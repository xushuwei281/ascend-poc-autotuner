"""Replay a fixed request DAG against a streamed Chat Completions endpoint.

This is a small, independent trace format, not the EvolvingInfra/Trie format.
Messages are frozen recorded inputs; generated text never changes later inputs.
Client queueing is included in TTFT/E2E. TPOT is an approximation from the span
of at least two content-bearing SSE chunks divided by the server's completion
token count minus one. Single-chunk/one-token outputs have unobservable TPOT.
No token counts are estimated from characters, and no agent KV headers are sent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit

from .storage import atomic_json


_MAX_TRACE_BYTES = 16 * 1024 * 1024
_MAX_BODY_BYTES = 4 * 1024 * 1024
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_EVENT_BYTES = 1024 * 1024
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')


class ReplayError(ValueError):
    """Invalid trace or sanitized benchmark transport failure."""


class _RequestFailure(ReplayError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ReplayError('Duplicate JSON key')
        result[key] = value
    return result


def _constant(_value):
    raise ReplayError('Non-finite JSON number')


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ReplayError('Invalid JSON data') from error


def _number(value, label, *, minimum=0, integer=False, maximum=None):
    try:
        finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum
            or (integer and type(value) is not int)
            or (maximum is not None and value > maximum)):
        raise ReplayError('Invalid ' + label)
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ReplayError('Invalid ' + label)
    return value


def load_trace(path):
    """Validate fixed messages, dependency graph and bounded trace inputs."""
    with Path(path).open('rb') as stream:
        raw = stream.read(_MAX_TRACE_BYTES + 1)
    if len(raw) > _MAX_TRACE_BYTES:
        raise ReplayError('Trace exceeds the size limit')
    trace = _json(raw)
    if not isinstance(trace, dict) or set(trace) != {'schema_version', 'requests'}:
        raise ReplayError('Trace must contain schema_version and requests')
    if type(trace['schema_version']) is not int or trace['schema_version'] != 1:
        raise ReplayError('Trace schema_version must be 1')
    rows = trace['requests']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ReplayError('Trace requires 1–10000 requests')
    seen = set()
    allowed = {'id', 'after', 'delay_s', 'messages', 'max_tokens', 'agent_id', 'branch_id', 'context_version'}
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ReplayError('Unknown request fields')
        name = _identifier(row.get('id'), 'request id')
        if name in seen:
            raise ReplayError('Duplicate request id')
        seen.add(name)
        after = row.setdefault('after', [])
        if not isinstance(after, list) or len(after) > 10000:
            raise ReplayError('Invalid dependency list')
        for dependency in after:
            _identifier(dependency, 'dependency id')
        if len(after) != len(set(after)) or name in after:
            raise ReplayError('Duplicate or self dependency')
        _number(row.setdefault('delay_s', 0), 'delay_s', maximum=86400)
        _number(row.get('max_tokens'), 'max_tokens', minimum=1, integer=True, maximum=1000000)
        for key in ('agent_id', 'branch_id'):
            if key in row:
                _identifier(row[key], key)
        if 'context_version' in row:
            _number(row['context_version'], 'context_version', integer=True)
        messages = row.get('messages')
        if not isinstance(messages, list) or not 1 <= len(messages) <= 10000:
            raise ReplayError('Request messages must be a nonempty list')
        for message in messages:
            if (not isinstance(message, dict)
                    or set(message) - {'role', 'content', 'name', 'tool_call_id', 'tool_calls'}
                    or message.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}):
                raise ReplayError('Invalid recorded message')
            content = message.get('content')
            if content is not None and not isinstance(content, (str, list)):
                raise ReplayError('Message content must be text or structured parts')
            if content is None and not isinstance(message.get('tool_calls'), list):
                raise ReplayError('Message requires content or recorded tool calls')
            # Structured image/tool content is data sent only to the requested
            # inference endpoint. It is never interpreted as local commands.
        if len(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode()) > _MAX_BODY_BYTES - 4096:
            raise ReplayError('Request messages exceed the size limit')
    dependents = {name: [] for name in seen}
    remaining = {}
    for row in rows:
        if any(name not in seen for name in row['after']):
            raise ReplayError('Unknown dependency')
        remaining[row['id']] = len(row['after'])
        for name in row['after']:
            dependents[name].append(row['id'])
    ready = [name for name, count in remaining.items() if count == 0]
    visited = 0
    while ready:
        name = ready.pop()
        visited += 1
        for child in dependents[name]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if visited != len(rows):
        raise ReplayError('Request dependency graph contains a cycle')
    return trace


def _endpoint(base_url):
    try:
        parts = urlsplit(base_url)
        host, port = parts.hostname, parts.port
        if not host or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError()
        local = host.lower() == 'localhost'
        try:
            local = local or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if parts.scheme != 'https' and not (parts.scheme == 'http' and local):
            raise ValueError()
        path = parts.path.rstrip('/')
        if not path.endswith('/chat/completions'):
            path += '/chat/completions'
        return parts.scheme, host, port, path
    except (ValueError, TypeError):
        raise ReplayError('Endpoint requires HTTPS or loopback HTTP without credentials, query or fragment') from None


def _percentile(values, fraction=.99):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _usage(value, max_output_tokens):
    if not isinstance(value, dict):
        raise ReplayError('Missing authoritative token usage')
    prompt = _number(value.get('prompt_tokens'), 'prompt token usage', integer=True)
    completion = _number(value.get('completion_tokens'), 'completion token usage', minimum=1,
                         integer=True, maximum=max_output_tokens)
    result = {'input_tokens': prompt, 'output_tokens': completion}
    details = value.get('prompt_tokens_details')
    if isinstance(details, dict) and details.get('cached_tokens') is not None:
        result['cached_tokens'] = _number(details['cached_tokens'], 'cached token usage', integer=True, maximum=prompt)
    return result


def _exchange(endpoint, body, headers, deadline, cancel, sockets, max_output_tokens):
    scheme, host, port, path = endpoint
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError()
    cls = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
    connection = cls(host, port=port, timeout=remaining)
    response = None
    try:
        connection.connect()
        if connection.sock:
            sockets.append(connection.sock)
            connection.sock.settimeout(max(.001, deadline - time.monotonic()))
        # DNS/connect can outlive the pool worker's timeout. Do not start a
        # generation after it has already returned a cancelled measurement.
        if cancel.is_set() or time.monotonic() >= deadline:
            raise TimeoutError()
        connection.request('POST', path, body=body, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise _RequestFailure('http_' + str(response.status))
        if 'text/event-stream' not in response.getheader('Content-Type', '').lower():
            raise ReplayError('Endpoint did not return an SSE stream')
        buffer = b''
        data = []
        event_bytes = 0
        total = 0
        first = last = None
        content_chunks = 0
        usage = None
        done = False
        while not done:
            if cancel.is_set() or time.monotonic() >= deadline:
                raise TimeoutError()
            for sock in sockets:
                sock.settimeout(max(.001, deadline - time.monotonic()))
            chunk = response.read1(65536)
            if not chunk:
                raise ReplayError('Incomplete SSE stream')
            total += len(chunk)
            if total > _MAX_RESPONSE_BYTES:
                raise ReplayError('Response exceeds the size limit')
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                line = line.rstrip(b'\r')
                if len(line) > _MAX_EVENT_BYTES:
                    raise ReplayError('SSE line exceeds the size limit')
                if line.startswith(b'data:'):
                    data.append(line[5:].lstrip(b' '))
                    event_bytes += len(data[-1]) + 1
                    if event_bytes > _MAX_EVENT_BYTES:
                        raise ReplayError('SSE event exceeds the size limit')
                elif line == b'' and data:
                    raw = b'\n'.join(data)
                    data = []
                    event_bytes = 0
                    if raw == b'[DONE]':
                        done = True
                        break
                    item = _json(raw)
                    if not isinstance(item, dict) or 'error' in item:
                        raise ReplayError('Invalid SSE payload')
                    if item.get('usage') is not None:
                        try:
                            observed = _usage(item['usage'], max_output_tokens)
                        except ReplayError as error:
                            raise _RequestFailure('invalid_usage') from error
                        if usage is not None and usage != observed:
                            raise ReplayError('Conflicting usage records')
                        usage = observed
                    choices = item.get('choices', [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise ReplayError('Invalid streamed choice list')
                    for choice in choices:
                        if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                            raise ReplayError('Invalid streamed choice')
                        delta = choice.get('delta', {})
                        if not isinstance(delta, dict):
                            raise ReplayError('Invalid streamed delta')
                        if any(delta.get(key) for key in ('content', 'reasoning_content', 'reasoning', 'tool_calls', 'refusal')):
                            now = time.monotonic()
                            first = now if first is None else first
                            last = now
                            content_chunks += 1
            if len(buffer) > _MAX_EVENT_BYTES:
                raise ReplayError('SSE line exceeds the size limit')
        if time.monotonic() > deadline or cancel.is_set():
            raise TimeoutError()
        if usage is None or first is None:
            raise ReplayError('Missing token usage or generated chunks')
        return dict(usage, first=first, last=last, content_chunks=content_chunks, ended=time.monotonic())
    finally:
        if response is not None:
            response.close()
        connection.close()


def _request(endpoint, row, model, key, ready_at, deadline):
    """Bound the pool worker even if DNS or a dribbling server blocks transport.

    A transport timeout ends the whole replay; no replacement request overlaps
    a timed-out transport. Remote generation may still consume service resources.
    """
    result = queue.Queue(maxsize=1)
    cancel = threading.Event()
    sockets = []
    body = json.dumps({'model': model, 'messages': row['messages'], 'max_tokens': row['max_tokens'],
                       'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}},
                      ensure_ascii=False, allow_nan=False).encode()
    if len(body) > _MAX_BODY_BYTES:
        return {'status': 'error', 'error_kind': 'invalid_body', 'ended': time.monotonic(), 'ready_at': ready_at}
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if key:
        headers['Authorization'] = 'Bearer ' + key

    def fetch():
        try:
            result.put(_exchange(endpoint, body, headers, deadline, cancel, sockets, row['max_tokens']))
        except (TimeoutError, socket.timeout):
            result.put({'error_kind': 'timeout'})
        except _RequestFailure as error:
            result.put({'error_kind': error.code})
        except Exception:
            # Never return response bodies, URLs, prompts or credential details.
            result.put({'error_kind': 'request_failed'})

    threading.Thread(target=fetch, daemon=True, name='poc-replay-transport').start()
    try:
        response = result.get(timeout=max(.001, deadline - time.monotonic()))
    except queue.Empty:
        response = {'error_kind': 'timeout'}
    if 'error_kind' in response:
        cancel.set()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return dict(response, status='error', ended=time.monotonic(), ready_at=ready_at)
    return dict(response, status='ok', ready_at=ready_at)


def run_replay(trace_path, base_url, model, output_path, *, max_concurrency=8,
               timeout_s=60, total_timeout_s=300, api_key_env=None, telemetry_output=None):
    """Execute a trace and write canonical metrics plus request telemetry.

    ``delay_s`` starts after all dependencies completed, or at replay start for
    roots. Requests queue at the configured client concurrency. Failed parents
    fail their descendants without sending them. A transport timeout stops new
    admissions. Every original request is counted, including budget-skipped ones.
    """
    trace = load_trace(trace_path)
    endpoint = _endpoint(base_url)
    _number(max_concurrency, 'max_concurrency', minimum=1, integer=True, maximum=256)
    _number(timeout_s, 'timeout_s', minimum=.001, maximum=86400)
    _number(total_timeout_s, 'total_timeout_s', minimum=.001, maximum=86400)
    if not isinstance(model, str) or not model.strip() or len(model) > 1024:
        raise ReplayError('A bounded model identifier is required')
    key = None
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env):
            raise ReplayError('Invalid API key environment variable')
        key = os.getenv(api_key_env)
        if not key or not key.strip() or '\r' in key or '\n' in key:
            raise ReplayError('Inference API key is missing or invalid')
    output_path = Path(output_path).resolve()
    telemetry_path = Path(telemetry_output).resolve() if telemetry_output else output_path.parent / 'telemetry.jsonl'
    if output_path == telemetry_path or output_path == Path(trace_path).resolve() or telemetry_path == Path(trace_path).resolve():
        raise ReplayError('Trace and output paths must be distinct')
    rows = {row['id']: row for row in trace['requests']}
    pending = set(rows)
    results = {}
    running = {}
    start = time.monotonic()
    end = start + total_timeout_s
    stop_reason = 'finished'
    with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix='poc-replay') as pool:
        while pending or running:
            now = time.monotonic()
            if now >= end:
                stop_reason = 'total_timeout'
            if stop_reason != 'finished':
                for name in list(pending):
                    results[name] = {'status': 'error', 'error_kind': stop_reason, 'ended': now, 'ready_at': now}
                    pending.remove(name)
            else:
                available = []
                for name in list(pending):
                    row = rows[name]
                    if not all(parent in results for parent in row['after']):
                        continue
                    ready_at = max((results[parent]['ended'] for parent in row['after']), default=start) + row['delay_s']
                    if any(results[parent]['status'] != 'ok' for parent in row['after']):
                        results[name] = {'status': 'error', 'error_kind': 'dependency_failed', 'ended': now,
                                         'ready_at': min(now, ready_at)}
                        pending.remove(name)
                    else:
                        available.append((ready_at, name))
                for ready_at, name in sorted(available):
                    if len(running) >= max_concurrency or ready_at > now:
                        break
                    pending.remove(name)
                    future = pool.submit(_request, endpoint, rows[name], model, key, ready_at,
                                         min(end, time.monotonic() + timeout_s))
                    running[future] = name
            if running:
                completed, _ = wait(running, timeout=min(.02, max(.001, end - time.monotonic())),
                                    return_when=FIRST_COMPLETED)
                for future in completed:
                    name = running.pop(future)
                    results[name] = future.result()
                    if results[name].get('error_kind') == 'timeout':
                        stop_reason = 'transport_timeout'
            elif pending:
                time.sleep(min(.01, max(.001, end - time.monotonic())))
    finished = time.monotonic()
    duration = max(finished - start, 1e-9)
    success = [result for result in results.values() if result['status'] == 'ok']
    ttft = [(r['first'] - r['ready_at']) * 1000 for r in success]
    e2e = [(r['ended'] - r['ready_at']) * 1000 for r in success]
    tpot = [(r['last'] - r['first']) * 1000 / (r['output_tokens'] - 1)
            for r in success if r['output_tokens'] > 1 and r['content_chunks'] >= 2]
    tpot_eligible = sum(r['output_tokens'] > 1 for r in success)
    tpot_unobservable = tpot_eligible - len(tpot)
    outputs = sum(r['output_tokens'] for r in success)
    metrics = {'completed': len(success), 'failed': len(rows) - len(success),
               'error_rate': (len(rows) - len(success)) / len(rows), 'duration_s': duration,
               'request_per_second': len(success) / duration, 'output_tokens_per_second': outputs / duration,
               'total_tokens_per_second': sum(r['input_tokens'] + r['output_tokens'] for r in success) / duration,
               'tpot_measured_requests': len(tpot), 'tpot_unobservable_requests': tpot_unobservable,
               'tpot_ineligible_requests': len(success) - tpot_eligible,
               'tpot_observation_coverage': len(tpot) / tpot_eligible if tpot_eligible else None,
               'client_concurrency': max_concurrency,
               'stop_reason': stop_reason, 'timing_origin': 'dependency_ready_time_includes_client_queue',
               'tpot_method': 'content_chunk_span_divided_by_server_completion_tokens_minus_one',
               'workload_mode': 'fixed_recorded_messages_no_live_tools', 'telemetry_file': telemetry_path.name}
    metrics['failure_counts'] = {}
    for result in results.values():
        if result['status'] == 'error':
            code = result['error_kind']
            metrics['failure_counts'][code] = metrics['failure_counts'].get(code, 0) + 1
    for prefix, values in [('ttft', ttft), ('e2el', e2e)]:
        if values:
            metrics['p99_' + prefix + '_ms'] = _percentile(values)
            metrics['mean_' + prefix + '_ms'] = statistics.mean(values)
    if tpot:
        # Never present the observable subset's tail as the complete workload's
        # tail. One-token requests have no TPOT by definition; multi-token
        # requests delivered in one chunk leave it unobservable.
        metrics['observed_p99_tpot_ms'] = _percentile(tpot)
        metrics['observed_mean_tpot_ms'] = statistics.mean(tpot)
        if not tpot_unobservable:
            metrics['p99_tpot_ms'] = metrics['observed_p99_tpot_ms']
            metrics['mean_tpot_ms'] = metrics['observed_mean_tpot_ms']
    tasks = {}
    for name, row in rows.items():
        if 'agent_id' in row:
            tasks.setdefault(row['agent_id'], []).append(results[name])
    if tasks:
        completed_tasks = [members for members in tasks.values()
                           if all(result['status'] == 'ok' for result in members)]
        task_times = [(max(r['ended'] for r in members) - min(r['ready_at'] for r in members)) * 1000
                      for members in completed_tasks]
        metrics.update(task_completed=len(completed_tasks), task_failed=len(tasks) - len(completed_tasks),
                       task_observation_coverage=sum(map(len, tasks.values())) / len(rows))
        if task_times:
            metrics['mean_task_e2e_ms'] = statistics.mean(task_times)
            metrics['p99_task_e2e_ms'] = _percentile(task_times)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, result in sorted(results.items(), key=lambda item: (item[1]['ended'], item[0])):
        row = {'request_id': name, 'started_at_s': max(0, result['ready_at'] - start),
               'completed_at_s': max(0, result['ended'] - start), 'status': result['status']}
        for key_name in ('input_tokens', 'output_tokens', 'cached_tokens'):
            if key_name in result:
                row[key_name] = result[key_name]
        for key_name in ('agent_id', 'branch_id', 'context_version'):
            if key_name in rows[name]:
                row[key_name] = rows[name][key_name]
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
    telemetry_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(output_path, metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--timeout-s', type=float, default=60)
    parser.add_argument('--total-timeout-s', type=float, default=300)
    parser.add_argument('--api-key-env')
    parser.add_argument('--telemetry-output')
    args = parser.parse_args(argv)
    try:
        result = run_replay(args.trace, args.base_url, args.model, args.output,
                            max_concurrency=args.concurrency, timeout_s=args.timeout_s,
                            total_timeout_s=args.total_timeout_s, api_key_env=args.api_key_env,
                            telemetry_output=args.telemetry_output)
    except (ValueError, OSError):
        print('Replay failed: validate the trace, endpoint, credentials and output paths.')
        return 1
    print(json.dumps(result, ensure_ascii=False))
    # A completed benchmark may include HTTP/request failures. Its configured
    # error_rate requirement decides acceptance; exit status must not silently
    # replace that customer threshold. An incomplete timeout is an execution
    # failure and cannot be treated as a complete measurement matrix.
    return 0 if result['stop_reason'] == 'finished' else 2


if __name__ == '__main__':
    raise SystemExit(main())
