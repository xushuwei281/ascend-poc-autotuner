"""Summarize observed request metadata without exposing prompts or identities.

These observations describe work, not engine utilization. In particular, the
causal attention term is a full-attention work proxy, and agents observed in a
window are not a measurement of live agents or decode concurrency. Cache reuse
must be reported by the serving stack; omitted cache observations stay unknown.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


_MAX_ROWS = 100_000
_MAX_WINDOWS = 10_000
_MAX_TOKENS = 10**12
_ALLOWED = {
    'request_id', 'completed_at_s', 'started_at_s', 'input_tokens', 'output_tokens',
    'cached_tokens', 'restored_tokens', 'accepted_tokens_per_step', 'agent_id',
    'branch_id', 'context_version', 'event', 'status',
}
_REQUIRED = {'request_id', 'completed_at_s', 'started_at_s'}


def _number(value, label, *, positive=False):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0 and (not positive or value > 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{label} must be a finite {"positive" if positive else "non-negative"} number')
    return float(value)


def _integer(value, label):
    if type(value) is not int or not 0 <= value <= _MAX_TOKENS:
        raise ValueError(f'{label} must be a non-negative integer no larger than {_MAX_TOKENS}')
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value or len(value.encode('utf-8')) > 512:
        raise ValueError(f'{label} must be a nonempty string of at most 512 UTF-8 bytes')
    return value


def _validate(rows):
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
        raise ValueError(f'telemetry must be a list of at most {_MAX_ROWS} requests')
    seen = set()
    checked = []
    for row in rows:
        if not isinstance(row, dict) or not _REQUIRED <= set(row) or set(row) - _ALLOWED:
            raise ValueError('telemetry request has missing or unknown fields')
        item = dict(row)
        if not isinstance(item.setdefault('status', 'ok'), str) or item['status'] not in {'ok', 'error'}:
            raise ValueError('telemetry status must be ok or error')
        if item['status'] == 'ok' and not {'input_tokens', 'output_tokens'} <= set(item):
            raise ValueError('successful telemetry requires input_tokens and output_tokens')
        identifier = _identifier(item['request_id'], 'request_id')
        if identifier in seen:
            raise ValueError('telemetry contains duplicate request_id values')
        seen.add(identifier)
        for field in ('agent_id', 'branch_id'):
            if field in item:
                _identifier(item[field], field)
        for field in ('started_at_s', 'completed_at_s'):
            item[field] = _number(item[field], field)
        if item['completed_at_s'] < item['started_at_s']:
            raise ValueError('completed_at_s must not precede started_at_s')
        for field in ('input_tokens', 'output_tokens', 'cached_tokens', 'restored_tokens', 'context_version'):
            if field in item:
                _integer(item[field], field)
        if 'cached_tokens' in item and ('input_tokens' not in item or item['cached_tokens'] > item['input_tokens']):
            raise ValueError('cached_tokens requires input_tokens and cannot exceed it')
        if 'restored_tokens' in item and ('cached_tokens' not in item or item['restored_tokens'] > item['cached_tokens']):
            raise ValueError('restored_tokens requires cached_tokens and cannot exceed it')
        if 'accepted_tokens_per_step' in item:
            item['accepted_tokens_per_step'] = _number(item['accepted_tokens_per_step'], 'accepted_tokens_per_step', positive=True)
            if item['accepted_tokens_per_step'] > _MAX_TOKENS:
                raise ValueError('accepted_tokens_per_step is too large')
        if item.setdefault('event', 'request_completed') != 'request_completed':
            raise ValueError('only request_completed events are supported')
        checked.append(item)
    return checked


def _mean(values):
    return sum(values) / len(values) if values else None


def _aggregate(rows):
    good = [row for row in rows if row['status'] == 'ok']
    cached = [row for row in good if 'cached_tokens' in row]
    restored = [row['restored_tokens'] for row in good if 'restored_tokens' in row]
    accepted = [row['accepted_tokens_per_step'] for row in good if 'accepted_tokens_per_step' in row]
    known_input = sum(row['input_tokens'] for row in cached)
    agents = {row['agent_id'] for row in good if 'agent_id' in row}
    branches = {(row.get('agent_id'), row['branch_id']) for row in good if 'branch_id' in row}
    versions = {}
    changes = 0
    # Start order reflects the order in which context versions were submitted;
    # completion order may be reversed by parallel requests.
    for row in sorted(good, key=lambda r: (r['started_at_s'], r['completed_at_s'], r['request_id'])):
        if not {'agent_id', 'branch_id', 'context_version'} <= set(row):
            continue
        branch = (row['agent_id'], row['branch_id'])
        version = row['context_version']
        if branch in versions and versions[branch] != version:
            changes += 1
        versions[branch] = version
    uncached = [row['input_tokens'] - row['cached_tokens'] for row in cached]
    pairs = [u * row['cached_tokens'] + u * (u + 1) / 2 for u, row in zip(uncached, cached)]
    return {
        'request_count': len(rows), 'successful': len(good), 'failed': len(rows) - len(good),
        'input_tokens_total': sum(row['input_tokens'] for row in good),
        'output_tokens_total': sum(row['output_tokens'] for row in good),
        'cache_observed_requests': len(cached), 'cache_unknown_requests': len(good) - len(cached),
        'cache_observation_coverage': len(cached) / len(good) if good else None,
        'active_agent_count': len(agents), 'branch_count': len(branches),
        'context_version_changes': changes,
        'overall': {
            'mean_input_tokens': _mean([row['input_tokens'] for row in good]),
            'mean_output_tokens': _mean([row['output_tokens'] for row in good]),
            'mean_cached_tokens': _mean([row['cached_tokens'] for row in cached]),
            'mean_uncached_tokens': _mean(uncached),
            'cache_hit_ratio': sum(row['cached_tokens'] for row in cached) / known_input if known_input else None,
            'mean_causal_attention_pairs': _mean(pairs),
            'mean_restored_tokens': _mean(restored),
            'mean_accepted_tokens_per_step': _mean(accepted),
        },
    }


def summarize_requests(rows: list, window_s: float = 60) -> dict:
    """Aggregate completed-request observations and fixed-width time slices.

    Each row requires request_id, started_at_s and completed_at_s. Successful
    rows also require input_tokens and output_tokens; failed rows can omit
    unknown token counts. Optional cache counts are observations, never inferred from
    prompt overlap. Only successful requests contribute token means and totals;
    failed requests remain in request/error counts. Window assignment uses
    completion time, anchored at the earliest request start, and includes idle
    windows. The last window may be incomplete, indicated by observation_end_s.

    Identity strings are used locally only for counts and never returned. A
    context_version change is an observation, not proof that compaction occurred
    or that the inference backend evicted any KV cache.
    """
    interval = _number(window_s, 'window_s', positive=True)
    checked = _validate(rows)
    result = {
        'schema_version': 1, 'source': 'completed_request_metadata',
        'window_s': interval,
        'cache_ratio_population': 'successful_requests_with_cache_observations',
        'attention_work_semantics': 'full_causal_attention_proxy_not_measured_compute',
        'agent_count_semantics': 'distinct_agents_with_successful_completions_not_live_population',
        'context_change_semantics': 'observed_version_changes_not_backend_eviction',
        **_aggregate(checked), 'windows': [],
    }
    if not checked:
        result.update(observation_start_s=None, observation_end_s=None)
        return result
    start = min(row['started_at_s'] for row in checked)
    end = max(row['completed_at_s'] for row in checked)
    span = end - start
    count_float = span / interval
    if not math.isfinite(count_float) or count_float >= _MAX_WINDOWS:
        raise ValueError(f'telemetry spans more than {_MAX_WINDOWS} windows')
    count = math.floor(count_float) + 1
    windows = [[] for _ in range(count)]
    for row in checked:
        index = math.floor((row['completed_at_s'] - start) / interval)
        windows[index].append(row)
    for index, items in enumerate(windows):
        left = start + index * interval
        right = start + (index + 1) * interval
        if not math.isfinite(right) or right <= left:
            raise ValueError('window boundaries cannot be represented at the supplied timestamps')
        result['windows'].append({'start_s': left, 'end_s': right,
                                  'complete_window': right <= end, **_aggregate(items)})
    result.update(observation_start_s=start, observation_end_s=end)
    return result


_COUNTS = {'request_count', 'successful', 'failed', 'input_tokens_total', 'output_tokens_total',
           'cache_observed_requests', 'cache_unknown_requests', 'active_agent_count',
           'branch_count', 'context_version_changes'}
_MEANS = {'mean_input_tokens', 'mean_output_tokens', 'mean_cached_tokens', 'mean_uncached_tokens',
          'cache_hit_ratio', 'mean_causal_attention_pairs', 'mean_restored_tokens',
          'mean_accepted_tokens_per_step'}


def public_summary(summary: dict, max_windows: int = 6) -> dict:
    """Whitelist finite numeric observations for research/LLM requests.

    The input may be read from a local checkpoint; never trust its field names,
    explanatory strings or nested metadata. Retain the first and last windows
    when limiting size so a change in workload remains visible. Report the full
    window count so this sampling cannot be mistaken for complete coverage.
    """
    if type(max_windows) is not int or not 0 <= max_windows <= 1000:
        raise ValueError('max_windows must be an integer between 0 and 1000')
    if not isinstance(summary, dict):
        return {}

    def finite(value):
        try:
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0)
        except OverflowError:
            return False

    def clean(item):
        if not isinstance(item, dict):
            return {}
        safe = {key: item[key] for key in _COUNTS
                if key in item and type(item[key]) is int and 0 <= item[key] <= _MAX_TOKENS * _MAX_ROWS}
        coverage = item.get('cache_observation_coverage')
        if coverage is None or (finite(coverage) and coverage <= 1):
            safe['cache_observation_coverage'] = coverage
        original = item.get('overall')
        means = {}
        if isinstance(original, dict):
            for key in _MEANS:
                if key not in original:
                    continue
                value = original[key]
                if value is None or (finite(value) and (key != 'cache_hit_ratio' or value <= 1)):
                    means[key] = value
        safe['overall'] = means
        return safe

    result = clean(summary)
    # These explanations are fixed program text, never copied from telemetry.
    result.update(schema_version=1,
                  attention_work_semantics='full_causal_attention_proxy_not_measured_compute',
                  agent_count_semantics='distinct_agents_with_successful_completions_not_live_population',
                  context_change_semantics='observed_version_changes_not_backend_eviction',
                  cache_ratio_population='successful_requests_with_cache_observations')
    for key in ('window_s', 'observation_start_s', 'observation_end_s'):
        value = summary.get(key)
        if value is None or finite(value):
            result[key] = value
    windows = summary.get('windows', [])
    if not isinstance(windows, list):
        windows = []
    declared_count = summary.get('window_count')
    result['window_count'] = (declared_count if type(declared_count) is int
                              and len(windows) <= declared_count <= _MAX_WINDOWS else len(windows))
    if len(windows) > max_windows:
        head = (max_windows + 1) // 2
        tail = max_windows // 2
        selected = windows[:head] + (windows[-tail:] if tail else [])
    else:
        selected = windows
    result['windows'] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        safe = clean(item)
        for key in ('start_s', 'end_s'):
            if finite(item.get(key)):
                safe[key] = item[key]
        if type(item.get('complete_window')) is bool:
            safe['complete_window'] = item['complete_window']
        result['windows'].append(safe)
    result['windows_sampled'] = len(selected) < result['window_count']
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate telemetry JSON key')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('telemetry JSON must contain finite numbers')


def load_telemetry(path: Path, max_bytes: int = 8 * 1024 * 1024) -> dict:
    """Read bounded UTF-8 JSONL metadata and return a 60-second summary.

    An empty file is a valid empty observation, not evidence of passing a
    benchmark. Unknown fields (including prompts/messages) are rejected.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    path = Path(path)
    try:
        if not path.is_file():
            raise ValueError('telemetry path must name a regular file')
        with path.open('rb') as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError('telemetry file exceeds its byte limit')
        lines = payload.decode('utf-8').splitlines()
        rows = []
        for line in lines:
            if not line.strip():
                continue
            rows.append(json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant))
            if len(rows) > _MAX_ROWS:
                raise ValueError('telemetry has too many request rows')
    except (OSError, UnicodeError) as error:
        raise ValueError('telemetry file cannot be read as UTF-8') from error
    return summarize_requests(rows)
