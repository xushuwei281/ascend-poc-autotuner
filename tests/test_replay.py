"""Wire-level tests for a frozen, bounded agent trace replay."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ascend_poc.replay import ReplayError, _request, load_trace, main, run_replay


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text = body['messages'][0]['content']
                with cls.lock:
                    cls.requests.append({'body': body, 'headers': dict(self.headers), 'time': time.monotonic()})
                if text == 'redirect':
                    self.send_response(307)
                    self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/redirected')
                    self.end_headers()
                    return
                if text == 'error':
                    self.send_error(500)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    if text == 'timeout':
                        time.sleep(.4)
                    chunk = {'choices': [{'index': 0, 'delta': {'content': 'private first text'}}]}
                    self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
                    self.wfile.flush()
                    if text != 'single chunk':
                        time.sleep(.005)
                        chunk['choices'][0]['delta']['content'] = 'private second text'
                        self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
                    if text != 'missing usage':
                        usage = {'prompt_tokens': 32, 'completion_tokens': 1 if text == 'single' else 4}
                        if text == 'over limit':
                            usage['completion_tokens'] = 999
                        if text == 'cached':
                            usage['prompt_tokens_details'] = {'cached_tokens': 24}
                        self.wfile.write(b'data: ' + json.dumps({'choices': [], 'usage': usage}).encode() + b'\n\n')
                    if text != 'truncated':
                        self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trace = self.root / 'trace.json'
        self.output = self.root / 'metrics.json'
        with self.lock:
            self.requests.clear()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def row(name='r1', text='cached', **kwargs):
        return {'id': name, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': 8, **kwargs}

    def write(self, rows):
        self.trace.write_text(json.dumps({'schema_version': 1, 'requests': rows}))

    def run_rows(self, rows, **kwargs):
        self.write(rows)
        return run_replay(self.trace, self.url, 'fixed-model', self.output, **kwargs)

    def test_streamed_counts_metrics_and_private_telemetry(self):
        result = self.run_rows([self.row(agent_id='agent1', branch_id='branch1', context_version=1)])
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(result['error_rate'], 0)
        self.assertGreater(result['p99_ttft_ms'], 0)
        self.assertGreater(result['p99_tpot_ms'], 0)
        self.assertGreater(result['output_tokens_per_second'], 0)
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved, result)
        telemetry = (self.root / 'telemetry.jsonl').read_text()
        row = json.loads(telemetry)
        self.assertEqual(row['input_tokens'], 32)
        self.assertEqual(row['output_tokens'], 4)
        self.assertEqual(row['cached_tokens'], 24)
        self.assertEqual(row['agent_id'], 'agent1')
        self.assertNotIn('private', telemetry)
        self.assertNotIn('messages', telemetry)
        self.assertTrue(self.requests[0]['body']['stream_options']['include_usage'])
        self.assertNotIn('x-agent-context', {key.lower() for key in self.requests[0]['headers']})

    def test_unknown_cache_is_not_zero_and_one_token_has_no_tpot(self):
        result = self.run_rows([self.row(text='single')])
        self.assertNotIn('p99_tpot_ms', result)
        self.assertEqual(result['tpot_measured_requests'], 0)
        self.assertEqual(result['tpot_unobservable_requests'], 0)
        self.assertEqual(result['tpot_ineligible_requests'], 1)
        row = json.loads((self.root / 'telemetry.jsonl').read_text())
        self.assertNotIn('cached_tokens', row)

    def test_single_content_chunk_does_not_fabricate_zero_tpot(self):
        result = self.run_rows([self.row(text='single chunk')])
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['tpot_measured_requests'], 0)
        self.assertEqual(result['tpot_unobservable_requests'], 1)
        self.assertNotIn('p99_tpot_ms', result)
        self.assertNotIn('task_completed', result)

    def test_unobservable_tpot_cannot_be_hidden_by_observable_subset(self):
        result = self.run_rows([self.row('observable'), self.row('unobservable', 'single chunk')])
        self.assertEqual(result['tpot_observation_coverage'], .5)
        self.assertEqual(result['tpot_unobservable_requests'], 1)
        self.assertIn('observed_p99_tpot_ms', result)
        self.assertNotIn('p99_tpot_ms', result)
        self.assertNotIn('mean_tpot_ms', result)
        result = self.run_rows([self.row('observable'), self.row('ineligible', 'single')])
        self.assertEqual(result['tpot_observation_coverage'], 1)
        self.assertEqual(result['tpot_ineligible_requests'], 1)
        self.assertIn('p99_tpot_ms', result)

    def test_unordered_dag_preserves_dependencies_and_delay(self):
        result = self.run_rows([
            self.row('child', 'child recorded prompt', after=['parent'], delay_s=.025),
            self.row('parent', 'parent recorded prompt'),
        ], max_concurrency=2)
        self.assertEqual(result['completed'], 2)
        bodies = [item['body']['messages'][0]['content'] for item in self.requests]
        self.assertEqual(bodies, ['parent recorded prompt', 'child recorded prompt'])
        times = [item['time'] for item in self.requests]
        self.assertGreaterEqual(times[1] - times[0], .025)
        telemetry = [json.loads(line) for line in (self.root / 'telemetry.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(telemetry[1]['started_at_s'] - telemetry[0]['completed_at_s'], .024)

    def test_failed_parent_skips_descendants_and_counts_every_request(self):
        result = self.run_rows([self.row('bad', 'error'), self.row('child', after=['bad']),
                                self.row('grandchild', after=['child'])])
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['failed'], 3)
        self.assertEqual(result['error_rate'], 1)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn('p99_ttft_ms', result)
        for line in (self.root / 'telemetry.jsonl').read_text().splitlines():
            self.assertNotIn('input_tokens', json.loads(line))

    def test_task_latency_includes_waits_and_branch_join_without_exposing_ids(self):
        result = self.run_rows([
            self.row('root', agent_id='private-agent', branch_id='root'),
            self.row('fork-a', after=['root'], delay_s=.02, agent_id='private-agent', branch_id='a'),
            self.row('fork-b', after=['root'], delay_s=.01, agent_id='private-agent', branch_id='b'),
            self.row('join', after=['fork-a', 'fork-b'], delay_s=.02, agent_id='private-agent', branch_id='root'),
        ], max_concurrency=2)
        self.assertEqual(result['task_completed'], 1)
        self.assertEqual(result['task_failed'], 0)
        self.assertEqual(result['task_observation_coverage'], 1)
        self.assertGreater(result['mean_task_e2e_ms'], 40)
        self.assertGreater(result['p99_task_e2e_ms'], result['p99_e2el_ms'])
        self.assertNotIn('private-agent', json.dumps(result))

    def test_failed_task_is_not_an_artificially_fast_completed_task(self):
        result = self.run_rows([self.row('bad', 'error', agent_id='bad-agent'),
                                self.row('child', after=['bad'], agent_id='bad-agent'), self.row('unassigned')])
        self.assertEqual(result['task_completed'], 0)
        self.assertEqual(result['task_failed'], 1)
        self.assertEqual(result['task_observation_coverage'], 2 / 3)
        self.assertNotIn('p99_task_e2e_ms', result)

    def test_missing_usage_and_truncated_stream_cannot_succeed(self):
        for text in ('missing usage', 'truncated'):
            with self.subTest(text=text):
                result = self.run_rows([self.row(text=text)])
                self.assertEqual(result['failed'], 1)
                self.assertEqual(result['completed'], 0)

    def test_backend_ignoring_output_budget_is_rejected(self):
        result = self.run_rows([self.row(text='over limit')])
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['failure_counts'], {'invalid_usage': 1})

    def test_expired_connect_never_sends_late_generation(self):
        connected = threading.Event()
        finish = threading.Event()
        posted = []

        class Connection:
            sock = None

            def __init__(self, *_args, **_kwargs):
                pass

            def connect(self):
                connected.set()
                finish.wait(1)

            def request(self, *_args, **_kwargs):
                posted.append(True)

            def close(self):
                pass

        with patch('ascend_poc.replay.http.client.HTTPConnection', Connection):
            now = time.monotonic()
            result = _request(('http', '127.0.0.1', 1, '/v1/chat/completions'), self.row(),
                              'model', None, now, now + .02)
            self.assertTrue(connected.is_set())
            self.assertEqual(result['error_kind'], 'timeout')
            finish.set()
            time.sleep(.02)
            self.assertEqual(posted, [])

    def test_redirect_never_replays_credentials_to_new_url(self):
        with patch.dict(os.environ, {'REPLAY_TEST_KEY': 'test-only'}):
            result = self.run_rows([self.row(text='redirect')], api_key_env='REPLAY_TEST_KEY')
        self.assertEqual(result['failed'], 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]['headers']['Authorization'], 'Bearer test-only')

    def test_transport_timeout_stops_new_admissions_without_hanging(self):
        started = time.monotonic()
        result = self.run_rows([self.row('slow', 'timeout'), self.row('later', after=['slow'])],
                               timeout_s=.05, total_timeout_s=.2)
        self.assertLess(time.monotonic() - started, .3)
        self.assertEqual(result['stop_reason'], 'transport_timeout')
        self.assertEqual(result['failed'], 2)
        self.assertEqual(len(self.requests), 1)

    def test_overall_budget_counts_waiting_root_as_failed(self):
        result = self.run_rows([self.row(delay_s=5)], total_timeout_s=.02)
        self.assertEqual(result['stop_reason'], 'total_timeout')
        self.assertEqual(result['failed'], 1)
        self.assertEqual(len(self.requests), 0)

    def test_client_queue_is_included_in_request_latency(self):
        result = self.run_rows([self.row('a'), self.row('b')], max_concurrency=1)
        self.assertEqual(result['completed'], 2)
        rows = [json.loads(line) for line in (self.root / 'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['started_at_s'], 0)
        self.assertEqual(rows[1]['started_at_s'], 0)
        self.assertGreater(rows[1]['completed_at_s'], rows[0]['completed_at_s'])

    def test_invalid_trace_shapes_rejected_without_http(self):
        cases = [[], [self.row('same'), self.row('same')], [self.row(after=['missing'])],
                 [self.row('a', after=['b']), self.row('b', after=['a'])],
                 [self.row(delay_s=-1)], [self.row(max_tokens=True)],
                 [self.row(command='bad')], [self.row(after=['r1'])]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    load_trace(self.trace)
        self.assertEqual(self.requests, [])

    def test_duplicate_keys_nonfinite_and_unsafe_endpoint_rejected(self):
        self.trace.write_text('{"schema_version":1,"schema_version":1,"requests":[]}')
        with self.assertRaises(ValueError):
            load_trace(self.trace)
        self.write([self.row(delay_s=float('nan'))])
        with self.assertRaises(ValueError):
            load_trace(self.trace)
        self.write([self.row()])
        for url in ('http://example.com/v1', 'https://u:p@example.com/v1', 'https://example.com/v1?key=x'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_replay(self.trace, url, 'model', self.output)

    def test_cli_defers_recorded_request_failures_to_customer_error_rate_gate(self):
        self.write([self.row(text='error')])
        with redirect_stdout(io.StringIO()):
            code = main(['--trace', str(self.trace), '--base-url', self.url, '--model', 'test',
                         '--output', str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.output.read_text())['error_rate'], 1)
        self.write([self.row(delay_s=1)])
        with redirect_stdout(io.StringIO()):
            code = main(['--trace', str(self.trace), '--base-url', self.url, '--model', 'test',
                         '--output', str(self.output), '--total-timeout-s', '.01'])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
