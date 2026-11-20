"""Local HTTP/subprocess demonstration of open, intent-driven tuning.

The planner and Wiki answers are scripted protocol fixtures. All benchmark
numbers are synthetic; this does not evaluate real LLM intelligence or NPU speed.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading

from .storage import atomic_json


REQUEST = ('Hardware: two SYNTHETIC cards. Model family: demo. '
           'Maximize output_tokens_per_second. p99_tpot_ms <= 50.')

HELP = '''Synthetic CPU demo options (not installed vLLM capabilities):
  --host HOST  Controlled loopback endpoint.
  --port PORT  Controlled loopback port.
  --served-model-name NAME  Synthetic service identity.
  --max-num-seqs N  Synthetic sequence limit.
  --max-num-batched-tokens N  Synthetic token budget.
  --gpu-memory-utilization FRACTION  Synthetic device memory budget.
  --tensor-parallel-size N  Synthetic parallel layout.
  --enable-prefix-caching, --disable-prefix-caching  Synthetic cache switch.
  --compilation-config JSON  Synthetic nested configuration.
'''


def demo_intent():
    return {
        'schema_version': 1,
        'summary': 'Optimize synthetic throughput under the explicit TPOT limit.',
        'objective': {'metric': 'output_tokens_per_second', 'direction': 'maximize',
                      'scenario_aggregation': 'min'},
        'metric_requirements': {'p99_tpot_ms': {'max': 50}},
        'constraints': [
            {'scope': 'hardware', 'key': 'device_count', 'operator': 'eq', 'value': 2,
             'source_quote': 'two SYNTHETIC cards'},
            {'scope': 'model', 'key': 'family', 'operator': 'eq', 'value': 'demo',
             'source_quote': 'Model family: demo'},
        ],
        'requirement_evidence': {'p99_tpot_ms': 'p99_tpot_ms <= 50'},
        'objective_evidence': 'Maximize output_tokens_per_second',
        'inferred': ['Use the minimum across scenarios as the aggregate score.'],
        'unresolved': [],
    }


def demo_candidate(index):
    candidate = {
        'model': 'demo-bf16',
        'engine': {
            'max-num-seqs': 32,
            'max-num-batched-tokens': 2048,
            'gpu-memory-utilization': .85,
            'tensor-parallel-size': 1 if index < 2 else 2,
            'enable-prefix-caching': True,
            'compilation-config': {'mode': 3},
        },
        'env': {},
        'workload': {'concurrency': (16, 64, 24)[min(index, 2)]},
    }
    return candidate


def make_runtime(inputs: Path, serving_port: int, fixture_port: int):
    model = inputs / 'synthetic-model'
    model.mkdir(parents=True, exist_ok=True)
    (model / 'SYNTHETIC.txt').write_text('Not real model weights. Demo only.\n', encoding='utf-8')
    (inputs / 'engine-help.txt').write_text(HELP, encoding='utf-8')
    baseline = demo_candidate(0)
    baseline['engine']['enable-prefix-caching'] = False
    baseline['engine']['compilation-config'] = {'mode': 0}
    baseline['workload']['concurrency'] = 8
    return {
        'schema_version': 1,
        'name': 'Synthetic open tuning demo — NOT Ascend measurements',
        'hardware': {'device_type': 'SYNTHETIC', 'device_count': 2,
                     'devices': [f'synthetic-{serving_port}-0', f'synthetic-{serving_port}-1']},
        'models': [{'id': 'demo-bf16', 'path': str(model), 'family': 'demo', 'precision': 'bf16'}],
        'server': {
            'command': [sys.executable, '-m', 'ascend_poc.open_mock', 'serve', '{model}'],
            'fixed_options': {'host': '127.0.0.1', 'port': serving_port,
                              'served-model-name': 'synthetic-demo'},
            'env': {},
            'health_url': f'http://127.0.0.1:{serving_port}/health',
            'startup_timeout_s': 10,
            'shutdown_timeout_s': 2,
        },
        'discovery': {'help_file': str(inputs / 'engine-help.txt'), 'timeout_s': 10, 'env_names': []},
        'baseline': baseline,
        'scenarios': [{
            'name': 'synthetic',
            'command': [sys.executable, '-m', 'ascend_poc.open_mock', 'benchmark',
                        '--base-url', f'http://127.0.0.1:{serving_port}',
                        '--concurrency', '{workload.concurrency}',
                        '--output', '{trial_dir}/metrics.json'],
            'result_file': '{trial_dir}/metrics.json',
            'metrics_format': 'canonical',
            'timeout_s': 10,
        }],
        'workload_parameters': {'concurrency': {'type': 'integer',
            'description': 'Synthetic benchmark load; changes are not same-load engine speedups.',
            'minimum': 1, 'maximum': 128}},
        'metric_names': ['output_tokens_per_second', 'p99_tpot_ms', 'p99_ttft_ms', 'completed', 'failed'],
        'benchmark_env': {},
        'llm': {'base_url': f'http://127.0.0.1:{fixture_port}/v1',
                'model': 'scripted-synthetic-planner', 'api_key_env': 'POC_OPEN_DEMO_KEY',
                'timeout_s': 10, 'max_tokens': 3000, 'temperature': 0, 'json_mode': True,
                'max_retries': 0},
        'research': {'endpoint': f'http://127.0.0.1:{fixture_port}/mcp',
                     'repository': 'vllm-project/vllm-ascend', 'timeout_s': 10,
                     'max_calls': 4, 'on_error': 'stop'},
        'budget': {'max_experiments': 3, 'max_wall_time_s': 120, 'max_llm_calls': 10,
                   'repetitions': 1, 'final_repetitions': 2},
        'verification': {'mode': 'aba', 'max_baseline_drift': .05,
                         'min_relative_improvement': .02, 'drift_multiplier': 2.0},
    }


def run_open_demo(output: Path) -> dict:
    from .open_runtime import load_runtime
    from .open_tuner import run_autotune

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Open tuning demo requires a fresh, empty output directory')
    inputs = output / 'input'
    inputs.mkdir(parents=True)

    class Fixture(BaseHTTPRequestHandler):
        proposal_index = 0

        def send_json(self, value, status=200):
            encoded = json.dumps(value, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError()
                body = json.loads(self.rfile.read(size))
            except (ValueError, TypeError):
                self.send_error(400)
                return
            if self.path == '/v1/chat/completions':
                try:
                    payload = json.loads(body['messages'][-1]['content'])
                    stage = payload['stage']
                except (KeyError, TypeError, ValueError):
                    self.send_error(400)
                    return
                if stage == 'compile_intent':
                    decision = demo_intent()
                elif stage == 'diagnose':
                    failed = type(self).proposal_index >= 2
                    decision = {
                        'observations': ['SYNTHETIC measurements only; compare all configured gates.'],
                        'bottleneck': 'The aggressive synthetic load exceeds the TPOT limit.' if failed else
                                      'The synthetic baseline has room for cache and compilation changes.',
                        'research_question': 'For this synthetic demonstration, explain the tradeoff of '
                                             'prefix cache, compilation and TP layout under a fixed TPOT gate.',
                        'next_experiment': 'Reduce load and try the available two-card layout.' if failed else
                                           'Test a structured configuration beyond the old three knobs.',
                    }
                elif stage == 'propose':
                    index = min(type(self).proposal_index, 2)
                    evidence = payload.get('evidence_ids', [])
                    if not evidence and isinstance(payload.get('evidence'), dict):
                        evidence = [payload['evidence']['id']] if payload['evidence'].get('id') else []
                    decision = {
                        'candidate': demo_candidate(index),
                        'hypothesis': ('SYNTHETIC hypothesis: moderate cache/compilation improves the score.'
                                       if index == 0 else 'SYNTHETIC hypothesis: larger load raises throughput '
                                       'but may break TPOT.' if index == 1 else
                                       'SYNTHETIC correction: reduce load and use the available TP2 layout.'),
                        'expected_effect': 'Check throughput together with the unchanged TPOT <= 50 gate.',
                        'evidence_ids': evidence,
                    }
                    type(self).proposal_index += 1
                else:
                    self.send_error(400)
                    return
                self.send_json({'choices': [{'message': {'content': json.dumps(decision)}}],
                                'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}})
                return
            if self.path != '/mcp':
                self.send_error(404)
                return
            method = body.get('method')
            if method == 'notifications/initialized':
                self.send_response(202)
                self.end_headers()
                return
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'Synthetic Wiki fixture', 'version': '1'}}
            elif method == 'tools/list':
                result = {'tools': [{'name': 'ask_wiki_question', 'description': 'Synthetic research',
                                    'inputSchema': {'type': 'object', 'properties': {
                                        'repoName': {'type': 'string'}, 'question': {'type': 'string'}},
                                        'required': ['repoName', 'question']}}]}
            elif method == 'tools/call':
                result = {'content': [{'type': 'text', 'text':
                    'SYNTHETIC research fixture, not hardware advice. Prefix caching requires repeated '
                    'token prefixes; compilation and parallel layouts must be validated on the installed '
                    'stack. Higher load may increase throughput while breaking a latency gate. '
                    'Reference repository: https://deepwiki.com/vllm-project/vllm-ascend'}]}
            else:
                self.send_json({'jsonrpc': '2.0', 'id': body.get('id'),
                                'error': {'code': -32601, 'message': 'Unknown synthetic fixture method'}})
                return
            self.send_json({'jsonrpc': '2.0', 'id': body.get('id'), 'result': result})

        def log_message(self, *_):
            pass

    fixture = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    previous = os.environ.get('POC_OPEN_DEMO_KEY')
    thread.start()
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            serving_port = sock.getsockname()[1]
        runtime = make_runtime(inputs, serving_port, fixture.server_port)
        atomic_json(inputs / 'runtime.json', runtime)
        (inputs / 'request.txt').write_text(REQUEST + '\n', encoding='utf-8')
        os.environ['POC_OPEN_DEMO_KEY'] = 'synthetic-fixture-not-a-real-secret'
        print('DEMO: scripted planner and Wiki, synthetic benchmark numbers; real local HTTP and '
              'subprocess execution. No NPU or paid API is used.', flush=True)
        return run_autotune(load_runtime(inputs / 'runtime.json'), REQUEST, output,
                            workdir=Path(__file__).resolve().parents[1])
    finally:
        if previous is None:
            os.environ.pop('POC_OPEN_DEMO_KEY', None)
        else:
            os.environ['POC_OPEN_DEMO_KEY'] = previous
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=2)
