"""Exercise the real HTTP API client, subprocess runner and confirmation loop."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading

from .config import load_config
from .storage import atomic_json
from .tuner import tune


def make_config(server_port: int, llm_port: int) -> dict:
    return {
        'schema_version': 1, 'name': 'CPU synthetic demo — NOT Ascend performance',
        'resource_id': f'demo-{server_port}',
        'context': {'hardware': 'CPU mock', 'purpose': 'synthetic integration test'},
        'llm': {'base_url': f'http://127.0.0.1:{llm_port}/v1', 'model': 'mock-planner', 'api_key_env': 'POC_DEMO_KEY', 'timeout_s': 5},
        'search_space': {'max_num_seqs': [32, 64, 128], 'max_num_batched_tokens': [1024, 2048, 4096]},
        'baseline': {'max_num_seqs': 32, 'max_num_batched_tokens': 1024},
        'budget': {'max_trials': 4, 'max_wall_time_s': 90, 'repetitions': 2, 'final_repetitions': 2, 'finalists': 2},
        'objective': {'metric': 'output_tokens_per_second', 'direction': 'maximize', 'scenario_aggregation': 'min'},
        'requirements': {'p99_ttft_ms': {'max': 1800}, 'p99_tpot_ms': {'max': 50}, 'error_rate': {'max': 0}, 'quality_score': {'min': .98}},
        'server': {'command': [sys.executable, '-m', 'ascend_poc.mock', 'serve', '--port', str(server_port), '--seqs', '{param.max_num_seqs}', '--tokens', '{param.max_num_batched_tokens}'], 'health_url': f'http://127.0.0.1:{server_port}/health', 'startup_timeout_s': 5, 'shutdown_timeout_s': 1},
        'scenarios': [{'name': 'synthetic', 'command': [sys.executable, '-m', 'ascend_poc.mock', 'bench', '--port', str(server_port), '--result', '{trial_dir}/metrics.json'], 'result_file': '{trial_dir}/metrics.json', 'metrics_format': 'canonical', 'timeout_s': 5}]
    }


def run_demo(output: Path) -> dict:
    class PlannerStub(BaseHTTPRequestHandler):
        index = 0

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path != '/v1/chat/completions' or not body.get('messages'):
                self.send_error(400)
                return
            candidates = [(64, 2048), (128, 4096), (128, 1024)]
            seqs, tokens = candidates[type(self).index % len(candidates)]
            type(self).index += 1
            proposal = {'parameters': {'max_num_seqs': seqs, 'max_num_batched_tokens': tokens}, 'hypothesis': 'Synthetic proposal for exercising the API and acceptance gates'}
            response = {'choices': [{'message': {'content': json.dumps(proposal)}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 30}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_):
            pass

    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        server_port = sock.getsockname()[1]
    server = ThreadingHTTPServer(('127.0.0.1', 0), PlannerStub)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    previous = os.environ.get('POC_DEMO_KEY')
    try:
        os.environ['POC_DEMO_KEY'] = 'local-test-key-not-a-real-credential'
        config_path = output / 'demo-input.json'
        atomic_json(config_path, make_config(server_port, server.server_port))
        config = load_config(config_path)
        config['_repo_dir'] = str(Path(__file__).resolve().parents[1])
        print('DEMO: synthetic measurements and local mock LLM API; no NPU or paid API used.', flush=True)
        return tune(config, output, planner_name='llm')
    finally:
        if previous is None:
            os.environ.pop('POC_DEMO_KEY', None)
        else:
            os.environ['POC_DEMO_KEY'] = previous
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
