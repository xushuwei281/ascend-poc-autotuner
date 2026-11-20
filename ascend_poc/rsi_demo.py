"""CPU-only wire demo of research, experiments, reflection and RSI promotion.

The synthetic planner follows a deterministic script. It exercises the real HTTP
client and controller, not the intelligence or performance of an external LLM.
"""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading

from .a2a import create_research_server
from .deepwiki import DeepWikiClient
from .demo import make_config
from .rsi_config import load_agent
from .storage import atomic_json


def run_agent_demo(output: Path, *, verification_mode='aba') -> dict:
    """Run an actual local HTTP/subprocess campaign with synthetic measurements."""
    # Imported here so the demo module remains a lightweight CLI dependency.
    from .rsi import run_agent

    actions = [
        {'tool': 'inspect', 'arguments': {}},
        {'tool': 'recall', 'arguments': {}},
        {'tool': 'research', 'arguments': {'topic': 'decode'}},
        {'tool': 'experiment', 'arguments': {
            'profile_id': 'decode',
            'parameters': {'max_num_seqs': 64, 'max_num_batched_tokens': 2048},
            'hypothesis': 'The synthetic decode recipe can improve throughput while meeting all fixed gates.',
            'expected_effect': 'Higher measured output throughput with TPOT below the unchanged limit.'}},
        {'tool': 'reflect', 'arguments': {
            'experiment_id': 'experiment-0000',
            'lesson': 'The moderate synthetic decode configuration passed independent ABA verification; '
                      'this evidence applies only to this synthetic contract and environment.'}},
        {'tool': 'promote', 'arguments': {'experiment_id': 'experiment-0000'}},
        {'tool': 'experiment', 'arguments': {
            'profile_id': 'decode',
            'parameters': {'max_num_seqs': 128, 'max_num_batched_tokens': 4096},
            'hypothesis': 'A larger synthetic batch may raise throughput, but can breach the fixed TPOT gate.',
            'expected_effect': 'Test the throughput/TPOT tradeoff against the newly promoted reference.'}},
        {'tool': 'reflect', 'arguments': {
            'experiment_id': 'experiment-0001',
            'lesson': 'The aggressive synthetic configuration fails the fixed TPOT requirement; '
                      'higher throughput does not qualify it for promotion.'}},
        {'tool': 'finish', 'arguments': {
            'reason': 'Retain the independently improved moderate configuration and the rejected counterexample.'}},
    ]

    class WireStub(BaseHTTPRequestHandler):
        action_index = 0

        def send_json(self, value, status=200):
            encoded = json.dumps(value).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > 1024 * 1024:
                    raise ValueError('Invalid local fixture request size')
                body = json.loads(self.rfile.read(size))
            except (TypeError, ValueError):
                self.send_error(400)
                return
            if self.path == '/v1/chat/completions':
                if not body.get('messages'):
                    self.send_error(400)
                    return
                index = min(type(self).action_index, len(actions) - 1)
                type(self).action_index += 1
                self.send_json({
                    'choices': [{'message': {'content': json.dumps(actions[index])}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                })
            elif self.path == '/mcp':
                method = body.get('method')
                if method == 'notifications/initialized':
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == 'initialize':
                    result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'Synthetic DeepWiki fixture', 'version': '1'}}
                elif method == 'tools/list':
                    result = {'tools': [{
                        'name': 'ask_wiki_question', 'description': 'Synthetic repository research',
                        'inputSchema': {'type': 'object', 'properties': {
                            'repoName': {'type': 'string'}, 'question': {'type': 'string'}},
                            'required': ['repoName', 'question']},
                    }]}
                elif method == 'tools/call':
                    result = {'content': [{'type': 'text', 'text':
                        'SYNTHETIC research fixture, not hardware advice. A larger decode batch can '
                        'trade latency for throughput. Test the complete fixed quality and latency '
                        'contract, then independently compare against the incumbent. No NPU support '
                        'or performance is established by this answer. Reference repository '
                        'https://deepwiki.com/vllm-project/vllm-ascend'}]}
                else:
                    self.send_json({'jsonrpc': '2.0', 'id': body.get('id'),
                                    'error': {'code': -32601, 'message': 'Unknown fixture method'}})
                    return
                self.send_json({'jsonrpc': '2.0', 'id': body.get('id'), 'result': result})
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Agent demo requires a fresh, empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'input'
    inputs.mkdir()
    stub = ThreadingHTTPServer(('127.0.0.1', 0), WireStub)
    stub_thread = threading.Thread(target=stub.serve_forever, daemon=True)
    bridge = bridge_thread = None
    previous = os.environ.get('POC_DEMO_KEY')
    stub_thread.start()
    try:
        bridge = create_research_server(
            port=0,
            researcher=DeepWikiClient(f'http://127.0.0.1:{stub.server_port}/mcp', timeout_s=5),
            timeout_s=5,
        )
        bridge_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        bridge_thread.start()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            serving_port = sock.getsockname()[1]
        contract = make_config(serving_port, stub.server_port)
        contract['name'] = 'Synthetic RSI contract — NOT Ascend performance'
        alternate = copy.deepcopy(contract['server'])
        alternate['command'] += ['--strategy', 'decode-optimized']
        compatibility = {'status': 'verified', 'evidence': [
            'Synthetic fixture only; this is not evidence of hardware compatibility']}
        workflow = {
            'schema_version': 1, 'name': 'Synthetic RSI strategy catalog',
            'contract': 'contract.json', 'verification': {'mode': verification_mode},
            'budget': {'early_stop': True, 'confirmation_reserve': True, 'cost_safety_factor': 1.25},
            'research': {'transport': 'a2a', 'endpoint': bridge.endpoint,
                         'timeout_s': 5, 'max_calls': 3, 'on_error': 'stop'},
            'profiles': [
                {'id': 'baseline', 'description': 'Unchanged synthetic baseline',
                 'compatibility': compatibility},
                {'id': 'decode', 'description': 'Synthetic decode fixture',
                 'server': alternate, 'compatibility': compatibility},
            ],
        }
        if verification_mode == 'aba_ci':
            workflow['verification'].update(ci_blocks=5, ci_confidence=.95, ci_max_comparisons=3)
        config = {
            'schema_version': 1, 'name': 'Synthetic bounded Inference RSI Agent',
            'goal': 'Exercise real research and experiment tools; promote only a verified improvement '
                    'and retain a failed candidate as a synthetic counterexample.',
            'workflow': 'workflow.json', 'memory_file': '../memory.json',
            'settings': {'max_steps': 20, 'max_experiments': 3, 'max_wall_time_s': 180,
                         'experiment_timeout_s': 120 if verification_mode == 'aba_ci' else 60, 'max_llm_calls': 30,
                         'max_research_calls': 3, 'max_consecutive_tool_errors': 3},
        }
        atomic_json(inputs / 'contract.json', contract)
        atomic_json(inputs / 'workflow.json', workflow)
        atomic_json(inputs / 'agent.json', config)
        normalized = load_agent(inputs / 'agent.json')
        normalized['_workflow']['_contract']['_repo_dir'] = str(Path(__file__).resolve().parents[1])
        os.environ['POC_DEMO_KEY'] = 'synthetic-local-fixture-not-a-real-key'
        print('DEMO: synthetic planner, research and performance; real local HTTP and subprocess '
              'execution. No NPU or paid API is used.', flush=True)
        return run_agent(normalized, output)
    finally:
        if previous is None:
            os.environ.pop('POC_DEMO_KEY', None)
        else:
            os.environ['POC_DEMO_KEY'] = previous
        if bridge is not None:
            bridge.shutdown()
            bridge.server_close()
        if bridge_thread is not None:
            bridge_thread.join(timeout=2)
        stub.shutdown()
        stub.server_close()
        stub_thread.join(timeout=2)
