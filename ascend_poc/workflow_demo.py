"""Complete local wire-level A2A -> MCP -> evidence -> LLM -> subprocess demo."""
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
from .storage import atomic_json
from .strategy import load_workflow
from .workflow import run_workflow


def run_workflow_demo(output: Path) -> dict:
    """No external HTTP traffic, NPU, or paid LLM API is used."""
    class WireStub(BaseHTTPRequestHandler):
        proposals = 0

        def send_json(self, value, status=200):
            encoded = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/mcp':
                method = body['method']
                if method == 'notifications/initialized':
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == 'initialize':
                    result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'Synthetic DeepWiki fixture', 'version': '1'}}
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'ask_wiki_question', 'description': 'Synthetic repository question',
                        'inputSchema': {'type': 'object', 'properties': {'repoName': {'type': 'string'}, 'question': {'type': 'string'}},
                                        'required': ['repoName', 'question']}}]}
                elif method == 'tools/call':
                    result = {'content': [{'type': 'text', 'text':
                        'SYNTHETIC reference for integration testing. Graph/decode changes are hypotheses; '
                        'test the full fixed latency and quality contract. The demo profile is not a real NPU feature. '
                        'Reference repository https://deepwiki.com/vllm-project/vllm-ascend'}]}
                else:
                    self.send_json({'jsonrpc': '2.0', 'id': body.get('id'), 'error': {'code': -32601, 'message': 'Unknown method'}})
                    return
                self.send_json({'jsonrpc': '2.0', 'id': body['id'], 'result': result})
            elif self.path == '/v1/chat/completions':
                prompt = json.loads(body['messages'][-1]['content'])
                options = [('decode', 64, 2048), ('decode', 128, 4096), ('baseline', 64, 2048)]
                profile, seqs, tokens = options[type(self).proposals % len(options)]
                type(self).proposals += 1
                proposal = {'profile_id': profile, 'parameters': {'max_num_seqs': seqs, 'max_num_batched_tokens': tokens},
                            'hypothesis': 'Synthetic strategy experiment; validate latency and quality after changing decode behavior',
                            'evidence_ids': [item['id'] for item in prompt['evidence']]}
                self.send_json({'choices': [{'message': {'content': json.dumps(proposal)}}],
                                'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'workflow_state.json').exists():
        raise ValueError('Demo output already exists; choose a new output directory')
    stub = ThreadingHTTPServer(('127.0.0.1', 0), WireStub)
    stub_thread = threading.Thread(target=stub.serve_forever, daemon=True)
    stub_thread.start()
    bridge = create_research_server(port=0, researcher=DeepWikiClient(f'http://127.0.0.1:{stub.server_port}/mcp', timeout_s=5), timeout_s=5)
    bridge_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    bridge_thread.start()
    previous = os.environ.get('POC_DEMO_KEY')
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            serving_port = sock.getsockname()[1]
        config = make_config(serving_port, stub.server_port)
        config['name'] = 'Synthetic workflow contract — NOT NPU measurements'
        atomic_json(output / 'demo-contract.json', config)
        alternate = copy.deepcopy(config['server'])
        alternate['command'] += ['--strategy', 'decode-optimized']
        compatibility = {'status': 'verified', 'evidence': ['Local synthetic fixture only; not hardware compatibility']}
        spec = {'schema_version': 1, 'name': 'Synthetic full workflow demo', 'contract': 'demo-contract.json',
                'verification': {'mode': 'aba'},
                'research': {'transport': 'a2a', 'endpoint': bridge.endpoint, 'timeout_s': 5, 'max_calls': 3, 'on_error': 'stop'},
                'profiles': [{'id': 'baseline', 'description': 'Unchanged synthetic server', 'compatibility': compatibility},
                             {'id': 'decode', 'description': 'Synthetic decode optimization fixture', 'server': alternate,
                              'compatibility': compatibility}]}
        atomic_json(output / 'demo-workflow.json', spec)
        normalized = load_workflow(output / 'demo-workflow.json')
        normalized['_contract']['_repo_dir'] = str(Path(__file__).resolve().parents[1])
        os.environ['POC_DEMO_KEY'] = 'local-fixture-not-a-real-key'
        print('DEMO: all metrics, LLM answers and DeepWiki answers are synthetic local fixtures.', flush=True)
        return run_workflow(normalized, output)
    finally:
        if previous is None:
            os.environ.pop('POC_DEMO_KEY', None)
        else:
            os.environ['POC_DEMO_KEY'] = previous
        bridge.shutdown()
        bridge.server_close()
        bridge_thread.join(timeout=2)
        stub.shutdown()
        stub.server_close()
        stub_thread.join(timeout=2)
