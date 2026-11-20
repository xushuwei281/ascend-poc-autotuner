"""Synthetic CPU-only serving/benchmark fixture. Never actual NPU measurements."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.request


class HealthHandler(BaseHTTPRequestHandler):
    configuration = {'seqs': 32, 'tokens': 1024}

    def do_GET(self):
        self.send_response(200 if self.path in ['/health', '/config'] else 404)
        self.end_headers()
        self.wfile.write(json.dumps(self.configuration if self.path == '/config' else {}).encode())

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'bench'])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--result')
    parser.add_argument('--seqs', type=int, default=32)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--strategy', choices=['baseline', 'decode-optimized'], default='baseline')
    args = parser.parse_args()
    if args.mode == 'serve':
        HealthHandler.configuration = {'seqs': args.seqs, 'tokens': args.tokens, 'strategy': args.strategy}
        ThreadingHTTPServer(('127.0.0.1', args.port), HealthHandler).serve_forever()
    else:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
            if response.status != 200:
                raise RuntimeError('Mock server unavailable')
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/config', timeout=2) as response:
            effective = json.load(response)
        args.seqs, args.tokens = effective['seqs'], effective['tokens']
        metrics = {
            'output_tokens_per_second': 300 + 4 * args.seqs + args.tokens / 20,
            'p99_ttft_ms': 900 + args.seqs * 8 - args.tokens / 10,
            'p99_tpot_ms': 20 + args.tokens / 100 + args.seqs / 10,
            'completed': 1000, 'failed': 0, 'quality_score': .99,
        }
        if effective.get('strategy') == 'decode-optimized':
            # Synthetic only: exercise a strategy switch without any NPU claims.
            metrics['output_tokens_per_second'] *= 1.1
            metrics['p99_tpot_ms'] *= .9
        target = Path(args.result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics), encoding='utf-8')


if __name__ == '__main__':
    main()
