"""Synthetic CPU fixture for the open tuning demo, never NPU performance.

The benchmark deliberately uses a deterministic formula so the demo can show a
successful candidate, a latency failure and a corrected candidate reproducibly.
Neither the numbers nor these option values are hardware recommendations.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.request import urlopen


def synthetic_metrics(configuration: dict, concurrency: int) -> dict:
    """Return clearly labelled fixture measurements, not measured inference."""
    throughput = 100 + 5 * concurrency
    if configuration.get('prefix_cache'):
        throughput += 30
    if configuration.get('compilation', {}).get('mode') == 3:
        throughput += 20
    if configuration.get('tensor_parallel_size') == 2:
        throughput += 20
    return {
        'output_tokens_per_second': throughput,
        'p99_tpot_ms': 10 + concurrency,
        'p99_ttft_ms': 100 + concurrency * 2,
        'completed': 100,
        'failed': 0,
        'synthetic_fixture': 1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('model')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, required=True)
    serve.add_argument('--served-model-name', default='synthetic-demo')
    serve.add_argument('--max-num-seqs', type=int, default=32)
    serve.add_argument('--max-num-batched-tokens', type=int, default=2048)
    serve.add_argument('--gpu-memory-utilization', type=float, default=.85)
    serve.add_argument('--tensor-parallel-size', type=int, default=1)
    cache = serve.add_mutually_exclusive_group()
    cache.add_argument('--enable-prefix-caching', dest='prefix_cache', action='store_true')
    cache.add_argument('--disable-prefix-caching', dest='prefix_cache', action='store_false')
    serve.set_defaults(prefix_cache=False)
    serve.add_argument('--compilation-config', type=json.loads, default={})
    benchmark = commands.add_parser('benchmark')
    benchmark.add_argument('--base-url', required=True)
    benchmark.add_argument('--concurrency', type=int, required=True)
    benchmark.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'benchmark':
        if args.concurrency < 1:
            parser.error('concurrency must be positive')
        with urlopen(args.base_url.rstrip('/') + '/config', timeout=3) as response:
            configuration = json.load(response)
        metrics = synthetic_metrics(configuration, args.concurrency)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics), encoding='utf-8')
        return 0

    if not Path(args.model).exists():
        parser.error('synthetic model fixture directory is missing')
    if not isinstance(args.compilation_config, dict):
        parser.error('compilation config must be a JSON object')
    configuration = {
        'prefix_cache': args.prefix_cache,
        'compilation': args.compilation_config,
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_num_seqs': args.max_num_seqs,
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {'/health', '/config'}:
                self.send_error(404)
                return
            encoded = json.dumps(configuration if self.path == '/config' else {
                'synthetic_fixture': True}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        server.serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
