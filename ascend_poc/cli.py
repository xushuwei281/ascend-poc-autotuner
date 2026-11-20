"""Command line interface; run on the dedicated Ascend PoC host/container."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys

from .config import load_config, _json_object, _reject_constant
from .demo import run_demo
from .tuner import tune


def doctor() -> dict:
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {}}
    for package in ['vllm', 'vllm-ascend', 'torch', 'torch-npu']:
        try:
            info['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info['packages'][package] = None
    info['npu_smi_available'] = shutil.which('npu-smi') is not None
    if info['npu_smi_available']:
        try:
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=10)
            info['npu_smi'] = result.stdout[:20_000]
        except (OSError, subprocess.TimeoutExpired):
            info['npu_smi'] = 'unavailable'
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='LLM-assisted Ascend PoC acceptance tuning')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Validate a customer contract without execution')
    validate.add_argument('--config', type=Path, required=True)
    start = commands.add_parser('tune', help='Run a baseline, LLM proposals, and final confirmation')
    start.add_argument('--config', type=Path, required=True)
    start.add_argument('--output', type=Path, required=True)
    start.add_argument('--planner', choices=['llm', 'random'], default='llm')
    start.add_argument('--resume', action='store_true')
    start.add_argument('--workdir', type=Path, default=Path.cwd(), help='Working directory of all fixed commands')
    demo = commands.add_parser('demo', help='CPU-only synthetic end-to-end demonstration')
    demo.add_argument('--output', type=Path, default=Path('runs/demo'))
    flow = commands.add_parser('workflow', help='Research, select inference strategies, benchmark and confirm')
    flow.add_argument('--config', type=Path, required=True)
    flow.add_argument('--output', type=Path, required=True)
    flow.add_argument('--resume', action='store_true')
    flow.add_argument('--workdir', type=Path, default=Path.cwd())
    flow_validate = commands.add_parser('workflow-validate', help='Validate immutable workload and strategy profiles')
    flow_validate.add_argument('--config', type=Path, required=True)
    flow_demo = commands.add_parser('workflow-demo', help='Local synthetic A2A, MCP, LLM and benchmark workflow')
    flow_demo.add_argument('--output', type=Path, default=Path('runs/workflow-demo'))
    research = commands.add_parser('research-agent', help='Loopback A2A 0.3 research bridge to DeepWiki MCP')
    research.add_argument('--port', type=int, default=8766)
    research.add_argument('--mcp-endpoint', default='https://mcp.deepwiki.com/mcp')
    research.add_argument('--timeout', type=float, default=60)
    commands.add_parser('doctor', help='Print local versions and NPU inventory; no API calls')
    workload = commands.add_parser('workload-analyze', help='Summarize measured request metadata without prompts or identities')
    workload.add_argument('--input', type=Path, required=True)
    workload.add_argument('--output', type=Path, required=True)
    adaptive = commands.add_parser('adaptive-plan', help='Predict a P/D split in shadow mode; never change a running service')
    adaptive.add_argument('--config', type=Path, required=True)
    adaptive.add_argument('--windows', type=Path, required=True)
    adaptive.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('report', help='Regenerate report from saved measurements')
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == 'doctor':
            print(json.dumps(doctor(), ensure_ascii=False, indent=2))
        elif args.command == 'workload-analyze':
            from .workload import load_telemetry
            from .storage import atomic_json
            atomic_json(args.output, load_telemetry(args.input))
            print(f'Workload summary: {args.output.resolve()}')
        elif args.command == 'adaptive-plan':
            from .adaptive import advise_pd
            from .storage import atomic_json
            def read_bounded(path):
                if path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError('Adaptive input exceeds 8 MiB')
                return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_json_object, parse_constant=_reject_constant)
            decision = advise_pd(read_bounded(args.config), read_bounded(args.windows))
            atomic_json(args.output, decision)
            print(f"Shadow decision: {decision['action']}; no backend changes. {args.output.resolve()}")
        elif args.command == 'validate':
            config = load_config(args.config)
            print(f"Valid contract: {config['name']}; {len(config['scenarios'])} scenarios")
        elif args.command == 'demo':
            run_demo(args.output)
            print(f"Demo report: {args.output.resolve() / 'report.md'}")
        elif args.command == 'research-agent':
            from .a2a import serve_research_agent
            serve_research_agent(port=args.port, mcp_endpoint=args.mcp_endpoint, timeout_s=args.timeout)
        elif args.command == 'workflow-demo':
            from .workflow_demo import run_workflow_demo
            run_workflow_demo(args.output)
            print(f"Workflow demo report: {args.output.resolve() / 'workflow_report.md'}")
        elif args.command in {'workflow', 'workflow-validate'}:
            from .strategy import load_workflow, eligible_profiles
            from .workflow import run_workflow
            spec = load_workflow(args.config)
            if args.command == 'workflow-validate':
                print(f"Valid workflow: {spec['name']}; {len(eligible_profiles(spec))} eligible strategies")
            else:
                spec['_contract']['_repo_dir'] = str(args.workdir.resolve())
                run_workflow(spec, args.output, resume=args.resume)
                print(f"Workflow report: {args.output.resolve() / 'workflow_report.md'}")
                if not (args.output / 'best.json').exists():
                    print('No independently confirmed feasible strategy.', file=sys.stderr)
                    return 2
        elif args.command == 'report':
            from .reporting import regenerate_report
            print(regenerate_report(args.output))
        else:
            config = load_config(args.config)
            config['_repo_dir'] = str(args.workdir.resolve())
            # Local runtime information is never included in planner requests.
            args.output.mkdir(parents=True, exist_ok=True)
            state = tune(config, args.output, planner_name=args.planner, resume=args.resume)
            print(f"Report: {args.output.resolve() / 'report.md'}")
            if not (args.output / 'best.json').exists():
                print('No fully confirmed feasible candidate. Inspect the acceptance report.', file=sys.stderr)
                return 2
        return 0
    except KeyboardInterrupt:
        print('Interrupted; owned server processes cleaned up. Resume with the same contract/output.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
