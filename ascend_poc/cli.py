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
    opened = commands.add_parser('autotune', help='Optimize all discovered inference options from a natural-language request')
    opened.add_argument('--runtime', type=Path, required=True)
    request = opened.add_mutually_exclusive_group(required=True)
    request.add_argument('--request', help='Natural-language customer requirements')
    request.add_argument('--request-file', type=Path, help='UTF-8 text requirements')
    opened.add_argument('--output', type=Path, required=True)
    opened.add_argument('--workdir', type=Path, default=Path.cwd())
    opened.add_argument('--plan-only', action='store_true', help='Discover options and compile intent without starting a model service')
    opened.add_argument('--resume', action='store_true')
    opened_validate = commands.add_parser('autotune-validate', help='Validate open-mode runtime wiring without API or hardware execution')
    opened_validate.add_argument('--runtime', type=Path, required=True)
    opened_demo = commands.add_parser('autotune-demo', help='CPU-only synthetic natural-language open-parameter demo')
    opened_demo.add_argument('--output', type=Path, default=Path('runs/open-demo'))
    opened_report = commands.add_parser('autotune-report', help='Verify open-mode evidence and regenerate its report')
    opened_report.add_argument('--output', type=Path, required=True)
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
    flow_validate.add_argument('--estimate', action='store_true', help='Show confirmation costs from budget.estimated_trial_s, without execution')
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
    agent = commands.add_parser('agent', help='Run the bounded inference RSI tool-using agent')
    agent.add_argument('--config', type=Path, required=True)
    agent.add_argument('--output', type=Path, required=True)
    agent.add_argument('--workdir', type=Path, default=Path.cwd())
    agent.add_argument('--resume', action='store_true')
    agent.add_argument('--ignore-memory', action='store_true', help='Isolated control run: neither read nor write persistent experience')
    agent_validate = commands.add_parser('agent-validate', help='Validate an inference RSI agent configuration')
    agent_validate.add_argument('--config', type=Path, required=True)
    agent_demo = commands.add_parser('agent-demo', help='CPU-only synthetic tool-using RSI agent demo')
    agent_demo.add_argument('--output', type=Path, default=Path('runs/rsi-demo'))
    agent_demo.add_argument('--verification', choices=['aba', 'aba_ci'], default='aba')
    agent_report = commands.add_parser('agent-report', help='Verify agent evidence and regenerate its report')
    agent_report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command in {'autotune', 'autotune-validate'}:
            from .open_runtime import load_runtime
            runtime = load_runtime(args.runtime)
            if args.command == 'autotune-validate':
                print(f"Valid open tuning runtime: {runtime['name']}")
            else:
                from .open_tuner import run_autotune
                if args.request_file:
                    if args.request_file.stat().st_size > 100_000:
                        raise ValueError('Request text exceeds 100 KiB')
                    text = args.request_file.read_text(encoding='utf-8')
                else:
                    text = args.request
                state = run_autotune(runtime, text, args.output, workdir=args.workdir,
                                     plan_only=args.plan_only, resume=args.resume)
                print(args.output.resolve() / 'report.md')
                if state['status'] == 'needs_input' or not args.plan_only and not state.get('champion'):
                    print(state.get('stop_reason', 'No independently improved champion.'), file=sys.stderr)
                    return 2
        elif args.command == 'autotune-demo':
            from .open_demo import run_open_demo
            run_open_demo(args.output)
            print(args.output.resolve() / 'report.md')
        elif args.command == 'autotune-report':
            from .open_tuner import report_autotune
            print(report_autotune(args.output))
        elif args.command in {'agent', 'agent-validate'}:
            from .rsi_config import load_agent
            config = load_agent(args.config)
            if args.command == 'agent-validate':
                print(f"Valid inference RSI agent: {config['name']}")
            else:
                from .rsi import run_agent
                config['_workflow']['_contract']['_repo_dir'] = str(args.workdir.resolve())
                state = run_agent(config, args.output, resume=args.resume, ignore_memory=args.ignore_memory)
                print(args.output.resolve() / 'agent_report.md')
                if not state['champion'].get('experiment_id'):
                    print('No independently improved champion was promoted.', file=sys.stderr)
                    return 2
        elif args.command == 'agent-demo':
            from .rsi_demo import run_agent_demo
            run_agent_demo(args.output, verification_mode=args.verification)
            print(args.output.resolve() / 'agent_report.md')
        elif args.command == 'agent-report':
            from .rsi import report_agent
            print(report_agent(args.output))
        elif args.command == 'doctor':
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
                if args.estimate:
                    from .budget import confirmation_cost
                    print(json.dumps(confirmation_cost([], spec['budget'], spec['verification'],
                        spec['profiles'][0]['id'], [None] * spec['budget']['finalists']), indent=2))
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
