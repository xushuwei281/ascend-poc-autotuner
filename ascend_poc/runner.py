"""Isolated, deadline-bound execution of trusted PoC commands.

The LLM never supplies commands: it supplies values in the configured search
space. All commands are argument arrays from the user-owned configuration.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


class RunnerError(RuntimeError):
    """A deployment or measurement failed; never contains command arguments."""


class TrialResults(list):
    """Backward-compatible scenario list with execution metadata for the owner.

    The list contains only measurements actually collected. Callers persist the
    attributes separately in their trial record; ordinary JSON serialization of
    the list intentionally remains compatible with the historical API.
    """

    def __init__(self, repetitions: int):
        super().__init__()
        self.status = 'failed'
        self.aborted_after = None
        self.timings = {'startup_s': 0.0, 'benchmark_s': 0.0, 'quality_s': 0.0,
                        'other_s': 0.0, 'total_s': 0.0, 'repetitions': repetitions}


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.:]*)\}")
_MAX_RESULT_BYTES = 32 * 1024 * 1024


def _string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise RunnerError("A parameter is not finite")
        return str(value)
    if isinstance(value, str):
        return value
    raise RunnerError("Only scalar parameter values are supported")


def render_argument(
    argument: str,
    candidate: dict[str, Any],
    trial_dir: Path,
    repo_dir: Path,
    env: dict[str, str],
) -> str:
    """Expand known placeholders only; preserve JSON and shell metacharacters."""
    if not isinstance(argument, str):
        raise RunnerError("Command arguments must be strings")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "trial_dir":
            return str(trial_dir)
        if key == "repo_dir":
            return str(repo_dir)
        if key.startswith("param."):
            name = key[6:]
            if name not in candidate:
                raise RunnerError(f"Unknown parameter placeholder: {name}")
            return _string(candidate[name])
        if key.startswith("env:"):
            name = key[4:]
            if name not in env:
                raise RunnerError(f"Required environment variable is missing: {name}")
            return env[name]
        raise RunnerError(f"Unknown command placeholder: {key}")

    result = _PLACEHOLDER.sub(replace, argument)
    if "\x00" in result:
        raise RunnerError("Command arguments cannot contain NUL bytes")
    return result


def render_command(
    command: list[str], candidate: dict[str, Any], trial_dir: Path,
    repo_dir: Path, env: dict[str, str],
) -> list[str]:
    if not isinstance(command, list) or not command:
        raise RunnerError("A non-empty argument array is required")
    return [render_argument(arg, candidate, trial_dir, repo_dir, env) for arg in command]


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunnerError("Global experiment time budget exhausted")
    return remaining


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_process_group(process: subprocess.Popen, grace_s: float = 2.0) -> None:
    """Stop our complete process group, even if its original leader exited."""
    pgid = process.pid  # Every process launched here has start_new_session=True.
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return
    end = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < end:
        process.poll()  # Reap the leader so it does not prolong group existence.
        if not _group_exists(pgid):
            return
        time.sleep(min(0.05, max(0.0, end - time.monotonic())))
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def run_command(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path,
    timeout_s: float, deadline: float, label: str = "Command",
) -> None:
    """Execute an argv array with bounded wall time and scoped cleanup."""
    duration = min(float(timeout_s), _remaining(deadline))
    if duration <= 0:
        raise RunnerError(f"{label} has no time remaining")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log_path.open("wb") as log:
            try:
                process = subprocess.Popen(
                    command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                    shell=False, start_new_session=True, close_fds=True,
                )
            except (OSError, ValueError) as exc:
                raise RunnerError(f"{label} could not be started ({type(exc).__name__})") from None
            try:
                code = process.wait(timeout=duration)
            except subprocess.TimeoutExpired:
                raise RunnerError(f"{label} timed out") from None
            if code != 0:
                raise RunnerError(f"{label} exited with status {code}")
    finally:
        if process is not None:
            # Shell wrappers can finish while a child remains alive.
            stop_process_group(process, min(1.0, max(0.0, deadline - time.monotonic())))


def _check_endpoint_free(health_url: str) -> None:
    try:
        parts = urlsplit(health_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError
        if parts.username or parts.password:
            raise ValueError
        port = parts.port or (443 if parts.scheme == "https" else 80)
        host = parts.hostname
    except (TypeError, ValueError):
        raise RunnerError("The health URL must be HTTP(S) without embedded credentials") from None
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise RunnerError("The configured serving endpoint is already occupied; choose a dedicated port")
    except (ConnectionRefusedError, TimeoutError, socket.timeout):
        return
    except OSError as exc:
        # Invalid hostnames and routing errors must not be mistaken for a free port.
        raise RunnerError(f"Cannot check serving endpoint ({type(exc).__name__})") from None


def _wait_healthy(process: subprocess.Popen, health_url: str, deadline: float) -> None:
    while True:
        remaining = _remaining(deadline)
        if process.poll() is not None:
            raise RunnerError("Server exited before it became healthy")
        try:
            with urlopen(health_url, timeout=min(0.5, remaining)) as response:
                if response.status == 200:
                    if process.poll() is not None:
                        raise RunnerError("Server exited during its health check")
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(min(0.1, _remaining(deadline)))


def _result_path(template: str, candidate: dict[str, Any], run_dir: Path,
                 repo_dir: Path, env: dict[str, str]) -> Path:
    path = Path(render_argument(template, candidate, run_dir, repo_dir, env))
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        raise RunnerError("Result files must be inside the current repetition directory") from None
    return path


def _read_result(path: Path) -> dict[str, Any]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate result key")
            result[key] = value
        return result

    try:
        if not path.is_file() or path.stat().st_size > _MAX_RESULT_BYTES:
            raise RunnerError("A result file is missing or exceeds the size limit")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (OSError, UnicodeError, ValueError):
        raise RunnerError("A result file is not valid finite JSON") from None
    if not isinstance(raw, dict):
        raise RunnerError("Result files must contain a JSON object")
    return raw


def _safe_record(command: list[str], original: list[str], env: dict[str, str]) -> list[str]:
    """Never persist values expanded from environment placeholders."""
    safe = []
    for expanded, template in zip(command, original):
        for name in re.findall(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", template):
            value = env.get(name)
            if value:
                expanded = expanded.replace(value, f"<env:{name}>")
        safe.append(expanded)
    return safe


def run_trial(
    config: dict[str, Any], candidate: dict[str, Any], trial_dir: Path,
    deadline: float, repetitions: int, *, phase: str | None = None,
) -> TrialResults:
    """Measure every scenario under one server configuration.

    Each repetition has a fresh directory. Hooks run in this order:
    before_each_command -> warmup_command -> measured command -> quality_command.
    Cached-state policy is therefore explicitly controlled by the user's hooks.
    Server environment overrides expand against the original inherited
    environment (with the tuning API credential removed), so overrides do not
    depend on each other or on dictionary order.
    Explicit benchmark_env creates a separate environment for all measurement
    hooks and results; omitted benchmark_env preserves legacy tune behavior.
    The server is always stopped before this function returns or raises.

    Only explicit exploration with ``budget.early_stop`` enabled may return a
    partial, ``infeasible_early`` list. Confirmation and reference measurements
    always run their full matrix. Timings use monotonic elapsed seconds:
    startup is one server launch/health wait; benchmark includes repeated setup,
    warmup and benchmark commands; quality includes repeated quality commands;
    other includes preparation, result processing and cleanup. ``repetitions``
    is the requested count per scenario, not a count of independent restarts.
    """
    from .metrics import evaluate_repetition, normalize_metrics
    from .storage import atomic_json

    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise RunnerError("Repetitions must be a positive integer")
    started = time.monotonic()
    results = TrialResults(repetitions)
    early_stop = phase == 'exploration' and config.get('budget', {}).get('early_stop', False) is True
    trial_dir = Path(trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(config.get("_repo_dir", Path.cwd())).resolve()
    api_key_env = config.get("llm", {}).get("api_key_env", "POC_LLM_API_KEY")
    inherited_env = dict(os.environ)
    inherited_env.pop(api_key_env, None)
    env = dict(inherited_env)
    env.update({
        key: render_argument(value, candidate, trial_dir, repo_dir, inherited_env)
        for key, value in config["server"].get("env", {}).items()
    })
    env.pop(api_key_env, None)
    benchmark_env = env
    if "benchmark_env" in config:
        benchmark_env = dict(inherited_env)
        benchmark_env.update({
            key: render_argument(value, {}, trial_dir, repo_dir, inherited_env)
            for key, value in config["benchmark_env"].items()
        })
        benchmark_env.pop(api_key_env, None)
    commands: list[dict[str, Any]] = []

    def record(label: str, rendered: list[str], original: list[str], effective_env: dict[str, str]) -> None:
        commands.append({"stage": label, "argv": _safe_record(rendered, original, effective_env)})
        (trial_dir / "commands.json").write_text(
            json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")

    def stage(command: list[str], run_dir: Path, label: str, timeout: float) -> None:
        rendered = render_command(command, candidate, run_dir, repo_dir, benchmark_env)
        record(label, rendered, command, benchmark_env)
        stage_started = time.monotonic()
        try:
            run_command(rendered, cwd=repo_dir, env=benchmark_env, log_path=run_dir / f"{label}.log",
                        timeout_s=timeout, deadline=deadline, label=label)
        finally:
            bucket = 'quality_s' if label == 'quality' else 'benchmark_s'
            results.timings[bucket] += max(0.0, time.monotonic() - stage_started)

    server = config["server"]
    _remaining(deadline)
    _check_endpoint_free(server["health_url"])
    argv = render_command(server["command"], candidate, trial_dir, repo_dir, env)
    record("server", argv, server["command"], env)
    process = None
    with (trial_dir / "server.log").open("wb") as log:
        try:
            startup_started = time.monotonic()
            try:
                try:
                    process = subprocess.Popen(argv, cwd=repo_dir, env=env, stdout=log,
                        stderr=subprocess.STDOUT, shell=False, start_new_session=True, close_fds=True)
                except (OSError, ValueError) as exc:
                    raise RunnerError(f"Server could not be started ({type(exc).__name__})") from None
                _wait_healthy(process, server["health_url"],
                              min(deadline, time.monotonic() + server.get("startup_timeout_s", 600)))
            finally:
                results.timings['startup_s'] = max(0.0, time.monotonic() - startup_started)
            (trial_dir / "server-process.json").write_text(
                json.dumps({"pid": process.pid, "started_at_unix": time.time()}), encoding="utf-8")
            for scenario in config["scenarios"]:
                collected = []
                workloads = []
                # Include an index: scenario labels can never escape the output directory.
                scenario_index = len(results)
                for repeat in range(repetitions):
                    _remaining(deadline)
                    if process.poll() is not None:
                        raise RunnerError("Server exited during measurement")
                    run_dir = trial_dir / f"scenario-{scenario_index:02d}" / f"repeat-{repeat:03d}"
                    try:
                        run_dir.mkdir(parents=True, exist_ok=False)
                    except FileExistsError:
                        raise RunnerError("Measurement directory already exists; stale results are not reusable") from None
                    timeout = float(scenario.get("timeout_s", 600))
                    for hook in ("before_each_command", "warmup_command"):
                        if scenario.get(hook):
                            stage(scenario[hook], run_dir, hook.removesuffix("_command"), timeout)
                    measured_file = _result_path(scenario["result_file"], candidate, run_dir, repo_dir, benchmark_env)
                    if measured_file.exists():
                        raise RunnerError("A hook created the measured result file before measurement")
                    telemetry_path = None
                    if scenario.get("telemetry_file"):
                        telemetry_path = _result_path(scenario["telemetry_file"], candidate, run_dir, repo_dir, benchmark_env)
                        if telemetry_path.exists() or telemetry_path == measured_file:
                            raise RunnerError("Telemetry must be a fresh, separate measurement artifact")
                    declared_outputs = [measured_file]
                    if telemetry_path is not None:
                        declared_outputs.append(telemetry_path)
                    if scenario.get('quality_result_file'):
                        declared_outputs.append(_result_path(scenario['quality_result_file'], candidate, run_dir, repo_dir, benchmark_env))
                    reserved = {run_dir / 'normalized.json', run_dir / 'workload-summary.json'}
                    if any(path in reserved for path in declared_outputs) or len(set(declared_outputs)) != len(declared_outputs):
                        raise RunnerError('Measurement outputs must be distinct from controller artifacts and each other')
                    stage(scenario["command"], run_dir, "benchmark", timeout)
                    measured_file = _result_path(scenario["result_file"], candidate, run_dir, repo_dir, benchmark_env)
                    try:
                        metrics = normalize_metrics(_read_result(measured_file), scenario.get("metrics_format", "canonical"))
                    except ValueError:
                        raise RunnerError("Benchmark output contains invalid metrics") from None
                    if telemetry_path is not None:
                        from .workload import load_telemetry
                        try:
                            workload = load_telemetry(telemetry_path)
                        except (ValueError, OSError):
                            raise RunnerError("Workload telemetry is missing or invalid") from None
                        if ('completed' in metrics and metrics['completed'] != workload['successful']) or ('failed' in metrics and metrics['failed'] != workload['failed']):
                            raise RunnerError("Workload telemetry counts disagree with benchmark metrics")
                        workloads.append(workload)
                        (run_dir / 'workload-summary.json').write_text(json.dumps(workload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                    if scenario.get("quality_command"):
                        quality_file = _result_path(scenario["quality_result_file"], candidate, run_dir, repo_dir, benchmark_env)
                        if quality_file.exists():
                            raise RunnerError("Quality result file exists before its command")
                        stage(scenario["quality_command"], run_dir, "quality", timeout)
                        quality_file = _result_path(scenario["quality_result_file"], candidate, run_dir, repo_dir, benchmark_env)
                        try:
                            quality = normalize_metrics(_read_result(quality_file), "canonical")
                        except ValueError:
                            raise RunnerError("Quality output contains invalid metrics") from None
                        if set(quality) & set(metrics):
                            raise RunnerError("Quality metrics cannot overwrite benchmark metrics")
                        metrics.update(quality)
                    if process.poll() is not None:
                        raise RunnerError("Server exited during measurement")
                    collected.append(metrics)
                    (run_dir / "normalized.json").write_text(
                        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                    if early_stop:
                        observed = evaluate_repetition(config, scenario, metrics, repeat + 1)
                        if observed['failures']:
                            results.status = 'infeasible_early'
                            results.aborted_after = {'scenario': scenario['name'], 'repetition': repeat + 1,
                                                     'violated_metrics': observed['violated_metrics']}
                            break
                result = {"name": scenario["name"], "metrics": collected}
                if workloads:
                    result['workload_summaries'] = workloads
                results.append(result)
                if results.status == 'infeasible_early':
                    break
            if results.status != 'infeasible_early':
                results.status = 'completed'
            return results
        except KeyboardInterrupt:
            results.status = 'interrupted'
            raise
        finally:
            if process is not None:
                grace = min(float(server.get("shutdown_timeout_s", 20)),
                            max(0.0, deadline - time.monotonic()))
                stop_process_group(process, grace)
            results.timings['total_s'] = max(0.0, time.monotonic() - started)
            accounted = sum(results.timings[key] for key in ('startup_s', 'benchmark_s', 'quality_s'))
            results.timings['other_s'] = max(0.0, results.timings['total_s'] - accounted)
            atomic_json(trial_dir / 'trial-execution.json', {
                'schema_version': 1, 'phase': phase, 'status': results.status,
                'aborted_after': results.aborted_after, 'timings': results.timings,
            })
