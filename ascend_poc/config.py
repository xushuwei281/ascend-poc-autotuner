"""Validation for the trusted PoC contract and untrusted LLM candidates.

Only values from the operator's finite search space may be proposed by the LLM.
Commands, workloads, acceptance thresholds and API destinations are never supplied
by the planner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _error(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _object(value, path: str) -> dict:
    if not isinstance(value, dict):
        _error(path, "must be an object")
    return value


def _keys(value: dict, allowed: set[str], path: str) -> None:
    extra = set(value) - allowed
    if extra:
        _error(path, f"unknown fields: {', '.join(sorted(extra))}")


def _string(value, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()) or "\x00" in value:
        _error(path, "must be a non-empty string without NUL characters")
    return value


def _number(value, path: str, *, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a number")
    if not math.isfinite(value):
        _error(path, "must be finite")
    if integer and not isinstance(value, int):
        _error(path, "must be an integer")
    if minimum is not None and value < minimum:
        _error(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _error(path, f"must be <= {maximum}")
    return value


def _scalar(value, path: str):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(value, path)
    if isinstance(value, str):
        return _string(value, path, empty=True)
    _error(path, "must be a string, finite number or boolean")


def _equal(left, right) -> bool:
    # Python otherwise considers True == 1, which would widen the allowlist.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _member(value, choices) -> bool:
    return any(_equal(value, choice) for choice in choices)


def _argv(value, path: str) -> None:
    if not isinstance(value, list) or not value:
        _error(path, "must be a non-empty argv array (shell strings are not supported)")
    for index, token in enumerate(value):
        _string(token, f"{path}[{index}]", empty=index > 0)


def _url(value, path: str) -> None:
    _string(value, path)
    try:
        url = urlsplit(value)
        valid = url.scheme in {"http", "https"} and url.hostname and not url.username and not url.password
        url.port
    except ValueError:
        valid = False
    if not valid:
        _error(path, "must be an http(s) URL without embedded credentials")


def _requirements(value, path: str) -> None:
    _object(value, path)
    for metric, limits in value.items():
        _string(metric, f"{path} metric")
        _object(limits, f"{path}.{metric}")
        _keys(limits, {"min", "max"}, f"{path}.{metric}")
        if not limits:
            _error(f"{path}.{metric}", "must contain min and/or max")
        for bound, threshold in limits.items():
            _number(threshold, f"{path}.{metric}.{bound}")
        if "min" in limits and "max" in limits and limits["min"] > limits["max"]:
            _error(f"{path}.{metric}", "min cannot exceed max")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_config(path: str | Path) -> dict:
    """Read a JSON contract, reject typos/invalid domains, and apply defaults."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream, object_pairs_hook=_json_object, parse_constant=_reject_constant)
    return validate_config(config)


def validate_config(config: dict) -> dict:
    """Validate an in-memory contract without mutating its caller's data."""
    config = copy.deepcopy(config)
    _object(config, "config")
    _keys(config, {"schema_version", "name", "description", "resource_id", "context", "search_space", "baseline", "constraints", "server", "benchmark_env", "scenarios", "requirements", "objective", "budget", "llm"}, "config")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        _error("schema_version", "must be 1")
    _string(config.get("name"), "name")
    _string(config.setdefault("resource_id", "ascend-default"), "resource_id")
    if "description" in config:
        _string(config["description"], "description", empty=True)
    _object(config.setdefault("context", {}), "context")

    space = _object(config.get("search_space"), "search_space")
    if not space:
        _error("search_space", "must contain at least one parameter")
    for name, choices in space.items():
        if not _NAME.fullmatch(name):
            _error("search_space", f"invalid parameter name {name!r}")
        if not isinstance(choices, list) or not choices:
            _error(f"search_space.{name}", "must be a non-empty list of choices")
        for index, choice in enumerate(choices):
            _scalar(choice, f"search_space.{name}[{index}]")
            if _member(choice, choices[:index]):
                _error(f"search_space.{name}", "contains duplicate choices")

    constraints = config.setdefault("constraints", [])
    if not isinstance(constraints, list):
        _error("constraints", "must be a list")
    for index, constraint in enumerate(constraints):
        loc = f"constraints[{index}]"
        _object(constraint, loc)
        _keys(constraint, {"when", "require"}, loc)
        when = _object(constraint.get("when"), f"{loc}.when")
        required = _object(constraint.get("require"), f"{loc}.require")
        if not when or not required:
            _error(loc, "when and require must be non-empty")
        for name, value in when.items():
            if name not in space or not _member(value, space[name]):
                _error(f"{loc}.when", f"{name!r} is not an allowed parameter/value")
        for name, values in required.items():
            if name not in space or not isinstance(values, list) or not values:
                _error(f"{loc}.require", "must map known parameters to non-empty choice lists")
            if any(not _member(value, space[name]) for value in values):
                _error(f"{loc}.require.{name}", "contains values outside search_space")
    config["baseline"] = validate_candidate(config.get("baseline"), config)

    server = _object(config.get("server"), "server")
    _keys(server, {"command", "env", "health_url", "startup_timeout_s", "shutdown_timeout_s"}, "server")
    _argv(server.get("command"), "server.command")
    _url(server.get("health_url"), "server.health_url")
    for key, value in _object(server.setdefault("env", {}), "server.env").items():
        if not _ENV_NAME.fullmatch(key):
            _error("server.env", f"invalid variable name {key!r}")
        _string(value, f"server.env.{key}", empty=True)
    if "benchmark_env" in config:
        for key, value in _object(config["benchmark_env"], "benchmark_env").items():
            if not _ENV_NAME.fullmatch(key):
                _error("benchmark_env", f"invalid variable name {key!r}")
            _string(value, f"benchmark_env.{key}", empty=True)
            if "{param." in value:
                _error("benchmark_env", "must not reference tunable parameters")
    # A parameter that only changes a benchmark argument does not configure the
    # server. Reject it instead of wasting trials or changing the test workload.
    bound_parameters = set()
    references = re.compile(r"\{param\.([^{}]*)\}")
    templates = list(server["command"]) + list(server["env"].values())
    for template in templates:
        bound_parameters.update(references.findall(template))
        if "{param." in references.sub("", template):
            _error("server", "malformed parameter placeholder")
    unknown_parameters = bound_parameters - set(space)
    if unknown_parameters:
        _error("server", f"unknown parameter references: {', '.join(sorted(unknown_parameters))}")
    unbound_parameters = set(space) - bound_parameters
    if unbound_parameters:
        _error("search_space", f"parameters must be bound in server.command or server.env: {', '.join(sorted(unbound_parameters))}")
    _number(server.setdefault("startup_timeout_s", 600), "server.startup_timeout_s", minimum=0.01)
    _number(server.setdefault("shutdown_timeout_s", 20), "server.shutdown_timeout_s", minimum=0.01)

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        _error("scenarios", "must be a non-empty list")
    seen = set()
    for index, scenario in enumerate(scenarios):
        loc = f"scenarios[{index}]"
        _object(scenario, loc)
        _keys(scenario, {"name", "command", "result_file", "metrics_format", "timeout_s", "requirements", "quality_command", "quality_result_file", "warmup_command", "before_each_command", "telemetry_file"}, loc)
        name = _string(scenario.get("name"), f"{loc}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            _error(f"{loc}.name", "use letters, digits, underscores, hyphens or dots")
        if name in seen:
            _error("scenarios", f"duplicate name: {name}")
        seen.add(name)
        _argv(scenario.get("command"), f"{loc}.command")
        _string(scenario.get("result_file"), f"{loc}.result_file")
        if "telemetry_file" in scenario:
            _string(scenario["telemetry_file"], f"{loc}.telemetry_file")
            if scenario["telemetry_file"] in {scenario["result_file"], scenario.get("quality_result_file")}:
                _error(loc, "telemetry_file must differ from metric and quality result files")
        if scenario.setdefault("metrics_format", "canonical") not in {"canonical", "vllm"}:
            _error(f"{loc}.metrics_format", "must be canonical or vllm")
        _number(scenario.setdefault("timeout_s", 600), f"{loc}.timeout_s", minimum=0.01)
        _requirements(scenario.setdefault("requirements", {}), f"{loc}.requirements")
        for command in ("quality_command", "warmup_command", "before_each_command"):
            if command in scenario:
                _argv(scenario[command], f"{loc}.{command}")
        if ("quality_command" in scenario) != ("quality_result_file" in scenario):
            _error(loc, "quality_command and quality_result_file must be provided together")
        if "quality_result_file" in scenario:
            _string(scenario["quality_result_file"], f"{loc}.quality_result_file")

    _requirements(config.setdefault("requirements", {}), "requirements")
    objective = _object(config.get("objective"), "objective")
    _keys(objective, {"metric", "direction", "scenario_aggregation"}, "objective")
    _string(objective.get("metric"), "objective.metric")
    if objective.setdefault("direction", "maximize") not in {"maximize", "minimize"}:
        _error("objective.direction", "must be maximize or minimize")
    if objective.setdefault("scenario_aggregation", "min") not in {"min", "mean", "max"}:
        _error("objective.scenario_aggregation", "must be min, mean or max")

    budget = _object(config.setdefault("budget", {}), "budget")
    defaults = {"max_trials": 12, "max_wall_time_s": 7200, "repetitions": 3, "final_repetitions": 3, "finalists": 2}
    _keys(budget, set(defaults) | {"early_stop", "confirmation_reserve", "cost_safety_factor", "estimated_trial_s"}, "budget")
    for key, value in defaults.items():
        _number(budget.setdefault(key, value), f"budget.{key}", minimum=0.01 if key == "max_wall_time_s" else 1, integer=key != "max_wall_time_s")
    # Optional execution policies stay absent in legacy snapshots so historical
    # fingerprints remain reproducible. Callers use get() for their defaults.
    for key in ("early_stop", "confirmation_reserve"):
        if key in budget and type(budget[key]) is not bool:
            _error("budget." + key, "must be a boolean")
    if "cost_safety_factor" in budget:
        _number(budget["cost_safety_factor"], "budget.cost_safety_factor", minimum=1)
    if "estimated_trial_s" in budget:
        _number(budget["estimated_trial_s"], "budget.estimated_trial_s", minimum=.01)

    llm = _object(config.get("llm"), "llm")
    _keys(llm, {"base_url", "model", "api_key_env", "timeout_s", "max_tokens", "temperature", "json_mode", "max_retries"}, "llm")
    _url(llm.get("base_url"), "llm.base_url")
    _string(llm.get("model"), "llm.model")
    api_key_env = _string(llm.setdefault("api_key_env", "POC_LLM_API_KEY"), "llm.api_key_env")
    if not _ENV_NAME.fullmatch(api_key_env):
        _error("llm.api_key_env", "must name an environment variable, never contain an API key")
    if api_key_env in server["env"]:
        _error("server.env", "must not contain the planner API-key environment variable")
    if api_key_env in config.get("benchmark_env", {}):
        _error("benchmark_env", "must not contain the planner API-key environment variable")
    if any("{env:" + api_key_env + "}" in value for value in config.get("benchmark_env", {}).values()):
        _error("benchmark_env", "must not reference the planner API-key environment variable")
    _number(llm.setdefault("timeout_s", 60), "llm.timeout_s", minimum=0.01)
    _number(llm.setdefault("max_tokens", 1200), "llm.max_tokens", minimum=1, integer=True)
    if llm.setdefault("temperature", 0.2) is not None:
        _number(llm["temperature"], "llm.temperature", minimum=0, maximum=2)
    _number(llm.setdefault("max_retries", 2), "llm.max_retries", minimum=0, maximum=10, integer=True)
    if not isinstance(llm.setdefault("json_mode", True), bool):
        _error("llm.json_mode", "must be a boolean")
    # JSON's exponent notation can parse to infinity without parse_constant.
    # Traverse even free-form context so snapshots always remain valid JSON.
    def check_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            _error("config", "non-finite numbers are not allowed")
        if isinstance(value, dict):
            for child in value.values():
                check_finite(child)
        elif isinstance(value, list):
            for child in value:
                check_finite(child)
    check_finite(config)
    return config


def validate_candidate(candidate, config: dict) -> dict:
    """Reject missing/extra keys, out-of-domain values and incompatible choices."""
    _object(candidate, "candidate")
    space = config["search_space"]
    if set(candidate) != set(space):
        missing = sorted(set(space) - set(candidate))
        extra = sorted(set(candidate) - set(space))
        _error("candidate", f"parameter keys must match search_space; missing={missing}, extra={extra}")
    canonical = {}
    for name, value in candidate.items():
        _scalar(value, f"candidate.{name}")
        if not _member(value, space[name]):
            _error(f"candidate.{name}", "value is outside the allowed choices")
        # A JSON LLM may emit 32.0 for an integer choice. Render/hash the trusted
        # representative, so CLI integer parsers receive "32", not "32.0".
        canonical[name] = next(choice for choice in space[name] if _equal(value, choice))
    for index, constraint in enumerate(config.get("constraints", [])):
        if all(_equal(candidate[key], value) for key, value in constraint["when"].items()):
            for key, choices in constraint["require"].items():
                if not _member(candidate[key], choices):
                    _error("candidate", f"violates compatibility constraint {index}: {key}")
    return copy.deepcopy(canonical)


def config_hash(config: dict) -> str:
    """Stable identity, excluding private runtime metadata at the root."""
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
