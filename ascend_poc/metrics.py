"""Normalize benchmark output and evaluate customer acceptance requirements."""

from __future__ import annotations

import math
import statistics


_ALIASES = {
    "output_throughput": "output_tokens_per_second",
    "request_throughput": "request_per_second",
    "total_token_throughput": "total_tokens_per_second",
    "request_goodput": "goodput_requests_per_second",
}


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def normalize_metrics(raw: dict, metrics_format: str = "canonical") -> dict:
    """Return numeric metrics; metadata is ignored, non-finite numbers rejected.

    vLLM's ``completed`` counts successful requests. Error rate is derived only
    when both completed and failed are present; omitted failures are never
    silently interpreted as zero. A caller must explicitly measure error rate.
    """
    if not isinstance(raw, dict):
        raise ValueError("benchmark result must be a JSON object")
    if metrics_format not in {"canonical", "vllm"}:
        raise ValueError("unknown metrics format")
    result = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = _finite(value, key)
    if metrics_format == "vllm":
        for source, destination in _ALIASES.items():
            if source in raw:
                value = _finite(raw[source], source)
                if destination in result and result[destination] != value:
                    raise ValueError(f"conflicting metrics: {source} and {destination}")
                result[destination] = value
        # Supports both flattened benchmark JSON and API-style percentile pairs.
        for metric in ("ttft", "tpot", "itl", "e2el"):
            key = f"percentiles_{metric}_ms"
            if key not in raw:
                continue
            if not isinstance(raw[key], list):
                raise ValueError(f"{key} must contain percentile/value pairs")
            for pair in raw[key]:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(f"invalid percentile pair in {key}")
                percentile = _finite(pair[0], key)
                value = _finite(pair[1], key)
                if percentile < 0 or percentile > 100:
                    raise ValueError(f"invalid percentile in {key}")
                name = f"p{percentile:g}_{metric}_ms"
                if name in result and result[name] != value:
                    raise ValueError(f"conflicting values for {name}")
                result[name] = value
    for count in ("completed", "failed"):
        if count in raw:
            value = _finite(raw[count], count)
            if value < 0 or not value.is_integer():
                raise ValueError(f"{count} must be a non-negative integer")
            result[count] = value
    if "completed" in result and "failed" in result:
        total = result["completed"] + result["failed"]
        if total <= 0:
            raise ValueError("benchmark completed no requests; error rate is undefined")
        derived = result["failed"] / total
        if "error_rate" in result and not math.isclose(result["error_rate"], derived, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("error_rate disagrees with completed/failed request counts")
        result["error_rate"] = derived
    if "error_rate" in result and not 0 <= result["error_rate"] <= 1:
        raise ValueError("error_rate must lie between 0 and 1")
    return result


def evaluate_trial(config: dict, scenario_results: list[dict], expected_repetitions: int | None = None) -> dict:
    """Apply all hard gates to all runs and aggregate only the objective.

    Each element is ``{name: str, metrics: [numeric_metrics, ...]}``. Each
    requirement's threshold is checked per repetition, including P99 values;
    P99 values are never averaged to claim a pooled quantile. Missing scenarios,
    runs or objective metrics make the trial infeasible.
    """
    checks = []
    failures = []
    violation = 0.0
    objectives = {}
    expected = expected_repetitions if expected_repetitions is not None else config.get("budget", {}).get("repetitions", 1)
    by_name = {}
    known_names = {scenario["name"] for scenario in config["scenarios"]}
    for item in scenario_results:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            failures.append("malformed scenario result")
            violation += 1
            continue
        name = item["name"]
        if name not in known_names or name in by_name:
            failures.append(f"unknown or duplicate scenario result: {name}")
            violation += 1
            continue
        by_name[name] = item

    objective_metric = config["objective"]["metric"]
    for scenario in config["scenarios"]:
        name = scenario["name"]
        item = by_name.get(name, {})
        repetitions = item.get("metrics")
        if not isinstance(repetitions, list) or not repetitions:
            failures.append(f"{name}: no completed metric repetitions")
            violation += 1
            continue
        if len(repetitions) != expected:
            failures.append(f"{name}: expected {expected} repetitions, received {len(repetitions)}")
            violation += 1
        limits = {**config.get("requirements", {}), **scenario.get("requirements", {})}
        values = []
        for index, metrics in enumerate(repetitions, start=1):
            if not isinstance(metrics, dict):
                failures.append(f"{name} repetition {index}: invalid metrics")
                violation += 1
                continue
            for metric, requirement in limits.items():
                value = metrics.get(metric)
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                passed = valid
                deficit = 0.0
                if valid:
                    if "min" in requirement and value < requirement["min"]:
                        passed = False
                        deficit += min(1e12, (requirement["min"] - value) / max(abs(requirement["min"]), 1.0))
                    if "max" in requirement and value > requirement["max"]:
                        passed = False
                        deficit += min(1e12, (value - requirement["max"]) / max(abs(requirement["max"]), 1.0))
                else:
                    deficit = 1.0
                checks.append({"scenario": name, "repetition": index, "metric": metric, "value": value if valid else None, "pass": bool(passed), "requirement": dict(requirement)})
                if not passed:
                    failures.append(f"{name} repetition {index}: {metric} {'outside requirement' if valid else 'missing or non-finite'}")
                    violation += deficit
            objective_value = metrics.get(objective_metric)
            if isinstance(objective_value, (int, float)) and not isinstance(objective_value, bool) and math.isfinite(objective_value):
                values.append(float(objective_value))
            else:
                failures.append(f"{name} repetition {index}: objective {objective_metric} missing or non-finite")
                violation += 1
        if len(values) == len(repetitions):
            objectives[name] = statistics.median(values)

    objective = None
    if len(objectives) == len(config["scenarios"]):
        aggregator = config["objective"].get("scenario_aggregation", "min")
        aggregate = {"min": min, "mean": statistics.mean, "max": max}[aggregator]
        objective = float(aggregate(objectives.values()))
    return {
        "feasible": not failures and objective is not None,
        "objective": objective,
        "violation": float(violation),
        "checks": checks,
        "failures": failures,
        "scenario_objectives": objectives,
    }
