"""Evidence-bounded diagnostics for research questions, never causal proof.

Only customer metric identifiers, finite measurements and fixed explanatory
templates leave this layer. Raw logs, command lines and process errors are not
inspected or returned. TTFT alone cannot prove a prefill or network bottleneck.
"""
from __future__ import annotations

import math
import re


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TEMPLATES = {
    "prefill_or_queueing": {
        "summary": "TTFT exceeds a customer limit; prefill, queueing and transport overhead are competing hypotheses, not a proven cause.",
        "families": ["chunked_prefill", "prefix_caching", "batch_scheduling"],
        "question": "For vLLM-Ascend, which supported chunked-prefill, prefix-cache and scheduling strategies can improve TTFT, and what prerequisites, tradeoffs and version-specific flags apply? Cite source files and explain how to distinguish prefill from queueing bottlenecks.",
    },
    "decode_latency": {
        "summary": "A decode-token latency limit failed; graph execution, batch interference and decode work are hypotheses requiring comparative tests.",
        "families": ["graph_execution", "speculative_decoding", "batch_scheduling"],
        "question": "For vLLM-Ascend, which graph execution and speculative decoding strategies can reduce TPOT or ITL? Explain supported model/hardware/version combinations, draft-model requirements, acceptance-rate tradeoffs and flags, with source references.",
    },
    "end_to_end_latency": {
        "summary": "End-to-end latency failed; separate queueing, TTFT and decode contributions before attributing the cause.",
        "families": ["batch_scheduling", "graph_execution"],
        "question": "In vLLM-Ascend, how should end-to-end tail-latency regressions be diagnosed using TTFT, TPOT and queue measurements, and which scheduling or graph strategies are supported? Cite implementation and configuration references.",
    },
    "throughput": {
        "summary": "A throughput requirement failed; batching, compute, memory and communication limits require profiling to distinguish.",
        "families": ["batch_scheduling", "parallelism", "quantization"],
        "question": "For vLLM-Ascend, which batching, parallelism and quantization strategies can improve throughput while preserving latency and model quality? State hardware, model and version restrictions and cite the relevant configuration and implementation files.",
    },
    "memory_pressure": {
        "summary": "A measured memory-related requirement failed; memory pressure is a hypothesis and is not proof of an OOM event.",
        "families": ["kv_cache", "batch_scheduling", "quantization"],
        "question": "In vLLM-Ascend, which KV-cache, batch-size and quantization controls affect memory pressure, and what compatibility and accuracy restrictions apply? Cite source files and explain the measurements needed before attributing a failure to OOM.",
    },
    "quality_regression": {
        "summary": "A measured quality requirement failed; compare the same evaluation data and restore an accuracy-preserving profile before optimizing speed.",
        "families": ["quality_preserving_baseline", "quantization"],
        "question": "For vLLM-Ascend, how should accuracy regressions after quantization or speculative decoding be validated against the unchanged model/evaluation contract? Identify supported correctness checks, compatibility restrictions and source references.",
    },
    "reliability": {
        "summary": "A completion or error-rate requirement failed; failed requests must remain in acceptance accounting.",
        "families": ["batch_scheduling", "compatibility"],
        "question": "For vLLM-Ascend, how should request failures or incomplete benchmarks be diagnosed without changing the workload? Identify supported-stack checks, scheduling limits, relevant metrics and source references.",
    },
    "measurement_failure": {
        "summary": "Measurements are failed, interrupted or missing; no hardware bottleneck or performance gain can be concluded.",
        "families": ["compatibility", "measurement_validation"],
        "question": "What vLLM-Ascend supported-stack, startup and benchmarking checks should precede performance tuning when a run fails or required metrics are missing? Cite version-compatibility and benchmark documentation; do not infer OOM without evidence.",
    },
    "feasible_optimization": {
        "summary": "The measured run meets its gates; further strategy changes remain hypotheses and need the full unchanged acceptance matrix.",
        "families": ["batch_scheduling", "graph_execution", "parallelism"],
        "question": "For vLLM-Ascend, what measured bottleneck evidence should guide selection among graph execution, batching, parallelism, prefix caching and speculative decoding after an acceptance baseline passes? Provide compatibility constraints, tradeoffs and source references.",
    },
}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe_name(value, fallback="unknown") -> str:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else fallback


def _kind(metric: str) -> str:
    if "ttft" in metric:
        return "prefill_or_queueing"
    if "tpot" in metric or "itl" in metric:
        return "decode_latency"
    if "e2el" in metric or "e2e_latency" in metric:
        return "end_to_end_latency"
    if any(word in metric for word in ("accuracy", "quality", "rouge", "bleu", "perplexity", "exact_match")):
        return "quality_regression"
    if any(word in metric for word in ("throughput", "tokens_per_second", "request_per_second", "goodput")):
        return "throughput"
    if any(word in metric for word in ("memory", "kv_cache", "cache_usage")):
        return "memory_pressure"
    return "reliability"


def analyze_trial(record: dict, contract: dict) -> dict:
    """Return compact numeric gate evidence and <=3 targeted research questions.

    Boundaries are always taken from the frozen contract. Untrusted evaluation
    text and check-provided boundaries are ignored. Numeric checks are validated
    again instead of trusting a supplied ``pass`` flag.
    """
    status = record.get("status")
    status = status if status in {"completed", "infeasible_early", "failed", "interrupted", "running"} else "unknown"
    evaluation = record.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    objective = evaluation.get("objective")
    result = {
        "trial_id": _safe_name(record.get("id")), "status": status,
        "feasible": status == "completed" and evaluation.get("feasible") is True and _finite(objective),
        "objective": objective if _finite(objective) else None,
        "violations": [], "hypotheses": [], "strategy_families": [], "research_questions": [],
    }
    scenarios = {item["name"]: item for item in contract["scenarios"]}
    from .workload import public_summary
    result['workloads'] = []
    for scenario_result in record.get('scenario_results', [])[:32]:
        if not isinstance(scenario_result, dict):
            continue
        name = scenario_result.get('name')
        if name not in scenarios or not scenarios[name].get('telemetry_file'):
            continue
        summaries = scenario_result.get('workload_summaries', [])
        if not isinstance(summaries, list):
            continue
        for index, summary in enumerate(summaries[:4], 1):
            clean = public_summary(summary, max_windows=4)
            if clean:
                result['workloads'].append({'scenario': name, 'repetition': index, 'summary': clean})
        if len(result['workloads']) >= 8:
            result['workloads'] = result['workloads'][:8]
            break
    evidence = {}
    checks = evaluation.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    for check in checks[:1000]:
        if not isinstance(check, dict) or check.get("scenario") not in scenarios:
            continue
        scenario = scenarios[check["scenario"]]
        limits = {**contract.get("requirements", {}), **scenario.get("requirements", {})}
        metric = check.get("metric")
        if metric not in limits or _safe_name(metric) == "unknown":
            continue
        bounds = limits[metric]
        value = check.get("value")
        valid = _finite(value)
        failed = not valid or any(value < threshold if bound == "min" else value > threshold for bound, threshold in bounds.items())
        if not failed:
            continue
        repetition = check.get("repetition")
        result["violations"].append({
            "metric": metric, "scenario": _safe_name(scenario["name"]),
            "repetition": repetition if isinstance(repetition, int) and not isinstance(repetition, bool) and repetition > 0 else None,
            "value": value if valid else None, "bounds": dict(bounds),
            "reason": "outside_requirement" if valid else "missing_metric",
        })
        kind = _kind(metric) if valid else "measurement_failure"
        evidence.setdefault(kind, set()).add(metric)
    if status != "completed" or not _finite(objective):
        evidence.setdefault("measurement_failure", set())
    elif not evidence:
        evidence["feasible_optimization" if result["feasible"] else "measurement_failure"] = set()
    if result["violations"]:
        result["feasible"] = False
    # Quality/reliability problems precede speed optimization; every diagnosis is
    # retained, while only three targeted questions consume remote research time.
    priority = ["measurement_failure", "quality_regression", "reliability", "prefill_or_queueing", "decode_latency", "end_to_end_latency", "memory_pressure", "throughput", "feasible_optimization"]
    for kind in priority:
        if kind not in evidence:
            continue
        template = _TEMPLATES[kind]
        result["hypotheses"].append({"kind": kind, "confidence": "hypothesis", "evidence_metrics": sorted(evidence[kind]), "summary": template["summary"]})
        for family in template["families"]:
            if family not in result["strategy_families"]:
                result["strategy_families"].append(family)
        if len(result["research_questions"]) < 3:
            result["research_questions"].append(template["question"])
    return result
