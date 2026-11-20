"""Trusted strategy catalog and immutable customer acceptance contract.

The research agent can recommend a profile, but cannot add flags, change a
workload, or promote unverified compatibility. Only operator-authored, enabled
and verified profiles are executable.
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .config import (
    _json_object, _keys, _number, _object, _reject_constant, _string,
    load_config, validate_candidate, validate_config,
)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_REPLACEMENTS = ("server", "search_space", "baseline", "constraints")
_BUDGET_KEYS = {"max_trials", "max_wall_time_s", "repetitions", "final_repetitions", "finalists",
                "early_stop", "confirmation_reserve", "cost_safety_factor", "estimated_trial_s"}
_REPOSITORY = "vllm-project/vllm-ascend"


def _endpoint(value: str) -> str:
    _string(value, "research.endpoint")
    try:
        url = urlsplit(value)
        host = url.hostname
        url.port
        local = host == "localhost"
        if host:
            try:
                local = local or ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        valid = bool(host) and not any((url.username, url.password, url.query, url.fragment))
        valid = valid and (url.scheme == "https" or (url.scheme == "http" and local))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("research.endpoint must use HTTPS (or loopback HTTP), without credentials, query or fragment")
    return value.rstrip("/")


def _quality_ready(profile: dict, contract: dict) -> None:
    if not profile["requires_quality"]:
        return
    metric = profile["quality_metric"]
    for scenario in contract["scenarios"]:
        limits = {**contract.get("requirements", {}), **scenario.get("requirements", {})}
        if not scenario.get("quality_command") or not scenario.get("quality_result_file"):
            raise ValueError(f"profile {profile['id']}: requires a quality hook for every scenario")
        if metric not in limits:
            raise ValueError(f"profile {profile['id']}: quality metric {metric!r} needs an acceptance threshold for every scenario")


def load_workflow(path: str | Path) -> dict:
    """Load a workflow plus its referenced frozen contract; reject unknown fields."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        spec = json.load(stream, object_pairs_hook=_json_object, parse_constant=_reject_constant)
    _object(spec, "workflow")
    _keys(spec, {"schema_version", "name", "contract", "budget", "research", "profiles", "verification"}, "workflow")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("workflow.schema_version must be 1")
    _string(spec.get("name"), "workflow.name")
    contract_ref = _string(spec.get("contract"), "workflow.contract")
    contract_path = Path(contract_ref)
    if contract_path.is_absolute():
        raise ValueError("workflow.contract must be a path relative to the workflow file")
    contract = load_config(source.parent / contract_path)
    # Legacy tune contracts share server env with the benchmark. A workflow
    # freezes that baseline environment explicitly, including parameters read
    # directly by benchmark code through os.environ rather than argv templates.
    if "benchmark_env" not in contract:
        def freeze(match):
            value = contract["baseline"][match.group(1)]
            return ("true" if value else "false") if isinstance(value, bool) else str(value)
        contract["benchmark_env"] = {
            key: re.sub(r"\{param\.([A-Za-z][A-Za-z0-9_]*)\}", freeze, value)
            for key, value in contract["server"]["env"].items()
        }
        contract = validate_config(contract)
    # A fixed benchmark may not take tunable values as input. Keeping an argv
    # template identical would otherwise still permit changing the workload.
    for scenario in contract["scenarios"]:
        if "{param." in json.dumps(scenario):
            raise ValueError("workflow scenarios cannot reference tunable parameters; freeze the customer workload")

    budget = _object(spec.setdefault("budget", {}), "workflow.budget")
    _keys(budget, _BUDGET_KEYS, "workflow.budget")
    spec["budget"] = {**contract["budget"], **budget}
    for key, value in spec["budget"].items():
        if key in {"early_stop", "confirmation_reserve"}:
            if type(value) is not bool:
                raise ValueError(f"workflow.budget.{key} must be boolean")
        else:
            seconds = key in {"max_wall_time_s", "estimated_trial_s"}
            _number(value, f"workflow.budget.{key}", minimum=.01 if seconds else 1,
                    integer=not seconds and key != "cost_safety_factor")

    verification = _object(spec.setdefault("verification", {}), "workflow.verification")
    _keys(verification, {"mode", "max_baseline_drift", "min_relative_improvement", "drift_multiplier", "frozen_files",
                         "confirmation_profiles", "ci_blocks", "ci_confidence", "ci_max_comparisons"}, "workflow.verification")
    if verification.setdefault("mode", "single") not in {"single", "aba", "aba_ci"}:
        raise ValueError("workflow.verification.mode must be single, aba or aba_ci")
    if verification['mode'] == 'aba_ci':
        _number(verification.setdefault('ci_blocks', 5), 'verification.ci_blocks', minimum=3, maximum=100, integer=True)
        _number(verification.setdefault('ci_confidence', .95), 'verification.ci_confidence', minimum=.5, maximum=1)
        if not .5 < verification['ci_confidence'] < 1:
            raise ValueError('verification.ci_confidence must be between 0.5 and 1 exclusively')
        _number(verification.setdefault('ci_max_comparisons', 8), 'verification.ci_max_comparisons', minimum=1, maximum=10000, integer=True)
        if spec['budget']['finalists'] > verification['ci_max_comparisons']:
            raise ValueError('finalists exceeds the predeclared CI comparison budget')
    elif any(key.startswith('ci_') for key in verification):
        raise ValueError('ci settings require verification.mode=aba_ci')
    for key, default in (("max_baseline_drift", .05), ("min_relative_improvement", .02), ("drift_multiplier", 2.0)):
        _number(verification.setdefault(key, default), f"workflow.verification.{key}", minimum=0)
    if verification["max_baseline_drift"] > 1:
        raise ValueError("workflow.verification.max_baseline_drift must be at most 1")
    files = verification.setdefault("frozen_files", [])
    if not isinstance(files, list):
        raise ValueError("workflow.verification.frozen_files must be a list")
    seen_paths = set()
    for entry in files:
        _object(entry, "verification.frozen_files entry")
        _keys(entry, {"path", "sha256"}, "verification.frozen_files entry")
        path_value = _string(entry.get("path"), "verification.frozen_files.path")
        if path_value in seen_paths:
            raise ValueError("duplicate verification.frozen_files path")
        seen_paths.add(path_value)
        digest = _string(entry.get("sha256"), "verification.frozen_files.sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("verification.frozen_files.sha256 must be 64 lowercase hex characters")

    research = _object(spec.setdefault("research", {}), "workflow.research")
    _keys(research, {"transport", "endpoint", "repository", "timeout_s", "max_calls", "on_error"}, "workflow.research")
    if research.setdefault("transport", "mcp") not in {"mcp", "a2a"}:
        raise ValueError("workflow.research.transport must be mcp or a2a")
    default_endpoint = "https://mcp.deepwiki.com/mcp" if research["transport"] == "mcp" else "http://127.0.0.1:8766/a2a"
    research["endpoint"] = _endpoint(research.setdefault("endpoint", default_endpoint))
    if research.setdefault("repository", _REPOSITORY) != _REPOSITORY:
        raise ValueError(f"workflow.research.repository must be {_REPOSITORY}")
    _number(research.setdefault("timeout_s", 60), "workflow.research.timeout_s", minimum=.01)
    _number(research.setdefault("max_calls", 12), "workflow.research.max_calls", minimum=1, integer=True)
    if research.setdefault("on_error", "stop") not in {"stop", "continue"}:
        raise ValueError("workflow.research.on_error must be stop or continue")

    profiles = spec.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("workflow.profiles must be a non-empty list")
    configs = {}
    for index, profile in enumerate(profiles):
        loc = f"workflow.profiles[{index}]"
        _object(profile, loc)
        _keys(profile, {"id", "description", "enabled", *_REPLACEMENTS, "requires_quality", "quality_metric", "compatibility"}, loc)
        name = _string(profile.get("id"), f"{loc}.id")
        if not _ID.fullmatch(name):
            raise ValueError(f"{loc}.id: use 1–64 letters, digits, underscores or hyphens")
        if name in configs:
            raise ValueError(f"duplicate profile id: {name}")
        _string(profile.get("description"), f"{loc}.description")
        for field, default in (("enabled", True), ("requires_quality", False)):
            if type(profile.setdefault(field, default)) is not bool:
                raise ValueError(f"{loc}.{field} must be a boolean")
        quality_metric = _string(profile.setdefault("quality_metric", "accuracy"), f"{loc}.quality_metric")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", quality_metric):
            raise ValueError(f"{loc}.quality_metric must be a metric identifier")
        compatibility = _object(profile.get("compatibility"), f"{loc}.compatibility")
        _keys(compatibility, {"status", "evidence", "reason"}, f"{loc}.compatibility")
        if compatibility.get("status") not in {"verified", "unverified"}:
            raise ValueError(f"{loc}.compatibility.status must be verified or unverified")
        _string(compatibility.setdefault("reason", ""), f"{loc}.compatibility.reason", empty=compatibility["status"] == "verified")
        evidence = compatibility.setdefault("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"{loc}.compatibility.evidence must be a list")
        for item in evidence:
            _string(item, f"{loc}.compatibility.evidence item")
        if compatibility["status"] == "verified" and not evidence:
            raise ValueError(f"{loc}: verified profiles require operator-supplied compatibility evidence")
        candidate = copy.deepcopy(contract)
        candidate["budget"] = copy.deepcopy(spec["budget"])
        for field in _REPLACEMENTS:
            if field in profile:
                candidate[field] = copy.deepcopy(profile[field])
        candidate = validate_config(candidate)
        # Different profiles cannot quietly move the benchmark to another
        # endpoint; all use the same dedicated service address/contract.
        if candidate["server"]["health_url"] != contract["server"]["health_url"]:
            raise ValueError(f"profile {name}: health_url must match the frozen contract")
        if profile["enabled"] and compatibility["status"] == "verified":
            _quality_ready(profile, candidate)
        configs[name] = candidate
    if not profiles[0]["enabled"] or profiles[0]["compatibility"]["status"] != "verified":
        raise ValueError("the first profile is the baseline and must be enabled and verified")
    if 'confirmation_profiles' in verification:
        selected = verification['confirmation_profiles']
        allowed = {p['id'] for p in profiles if p['enabled'] and p['compatibility']['status'] == 'verified'}
        if (not isinstance(selected, list) or not selected or any(not isinstance(p, str) or p not in allowed for p in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('verification.confirmation_profiles must name unique eligible profiles')
    # Prevent misleading baseline labels that actually run a modified strategy.
    first = configs[profiles[0]["id"]]
    if any(first[key] != contract[key] for key in _REPLACEMENTS):
        raise ValueError("the first profile must preserve the frozen contract baseline strategy")
    spec["_contract"] = contract
    spec["_profiles"] = configs
    spec["_source"] = str(source)
    return spec


def eligible_profiles(spec: dict) -> list[dict]:
    """Return copies of operator-enabled, verified metadata in catalog order."""
    return [copy.deepcopy(profile) for profile in spec["profiles"]
            if profile["enabled"] and profile["compatibility"]["status"] == "verified"]


def materialize_profile(spec: dict, profile_id: str, parameters: dict | None = None) -> dict:
    """Resolve an allowed profile without ever modifying the acceptance matrix."""
    eligible = {profile["id"] for profile in spec["profiles"]
                if profile["enabled"] and profile["compatibility"]["status"] == "verified"}
    if profile_id not in eligible:
        raise ValueError("profile is unknown, disabled or has unverified compatibility")
    config = copy.deepcopy(spec["_profiles"][profile_id])
    if "_repo_dir" in spec["_contract"]:
        config["_repo_dir"] = spec["_contract"]["_repo_dir"]
    if parameters is not None:
        config["baseline"] = validate_candidate(parameters, config)
    return config


def workflow_hash(spec: dict) -> str:
    """Freeze the public catalog and complete contract, excluding runtime paths."""
    data = {key: value for key, value in spec.items() if not key.startswith("_")}
    data["resolved_contract"] = {key: value for key, value in spec["_contract"].items() if not key.startswith("_")}
    content = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
