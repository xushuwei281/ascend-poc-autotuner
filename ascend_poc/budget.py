"""Conservative, observed measurement-cost estimates, never deadline guarantees.

Only complete trials are extrapolated. Startup and other fixed work are kept
separate from repetition work when the runner supplies timing telemetry. Old
records with only a total duration are never scaled down for fewer repetitions.
"""
from __future__ import annotations

import math
import statistics


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def trial_cost(trials, profile_id, repetitions, budget):
    """Estimate a full matrix in seconds, before the configured safety margin.

    The profile-specific median is preferred. For an unmeasured profile use the
    slowest observed profile as an explicitly labelled fallback, or an operator
    estimate. Different parameters within one profile can still change duration.
    """
    grouped = {}
    for trial in trials:
        if trial.get('status') != 'completed':
            continue
        timings = trial.get('timings') or {}
        source_repeats = timings.get('repetitions')
        if not _positive(source_repeats):
            rows = trial.get('scenario_results') or []
            counts = [len(row.get('metrics', [])) for row in rows if isinstance(row, dict)]
            source_repeats = counts[0] if counts and len(set(counts)) == 1 and counts[0] > 0 else None
        if not _positive(source_repeats):
            continue
        keys = ('startup_s', 'benchmark_s', 'quality_s', 'other_s')
        if all(_nonnegative(timings.get(key)) for key in keys):
            duration = (timings['startup_s'] + timings['other_s']
                        + (timings['benchmark_s'] + timings['quality_s']) * repetitions / source_repeats)
            source = 'observed_components'
        else:
            start, finish = trial.get('started_epoch'), trial.get('ended_epoch')
            total = timings.get('total_s')
            if not _positive(total) and _nonnegative(start) and _nonnegative(finish):
                total = finish - start
            if not _positive(total):
                continue
            duration = total * max(1.0, repetitions / source_repeats)
            source = 'observed_total_conservative'
        if _positive(duration):
            grouped.setdefault(trial.get('profile_id'), []).append((duration, source))
    if profile_id in grouped:
        values = grouped[profile_id]
        return {'seconds': statistics.median(value[0] for value in values),
                'source': values[-1][1], 'observations': len(values), 'profile_id': profile_id}
    estimate = budget.get('estimated_trial_s')
    if _positive(estimate):
        repeats = budget.get('repetitions', 1)
        return {'seconds': estimate * max(1.0, repetitions / repeats),
                'source': 'operator_estimate', 'observations': 0, 'profile_id': profile_id}
    if grouped:
        return {'seconds': max(statistics.median(item[0] for item in values) for values in grouped.values()),
                'source': 'cross_profile_fallback', 'observations': sum(map(len, grouped.values())),
                'profile_id': profile_id}
    return {'seconds': None, 'source': 'unknown', 'observations': 0, 'profile_id': profile_id}


def confirmation_cost(trials, budget, verification, baseline_profile, candidate_profiles):
    """Estimate all requested final comparisons including independent CI blocks."""
    mode = verification.get('mode', 'single')
    blocks = verification.get('ci_blocks', 5) if mode == 'aba_ci' else 1
    repeats = budget['final_repetitions']
    components = []
    for profile in candidate_profiles:
        for _ in range(blocks):
            if mode in {'aba', 'aba_ci'}:
                components.append(trial_cost(trials, baseline_profile, repeats, budget))
            components.append(trial_cost(trials, profile, repeats, budget))
            if mode in {'aba', 'aba_ci'}:
                components.append(trial_cost(trials, baseline_profile, repeats, budget))
    factor = budget.get('cost_safety_factor', 1.25)
    known = all(item['seconds'] is not None for item in components)
    return {'estimated_s': sum(item['seconds'] for item in components) * factor if known else None,
            'measurements': len(components), 'safety_factor': factor,
            'sources': sorted({item['source'] for item in components}),
            'is_guarantee': False}


def search_cutoff(end, legacy_cutoff, trials, budget, verification, baseline_profile, candidate_profiles):
    """Use measured confirmation reserve; retain the old cutoff when unknown."""
    if not budget.get('confirmation_reserve', False):
        return legacy_cutoff, None
    estimate = confirmation_cost(trials, budget, verification, baseline_profile, candidate_profiles)
    if estimate['estimated_s'] is None:
        return legacy_cutoff, estimate
    return end - estimate['estimated_s'], estimate


def preflight(remaining_s, estimate):
    """A known insufficient estimate skips; unknown cost remains explicitly unknown."""
    seconds = estimate['estimated_s']
    return {'remaining_s': max(0.0, remaining_s), **estimate,
            'can_start': seconds is None or remaining_s >= seconds}
