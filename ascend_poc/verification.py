"""Independent baseline-candidate-baseline verification of a fixed PoC matrix.

Acceptance and improvement are separate: a stable candidate can satisfy the
customer's gates without demonstrating a speedup over the unchanged baseline.
These checks use raw measurements, never the planner or a cached verdict.
"""
from __future__ import annotations

import math

from .metrics import evaluate_trial


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _measurement(contract, record, repetitions, label):
    """Check completeness before evaluating gates; return an error if malformed."""
    if not isinstance(record, dict) or record.get('status') != 'completed':
        return None, [f'{label}: measurement did not complete']
    rows = record.get('scenario_results')
    if not isinstance(rows, list):
        return None, [f'{label}: scenario results are missing']
    names = {scenario['name'] for scenario in contract['scenarios']}
    seen = set()
    problems = []
    metric = contract['objective']['metric']
    scenarios = {scenario['name']: scenario for scenario in contract['scenarios']}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str):
            problems.append(f'{label}: malformed scenario result')
            continue
        name = row['name']
        if name not in names or name in seen:
            problems.append(f'{label}: unknown or duplicate scenario')
            continue
        seen.add(name)
        values = row.get('metrics')
        if not isinstance(values, list) or len(values) != repetitions:
            problems.append(f'{label}/{name}: incomplete repetition count')
            continue
        required = set(contract.get('requirements', {})) | set(scenarios[name].get('requirements', {})) | {metric}
        for index, metrics in enumerate(values, 1):
            if not isinstance(metrics, dict):
                problems.append(f'{label}/{name}/{index}: malformed metrics')
                continue
            if any(not _finite(metrics.get(key)) for key in required):
                problems.append(f'{label}/{name}/{index}: required metric missing or non-finite')
            if any(isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value)
                   for value in metrics.values()):
                problems.append(f'{label}/{name}/{index}: non-finite measurement')
    if seen != names:
        problems.append(f'{label}: missing scenario')
    if problems:
        return None, problems
    evaluation = evaluate_trial(contract, rows, expected_repetitions=repetitions)
    objectives = evaluation.get('scenario_objectives', {})
    if set(objectives) != names or any(not _finite(value) for value in objectives.values()):
        return None, [f'{label}: invalid scenario objective']
    return evaluation, []


def evaluate_aba(contract: dict, before: dict, candidate: dict, after: dict, settings: dict) -> dict:
    """Return a fail-closed, JSON-serializable final verification result.

    All three records must contain the fixed contract's complete scenario matrix
    with ``budget.final_repetitions`` measurements. Reference SLO violations are
    allowed (improving an inadequate baseline is a valid PoC goal), but missing
    metrics, failed runs, nonpositive objective values or excessive baseline
    drift invalidate the comparison. Candidate hard gates apply per repetition.

    Drift is ``abs(A-B) / mean(A,B)`` for positive scenario objective medians.
    Improvement is ``C/mean(A,B)-1`` for maximize or ``mean(A,B)/C-1`` for
    minimize. The minimum scenario improvement must exceed the larger of the
    configured floor and ``drift_multiplier * maximum scenario drift``.
    """
    result = {'mode': 'aba', 'verdict': 'INVALID', 'valid': False, 'accepted': False,
              'improved': False, 'reasons': [], 'baseline_drift': None,
              'relative_improvement': None, 'required_improvement': None,
              'scenarios': {}}
    try:
        if not isinstance(settings, dict) or settings.get('mode', 'aba') != 'aba':
            raise ValueError('invalid ABA settings')
        thresholds = {key: settings.get(key, default) for key, default in (
            ('max_baseline_drift', .05), ('min_relative_improvement', .02), ('drift_multiplier', 2.0))}
        if any(not _finite(value) or value < 0 for value in thresholds.values()):
            raise ValueError('ABA thresholds must be finite and non-negative')
        repetitions = contract['budget']['final_repetitions']
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError('final repetition count must be a positive integer')
        direction = contract['objective']['direction']
        if direction not in {'maximize', 'minimize'}:
            raise ValueError('unknown objective direction')
        names = [scenario['name'] for scenario in contract['scenarios']]
        if not names or len(names) != len(set(names)):
            raise ValueError('scenario matrix must be non-empty and unique')
        reference_a, errors_a = _measurement(contract, before, repetitions, 'baseline_before')
        reference_b, errors_b = _measurement(contract, after, repetitions, 'baseline_after')
        if errors_a or errors_b:
            result['reasons'] = errors_a + errors_b
            return result
        for name in names:
            a = reference_a['scenario_objectives'][name]
            b = reference_b['scenario_objectives'][name]
            if a <= 0 or b <= 0:
                result['reasons'].append(f'{name}: baseline objective must be positive for a relative comparison')
                continue
            baseline = a / 2 + b / 2  # Avoid overflow from (a+b)/2.
            drift = abs(a - b) / baseline
            result['scenarios'][name] = {'before': a, 'after': b, 'baseline': baseline,
                                         'baseline_drift': drift, 'candidate': None,
                                         'relative_improvement': None, 'speedup': None}
        if result['reasons']:
            return result
        maximum_drift = max(row['baseline_drift'] for row in result['scenarios'].values())
        bar = max(thresholds['min_relative_improvement'], thresholds['drift_multiplier'] * maximum_drift)
        if not _finite(bar):
            result['reasons'].append('improvement threshold overflowed')
            return result
        result.update(baseline_drift=maximum_drift, required_improvement=bar)
        if maximum_drift > thresholds['max_baseline_drift']:
            result['reasons'].append('baseline drift exceeds the configured stability limit')
            return result

        checked, errors = _measurement(contract, candidate, repetitions, 'candidate')
        if errors:
            result.update(verdict='REJECTED', valid=True, reasons=errors)
            return result
        for name in names:
            value = checked['scenario_objectives'][name]
            baseline = result['scenarios'][name]['baseline']
            if value <= 0:
                result['reasons'].append(f'{name}: candidate objective must be positive for a relative comparison')
                continue
            ratio = value / baseline if direction == 'maximize' else baseline / value
            if not _finite(ratio):
                result['reasons'].append(f'{name}: relative comparison overflowed')
                continue
            result['scenarios'][name].update(candidate=value, speedup=ratio, relative_improvement=ratio - 1)
        if result['reasons']:
            return result
        improvement = min(row['relative_improvement'] for row in result['scenarios'].values())
        result.update(valid=True, relative_improvement=improvement)
        if not checked['feasible']:
            result.update(verdict='REJECTED', reasons=list(checked['failures']))
            return result
        # A rounded ratio exactly at the bar must not become a claimed speedup
        # because 102/100 - 1 evaluates slightly above 0.02 in binary floats.
        clear_improvement = improvement > bar and not math.isclose(improvement, bar, rel_tol=1e-12, abs_tol=1e-12)
        result.update(verdict='ACCEPTED', accepted=True, improved=clear_improvement)
        if not result['improved']:
            result['reasons'].append('customer gates passed; improvement did not exceed the conservative comparison bar')
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        result.update(verdict='INVALID', valid=False, accepted=False, improved=False,
                      reasons=['malformed verification input or invalid numeric comparison'])
    return result
