"""Independent ABA comparisons for arbitrary finite open-mode score metrics.

Strictly positive scenario medians use the existing ratio protocol unchanged.
If any measured objective median is zero or negative, use direction-adjusted
absolute differences divided by ``max(abs(A_before), abs(A_after), 1.0)``.
The floor of one is one unit of the objective metric, not a statistical confidence
claim or a hidden customer gate. In this mode the configured improvement bar is
a normalized-difference threshold, never a relative percentage or a speedup.
"""
from __future__ import annotations

import math

from .verification import _finite, _measurement, evaluate_aba


def evaluate_open_aba(contract, before, candidate, after, settings, *, bootstrap=False):
    """Compare scores and require every bootstrap restart to pass customer gates.

    A normal comparison may use an inadequate old reference. A bootstrap has no
    old reference: all three records are fresh starts of the same new candidate,
    so none may use that exemption. Recompute from raw repetitions, not cached
    feasibility fields, and retain the ordinary drift/improvement checks.
    """
    result = _evaluate_open_aba(contract, before, candidate, after, settings)
    if not bootstrap:
        return result
    problems = []
    try:
        repetitions = contract['budget']['final_repetitions']
        for label, measured in (('bootstrap_before', before), ('bootstrap_candidate', candidate),
                                ('bootstrap_after', after)):
            checked, errors = _measurement(contract, measured, repetitions, label)
            if errors:
                problems.extend(errors)
            elif not checked['feasible']:
                problems.extend(f'{label}: customer gates failed: {reason}'
                                for reason in checked['failures'])
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        result.update(verdict='INVALID', valid=False)
        problems.append('bootstrap: malformed independent measurement')
    if problems:
        result.update(verdict='REJECTED' if result['valid'] else 'INVALID',
                      accepted=False, improved=False,
                      reasons=list(result['reasons']) + problems)
    return result


def _evaluate_open_aba(contract, before, candidate, after, settings):
    """Recompute a verdict from complete raw measurements, including zero scores.

    Output retains the ordinary ABA verdict/valid/accepted/improved fields.
    The signed/zero protocol adds ``method``, ``normalized_improvement`` and
    ``threshold_interpretation``. Per-scenario records additionally contain
    ``absolute_improvement`` and ``comparison_scale``; relative-improvement and
    speedup fields are null to prevent reporting an absolute gain as a percent.
    """
    legacy = evaluate_aba(contract, before, candidate, after, settings)
    result = {
        'mode': 'aba', 'method': 'scaled_absolute_difference',
        'verdict': 'INVALID', 'valid': False, 'accepted': False, 'improved': False,
        'reasons': [], 'baseline_drift': None, 'relative_improvement': None,
        'normalized_improvement': None, 'required_improvement': None, 'scenarios': {},
        'threshold_interpretation':
            'Direction-adjusted absolute gain / max(abs(reference_before), '
            'abs(reference_after), 1 objective unit). This is not a relative '
            'percentage, speedup ratio, or confidence interval.',
    }
    try:
        if not isinstance(settings, dict) or settings.get('mode', 'aba') != 'aba':
            return legacy
        thresholds = {key: settings.get(key, default) for key, default in (
            ('max_baseline_drift', .05), ('min_relative_improvement', .02),
            ('drift_multiplier', 2.0))}
        if any(not _finite(value) or value < 0 for value in thresholds.values()):
            return legacy
        repetitions = contract['budget']['final_repetitions']
        if type(repetitions) is not int or repetitions < 1:
            return legacy
        direction = contract['objective']['direction']
        if direction not in ('maximize', 'minimize'):
            return legacy
        names = [scene['name'] for scene in contract['scenarios']]
        if not names or len(names) != len(set(names)):
            return legacy
        checked_before, errors_before = _measurement(contract, before, repetitions, 'baseline_before')
        checked_after, errors_after = _measurement(contract, after, repetitions, 'baseline_after')
        if errors_before or errors_after:
            return legacy
        checked_candidate, candidate_errors = _measurement(contract, candidate, repetitions, 'candidate')
        # Completeness failures with otherwise-positive references retain the
        # existing stable-reference rejection behavior. No score is fabricated.
        all_reference_positive = all(checked['scenario_objectives'][name] > 0
                                     for checked in (checked_before, checked_after) for name in names)
        if all_reference_positive and (candidate_errors or all(
                checked_candidate['scenario_objectives'][name] > 0 for name in names)):
            return legacy

        for name in names:
            a = checked_before['scenario_objectives'][name]
            b = checked_after['scenario_objectives'][name]
            scale = max(abs(a), abs(b), 1.0)
            baseline = a / 2 + b / 2
            # Divide before subtracting to avoid overflowing two finite signed
            # reference values. This is algebraically abs(a-b)/scale.
            drift = abs(a / scale - b / scale)
            result['scenarios'][name] = {
                'before': a, 'after': b, 'baseline': baseline,
                'baseline_drift': drift, 'candidate': None,
                'relative_improvement': None, 'speedup': None,
                'comparison_scale': scale, 'absolute_improvement': None,
                'normalized_improvement': None,
            }
        maximum_drift = max(row['baseline_drift'] for row in result['scenarios'].values())
        bar = max(thresholds['min_relative_improvement'], thresholds['drift_multiplier'] * maximum_drift)
        if not _finite(bar):
            result['reasons'].append('normalized improvement threshold overflowed')
            return result
        result.update(baseline_drift=maximum_drift, required_improvement=bar)
        if maximum_drift > thresholds['max_baseline_drift']:
            result['reasons'].append('baseline drift exceeds the configured stability limit')
            return result
        if candidate_errors:
            result.update(verdict='REJECTED', valid=True, reasons=candidate_errors)
            return result
        sign = 1.0 if direction == 'maximize' else -1.0
        for name, row in result['scenarios'].items():
            score = checked_candidate['scenario_objectives'][name]
            absolute = sign * (score - row['baseline'])
            normalized = absolute / row['comparison_scale']
            if not _finite(absolute) or not _finite(normalized):
                result['reasons'].append(f'{name}: absolute comparison overflowed')
                continue
            row.update(candidate=score, absolute_improvement=absolute,
                       normalized_improvement=normalized)
        if result['reasons']:
            return result
        improvement = min(row['normalized_improvement'] for row in result['scenarios'].values())
        result.update(valid=True, normalized_improvement=improvement)
        if not checked_candidate['feasible']:
            result.update(verdict='REJECTED', reasons=list(checked_candidate['failures']))
            return result
        improved = improvement > bar and not math.isclose(improvement, bar, rel_tol=1e-12, abs_tol=1e-12)
        result.update(verdict='ACCEPTED', accepted=True, improved=improved)
        if not improved:
            result['reasons'].append(
                'customer gates passed; normalized absolute improvement did not exceed '
                'the conservative comparison bar (not a relative percentage)')
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        result.update(verdict='INVALID', valid=False, accepted=False, improved=False,
                      reasons=['malformed verification input or invalid absolute comparison'])
    return result
