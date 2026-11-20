"""Independent baseline-candidate-baseline verification of a fixed PoC matrix.

Acceptance and improvement are separate: a stable candidate can satisfy the
customer's gates without demonstrating a speedup over the unchanged baseline.
These checks use raw measurements, never the planner or a cached verdict.
"""
from __future__ import annotations

import math
import statistics

from .metrics import evaluate_trial
from .statistics_ci import student_t_critical


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


def evaluate_aba_ci(contract: dict, blocks: list[dict], settings: dict) -> dict:
    """Verify fresh ABA blocks and bound the mean paired log-ratio per scenario.

    A block is three separately started full-matrix trials. In-process benchmark
    repetitions are summarized by the existing scenario median; they are NOT
    counted as independent observations for this interval. The execution layer
    must establish fresh, distinct trials and a predeclared comparison budget.

    The interval assumes independent, approximately normal block log-effects.
    It cannot certify that those assumptions hold on a particular accelerator.
    Drift gates remain mandatory within AND between blocks. All customer gates
    apply to every candidate repetition, exactly as in the original ABA mode.
    A Bonferroni allocation covers the declared maximum comparisons and scenario
    count, conditional on those sampling assumptions and fresh confirmation data.

    Raw metrics and their objective aggregation are unchanged. The NEW interval
    estimand is explicitly the geometric-mean block speedup, not the median of
    all raw requests or an interval for the customer's aggregate objective.
    """
    result = {'mode': 'aba_ci', 'verdict': 'INVALID', 'valid': False, 'accepted': False,
              'improved': False, 'reasons': [], 'baseline_drift': None,
              'within_block_drift': None, 'between_block_drift': None,
              'relative_improvement': None, 'relative_improvement_lower_bound': None,
              'minimum_observed_relative_improvement': None, 'required_improvement': None,
              'scenarios': {}, 'block_verifications': []}
    try:
        if not isinstance(settings, dict) or settings.get('mode') != 'aba_ci':
            raise ValueError('invalid block CI settings')
        count = settings.get('ci_blocks', 5)
        confidence = settings.get('ci_confidence', .95)
        comparisons = settings.get('ci_max_comparisons', 8)
        if (type(count) is not int or not 3 <= count <= 100
                or not _finite(confidence) or not .5 < confidence < 1
                or type(comparisons) is not int or not 1 <= comparisons <= 10000):
            raise ValueError('invalid block CI budget or confidence')
        if not isinstance(blocks, list) or len(blocks) != count:
            raise ValueError('independent block count differs from the declared protocol')
        names = [scenario['name'] for scenario in contract['scenarios']]
        if not names or len(set(names)) != len(names):
            raise ValueError('invalid scenario matrix')
        settings_aba = dict(settings, mode='aba')
        seen_ids = set()
        for block in blocks:
            if not isinstance(block, dict) or set(block) != {'before', 'candidate', 'after'}:
                raise ValueError('invalid ABA block')
            for record in block.values():
                # Real workflow records have unique IDs. Bare metric fixtures may
                # omit IDs; no pure statistical function can prove fresh startup.
                if isinstance(record, dict) and 'id' in record:
                    identifier = record['id']
                    if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                        raise ValueError('duplicate or malformed trial identity across ABA blocks')
                    seen_ids.add(identifier)
            check = evaluate_aba(contract, block['before'], block['candidate'], block['after'], settings_aba)
            result['block_verifications'].append(check)
        checks = result['block_verifications']
        for index, check in enumerate(checks, 1):
            if not check['valid']:
                result['reasons'].extend(f'block {index}: {reason}' for reason in check['reasons'])
        if result['reasons']:
            return result
        within_drift = max(check['baseline_drift'] for check in checks)
        reference_centers = {name: [check['scenarios'][name]['baseline'] for check in checks] for name in names}
        between = {}
        for name, values in reference_centers.items():
            low, high = min(values), max(values)
            between[name] = (high - low) / (high / 2.0 + low / 2.0)
        between_drift = max(between.values())
        maximum_drift = max(within_drift, between_drift)
        bar = max(settings.get('min_relative_improvement', .02),
                  settings.get('drift_multiplier', 2.0) * maximum_drift)
        if not _finite(bar):
            raise ValueError('block CI improvement bar is not finite')
        result.update(baseline_drift=maximum_drift, within_block_drift=within_drift,
                      between_block_drift=between_drift, required_improvement=bar)
        if between_drift > settings.get('max_baseline_drift', .05):
            result['reasons'].append('baseline centers drift across blocks beyond the configured stability limit')
            return result
        # A candidate failure cannot turn an unstable reference comparison into
        # a valid rejection. Both reference drift checks take precedence before
        # interpreting candidate hard-gate or completeness failures.
        rejected = [(index, check) for index, check in enumerate(checks, 1) if not check['accepted']]
        if rejected:
            result.update(verdict='REJECTED', valid=True,
                          reasons=[f'block {index}: {reason}' for index, check in rejected
                                   for reason in check['reasons']])
            return result
        alpha = (1.0 - confidence) / (comparisons * len(names))
        critical = student_t_critical(alpha, count - 1)
        result['ci'] = {'method': 'paired_block_log_ratio_student_t', 'confidence': confidence,
                        'alpha_per_comparison': alpha, 'max_comparisons': comparisons,
                        'scenario_count': len(names), 'blocks': count, 'degrees_of_freedom': count - 1,
                        'critical_value': critical, 'sampling_unit': 'fresh_aba_block',
                        'within_block_objective': 'per_scenario_median',
                        'point_estimate': 'geometric_mean_of_direction_normalized_block_ratios_minus_one',
                        'assumptions': ['independent fresh ABA blocks', 'approximately normal block log-ratios',
                                        'predeclared comparison budget with fresh confirmation data',
                                        'observed zero variance does not establish absence of unobserved noise']}
        for name in names:
            effects = [check['scenarios'][name]['relative_improvement'] for check in checks]
            logs = [math.log(check['scenarios'][name]['speedup']) for check in checks]
            center = statistics.fmean(logs)
            error = statistics.stdev(logs) / math.sqrt(count)
            lower = math.expm1(center - critical * error)
            point = math.expm1(center)
            if not all(_finite(value) for value in (point, lower, error)):
                raise ValueError('non-finite block CI estimate')
            result['scenarios'][name] = {
                'block_relative_improvements': effects,
                'geometric_mean_relative_improvement': point,
                'relative_improvement_lower_bound': lower,
                'standard_error_log_ratio': error, 'zero_observed_variance': error == 0,
                'between_block_baseline_drift': between[name]}
        point = min(row['geometric_mean_relative_improvement'] for row in result['scenarios'].values())
        lower = min(row['relative_improvement_lower_bound'] for row in result['scenarios'].values())
        observed = min(check['relative_improvement'] for check in checks)
        def exceeds(value):
            return value > bar and not math.isclose(value, bar, rel_tol=1e-12, abs_tol=1e-12)
        result.update(verdict='ACCEPTED', valid=True, accepted=True,
                      relative_improvement=point, relative_improvement_lower_bound=lower,
                      minimum_observed_relative_improvement=observed,
                      improved=exceeds(lower) and exceeds(observed))
        if not result['improved']:
            result['reasons'].append('customer gates passed; block CI lower bound and every observed block must exceed the comparison bar')
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, statistics.StatisticsError):
        result.update(verdict='INVALID', valid=False, accepted=False, improved=False,
                      reasons=['malformed block CI input, incomplete protocol, or invalid numeric comparison'])
    return result
