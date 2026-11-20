"""Workload-aware P/D capacity recommendations, with no backend side effects.

This small shadow advisor borrows the phase-capacity and sustained-evidence
ideas of EvolvingInfra. Its operator-supplied coefficients must be calibrated
for the actual model, hardware and parallelism. It is not that project's
runtime planner, an online fitter, a scheduler simulator, or an SLO verifier.
"""
from __future__ import annotations

import math
import statistics


_POLICY = {
    'min_windows': 3,
    'standard_errors': 2.0,
    'min_relative_gain': .02,
    'max_observation_gap_s': 15.0,
    'cooldown_s': 120.0,
}
_CONFIG_KEYS = {'schema_version', 'mode', 'workers', 'current_prefill', 'calibration',
                'policy', 'seconds_since_last_switch'}
_CALIBRATION_KEYS = {
    'source', 'hardware', 'model', 'prefill_coefficients_ms', 'decode_coefficients_ms',
    'experts', 'experts_per_token', 'forward_tokens', 'max_running_requests', 'kv_capacity_tokens',
}
_WINDOW_KEYS = {
    'start_s', 'end_s', 'completed_requests', 'mean_uncached_tokens', 'mean_cached_tokens',
    'mean_output_tokens', 'mean_accepted_tokens_per_step', 'decode_population',
    'mean_new_tokens_per_prefill_iteration', 'restore_fraction', 'max_observation_gap_s',
    'mean_causal_attention_pairs',
}
_WINDOW_REQUIRED = _WINDOW_KEYS - {'mean_causal_attention_pairs'}


def _number(value, label, *, minimum=0, positive=False, integer=False):
    try:
        valid = (type(value) is int if integer else
                 isinstance(value, (int, float)) and not isinstance(value, bool))
        valid = valid and math.isfinite(value) and value >= minimum and (not positive or value > 0)
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'{label} must be a finite {"integer" if integer else "number"} in range')
    return value


def _object(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f'{label} has missing or unknown fields')


def _config(config):
    _object(config, _CONFIG_KEYS, _CONFIG_KEYS - {'policy', 'seconds_since_last_switch'}, 'adaptive config')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('adaptive schema_version must be 1')
    if config['mode'] != 'shadow':
        raise ValueError('adaptive mode must be shadow; live role changes are not supported')
    count = _number(config['workers'], 'workers', minimum=2, integer=True)
    if count > 1024:
        raise ValueError('workers exceeds the shadow advisor limit of 1024')
    current = _number(config['current_prefill'], 'current_prefill', minimum=1, integer=True)
    if current >= count:
        raise ValueError('current_prefill must leave at least one decode worker')
    calibration = config['calibration']
    _object(calibration, _CALIBRATION_KEYS, _CALIBRATION_KEYS, 'calibration')
    for key in ('source', 'hardware', 'model'):
        if not isinstance(calibration[key], str) or not calibration[key].strip() or len(calibration[key]) > 4096:
            raise ValueError(f'calibration.{key} must be a nonempty string of at most 4096 characters')
    for key, size in (('prefill_coefficients_ms', 4), ('decode_coefficients_ms', 3)):
        coefficients = calibration[key]
        if not isinstance(coefficients, list) or len(coefficients) != size:
            raise ValueError(f'calibration.{key} must contain {size} coefficients')
        for value in coefficients:
            _number(value, f'calibration.{key}')
        if not any(coefficients):
            raise ValueError(f'calibration.{key} cannot be all zero')
    for key in ('experts', 'experts_per_token', 'max_running_requests', 'kv_capacity_tokens'):
        _number(calibration[key], f'calibration.{key}', minimum=1, integer=True)
    if calibration['experts_per_token'] > calibration['experts']:
        raise ValueError('experts_per_token cannot exceed experts')
    _number(calibration['forward_tokens'], 'calibration.forward_tokens', minimum=1)
    configured = config.get('policy', {})
    _object(configured, set(_POLICY), set(), 'policy')
    policy = {**_POLICY, **configured}
    for key, value in policy.items():
        _number(value, f'policy.{key}', minimum=2 if key == 'min_windows' else 0,
                positive=key == 'max_observation_gap_s', integer=key == 'min_windows')
    if 'seconds_since_last_switch' in config:
        _number(config['seconds_since_last_switch'], 'seconds_since_last_switch')
    return calibration, policy


def _windows(windows, policy):
    if not isinstance(windows, list):
        raise ValueError('observations must be a list of completed windows')
    if len(windows) < policy['min_windows']:
        raise ValueError('insufficient completed observation windows')
    if len(windows) > 10000:
        raise ValueError('observation window limit of 10000 exceeded')
    previous_end = None
    reference_duration = None
    for index, window in enumerate(windows):
        label = f'window {index}'
        _object(window, _WINDOW_KEYS, _WINDOW_REQUIRED, label)
        for key, value in window.items():
            _number(value, f'{label}.{key}', minimum=1 if key == 'completed_requests' else 0,
                    positive=key in {'mean_output_tokens', 'mean_accepted_tokens_per_step',
                                     'mean_new_tokens_per_prefill_iteration'},
                    integer=key == 'completed_requests')
        if window['end_s'] <= window['start_s']:
            raise ValueError(f'{label}: window must have positive duration')
        duration = window['end_s'] - window['start_s']
        if reference_duration is None:
            reference_duration = duration
        elif not math.isclose(duration, reference_duration, rel_tol=1e-9, abs_tol=1e-6):
            # Each slice has one vote in the mean and standard error below.
            # Unequal durations would over-weight short transients.
            raise ValueError(f'{label}: observation windows must have equal durations')
        if window['restore_fraction'] > 1:
            raise ValueError(f'{label}: restore_fraction must be at most 1')
        if previous_end is not None:
            if window['start_s'] < previous_end:
                raise ValueError(f'{label}: observations must be ordered and nonoverlapping')
            if window['start_s'] - previous_end > policy['max_observation_gap_s']:
                raise ValueError(f'{label}: gap between observations exceeds policy')
        if window['max_observation_gap_s'] > policy['max_observation_gap_s']:
            raise ValueError(f'{label}: observation gap exceeds policy')
        previous_end = window['end_s']


def _positive(value, label):
    return float(_number(value, label, positive=True))


def _capacity(calibration, window, prefill, workers):
    """Homogeneous worker approximation; all times supplied in milliseconds."""
    decode = workers - prefill
    new, cached, output = (window[key] for key in
                           ('mean_uncached_tokens', 'mean_cached_tokens', 'mean_output_tokens'))
    attention = window.get('mean_causal_attention_pairs')
    if attention is None:
        # This is a full-causal proxy at the mean lengths, not E[attention]
        # and not a sparse/linear/vision model inferred from a model name.
        attention = new * cached + new * (new + 1) / 2
    c0, c1, c2, c3 = calibration['prefill_coefficients_ms']
    prefill_ms = (c0 * new / window['mean_new_tokens_per_prefill_iteration']
                  + c1 * new + c2 * attention + c3 * window['restore_fraction'] * cached)
    _positive(prefill_ms, 'predicted prefill service time')
    prefill_rate = _positive(prefill * (1000 / prefill_ms), 'predicted prefill capacity')

    # Context grows during decode. The half-output estimate is explicit and
    # must be replaced by a richer backend model when unsuitable for a PoC.
    context = _positive(cached + new + output / 2, 'estimated decode context')
    kv_batch_limit = math.floor(.95 * calibration['kv_capacity_tokens'] / context)
    occupancy = min(window['decode_population'] / decode,
                    calibration['max_running_requests'], kv_batch_limit)
    if occupancy <= 0:
        return 0.0
    duty = min(1.0, occupancy)
    batch = max(1.0, occupancy)
    fraction = calibration['experts_per_token'] / calibration['experts']
    phi = 1.0 if fraction == 1 else -math.expm1(
        calibration['forward_tokens'] * batch * math.log1p(-fraction))
    d0, d_batch, d_expert = calibration['decode_coefficients_ms']
    iteration_ms = _positive(d0 + d_batch * batch + d_expert * phi, 'predicted decode iteration time')
    decode_rate = (decode * duty * batch * window['mean_accepted_tokens_per_step'] * 1000
                   / (iteration_ms * output))
    _positive(decode_rate, 'predicted decode capacity')
    return min(prefill_rate, decode_rate)


def _above(value, threshold):
    return value > threshold and not math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def advise_pd(config: dict, windows: list[dict]) -> dict:
    """Recommend an unexecuted P/D split from already completed observations.

    Invalid configuration raises ValueError. Missing, inconsistent or
    numerically unusable observations produce ``action: hold``. A recommendation
    never changes a service, validates a backend capability, or passes an SLO.
    Windows must have equal durations and acceptable internal/inter-window gaps;
    this historical shadow calculation does not compare their age to a live clock.
    ``seconds_since_last_switch`` is operator telemetry, not a maintained clock;
    omitting it denotes that no previous switch is being asserted.
    """
    calibration, policy = _config(config)
    workers, current = config['workers'], config['current_prefill']
    result = {
        'schema_version': 1, 'mode': 'shadow', 'action': 'hold',
        'measurement_status': 'prediction_only',
        'current_split': {'prefill': current, 'decode': workers - current},
        'recommended_split': None, 'predicted_gain_rps': None,
        'predicted_relative_gain': None, 'standard_error_rps': None,
        'candidates': [], 'reasons': [],
        'requires_backend_capabilities': ['runtime_pd_role_switch', 'kv_preserving_transition', 'inflight_drain'],
        'calibration': {key: calibration[key] for key in ('source', 'hardware', 'model')},
        'calibration_verified': False,
        'assumptions': [
            'Homogeneous workers; supplied calibration provenance is not automatically verified',
            'Window means describe completed work, not future requests or guaranteed demand',
            'Decode KV bound uses cached + uncached + half of output tokens',
            'Capacity predictions do not establish latency, quality, or customer acceptance',
        ],
    }
    if config.get('seconds_since_last_switch', policy['cooldown_s']) < policy['cooldown_s']:
        result['reasons'] = ['cooldown_active']
        return result
    try:
        _windows(windows, policy)
        if (workers - 1) * len(windows) > 1000000:
            raise ValueError('shadow capacity evaluation budget exceeded')
        result['observation_window_count'] = len(windows)
        result['observation_start_s'] = windows[0]['start_s']
        result['observation_end_s'] = windows[-1]['end_s']
        result['attention_model'] = ('observed_window_mean' if all('mean_causal_attention_pairs' in w for w in windows)
                                     else 'full_causal_proxy_for_windows_without_observed_mean')
        baseline = [_capacity(calibration, w, current, workers) for w in windows]
        if any(value <= 0 for value in baseline):
            raise ValueError('zero predicted baseline capacity; population or KV capacity is insufficient')
        mean_baseline = _positive(statistics.fmean(baseline), 'mean baseline capacity')
        result['predicted_current_capacity_rps'] = mean_baseline
        for count in range(1, workers):
            if count == current:
                continue
            predicted = [_capacity(calibration, window, count, workers) for window in windows]
            gains = [candidate - reference for candidate, reference in zip(predicted, baseline)]
            mean_gain = statistics.fmean(gains)
            error = statistics.stdev(gains) / math.sqrt(len(gains))
            relative = mean_gain / mean_baseline
            threshold = policy['standard_errors'] * error
            # Reject all non-finite intermediate values before returning JSON.
            for value in (mean_gain, error, relative, threshold, *predicted):
                if isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError('non-finite capacity comparison')
            candidate = {
                'split': {'prefill': count, 'decode': workers - count},
                'changed_workers': abs(count - current),
                'predicted_capacity_rps': statistics.fmean(predicted),
                'predicted_gain_rps': mean_gain,
                'predicted_relative_gain': relative,
                'standard_error_rps': error,
                'required_gain_rps': threshold,
                'qualifies': _above(mean_gain, threshold) and _above(relative, policy['min_relative_gain']),
            }
            result['candidates'].append(candidate)
        qualifying = [item for item in result['candidates'] if item['qualifies']]
        if not qualifying:
            result['reasons'] = ['no_candidate_exceeds_sustained_gain_and_relative_gain_thresholds']
            return result
        best = max(qualifying, key=lambda item: (item['predicted_gain_rps'], -item['changed_workers'], -item['split']['prefill']))
        result.update(action='recommend', recommended_split=best['split'],
                      predicted_gain_rps=best['predicted_gain_rps'],
                      predicted_relative_gain=best['predicted_relative_gain'],
                      standard_error_rps=best['standard_error_rps'],
                      reasons=['predicted_sustained_capacity_gain; no backend action executed'])
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as error:
        # Discard partial candidate scores rather than publishing a tempting
        # result from an incomplete observation set.
        result.update(action='hold', recommended_split=None, predicted_gain_rps=None,
                      predicted_relative_gain=None, standard_error_rps=None, candidates=[],
                      reasons=['unusable_observations: ' + str(error)])
    return result
