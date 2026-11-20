"""Shadow planning must distinguish observations from deployable proof."""
import copy
import json
import math
import unittest

from ascend_poc.adaptive import advise_pd


def config():
    return {
        'schema_version': 1, 'mode': 'shadow', 'workers': 4, 'current_prefill': 2,
        'calibration': {
            'source': 'SYNTHETIC UNIT TEST; not hardware calibration',
            'hardware': 'fake four identical workers', 'model': 'fake model',
            'prefill_coefficients_ms': [0, .1, 0, 0],
            'decode_coefficients_ms': [20, 2, 0],
            'experts': 8, 'experts_per_token': 2, 'forward_tokens': 1,
            'max_running_requests': 128, 'kv_capacity_tokens': 10000000,
        },
    }


def windows(new=100, cached=0, output=100, count=3):
    return [{
        'start_s': i * 60, 'end_s': (i + 1) * 60,
        'completed_requests': 100, 'mean_uncached_tokens': new,
        'mean_cached_tokens': cached, 'mean_output_tokens': output,
        'mean_accepted_tokens_per_step': 1, 'decode_population': 16,
        'mean_new_tokens_per_prefill_iteration': 2048, 'restore_fraction': 0,
        'max_observation_gap_s': 3,
    } for i in range(count)]


class AdaptiveTests(unittest.TestCase):
    def test_workload_shift_changes_split_in_correct_direction(self):
        decode = advise_pd(config(), windows(new=100))
        prefill = advise_pd(config(), windows(new=10000))
        self.assertEqual(decode['action'], 'recommend')
        self.assertEqual(decode['recommended_split'], {'prefill': 1, 'decode': 3})
        self.assertEqual(prefill['recommended_split'], {'prefill': 3, 'decode': 1})
        self.assertGreater(decode['predicted_gain_rps'], 0)
        self.assertEqual(decode['standard_error_rps'], 0)

    def test_cached_prefix_is_not_charged_as_new_prefill_work(self):
        uncached = advise_pd(config(), windows(new=10000, cached=0))
        reused = advise_pd(config(), windows(new=100, cached=9900))
        self.assertEqual(uncached['recommended_split']['prefill'], 3)
        self.assertEqual(reused['recommended_split']['prefill'], 1)

    def test_missing_cache_observation_holds_instead_of_assuming_no_reuse(self):
        rows = windows()
        del rows[0]['mean_cached_tokens']
        outcome = advise_pd(config(), rows)
        self.assertEqual(outcome['action'], 'hold')
        self.assertIsNone(outcome['recommended_split'])
        self.assertEqual(outcome['candidates'], [])

    def test_one_burst_does_not_satisfy_standard_error_gate(self):
        # Steady windows favor the existing 2P2D split; one dramatic input
        # burst must not become a sustained 3P1D recommendation.
        rows = windows(new=5000, count=5)
        rows[-1]['mean_uncached_tokens'] = 10000
        cfg = config()
        cfg['policy'] = {'min_relative_gain': 0}
        outcome = advise_pd(cfg, rows)
        self.assertEqual(outcome['action'], 'hold')
        self.assertFalse(any(item['qualifies'] for item in outcome['candidates']))

    def test_cooldown_and_short_observation_intervals_hold(self):
        cfg = config()
        cfg['seconds_since_last_switch'] = 119
        outcome = advise_pd(cfg, windows(new=10000))
        self.assertEqual(outcome['reasons'], ['cooldown_active'])
        cfg['seconds_since_last_switch'] = 120
        self.assertEqual(advise_pd(cfg, windows(new=10000))['action'], 'recommend')
        self.assertEqual(advise_pd(config(), windows(count=2))['action'], 'hold')

    def test_missing_or_gapped_observations_fail_closed(self):
        mutations = [
            lambda rows: rows[0].update(max_observation_gap_s=16),
            lambda rows: rows[1].update(start_s=80),
            lambda rows: rows[1].update(start_s=59),
            lambda rows: rows[0].update(mean_output_tokens=math.nan),
            lambda rows: rows[0].update(decode_population=math.inf),
            lambda rows: rows[0].update(restore_fraction=1.1),
            lambda rows: rows[0].update(completed_requests=0),
            lambda rows: rows[0].update(completed_requests=True),
            lambda rows: rows[0].update(mean_accepted_tokens_per_step=0),
            lambda rows: rows[0].update(end_s=0),
            lambda rows: rows[0].update(future_output_tokens=50),
        ]
        for mutate in mutations:
            rows = windows()
            mutate(rows)
            with self.subTest(rows=rows):
                result = advise_pd(config(), rows)
                self.assertEqual(result['action'], 'hold')
                self.assertEqual(result['candidates'], [])
                json.dumps(result, allow_nan=False)

    def test_unequal_duration_windows_cannot_overweight_a_brief_burst(self):
        rows = windows(new=10000, count=11)
        for index, row in enumerate(rows):
            row['start_s'] = index
            row['end_s'] = index + 1 if index < 10 else index + 600
        rows[-1]['mean_uncached_tokens'] = 100
        result = advise_pd(config(), rows)
        self.assertEqual(result['action'], 'hold')
        self.assertEqual(result['candidates'], [])
        self.assertIn('equal durations', result['reasons'][0])

    def test_window_duration_rounding_tolerance_does_not_reject_equivalent_slices(self):
        rows = windows(new=10000)
        rows[1]['start_s'] += 1e-8
        result = advise_pd(config(), rows)
        self.assertEqual(result['action'], 'recommend')

    def test_idle_or_unserviceable_decode_cannot_claim_a_capacity_gain(self):
        for mutation in ('idle', 'kv_limit'):
            cfg, rows = config(), windows()
            if mutation == 'idle':
                for row in rows:
                    row['decode_population'] = 0
            else:
                cfg['calibration']['kv_capacity_tokens'] = 1
            result = advise_pd(cfg, rows)
            self.assertEqual(result['action'], 'hold')
            self.assertIn('zero predicted baseline capacity', result['reasons'][0])

    def test_all_cached_no_restore_cost_is_unknown_not_infinite_capacity(self):
        result = advise_pd(config(), windows(new=0, cached=1000))
        self.assertEqual(result['action'], 'hold')
        self.assertIn('predicted prefill service time', result['reasons'][0])

    def test_observed_attention_work_replaces_proxy_and_does_not_mutate_inputs(self):
        cfg, rows = config(), windows(new=100, cached=100)
        cfg['calibration']['prefill_coefficients_ms'] = [0, 0, .1, 0]
        for row in rows:
            row['mean_causal_attention_pairs'] = 100
        before = copy.deepcopy((cfg, rows))
        observed = advise_pd(cfg, rows)
        proxy_rows = copy.deepcopy(rows)
        for row in proxy_rows:
            del row['mean_causal_attention_pairs']
        proxy = advise_pd(cfg, proxy_rows)
        self.assertEqual(observed['attention_model'], 'observed_window_mean')
        self.assertIn('proxy', proxy['attention_model'])
        self.assertGreater(observed['predicted_current_capacity_rps'], proxy['predicted_current_capacity_rps'])
        self.assertEqual(before, (cfg, rows))

    def test_relative_gain_floor_can_prevent_otherwise_stable_recommendation(self):
        cfg = config()
        cfg['policy'] = {'min_relative_gain': .5}
        result = advise_pd(cfg, windows())
        self.assertEqual(result['action'], 'hold')
        self.assertTrue(result['candidates'])

    def test_only_shadow_actions_and_constant_worker_budget_are_returned(self):
        result = advise_pd(config(), windows(new=10000))
        self.assertEqual(result['mode'], 'shadow')
        self.assertEqual(result['measurement_status'], 'prediction_only')
        self.assertFalse(result['calibration_verified'])
        self.assertIn(result['action'], {'hold', 'recommend'})
        for item in result['candidates']:
            self.assertEqual(sum(item['split'].values()), 4)
            self.assertGreater(min(item['split'].values()), 0)
        self.assertEqual(result['requires_backend_capabilities'],
                         ['runtime_pd_role_switch', 'kv_preserving_transition', 'inflight_drain'])
        json.dumps(result, allow_nan=False)

    def test_fractional_decode_population_uses_duty_cycle(self):
        rows = windows()
        for row in rows:
            row['decode_population'] = .5
        outcome = advise_pd(config(), rows)
        # The same half-active request population is split among workers;
        # adding decode workers cannot turn it into several active requests.
        self.assertEqual(outcome['action'], 'hold')
        self.assertAlmostEqual(outcome['predicted_current_capacity_rps'], .5 * 1000 / (22 * 100))

    def test_bad_configuration_is_an_error_not_an_implicit_prior(self):
        cases = []
        for key, value in [('mode', 'live'), ('workers', 1), ('workers', 1025),
                           ('current_prefill', 4), ('schema_version', True), ('unknown', 1)]:
            cfg = config()
            cfg[key] = value
            cases.append(cfg)
        for key, value in [('source', ''), ('hardware', ''), ('model', ''),
                           ('prefill_coefficients_ms', [0, 0, 0, 0]),
                           ('decode_coefficients_ms', [1, -1, 1]),
                           ('decode_coefficients_ms', [1, math.inf, 1]),
                           ('prefill_coefficients_ms', [1, 2]), ('experts_per_token', 9),
                           ('forward_tokens', .5), ('max_running_requests', True)]:
            cfg = config()
            cfg['calibration'][key] = value
            cases.append(cfg)
        for policy in ({'min_windows': 1}, {'max_observation_gap_s': 0}, {'standard_errors': -1},
                       {'cooldown_s': math.nan}, {'unknown': 2}):
            cfg = config()
            cfg['policy'] = policy
            cases.append(cfg)
        for cfg in cases:
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                advise_pd(cfg, windows())

    def test_overflowed_predictions_never_return_nan_or_infinity(self):
        cfg, rows = config(), windows(new=1e308)
        cfg['calibration']['prefill_coefficients_ms'] = [0, 1e308, 1, 0]
        result = advise_pd(cfg, rows)
        self.assertEqual(result['action'], 'hold')
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
