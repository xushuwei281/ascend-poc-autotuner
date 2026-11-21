"""Zero and signed objective values are measured values, not missing data."""
import json
import unittest

from ascend_poc.open_verification import evaluate_open_aba
from ascend_poc.verification import evaluate_aba


def contract(direction='maximize'):
    return {'objective': {'metric': 'score', 'direction': direction, 'scenario_aggregation': 'min'},
            'requirements': {'quality': {'min': .9}}, 'budget': {'final_repetitions': 2},
            'scenarios': [{'name': 'short', 'requirements': {}}, {'name': 'long', 'requirements': {}}]}


def record(scores=(0, 0), quality=1):
    return {'status': 'completed', 'scenario_results': [
        {'name': name, 'metrics': [{'score': value, 'quality': quality} for _ in range(2)]}
        for name, value in zip(('short', 'long'), scores)],
        'evaluation': {'feasible': True, 'objective': 999999}}


class OpenVerificationTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract()
        self.settings = {'mode': 'aba', 'max_baseline_drift': .05,
                         'min_relative_improvement': .02, 'drift_multiplier': 2.0}

    def verify(self, a=(0, 0), c=(1, 1), b=None, **kwargs):
        result = evaluate_open_aba(self.contract, record(a), record(c, **kwargs),
                                   record(a if b is None else b), self.settings)
        json.dumps(result, allow_nan=False)
        return result

    def test_single_positive_scene_preserves_ratio_and_legacy_protocol(self):
        self.contract['scenarios'] = self.contract['scenarios'][:1]
        a, c, b = record((100, 150)), record((125, 180)), record((100, 150))
        for item in (a, c, b):
            item['scenario_results'] = item['scenario_results'][:1]
        for direction in ('maximize', 'minimize'):
            self.contract['objective']['direction'] = direction
            result = evaluate_open_aba(self.contract, a, c, b, self.settings)
            legacy = evaluate_aba(self.contract, a, c, b, self.settings)
            self.assertEqual(result['relative_improvement'], legacy['relative_improvement'])
            self.assertEqual(result['accepted'], legacy['accepted'])
            self.assertEqual(result['improved'], legacy['improved'])
            self.assertEqual(result['baseline_drift'], legacy['baseline_drift'])
            self.assertEqual(result['scenarios'], legacy['scenarios'])
            self.assertEqual(result['objective_comparison']['before'], 100)

    def test_customer_aggregate_can_improve_despite_one_scene_regressing(self):
        cases = [
            ('mean', 'maximize', (100, 100), (200, 90), 100, 145, .45),
            ('min', 'maximize', (100, 1000), (120, 900), 100, 120, .2),
            ('max', 'maximize', (100, 200), (90, 300), 200, 300, .5),
            ('mean', 'minimize', (100, 100), (150, 40), 100, 95, 100 / 95 - 1),
            ('max', 'minimize', (100, 1000), (110, 800), 1000, 800, .25),
            ('min', 'minimize', (100, 1000), (80, 1100), 100, 80, .25),
        ]
        for aggregation, direction, a, c, old_score, new_score, gain in cases:
            with self.subTest(aggregation=aggregation, direction=direction):
                self.contract['objective'].update(scenario_aggregation=aggregation, direction=direction)
                result = self.verify(a=a, c=c)
                self.assertTrue(result['accepted'])
                self.assertTrue(result['improved'])
                self.assertAlmostEqual(result['relative_improvement'], gain)
                comparison = result['objective_comparison']
                self.assertEqual(comparison['before'], old_score)
                self.assertEqual(comparison['candidate'], new_score)
                self.assertEqual(comparison['scenario_aggregation'], aggregation)
                # Legacy modes retain their original all-scenes comparison.
                self.assertFalse(evaluate_aba(self.contract, record(a), record(c), record(a),
                                              self.settings)['improved'])

    def test_aggregate_gain_does_not_bypass_explicit_scenario_gate(self):
        self.contract['objective']['scenario_aggregation'] = 'mean'
        self.contract['scenarios'][1]['requirements']['score'] = {'min': 95}
        result = self.verify(a=(100, 100), c=(200, 90))
        self.assertFalse(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertEqual(result['verdict'], 'REJECTED')

    def test_aggregate_gain_does_not_hide_drifting_reference_scene(self):
        self.contract['objective']['scenario_aggregation'] = 'mean'
        result = self.verify(a=(100, 100), b=(80, 120), c=(300, 100))
        self.assertFalse(result['valid'])
        self.assertFalse(result['improved'])

    def test_signed_and_zero_aggregates_use_customer_objective(self):
        cases = [
            ('mean', 'maximize', (-100, -100), (-50, -110), -100, -80, .2),
            ('min', 'maximize', (-100, -10), (-80, -20), -100, -80, .2),
            ('max', 'maximize', (-100, -10), (-110, -8), -10, -8, .2),
            ('mean', 'minimize', (-100, -100), (-150, -90), -100, -120, .2),
            ('mean', 'maximize', (-10, 10), (-5, 9), 0, 2, 2),
        ]
        for aggregation, direction, a, c, old_score, new_score, gain in cases:
            with self.subTest(aggregation=aggregation, direction=direction, baseline=a):
                self.contract['objective'].update(scenario_aggregation=aggregation, direction=direction)
                result = self.verify(a=a, c=c)
                self.assertTrue(result['improved'])
                self.assertEqual(result['method'], 'scaled_absolute_difference')
                self.assertIsNone(result['relative_improvement'])
                self.assertAlmostEqual(result['normalized_improvement'], gain)
                comparison = result['objective_comparison']
                self.assertEqual(comparison['before'], old_score)
                self.assertEqual(comparison['candidate'], new_score)
                self.assertIsNone(comparison['speedup'])

    def test_signed_cancellation_cannot_hide_aggregate_reference_drift(self):
        self.contract['objective']['scenario_aggregation'] = 'mean'
        result = self.verify(a=(-100, 100), b=(-99, 101), c=(-50, 100))
        self.assertFalse(result['valid'])
        self.assertEqual(result['objective_comparison']['baseline_drift'], 1)

    def test_minimizing_error_rate_can_accept_a_zero_error_candidate(self):
        self.contract['objective']['direction'] = 'minimize'
        result = self.verify(a=(.1, .2), c=(0, 0))
        self.assertTrue(result['accepted'])
        self.assertTrue(result['improved'])
        self.assertEqual(result['method'], 'scaled_absolute_difference')
        self.assertIsNone(result['relative_improvement'])
        self.assertAlmostEqual(result['normalized_improvement'], .1)
        for row in result['scenarios'].values():
            self.assertIsNone(row['speedup'])
            self.assertIsNone(row['relative_improvement'])
            self.assertEqual(row['candidate'], 0)
            self.assertEqual(row['comparison_scale'], 1)

    def test_small_zero_error_gain_can_pass_gates_without_beating_noise_bar(self):
        self.contract['objective']['direction'] = 'minimize'
        result = self.verify(a=(.001, .001), c=(0, 0))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertAlmostEqual(result['normalized_improvement'], .001)
        self.settings['min_relative_improvement'] = .0001
        self.assertTrue(self.verify(a=(.001, .001), c=(0, 0))['improved'])

    def test_zero_baseline_to_positive_score_is_a_valid_absolute_gain(self):
        result = self.verify(c=(.2, .1))
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['normalized_improvement'], .1)
        self.assertAlmostEqual(result['scenarios']['short']['absolute_improvement'], .2)
        self.assertIn('1 objective unit', result['threshold_interpretation'])

    def test_signed_score_changes_honor_direction(self):
        result = self.verify(a=(-100, -50), c=(-90, -40))
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['normalized_improvement'], .1)
        self.assertEqual(result['scenarios']['short']['comparison_scale'], 100)
        self.contract['objective']['direction'] = 'minimize'
        result = self.verify(a=(-100, -50), c=(-90, -40))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertLess(result['normalized_improvement'], 0)

    def test_mixed_sign_scenarios_use_one_explicit_absolute_protocol(self):
        result = self.verify(a=(100, -10), c=(110, -9))
        self.assertEqual(result['method'], 'scaled_absolute_difference')
        self.assertAlmostEqual(result['normalized_improvement'], .1)
        self.assertEqual(result['scenarios']['short']['absolute_improvement'], 10)
        self.assertEqual(result['scenarios']['long']['absolute_improvement'], 1)
        self.assertIsNone(result['scenarios']['short']['speedup'])

    def test_drift_protection_precedes_candidate_quality_failure(self):
        result = self.verify(a=(-1, -1), b=(-.8, -.8), c=(0, 0), quality=.1)
        self.assertFalse(result['valid'])
        self.assertEqual(result['verdict'], 'INVALID')
        self.assertGreater(result['baseline_drift'], .05)

    def test_signed_reference_quality_failure_does_not_invalidate_real_candidate(self):
        a = record((-1, -1), quality=.1)
        result = evaluate_open_aba(self.contract, a, record((0, 0)), a, self.settings)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['improved'])

    def test_candidate_must_pass_every_repetition_not_cached_verdict(self):
        c = record((1, 1)); c['scenario_results'][1]['metrics'][0]['quality'] = .1
        result = evaluate_open_aba(self.contract, record(), c, record(), self.settings)
        self.assertTrue(result['valid'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['verdict'], 'REJECTED')

    def test_missing_reference_or_candidate_measurements_never_promote(self):
        incomplete = record(); incomplete['scenario_results'][0]['metrics'].pop()
        result = evaluate_open_aba(self.contract, incomplete, record((1, 1)), record(), self.settings)
        self.assertFalse(result['valid'])
        result = evaluate_open_aba(self.contract, record(), incomplete, record(), self.settings)
        self.assertTrue(result['valid'])
        self.assertEqual(result['verdict'], 'REJECTED')
        result = evaluate_open_aba(self.contract, record(), {'status': 'failed'}, record(), self.settings)
        self.assertTrue(result['valid'])
        self.assertFalse(result['accepted'])

    def test_zero_to_zero_does_not_claim_improvement(self):
        result = self.verify(c=(0, 0))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertEqual(result['normalized_improvement'], 0)

    def test_gain_exactly_at_bar_does_not_claim_improvement(self):
        result = self.verify(c=(.02, .02))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])

    def test_noise_adjusted_bar_still_applies_to_absolute_differences(self):
        result = self.verify(a=(-.01, -.01), b=(.01, .01), c=(.03, .03))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertAlmostEqual(result['required_improvement'], .04)

    def test_invalid_or_nonfinite_scores_fail_closed_without_infinite_outputs(self):
        for value in (float('nan'), float('inf'), True):
            for where in range(3):
                records = [record(), record((1, 1)), record()]
                records[where]['scenario_results'][0]['metrics'][0]['score'] = value
                with self.subTest(value=value, where=where):
                    result = evaluate_open_aba(self.contract, *records, self.settings)
                    self.assertFalse(result['accepted'])
                    json.dumps(result, allow_nan=False)
        for setting in (float('nan'), -1, True):
            settings = dict(self.settings, drift_multiplier=setting)
            result = evaluate_open_aba(self.contract, record(), record((1, 1)), record(), settings)
            self.assertFalse(result['valid'])

    def test_signed_arithmetic_overflow_does_not_leak_nonfinite_numbers(self):
        result = self.verify(a=(-1e308, -1e308), c=(1e308, 1e308))
        self.assertFalse(result['valid'])
        self.assertFalse(result['improved'])

    def test_bootstrap_requires_gates_on_each_independent_restart(self):
        for score in (100, 0, -1):
            for index in range(3):
                records = [record((score, score)) for _ in range(3)]
                # One failed repetition must win over the forged cached verdict.
                records[index]['scenario_results'][0]['metrics'][0]['quality'] = .1
                with self.subTest(score=score, index=index):
                    result = evaluate_open_aba(self.contract, *records, self.settings, bootstrap=True)
                    self.assertTrue(result['valid'])
                    self.assertFalse(result['accepted'])
                    self.assertFalse(result['improved'])
                    self.assertEqual(result['verdict'], 'REJECTED')
                    self.assertTrue(any('bootstrap_' in reason and 'customer gates failed' in reason
                                        for reason in result['reasons']))

    def test_bootstrap_passes_complete_feasible_restarts_without_claiming_gain(self):
        for score in (100, 0, -1):
            with self.subTest(score=score):
                measurements = [record((score, score)) for _ in range(3)]
                result = evaluate_open_aba(self.contract, *measurements, self.settings, bootstrap=True)
                self.assertTrue(result['valid'])
                self.assertTrue(result['accepted'])
                self.assertFalse(result['improved'])

    def test_bootstrap_incomplete_reference_is_not_accepted(self):
        result = evaluate_open_aba(self.contract, {'status': 'failed'}, record(), record(),
                                   self.settings, bootstrap=True)
        self.assertFalse(result['valid'])
        self.assertFalse(result['accepted'])
        self.assertTrue(any('bootstrap_before' in reason for reason in result['reasons']))

    def test_normal_positive_reference_gate_exemption_is_preserved(self):
        a = record((100, 100), quality=.1)
        result = evaluate_open_aba(self.contract, a, record((110, 110)), a, self.settings)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['improved'])


if __name__ == '__main__':
    unittest.main()
