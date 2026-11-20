"""Independent final verdicts never substitute speedup for customer gates."""
import copy
import json
import unittest

from ascend_poc.metrics import evaluate_trial
from ascend_poc.verification import evaluate_aba
from tests.test_tuner import contract, measured


class ABAVerificationTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract('aba-fixture')
        self.settings = {'mode': 'aba', 'max_baseline_drift': .05,
                         'min_relative_improvement': .02, 'drift_multiplier': 2.0}

    def record(self, values=(100, 100), **kwargs):
        rows = measured(self.contract, {'batch': 1}, 3, **kwargs)
        for row, value in zip(rows, values):
            for metrics in row['metrics']:
                metrics[self.contract['objective']['metric']] = value
        return {'status': 'completed', 'scenario_results': rows,
                'evaluation': evaluate_trial(self.contract, rows, 3)}

    def verify(self, before=None, candidate=None, after=None):
        result = evaluate_aba(self.contract, before if before is not None else self.record(),
                              candidate if candidate is not None else self.record((150, 140)),
                              after if after is not None else self.record(), self.settings)
        json.dumps(result, allow_nan=False)
        return result

    def test_maximize_uses_conservative_scenario_improvement(self):
        result = self.verify()
        self.assertEqual(result['verdict'], 'ACCEPTED')
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['relative_improvement'], .4)
        self.assertEqual(result['baseline_drift'], 0)
        self.assertAlmostEqual(result['scenarios']['chat']['speedup'], 1.5)

    def test_minimize_uses_inverse_ratio(self):
        self.contract['objective']['direction'] = 'minimize'
        result = self.verify(candidate=self.record((50, 80)))
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['relative_improvement'], .25)

    def test_passed_customer_gates_do_not_require_a_speedup(self):
        result = self.verify(candidate=self.record((100, 99)))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertLess(result['relative_improvement'], 0)

    def test_rounding_at_improvement_bar_cannot_claim_a_clear_speedup(self):
        result = self.verify(candidate=self.record((102, 102)))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])

    def test_reference_drift_uses_each_scenario_not_aggregate(self):
        # Both reference aggregates are 100; changing the slow scenario is unstable.
        result = self.verify(before=self.record((100, 200)), after=self.record((200, 100)))
        self.assertEqual(result['verdict'], 'INVALID')
        self.assertFalse(result['accepted'])
        self.assertGreater(result['baseline_drift'], .05)

    def test_improvement_must_exceed_noise_adjusted_bar(self):
        result = self.verify(before=self.record((98, 98)), after=self.record((102, 102)),
                             candidate=self.record((106, 106)))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertAlmostEqual(result['baseline_drift'], .04)
        self.assertAlmostEqual(result['required_improvement'], .08)

    def test_infeasible_baseline_can_support_valid_accepted_candidate(self):
        baseline = self.record(latency=3000, quality=.8)
        self.assertFalse(baseline['evaluation']['feasible'])
        result = self.verify(before=baseline, after=baseline)
        self.assertTrue(result['accepted'])

    def test_candidate_failing_one_repetition_is_rejected(self):
        candidate = self.record()
        candidate['scenario_results'][0]['metrics'][0]['p99_ttft_ms'] = 5000
        # A forged cached feasibility verdict cannot override the raw gate.
        candidate['evaluation']['feasible'] = True
        result = self.verify(candidate=candidate)
        self.assertEqual(result['verdict'], 'REJECTED')
        self.assertFalse(result['accepted'])
        self.assertTrue(result['valid'])

    def test_incomplete_and_failed_reference_runs_invalidate_comparison(self):
        variants = []
        missing_repeat = self.record()
        missing_repeat['scenario_results'][0]['metrics'].pop()
        variants.append(missing_repeat)
        missing_scenario = self.record()
        missing_scenario['scenario_results'].pop()
        variants.append(missing_scenario)
        duplicate = self.record()
        duplicate['scenario_results'].append(copy.deepcopy(duplicate['scenario_results'][0]))
        variants.append(duplicate)
        missing_metric = self.record()
        del missing_metric['scenario_results'][0]['metrics'][0]['quality_score']
        variants.append(missing_metric)
        variants.extend([{'status': 'failed'}, None, [], {'status': 'completed'}])
        for record in variants:
            with self.subTest(record=record):
                result = evaluate_aba(self.contract, record, self.record(), self.record(), self.settings)
                self.assertEqual(result['verdict'], 'INVALID')
                self.assertFalse(result['accepted'])

    def test_invalid_candidate_records_rejected_after_stable_controls(self):
        candidate = self.record()
        candidate['scenario_results'][0]['metrics'].pop()
        for record in (candidate, {'status': 'failed'}):
            result = self.verify(candidate=record)
            self.assertEqual(result['verdict'], 'REJECTED')
            self.assertFalse(result['accepted'])

    def test_nonpositive_and_nonfinite_objectives_never_produce_speedup(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            for location in ('before', 'candidate', 'after'):
                with self.subTest(value=value, location=location):
                    record = self.record()
                    record['scenario_results'][0]['metrics'][0]['output_tokens_per_second'] = value
                    # Change all repetitions so the median cannot hide a bad value.
                    for row in record['scenario_results']:
                        for metrics in row['metrics']:
                            metrics['output_tokens_per_second'] = value
                    result = self.verify(**{location: record})
                    self.assertFalse(result['accepted'])
                    self.assertFalse(result['improved'])

    def test_cached_objective_is_not_authoritative(self):
        candidate = self.record((100, 100))
        candidate['evaluation']['scenario_objectives'] = {'chat': 10000, 'long_context': 10000}
        result = self.verify(candidate=candidate)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])

    def test_malformed_settings_fail_closed(self):
        for value in (float('nan'), float('inf'), -1, True, '5%'):
            self.settings['max_baseline_drift'] = value
            result = self.verify()
            self.assertEqual(result['verdict'], 'INVALID')

    def test_candidate_ratio_overflow_does_not_leak_infinity(self):
        result = self.verify(before=self.record((1e-300, 1e-300)),
                             after=self.record((1e-300, 1e-300)),
                             candidate=self.record((1e300, 1e300)))
        self.assertEqual(result['verdict'], 'INVALID')


if __name__ == '__main__':
    unittest.main()
