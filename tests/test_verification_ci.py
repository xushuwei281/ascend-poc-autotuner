"""Independent block statistics must not turn correlated repeats into certainty."""
import copy
import json
import math
import unittest

from ascend_poc.metrics import evaluate_trial
from ascend_poc.statistics_ci import student_t_critical, student_t_survival
from ascend_poc.verification import evaluate_aba_ci
from tests.test_tuner import contract, measured


class StudentTTests(unittest.TestCase):
    def test_reference_quantiles_use_finite_sample_t_not_normal(self):
        # Standard Student t table values; alpha is the ONE-sided upper tail.
        for alpha, df, expected in ((.025, 1, 12.7062047364), (.025, 2, 4.30265272975),
                                    (.025, 4, 2.7764451052), (.05, 10, 1.81246112281),
                                    (.005, 30, 2.74999565357)):
            with self.subTest(alpha=alpha, df=df):
                self.assertAlmostEqual(student_t_critical(alpha, df), expected, places=8)

    def test_analytic_cauchy_and_df_two_tails(self):
        for alpha in (.2, .05, .001, 1e-7):
            actual = student_t_critical(alpha, 1)
            expected = 1 / math.tan(math.pi * alpha)
            self.assertLess(abs(actual / expected - 1), 2e-10)
        for value in (.01, .5, 2, 10, 100):
            expected = .5 * (1 - value / math.sqrt(value * value + 2))
            self.assertAlmostEqual(student_t_survival(value, 2), expected, places=12)

    def test_tail_roundtrip_and_stronger_confidence_increases_threshold(self):
        for df in (3, 7, 20, 99):
            values = [student_t_critical(alpha, df) for alpha in (.1, .01, .0001, 1e-8)]
            self.assertEqual(values, sorted(values))
            for alpha, value in zip((.1, .01, .0001, 1e-8), values):
                self.assertLess(abs(student_t_survival(value, df) / alpha - 1), 1e-9)

    def test_invalid_quantile_inputs_rejected(self):
        for alpha, df in ((0, 3), (.5, 3), (.1, 0), (True, 3), (.05, True),
                          (float('nan'), 3), (.05, 2.5)):
            with self.subTest(alpha=alpha, df=df), self.assertRaises(ValueError):
                student_t_critical(alpha, df)


class ABACIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract('block-ci-fixture')
        self.settings = {'mode': 'aba_ci', 'max_baseline_drift': .05,
                         'min_relative_improvement': .02, 'drift_multiplier': 2.0,
                         'ci_blocks': 5, 'ci_confidence': .95, 'ci_max_comparisons': 8}

    def record(self, identifier, values=(100, 100), **kwargs):
        repetitions = self.contract['budget']['final_repetitions']
        rows = measured(self.contract, {'batch': 1}, repetitions, **kwargs)
        for row, value in zip(rows, values):
            for metric in row['metrics']:
                metric[self.contract['objective']['metric']] = value
        return {'id': identifier, 'status': 'completed', 'scenario_results': rows,
                'evaluation': evaluate_trial(self.contract, rows, repetitions)}

    def blocks(self, candidate=(150, 140)):
        return [{'before': self.record(f'{i}-A'), 'candidate': self.record(f'{i}-C', candidate),
                 'after': self.record(f'{i}-B')} for i in range(self.settings['ci_blocks'])]

    def verify(self, blocks=None):
        result = evaluate_aba_ci(self.contract, self.blocks() if blocks is None else blocks, self.settings)
        json.dumps(result, allow_nan=False)
        return result

    def test_stable_maximize_has_explicit_point_lower_bound_and_sampling_unit(self):
        result = self.verify()
        self.assertEqual(result['mode'], 'aba_ci')
        self.assertEqual(result['verdict'], 'ACCEPTED')
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['relative_improvement'], .4)
        self.assertAlmostEqual(result['relative_improvement_lower_bound'], .4)
        self.assertEqual(result['ci']['sampling_unit'], 'fresh_aba_block')
        self.assertEqual(result['ci']['degrees_of_freedom'], 4)
        self.assertEqual(result['ci']['blocks'], 5)
        self.assertAlmostEqual(result['ci']['alpha_per_comparison'], .05 / 16)
        self.assertTrue(result['scenarios']['chat']['zero_observed_variance'])
        self.assertTrue(any('unobserved noise' in text for text in result['ci']['assumptions']))

    def test_minimize_normalizes_the_improvement_direction(self):
        self.contract['objective']['direction'] = 'minimize'
        result = self.verify(self.blocks((50, 80)))
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['relative_improvement'], .25)

    def test_zero_effect_never_improves_even_with_zero_observed_variance(self):
        result = self.verify(self.blocks((100, 100)))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertEqual(result['relative_improvement_lower_bound'], 0)

    def test_equality_at_practical_improvement_threshold_is_not_a_win(self):
        result = self.verify(self.blocks((102, 102)))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])

    def test_one_unimproved_block_cannot_be_hidden_by_average(self):
        blocks = self.blocks((200, 200))
        blocks[0]['candidate'] = self.record('0-C', (101, 101))
        result = self.verify(blocks)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['improved'])
        self.assertAlmostEqual(result['minimum_observed_relative_improvement'], .01)

    def test_variable_block_effects_reduce_bound_and_multiple_comparisons_reduce_it_further(self):
        blocks = self.blocks()
        for index, value in enumerate((120, 130, 140, 150, 160)):
            blocks[index]['candidate'] = self.record(f'{index}-C', (value, value))
        self.settings['ci_max_comparisons'] = 1
        once = self.verify(blocks)
        self.settings['ci_max_comparisons'] = 8
        many = self.verify(blocks)
        self.assertLess(once['relative_improvement_lower_bound'], once['relative_improvement'])
        self.assertLess(many['relative_improvement_lower_bound'], once['relative_improvement_lower_bound'])
        self.assertEqual(many['relative_improvement'], once['relative_improvement'])
        self.settings['ci_confidence'] = .99
        stronger = self.verify(blocks)
        self.assertLess(stronger['relative_improvement_lower_bound'], many['relative_improvement_lower_bound'])

    def test_every_block_can_clear_old_bar_while_the_interval_still_prevents_promotion(self):
        blocks = self.blocks()
        for index, value in enumerate((103, 104, 105, 106, 130)):
            blocks[index]['candidate'] = self.record(f'{index}-C', (value, value))
        result = self.verify(blocks)
        self.assertTrue(result['accepted'])
        self.assertTrue(all(item['improved'] for item in result['block_verifications']))
        self.assertGreater(result['minimum_observed_relative_improvement'], .02)
        self.assertLess(result['relative_improvement_lower_bound'], .02)
        self.assertFalse(result['improved'])

    def test_adding_same_process_repetitions_does_not_increase_independent_sample_count(self):
        first = self.verify()
        self.contract['budget']['final_repetitions'] = 9
        extra_repeats = self.verify()
        self.assertEqual(first['ci'], extra_repeats['ci'])
        self.assertEqual(first['relative_improvement_lower_bound'], extra_repeats['relative_improvement_lower_bound'])

    def test_within_block_drift_still_invalidates_entire_comparison(self):
        blocks = self.blocks()
        blocks[2]['after'] = self.record('2-B', (120, 100))
        result = self.verify(blocks)
        self.assertEqual(result['verdict'], 'INVALID')
        self.assertFalse(result['accepted'])

    def test_between_block_drift_is_not_hidden_by_individually_stable_blocks(self):
        blocks = self.blocks()
        blocks[3]['before'] = self.record('3-A', (110, 100))
        blocks[3]['after'] = self.record('3-B', (110, 100))
        result = self.verify(blocks)
        self.assertEqual(result['verdict'], 'INVALID')
        self.assertEqual(result['within_block_drift'], 0)
        self.assertGreater(result['between_block_drift'], .05)

    def test_between_block_drift_takes_precedence_over_candidate_failure(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                blocks = self.blocks()
                blocks[3]['before'] = self.record('3-A', (110, 100))
                blocks[3]['after'] = self.record('3-B', (110, 100))
                metrics = blocks[1]['candidate']['scenario_results'][0]['metrics']
                if incomplete:
                    metrics.pop()
                else:
                    metrics[0]['quality_score'] = .1
                result = self.verify(blocks)
                self.assertEqual(result['verdict'], 'INVALID')
                self.assertFalse(result['valid'])
                self.assertFalse(result['accepted'])
                self.assertGreater(result['between_block_drift'], .05)
                self.assertIsNone(result['relative_improvement_lower_bound'])

    def test_drift_within_limit_still_raises_improvement_bar(self):
        blocks = self.blocks((110, 110))
        blocks[0]['before'] = self.record('0-A', (98, 98))
        blocks[0]['after'] = self.record('0-B', (102, 102))
        result = self.verify(blocks)
        self.assertAlmostEqual(result['required_improvement'], .08)
        self.assertTrue(result['improved'])

    def test_each_candidate_repetition_still_must_satisfy_quality_and_latency(self):
        for field, value in (('quality_score', .1), ('p99_ttft_ms', 100000)):
            blocks = self.blocks()
            blocks[1]['candidate']['scenario_results'][0]['metrics'][0][field] = value
            blocks[1]['candidate']['evaluation']['feasible'] = True
            result = self.verify(blocks)
            self.assertEqual(result['verdict'], 'REJECTED')
            self.assertFalse(result['accepted'])

    def test_fewer_blocks_cannot_be_used_as_an_early_success(self):
        result = self.verify(self.blocks()[:-1])
        self.assertEqual(result['verdict'], 'INVALID')
        self.assertFalse(result['improved'])

    def test_reusing_a_trial_as_fresh_evidence_is_rejected(self):
        blocks = self.blocks()
        blocks[1]['before'] = copy.deepcopy(blocks[0]['after'])
        result = self.verify(blocks)
        self.assertEqual(result['verdict'], 'INVALID')

    def test_incomplete_records_and_nonpositive_objectives_do_not_establish_interval(self):
        for part in ('before', 'candidate', 'after'):
            blocks = self.blocks()
            blocks[2][part]['scenario_results'][0]['metrics'].pop()
            result = self.verify(blocks)
            self.assertFalse(result['accepted'])
            self.assertIsNone(result['relative_improvement_lower_bound'])
        for value in (0, -1, float('nan'), float('inf'), True):
            blocks = self.blocks()
            blocks[0]['candidate'] = self.record('0-C', (value, value))
            result = self.verify(blocks)
            self.assertFalse(result['accepted'])

    def test_input_records_are_unchanged_and_cached_evaluation_is_not_used(self):
        blocks = self.blocks((100, 100))
        for block in blocks:
            block['candidate']['evaluation']['scenario_objectives'] = {'chat': 10000, 'long_context': 10000}
        original = copy.deepcopy((self.contract, blocks, self.settings))
        result = self.verify(blocks)
        self.assertFalse(result['improved'])
        self.assertEqual((self.contract, blocks, self.settings), original)

    def test_bad_settings_never_produce_a_valid_interval(self):
        for key, value in (('ci_blocks', 2), ('ci_blocks', True), ('ci_blocks', 101),
                           ('ci_confidence', .5), ('ci_confidence', 1), ('ci_confidence', float('nan')),
                           ('ci_max_comparisons', 0), ('ci_max_comparisons', 10001),
                           ('max_baseline_drift', -1)):
            settings = dict(self.settings, **{key: value})
            result = evaluate_aba_ci(self.contract, self.blocks(), settings)
            self.assertEqual(result['verdict'], 'INVALID')


if __name__ == '__main__':
    unittest.main()
