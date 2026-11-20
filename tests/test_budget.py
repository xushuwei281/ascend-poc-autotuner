"""Cost-aware execution preserves full independent confirmation and old defaults."""
from contextlib import redirect_stdout
import io
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.budget import confirmation_cost, preflight, search_cutoff, trial_cost
from ascend_poc.runner import TrialResults
from ascend_poc.workflow import run_workflow
from tests import test_workflow as fixtures
from tests import test_tuner as tuner_fixtures
from tests.test_tuner import measured


def observed(profile='baseline', *, status='completed', startup=20, benchmark=30, quality=10, other=5, repeats=2):
    return {'profile_id': profile, 'status': status,
            'timings': {'startup_s': startup, 'benchmark_s': benchmark, 'quality_s': quality,
                        'other_s': other, 'repetitions': repeats,
                        'total_s': startup + benchmark + quality + other}}


class CostTests(TestCase):
    def setUp(self):
        self.budget = {'repetitions': 2, 'final_repetitions': 4, 'cost_safety_factor': 1.25}

    def test_repetition_scaling_keeps_startup_and_other_work_fixed(self):
        cost = trial_cost([observed()], 'baseline', 4, self.budget)
        self.assertEqual(cost['seconds'], 105)
        self.assertEqual(cost['source'], 'observed_components')

    def test_partial_early_failures_are_not_full_matrix_costs(self):
        records = [observed(status='infeasible_early', startup=1, benchmark=1), observed()]
        cost = trial_cost(records, 'baseline', 4, self.budget)
        self.assertEqual(cost['seconds'], 105)
        self.assertEqual(cost['observations'], 1)

    def test_profile_specific_and_cross_profile_fallback_are_distinct(self):
        records = [observed(), observed('slow', startup=100)]
        self.assertEqual(trial_cost(records, 'baseline', 4, self.budget)['seconds'], 105)
        fallback = trial_cost(records, 'unseen', 4, self.budget)
        self.assertEqual(fallback['seconds'], 185)
        self.assertEqual(fallback['source'], 'cross_profile_fallback')

    def test_total_only_estimates_never_shrink_fixed_cost(self):
        record = {'status': 'completed', 'started_epoch': 10, 'ended_epoch': 50,
                  'scenario_results': [{'metrics': [{}, {}, {}]}]}
        self.assertEqual(trial_cost([record], None, 1, self.budget)['seconds'], 40)
        self.assertEqual(trial_cost([record], None, 6, self.budget)['seconds'], 80)

    def test_operator_estimate_and_unknown_are_explicit(self):
        self.assertIsNone(trial_cost([], 'new', 3, self.budget)['seconds'])
        budget = dict(self.budget, estimated_trial_s=30)
        self.assertEqual(trial_cost([], 'new', 4, budget)['seconds'], 60)
        self.assertEqual(trial_cost([], 'new', 1, budget)['seconds'], 30)

    def test_ci_cost_includes_each_independent_restart_and_safety_factor(self):
        value = confirmation_cost([observed()], self.budget, {'mode': 'aba_ci', 'ci_blocks': 5}, 'baseline', ['new'])
        self.assertEqual(value['measurements'], 15)
        self.assertEqual(value['estimated_s'], 15 * 105 * 1.25)
        self.assertFalse(value['is_guarantee'])
        single = confirmation_cost([observed()], self.budget, {'mode': 'single'}, 'baseline', ['new'])
        self.assertEqual(single['measurements'], 1)

    def test_preflight_distinguishes_unknown_from_insufficient(self):
        unknown = confirmation_cost([], self.budget, {'mode': 'aba'}, 'baseline', ['new'])
        self.assertTrue(preflight(100, unknown)['can_start'])
        known = confirmation_cost([observed()], self.budget, {'mode': 'aba'}, 'baseline', ['new'])
        self.assertFalse(preflight(100, known)['can_start'])

    def test_dynamic_reserve_is_not_capped_at_the_old_quarter(self):
        args = ([observed()], dict(self.budget, confirmation_reserve=True), {'mode': 'aba'}, 'baseline', ['new'])
        cutoff, estimate = search_cutoff(2000, 1500, *args)
        self.assertEqual(cutoff, 2000 - 3 * 105 * 1.25)
        self.assertGreater(cutoff, 1500)
        cutoff, _ = search_cutoff(500, 375, *args)
        self.assertLess(cutoff, 375)
        old, estimate = search_cutoff(2000, 1500, [], self.budget, {'mode': 'aba'}, 'baseline', ['new'])
        self.assertEqual(old, 1500)
        self.assertIsNone(estimate)


class BudgetWorkflowTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.source['verification'] = {'mode': 'aba'}
        self.fixture.reload()

    def run_workflow(self, runner, clock=None):
        with patch('ascend_poc.workflow.run_trial', side_effect=runner) as mocked, \
             patch('ascend_poc.workflow.runtime_signatures', side_effect=lambda configs: {key: {'fixture': 1} for key in configs}), \
             redirect_stdout(io.StringIO()):
            arguments = {'planner': fixtures.PlannerDouble(), 'researcher': fixtures.ResearchDouble()}
            if clock:
                with patch('ascend_poc.workflow.time', clock):
                    state = run_workflow(self.fixture.spec, self.fixture.output, **arguments)
            else:
                state = run_workflow(self.fixture.spec, self.fixture.output, **arguments)
        return state, mocked

    def test_insufficient_budget_skips_whole_bracket_without_partial_reference(self):
        self.fixture.source['budget'] = {'confirmation_reserve': True}
        self.fixture.reload()
        clock = fixtures.Clock()
        clock.now = 1000

        def runner(config, params, folder, deadline, repeats):
            clock.now += 5 if params['batch'] == 1 else 25
            return measured(config, params, repeats)

        state, hardware = self.run_workflow(runner, clock)
        self.assertEqual(hardware.call_count, 2)
        self.assertTrue(all(t['phase'] == 'exploration' for t in state['trials']))
        self.assertEqual(state['verification_attempts'][0]['status'], 'skipped_insufficient_budget')
        self.assertFalse((self.fixture.output / 'best.json').exists())

    def test_candidate_only_confirmation_does_not_promote_losing_candidate(self):
        self.fixture.source['verification']['confirmation_profiles'] = ['eager']
        self.fixture.reload()

        def runner(config, params, folder, deadline, repeats):
            # The candidate is feasible, but slower than the baseline. It must
            # not be raised into first place merely by filtering the finalist.
            return measured(config, {'batch': 3 if params['batch'] == 1 else 2}, repeats)

        state, hardware = self.run_workflow(runner)
        self.assertEqual(hardware.call_count, 2)
        self.assertEqual(state['verification_attempts'][0]['status'], 'skipped_confirmation_profile')
        self.assertFalse((self.fixture.output / 'best.json').exists())

    def test_ci_restarts_each_block_and_only_final_candidate_gets_verdict(self):
        self.fixture.source['verification'].update(mode='aba_ci', ci_blocks=3, ci_max_comparisons=1)
        self.fixture.reload()
        state, hardware = self.run_workflow(lambda config, params, folder, deadline, repeats: measured(config, params, repeats))
        self.assertEqual(hardware.call_count, 11)
        intermediate = [r for r in state['trials'] if r['phase'] == 'confirmation_block']
        final = [r for r in state['trials'] if r['phase'] == 'confirmation']
        self.assertEqual(len(intermediate), 2)
        self.assertTrue(all('verification' not in row for row in intermediate))
        self.assertEqual(len(final), 1)
        result = final[0]['verification']
        self.assertTrue(result['accepted'])
        self.assertTrue(result['improved'])
        self.assertEqual(len(result['reference_blocks']), 3)
        self.assertEqual(result['reference_blocks'][-1]['candidate'], final[0]['id'])
        self.assertEqual(len({x for row in result['reference_blocks'] for x in row.values()}), 9)

    def test_incomplete_ci_before_final_block_never_exports_winner(self):
        self.fixture.source['verification'].update(mode='aba_ci', ci_blocks=3, ci_max_comparisons=1)
        self.fixture.reload()
        clock = fixtures.Clock()
        clock.now = 1000
        calls = 0

        def runner(config, params, folder, deadline, repeats):
            nonlocal calls
            calls += 1
            clock.now = 1061 if calls == 6 else clock.now + 1
            return measured(config, params, repeats)

        state, hardware = self.run_workflow(runner, clock)
        self.assertEqual(hardware.call_count, 6)
        self.assertFalse(any(row['phase'] == 'confirmation' for row in state['trials']))
        self.assertEqual(state['verification_attempts'][0]['status'], 'incomplete')
        self.assertFalse(state['verification_attempts'][0]['verification']['accepted'])
        self.assertFalse((self.fixture.output / 'best.json').exists())

    def test_early_stop_metadata_is_preserved_without_fake_completed_matrix(self):
        self.fixture.source['budget'] = {'early_stop': True}
        self.fixture.source['verification']['confirmation_profiles'] = ['eager']
        self.fixture.reload()
        phases = []

        def runner(config, params, folder, deadline, repeats, **options):
            phases.append(options.get('phase'))
            if params['batch'] == 1:
                return measured(config, params, repeats)
            rows = TrialResults(measured(config, params, 1, latency=3000))
            rows.status = 'infeasible_early'
            rows.aborted_after = {'scenario': 'chat', 'repetition': 1}
            rows.timings = {'startup_s': 1, 'benchmark_s': 1, 'quality_s': 0,
                            'other_s': 0, 'total_s': 2, 'repetitions': repeats}
            return rows

        state, _ = self.run_workflow(runner)
        candidate = state['trials'][1]
        self.assertEqual(phases, ['exploration', 'exploration'])
        self.assertEqual(candidate['status'], 'infeasible_early')
        self.assertFalse(candidate['evaluation']['feasible'])
        self.assertEqual(candidate['aborted_after']['repetition'], 1)
        self.assertEqual(candidate['timings']['total_s'], 2)


class BudgetTunerTests(TestCase):
    def test_tuner_skips_unaffordable_single_confirmation(self):
        fixture = tuner_fixtures.TunerTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.config['budget'].update(max_trials=2, confirmation_reserve=True)
        clock = fixtures.Clock()
        clock.now = 1000

        def runner(config, params, folder, deadline, repeats):
            clock.now += 5 if params['batch'] == 1 else 40
            return measured(config, params, repeats)

        with patch('ascend_poc.tuner.time', clock):
            state, hardware = fixture.run_tune(runner)
        self.assertEqual(hardware.call_count, 2)
        preflight_event = next(item for item in state['events'] if item['type'] == 'verification_budget_preflight')
        self.assertEqual(preflight_event['estimate']['measurements'], 1)
        self.assertTrue(any(item['type'] == 'skipped_insufficient_budget' for item in state['events']))
        self.assertFalse((fixture.output / 'best.json').exists())
