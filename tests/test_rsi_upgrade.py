"""Cross-component regressions for bounded measurement improvements."""
import copy
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.confirmation import revalidate_confirmation
from ascend_poc.rsi import AgentIntegrityError, run_agent, report_agent
from ascend_poc.rsi_experiment import read_experiment
from ascend_poc.rsi_research import research_question
from ascend_poc.runner import TrialResults
from tests import test_rsi as parent_tests
from tests import test_workflow as fixtures
from tests.test_tuner import measured


class UpgradeTests(TestCase):
    def setUp(self):
        self.case = parent_tests.RSIControllerTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def run_actions(self, actions, *, ignore_memory=False, resume=False, hook=None, runner=None):
        planner = parent_tests.ScriptedPlanner(actions, hook=hook)
        research = fixtures.ResearchDouble()
        with patch('ascend_poc.rsi._runtime', return_value={'fixture': 1}), \
             patch('ascend_poc.workflow.runtime_signatures', side_effect=lambda configs: {k: {'fixture': 1} for k in configs}), \
             patch('ascend_poc.workflow.run_trial', side_effect=runner or self.case.fixture.runner) as hardware, \
             redirect_stdout(io.StringIO()):
            state = run_agent(self.case.config, self.case.output, planner=planner, researcher=research,
                              ignore_memory=ignore_memory, resume=resume)
        return state, hardware, research, planner

    def test_ignore_memory_neither_reads_nor_writes_and_reports_milestones(self):
        memory = self.case.fixture.root / 'memory.json'
        memory.write_text('This deliberately invalid file must never be read')
        actions = [parent_tests.action('recall'), *self.case.successful_actions()]
        state, _, _, planner = self.run_actions(actions, ignore_memory=True)
        self.assertEqual(memory.read_text(), 'This deliberately invalid file must never be read')
        self.assertEqual(state['memory']['records'], [])
        self.assertEqual(state['memory_mode'], 'disabled')
        self.assertTrue(any(item.get('lesson') for item in state['experiments']))
        self.assertFalse(state['experiments'][0].get('remembered', False))
        milestones = state['milestones']
        self.assertLessEqual(milestones['first_observed_feasible']['elapsed_s'], milestones['first_independent_acceptance']['elapsed_s'])
        self.assertLessEqual(milestones['first_independent_acceptance']['elapsed_s'], milestones['first_promotion']['elapsed_s'])
        self.assertEqual(milestones['first_promotion']['experiments'], 1)
        self.assertIn('first_promotion', (self.case.output / 'agent_report.md').read_text())
        self.assertEqual(planner.observations[1]['state']['memory']['scope'], 'disabled_control')
        with self.assertRaises(AgentIntegrityError):
            self.run_actions([], ignore_memory=False, resume=True)

    def test_disabled_memory_does_not_disable_final_audit_of_nonchampion(self):
        def corrupt_after_first_result():
            file = self.case.output / 'experiments/experiment-0000/events.jsonl'
            if file.exists():
                file.write_text('corrupted')
        with self.assertRaises(AgentIntegrityError):
            self.run_actions([parent_tests.experiment(), parent_tests.finish()],
                             ignore_memory=True, hook=corrupt_after_first_result)
        self.assertFalse((self.case.output / 'champion.json').exists())

    def test_observation_milestone_survives_failed_confirmation(self):
        def runner(config, params, folder, deadline, repeats):
            latency = 2500 if folder.name.endswith('confirmation') else 500
            return measured(config, params, repeats, latency=latency)
        state, _, _, _ = self.run_actions([parent_tests.experiment(), parent_tests.finish()], runner=runner)
        self.assertIn('first_observed_feasible', state['milestones'])
        self.assertNotIn('first_independent_acceptance', state['milestones'])
        result = state['experiments'][0]['result']
        self.assertFalse(result['evaluation']['feasible'])
        self.assertIsNotNone(result['milestones']['first_observed_feasible_epoch'])

    def test_losing_candidate_stops_after_two_explorations(self):
        def runner(config, params, folder, deadline, repeats):
            return measured(config, params, repeats, latency=2500 if params['batch'] == 2 else 500)
        state, hardware, _, _ = self.run_actions([parent_tests.experiment(), parent_tests.finish()], runner=runner)
        self.assertEqual(hardware.call_count, 2)
        self.assertFalse(state['experiments'][0]['result']['verification']['accepted'])
        child = json.loads((self.case.output / 'experiments/experiment-0000/workflow_state.json').read_text())
        self.assertEqual(child['verification_attempts'][0]['status'], 'skipped_confirmation_profile')

    def test_partial_exploration_is_preserved_without_independent_acceptance(self):
        self.case.fixture.source['budget'] = {'early_stop': True}
        self.case.fixture.reload()
        self.case.reload()
        def runner(config, params, folder, deadline, repeats, **options):
            rows = measured(config, params, repeats, latency=2500 if params['batch'] == 2 else 500)
            if params['batch'] != 2:
                return rows
            self.assertEqual(options['phase'], 'exploration')
            rows[0]['metrics'] = rows[0]['metrics'][:1]
            result = TrialResults(repeats)
            result.extend(rows)
            result.status = 'infeasible_early'
            result.aborted_after = {'scenario': rows[0]['name'], 'repetition': 1, 'violated_metrics': ['p99_ttft_ms']}
            return result
        state, hardware, _, _ = self.run_actions([parent_tests.experiment(), parent_tests.finish()], runner=runner)
        result = state['experiments'][0]['result']
        self.assertEqual(hardware.call_count, 2)
        self.assertEqual(result['candidate_status'], 'infeasible_early')
        self.assertFalse(result['evaluation']['feasible'])
        self.assertEqual(result['analysis']['status'], 'infeasible_early')
        self.assertFalse(result['verification']['accepted'])
        child = json.loads((self.case.output / 'experiments/experiment-0000/workflow_state.json').read_text())
        self.assertEqual(len(child['trials'][1]['scenario_results'][0]['metrics']), 1)
        self.assertEqual(result, read_experiment(self.case.output / 'experiments/experiment-0000'))

    def test_research_receives_champion_and_recent_metrics_without_raw_logs(self):
        actions = [parent_tests.experiment(), parent_tests.action('promote', experiment_id='experiment-0000'),
                   parent_tests.action('research', topic='decode'), parent_tests.finish()]
        state, _, research, _ = self.run_actions(actions)
        query = research.questions[0][1]
        payload = json.loads(query.split('\n', 1)[1])
        self.assertEqual(payload['champion']['experiment_id'], 'experiment-0000')
        self.assertTrue(payload['measurements'][0]['is_current_champion'])
        self.assertIn('objective', payload['measurements'][0]['analysis'])
        state['experiments'][0]['result']['analysis']['stdout'] = 'PRIVATE_LOG'
        state['experiments'][0]['result']['analysis']['prompt'] = 'PRIVATE_PROMPT'
        state['experiments'][0]['lesson'] = 'PRIVATE_LESSON'
        query = research_question(self.case.config['_workflow'], state, 'decode')
        self.assertNotIn('PRIVATE', query)
        self.assertLess(len(query), 16000)

    def enable_ci(self):
        self.case.fixture.source['verification'] = {'mode': 'aba_ci', 'ci_blocks': 3,
            'ci_confidence': .95, 'ci_max_comparisons': 5}
        self.case.fixture.reload()
        self.case.reload()

    def test_ci_full_agent_promotes_and_replays_independent_blocks(self):
        self.enable_ci()
        state, hardware, _, _ = self.run_actions(self.case.successful_actions())
        self.assertEqual(hardware.call_count, 11)  # two exploration plus three fresh ABA blocks
        self.assertEqual(state['generation'], 1)
        result = state['experiments'][0]['result']
        verification = result['verification']
        self.assertEqual(verification['mode'], 'aba_ci')
        self.assertTrue(verification['improved'])
        self.assertGreater(verification['relative_improvement_lower_bound'], .02)
        ids = [item for block in verification['reference_blocks'] for item in block.values()]
        self.assertEqual(len(set(ids)), 9)
        self.assertEqual(result, read_experiment(self.case.output / 'experiments/experiment-0000'))
        report_agent(self.case.output)

    def test_ci_rejects_reordered_reused_or_incorrectly_labelled_records(self):
        self.enable_ci()
        self.run_actions(self.case.successful_actions())
        folder = self.case.output / 'experiments/experiment-0000'
        state = json.loads((folder / 'workflow_state.json').read_text())
        contract = json.loads((folder / 'contract.json').read_text())
        workflow = json.loads((folder / 'workflow.json').read_text())
        contract['budget'] = workflow['budget']
        def check(trials):
            candidate = next(t for t in reversed(trials) if t['phase'] == 'confirmation')
            return revalidate_confirmation(contract, trials, candidate, workflow['verification'], 'reference')
        trials = copy.deepcopy(state['trials'])
        trials[0], trials[-1] = trials[-1], trials[0]
        with self.assertRaises(ValueError):
            check(trials)
        trials = copy.deepcopy(state['trials'])
        first = next(t for t in trials if t['phase'] == 'confirmation_block')
        first['phase'] = 'confirmation'
        with self.assertRaises(ValueError):
            check(trials)
        trials = copy.deepcopy(state['trials'])
        candidate = next(t for t in trials if t['phase'] == 'confirmation')
        candidate['verification']['reference_blocks'][1] = candidate['verification']['reference_blocks'][0]
        with self.assertRaises(ValueError):
            check(trials)

    def test_ci_comparison_budget_is_frozen_before_any_execution(self):
        self.case.fixture.source['verification'] = {'mode': 'aba_ci', 'ci_max_comparisons': 2}
        self.case.fixture.reload()
        with self.assertRaisesRegex(ValueError, 'comparison budget'):
            self.case.reload()
