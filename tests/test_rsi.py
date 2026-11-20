"""Parent RSI integration: real child workflows, immutable judgments and lineage."""
import copy
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.rsi import AgentIntegrityError, run_agent, report_agent
from ascend_poc.rsi_config import load_agent
from tests import test_workflow as fixtures
from tests.test_tuner import measured


def action(tool, **arguments):
    return {'tool': tool, 'arguments': arguments}


def experiment(batch=2, profile='eager'):
    return action('experiment', profile_id=profile, parameters={'batch': batch},
                  hypothesis='A larger allowed batch can raise throughput.',
                  expected_effect='Measure throughput while retaining every latency and quality gate.')


def finish():
    return action('finish', reason='All planned hypotheses have been checked.')


class ScriptedPlanner:
    def __init__(self, actions, *, hook=None):
        self.actions = list(actions)
        self.hook = hook
        self.observations = []
        self.calls = 0
        self.last_usage = {}

    def decide(self, observation, remaining_s):
        self.calls += 1
        self.last_usage = {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}
        self.observations.append(copy.deepcopy(observation))
        if self.hook:
            self.hook()
        return copy.deepcopy(self.actions.pop(0) if self.actions else finish())


class RSIControllerTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.output = self.fixture.root / 'agent-run'
        self.agent_path = self.fixture.root / 'agent.json'
        self.source = {'schema_version': 1, 'name': 'RSI integration fixture',
                       'goal': 'Improve the fixed customer acceptance objective.',
                       'workflow': 'workflow.json', 'memory_file': 'memory.json',
                       'settings': {'max_steps': 30, 'max_experiments': 5, 'max_wall_time_s': 120,
                                    'experiment_timeout_s': 45, 'max_llm_calls': 50,
                                    'max_research_calls': 3, 'max_consecutive_tool_errors': 3}}
        self.reload()

    def reload(self):
        self.agent_path.write_text(json.dumps(self.source))
        self.config = load_agent(self.agent_path)

    def execute(self, actions=(), *, runner=None, resume=False, output=None, planner=None, researcher=None):
        planner = planner or ScriptedPlanner(actions)
        researcher = researcher or fixtures.ResearchDouble()
        with patch('ascend_poc.rsi._runtime', return_value={'fixture': 1}), \
             patch('ascend_poc.workflow.runtime_signature', return_value={'fixture': 1}), \
             patch('ascend_poc.workflow.run_trial', side_effect=runner or self.fixture.runner) as hardware, \
             redirect_stdout(io.StringIO()):
            state = run_agent(self.config, output or self.output, resume=resume,
                              planner=planner, researcher=researcher)
        return state, hardware, planner

    def read_state(self):
        return json.loads((self.output / 'agent_state.json').read_text())

    def successful_actions(self):
        return [experiment(), action('reflect', experiment_id='experiment-0000',
                                     lesson='Observed throughput gain in this fixed fixture; revalidate reuse.'),
                action('promote', experiment_id='experiment-0000'), finish()]

    def test_experiment_reflection_promotion_export_and_historical_report(self):
        state, hardware, _ = self.execute(self.successful_actions())
        self.assertEqual(hardware.call_count, 5)
        self.assertEqual(state['generation'], 1)
        self.assertEqual(state['champion']['parameters'], {'batch': 2})
        self.assertEqual(state['lineage'][0]['kind'], 'promotion')
        self.assertTrue(state['experiments'][0]['remembered'])
        self.assertIn('Observed throughput', state['experiments'][0]['lesson'])
        exported = json.loads((self.output / 'champion.json').read_text())
        self.assertEqual(exported['deployment'], 'export_only')
        self.assertTrue(exported['verification']['improved'])
        replay = json.loads((self.output / 'champion-contract.json').read_text())
        for key in ('scenarios', 'requirements', 'objective'):
            self.assertEqual(replay[key], self.config['_workflow']['_contract'][key])
        self.assertEqual(replay['baseline'], {'batch': 2})
        (self.output / 'agent_report.md').unlink()
        with patch('ascend_poc.rsi._runtime', side_effect=AssertionError('no current hardware')):
            result = report_agent(self.output)
        self.assertEqual(result, self.output / 'agent_report.md')
        self.assertIn('generation 1', result.read_text())

    def test_customer_violation_and_weak_candidate_cannot_promote(self):
        def runner(config, params, folder, deadline, repeats):
            return measured(config, params, repeats, latency=2500 if params['batch'] == 2 else 500)
        state, _, _ = self.execute([experiment(), action('promote', experiment_id='experiment-0000'), finish()], runner=runner)
        self.assertEqual(state['generation'], 0)
        self.assertFalse(state['experiments'][0]['result']['verification']['accepted'])
        self.assertFalse((self.output / 'champion.json').exists())
        second = self.fixture.root / 'weak-run'
        def unchanged(config, params, folder, deadline, repeats):
            return measured(config, {'batch': 1}, repeats)
        weak, _, _ = self.execute([experiment(), action('promote', experiment_id='experiment-0000'), finish()],
                                  runner=unchanged, output=second)
        self.assertEqual(weak['generation'], 0)
        self.assertFalse(weak['experiments'][0]['result']['verification']['improved'])
        self.assertFalse((second / 'champion.json').exists())

    def test_recursive_experiment_references_current_champion_and_preserves_contract(self):
        state, hardware, _ = self.execute([experiment(), action('promote', experiment_id='experiment-0000'),
                                          experiment(3, 'baseline'), action('promote', experiment_id='experiment-0001'), finish()])
        self.assertEqual(hardware.call_count, 10)
        self.assertEqual(state['generation'], 2)
        self.assertEqual(state['experiments'][1]['reference'], {'profile_id': 'eager', 'parameters': {'batch': 2}})
        self.assertEqual(state['experiments'][1]['source_generation'], 1)
        self.assertEqual(state['champion']['parameters'], {'batch': 3})
        first_input = json.loads((self.output / 'experiments/experiment-0000/experiment-input.json').read_text())
        second_input = json.loads((self.output / 'experiments/experiment-0001/experiment-input.json').read_text())
        self.assertEqual(first_input['source_workflow_hash'], second_input['source_workflow_hash'])
        self.assertNotEqual(first_input['reference_config_hash'], second_input['reference_config_hash'])

    def test_stale_candidate_cannot_promote_after_incumbent_changes(self):
        state, _, _ = self.execute([experiment(), experiment(3, 'baseline'),
                                    action('promote', experiment_id='experiment-0000'),
                                    action('promote', experiment_id='experiment-0001'), finish()])
        self.assertTrue(state['experiments'][1]['result']['verification']['improved'])
        self.assertEqual(state['generation'], 1)
        self.assertEqual(state['champion']['experiment_id'], 'experiment-0000')
        self.assertEqual(len(state['lineage']), 1)

    def test_rollback_to_original_baseline_revokes_measured_exports(self):
        state, _, _ = self.execute([experiment(), action('promote', experiment_id='experiment-0000'),
                                    action('rollback', reason='Return to the original reference.'), finish()])
        self.assertEqual(state['generation'], 2)  # Monotonic lineage revision, not stack depth.
        self.assertIsNone(state['champion']['experiment_id'])
        self.assertEqual(state['champion']['parameters'], {'batch': 1})
        self.assertEqual([row['kind'] for row in state['lineage']], ['promotion', 'rollback'])
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())
        report_agent(self.output)
        self.assertFalse((self.output / 'champion.json').exists())

    def test_parent_tampering_rejects_report_and_revokes_exports(self):
        self.execute(self.successful_actions())
        path = self.output / 'agent_state.json'
        state = json.loads(path.read_text())
        state['generation'] = 999
        path.write_text(json.dumps(state))
        with self.assertRaises(AgentIntegrityError):
            report_agent(self.output)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())

    def test_child_tampering_rejects_report_and_revokes_exports(self):
        self.execute(self.successful_actions())
        path = self.output / 'experiments/experiment-0000/workflow_state.json'
        state = json.loads(path.read_text())
        state['elapsed_s'] += 1
        path.write_text(json.dumps(state))
        with self.assertRaises(AgentIntegrityError):
            report_agent(self.output)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())

    def test_broken_parent_journal_on_resume_revokes_existing_champion(self):
        self.execute(self.successful_actions())
        path = self.output / 'events.jsonl'
        path.write_bytes(path.read_bytes() + b'{"interrupted-record":')
        with self.assertRaises(AgentIntegrityError):
            self.execute(resume=True)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())

    def test_non_champion_child_journal_tamper_rejects_report_and_revokes_exports(self):
        state, _, _ = self.execute([experiment(), experiment(3, 'baseline'),
                                   action('promote', experiment_id='experiment-0000'), finish()])
        self.assertEqual(state['champion']['experiment_id'], 'experiment-0000')
        path = self.output / 'experiments/experiment-0001/events.jsonl'
        path.write_bytes(path.read_bytes() + b'{"broken-record":')
        with self.assertRaises(AgentIntegrityError):
            report_agent(self.output)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())

    def test_step_budget_stops_even_when_planner_wants_more(self):
        self.source['settings']['max_steps'] = 2
        self.reload()
        state, _, planner = self.execute([action('inspect')] * 5)
        self.assertEqual(state['steps'], 2)
        self.assertEqual(state['stop_reason'], 'max_steps')
        self.assertEqual(len(planner.observations), 2)

    def test_llm_attempt_budget_stops_new_decisions(self):
        self.source['settings']['max_llm_calls'] = 2
        self.reload()
        state, _, planner = self.execute([action('inspect')] * 5)
        self.assertEqual(state['llm_calls'], 2)
        self.assertEqual(state['stop_reason'], 'max_llm_calls')
        self.assertEqual(len(planner.observations), 2)

    def test_experiment_and_research_limits_are_controller_owned(self):
        self.source['settings'].update(max_experiments=1, max_research_calls=1)
        self.reload()
        research = fixtures.ResearchDouble()
        state, hardware, _ = self.execute([action('research', topic='decode'), action('research', topic='cache'),
                                          experiment(), experiment(3, 'baseline'), finish()], researcher=research)
        self.assertEqual(len(research.questions), 1)
        self.assertEqual(state['research_calls'], 1)
        self.assertEqual(len(state['experiments']), 1)
        self.assertEqual(hardware.call_count, 5)

    def test_absolute_deadline_counts_time_spent_in_a_planner_request(self):
        clock = fixtures.Clock()
        planner = ScriptedPlanner([experiment()], hook=lambda: setattr(clock, 'now', clock.now + 121))
        with patch('ascend_poc.rsi.time.time', side_effect=clock.time), \
             patch('ascend_poc.rsi.time.monotonic', side_effect=clock.monotonic):
            state, hardware, _ = self.execute(planner=planner)
        self.assertEqual(state['stop_reason'], 'deadline')
        self.assertEqual(hardware.call_count, 0)
        self.assertEqual(state['deadline_epoch'] - state['started_epoch'], 120)

    def test_pending_interrupted_experiment_resumes_without_another_decision(self):
        def interrupt(config, params, folder, deadline, repeats):
            if folder.name.endswith('confirmation'):
                raise KeyboardInterrupt()
            return measured(config, params, repeats)
        with self.assertRaises(KeyboardInterrupt):
            self.execute([experiment()], runner=interrupt)
        old = self.read_state()
        self.assertEqual(old['pending']['experiment_id'], 'experiment-0000')
        self.assertEqual(old['steps'], 1)
        state, hardware, planner = self.execute([action('promote', experiment_id='experiment-0000'), finish()], resume=True)
        self.assertEqual(state['deadline_epoch'], old['deadline_epoch'])
        self.assertEqual(state['steps'], 3)
        self.assertEqual(hardware.call_count, 3)
        self.assertEqual(len(planner.observations), 2)
        self.assertEqual(state['generation'], 1)

    def test_elapsed_pause_consumes_absolute_budget_on_resume(self):
        clock = fixtures.Clock()
        def interrupt(config, params, folder, deadline, repeats):
            if folder.name.endswith('confirmation'):
                raise KeyboardInterrupt()
            return measured(config, params, repeats)
        with patch('ascend_poc.rsi.time.time', side_effect=clock.time), \
             patch('ascend_poc.rsi.time.monotonic', side_effect=clock.monotonic):
            with self.assertRaises(KeyboardInterrupt):
                self.execute([experiment()], runner=interrupt)
            old = self.read_state()
            clock.now = old['deadline_epoch'] + 1
            state, hardware, planner = self.execute([finish()], resume=True)
        self.assertEqual(state['deadline_epoch'], old['deadline_epoch'])
        self.assertEqual(state['stop_reason'], 'deadline')
        self.assertEqual(hardware.call_count, 0)
        self.assertEqual(planner.observations, [])
        self.assertFalse((self.output / 'champion.json').exists())

    def test_prior_memory_is_hint_only_and_cannot_supply_an_unmeasured_champion(self):
        self.execute(self.successful_actions())
        second_output = self.fixture.root / 'second-agent'
        state, hardware, planner = self.execute([action('recall'), action('inspect'), finish()], output=second_output)
        self.assertTrue(state['memory']['records'])
        self.assertTrue(state['memory']['priors'])
        self.assertEqual(hardware.call_count, 0)
        self.assertEqual(state['generation'], 0)
        self.assertIsNone(state['champion']['experiment_id'])
        self.assertEqual(state['experiments'], [])
        self.assertFalse((second_output / 'champion.json').exists())
        self.assertIsNone(planner.observations[1]['state']['champion']['experiment_id'])

    def test_failed_measurement_is_retained_without_crashing_report_or_memory(self):
        def failed(config, params, folder, deadline, repeats):
            if params['batch'] == 2:
                raise OSError('fixture benchmark failed')
            return measured(config, params, repeats)
        state, _, _ = self.execute([experiment(), finish()], runner=failed)
        self.assertEqual(state['generation'], 0)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertEqual(state['experiments'][0]['result']['candidate_status'], 'failed')
        report_agent(self.output)

    def test_config_rejects_unknown_fields_and_invalid_limits(self):
        examples = [dict(self.source, autonomous_shell=True),
                    dict(self.source, schema_version=True),
                    dict(self.source, workflow='/tmp/absolute.json')]
        for field, value in (('max_steps', 0), ('max_llm_calls', 1.5), ('max_experiments', 10001),
                             ('experiment_timeout_s', float('nan')), ('unknown_budget', 1)):
            example = copy.deepcopy(self.source)
            example['settings'][field] = value
            examples.append(example)
        for example in examples:
            with self.subTest(example=example):
                self.agent_path.write_text(json.dumps(example))
                with self.assertRaises(ValueError):
                    load_agent(self.agent_path)
