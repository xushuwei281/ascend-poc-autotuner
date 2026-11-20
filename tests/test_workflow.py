"""Exercise research, strategy selection, immutable acceptance and confirmation."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_poc.llm import LLMError
from ascend_poc.strategy import load_workflow
from ascend_poc.workflow import run_workflow, _question
from tests.test_tuner import contract, measured


class ResearchDouble:
    def __init__(self, *, error=None, events=None, clock=None, delay=0):
        self.error = error
        self.events = events if events is not None else []
        self.clock, self.delay = clock, delay
        self.questions = []

    def ask(self, repository, question, remaining_s=None):
        self.events.append('research')
        self.questions.append((repository, question, remaining_s))
        if self.clock:
            self.clock.now += self.delay
        if self.error:
            raise self.error
        return {'answer': 'Compare graph modes; this is a hypothesis, benchmark it.',
                'sources': ['https://github.com/vllm-project/vllm-ascend/blob/main/README.md'],
                'repository': repository, 'transport': 'mcp', 'tool': 'ask_wiki_question'}


class PlannerDouble:
    def __init__(self, profile='eager', batch=2, *, error=None, events=None):
        self.profile, self.batch, self.error = profile, batch, error
        self.events = events if events is not None else []
        self.calls = 0
        self.last_usage = {}
        self.received = []

    def propose(self, history, analysis, evidence, remaining_s):
        self.events.append('planner')
        self.calls += 1
        self.last_usage = {'prompt_tokens': 10, 'completion_tokens': 5}
        self.received.append((copy.deepcopy(history), copy.deepcopy(analysis), copy.deepcopy(evidence)))
        if self.error:
            raise self.error
        return {'profile_id': self.profile, 'parameters': {'batch': self.batch},
                'hypothesis': 'Measure the operator-approved eager strategy.',
                'evidence_ids': [item['id'] for item in evidence]}


class Clock:
    now = 1000.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poc-workflow-tests-')
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        base = contract(str(self.root))
        base['budget'].update(max_trials=2, finalists=1)
        self.contract_path = self.root / 'contract.json'
        self.contract_path.write_text(json.dumps(base))
        self.workflow_path = self.root / 'workflow.json'
        eager = copy.deepcopy(base['server'])
        eager['command'] += ['--enforce-eager']
        self.source = {
            'schema_version': 1, 'name': 'Workflow integration test', 'contract': 'contract.json',
            'research': {'transport': 'mcp', 'max_calls': 10, 'on_error': 'stop'},
            'profiles': [
                {'id': 'baseline', 'description': 'Frozen baseline',
                 'compatibility': {'status': 'verified', 'evidence': ['test fixture']}},
                {'id': 'eager', 'description': 'Eager execution', 'server': eager,
                 'compatibility': {'status': 'verified', 'evidence': ['test fixture']}},
            ],
        }
        self.reload()

    def reload(self):
        self.workflow_path.write_text(json.dumps(self.source))
        self.spec = load_workflow(self.workflow_path)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def runner(config, candidate, trial_dir, deadline, repeats):
        return measured(config, candidate, repeats)

    def execute(self, *, runner=None, planner=None, researcher=None, resume=False, runtime=None):
        with patch('ascend_poc.workflow.run_trial', side_effect=runner or self.runner) as hardware, \
             patch('ascend_poc.workflow.runtime_signatures', side_effect=lambda configs: {key: runtime or {'fixture': 1} for key in configs}), \
             redirect_stdout(io.StringIO()):
            state = run_workflow(self.spec, self.output, resume=resume,
                                 planner=planner or PlannerDouble(),
                                 researcher=researcher or ResearchDouble())
        return state, hardware

    def read_state(self):
        return json.loads((self.output / 'workflow_state.json').read_text())

    def test_closed_loop_orders_measure_research_plan_and_exports_confirmed_strategy(self):
        events = []
        planner = PlannerDouble(events=events)
        research = ResearchDouble(events=events)
        snapshots = []

        def runner(config, candidate, folder, deadline, repeats):
            # No winner can be exported before any confirmation has completed.
            self.assertFalse((self.output / 'best.json').exists())
            events.append('confirmation' if folder.name.endswith('confirmation') else 'measure')
            snapshots.append(copy.deepcopy(config))
            return measured(config, candidate, repeats)

        state, hardware = self.execute(runner=runner, planner=planner, researcher=research)
        self.assertEqual(events, ['measure', 'research', 'planner', 'measure', 'confirmation'])
        self.assertEqual(hardware.call_count, 3)
        self.assertEqual(state['trials'][0]['profile_id'], 'baseline')
        self.assertEqual(state['trials'][1]['profile_id'], 'eager')
        self.assertEqual(state['research_calls'], 1)
        self.assertEqual(state['llm_calls'], 1)
        self.assertEqual(state['decisions'][0]['evidence_ids'], ['evidence-0000'])
        for candidate in snapshots:
            self.assertEqual(candidate['scenarios'], self.spec['_contract']['scenarios'])
            self.assertEqual(candidate['requirements'], self.spec['_contract']['requirements'])
        self.assertIn('--enforce-eager', snapshots[-1]['server']['command'])
        winner = json.loads((self.output / 'best.json').read_text())
        replay = json.loads((self.output / 'best-contract.json').read_text())
        self.assertEqual(winner['profile_id'], 'eager')
        self.assertEqual(winner['parameters'], {'batch': 2})
        self.assertEqual(replay['baseline'], {'batch': 2})
        self.assertEqual(replay['budget']['max_trials'], 1)
        self.assertIn('--enforce-eager', replay['server']['command'])
        self.assertTrue((self.output / 'research/evidence-0000.json').exists())
        self.assertTrue((self.output / 'trials/0001-exploration/analysis.json').exists())
        self.assertEqual(len(state['analyses']), 3)
        effective = json.loads((self.output / 'trials/0001-exploration/effective-contract.json').read_text())
        self.assertEqual(effective['baseline'], {'batch': 2})
        self.assertIn('--enforce-eager', effective['server']['command'])
        self.assertEqual(state['stop_reason'], 'max_trials')
        self.assertIn('探索终止原因：max_trials', (self.output / 'workflow_report.md').read_text())

    def test_higher_throughput_cannot_override_failed_latency(self):
        def runner(config, candidate, folder, deadline, repeats):
            return measured(config, candidate, repeats,
                            latency=2500 if candidate['batch'] == 3 else 500)

        state, _ = self.execute(runner=runner, planner=PlannerDouble(batch=3))
        self.assertFalse(state['trials'][1]['evaluation']['feasible'])
        self.assertGreater(state['trials'][1]['evaluation']['objective'], state['trials'][0]['evaluation']['objective'])
        winner = json.loads((self.output / 'best.json').read_text())
        self.assertEqual(winner['profile_id'], 'baseline')

    def test_final_confirmation_failure_never_exports_exploration_winner(self):
        def runner(config, candidate, folder, deadline, repeats):
            rows = measured(config, candidate, repeats)
            if folder.name.endswith('confirmation'):
                rows[-1]['metrics'][-1]['quality_score'] = .1
            return rows

        state, _ = self.execute(runner=runner)
        self.assertTrue(state['trials'][1]['evaluation']['feasible'])
        self.assertFalse(state['trials'][-1]['evaluation']['feasible'])
        self.assertFalse((self.output / 'best.json').exists())
        self.assertFalse((self.output / 'best-contract.json').exists())

    def test_research_failure_stops_new_proposals_but_confirms_measured_baseline(self):
        planner = PlannerDouble()
        state, hardware = self.execute(planner=planner, researcher=ResearchDouble(error=RuntimeError('SECRET NETWORK DETAIL')))
        self.assertEqual(hardware.call_count, 2)
        self.assertEqual(len(planner.received), 0)
        self.assertEqual(state['research'][0]['status'], 'failed')
        self.assertNotIn('SECRET NETWORK DETAIL', json.dumps(state))
        self.assertEqual(state['trials'][-1]['profile_id'], 'baseline')
        self.assertEqual(state['trials'][-1]['phase'], 'confirmation')
        self.assertEqual(state['stop_reason'], 'research_unavailable')
        self.assertIn('探索终止原因：research_unavailable', (self.output / 'workflow_report.md').read_text())

    def test_explicit_continue_mode_can_plan_without_research_evidence(self):
        self.source['research']['on_error'] = 'continue'
        self.reload()
        planner = PlannerDouble()
        state, _ = self.execute(planner=planner, researcher=ResearchDouble(error=RuntimeError('unavailable')))
        self.assertEqual(len(planner.received), 1)
        self.assertEqual(planner.received[0][2], [])
        self.assertEqual(state['trials'][1]['profile_id'], 'eager')

    def test_research_call_limit_stops_search_without_losing_confirmed_candidate(self):
        self.source['budget'] = {'max_trials': 4}
        self.source['research']['max_calls'] = 1
        self.reload()
        researcher = ResearchDouble()
        planner = PlannerDouble()
        state, hardware = self.execute(planner=planner, researcher=researcher)
        self.assertEqual(len(researcher.questions), 1)
        self.assertEqual(len(planner.received), 1)
        self.assertEqual(state['research_calls'], 1)
        self.assertEqual(hardware.call_count, 3)
        self.assertEqual(state['trials'][-1]['phase'], 'confirmation')
        self.assertTrue((self.output / 'best.json').exists())
        self.assertEqual(state['stop_reason'], 'research_budget_exhausted')

    def test_empty_research_response_cannot_be_used_as_evidence(self):
        researcher = ResearchDouble()
        researcher.ask = lambda *_args, **_kwargs: {'answer': ' ', 'sources': []}
        planner = PlannerDouble()
        state, _ = self.execute(planner=planner, researcher=researcher)
        self.assertEqual(planner.received, [])
        self.assertEqual(state['research'][0]['status'], 'failed')
        self.assertEqual(state['trials'][-1]['profile_id'], 'baseline')

    def test_foreign_repository_research_cannot_reach_planner(self):
        researcher = ResearchDouble()
        answer = researcher.ask('another/repository', 'fixture')
        researcher.ask = lambda *_args, **_kwargs: answer
        planner = PlannerDouble()
        state, _ = self.execute(planner=planner, researcher=researcher)
        self.assertEqual(planner.received, [])
        self.assertEqual(state['research'][0]['status'], 'failed')
        self.assertEqual(state['stop_reason'], 'research_unavailable')
        self.assertEqual(state['trials'][-1]['profile_id'], 'baseline')
        self.assertEqual(state['trials'][-1]['phase'], 'confirmation')

    def test_malformed_research_metadata_cannot_reach_planner(self):
        original = ResearchDouble().ask('vllm-project/vllm-ascend', 'fixture')
        for field, value in [('sources', 'not-a-list'), ('transport', {'nested': 'bad'}),
                             ('tool', ['bad']), ('tool', 'x' * 101)]:
            with self.subTest(field=field, value=type(value).__name__):
                self.output = self.root / (field + '-' + type(value).__name__)
                researcher = ResearchDouble()
                answer = dict(original, **{field: value})
                researcher.ask = lambda *_args, **_kwargs: answer
                planner = PlannerDouble()
                state, _ = self.execute(planner=planner, researcher=researcher)
                self.assertEqual(planner.received, [])
                self.assertEqual(state['research'][0]['status'], 'failed')
                self.assertEqual(state['stop_reason'], 'research_unavailable')
                self.assertEqual(state['trials'][-1]['profile_id'], 'baseline')
                self.assertEqual(state['trials'][-1]['phase'], 'confirmation')

    def test_illegal_duplicate_or_unavailable_proposals_fall_back_to_legal_strategy(self):
        for case, planner in [('unknown', PlannerDouble(profile='undeclared')),
                              ('invalid', PlannerDouble(batch=1000)),
                              ('duplicate', PlannerDouble(profile='baseline', batch=1)),
                              ('api_failure', PlannerDouble(error=LLMError('PRIVATE API DETAIL')))]:
            with self.subTest(case=case):
                self.output = self.root / case
                state, _ = self.execute(planner=planner)
                self.assertEqual(state['trials'][1]['profile_id'], 'eager')
                self.assertEqual(state['trials'][1]['parameters'], {'batch': 1})
                self.assertEqual(state['decisions'][0]['source'], 'bounded_fallback')
                self.assertNotIn('PRIVATE API DETAIL', json.dumps(state))

    def test_research_time_consumes_search_budget_and_preserves_confirmation_time(self):
        clock = Clock()
        planner = PlannerDouble()
        researcher = ResearchDouble(clock=clock, delay=46)
        with patch('ascend_poc.workflow.time', clock):
            state, hardware = self.execute(planner=planner, researcher=researcher)
        self.assertEqual(len(planner.received), 0)
        self.assertEqual(hardware.call_count, 2)
        self.assertEqual(state['trials'][-1]['phase'], 'confirmation')
        self.assertGreater(researcher.questions[0][2], 0)
        self.assertLessEqual(researcher.questions[0][2], 45)
        self.assertEqual(state['elapsed_s'], 46)
        self.assertEqual(state['stop_reason'], 'search_deadline')

    def test_exhausted_finite_space_needs_no_research_call(self):
        self.source['profiles'] = self.source['profiles'][:1]
        base = json.loads(self.contract_path.read_text())
        base['search_space']['batch'] = [1]
        base['budget']['max_trials'] = 10
        self.contract_path.write_text(json.dumps(base))
        self.reload()
        researcher = ResearchDouble()
        state, hardware = self.execute(researcher=researcher)
        self.assertEqual(hardware.call_count, 2)
        self.assertEqual(researcher.questions, [])
        self.assertTrue(any(e['type'] == 'search_exhausted' for e in state['events']))
        self.assertEqual(state['stop_reason'], 'search_exhausted')

    def test_resume_rejects_changed_profile_or_contract(self):
        self.execute()
        original = copy.deepcopy(self.source)
        self.source['profiles'][1]['description'] += ' changed'
        self.reload()
        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
            self.execute(resume=True)
        self.source = original
        base = json.loads(self.contract_path.read_text())
        base['requirements']['p99_ttft_ms']['max'] = 1000
        self.contract_path.write_text(json.dumps(base))
        self.reload()
        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
            self.execute(resume=True)

    def test_resume_rejects_changed_runtime_and_completed_resume_does_no_work(self):
        self.execute()
        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
            self.execute(resume=True, runtime={'fixture': 2})
        researcher = ResearchDouble()
        state, hardware = self.execute(resume=True, researcher=researcher)
        hardware.assert_not_called()
        self.assertEqual(researcher.questions, [])
        self.assertEqual(state['status'], 'completed')

    def test_interruption_keeps_partial_trial_unaccepted_and_resume_confirms_baseline(self):
        def runner(config, candidate, folder, deadline, repeats):
            if folder.name == '0001-exploration':
                raise KeyboardInterrupt()
            return measured(config, candidate, repeats)

        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner=runner)
        saved = self.read_state()
        self.assertEqual(saved['status'], 'interrupted')
        self.assertEqual(saved['trials'][-1]['status'], 'interrupted')
        self.assertNotIn('evaluation', saved['trials'][-1])
        self.assertFalse((self.output / 'best.json').exists())
        state, hardware = self.execute(resume=True)
        self.assertEqual(hardware.call_count, 1)
        self.assertEqual(state['trials'][-1]['profile_id'], 'baseline')
        self.assertEqual(state['trials'][-1]['phase'], 'confirmation')

    def test_research_query_omits_secret_context_keys_and_process_error_text(self):
        base = json.loads(self.contract_path.read_text())
        base['context'].update(api_key='SHOULD_NOT_SEND', prompt='CUSTOMER_PROMPT', hardware='Ascend test')
        self.contract_path.write_text(json.dumps(base))
        self.reload()
        researcher = ResearchDouble()
        self.execute(researcher=researcher)
        query = researcher.questions[0][1]
        self.assertNotIn('SHOULD_NOT_SEND', query)
        self.assertNotIn('CUSTOMER_PROMPT', query)
        self.assertNotIn('fake-server', query)
        self.assertIn('Ascend test', query)

    def test_large_valid_workflow_query_is_bounded_without_truncating_json(self):
        base = json.loads(self.contract_path.read_text())
        base['context'] = {f'shareable_{i}': ['metadata' * 500] * 20 for i in range(30)}
        # Scenario identifiers have no length cap in the contract. Even these
        # unusual but valid names must not break the transport's query budget.
        base['scenarios'][0]['name'] = 'scenario_' + 'x' * 4000
        self.contract_path.write_text(json.dumps(base))
        template = copy.deepcopy(self.source['profiles'][1])
        for index in range(30):
            self.source['profiles'].append(dict(copy.deepcopy(template), id=f'strategy_{index}', description='strategy' * 3000))
        self.reload()
        scenario = self.spec['_contract']['scenarios'][0]['name']
        analysis = {'trial_id': '0000-exploration', 'status': 'completed', 'feasible': False,
                    'objective': 123, 'violations': [
                        {'metric': 'p99_ttft_ms', 'scenario': scenario, 'value': 3000, 'bounds': {'max': 2000}}
                        for _ in range(30)],
                    'hypotheses': [{'kind': f'kind_{i}', 'summary': 'hypothesis' * 3000,
                                    'evidence_metrics': ['p99_ttft_ms']} for i in range(20)],
                    'research_questions': ['question' * 2000] * 10}
        question = _question(self.spec, analysis)
        self.assertLessEqual(len(question), 16000)
        payload = json.loads(question.split('diagnostic data:\n', 1)[1])
        self.assertIn('measurement_analysis', payload)
        self.assertIsInstance(payload['operator_allowed_strategies'], list)


if __name__ == '__main__':
    unittest.main()
