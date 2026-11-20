"""RSI experiments retain the independent evaluator and fail-closed promotion evidence."""
import copy
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.rsi_experiment import run_experiment, read_experiment, _hash
from ascend_poc.audit import append_trace
from ascend_poc.workflow import _checkpoint_hash
from ascend_poc.workload import summarize_requests
from tests import test_workflow as fixtures
from tests.test_tuner import measured


class RSIExperimentTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.output = self.fixture.output
        self.reference = {'profile_id': 'baseline', 'parameters': {'batch': 1}}
        self.candidate = {'profile_id': 'eager', 'parameters': {'batch': 2}}

    def execute(self, *, candidate=None, reference=None, runner=None, resume=False, remaining=60):
        with patch('ascend_poc.workflow.run_trial', side_effect=runner or self.fixture.runner) as hardware, \
             patch('ascend_poc.workflow.runtime_signature', return_value={'fixture': 1}), \
             patch('ascend_poc.workflow.WorkflowPlanner', side_effect=AssertionError('no child LLM')), \
             patch('ascend_poc.deepwiki.DeepWikiClient', side_effect=AssertionError('no child network')), \
             redirect_stdout(io.StringIO()):
            result = run_experiment(self.fixture.spec, candidate or self.candidate, reference or self.reference,
                                    self.output, remaining, 45, resume=resume)
        return result, hardware

    def state(self):
        return json.loads((self.output / 'workflow_state.json').read_text())

    def test_full_candidate_aba_uses_same_customer_matrix_without_llm_or_network(self):
        seen = []
        original = copy.deepcopy(self.fixture.spec)
        def runner(config, params, folder, deadline, repeats):
            seen.append(copy.deepcopy(config))
            return measured(config, params, repeats)
        result, hardware = self.execute(runner=runner)
        self.assertEqual(hardware.call_count, 5)
        self.assertEqual(self.fixture.spec, original)
        self.assertTrue(result['verification']['accepted'])
        self.assertTrue(result['verification']['improved'])
        self.assertEqual(result['confirmed_objective'], 190)
        self.assertEqual(result['candidate'], self.candidate)
        self.assertEqual(result['reference'], self.reference)
        for config in seen:
            for field in ('scenarios', 'requirements', 'objective', 'benchmark_env'):
                self.assertEqual(config[field], self.fixture.spec['_contract'][field])
        self.assertEqual(self.state()['llm_calls'], 0)
        evidence = json.loads((self.output / 'research/evidence-0000.json').read_text())
        self.assertEqual(evidence['transport'], 'local')
        self.assertEqual(evidence['sources'], [])
        self.assertEqual(result, read_experiment(self.output))

    def test_new_generation_uses_incumbent_recipe_and_parameters_as_reference(self):
        reference = {'profile_id': 'eager', 'parameters': {'batch': 2}}
        candidate = {'profile_id': 'baseline', 'parameters': {'batch': 3}}
        seen = []
        def runner(config, params, folder, deadline, repeats):
            seen.append((folder.name, config['server']['command'], params))
            return measured(config, params, repeats)
        result, _ = self.execute(candidate=candidate, reference=reference, runner=runner)
        self.assertTrue(result['verification']['improved'])
        self.assertEqual(result['reference'], reference)
        reference_runs = [row for row in seen if 'reference_' in row[0]]
        self.assertEqual(len(reference_runs), 2)
        for _, command, params in reference_runs:
            self.assertIn('--enforce-eager', command)
            self.assertEqual(params, {'batch': 2})
        self.assertEqual(self.fixture.spec['_contract']['baseline'], {'batch': 1})

    def test_analysis_retains_safe_cache_windows_without_raw_prompts_or_logs(self):
        config = json.loads(self.fixture.contract_path.read_text())
        config['scenarios'][0]['telemetry_file'] = '{trial_dir}/telemetry.json'
        self.fixture.contract_path.write_text(json.dumps(config))
        self.fixture.reload()
        def runner(config, params, folder, deadline, repeats):
            rows = measured(config, params, repeats)
            # The final candidate confirmation is the most recent observation.
            cached = 80 if folder.name.endswith('confirmation') else 20
            summary = summarize_requests([
                {'request_id': 'PRIVATE_PROMPT', 'started_at_s': 0, 'completed_at_s': 601,
                 'input_tokens': 100, 'output_tokens': 20, 'cached_tokens': cached}])
            summary.update(prompt='PRIVATE_PROMPT', raw_log='PRIVATE_LOG')
            summary['overall']['api_key'] = 'PRIVATE_API_KEY'
            summary['windows'][0]['log'] = 'PRIVATE_LOG'
            rows[0].update(workload_summaries=[summary] * repeats,
                           prompt='PRIVATE_PROMPT', error='PRIVATE_LOG')
            return rows
        result, _ = self.execute(runner=runner)
        analysis = result['analysis']
        self.assertEqual(analysis['trial_id'], next(trial['id'] for trial in self.state()['trials']
                                                  if trial['phase'] == 'confirmation'))
        self.assertTrue(analysis['feasible'])
        self.assertNotIn('PRIVATE', json.dumps(analysis))
        self.assertNotIn('research_questions', analysis)
        workload = analysis['workloads'][0]['summary']
        self.assertEqual(workload['overall']['mean_cached_tokens'], 80)
        self.assertEqual(workload['overall']['mean_uncached_tokens'], 20)
        self.assertEqual(workload['window_count'], 11)
        self.assertTrue(workload['windows_sampled'])
        self.assertEqual([window['start_s'] for window in workload['windows']], [0, 60, 540, 600])
        self.assertEqual(result, read_experiment(self.output))

    def test_no_candidate_observation_has_empty_analysis(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
        result = read_experiment(self.output)
        self.assertEqual(result['analysis'], {})
        self.assertFalse(result['verification']['accepted'])

    def test_higher_throughput_with_failed_slo_cannot_be_promoted(self):
        def runner(config, params, folder, deadline, repeats):
            return measured(config, params, repeats, latency=2500 if params['batch'] == 2 else 500)
        result, _ = self.execute(runner=runner)
        self.assertFalse(result['evaluation']['feasible'])
        self.assertFalse(result['verification']['accepted'])
        self.assertIsNone(result['confirmed_objective'])
        self.assertEqual([t['profile_id'] for t in self.state()['trials'] if t['phase'] == 'confirmation'], ['reference'])

    def test_slower_feasible_candidate_has_no_false_confirmation(self):
        result, _ = self.execute(candidate=self.reference, reference=self.candidate)
        self.assertTrue(result['evaluation']['feasible'])
        self.assertFalse(result['verification']['accepted'])
        self.assertFalse(result['verification']['improved'])
        self.assertIsNone(result['confirmed_objective'])

    def test_drift_invalidates_apparently_faster_candidate(self):
        def runner(config, params, folder, deadline, repeats):
            rows = measured(config, params, repeats)
            if folder.name.endswith('reference_after'):
                for row in rows:
                    for metrics in row['metrics']:
                        metrics['output_tokens_per_second'] *= 2
            return rows
        result, _ = self.execute(runner=runner)
        self.assertEqual(result['verification']['verdict'], 'INVALID')
        self.assertFalse(result['verification']['accepted'])

    def test_unverified_disabled_unknown_and_invalid_candidates_rejected(self):
        for candidate in ({'profile_id': 'unknown', 'parameters': {'batch': 2}},
                          {'profile_id': 'eager', 'parameters': {'batch': 999}},
                          dict(self.candidate, command=['echo', 'bad']), self.reference):
            with self.assertRaises(ValueError):
                self.execute(candidate=candidate)
        self.fixture.spec['profiles'][1]['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'disabled'):
            self.execute()
        self.fixture.spec['profiles'][1]['enabled'] = True
        self.fixture.spec['profiles'][1]['compatibility']['status'] = 'unverified'
        with self.assertRaisesRegex(ValueError, 'unverified'):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_protocol_edits_are_bound_into_workflow_checkpoint(self):
        self.execute()
        path = self.output / 'experiment-input.json'
        data = json.loads(path.read_text())
        data['candidate']['profile_id'] = 'forged'
        data['input_hash'] = _hash({key: value for key, value in data.items() if key != 'input_hash'})
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'bound'):
            read_experiment(self.output)

    def test_saved_state_or_source_tamper_is_rejected(self):
        self.execute()
        path = self.output / 'experiment-contract.json'
        original = path.read_bytes()
        config = json.loads(original)
        config['requirements'] = {}
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            read_experiment(self.output)
        path.write_bytes(original)
        state = self.state()
        state['trials'][1]['evaluation']['objective'] = 123456
        (self.output / 'workflow_state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            read_experiment(self.output)

    def test_aba_verdict_is_recomputed_even_with_coherent_new_checkpoint(self):
        self.execute()
        state = self.state()
        confirm = next(row for row in state['trials'] if row['phase'] == 'confirmation')
        confirm['verification']['relative_improvement'] = 99
        checkpoint = append_trace(self.output, {'type': 'checkpoint', 'state_hash': _checkpoint_hash(state)})
        state['trace_anchor'] = {'sequence': checkpoint['sequence'], 'hash': checkpoint['hash']}
        (self.output / 'workflow_state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'independent ABA'):
            read_experiment(self.output)

    def test_completed_resume_retains_original_deadline_and_does_not_measure(self):
        result, _ = self.execute()
        deadline = self.state()['deadline_epoch']
        again, hardware = self.execute(resume=True, remaining=20)
        self.assertEqual(again, result)
        self.assertEqual(hardware.call_count, 0)
        self.assertEqual(self.state()['deadline_epoch'], deadline)
        with self.assertRaisesRegex(ValueError, 'inputs'):
            self.execute(candidate={'profile_id': 'eager', 'parameters': {'batch': 3}}, resume=True)

    def test_interrupted_aba_resume_reruns_complete_group_without_extending_deadline(self):
        def interrupt(config, params, folder, deadline, repeats):
            if folder.name.endswith('confirmation'):
                raise KeyboardInterrupt()
            return measured(config, params, repeats)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner=interrupt)
        deadline = self.state()['deadline_epoch']
        self.assertFalse(read_experiment(self.output)['verification']['accepted'])
        with self.assertRaisesRegex(ValueError, 'Remaining global budget'):
            self.execute(resume=True, remaining=1)
        result, hardware = self.execute(resume=True)
        self.assertEqual(hardware.call_count, 3)
        self.assertTrue(result['verification']['improved'])
        self.assertEqual(self.state()['deadline_epoch'], deadline)

    def test_incomplete_initialization_cannot_restart_a_deadline(self):
        with patch('ascend_poc.rsi_experiment.run_workflow', side_effect=OSError('interrupted initialization')):
            with self.assertRaises(OSError):
                self.execute()
        with self.assertRaisesRegex(ValueError, 'initialization'):
            self.execute(resume=True)

    def test_historical_read_does_not_require_original_catalog_or_current_hardware(self):
        result, _ = self.execute()
        self.fixture.workflow_path.unlink()
        self.fixture.contract_path.unlink()
        with patch('ascend_poc.workflow.runtime_signature', side_effect=AssertionError('not current hardware')), \
             patch('ascend_poc.workflow.run_trial', side_effect=AssertionError('not a replay')):
            self.assertEqual(read_experiment(self.output), result)

    def test_invalid_budget_and_nonempty_output_rejected(self):
        for remaining in (0, -1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                self.execute(remaining=remaining)
        self.output.mkdir()
        (self.output / 'foreign.txt').write_text('Do not overwrite')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            self.execute()
        self.assertEqual((self.output / 'foreign.txt').read_text(), 'Do not overwrite')

    def test_quality_requirement_is_preserved_when_candidate_is_incumbent(self):
        for scenario in self.fixture.spec['_contract']['scenarios']:
            scenario['quality_command'] = ['quality-script']
            scenario['quality_result_file'] = '{trial_dir}/quality.json'
        # Rebuild the valid source config, not its private materialized copies.
        path = self.fixture.contract_path
        config = json.loads(path.read_text())
        config['scenarios'] = self.fixture.spec['_contract']['scenarios']
        path.write_text(json.dumps(config))
        self.fixture.source['profiles'][1].update(requires_quality=True, quality_metric='quality_score')
        self.fixture.reload()
        self.execute(reference=self.candidate, candidate={'profile_id': 'baseline', 'parameters': {'batch': 3}})
        child = json.loads((self.output / 'workflow.json').read_text())
        self.assertTrue(child['profiles'][0]['requires_quality'])
        self.assertEqual(child['profiles'][0]['quality_metric'], 'quality_score')
