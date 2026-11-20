"""Request telemetry is measured locally and only bounded aggregates reach planners."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from ascend_poc.cli import main
from ascend_poc.config import validate_config
from ascend_poc.diagnosis import analyze_trial
from ascend_poc.runner import RunnerError, run_trial
from ascend_poc.strategy import load_workflow
from ascend_poc.workflow import _question
from ascend_poc.workflow_planner import _analysis_summary
from ascend_poc.workload import summarize_requests
from tests.test_runner import SERVER, free_port
from tests.test_tuner import contract


class DynamicIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poc-dynamic-integration-')
        self.root = Path(self.temp.name)
        self.rows = [
            {'request_id': 'private-request-1', 'agent_id': 'PRIVATE_AGENT', 'branch_id': 'PRIVATE_BRANCH',
             'started_at_s': 0, 'completed_at_s': .5, 'input_tokens': 100, 'output_tokens': 20,
             'cached_tokens': 80, 'context_version': 0},
            {'request_id': 'private-request-2', 'agent_id': 'PRIVATE_AGENT', 'branch_id': 'PRIVATE_BRANCH',
             'started_at_s': 61, 'completed_at_s': 62, 'input_tokens': 200, 'output_tokens': 30,
             'context_version': 1},
        ]

    def tearDown(self):
        self.temp.cleanup()

    def runner_config(self, *, completed=2, failed=0, telemetry=True):
        port = free_port()
        script = self.root / 'benchmark.py'
        script.write_text(
            'import json, pathlib, sys\n'
            f'metrics = {repr({"completed": completed, "failed": failed, "output_tokens_per_second": 123, "p99_ttft_ms": 30})}\n'
            'pathlib.Path(sys.argv[1]).write_text(json.dumps(metrics))\n'
            + (f'rows = {repr(self.rows)}\n'
               'pathlib.Path(sys.argv[2]).write_text("".join(json.dumps(row) + "\\n" for row in rows))\n'
               if telemetry else ''), encoding='utf-8')
        return {
            '_repo_dir': str(self.root), 'llm': {'api_key_env': 'TEST_DYNAMIC_PLANNER_KEY'},
            'server': {'command': [sys.executable, '-c', SERVER, str(port)],
                       'health_url': f'http://127.0.0.1:{port}/health',
                       'startup_timeout_s': 2, 'shutdown_timeout_s': .2},
            'scenarios': [{'name': 'agent_dag',
                           'command': [sys.executable, str(script), '{trial_dir}/metrics.json', '{trial_dir}/telemetry.jsonl'],
                           'result_file': '{trial_dir}/metrics.json',
                           'telemetry_file': '{trial_dir}/telemetry.jsonl',
                           'metrics_format': 'canonical', 'timeout_s': 2}],
        }

    def test_measured_telemetry_is_collected_per_repetition_without_identity_leaks(self):
        result = run_trial(self.runner_config(), {}, self.root / 'run', time.monotonic() + 10, 2)
        summaries = result[0]['workload_summaries']
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertEqual(summary['successful'], 2)
            self.assertEqual(summary['cache_observation_coverage'], .5)
            self.assertEqual(summary['overall']['mean_uncached_tokens'], 20)
            self.assertEqual(summary['overall']['mean_output_tokens'], 25)
            self.assertEqual(summary['context_version_changes'], 1)
        for repetition in (0, 1):
            folder = self.root / 'run' / 'scenario-00' / f'repeat-{repetition:03d}'
            saved = (folder / 'workload-summary.json').read_text()
            self.assertNotIn('PRIVATE_', saved)
            self.assertNotIn('private-request-', saved)
            self.assertEqual(json.loads((folder / 'normalized.json').read_text())['completed'], 2)

    def test_missing_telemetry_fails_an_otherwise_successful_measurement(self):
        with self.assertRaisesRegex(RunnerError, 'telemetry.*missing|telemetry.*invalid'):
            run_trial(self.runner_config(telemetry=False), {}, self.root / 'run', time.monotonic() + 5, 1)

    def test_hook_cannot_preseed_a_stale_telemetry_result(self):
        config = self.runner_config()
        config['scenarios'][0]['before_each_command'] = [
            sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("")',
            '{trial_dir}/telemetry.jsonl']
        with self.assertRaisesRegex(RunnerError, 'fresh'):
            run_trial(config, {}, self.root / 'run', time.monotonic() + 5, 1)

    def test_success_and_failure_counts_must_match_benchmark_metrics(self):
        for index, values in enumerate(((1, 0), (2, 1))):
            with self.subTest(counts=values), self.assertRaisesRegex(RunnerError, 'counts disagree'):
                run_trial(self.runner_config(completed=values[0], failed=values[1]), {},
                          self.root / f'run-{index}', time.monotonic() + 5, 1)

    def test_contract_rejects_telemetry_that_overwrites_metric_or_quality_files(self):
        for target in ('{trial_dir}/metrics.json', '{trial_dir}/quality.json'):
            base = contract('dynamic-schema-test')
            scenario = base['scenarios'][0]
            scenario.update(quality_command=['quality-test'], quality_result_file='{trial_dir}/quality.json',
                            telemetry_file=target)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'telemetry_file'):
                validate_config(base)

    def test_controller_summary_paths_cannot_overwrite_measurement_evidence(self):
        for index, (field, name) in enumerate((('result_file', 'workload-summary.json'),
                                               ('telemetry_file', 'workload-summary.json'),
                                               ('telemetry_file', 'normalized.json'))):
            config = self.runner_config()
            config['scenarios'][0][field] = '{trial_dir}/' + name
            with self.subTest(field=field, name=name), self.assertRaisesRegex(RunnerError, 'controller artifacts'):
                run_trial(config, {}, self.root / f'collision-{index}', time.monotonic() + 5, 1)

    def test_only_explicit_telemetry_scenarios_supply_workload_diagnostics(self):
        base = contract('diagnostic-test')
        base['scenarios'][0]['telemetry_file'] = '{trial_dir}/telemetry.jsonl'
        summary = summarize_requests(self.rows)
        record = {'id': '0000-exploration', 'status': 'completed',
                  'evaluation': {'feasible': True, 'objective': 123, 'checks': []},
                  'scenario_results': [{'name': scenario['name'], 'workload_summaries': [summary]}
                                       for scenario in base['scenarios']]}
        analysis = analyze_trial(record, base)
        self.assertEqual([w['scenario'] for w in analysis['workloads']], ['chat'])
        self.assertEqual(analysis['workloads'][0]['summary']['cache_observation_coverage'], .5)
        self.assertNotIn('PRIVATE_', json.dumps(analysis))

    def test_planner_sanitizes_even_forged_workload_summaries(self):
        base = contract('planner-test')
        summary = summarize_requests(self.rows)
        summary.update(messages=['PRIVATE_PROMPT'], agent_id='PRIVATE_AGENT', raw_log='PRIVATE_LOG')
        summary['overall']['secret'] = 'PRIVATE_SECRET'
        summary['overall']['mean_cached_tokens'] = None
        summary['windows'][0]['messages'] = 'PRIVATE_WINDOW'
        analysis = {'workloads': [{'scenario': 'chat', 'repetition': 1, 'summary': summary},
                                 {'scenario': 'foreign', 'summary': summary}]}
        clean = _analysis_summary(analysis, base)
        self.assertEqual(len(clean['workloads']), 1)
        self.assertEqual(clean['workloads'][0]['summary']['successful'], 2)
        self.assertIsNone(clean['workloads'][0]['summary']['overall']['mean_cached_tokens'])
        self.assertNotIn('PRIVATE_', json.dumps(clean))
        self.assertNotIn('secret', json.dumps(clean))

    def test_large_workload_research_query_is_bounded_and_has_valid_json(self):
        base = contract('research-test')
        base['scenarios'][0]['name'] = 'scenario_' + 'x' * 3000
        (self.root / 'contract.json').write_text(json.dumps(base))
        spec_raw = {'schema_version': 1, 'name': 'test', 'contract': 'contract.json',
                    'profiles': [{'id': 'baseline', 'description': 'Baseline',
                                  'compatibility': {'status': 'verified', 'evidence': ['local fixture']}}]}
        path = self.root / 'workflow.json'
        path.write_text(json.dumps(spec_raw))
        spec = load_workflow(path)
        summary = summarize_requests(self.rows)
        summary['windows'] = [copy.deepcopy(summary['windows'][0]) for _ in range(100)]
        summary['agent_id'] = 'PRIVATE_AGENT'
        analysis = {'trial_id': 'test', 'workloads': [
            {'scenario': base['scenarios'][0]['name'], 'repetition': i + 1, 'summary': summary}
            for i in range(100)]}
        query = _question(spec, analysis)
        self.assertLessEqual(len(query), 16000)
        payload = json.loads(query.split('diagnostic data:\n', 1)[1])
        self.assertIn('measurement_analysis', payload)
        self.assertNotIn('PRIVATE_', query)

    def test_cli_shadow_plan_never_claims_to_apply_a_backend_change(self):
        repo = Path(__file__).resolve().parents[1]
        destination = self.root / 'shadow.json'
        with redirect_stdout(io.StringIO()) as output:
            status = main(['adaptive-plan', '--config', str(repo / 'examples/adaptive-shadow.json'),
                           '--windows', str(repo / 'examples/adaptive-windows.json'),
                           '--output', str(destination)])
        self.assertEqual(status, 0)
        decision = json.loads(destination.read_text())
        self.assertEqual(decision['measurement_status'], 'prediction_only')
        self.assertIn(decision['action'], {'hold', 'recommend'})
        self.assertIn('no backend changes', output.getvalue())


if __name__ == '__main__':
    unittest.main()
