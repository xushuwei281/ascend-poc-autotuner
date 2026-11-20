"""Real subprocess coverage for partial exploration and measured execution cost."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from ascend_poc.metrics import evaluate_trial
from ascend_poc.runner import RunnerError, TrialResults, run_trial
from tests.test_runner import SERVER, alive, free_port


class EarlyStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, metrics=None):
        port = free_port()
        metrics = metrics if metrics is not None else {'throughput': 100, 'latency': 500}
        command = [sys.executable, '-c',
                   'import json,sys; json.dump(json.loads(sys.argv[2]),open(sys.argv[1],"w"))',
                   '{trial_dir}/metrics.json', json.dumps(metrics)]
        return {'_repo_dir': str(self.root), 'budget': {'early_stop': True, 'repetitions': 3},
                'objective': {'metric': 'throughput'}, 'requirements': {'latency': {'max': 100}},
                'server': {'command': [sys.executable, '-c', SERVER, str(port)],
                           'health_url': f'http://127.0.0.1:{port}/health',
                           'startup_timeout_s': 2, 'shutdown_timeout_s': .2},
                'scenarios': [{'name': name, 'command': command, 'result_file': '{trial_dir}/metrics.json',
                               'timeout_s': 2} for name in ('short', 'long')]}

    def measure(self, config, folder='trial', phase='exploration', repetitions=3):
        return run_trial(config, {}, self.root / folder, time.monotonic() + 10,
                         repetitions, phase=phase)

    def test_first_observed_violation_stops_remaining_matrix_and_cleans_up(self):
        config = self.config()
        rows = self.measure(config)
        self.assertIsInstance(rows, list)
        self.assertIsInstance(rows, TrialResults)
        self.assertEqual(rows.status, 'infeasible_early')
        self.assertEqual(rows.aborted_after,
                         {'scenario': 'short', 'repetition': 1, 'violated_metrics': ['latency']})
        self.assertEqual([r['name'] for r in rows], ['short'])
        self.assertEqual(len(rows[0]['metrics']), 1)
        self.assertFalse((self.root / 'trial/scenario-00/repeat-001').exists())
        self.assertFalse((self.root / 'trial/scenario-01').exists())
        evaluated = evaluate_trial(config, rows)
        self.assertFalse(evaluated['feasible'])
        self.assertIsNone(evaluated['objective'])
        self.assertEqual(rows[0]['metrics'][0]['latency'], 500)
        process = json.loads((self.root / 'trial/server-process.json').read_text())
        self.assertFalse(alive(process['pid']))
        saved = json.loads((self.root / 'trial/trial-execution.json').read_text())
        self.assertEqual(saved['aborted_after'], rows.aborted_after)
        self.assertEqual(saved['timings'], rows.timings)
        # Existing JSON consumers still see the historical scenario-list shape.
        self.assertEqual(json.loads(json.dumps(rows)), list(rows))

    def test_missing_future_measurements_do_not_trigger_early_stop(self):
        config = self.config({'throughput': 100, 'latency': 50})
        rows = self.measure(config)
        self.assertEqual(rows.status, 'completed')
        self.assertIsNone(rows.aborted_after)
        self.assertEqual([len(r['metrics']) for r in rows], [3, 3])
        self.assertTrue(evaluate_trial(config, rows)['feasible'])

    def test_later_violation_preserves_completed_scenario_and_prior_repetitions(self):
        config = self.config({'throughput': 100, 'latency': 50})
        config['scenarios'][1]['command'] = [sys.executable, '-c',
            'import json,sys; latency=500 if "repeat-001" in sys.argv[1] else 50; '
            'json.dump({"throughput":100,"latency":latency},open(sys.argv[1],"w"))',
            '{trial_dir}/metrics.json']
        rows = self.measure(config)
        self.assertEqual(rows.status, 'infeasible_early')
        self.assertEqual([len(r['metrics']) for r in rows], [3, 2])
        self.assertEqual(rows.aborted_after,
                         {'scenario': 'long', 'repetition': 2, 'violated_metrics': ['latency']})
        self.assertFalse((self.root / 'trial/scenario-01/repeat-002').exists())
        self.assertFalse(evaluate_trial(config, rows)['feasible'])

    def test_confirmation_references_and_legacy_calls_never_early_stop(self):
        for index, phase in enumerate((None, 'confirmation', 'reference_before', 'reference_after')):
            with self.subTest(phase=phase):
                config = self.config()
                rows = self.measure(config, f'full-{index}', phase)
                self.assertEqual(rows.status, 'completed')
                self.assertIsNone(rows.aborted_after)
                self.assertEqual([len(r['metrics']) for r in rows], [3, 3])
                self.assertFalse(evaluate_trial(config, rows)['feasible'])

    def test_explicit_setting_is_required_even_for_exploration(self):
        config = self.config()
        config['budget']['early_stop'] = False
        rows = self.measure(config)
        self.assertEqual(rows.status, 'completed')
        self.assertEqual([len(r['metrics']) for r in rows], [3, 3])

    def test_quality_is_measured_before_early_stop_and_timing_is_partitioned(self):
        config = self.config({'throughput': 100, 'latency': 50})
        config['requirements']['accuracy'] = {'min': .9}
        for scenario in config['scenarios']:
            scenario['quality_command'] = [sys.executable, '-c',
                'import json,sys;json.dump({"accuracy":.5},open(sys.argv[1],"w"))',
                '{trial_dir}/quality.json']
            scenario['quality_result_file'] = '{trial_dir}/quality.json'
            scenario['before_each_command'] = [sys.executable, '-c', 'pass']
            scenario['warmup_command'] = [sys.executable, '-c', 'pass']
        rows = self.measure(config)
        self.assertEqual(rows.status, 'infeasible_early')
        self.assertEqual(rows.aborted_after['violated_metrics'], ['accuracy'])
        self.assertEqual(rows[0]['metrics'][0]['accuracy'], .5)
        commands = json.loads((self.root / 'trial/commands.json').read_text())
        self.assertEqual([c['stage'] for c in commands],
                         ['server', 'before_each', 'warmup', 'benchmark', 'quality'])
        for name in ('startup_s', 'benchmark_s', 'quality_s'):
            self.assertGreater(rows.timings[name], 0)
        self.assertGreaterEqual(rows.timings['other_s'], 0)
        self.assertEqual(rows.timings['repetitions'], 3)
        self.assertAlmostEqual(rows.timings['total_s'], sum(rows.timings[name] for name in
                               ('startup_s', 'benchmark_s', 'quality_s', 'other_s')), places=6)

    def test_missing_metric_is_a_failure_in_observed_run_not_a_zero(self):
        for index, metrics in enumerate(({'throughput': 100}, {'latency': 50})):
            with self.subTest(metrics=metrics):
                rows = self.measure(self.config(metrics), f'missing-{index}')
                self.assertEqual(rows.status, 'infeasible_early')
                missing = 'latency' if index == 0 else 'throughput'
                self.assertEqual(rows.aborted_after['violated_metrics'], [missing])
                self.assertNotIn(missing, rows[0]['metrics'][0])

    def test_execution_failure_remains_failure_and_records_elapsed_cost(self):
        config = self.config()
        config['scenarios'][0]['command'] = [sys.executable, '-c', 'raise SystemExit(7)']
        with self.assertRaisesRegex(RunnerError, 'status 7'):
            self.measure(config)
        saved = json.loads((self.root / 'trial/trial-execution.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertIsNone(saved['aborted_after'])
        self.assertGreater(saved['timings']['benchmark_s'], 0)
        self.assertGreater(saved['timings']['total_s'], saved['timings']['startup_s'])
        process = json.loads((self.root / 'trial/server-process.json').read_text())
        self.assertFalse(alive(process['pid']))


if __name__ == '__main__':
    unittest.main()
