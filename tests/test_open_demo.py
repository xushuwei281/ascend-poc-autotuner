"""Exercise the open mode through real local HTTP and owned subprocesses."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.open_demo import run_open_demo
from ascend_poc.open_tuner import report_autotune


class OpenWireDemoTests(TestCase):
    def test_dynamic_options_feedback_recovery_and_saved_evidence(self):
        with TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'POC_OPEN_DEMO_KEY': 'existing-value-must-be-restored'}), \
             redirect_stdout(io.StringIO()):
            output = Path(folder) / 'demo'
            state = run_open_demo(output)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['generation'], 2)
            self.assertEqual(state['counters'], {'llm_calls': 10, 'research_calls': 6})
            self.assertEqual(os.environ['POC_OPEN_DEMO_KEY'], 'existing-value-must-be-restored')
            self.assertEqual(state['intent']['metric_requirements'], {'p99_tpot_ms': {'max': 50}})

            first, failed, corrected = state['experiments']
            self.assertEqual(first['status'], 'promoted')
            self.assertTrue(first['candidate']['engine']['enable-prefix-caching'])
            self.assertEqual(first['candidate']['engine']['compilation-config'], {'mode': 3})
            self.assertEqual(failed['status'], 'exploration_failed')
            self.assertGreater(failed['feedback']['evaluation']['objective'],
                               first['feedback']['evaluation']['objective'])
            self.assertFalse(failed['feedback']['evaluation']['feasible'])
            self.assertNotIn('verification_trials', failed)
            self.assertEqual(corrected['status'], 'promoted')
            self.assertEqual(corrected['candidate']['engine']['tensor-parallel-size'], 2)
            self.assertLess(corrected['candidate']['workload']['concurrency'],
                            failed['candidate']['workload']['concurrency'])
            self.assertEqual(state['champion']['experiment_id'], corrected['id'])
            self.assertFalse(state['champion']['comparison']['same_workload'])
            self.assertEqual(state['champion']['comparison']['kind'], 'different_operating_point')

            # The executable argv proves the nested JSON and boolean survived
            # the complete provider -> validation -> rendering -> process path.
            first_trial = output / 'trials' / first['exploration_trial']
            commands = json.loads((first_trial / 'commands.json').read_text())
            serving = next(item['argv'] for item in commands if item['stage'] == 'server')
            self.assertIn('--enable-prefix-caching', serving)
            self.assertIn('--compilation-config={"mode":3}', serving)

            (output / 'report.md').unlink()
            report = report_autotune(output)
            self.assertEqual(report, output / 'report.md')
            self.assertIn('SYNTHETIC', report.read_text())
            self.assertIn('different_operating_point', report.read_text())
            with self.assertRaisesRegex(ValueError, 'fresh, empty'):
                run_open_demo(output)

            # Historical reporting cannot silently bless changed raw results.
            measured = first_trial / 'scenario-00' / 'repeat-000' / 'metrics.json'
            data = json.loads(measured.read_text())
            data['output_tokens_per_second'] += 1
            measured.write_text(json.dumps(data))
            with self.assertRaises((ValueError, RuntimeError)):
                report_autotune(output)
