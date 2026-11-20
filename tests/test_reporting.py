"""Historical reports must use saved evidence without rerunning a controller."""
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import io
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.audit import append_trace
from ascend_poc.cli import main
from ascend_poc.config import load_config
from ascend_poc.reporting import regenerate_report
from ascend_poc.tuner import tune
from ascend_poc.workflow import _checkpoint_hash
from tests import test_workflow as fixtures


class ReportingTests(TestCase):
    def setUp(self):
        # Compose the workflow fixture without inheriting its test methods.
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.output = self.fixture.output

    def run_workflow(self):
        return self.fixture.execute()[0]

    def discard_exports(self):
        for name in ('workflow_report.md', 'best.json', 'best-contract.json', 'anytime.csv'):
            (self.output / name).unlink(missing_ok=True)

    def save_state(self, state):
        (self.output / 'workflow_state.json').write_text(json.dumps(state), encoding='utf-8')

    def assert_no_exports(self):
        for name in ('workflow_report.md', 'best.json', 'best-contract.json', 'anytime.csv'):
            self.assertFalse((self.output / name).exists(), name)

    def test_workflow_report_restores_confirmed_exports_without_execution(self):
        self.run_workflow()
        expected_best = (self.output / 'best.json').read_bytes()
        expected_report = (self.output / 'workflow_report.md').read_bytes()
        self.discard_exports()
        with patch('ascend_poc.workflow.run_trial', side_effect=AssertionError('no measurements')), \
             patch('ascend_poc.workflow.runtime_signature', side_effect=AssertionError('no runtime probe')), \
             patch('ascend_poc.workflow.verify_frozen_files', side_effect=AssertionError('no live inputs')), \
             patch('ascend_poc.workflow.WorkflowPlanner', side_effect=AssertionError('no LLM')):
            result = regenerate_report(self.output)
        self.assertEqual(result, self.output / 'workflow_report.md')
        self.assertEqual(result.read_bytes(), expected_report)
        self.assertEqual((self.output / 'best.json').read_bytes(), expected_best)
        self.assertTrue((self.output / 'best-contract.json').exists())
        self.assertTrue((self.output / 'anytime.csv').exists())

    def test_original_contract_reference_is_preserved_and_not_dereferenced(self):
        nested_contract = self.fixture.root / 'customer' / 'contracts' / 'base.json'
        nested_contract.parent.mkdir(parents=True)
        self.fixture.contract_path.rename(nested_contract)
        self.fixture.source['contract'] = 'customer/contracts/base.json'
        self.fixture.reload()
        self.run_workflow()
        snapshots = {name: (self.output / name).read_bytes()
                     for name in ('workflow.json', 'contract.json', 'workflow_state.json', 'events.jsonl')}
        nested_contract.unlink()
        self.fixture.workflow_path.unlink()
        self.discard_exports()
        regenerate_report(self.output)
        for name, content in snapshots.items():
            self.assertEqual((self.output / name).read_bytes(), content, name)
        self.assertTrue((self.output / 'best.json').exists())

    def test_changed_state_cannot_regenerate_winner(self):
        state = self.run_workflow()
        state['trials'][-1]['evaluation']['objective'] = 999999
        self.save_state(state)
        self.discard_exports()
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            regenerate_report(self.output)
        self.assert_no_exports()

    def test_broken_trace_cannot_regenerate_winner(self):
        self.run_workflow()
        journal = self.output / 'events.jsonl'
        journal.write_bytes(journal.read_bytes()[:-1])
        self.discard_exports()
        with self.assertRaisesRegex(ValueError, 'Trace'):
            regenerate_report(self.output)
        self.assert_no_exports()

    def test_changed_contract_cannot_regenerate_winner(self):
        self.run_workflow()
        path = self.output / 'contract.json'
        contract = json.loads(path.read_text())
        contract['name'] = 'Another customer'
        path.write_text(json.dumps(contract), encoding='utf-8')
        self.discard_exports()
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            regenerate_report(self.output)
        self.assert_no_exports()

    def test_invalidated_inputs_cannot_regenerate_winner(self):
        state = self.run_workflow()
        state['integrity_valid'] = False
        # Produce a coherent journal checkpoint for an invalidated campaign.
        checkpoint = append_trace(self.output, {'type': 'checkpoint', 'state_hash': _checkpoint_hash(state)})
        state['trace_anchor'] = {'sequence': checkpoint['sequence'], 'hash': checkpoint['hash']}
        self.save_state(state)
        self.discard_exports()
        with self.assertRaisesRegex(ValueError, 'invalidated'):
            regenerate_report(self.output)
        self.assert_no_exports()

    def test_ambiguous_output_is_rejected_before_writing(self):
        self.run_workflow()
        (self.output / 'state.json').write_text('{}', encoding='utf-8')
        self.discard_exports()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            regenerate_report(self.output)
        self.assert_no_exports()

    def test_active_controller_output_is_locked(self):
        self.run_workflow()
        self.discard_exports()
        with (self.output / '.run.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another process'):
                regenerate_report(self.output)
        self.assert_no_exports()

    def test_cli_prints_workflow_report_path_and_reports_corruption(self):
        state = self.run_workflow()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(['report', '--output', str(self.output)]), 0)
        self.assertEqual(stdout.getvalue().strip(), str(self.output / 'workflow_report.md'))
        state['elapsed_s'] += 1
        self.save_state(state)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main(['report', '--output', str(self.output)]), 1)
        self.assertIn('checkpoint', stderr.getvalue())

    def test_legacy_tune_report_and_contract_fingerprint(self):
        config = load_config(self.fixture.contract_path)
        config['budget']['max_trials'] = 1
        with patch('ascend_poc.tuner.run_trial', side_effect=self.fixture.runner), \
             patch('ascend_poc.tuner.runtime_signature', return_value={'fixture': 1}), \
             redirect_stdout(io.StringIO()):
            tune(config, self.output, planner_name='random')
        expected_report = (self.output / 'report.md').read_bytes()
        (self.output / 'report.md').unlink()
        (self.output / 'best.json').unlink()
        result = regenerate_report(self.output)
        self.assertEqual(result, self.output / 'report.md')
        self.assertEqual(result.read_bytes(), expected_report)
        self.assertTrue((self.output / 'best.json').exists())
        state_path = self.output / 'state.json'
        state = json.loads(state_path.read_text())
        state['config_hash'] = '0' * 64
        state_path.write_text(json.dumps(state), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            regenerate_report(self.output)

    def test_missing_output_or_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            regenerate_report(self.output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'No workflow_state.json or state.json'):
            regenerate_report(self.output)
