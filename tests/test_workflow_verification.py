"""End-to-end acceptance checks for bracketed verification and frozen inputs."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.audit import append_trace, read_trace
from ascend_poc.runner import RunnerError
from ascend_poc.workflow import WorkflowIntegrityError, run_workflow
from tests import test_workflow as fixtures
from tests.test_tuner import measured


class WorkflowVerificationTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.source['verification'] = {'mode': 'aba'}
        self.fixture.reload()
        self.output = self.fixture.output

    def execute(self, runner=None, *, resume=False, runtime=None, clock=None):
        def default_runner(config, parameters, folder, deadline, repeats):
            return measured(config, parameters, repeats)

        with patch('ascend_poc.workflow.run_trial', side_effect=runner or default_runner) as hardware, \
             patch('ascend_poc.workflow.runtime_signatures', side_effect=lambda configs: {
                 key: (runtime(config) if runtime else {'fixture': 1}) for key, config in configs.items()}), \
             redirect_stdout(io.StringIO()):
            self.hardware = hardware
            if clock:
                with patch('ascend_poc.workflow.time', clock):
                    state = run_workflow(self.fixture.spec, self.output, resume=resume,
                                         planner=fixtures.PlannerDouble(), researcher=fixtures.ResearchDouble())
            else:
                state = run_workflow(self.fixture.spec, self.output, resume=resume,
                                     planner=fixtures.PlannerDouble(), researcher=fixtures.ResearchDouble())
        return state, hardware

    def state(self):
        return json.loads((self.output / 'workflow_state.json').read_text())

    def confirmation(self, state):
        return next(t for t in reversed(state['trials']) if t['phase'] == 'confirmation')

    def assert_no_winner(self):
        self.assertFalse((self.output / 'best.json').exists())
        self.assertFalse((self.output / 'best-contract.json').exists())

    def test_aba_order_and_no_winner_before_complete_bracket(self):
        order = []

        def runner(config, parameters, folder, deadline, repeats):
            self.assert_no_winner()
            phase = folder.name.split('-', 1)[1]
            order.append((phase, parameters['batch'], repeats))
            return measured(config, parameters, repeats)

        state, hardware = self.execute(runner)
        self.assertEqual(order, [('exploration', 1, 2), ('exploration', 2, 2),
                                 ('reference_before', 1, 3), ('confirmation', 2, 3),
                                 ('reference_after', 1, 3)])
        self.assertEqual(hardware.call_count, 5)
        verification = self.confirmation(state)['verification']
        self.assertEqual(verification['verdict'], 'ACCEPTED')
        self.assertTrue(verification['accepted'])
        self.assertTrue(verification['improved'])
        winner = json.loads((self.output / 'best.json').read_text())
        self.assertEqual(winner['verification'], verification)
        self.assertEqual(winner['profile_id'], 'eager')

    def test_noisy_baselines_make_otherwise_feasible_candidate_invalid(self):
        def runner(config, parameters, folder, deadline, repeats):
            result = measured(config, parameters, repeats)
            if folder.name.endswith('reference_after'):
                for row in result:
                    for metrics in row['metrics']:
                        metrics['output_tokens_per_second'] *= 1.25
            return result

        state, _ = self.execute(runner)
        candidate = self.confirmation(state)
        self.assertTrue(candidate['evaluation']['feasible'])
        self.assertEqual(candidate['verification']['verdict'], 'INVALID')
        self.assertFalse(candidate['verification']['accepted'])
        self.assert_no_winner()

    def test_failed_last_reference_does_not_fall_back_to_single_confirmation(self):
        def runner(config, parameters, folder, deadline, repeats):
            if folder.name.endswith('reference_after'):
                raise RunnerError('reference unavailable')
            return measured(config, parameters, repeats)

        state, _ = self.execute(runner)
        self.assertEqual(state['trials'][-1]['status'], 'failed')
        self.assertEqual(self.confirmation(state)['verification']['verdict'], 'INVALID')
        self.assert_no_winner()

    def test_deadline_after_candidate_without_last_reference_is_invalid(self):
        clock = fixtures.Clock()
        clock.now = 1000

        def runner(config, parameters, folder, deadline, repeats):
            clock.now += 1
            if folder.name.endswith('confirmation'):
                clock.now = 1061
            return measured(config, parameters, repeats)

        state, hardware = self.execute(runner, clock=clock)
        self.assertEqual(hardware.call_count, 4)
        candidate = self.confirmation(state)
        self.assertEqual(candidate['verification']['verdict'], 'INVALID')
        self.assertIsNone(candidate['verification']['reference_trials'][-1])
        self.assert_no_winner()

    def test_customer_can_accept_improved_candidate_even_if_baseline_misses_slo(self):
        def runner(config, parameters, folder, deadline, repeats):
            return measured(config, parameters, repeats, latency=3000 if parameters['batch'] == 1 else 500)

        state, _ = self.execute(runner)
        references = [t for t in state['trials'] if t['phase'].startswith('reference_')]
        self.assertEqual(len(references), 2)
        self.assertTrue(all(not t['evaluation']['feasible'] for t in references))
        self.assertTrue(self.confirmation(state)['verification']['accepted'])
        self.assertTrue((self.output / 'best.json').exists())

    def test_later_runtime_change_revokes_an_already_exported_winner(self):
        self.fixture.source['verification']['mode'] = 'single'
        self.fixture.source['budget'] = {'finalists': 2}
        self.fixture.reload()
        changed = False
        confirmations = 0

        def runner(config, parameters, folder, deadline, repeats):
            nonlocal changed, confirmations
            if folder.name.endswith('confirmation'):
                confirmations += 1
                if confirmations == 2:
                    self.assertTrue((self.output / 'best.json').exists())
                    changed = True
            return measured(config, parameters, repeats)

        with self.assertRaises(WorkflowIntegrityError):
            self.execute(runner, runtime=lambda _config: {'fixture': 2 if changed else 1})
        state = self.state()
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(state['integrity_valid'])
        self.assertEqual(state['stop_reason'], 'frozen_inputs_changed')
        self.assert_no_winner()

    def test_frozen_data_mutation_during_measurement_invalidates_campaign(self):
        artifact = self.fixture.root / 'customer-data.jsonl'
        artifact.write_bytes(b'original customer workload\n')
        self.fixture.source['verification']['frozen_files'] = [
            {'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}]
        self.fixture.reload()

        def runner(config, parameters, folder, deadline, repeats):
            artifact.write_bytes(b'changed workload\n')
            return measured(config, parameters, repeats)

        with self.assertRaises(WorkflowIntegrityError):
            self.execute(runner)
        self.assertFalse(self.state()['integrity_valid'])
        self.assert_no_winner()

    def test_complete_trace_tail_deletion_is_detected_against_state_anchor(self):
        self.execute()
        path = self.output / 'events.jsonl'
        lines = path.read_bytes().splitlines(keepends=True)
        self.assertGreater(len(lines), 1)
        path.write_bytes(b''.join(lines[:-1]))
        # This is still a valid prefix, but it no longer reaches the checkpoint.
        self.assertTrue(read_trace(path))
        with self.assertRaisesRegex(ValueError, 'trace.*checkpoint'):
            self.execute(resume=True)

    def test_completed_state_edits_are_rejected_before_any_hardware_runs(self):
        state, _ = self.execute()
        path = self.output / 'workflow_state.json'
        for field in ('verification', 'evaluation'):
            with self.subTest(field=field):
                altered = copy.deepcopy(state)
                candidate = self.confirmation(altered)
                if field == 'verification':
                    candidate['verification']['accepted'] = False
                else:
                    candidate['evaluation']['objective'] = 999999
                path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    self.execute(resume=True)
                self.assertEqual(self.hardware.call_count, 0)

    def test_audit_append_failure_revokes_candidate_and_previous_winner(self):
        self.fixture.source['verification']['mode'] = 'single'
        self.fixture.source['budget'] = {'finalists': 2}
        self.fixture.reload()
        for fail_on in (1, 2):
            with self.subTest(fail_on_verification=fail_on):
                self.output = self.fixture.root / f'audit-failure-{fail_on}'
                confirmations = 0

                def append(output, event):
                    nonlocal confirmations
                    if event['type'] == 'verification_finished':
                        confirmations += 1
                        self.assertTrue(event['verification']['accepted'])
                        if confirmations == fail_on:
                            self.assertEqual((self.output / 'best.json').exists(), fail_on == 2)
                            raise OSError('audit journal write failed')
                    return append_trace(output, event)

                with patch('ascend_poc.workflow.append_trace', side_effect=append), \
                     self.assertRaises(WorkflowIntegrityError):
                    self.execute()
                state = self.state()
                self.assertFalse(state['integrity_valid'])
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['stop_reason'], 'audit_trace_invalid')
                self.assertEqual(confirmations, fail_on)
                self.assert_no_winner()

    def test_anytime_curve_publishes_incumbent_only_at_verification_time(self):
        clock = fixtures.Clock()
        clock.now = 1000

        def runner(config, parameters, folder, deadline, repeats):
            clock.now += 1
            return measured(config, parameters, repeats)

        state, _ = self.execute(runner, clock=clock)
        trace = read_trace(self.output / 'events.jsonl')
        events = [entry['event'] for entry in trace]
        types = [event['type'] for event in events]
        self.assertIn('proposal_selected', types)
        self.assertIn('research_finished', types)
        final = next(event for event in events if event['type'] == 'verification_finished')
        candidate = self.confirmation(state)
        after = next(t for t in state['trials'] if t['phase'] == 'reference_after')
        self.assertGreaterEqual(final['epoch'], after['ended_epoch'])
        self.assertGreater(candidate['verified_epoch'], candidate['ended_epoch'])
        with (self.output / 'anytime.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        times = [float(row['elapsed_s']) for row in rows]
        self.assertEqual(times, sorted(times))
        winners = [row for row in rows if row['best_confirmed_objective']]
        self.assertTrue(winners)
        self.assertEqual(winners[0]['trial_id'], candidate['id'])
        self.assertEqual(float(winners[0]['elapsed_s']), final['elapsed_s'])
        self.assertTrue(all(float(row['elapsed_s']) >= final['elapsed_s'] for row in winners))

    def test_resume_after_interrupted_last_reference_restarts_entire_bracket(self):
        interrupted = False

        def runner(config, parameters, folder, deadline, repeats):
            nonlocal interrupted
            if folder.name.endswith('reference_after') and not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            return measured(config, parameters, repeats)

        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner)
        self.assert_no_winner()
        state, hardware = self.execute(runner, resume=True)
        self.assertEqual(hardware.call_count, 3)
        self.assertTrue(self.confirmation(state)['verification']['accepted'])
        self.assertTrue((self.output / 'best.json').exists())
