"""Open campaign integration: explicit constraints, recovery and evidence rules."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.deepwiki import DeepWikiError
from ascend_poc.open_demo import REQUEST, demo_candidate, demo_intent, make_runtime
from ascend_poc.open_mock import synthetic_metrics
from ascend_poc.open_runtime import validate_runtime
from ascend_poc.open_tuner import IntegrityError, run_autotune, report_autotune
from ascend_poc.runner import RunnerError


class ScriptedModel:
    def __init__(self, candidates=(), intent=None):
        self.candidates = list(copy.deepcopy(candidates))
        self.intent = copy.deepcopy(intent or demo_intent())
        self.payloads = []

    def ask(self, system, payload, remaining_s):
        self.payloads.append(copy.deepcopy(payload))
        stage = payload['stage']
        if stage == 'compile_intent':
            return copy.deepcopy(self.intent)
        if stage == 'diagnose':
            return {'observations': ['Measured fixture feedback is available.'],
                    'bottleneck': 'Test one new complete configuration.',
                    'research_question': 'How do cache, compilation and load affect this installed stack?',
                    'next_experiment': 'Check the next candidate against explicit customer constraints.'}
        if stage == 'propose':
            if not self.candidates:
                raise AssertionError('Unexpected extra proposal request')
            return {'candidate': self.candidates.pop(0), 'hypothesis': 'A different configuration may improve the score.',
                    'expected_effect': 'Measure the objective and every explicitly requested gate.',
                    'evidence_ids': payload['evidence_ids']}
        raise AssertionError('Unexpected model stage: ' + str(stage))


class Research:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def ask(self, repository, question, remaining_s=None):
        self.calls.append((repository, question))
        if self.fail:
            raise DeepWikiError('Synthetic research service unavailable')
        return {'answer': 'Synthetic evidence: test supported cache and compilation options.',
                'sources': ['https://deepwiki.com/vllm-project/vllm-ascend'],
                'repository': repository, 'transport': 'mcp', 'tool': 'ask_wiki_question', 'truncated': False}


class Measurements:
    def __init__(self, fail_calls=(), interrupt_calls=()):
        self.fail_calls = set(fail_calls)
        self.interrupt_calls = set(interrupt_calls)
        self.calls = []

    def __call__(self, config, parameters, directory, deadline, repetitions):
        self.calls.append({'config': copy.deepcopy(config), 'directory': Path(directory), 'repetitions': repetitions})
        index = len(self.calls)
        if index in self.interrupt_calls:
            (directory / 'server.log').write_text('Interrupted synthetic process.\n')
            raise KeyboardInterrupt()
        if index in self.fail_calls:
            (directory / 'server.log').write_text('Out of memory in synthetic initial configuration.\n')
            raise RunnerError('Server exited before it became healthy')
        argv = config['server']['command']
        flags = dict(a[2:].split('=', 1) for a in argv if a.startswith('--') and '=' in a)
        effective = {'prefix_cache': '--enable-prefix-caching' in argv,
                     'compilation': json.loads(flags.get('compilation-config', '{}')),
                     'tensor_parallel_size': int(flags.get('tensor-parallel-size', '1'))}
        rows = []
        for scenario in config['scenarios']:
            command = scenario['command']
            load = int(command[command.index('--concurrency') + 1])
            rows.append({'name': scenario['name'],
                         'metrics': [synthetic_metrics(effective, load) for _ in range(repetitions)]})
        (directory / 'fixture-raw.json').write_text(json.dumps(rows))
        return rows


class OpenTunerTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.raw = make_runtime(self.inputs, 19876, 19877)
        self.raw['budget']['max_experiments'] = 1
        self.output = self.root / 'output'
        self.patch = patch('ascend_poc.open_tuner.runtime_signature', return_value={'synthetic_runtime': 1})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def runtime(self):
        return validate_runtime(copy.deepcopy(self.raw), base_dir=self.inputs)

    def run_campaign(self, candidates=None, *, intent=None, request=REQUEST, resume=False,
                     plan_only=False, runner=None, researcher=None, output=None, model=None):
        model = model or ScriptedModel(candidates if candidates is not None else [demo_candidate(0)], intent)
        runner = runner or Measurements()
        researcher = researcher or Research()
        state = run_autotune(self.runtime(), request, output or self.output, workdir=Path.cwd(),
                             resume=resume, plan_only=plan_only, client=model,
                             researcher=researcher, trial_runner=runner)
        return state, model, runner, researcher

    def test_initial_startup_failure_can_establish_independently_verified_baseline(self):
        state, _, runner, _ = self.run_campaign(runner=Measurements(fail_calls={1}))
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['trials'][0]['failure_category'], 'out_of_memory')
        experiment = state['experiments'][0]
        self.assertEqual(experiment['verification_kind'], 'bootstrap')
        self.assertEqual(experiment['status'], 'baseline_established')
        self.assertTrue(experiment['verification']['accepted'])
        self.assertFalse(experiment['verification']['improved'])
        self.assertEqual(state['champion']['kind'], 'bootstrap')
        self.assertEqual(len(runner.calls), 5)
        self.assertTrue((self.output / 'champion.json').exists())
        report_autotune(self.output)

    def test_initial_oom_category_reaches_the_next_model_diagnosis(self):
        state, model, runner, _ = self.run_campaign(
            runner=Measurements(fail_calls={1}), researcher=Research(fail=True))
        diagnosis = next(payload for payload in model.payloads if payload['stage'] == 'diagnose')
        self.assertEqual(diagnosis['reference']['status'], 'failed')
        self.assertEqual(diagnosis['reference']['failure_category'], 'out_of_memory')
        self.assertEqual(len(runner.calls), 1)
        self.assertIsNone(state['champion'])

    def test_bootstrap_requires_all_three_independent_measurements_to_meet_gates(self):
        measured = Measurements(fail_calls={1})

        def unstable_bootstrap(config, parameters, directory, deadline, repetitions):
            rows = measured(config, parameters, directory, deadline, repetitions)
            if len(measured.calls) in {3, 5}:
                # Initial startup failed; the first candidate explores well.
                # Its independent before/after restarts violate the customer
                # gate even though the middle restart still looks successful.
                for row in rows:
                    for metrics in row['metrics']:
                        metrics['p99_tpot_ms'] = 100
                (directory / 'fixture-raw.json').write_text(json.dumps(rows))
            return rows

        state, _, _, _ = self.run_campaign(runner=unstable_bootstrap)
        experiment = state['experiments'][0]
        self.assertEqual(experiment['verification_kind'], 'bootstrap')
        self.assertTrue(experiment['feedback']['evaluation']['feasible'])
        self.assertFalse(experiment['verification']['accepted'])
        self.assertNotEqual(experiment['status'], 'baseline_established')
        self.assertIsNone(state['champion'])
        self.assertEqual(state['generation'], 0)
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())

    def test_feasible_lower_score_can_replace_an_infeasible_high_score_starting_point(self):
        self.raw['baseline']['workload']['concurrency'] = 64
        state, _, _, _ = self.run_campaign()
        initial = state['trials'][0]['evaluation']
        experiment = state['experiments'][0]
        self.assertFalse(initial['feasible'])
        self.assertGreater(initial['objective'], experiment['feedback']['evaluation']['objective'])
        self.assertEqual(experiment['verification_kind'], 'bootstrap')
        self.assertEqual(experiment['status'], 'baseline_established')
        self.assertTrue(experiment['verification']['accepted'])
        self.assertFalse(experiment['verification']['improved'])
        self.assertEqual(state['champion']['kind'], 'bootstrap')
        self.assertEqual(state['champion']['candidate'], demo_candidate(0))
        report_autotune(self.output)

    def test_interrupted_measurement_is_retained_and_resume_uses_new_trial_directory(self):
        self.raw['budget']['max_experiments'] = 2
        with self.assertRaises(KeyboardInterrupt):
            self.run_campaign(runner=Measurements(interrupt_calls={2}))
        saved = json.loads((self.output / 'state.json').read_text())
        self.assertEqual(saved['status'], 'interrupted')
        interrupted = [row for row in saved['trials'] if row['status'] == 'interrupted']
        self.assertEqual(len(interrupted), 1)
        old_id = interrupted[0]['id']
        old_deadline = saved['deadline_epoch']
        state, model, runner, _ = self.run_campaign([demo_candidate(2)], resume=True)
        self.assertEqual(state['deadline_epoch'], old_deadline)
        self.assertEqual(len({row['id'] for row in state['trials']}), len(state['trials']))
        self.assertNotIn(old_id, [call['directory'].name for call in runner.calls])
        self.assertNotIn('compile_intent', [p['stage'] for p in model.payloads])
        self.assertEqual(state['champion']['candidate'], demo_candidate(2))
        report_autotune(self.output)

    def test_plan_only_then_resume_keeps_deadline_and_previously_spent_api_budget(self):
        state, model, runner, researcher = self.run_campaign(plan_only=True)
        self.assertEqual(state['status'], 'planned')
        self.assertEqual(state['counters'], {'llm_calls': 1, 'research_calls': 0})
        self.assertEqual(runner.calls, [])
        self.assertEqual(researcher.calls, [])
        deadline = state['deadline_epoch']
        resumed, next_model, _, _ = self.run_campaign(resume=True)
        self.assertEqual(resumed['deadline_epoch'], deadline)
        self.assertEqual(resumed['counters']['llm_calls'], 3)
        self.assertEqual([p['stage'] for p in next_model.payloads], ['diagnose', 'propose'])
        self.assertEqual(resumed['generation'], 1)

    def test_explicit_user_parameter_lock_rejects_proposal_then_allows_correction(self):
        self.raw['budget']['max_experiments'] = 2
        intent = demo_intent()
        request = REQUEST + ' max-num-seqs must remain 32.'
        intent['constraints'].append({'scope': 'engine', 'key': 'max-num-seqs', 'operator': 'eq',
                                      'value': 32, 'source_quote': 'max-num-seqs must remain 32'})
        invalid = demo_candidate(0)
        invalid['engine']['max-num-seqs'] = 64
        state, model, runner, _ = self.run_campaign([invalid, demo_candidate(0)], intent=intent, request=request)
        self.assertEqual(state['experiments'][0]['status'], 'proposal_failed')
        self.assertIn('explicit', state['experiments'][0]['feedback']['reason'])
        self.assertEqual(state['experiments'][1]['status'], 'promoted')
        self.assertEqual(len(runner.calls), 5)
        # The failed proposal must appear in the next model diagnosis, rather
        # than being discarded and causing an identical blind retry.
        diagnoses = [p for p in model.payloads if p['stage'] == 'diagnose']
        self.assertEqual(diagnoses[1]['recent_experiments'][0]['status'], 'proposal_failed')

    def test_unresolved_customer_goal_never_starts_measurement_or_research(self):
        intent = demo_intent()
        intent['unresolved'] = ['Two customer score goals have no stated priority.']
        state, _, runner, researcher = self.run_campaign(intent=intent)
        self.assertEqual(state['status'], 'needs_input')
        self.assertEqual(runner.calls, [])
        self.assertEqual(researcher.calls, [])
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertIn('Two customer score goals', (self.output / 'report.md').read_text())

    def test_wiki_failure_stop_does_not_propose_or_measure_candidate(self):
        state, model, runner, _ = self.run_campaign(researcher=Research(fail=True))
        self.assertEqual(state['experiments'][0]['status'], 'research_unavailable')
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual([p['stage'] for p in model.payloads], ['compile_intent', 'diagnose'])
        self.assertIsNone(state['champion'])

    def test_wiki_failure_continue_is_explicit_and_never_fabricates_evidence(self):
        self.raw['research']['on_error'] = 'continue'
        state, model, _, _ = self.run_campaign(researcher=Research(fail=True))
        item = state['experiments'][0]
        self.assertEqual(item['research_status'], 'failed')
        self.assertEqual(item['evidence_ids'], [])
        self.assertEqual(item['status'], 'promoted')
        payload = next(p for p in model.payloads if p['stage'] == 'propose')
        self.assertIsNone(payload['evidence'])
        self.assertEqual(payload['evidence_ids'], [])

    def test_unknown_parameter_error_is_feedback_and_later_valid_configuration_runs(self):
        self.raw['budget']['max_experiments'] = 2
        unknown = demo_candidate(0)
        unknown['engine']['not-installed-on-this-engine'] = 4
        state, _, runner, _ = self.run_campaign([unknown, demo_candidate(0)])
        self.assertEqual(state['experiments'][0]['status'], 'proposal_failed')
        self.assertEqual(state['experiments'][1]['status'], 'promoted')
        self.assertEqual(len(runner.calls), 5)

    def test_changed_request_or_runtime_cannot_resume_saved_plan(self):
        state, _, _, _ = self.run_campaign(plan_only=True)
        for changed in ('request', 'runtime'):
            with self.subTest(changed=changed):
                request = REQUEST
                if changed == 'request':
                    request += ' Different goal.'
                else:
                    self.raw['budget']['max_experiments'] += 1
                with self.assertRaisesRegex(ValueError, 'same request, runtime'):
                    self.run_campaign(request=request, resume=True)

    def test_snapshot_and_export_tampering_revokes_champion_on_report(self):
        self.run_campaign()
        targets = ('runtime.json', 'intent.json', 'request.txt', 'capabilities.json',
                   'champion.json', 'champion-contract.json', 'best_observed.json')
        for index, target in enumerate(targets):
            with self.subTest(target=target):
                copied = self.root / f'tampered-{index}'
                shutil.copytree(self.output, copied)
                path = copied / target
                self.assertTrue(path.exists(), target)
                if path.suffix == '.json':
                    value = json.loads(path.read_text())
                    value['tampered'] = True
                    path.write_text(json.dumps(value))
                else:
                    path.write_text(path.read_text() + ' changed')
                with self.assertRaises((ValueError, RuntimeError)):
                    report_autotune(copied)
                self.assertFalse((copied / 'champion.json').exists())
                self.assertFalse((copied / 'champion-contract.json').exists())

    def test_raw_measurement_tampering_invalidates_saved_verdict(self):
        state, _, _, _ = self.run_campaign()
        identifier = state['experiments'][0]['verification_trials'][1]
        raw = self.output / 'trials' / identifier / 'fixture-raw.json'
        raw.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'raw measurement evidence'):
            report_autotune(self.output)
        self.assertFalse((self.output / 'champion.json').exists())

    def test_frozen_evaluator_change_during_measurement_stops_campaign_immediately(self):
        self.raw['budget']['max_experiments'] = 2
        frozen = self.inputs / 'customer-evaluation.json'
        frozen.write_text('{"fixed_test_set":1}\n')
        self.raw['verification']['frozen_files'] = [
            {'path': str(frozen), 'sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()}]
        measured = Measurements()

        def changing_runner(config, parameters, directory, deadline, repetitions):
            rows = measured(config, parameters, directory, deadline, repetitions)
            if len(measured.calls) == 2:
                # Changing the customer's fixed evaluator input is an integrity
                # failure, not an ordinary bad parameter choice to recover from.
                frozen.write_text('{"different_test_set":2}\n')
            return rows

        with self.assertRaises(IntegrityError):
            self.run_campaign([demo_candidate(0), demo_candidate(2)], runner=changing_runner)
        saved = json.loads((self.output / 'state.json').read_text())
        self.assertEqual(saved['status'], 'error')
        self.assertEqual(len(measured.calls), 2)
        self.assertEqual(len(saved['experiments']), 1)
        self.assertNotEqual(saved['experiments'][0]['status'], 'proposal_failed')
        self.assertFalse((self.output / 'champion.json').exists())
        self.assertFalse((self.output / 'champion-contract.json').exists())
