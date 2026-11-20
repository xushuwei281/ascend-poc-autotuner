"""Measurement-scoped hints remain useful without relaxing campaign acceptance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_poc.rsi_memory import (MEASUREMENT_PROTOCOL, memory_namespace,
                                   recall_memory, remember_experiment)
from ascend_poc.strategy import workflow_hash
from tests.test_rsi_memory import record
from tests import test_strategy as strategy_fixtures


class MemoryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = strategy_fixtures.StrategyTests().load()
        self.runtime = {'baseline': {'python': '3.12', 'packages': {'vllm': 'fixed'},
                                    'controller_code': {'metrics.py': 'digest'},
                                    'referenced_env_hashes': {'MODEL_PATH': 'fixed'}}}
        self.frozen = [{'sha256': 'a' * 64, 'size': 123}]
        self.scope = memory_namespace(self.spec, self.runtime, self.frozen)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.json'

    def test_search_names_planners_and_limits_do_not_isolate_new_campaigns(self):
        changed = copy.deepcopy(self.spec)
        changed['name'] = 'another campaign name'
        changed['contract'] = 'another-contract-file.json'
        changed['budget'].update(max_trials=100, max_wall_time_s=99999, finalists=10,
                                 early_stop=True, confirmation_reserve=True,
                                 cost_safety_factor=1.5, estimated_trial_s=100)
        changed['research'].update(endpoint='https://different.example/mcp', max_calls=99,
                                   timeout_s=20, transport='a2a')
        changed['verification']['confirmation_profiles'] = ['eager']
        changed['_contract'].update(name='renamed contract', description='presentation only')
        changed['_contract']['llm'].update(model='new-planner', base_url='https://other.example/v1',
                                           timeout_s=99)
        for profile in changed['profiles']:
            profile['description'] = 'new presentation'
        for config in changed['_profiles'].values():
            config.update(name='renamed', description='new text')
            config['llm'].update(model='new-planner', timeout_s=99)
            config['budget'].update(max_trials=100, max_wall_time_s=99999, finalists=10)
        self.assertNotEqual(workflow_hash(self.spec), workflow_hash(changed))
        self.assertEqual(self.scope, memory_namespace(changed, self.runtime, self.frozen))

    def test_sampling_changes_still_isolate_historical_evidence(self):
        for key in ('repetitions', 'final_repetitions'):
            changed = copy.deepcopy(self.spec)
            changed['budget'][key] += 1
            with self.subTest(key=key):
                self.assertNotEqual(self.scope, memory_namespace(changed, self.runtime, self.frozen))

    def test_acceptance_model_workload_quality_and_cache_changes_are_scoped(self):
        mutations = [
            lambda s: s['_contract']['requirements'].update(p99_ttft_ms={'max': 1}),
            lambda s: s['_contract']['objective'].update(direction='minimize'),
            lambda s: s['_contract']['context'].update(model='different-model'),
            lambda s: s['_contract']['scenarios'][0].update(name='other-scenario'),
            lambda s: s['_contract']['scenarios'][0].update(command=['python', 'different-workload.py']),
            lambda s: s['_contract']['scenarios'][0].update(quality_command=['python', 'quality.py']),
            lambda s: s['_contract']['scenarios'][0].update(before_each_command=['python', 'flush-cache.py']),
            lambda s: s['_contract']['benchmark_env'].update(CACHE_POLICY='cold'),
            lambda s: s['_profiles']['eager']['server']['env'].update(PRECISION='other'),
            lambda s: s['_profiles']['eager']['server']['command'].append('--different'),
            lambda s: s['_profiles']['eager']['search_space']['batch'].append(123),
            lambda s: s['_profiles']['eager']['constraints'].append({'when': {'graph': True}, 'require': {'batch': [16]}}),
            lambda s: s['profiles'][1].update(requires_quality=True),
        ]
        for index, change in enumerate(mutations):
            changed = copy.deepcopy(self.spec)
            change(changed)
            with self.subTest(index=index):
                self.assertNotEqual(self.scope, memory_namespace(changed, self.runtime, self.frozen))

    def test_verification_protocol_and_runtime_are_not_pruned(self):
        for name, value in (('mode', 'aba_ci'), ('max_baseline_drift', .01),
                            ('min_relative_improvement', .2), ('drift_multiplier', 4),
                            ('ci_blocks', 5), ('ci_confidence', .99), ('ci_max_comparisons', 8)):
            changed = copy.deepcopy(self.spec)
            changed['verification'][name] = value
            with self.subTest(name=name):
                self.assertNotEqual(self.scope, memory_namespace(changed, self.runtime, self.frozen))
        runtime = copy.deepcopy(self.runtime)
        runtime['baseline']['referenced_env_hashes']['MODEL_PATH'] = 'changed'
        self.assertNotEqual(self.scope, memory_namespace(self.spec, runtime, self.frozen))
        self.assertNotEqual(self.scope, memory_namespace(self.spec, self.runtime, []))

    def test_scope_is_versioned_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.spec)
        self.assertEqual(self.scope, memory_namespace(self.spec, self.runtime, self.frozen))
        self.assertEqual(self.spec, before)
        with patch('ascend_poc.rsi_memory.MEASUREMENT_PROTOCOL', 'measurement-scope-next'):
            self.assertNotEqual(self.scope, memory_namespace(self.spec, self.runtime, self.frozen))
        with self.assertRaisesRegex(ValueError, 'normalized'):
            memory_namespace({}, self.runtime, self.frozen)

    def test_ci_confirmed_record_preserves_origin_without_reusing_acceptance(self):
        item = dict(record(mode='aba_ci'), source_workflow_hash=workflow_hash(self.spec),
                    measurement_protocol=MEASUREMENT_PROTOCOL)
        remember_experiment(self.path, self.scope, item)
        found = recall_memory(self.path, self.scope)
        self.assertEqual(found['records'][0]['measurement_protocol'], MEASUREMENT_PROTOCOL)
        self.assertEqual(found['records'][0]['evidence_level'], 'independently_confirmed')
        self.assertEqual(found['priors'][0]['improved_count'], 1)
        self.assertEqual(found['scope'], 'same_fingerprint_hints_only')
        self.assertEqual(found['marginals_scope'], 'descriptive_not_causal')
        self.assertEqual(recall_memory(self.path, 'b' * 64)['records'], [])

    def test_origin_metadata_is_bounded_and_validation_remains_strict(self):
        for extra in ({'source_workflow_hash': 'bad'}, {'source_workflow_hash': []},
                      {'measurement_protocol': 'contains\ncommand'},
                      {'measurement_protocol': 'x' * 161},
                      {'measurement_protocol': {'command': 'bad'}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                remember_experiment(self.path, self.scope, dict(record(), **extra))
        invalid = record(mode='aba_ci')
        invalid['verification']['valid'] = False
        with self.assertRaises(ValueError):
            remember_experiment(self.path, self.scope, invalid)

    def test_typed_marginals_distinguish_profile_and_scalar_types(self):
        for i, scalar in enumerate((True, 1, 1.0, '1')):
            item = record(i)
            item['parameters'] = {'knob': scalar}
            remember_experiment(self.path, self.scope, item)
        other = record(4, profile='eager')
        other['parameters'] = {'knob': True}
        remember_experiment(self.path, self.scope, other)
        cells = recall_memory(self.path, self.scope)['marginals']
        self.assertEqual(len(cells), 5)
        baseline = [c for c in cells if c['profile_id'] == 'baseline']
        self.assertEqual({c['value_type'] for c in baseline}, {'bool', 'int', 'float', 'str'})
        self.assertTrue(all(c['observed_count'] == 1 for c in cells))
        self.assertTrue(all(c['confirmed_count'] == 1 for c in cells))

    def test_marginals_count_all_retained_cases_and_distinguish_early_failure(self):
        remember_experiment(self.path, self.scope, record(0, mode='aba_ci'))
        failed = record(1, accepted=False, improved=False)
        failed['verification'].update(verdict='INVALID', valid=False)
        failed.update(observed_feasible=False, violated_metrics=['p99_ttft_ms'])
        remember_experiment(self.path, self.scope, failed)
        remember_experiment(self.path, self.scope, record(2, improved=False))
        found = recall_memory(self.path, self.scope, limit=1)
        self.assertEqual(len(found['records']), 1)
        cell = next(c for c in found['marginals'] if c['parameter'] == 'batch')
        self.assertEqual(cell['observed_count'], 3)
        self.assertEqual(cell['observed_infeasible_count'], 1)
        self.assertEqual(cell['confirmed_count'], 2)
        self.assertEqual(cell['improved_count'], 1)

    def test_marginals_limit_and_byte_budget_are_explicit(self):
        for i in range(30):
            item = record(i)
            item['parameters'] = {'knob': str(i) + 'x' * 990}
            remember_experiment(self.path, self.scope, item)
        found = recall_memory(self.path, self.scope)
        self.assertLessEqual(len(json.dumps(found['marginals'], separators=(',', ':'))), 24000)
        self.assertTrue(found['marginals_truncated'])
        self.assertTrue(found['marginals'][0]['value'].startswith('29'))
        with patch('ascend_poc.rsi_memory._MAX_MARGINALS', 3):
            limited = recall_memory(self.path, self.scope)
            self.assertEqual(len(limited['marginals']), 3)
            self.assertTrue(limited['marginals_truncated'])
        with self.assertRaises(ValueError):
            remember_experiment(self.path, self.scope, dict(record(31), parameters={'knob': float('nan')}))


if __name__ == '__main__':
    unittest.main()
