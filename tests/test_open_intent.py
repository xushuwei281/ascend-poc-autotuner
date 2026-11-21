"""Explicit customer constraints stay separate from available starting recipes."""
import copy
import json
import unittest

from ascend_poc.open_intent import check_candidate, compile_intent, validate_intent


TEXT = ('设备为两张 A2，模型使用 demo 系列。最大化 output_tokens_per_second；'
        'p99_ttft_ms 不超过 2000。其余不限制。')


def runtime():
    return {'metric_names': ['output_tokens_per_second', 'p99_ttft_ms', 'quality_score'],
            'hardware': {'device_type': 'A2', 'device_count': 2, 'devices': ['0', '1']},
            'models': [{'id': 'demo-bf16', 'path': '/PRIVATE/model', 'family': 'demo', 'precision': 'bf16'},
                       {'id': 'demo-w8', 'path': '/PRIVATE/quantized', 'family': 'demo', 'precision': 'w8a8'},
                       {'id': 'other', 'path': '/PRIVATE/other', 'family': 'other', 'precision': 'bf16'}],
            'baseline': {'model': 'demo-bf16', 'engine': {'tensor-parallel-size': 2},
                         'env': {}, 'workload': {'output_len': 256}},
            'workload_parameters': {'output_len': {'type': 'integer', 'minimum': 1,
                                                  'description': 'Output token count'}},
            'discovery': {'env_names': ['HCCL_BUFFSIZE', 'SOME_FEATURE']},
            'requirements': {'quality_score': {'min': 0.999}},
            'llm': {'api_key': 'PRIVATE-KEY'}, 'command': ['PRIVATE-COMMAND']}


def intent():
    return {'schema_version': 1, 'summary': '在明确条件内提高吞吐。',
            'objective': {'metric': 'output_tokens_per_second', 'direction': 'maximize', 'scenario_aggregation': 'min'},
            'metric_requirements': {'p99_ttft_ms': {'max': 2000}},
            'constraints': [
                {'scope': 'hardware', 'key': 'device_count', 'operator': 'eq', 'value': 2, 'source_quote': '两张 A2'},
                {'scope': 'model', 'key': 'family', 'operator': 'eq', 'value': 'demo', 'source_quote': '模型使用 demo 系列'},
            ],
            'requirement_evidence': {'p99_ttft_ms': 'p99_ttft_ms 不超过 2000'},
            'objective_evidence': '最大化 output_tokens_per_second',
            'inferred': ['多个场景按最小吞吐聚合。'], 'unresolved': []}


def candidate():
    return {'model': 'demo-w8', 'engine': {'tensor-parallel-size': 1, 'max-model-len': 4096,
            'speculative-config': {'method': 'ngram', 'num_speculative_tokens': 3}},
            'env': {}, 'workload': {'output_len': 128}}


class IntentTests(unittest.TestCase):
    def test_explicit_only_leaves_precision_tp_workload_and_quality_free(self):
        before = intent()
        result = validate_intent(before, TEXT, runtime())
        self.assertEqual(result['metric_requirements'], {'p99_ttft_ms': {'max': 2000}})
        self.assertEqual(result['constraints'], before['constraints'])
        self.assertEqual(check_candidate(result, candidate(), runtime()), [])
        self.assertIsNot(result, before)
        self.assertNotIn('quality_score', result['metric_requirements'])

    def test_compilation_context_excludes_paths_commands_keys_and_template_locks(self):
        class Client:
            def ask(self, system, payload, remaining_s):
                self.system, self.payload, self.remaining_s = system, payload, remaining_s
                return intent()
        client = Client()
        result = compile_intent(TEXT, runtime(), client, 23)
        self.assertEqual(result, intent())
        self.assertEqual(client.payload['stage'], 'compile_intent')
        self.assertEqual(client.payload['request'], TEXT)
        self.assertEqual(client.remaining_s, 23)
        encoded = json.dumps(client.payload)
        for secret in ('PRIVATE', 'baseline', 'requirements', 'path', 'command'):
            self.assertNotIn(secret, encoded)
        self.assertIn('Two available cards does NOT imply tensor-parallel-size=2', client.system)

    def test_missing_or_invented_evidence_is_rejected(self):
        cases = []
        value = intent(); value['objective_evidence'] = 'maximize latency'; cases.append(value)
        value = intent(); value['requirement_evidence'] = {}; cases.append(value)
        value = intent(); value['requirement_evidence']['quality_score'] = '两张 A2'; cases.append(value)
        value = intent(); value['constraints'][0]['source_quote'] = '四张 A2'; cases.append(value)
        value = intent(); value['constraints'][0]['source_quote'] = ' '; cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_intent(value, TEXT, runtime())

    def test_compiler_can_map_customer_concurrency_to_actual_adapter_field(self):
        resources = runtime()
        resources['workload_parameters'] = {
            'num_clients': {'type': 'integer', 'minimum': 1,
                            'description': 'Number of concurrent benchmark clients'}}
        caps = {'flags': {'max-num-seqs': {'kind': 'value', 'metavar': 'N',
                                          'help': 'Maximum concurrent engine sequences'}},
                'excluded': {'port': {'reason': 'controller connection'}}}
        text = TEXT + ' 并发固定为 32。'

        class Client:
            def ask(self, system, payload, remaining_s):
                self.payload = payload
                result = intent()
                key = next(name for name, spec in payload['workload_parameters'].items()
                           if 'concurrent benchmark clients' in spec['description'])
                result['constraints'].append(dict(scope='workload', key=key, operator='eq',
                                                  value=32, source_quote='并发固定为 32'))
                return result

        client = Client()
        result = compile_intent(text, resources, client, 10, caps=caps)
        self.assertEqual(result['unresolved'], [])
        self.assertEqual(result['constraints'][-1], dict(scope='workload', key='num_clients',
                         operator='eq', value=32, source_quote='并发固定为 32'))
        self.assertEqual(client.payload['workload_parameters'], resources['workload_parameters'])
        self.assertEqual(client.payload['environment_names'], ['HCCL_BUFFSIZE', 'SOME_FEATURE'])
        self.assertEqual(client.payload['engine_capabilities']['max-num-seqs']['metavar'], 'N')
        self.assertNotIn('port', client.payload['engine_capabilities'])
        self.assertNotIn('baseline', client.payload)
        self.assertNotIn('PRIVATE', json.dumps(client.payload))
        self.assertEqual(result['metric_requirements'], intent()['metric_requirements'])

    def test_unconnected_parameter_locks_are_unresolved_before_candidate_search(self):
        resources = runtime()
        resources['workload_parameters'] = {'num_clients': {'type': 'integer', 'minimum': 1}}
        caps = {'flags': {'max-num-seqs': {'kind': 'value'}}}
        text = TEXT + ' 并发固定为 32；任务队列必须为 1；批大小固定为 32。'
        for scope, key, number, quote in [
            ('workload', 'concurrency', 32, '并发固定为 32'),
            ('env', 'TASK_QUEUE_ENABLE', 1, '任务队列必须为 1'),
            ('engine', 'max-batch-size', 32, '批大小固定为 32'),
        ]:
            value = intent()
            condition = dict(scope=scope, key=key, operator='eq', value=number, source_quote=quote)
            value['constraints'].append(condition)
            with self.subTest(scope=scope):
                result = validate_intent(value, text, resources, caps=caps)
                self.assertEqual(result['constraints'][-1], condition)
                self.assertTrue(any(f'{scope}.{key}' in issue for issue in result['unresolved']))
                self.assertTrue(any('unresolved' in failure for failure in
                                    check_candidate(result, candidate(), resources)))
                self.assertEqual(value['unresolved'], [])

    def test_available_catalog_does_not_create_customer_locks(self):
        resources = runtime()
        caps = {'flags': {'max-num-seqs': {'kind': 'value'}, 'dtype': {'kind': 'value'}}}
        result = validate_intent(intent(), TEXT, resources, caps=caps)
        self.assertEqual(result, intent())

    def test_unknown_connected_metric_blocks_execution_instead_of_inventing_adapter(self):
        value = intent(); value['objective']['metric'] = 'new_score'
        result = validate_intent(value, TEXT, runtime())
        self.assertTrue(any('new_score' in item for item in result['unresolved']))
        self.assertIn('unresolved', check_candidate(result, candidate(), runtime())[0])

    def test_missing_hardware_or_model_resource_is_unresolved(self):
        resources = runtime(); resources['hardware']['device_count'] = 1
        result = validate_intent(intent(), TEXT, resources)
        self.assertTrue(any('device_count' in item for item in result['unresolved']))
        resources = runtime(); resources['models'] = [resources['models'][-1]]
        result = validate_intent(intent(), TEXT, resources)
        self.assertTrue(any('model asset' in item for item in result['unresolved']))

    def test_model_family_allows_variants_but_rejects_other_models_or_fictional_paths(self):
        normalized = validate_intent(intent(), TEXT, runtime())
        value = candidate(); value['model'] = 'other'
        self.assertTrue(any('model.family' in item for item in check_candidate(normalized, value, runtime())))
        value['model'] = '/new/invented/path'
        self.assertTrue(any('available runtime asset' in item for item in check_candidate(normalized, value, runtime())))

    def test_explicit_precision_blocks_quantized_variant(self):
        text = TEXT + ' 必须 bf16。'
        value = intent()
        value['constraints'].append({'scope': 'model', 'key': 'precision', 'operator': 'eq', 'value': 'bf16', 'source_quote': '必须 bf16'})
        normalized = validate_intent(value, text, runtime())
        self.assertTrue(any('model.precision' in item for item in check_candidate(normalized, candidate(), runtime())))

    def test_explicit_engine_workload_and_env_bounds_are_checked(self):
        text = TEXT + ' 批量至少 32，输出最多 256，HCCL 参数至少 2。'
        value = intent()
        value['constraints'] += [
            {'scope': 'engine', 'key': 'max-num-seqs', 'operator': 'min', 'value': 32, 'source_quote': '批量至少 32'},
            {'scope': 'workload', 'key': 'output_len', 'operator': 'max', 'value': 256, 'source_quote': '输出最多 256'},
            {'scope': 'env', 'key': 'HCCL_BUFFSIZE', 'operator': 'min', 'value': 2, 'source_quote': 'HCCL 参数至少 2'},
        ]
        normalized = validate_intent(value, text, runtime())
        proposal = candidate(); proposal['engine']['max-num-seqs'] = 64; proposal['env']['HCCL_BUFFSIZE'] = '4'
        self.assertEqual(check_candidate(normalized, proposal, runtime()), [])
        proposal['workload']['output_len'] = 512
        self.assertTrue(any('workload.output_len' in item for item in check_candidate(normalized, proposal, runtime())))
        proposal['env']['HCCL_BUFFSIZE'] = 'NaN'
        self.assertTrue(any('env.HCCL_BUFFSIZE' in item for item in check_candidate(normalized, proposal, runtime())))

    def test_missing_explicit_option_is_not_assumed_from_engine_defaults(self):
        value = intent()
        value['constraints'].append({'scope': 'engine', 'key': 'max-num-seqs', 'operator': 'eq', 'value': 32, 'source_quote': '其余不限制'})
        normalized = validate_intent(value, TEXT, runtime())
        # Grounding validates the quoted substring, not its semantic accuracy.
        # A full candidate must state a constrained option before it can pass.
        self.assertTrue(any('engine.max-num-seqs' in item for item in check_candidate(normalized, candidate(), runtime())))

    def test_env_equality_uses_actual_rendered_string_semantics(self):
        value = intent()
        value['constraints'] += [
            dict(scope='env', key='HCCL_BUFFSIZE', operator='eq', value=256, source_quote='其余不限制'),
            dict(scope='env', key='SOME_FEATURE', operator='in', value=[True, 'auto'], source_quote='其余不限制'),
        ]
        normalized = validate_intent(value, TEXT, runtime())
        proposal = candidate(); proposal['env'] = {'HCCL_BUFFSIZE': '256', 'SOME_FEATURE': '1'}
        self.assertEqual(check_candidate(normalized, proposal, runtime()), [])
        proposal['env']['SOME_FEATURE'] = '0'
        self.assertTrue(check_candidate(normalized, proposal, runtime()))
        value['constraints'][-1]['value'] = [None]
        with self.assertRaises(ValueError): validate_intent(value, TEXT, runtime())

    def test_json_equality_and_boolean_numeric_distinction(self):
        value = intent()
        value['constraints'].append({'scope': 'engine', 'key': 'speculative-config', 'operator': 'eq',
                                    'value': {'method': 'ngram', 'num_speculative_tokens': 3}, 'source_quote': '其余不限制'})
        normalized = validate_intent(value, TEXT, runtime())
        self.assertEqual(check_candidate(normalized, candidate(), runtime()), [])
        proposal = candidate(); proposal['engine']['speculative-config']['num_speculative_tokens'] = True
        self.assertTrue(check_candidate(normalized, proposal, runtime()))

    def test_multiple_allowed_values_and_bounds_intersect(self):
        value = intent()
        value['constraints'] += [
            {'scope': 'engine', 'key': 'max-num-seqs', 'operator': 'in', 'value': [16, 32, 64], 'source_quote': '其余不限制'},
            {'scope': 'engine', 'key': 'max-num-seqs', 'operator': 'min', 'value': 32, 'source_quote': '其余不限制'},
        ]
        normalized = validate_intent(value, TEXT, runtime())
        proposal = candidate(); proposal['engine']['max-num-seqs'] = 32
        self.assertEqual(check_candidate(normalized, proposal, runtime()), [])
        proposal['engine']['max-num-seqs'] = 16
        self.assertTrue(check_candidate(normalized, proposal, runtime()))

    def test_contradictory_constraints_are_rejected(self):
        for first, second in [(('min', 64), ('max', 32)), (('eq', 16), ('eq', 32)),
                              (('in', [16, 32]), ('min', 64)), (('eq', True), ('eq', 1))]:
            value = intent()
            value['constraints'] += [dict(scope='engine', key='max-num-seqs', operator=op, value=v, source_quote='其余不限制')
                                     for op, v in (first, second)]
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, 'Contradictory'):
                validate_intent(value, TEXT, runtime())

    def test_schema_rejects_arbitrary_fields_nonfinite_and_bad_constraint_keys(self):
        cases = []
        value = intent(); value['commands'] = ['new']; cases.append(value)
        value = intent(); value['schema_version'] = True; cases.append(value)
        value = intent(); value['metric_requirements']['p99_ttft_ms']['max'] = float('nan'); cases.append(value)
        value = intent(); value['constraints'][0]['operator'] = 'run'; cases.append(value)
        value = intent(); value['constraints'][0]['value'] = True; cases.append(value)
        value = intent(); value['constraints'].append(dict(scope='engine', key='--max-num-seqs', operator='eq', value=32, source_quote='其余不限制')); cases.append(value)
        value = intent(); value['constraints'].append(dict(scope='env', key='mixedCase', operator='eq', value='x', source_quote='其余不限制')); cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_intent(value, TEXT, runtime())

    def test_unresolved_ambiguity_is_preserved_and_blocking(self):
        value = intent(); value['unresolved'] = ['吞吐与延迟没有给出优先级。']
        normalized = validate_intent(value, TEXT, runtime())
        self.assertEqual(normalized['unresolved'], value['unresolved'])
        self.assertTrue(check_candidate(normalized, candidate(), runtime()))

    def test_runtime_must_declare_real_metric_names(self):
        for names in (None, [], ['x', 'x'], ['invalid space'], [1]):
            resources = runtime(); resources['metric_names'] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_intent(intent(), TEXT, resources)

    def test_malformed_candidate_does_not_pass_when_no_parameter_constraints(self):
        normalized = validate_intent(intent(), TEXT, runtime())
        for proposal in (None, {}, {'model': 'demo-bf16', 'engine': [], 'env': {}, 'workload': {}}):
            with self.subTest(proposal=proposal):
                self.assertTrue(check_candidate(normalized, proposal, runtime()))


if __name__ == '__main__':
    unittest.main()
