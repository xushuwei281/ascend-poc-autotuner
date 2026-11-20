import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ascend_poc.open_capabilities import parse_help
from ascend_poc.open_runtime import load_runtime, materialize, validate_candidate, validate_runtime
from ascend_poc.runner import run_trial
from ascend_poc.metrics import evaluate_trial


HELP = '''options:
  --max-num-seqs MAX_NUM_SEQS
                        Maximum sequences.
  --max-model-len MAX_MODEL_LEN
                        Context length.
  --tensor-parallel-size SIZE
                        Tensor parallel size.
  --pipeline-parallel-size SIZE
                        Pipeline parallel size.
  --data-parallel-size SIZE
                        Data parallel size.
  --enable-prefix-caching, --disable-prefix-caching
                        Enable or disable prefix caching.
  --compilation-config COMPILATION_CONFIG
                        JSON compilation configuration.
  --additional-config ADDITIONAL_CONFIG
                        JSON backend configuration.
  --speculative-config SPECULATIVE_CONFIG
                        JSON draft-model configuration.
  --host HOST
                        Runtime host.
  --port PORT
                        Runtime port.
'''


def fixture():
    return {
        'schema_version': 1, 'name': 'open runtime unit test',
        'hardware': {'device_type': 'Ascend test', 'device_count': 2, 'devices': ['0', '1']},
        'models': [{'id': 'bf16', 'path': 'model', 'family': 'TestModel', 'precision': 'bf16'},
                   {'id': 'int8', 'path': 'model-int8', 'family': 'TestModel', 'precision': 'int8'}],
        'server': {'command': ['vllm', 'serve', '{model}'],
                   'fixed_options': {'host': '127.0.0.1', 'port': 8123, 'served-model-name': 'poc'},
                   'env': {'ASCEND_RT_VISIBLE_DEVICES': '0,1'},
                   'health_url': 'http://127.0.0.1:8123/health'},
        'discovery': {'help_file': 'help.txt', 'env_names': ['HCCL_BUFFSIZE']},
        'baseline': {'model': 'bf16', 'engine': {'max-num-seqs': 32}, 'env': {},
                     'workload': {'concurrency': 16}},
        'workload_parameters': {'concurrency': {'type': 'integer', 'description': 'Load generator concurrency', 'minimum': 1}},
        'scenarios': [{'name': 'load', 'command': ['bench', '--concurrency', '{workload.concurrency}', '--output', '{trial_dir}/metrics.json'],
                       'result_file': '{trial_dir}/metrics.json'}],
        'metric_names': ['output_tokens_per_second', 'p99_ttft_ms', 'quality_score'],
        'llm': {'base_url': 'https://example.org/v1', 'model': 'planner'},
    }


def intent():
    return {'objective': {'metric': 'output_tokens_per_second', 'direction': 'maximize', 'scenario_aggregation': 'min'},
            'metric_requirements': {}, 'constraints': [], 'unresolved': []}


class OpenRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'model').mkdir()
        (self.root / 'model-int8').mkdir()
        (self.root / 'help.txt').write_text(HELP)
        self.caps = parse_help(HELP)
        self.runtime = validate_runtime(fixture(), self.root)

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self):
        return copy.deepcopy(self.runtime['baseline'])

    def test_defaults_never_add_customer_locks_or_quality_gates(self):
        result = materialize(self.runtime, self.candidate(), intent(), self.caps, self.root)
        self.assertEqual(result['requirements'], {})
        self.assertNotIn('search_space', result)
        self.assertNotIn('requirements', result['scenarios'][0])
        self.assertEqual(result['server']['command'][:3], ['vllm', 'serve', str(self.root / 'model')])
        self.assertEqual(result['budget']['repetitions'], 1)
        self.assertEqual(self.runtime['verification']['frozen_files'], [])

    def test_arbitrary_discovered_new_options_and_json_are_rendered(self):
        proposed = self.candidate()
        proposed['engine'] = {'max_model_len': 16384, 'enable-prefix-caching': True,
                              'compilation-config': {'mode': 3, 'cudagraph_capture_sizes': [1, 8]},
                              'max-num-seqs': None}
        result = materialize(self.runtime, proposed, intent(), self.caps, self.root)
        argv = result['server']['command']
        self.assertIn('--max-model-len=16384', argv)
        self.assertIn('--enable-prefix-caching', argv)
        encoded = next(x for x in argv if x.startswith('--compilation-config='))
        self.assertEqual(json.loads(encoded.split('=', 1)[1]), proposed['engine']['compilation-config'])
        self.assertFalse(any(x.startswith('--max-num-seqs') for x in argv))

    def test_unmentioned_precision_and_workload_can_change(self):
        proposed = self.candidate()
        proposed.update(model='int8', workload={'concurrency': 128})
        request = intent()
        request['constraints'] = [{'scope': 'model', 'key': 'family', 'operator': 'eq', 'value': 'TestModel'}]
        result = materialize(self.runtime, proposed, request, self.caps, self.root)
        self.assertEqual(result['context']['model']['precision'], 'int8')
        self.assertIn('128', result['scenarios'][0]['command'])
        self.assertEqual(result['requirements'], {})

    def test_nested_asset_references_resolve_to_real_argv_paths(self):
        draft = self.root / 'draft model with spaces'; draft.mkdir()
        self.runtime['models'].append({'id': 'draft', 'path': str(draft), 'family': 'Draft'})
        candidate = self.candidate()
        candidate['engine']['speculative-config'] = {
            'model': 'asset:draft', 'method': 'draft_model',
            'nested': {'assets': ['asset:int8', {'model': 'asset:bf16'}]},
        }
        unchanged = copy.deepcopy(candidate)
        config = materialize(self.runtime, candidate, intent(), self.caps, self.root)
        argv = config['server']['command'][3:]
        code = "import argparse,json; p=argparse.ArgumentParser(); p.add_argument('--speculative-config',type=json.loads); a,_=p.parse_known_args(); print(json.dumps(a.speculative_config))"
        process = subprocess.run([sys.executable, '-c', code, *argv], capture_output=True,
                                 text=True, check=True, timeout=3)
        parsed = json.loads(process.stdout)
        self.assertEqual(parsed['model'], str(draft))
        self.assertEqual(parsed['nested']['assets'][0], str(self.root / 'model-int8'))
        self.assertEqual(parsed['nested']['assets'][1]['model'], str(self.root / 'model'))
        self.assertEqual(candidate, unchanged)
        self.assertEqual(validate_candidate(self.runtime, candidate, intent(), self.caps)['engine'], candidate['engine'])

    def test_unknown_asset_reference_rejected_before_execution_including_json_string(self):
        for value in ({'model': 'asset:missing'}, {'inner': ['asset:missing']},
                      '{"model":"asset:missing"}', 'asset:'):
            candidate = self.candidate(); candidate['engine']['speculative-config'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'unknown local model asset'):
                validate_candidate(self.runtime, candidate, intent(), self.caps)

    def test_missing_nested_asset_path_does_not_download_or_mutate_candidate(self):
        self.runtime['models'].append({'id': 'missing-draft', 'path': 'not-prepared', 'family': 'Draft'})
        candidate = self.candidate()
        candidate['engine']['speculative-config'] = {'model': 'asset:missing-draft'}
        # Inventory validation and executable asset readiness are separate stages.
        validate_candidate(self.runtime, candidate, intent(), self.caps)
        with self.assertRaisesRegex(ValueError, 'local model asset is missing'):
            materialize(self.runtime, candidate, intent(), self.caps, self.root)
        self.assertEqual(candidate['engine']['speculative-config']['model'], 'asset:missing-draft')

    def test_asset_reference_in_json_string_is_resolved_without_touching_other_strings(self):
        candidate = self.candidate()
        candidate['engine']['speculative-config'] = '{"model":"asset:int8","note":"an asset:bf16 label"}'
        config = materialize(self.runtime, candidate, intent(), self.caps, self.root)
        encoded = next(arg for arg in config['server']['command'] if arg.startswith('--speculative-config='))
        parsed = json.loads(encoded.split('=', 1)[1])
        self.assertEqual(parsed, {'model': str(self.root / 'model-int8'), 'note': 'an asset:bf16 label'})

    def test_explicit_engine_model_and_workload_locks_are_enforced(self):
        candidates = []
        for scope, key, value in [('engine', 'max-num-seqs', 64), ('model', 'precision', 'int8'),
                                   ('workload', 'concurrency', 32)]:
            request = intent()
            request['constraints'] = [{'scope': scope, 'key': key, 'operator': 'eq', 'value': value}]
            candidates.append(request)
        for request in candidates:
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, 'Explicit request'):
                validate_candidate(self.runtime, self.candidate(), request, self.caps)

    def test_hardware_count_does_not_force_tensor_parallel_size(self):
        request = intent()
        request['constraints'] = [{'scope': 'hardware', 'key': 'device_count', 'operator': 'eq', 'value': 2}]
        candidate = self.candidate()
        candidate['engine'].update({'tensor-parallel-size': 1, 'data-parallel-size': 2})
        result = validate_candidate(self.runtime, candidate, request, self.caps)
        self.assertEqual(result['engine']['tensor-parallel-size'], 1)
        candidate['engine']['tensor-parallel-size'] = 2
        with self.assertRaisesRegex(ValueError, 'available device'):
            validate_candidate(self.runtime, candidate, request, self.caps)

    def test_parallel_layout_requires_positive_integers(self):
        for value in [0, -1, 1.5, True, '2']:
            candidate = self.candidate()
            candidate['engine']['tensor-parallel-size'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_candidate(self.runtime, candidate, intent(), self.caps)

    def test_environment_changes_are_isolated_from_benchmark(self):
        candidate = self.candidate()
        candidate['env'] = {'HCCL_BUFFSIZE': 512}
        request = intent()
        request['constraints'] = [{'scope': 'env', 'key': 'HCCL_BUFFSIZE', 'operator': 'eq', 'value': 512}]
        result = materialize(self.runtime, candidate, request, self.caps, self.root)
        self.assertEqual(result['server']['env']['HCCL_BUFFSIZE'], '512')
        self.assertEqual(result['benchmark_env'], {})
        self.assertEqual(self.runtime['server']['env'], {'ASCEND_RT_VISIBLE_DEVICES': '0,1'})

    def test_runtime_cannot_smuggle_in_customer_acceptance_requirements(self):
        for where in ('root', 'scenario'):
            value = fixture()
            if where == 'root': value['requirements'] = {'quality_score': {'min': .9}}
            else: value['scenarios'][0]['requirements'] = {'quality_score': {'min': .9}}
            with self.subTest(where=where), self.assertRaisesRegex(ValueError, 'unknown fields'):
                validate_runtime(value, self.root)

    def test_unknown_or_control_engine_flags_are_rejected(self):
        for flag in ('host', 'unknown-flag', 'port'):
            candidate = self.candidate()
            candidate['engine'][flag] = 3
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_candidate(self.runtime, candidate, intent(), self.caps)

    def test_workload_values_cannot_expand_runtime_secrets_or_inject_flags(self):
        value = fixture()
        value['workload_parameters']['concurrency'] = {'type': 'string', 'description': 'Test literal'}
        value['baseline']['workload']['concurrency'] = 'safe'
        runtime = validate_runtime(value, self.root)
        for attack in ('{env:SECRET}', '{trial_dir}', '{param.foo}', '--output=/tmp/stale', 'line\nbreak'):
            candidate = copy.deepcopy(runtime['baseline'])
            candidate['workload']['concurrency'] = attack
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                validate_candidate(runtime, candidate, intent(), self.caps)

    def test_workload_spec_requires_real_command_binding_and_fixed_outputs(self):
        variants = []
        value = fixture(); value['workload_parameters']['unbound'] = {'type': 'integer', 'description': 'Unused'}; variants.append(value)
        value = fixture(); value['scenarios'][0]['command'].append('{workload.unknown}'); variants.append(value)
        value = fixture(); value['scenarios'][0]['result_file'] = '{trial_dir}/{workload.concurrency}.json'; variants.append(value)
        value = fixture(); value['scenarios'][0]['command'].append('{param.old}'); variants.append(value)
        value = fixture(); value['scenarios'][0]['command'][0] = '{workload.concurrency}'; variants.append(value)
        value = fixture(); value['scenarios'][0]['command'] = ['python', '-c', '{workload.concurrency}']; variants.append(value)
        value = fixture(); value['scenarios'][0]['telemetry_file'] = '{trial_dir}/metrics.json'; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime(value, self.root)

    def test_candidate_workload_is_complete_typed_and_bounded(self):
        for workload in ({}, {'concurrency': 0}, {'concurrency': True}, {'concurrency': 1, 'extra': 2}):
            candidate = self.candidate(); candidate['workload'] = workload
            with self.subTest(workload=workload), self.assertRaises(ValueError):
                validate_candidate(self.runtime, candidate, intent(), self.caps)

    def test_metric_and_quality_results_only_use_explicit_request_gates(self):
        request = intent(); request['metric_requirements'] = {'quality_score': {'min': .95}}
        result = materialize(self.runtime, self.candidate(), request, self.caps, self.root)
        evaluation = evaluate_trial(result, [{'name': 'load', 'metrics': [{'output_tokens_per_second': 100}]}], 1)
        self.assertFalse(evaluation['feasible'])
        self.assertIn('quality_score', evaluation['failures'][0])
        request['metric_requirements'] = {'unknown': {'min': .1}}
        with self.assertRaisesRegex(ValueError, 'not exposed'):
            materialize(self.runtime, self.candidate(), request, self.caps, self.root)

    def test_custom_benchmark_metric_names_are_not_limited_to_builtin_scores(self):
        value = fixture(); value['metric_names'].append('customer.score-v2')
        runtime = validate_runtime(value, self.root)
        request = intent(); request['objective']['metric'] = 'customer.score-v2'
        result = materialize(runtime, self.candidate(), request, self.caps, self.root)
        self.assertEqual(result['objective']['metric'], 'customer.score-v2')

    def test_model_assets_must_be_declared_and_available_without_download(self):
        candidate = self.candidate(); candidate['model'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'available local'):
            validate_candidate(self.runtime, candidate, intent(), self.caps)
        self.runtime['models'][0]['path'] = 'missing-model'
        with self.assertRaisesRegex(ValueError, 'missing'):
            materialize(self.runtime, self.candidate(), intent(), self.caps, self.root)
        self.runtime['models'][0]['path'] = '{env:OPEN_TEST_MODEL}'
        with patch.dict(os.environ, {'OPEN_TEST_MODEL': str(self.root / 'model')}):
            result = materialize(self.runtime, self.candidate(), intent(), self.caps, self.root)
        self.assertEqual(result['server']['command'][2], str(self.root / 'model'))
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'requires environment'):
            materialize(self.runtime, self.candidate(), intent(), self.caps, self.root)

    def test_secrets_are_not_measurement_wiring(self):
        variants = []
        value = fixture(); value['server']['env']['POC_LLM_API_KEY'] = 'no'; variants.append(value)
        value = fixture(); value['discovery']['env_names'].append('POC_LLM_API_KEY'); variants.append(value)
        value = fixture(); value['models'][0]['path'] = '{env:POC_LLM_API_KEY}'; variants.append(value)
        value = fixture(); value['server']['fixed_options']['api-key'] = '{env:POC_LLM_API_KEY}'; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'planner API key'):
                validate_runtime(value, self.root)

    def test_invalid_shapes_duplicate_json_nonfinite_are_rejected(self):
        variants = []
        value = fixture(); value['schema_version'] = True; variants.append(value)
        value = fixture(); value['models'].append(copy.deepcopy(value['models'][0])); variants.append(value)
        value = fixture(); value['hardware']['devices'] = ['0', '0']; variants.append(value)
        value = fixture(); value['discovery']['command'] = ['vllm', 'serve', '--help=all']; variants.append(value)
        value = fixture(); value['server']['command'] = ['vllm', 'serve', '--model={model}']; variants.append(value)
        value = fixture(); value['server']['fixed_options']['max-model-len'] = 1024; variants.append(value)
        value = fixture(); value['baseline']['engine']['test'] = float('inf'); variants.append(value)
        value = fixture(); value['scenarios'][0]['quality_command'] = ['quality']; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime(value, self.root)
        path = self.root / 'runtime.json'
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_runtime(path)

    def test_load_resolves_help_relative_to_runtime_and_preserves_input(self):
        value = fixture(); original = copy.deepcopy(value)
        path = self.root / 'runtime.json'; path.write_text(json.dumps(value))
        result = load_runtime(path)
        self.assertEqual(result['discovery']['help_file'], str(self.root / 'help.txt'))
        self.assertEqual(result['_source_path'], str(path))
        self.assertEqual(value, original)
        reordered = fixture(); reordered['hardware']['devices'].reverse()
        self.assertEqual(result['_resource_id'], validate_runtime(reordered, self.root)['_resource_id'])

    def test_frozen_files_validate_shape_but_are_not_read_by_config_validation(self):
        value = fixture()
        value['verification'] = {'frozen_files': [{'path': 'not-yet-mounted', 'sha256': '0' * 64}]}
        self.assertEqual(validate_runtime(value, self.root)['verification']['frozen_files'], value['verification']['frozen_files'])
        value['verification']['frozen_files'][0]['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_runtime(value, self.root)

    def test_unresolved_intent_blocks_execution(self):
        request = intent(); request['unresolved'] = ['Metric priority is ambiguous']
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            validate_candidate(self.runtime, self.candidate(), request, self.caps)

    def test_real_runner_uses_materialized_workload_and_isolated_environment(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        server_code = '''import argparse,http.server
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--port',type=int);a,_=p.parse_known_args()
class H(http.server.BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'ok')
 def log_message(self,*args):pass
http.server.HTTPServer(('127.0.0.1',a.port),H).serve_forever()
'''
        bench_code = "import os,json,sys; json.dump({'output_tokens_per_second':int(sys.argv[2]),'env_isolated':int('HCCL_BUFFSIZE' not in os.environ)},open(sys.argv[1],'w'))"
        value = fixture()
        value['server'].update(command=[sys.executable, '-c', server_code, '{model}'],
                               fixed_options={'port': port}, health_url=f'http://127.0.0.1:{port}/health',
                               startup_timeout_s=2, shutdown_timeout_s=.2)
        value['scenarios'][0]['command'] = [sys.executable, '-c', bench_code, '{trial_dir}/metrics.json', '{workload.concurrency}']
        value['scenarios'][0]['timeout_s'] = 2
        runtime = validate_runtime(value, self.root)
        candidate = copy.deepcopy(runtime['baseline']); candidate['env'] = {'HCCL_BUFFSIZE': '512'}
        candidate['workload']['concurrency'] = 32
        config = materialize(runtime, candidate, intent(), self.caps, self.root)
        # urllib can have cached its proxy opener in earlier integration tests.
        # Clearing all environment variables removes NO_PROXY and makes the
        # existing opener route localhost through a proxy. Isolate the measured
        # variable while preserving explicitly local connectivity instead.
        with patch.dict(os.environ, {'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost'}):
            os.environ.pop('HCCL_BUFFSIZE', None)
            results = run_trial(config, {}, self.root / 'trial', time.monotonic() + 10, 1)
        self.assertEqual(results.status, 'completed')
        self.assertEqual(results[0]['metrics'][0]['output_tokens_per_second'], 32)
        self.assertEqual(results[0]['metrics'][0]['env_isolated'], 1)
        self.assertTrue(evaluate_trial(config, results, 1)['feasible'])


if __name__ == '__main__':
    unittest.main()
