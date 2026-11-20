import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ascend_poc.open_capabilities import (
    build_engine_argv, discover, parse_help, validate_engine, validate_env,
)
from ascend_poc.runner import RunnerError

HELP = '''usage: vllm serve [-h] [--max-num-seqs MAX_NUM_SEQS]
                  --usage-only IGNORED

options:
  -h, --help            Show help.
  --host HOST          Listener host.
  --port PORT          Listener port.
  --api-key API_KEY    API credential.
  --model MODEL        A model selected by the runtime.
  --served-model-name NAME [NAME ...]
                       API model name.
  --config CONFIG      Arbitrary configuration file.
  --trust-remote-code, --no-trust-remote-code
                       Run downloaded model code.
  --worker-cls CLS     Worker import path.
  --max-num-seqs MAX_NUM_SEQS
                       Maximum sequences per scheduling iteration.
  --block-size {16,32,64}
                       Block capacity.
  --tensor-parallel-size TP, -tp TP
                       Number of TP ranks.
  --enable-prefix-caching, --disable-prefix-caching
                       Cache matching token prefixes.
  --enable-chunked-prefill, --no-enable-chunked-prefill
                       Chunk large inputs.
  --enforce-eager       Skip graph capture.
  --additional-config ADDITIONAL_CONFIG
                       Ascend JSON configuration.
  --compilation-config JSON, -cc JSON
                       Compilation options.
  --speculative-config SPECULATIVE_CONFIG
                       Draft settings.
  --capture-sizes CAPTURE_SIZES [CAPTURE_SIZES ...]
                       Capture sizes.
  --optional-sizes [OPTIONAL_SIZES ...]
                       Optional sizes.
  --dtype {auto,float16,bfloat16}
                       Model dtype.
  --load-format LOAD_FORMAT
                       Weight format.
  --future-engine-knob FUTURE_ENGINE_KNOB
                       A new flag unknown to this controller release.
'''


class HelpParsingTests(unittest.TestCase):
    def test_discovers_future_parameter_without_hardcoded_catalog(self):
        caps = parse_help(HELP)
        self.assertIn('future-engine-knob', caps['flags'])
        self.assertNotIn('usage-only', caps['flags'])
        self.assertEqual(caps['source_hash'], hashlib.sha256(HELP.encode()).hexdigest())
        self.assertEqual(caps['flags']['block-size']['choices'], ['16', '32', '64'])

    def test_retains_real_boolean_alias_spelling(self):
        caps = parse_help(HELP)
        self.assertEqual(caps['flags']['enable-prefix-caching']['negative'], '--disable-prefix-caching')
        self.assertEqual(caps['flags']['enable-chunked-prefill']['negative'], '--no-enable-chunked-prefill')
        self.assertEqual(caps['flags']['enforce-eager']['kind'], 'boolean')
        self.assertEqual(caps['flags']['tensor-parallel-size']['kind'], 'value')

    def test_reports_operational_and_executable_exclusions(self):
        caps = parse_help(HELP)
        for name in ('help', 'host', 'port', 'api-key', 'model', 'config', 'worker-cls', 'trust-remote-code'):
            self.assertNotIn(name, caps['flags'])
            self.assertTrue(caps['excluded'][name]['reason'])

    def test_wrapped_alias_definition_is_not_a_second_option(self):
        caps = parse_help('  --enable-very-long-feature,\n                       --disable-very-long-feature\n                       A useful flag.\n')
        self.assertEqual(list(caps['flags']), ['enable-very-long-feature'])
        self.assertEqual(caps['flags']['enable-very-long-feature']['negative'], '--disable-very-long-feature')

    def test_separate_observed_boolean_definitions_pair(self):
        caps = parse_help('  --enable-cache  On.\n  --no-enable-cache  Off.\n')
        self.assertEqual(list(caps['flags']), ['enable-cache'])
        self.assertEqual(caps['flags']['enable-cache']['negative'], '--no-enable-cache')

    def test_short_alias_and_nargs_shapes_are_retained(self):
        caps = parse_help(HELP)
        self.assertIn('-tp', caps['flags']['tensor-parallel-size']['aliases'])
        self.assertEqual(caps['flags']['capture-sizes']['nargs'], '+')
        self.assertEqual(caps['flags']['optional-sizes']['nargs'], '*')
        self.assertEqual(caps['flags']['compilation-config']['encoding'], 'json')

    def test_empty_duplicate_and_oversized_help_rejected(self):
        for text in ('No engine help available', '  --foo FOO\n  --foo FOO\n', 'x' * (4 * 1024 * 1024 + 1)):
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                parse_help(text)


class EngineRenderingTests(unittest.TestCase):
    def setUp(self):
        self.caps = parse_help(HELP)

    def test_aliases_normalize_and_values_are_not_finite_search_choices(self):
        clean = validate_engine({'max_num_seqs': 777, '-tp': 4, 'future-engine-knob': 0.123}, self.caps)
        self.assertEqual(clean, {'max-num-seqs': 777, 'tensor-parallel-size': 4, 'future-engine-knob': 0.123})

    def test_duplicate_alias_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple aliases'):
            validate_engine({'tensor-parallel-size': 2, 'tp': 4}, self.caps)

    def test_unknown_and_protected_flags_rejected(self):
        for name in ('invented-speedup', 'host', 'api-key', 'model', 'trust-remote-code'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_engine({name: 1}, self.caps)

    def test_single_boolean_false_requires_actual_negative(self):
        with self.assertRaisesRegex(ValueError, 'no discovered negative'):
            validate_engine({'enforce-eager': False}, self.caps)
        self.assertEqual(build_engine_argv({'enforce-eager': None}, self.caps), [])
        self.assertEqual(build_engine_argv({'enforce-eager': True}, self.caps), ['--enforce-eager'])

    def test_real_negative_alias_and_inverted_alias_input(self):
        self.assertEqual(build_engine_argv({'enable-prefix-caching': False}, self.caps), ['--disable-prefix-caching'])
        self.assertEqual(validate_engine({'disable-prefix-caching': True}, self.caps), {'enable-prefix-caching': False})
        self.assertEqual(build_engine_argv({'no-enable-chunked-prefill': False}, self.caps), ['--enable-chunked-prefill'])

    def test_boolean_requires_bool(self):
        for value in (1, 'true', [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_engine({'enforce-eager': value}, self.caps)

    def test_choices_validate_values_and_preserve_unit_support_elsewhere(self):
        self.assertEqual(build_engine_argv({'block-size': 32, 'max-num-seqs': '1k'}, self.caps), ['--block-size=32', '--max-num-seqs=1k'])
        for value in (128, 'arbitrary', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_engine({'block-size': value}, self.caps)

    def test_json_configs_are_one_argv_item(self):
        options = {'additional-config': {'scheduler_config': {'enable_balance_scheduling': True}},
                   'compilation-config': [1, 2, 4], 'speculative-config': {'method': 'mtp', 'num_speculative_tokens': 3}}
        argv = build_engine_argv(options, self.caps)
        self.assertEqual(len(argv), 3)
        for arg in argv:
            name, value = arg.split('=', 1)
            self.assertEqual(json.loads(value), options[name[2:]])

    def test_multi_value_options_emit_bounded_separate_items(self):
        self.assertEqual(build_engine_argv({'capture-sizes': [1, 8, 32]}, self.caps), ['--capture-sizes', '1', '8', '32'])
        self.assertEqual(build_engine_argv({'optional-sizes': []}, self.caps), ['--optional-sizes'])
        with self.assertRaises(ValueError):
            validate_engine({'capture-sizes': []}, self.caps)
        with self.assertRaises(ValueError):
            validate_engine({'capture-sizes': [None]}, self.caps)

    def test_nargs_list_rejects_injected_flags_but_allows_numeric_negative(self):
        for value in ('--host', '-m', '-2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_engine({'capture-sizes': [value]}, self.caps)
        self.assertEqual(build_engine_argv({'capture-sizes': [-2, 4]}, self.caps), ['--capture-sizes', '-2', '4'])

    def test_scalar_leading_flags_and_shell_syntax_are_bound_data(self):
        value = '--model=another; $(touch /tmp/not-executed)'
        self.assertEqual(build_engine_argv({'future-engine-knob': value}, self.caps), ['--future-engine-knob=' + value])

    def test_placeholders_controls_and_nonfinite_values_rejected_recursively(self):
        for value in ('{env:SECRET}', '{param.x}', '{trial_dir}', '{repo_dir}', 'x\n', 'x\x00',
                      float('nan'), float('inf'), 10 ** 1000, {'nested': ['{env:SECRET}']}, {'x': '\x7f'}, {1: 'x'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_engine({'additional-config': value}, self.caps)

    def test_malformed_option_names_rejected(self):
        for name in ('max-num-seqs=10', 'max-num-seqs --model', 'max-num-seqs\n', 'additional-config.foo'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_engine({name: 1}, self.caps)

    def test_semantic_fake_paths_rejected_as_dict_or_json_text(self):
        values = [
            {'enable_force_eplb': True},
            {'rejection_sample_method': 'synthetic'},
            {'synthetic_acceptance_rates': [1, 1]},
            {'nested': {'synthetic_acceptance_length': 4}},
            {'trust_remote_code': True},
        ]
        for value in values:
            for form in (value, json.dumps(value)):
                with self.subTest(form=form), self.assertRaises(ValueError):
                    validate_engine({'additional-config': form}, self.caps)
        with self.assertRaises(ValueError):
            validate_engine({'load-format': 'dummy'}, self.caps)
        self.assertEqual(validate_engine({'additional-config': {'enable_force_eplb': False}}, self.caps),
                         {'additional-config': {'enable_force_eplb': False}})

    def test_bounds_and_null_restore_defaults(self):
        self.assertEqual(validate_engine({'max-num-seqs': None}, self.caps), {'max-num-seqs': None})
        self.assertEqual(build_engine_argv({'max-num-seqs': None}, self.caps), [])
        with self.assertRaisesRegex(ValueError, '128 KiB'):
            validate_engine({'future-engine-knob': 'a' * (128 * 1024)}, self.caps)
        value = 1
        for _ in range(20):
            value = [value]
        with self.assertRaises(ValueError):
            validate_engine({'additional-config': value}, self.caps)


class EnvironmentTests(unittest.TestCase):
    def test_registered_performance_env_values_normalize(self):
        self.assertEqual(validate_env({'OMP_NUM_THREADS': 4, 'VLLM_ASCEND_ENABLE_NZ': True}, ['OMP_NUM_THREADS', 'VLLM_ASCEND_ENABLE_NZ']),
                         {'OMP_NUM_THREADS': '4', 'VLLM_ASCEND_ENABLE_NZ': '1'})

    def test_protected_names_rejected_even_if_registered(self):
        for name in ('PATH', 'LD_PRELOAD', 'PYTHONPATH', 'OPENAI_API_KEY', 'HF_TOKEN', 'PASSWORD', 'HOME', 'HTTPS_PROXY', 'ASCEND_RT_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_env({name: 'x'}, [name])

    def test_unknown_name_invalid_value_and_placeholder_rejected(self):
        for values in ({'UNREGISTERED': 'x'}, {'OMP_NUM_THREADS': None}, {'OMP_NUM_THREADS': ['1']},
                       {'OMP_NUM_THREADS': '{env:SECRET}'}, {'OMP_NUM_THREADS': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_env(values, ['OMP_NUM_THREADS'])


class DiscoveryTests(unittest.TestCase):
    def test_help_file_relative_to_workdir_and_saved_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'help.txt').write_text(HELP)
            caps = discover({'help_file': 'help.txt'}, root / 'output', time.monotonic() + 5, root)
            self.assertEqual(caps['source'], 'help_file')
            self.assertEqual((root / 'output/capability-help.txt').read_text(), HELP)
            self.assertEqual(json.loads((root / 'output/capabilities.json').read_text()), caps)

    def test_real_argv_command_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            caps = discover({'command': [sys.executable, '-c', 'print(' + repr(HELP) + ')']}, root / 'out', time.monotonic() + 5, root)
            self.assertIn('future-engine-knob', caps['flags'])
            self.assertEqual(caps['source'], 'command')

    def test_discovery_does_not_pass_planner_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def fake_run(command, **kwargs):
                self.assertNotIn('POC_LLM_API_KEY', kwargs['env'])
                self.assertNotIn('HF_TOKEN', kwargs['env'])
                Path(kwargs['log_path']).write_text(HELP)
            with patch.dict(os.environ, {'POC_LLM_API_KEY': 'private', 'HF_TOKEN': 'private'}), patch('ascend_poc.open_capabilities.run_command', side_effect=fake_run):
                discover({'command': ['vllm', 'serve', '--help=all']}, root / 'out', time.monotonic() + 5, root)

    def test_discovery_strips_custom_credential_name_but_keeps_normal_env(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def fake_run(command, **kwargs):
                self.assertNotIn('FOO', kwargs['env'])
                self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '3')
                Path(kwargs['log_path']).write_text(HELP)
            with patch.dict(os.environ, {'FOO': 'custom-private-key', 'OMP_NUM_THREADS': '3'}), patch('ascend_poc.open_capabilities.run_command', side_effect=fake_run):
                discover({'command': ['vllm', 'serve', '--help=all'], 'env_names': ['OMP_NUM_THREADS']},
                         root / 'out', time.monotonic() + 5, root, exclude_env=('FOO',))

    def test_discovery_validates_env_registration_and_exact_exclusions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'help.txt').write_text(HELP)
            with self.assertRaises(ValueError):
                discover({'help_file': 'help.txt', 'env_names': ['ASCEND_RT_VISIBLE_DEVICES']},
                         root / 'out', time.monotonic() + 5, root)
            for excluded in ('FOO', ['INVALID=NAME'], [None]):
                with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                    discover({'help_file': 'help.txt'}, root / 'out', time.monotonic() + 5,
                             root, exclude_env=excluded)

    def test_ambiguous_discovery_config_and_expired_deadline_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for config in ({}, {'command': ['vllm'], 'help_file': 'a'}, {'command': 'vllm --help'}, {'command': ['vllm'], 'timeout_s': 0}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    discover(config, root / 'out', time.monotonic() + 5, root)
            with self.assertRaises(ValueError):
                discover({'command': ['vllm']}, root / 'out', time.monotonic() - 1, root)

    def test_failures_and_oversized_output_do_not_become_capabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(RunnerError):
                discover({'command': [sys.executable, '-c', 'raise SystemExit(2)']}, root / 'fail', time.monotonic() + 5, root)
            (root / 'huge.txt').write_bytes(b'x' * (4 * 1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                discover({'help_file': 'huge.txt'}, root / 'huge', time.monotonic() + 5, root)
            self.assertFalse((root / 'huge/capabilities.json').exists())


if __name__ == '__main__':
    unittest.main()
