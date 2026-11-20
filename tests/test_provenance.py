"""Runtime identity follows the same environment scopes as real commands."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_poc.provenance import runtime_signature
from ascend_poc.tuner import tune
from tests.test_tuner import contract, measured


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poc-provenance-')
        self.root = Path(self.temp.name)
        self.config = contract(str(self.root))
        self.config['_repo_dir'] = str(self.root)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def write(self, name, text='first'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path.resolve()

    def signature(self):
        return runtime_signature(self.config)

    def test_self_referential_server_template_hashes_original_input(self):
        self.config['server']['env'] = {'ASCEND_RT_VISIBLE_DEVICES': '{env:ASCEND_RT_VISIBLE_DEVICES}'}
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        first = self.signature()
        self.assertEqual(first['referenced_env_hashes']['ASCEND_RT_VISIBLE_DEVICES'],
                         hashlib.sha256(b'0,1').hexdigest())
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '2,3'
        self.assertNotEqual(first, self.signature())
        self.assertNotIn('"0,1"', json.dumps(first))

    def test_changed_device_selection_prevents_completed_experiment_resume(self):
        self.config['server']['env'] = {'ASCEND_RT_VISIBLE_DEVICES': '{env:ASCEND_RT_VISIBLE_DEVICES}'}
        self.config['budget']['max_trials'] = 1
        output = self.root / 'run'
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'

        def runner(config, candidate, folder, deadline, repeats):
            return measured(config, candidate, repeats)

        with patch('ascend_poc.tuner.run_trial', side_effect=runner), redirect_stdout(io.StringIO()):
            tune(self.config, output, planner_name='random')
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '2,3'
        with patch('ascend_poc.tuner.run_trial') as hardware:
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                tune(self.config, output, planner_name='random', resume=True)
            hardware.assert_not_called()

    def test_server_and_benchmark_command_files_use_independent_envs(self):
        server_script = self.write('server.py')
        benchmark_script = self.write('benchmark.py')
        self.config['server']['env'] = {'SCRIPT': 'server.py'}
        self.config['benchmark_env'] = {'SCRIPT': 'benchmark.py'}
        self.config['server']['command'] = ['python3', '{env:SCRIPT}']
        self.config['scenarios'][0]['command'] = ['python3', '{env:SCRIPT}']
        first = self.signature()
        self.assertIn(str(server_script), first['command_files'])
        self.assertIn(str(benchmark_script), first['command_files'])
        original_stat = benchmark_script.stat()
        benchmark_script.write_text('later', encoding='utf-8')
        os.utime(benchmark_script, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertNotEqual(first, self.signature())

    def test_benchmark_override_tracks_its_inherited_source(self):
        first_script = self.write('first.py')
        second_script = self.write('second.py')
        self.config['benchmark_env'] = {'SCRIPT': '{env:SOURCE_SCRIPT}'}
        self.config['scenarios'][0]['command'] = ['python3', '{env:SCRIPT}']
        os.environ['SOURCE_SCRIPT'] = 'first.py'
        first = self.signature()
        self.assertIn(str(first_script), first['command_files'])
        os.environ['SOURCE_SCRIPT'] = 'second.py'
        second = self.signature()
        self.assertIn(str(second_script), second['command_files'])
        self.assertNotEqual(first['referenced_env_hashes'], second['referenced_env_hashes'])

    def test_overrides_do_not_expand_against_each_other(self):
        original = self.write('original.py')
        self.write('override.py')
        self.config['server']['env'] = {'SCRIPT': 'override.py', 'SIDE': '{env:SCRIPT}'}
        self.config['server']['command'] = ['python3', '{env:SIDE}']
        os.environ['SCRIPT'] = 'original.py'
        signature = self.signature()
        self.assertIn(str(original), signature['command_files'])
        self.assertNotIn(str(self.root / 'override.py'), signature['command_files'])

    def test_planner_credential_is_never_fingerprinted(self):
        key = self.config['llm']['api_key_env']
        self.config['context'] = {'invalid_credential_reference': '{env:' + key + '}'}
        os.environ[key] = 'DO_NOT_PERSIST'
        first = self.signature()
        os.environ[key] = 'ROTATED_CREDENTIAL'
        self.assertEqual(first, self.signature())
        self.assertNotIn(key, first['referenced_env_hashes'])
        self.assertNotIn('DO_NOT_PERSIST', json.dumps(first))
        self.assertNotIn(hashlib.sha256(b'DO_NOT_PERSIST').hexdigest(), json.dumps(first))

    def test_command_path_resolution_uses_scoped_relative_path(self):
        server_tool = self.write('server-bin/tool', '#!/bin/sh\nexit 0\n')
        bench_tool = self.write('benchmark-bin/tool', '#!/bin/sh\nexit 0\n')
        server_tool.chmod(0o700)
        bench_tool.chmod(0o700)
        self.config['server']['env'] = {'PATH': 'server-bin'}
        self.config['benchmark_env'] = {'PATH': 'benchmark-bin'}
        self.config['server']['command'] = ['tool']
        self.config['scenarios'][0]['command'] = ['tool']
        signature = self.signature()
        self.assertIn(str(server_tool), signature['command_files'])
        self.assertIn(str(bench_tool), signature['command_files'])

    def test_legacy_contracts_keep_shared_server_environment(self):
        script = self.write('legacy.py')
        self.config.pop('benchmark_env', None)
        self.config['server']['env'] = {'SCRIPT': 'legacy.py'}
        self.config['scenarios'][0]['command'] = ['python3', '{env:SCRIPT}']
        self.assertIn(str(script), self.signature()['command_files'])


if __name__ == '__main__':
    unittest.main()
