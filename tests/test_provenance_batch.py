"""A shared integrity check saves reads without reusing old digests."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_poc.provenance import runtime_signature, runtime_signatures
from tests.test_tuner import contract


class BatchProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='poc-batch-provenance-')
        self.root = Path(self.temp.name)
        self.code_root = self.root / 'controller'
        self.code_root.mkdir()
        self.code_a = self.code_root / 'a.py'
        self.code_b = self.code_root / 'b.py'
        self.code_a.write_text('first = 1\n')
        self.code_b.write_text('second = 2\n')
        self.script = self.root / 'shared.py'
        self.script.write_text('print(1)\n')
        self.executable = self.root / 'tool'
        self.executable.write_text('#!/bin/sh\nexit 0\n')
        self.executable.chmod(0o700)
        self.config = contract(str(self.root))
        self.config['_repo_dir'] = str(self.root)
        self.config['server']['command'] = ['tool', 'shared.py']
        self.config['server']['env'] = {'PATH': str(self.root)}
        self.config['benchmark_env'] = {'PATH': str(self.root)}
        for scenario in self.config['scenarios']:
            scenario['command'] = ['tool', 'shared.py']
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.source = patch('ascend_poc.provenance.__file__', str(self.code_root / 'provenance.py'))
        self.source.start()

    def tearDown(self):
        self.source.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_batch_reads_shared_sources_commands_and_packages_once(self):
        original_read = Path.read_bytes
        counts = {}

        def tracked(path):
            counts[path] = counts.get(path, 0) + 1
            return original_read(path)

        with patch.object(Path, 'read_bytes', tracked), \
                patch('ascend_poc.provenance.importlib.metadata.version', return_value='fixture') as version:
            result = runtime_signatures({'one': self.config, 'two': copy.deepcopy(self.config)})
        self.assertEqual(result['one'], result['two'])
        self.assertEqual(set(counts), {self.code_a, self.code_b, self.script, self.executable})
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertEqual(version.call_count, 4)
        # Shared work must not make the returned mutable dictionaries aliases.
        result['one']['controller_code']['a.py'] = 'changed'
        result['one']['packages']['torch'] = 'changed'
        result['one']['command_files'][str(self.script)]['sha256'] = 'changed'
        self.assertNotEqual(result['one'], result['two'])
        self.assertNotEqual(result['two']['controller_code']['a.py'], 'changed')
        self.assertNotEqual(result['two']['packages']['torch'], 'changed')
        self.assertNotEqual(result['two']['command_files'][str(self.script)]['sha256'], 'changed')

    def test_single_signature_matches_batch_schema(self):
        expected = runtime_signature(self.config)
        self.assertEqual(runtime_signatures({'one': self.config})['one'], expected)
        self.assertEqual(expected['controller_code']['a.py'], hashlib.sha256(self.code_a.read_bytes()).hexdigest())

    def test_next_check_rereads_all_sources_and_detects_changed_contents(self):
        original_read = Path.read_bytes
        counts = {}

        def tracked(path):
            counts[path] = counts.get(path, 0) + 1
            return original_read(path)

        with patch.object(Path, 'read_bytes', tracked):
            before = runtime_signatures({'one': self.config, 'two': self.config})
            unchanged = runtime_signatures({'one': self.config, 'two': self.config})
        self.assertEqual(before, unchanged)
        self.assertTrue(all(count == 2 for count in counts.values()))
        old_stat = self.script.stat()
        self.script.write_text('print(2)\n')
        os.utime(self.script, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.code_a.write_text('first = 9\n')
        after = runtime_signatures({'one': self.config, 'two': self.config})
        self.assertNotEqual(before['one']['controller_code'], after['one']['controller_code'])
        self.assertNotEqual(before['one']['command_files'], after['one']['command_files'])

    def test_profile_environments_resolve_independent_command_targets(self):
        other_script = self.root / 'other.py'
        other_script.write_text('print(2)\n')
        one, two = copy.deepcopy(self.config), copy.deepcopy(self.config)
        one['server']['command'] = ['tool', '{env:SCRIPT}']
        two['server']['command'] = ['tool', '{env:SCRIPT}']
        one['server']['env']['SCRIPT'] = 'shared.py'
        two['server']['env']['SCRIPT'] = 'other.py'
        result = runtime_signatures({'one': one, 'two': two})
        self.assertNotIn(str(other_script), result['one']['command_files'])
        self.assertIn(str(other_script), result['two']['command_files'])

    def test_change_to_already_read_source_invalidates_whole_batch(self):
        original_read = Path.read_bytes

        def modified(path):
            content = original_read(path)
            if path == self.script:
                self.code_a.write_text('first = 9\n')
            return content

        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'changed'):
                runtime_signatures({'one': self.config})

    def test_source_inventory_change_invalidates_whole_batch(self):
        original_read = Path.read_bytes

        def modified(path):
            content = original_read(path)
            if path == self.script:
                (self.code_root / 'new.py').write_text('new = 1\n')
            return content

        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                runtime_signatures({'one': self.config})

    def test_symlink_retarget_after_read_invalidates_whole_batch(self):
        other_script = self.root / 'other.py'
        other_script.write_text('print(2)\n')
        link = self.root / 'link.py'
        link.symlink_to(self.script)
        self.config['server']['command'] = ['tool', 'link.py']
        original_read = Path.read_bytes

        def modified(path):
            content = original_read(path)
            if path == self.script:
                link.unlink()
                link.symlink_to(other_script)
            return content

        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'changed'):
                runtime_signatures({'one': self.config})

    def test_empty_batch_performs_no_reads(self):
        with patch.object(Path, 'read_bytes') as read:
            self.assertEqual(runtime_signatures({}), {})
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
