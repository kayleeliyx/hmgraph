"""CPU-only checks for the user's /data storage restriction."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code.hdepic_sft_qwen38 import storage


class StorageTests(unittest.TestCase):
    def test_outside_and_relative_paths_refused(self):
        for path in ('relative/path', '/tmp/new-cache', '/home/yaxuanli/new-cache'):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                storage.data_path(path)

    def test_real_data_paths_and_current_aliases(self):
        proof = storage.validate_storage()
        self.assertTrue(all(Path(p).is_relative_to('/data') for p in proof['environment'].values()))
        self.assertTrue(all(Path(p).is_relative_to('/data') for p in proof['legacy_aliases'].values()))
        self.assertEqual(tempfile.tempdir, os.environ['TMPDIR'])

    def test_missing_environment_refused(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'Source scripts'):
            storage.validate_storage()

    def test_external_cache_override_refused(self):
        with patch.dict(os.environ, {'TORCH_HOME': '/home/yaxuanli/.cache/torch'}):
            with self.assertRaisesRegex(RuntimeError, 'escapes /data'):
                storage.validate_storage()

    def test_missing_alias_refused_without_recreation(self):
        alias = Path('/tmp/hdepic-nonexistent-storage-test-alias')
        with patch.dict(storage.ALIASES, {alias: storage.STORAGE/'absent'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Missing relocated legacy alias'):
                storage.validate_storage()
        self.assertFalse(alias.exists())

    def test_space_block_and_no_outside_temp_fallback(self):
        used = type('Usage', (), {'free': storage.MIN_FREE_BYTES - 1})()
        with patch.object(storage.shutil, 'disk_usage', return_value=used):
            with self.assertRaisesRegex(RuntimeError, 'less than 3 GiB'):
                storage.validate_storage(require_space=True)
        with patch.object(storage.shutil, 'disk_usage', return_value=type('Usage', (), {'free': storage.MIN_FREE_BYTES})()):
            self.assertTrue(storage.validate_storage(require_space=True)['space_checked'])


if __name__ == '__main__':
    unittest.main()
