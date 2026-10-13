"""CPU-only blob verification and completion-gate tests using tiny fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from code.hdepic_qwen35_baseline.weight_integrity import (
    inspect_safetensors_header, require_completed_generation, verify_blob)


class WeightIntegrityTests(unittest.TestCase):
    def test_safetensors_payload_and_header_are_distinct_and_sum_to_file_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tiny.safetensors'
            header = json.dumps({
                'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
                'b': {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]},
            }).encode()
            header += b' ' * ((-len(header)) % 8)
            path.write_bytes(len(header).to_bytes(8, 'little') + header + b'12345678')
            result = inspect_safetensors_header(path)
            self.assertEqual(result['tensor_bytes'], 8)
            self.assertEqual(result['header_and_prefix_bytes'], 8 + len(header))
            self.assertEqual(result['file_bytes'], result['tensor_bytes'] + result['header_and_prefix_bytes'])
            self.assertEqual(result['tensor_count'], 2)
            self.assertTrue(result['file_length_matches_header_and_tensor_offsets'])

    def test_safetensors_trailing_bytes_and_offset_gaps_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'broken.safetensors'
            for offsets, payload, message in (([0, 4], b'12345', 'file length'),
                                               ([1, 4], b'1234', 'overlap or leave a gap')):
                with self.subTest(offsets=offsets):
                    header = json.dumps({'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': offsets}}).encode()
                    path.write_bytes(len(header).to_bytes(8, 'little') + header + payload)
                    with self.assertRaisesRegex(ValueError, message):
                        inspect_safetensors_header(path)

    def test_correct_sha256_blob_and_snapshot_symlink_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); blobs = root / 'blobs'; blobs.mkdir()
            content = b'pinned test weights'; key = hashlib.sha256(content).hexdigest()
            (blobs / key).write_bytes(content)
            link = root / 'model-00001-of-00014.safetensors'; link.symlink_to(blobs / key)
            result = verify_blob(link, blobs)
            self.assertTrue(result['passed'])
            self.assertEqual(result['actual_sha256'], key)
            self.assertEqual(result['bytes'], len(content))

    def test_corrupt_blob_returns_explicit_hash_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); blobs = root / 'blobs'; blobs.mkdir()
            key = hashlib.sha256(b'original').hexdigest()
            (blobs / key).write_bytes(b'changed')
            link = root / 'shard.safetensors'; link.symlink_to(blobs / key)
            self.assertFalse(verify_blob(link, blobs)['passed'])

    def test_incomplete_generation_blocks_before_weight_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'frozen_config.json').write_text(json.dumps({'configuration_sha256': 'fixture'}))
            with self.assertRaisesRegex(ValueError, 'Refuse weight hashing while generation is incomplete'):
                require_completed_generation(root)


if __name__ == '__main__':
    unittest.main()
