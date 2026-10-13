"""CPU-only blob verification and completion-gate tests using tiny fixtures."""
import hashlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code.hdepic_qwen38_baseline import weight_integrity as module

from code.hdepic_qwen38_baseline.weight_integrity import (
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
            link = root / 'model-00001-of-00018.safetensors'; link.symlink_to(blobs / key)
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


class CachedInventoryTests(unittest.TestCase):
    """Exercise the actual cached verify_weights branch with18 tiny HF blobs."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run, self.cache = root / 'run', root / 'cache'
        (self.run / 'model_metadata').mkdir(parents=True)
        (self.run / 'audit').mkdir()
        repository = self.cache / 'models--Qwen--Qwen3.8-27B'
        blobs = repository / 'blobs'; blobs.mkdir(parents=True)
        self.snapshot = repository / 'snapshots' / module.REVISION
        self.snapshot.mkdir(parents=True)
        self.names = [f'model-{index:05d}-of-00018.safetensors' for index in range(1, 19)]
        records = []
        for index, name in enumerate(self.names):
            header = json.dumps({f'tensor{index}': {'dtype': 'U8', 'shape': [8], 'data_offsets': [0, 8]}}).encode()
            header += b' ' * ((-len(header)) % 8)
            content = len(header).to_bytes(8, 'little') + header + bytes([index]) * 8
            blob = blobs / hashlib.sha256(content).hexdigest(); blob.write_bytes(content)
            source = self.snapshot / name; source.symlink_to(blob)
            record = verify_blob(source, blobs); record.update(inspect_safetensors_header(source))
            records.append(record)
        self.tensor_bytes = 18 * 8
        index = {'metadata': {'total_size': self.tensor_bytes},
                 'weight_map': {f'tensor{i}': name for i, name in enumerate(self.names)}}
        index_path = self.run / 'model_metadata/model.safetensors.index.json'
        index_path.write_text(json.dumps(index))
        identity = {'repo_id': module.REPO, 'revision': module.REVISION, 'weight_shards': self.names,
                    'metadata_sha256': {'model.safetensors.index.json': module.sha256(index_path)}}
        identity_path = self.run / 'model_identity.json'; identity_path.write_text(json.dumps(identity))
        self.config = {'repo_id': module.REPO, 'revision': module.REVISION,
                       'configuration_sha256': 'same-completed-run', 'model_identity_sha256': module.sha256(identity_path)}
        self.marker = {'generation_sha256': 'saved-generation-sha'}
        self.cached = {'passed': True, 'checkpoint_repo': module.REPO, 'checkpoint_revision': module.REVISION,
                       'checkpoint_snapshot': str(self.snapshot), 'generation_sha256': self.marker['generation_sha256'],
                       'configuration_sha256': self.config['configuration_sha256'],
                       'expected_shards': 18, 'checked_shards': 18, 'weight_shards': records,
                       'expected_tensor_bytes': self.tensor_bytes, 'actual_tensor_bytes': self.tensor_bytes,
                       'actual_file_bytes': sum(row['file_bytes'] for row in records),
                       'actual_header_and_prefix_bytes': sum(row['header_and_prefix_bytes'] for row in records),
                       'all_file_lengths_match_header_and_tensor_offsets': True}
        self.audit_path = self.run / 'audit/model_weight_integrity.json'
        self.save(self.cached)

    def save(self, value):
        self.audit_path.write_text(json.dumps(value))

    def verify_cached(self):
        with patch.object(module, 'require_completed_generation', return_value=(self.config, {}, self.marker)), \
             patch.object(module, 'WEIGHT_BYTES', self.tensor_bytes), \
             patch.object(module, 'verify_blob', side_effect=AssertionError('Cached valid proofs must not rehash payload')):
            return module.verify_weights(self.run, self.cache)

    def test_complete18_shard_cache_is_accepted_without_rehash_or_rewrite(self):
        before = self.audit_path.read_bytes(); before_mtime = self.audit_path.stat().st_mtime_ns
        self.assertEqual(self.verify_cached(), self.cached)
        self.assertEqual(self.audit_path.read_bytes(), before)
        self.assertEqual(self.audit_path.stat().st_mtime_ns, before_mtime)

    def test_incomplete_cached_inventory_is_rejected_even_when_listed_blob_is_valid(self):
        bad = copy.deepcopy(self.cached); bad['weight_shards'] = bad['weight_shards'][:1]
        self.save(bad)
        # The17 unlisted snapshot files can be absent while the single cached
        # row still has the exact size/mtime/hash that the old branch accepted.
        for name in self.names[1:]:
            (self.snapshot / name).unlink()
        with self.assertRaisesRegex(ValueError, 'inventory is incomplete'):
            self.verify_cached()

    def test_duplicate_or_substituted_names_cannot_cover_the_inventory(self):
        for changed_name in (self.names[0], 'unrelated.safetensors'):
            with self.subTest(changed_name=changed_name):
                bad = copy.deepcopy(self.cached); bad['weight_shards'][1]['filename'] = changed_name
                self.save(bad)
                with self.assertRaisesRegex(ValueError, 'filenames differ'):
                    self.verify_cached()

    def test_wrong_repo_snapshot_hash_or_total_is_rejected(self):
        changes = [
            ('repository', lambda value: value.update(checkpoint_repo='different/model')),
            ('snapshot', lambda value: value['weight_shards'][0].update(snapshot_path='/unrelated/shard.safetensors')),
            ('hash', lambda value: value['weight_shards'][0].update(actual_sha256='0' * 64)),
            ('totals', lambda value: value.update(actual_tensor_bytes=self.tensor_bytes - 1)),
        ]
        for description, change in changes:
            with self.subTest(description=description):
                bad = copy.deepcopy(self.cached); change(bad); self.save(bad)
                with self.assertRaises(ValueError):
                    self.verify_cached()

    def test_missing_listed_snapshot_file_is_rejected(self):
        (self.snapshot / self.names[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'missing or different pinned snapshot'):
            self.verify_cached()


if __name__ == '__main__':
    unittest.main()
