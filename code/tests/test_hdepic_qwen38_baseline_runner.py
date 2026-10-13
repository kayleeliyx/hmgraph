"""Standard-library tests for prediction identity and pre-allocation gates."""
import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from code.hdepic_qwen38_baseline import runner
from code.hdepic_qwen38_baseline.data import canonical_bytes


class RunnerIdentityAndGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.config = {'configuration_sha256': 'frozen-configuration'}
        self.records = [
            {'id': f'P07-video:explain:action{index}', 'row_index': index, 'prompt_sha256': f'prompt-{index}'}
            for index in range(2)
        ]

    def prediction(self, index=0):
        record = self.records[index]
        return {'id': record['id'], 'row_index': record['row_index'], 'task': 'explain',
                'status': 'completed', 'model_repo': runner.REPO, 'model_revision': runner.REVISION,
                'prompt_sha256': record['prompt_sha256'],
                'configuration_sha256': self.config['configuration_sha256'],
                'prediction': '<why>saved response</why>', 'generated_token_ids': [1, 2], 'generated_tokens': 2}

    def write_rows(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b''.join(canonical_bytes(value) + b'\n' for value in values))

    def test_native_loading_info_sets_and_shapes_are_json_safe_without_mutating_source(self):
        native = {
            'loading_info': {
                'missing_keys': set(),
                'unexpected_keys': {'mtp.layers.0.mlp.experts.1.up_proj.weight', 'mtp.fc.weight'},
                'mismatched_keys': [('some.weight', (2, 3), (2, 4))],
                'nested': [{'ignored_keys': frozenset({'mtp.z', 'mtp.a'})}],
                'error_msgs': [],
            },
            'device_map': {'model.visual': 0, 'model.language_model.layers.39': 1},
            'parameter_count': 35000000000,
        }
        converted = runner.json_safe(native)
        restored = json.loads(json.dumps(converted, allow_nan=False, sort_keys=True))
        self.assertEqual(restored['loading_info']['missing_keys'], [])
        self.assertEqual(restored['loading_info']['unexpected_keys'],
                         ['mtp.fc.weight', 'mtp.layers.0.mlp.experts.1.up_proj.weight'])
        self.assertEqual(restored['loading_info']['mismatched_keys'], [['some.weight', [2, 3], [2, 4]]])
        self.assertEqual(restored['loading_info']['nested'], [{'ignored_keys': ['mtp.a', 'mtp.z']}])
        self.assertEqual(restored['device_map'], native['device_map'])
        self.assertIsInstance(native['loading_info']['missing_keys'], set)
        self.assertIsInstance(native['loading_info']['nested'][0]['ignored_keys'], frozenset)
        self.assertEqual(native['loading_info']['mismatched_keys'], [('some.weight', (2, 3), (2, 4))])
        # Same diagnostics with a different set construction order must produce
        # identical serialized evidence and hence the same provenance digest.
        reordered = dict(native)
        reordered['loading_info'] = dict(native['loading_info'])
        reordered['loading_info']['unexpected_keys'] = set(reversed(sorted(native['loading_info']['unexpected_keys'])))
        self.assertEqual(canonical_bytes(runner.json_safe(reordered)), canonical_bytes(converted))

    def test_valid_completed_prediction_cache_is_reused_exactly(self):
        path = self.run / 'generation.jsonl'; prediction = self.prediction()
        self.write_rows(path, [prediction])
        self.assertEqual(runner.load_cached(path, self.records, self.config), {prediction['id']: prediction})

    def test_configuration_revision_and_prompt_changes_reject_cache(self):
        for field in ('configuration_sha256', 'model_revision', 'prompt_sha256'):
            with self.subTest(field=field):
                prediction = self.prediction(); prediction[field] = 'changed'
                path = self.run / f'{field}.jsonl'; self.write_rows(path, [prediction])
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    runner.load_cached(path, self.records, self.config)

    def test_duplicate_or_unknown_ids_reject_cache(self):
        first = self.prediction()
        unknown = self.prediction(); unknown['id'] = 'P07-video:explain:unapproved'
        for name, values in (('duplicate', [first, first]), ('unknown', [unknown])):
            with self.subTest(name=name):
                path = self.run / f'{name}.jsonl'; self.write_rows(path, values)
                with self.assertRaisesRegex(ValueError, 'Unknown/duplicate'):
                    runner.load_cached(path, self.records, self.config)

    def test_changed_model_repo_or_original_row_index_reject_cache(self):
        for field, value in (('model_repo', 'different/model'), ('row_index', 999)):
            with self.subTest(field=field):
                prediction = self.prediction(); prediction[field] = value
                path = self.run / f'{field}.jsonl'; self.write_rows(path, [prediction])
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    runner.load_cached(path, self.records, self.config)

    def test_missing_cache_returns_empty_without_creating_output(self):
        path = self.run / 'missing.jsonl'
        self.assertEqual(runner.load_cached(path, self.records, self.config), {})
        self.assertFalse(path.exists())

    def prepare_answering_rows(self):
        self.write_rows(self.run / 'inputs/validation_explain.prompts.jsonl', self.records)

    def deny_model_imports(self, real_import):
        def guarded(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.') or name == 'transformers' or name.startswith('transformers.'):
                raise AssertionError(f'Model/GPU module imported before resource gate: {name}')
            return real_import(name, *args, **kwargs)
        return guarded

    def test_busy_gpu_gate_prevents_download_and_any_model_gpu_import(self):
        self.prepare_answering_rows()
        resources = types.ModuleType('code.hdepic_qwen38_baseline.resources')
        resources.require_idle_pair = Mock(side_effect=ValueError('Allocated GPUs are occupied'))
        resources.require_disk_space = Mock()
        before = dict(os.environ)
        with patch.dict(sys.modules, {resources.__name__: resources}), \
             patch.object(runner, 'freeze', return_value=self.config), \
             patch.object(runner, 'fetch_weights') as weights, \
             patch('builtins.__import__', side_effect=self.deny_model_imports(builtins.__import__)):
            with self.assertRaisesRegex(ValueError, 'occupied'):
                runner.run_inference(self.run)
        resources.require_idle_pair.assert_called_once_with(self.run)
        resources.require_disk_space.assert_not_called()
        weights.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        self.assertFalse((self.run / 'predictions/generation.jsonl').exists())

    def test_insufficient_disk_gate_prevents_download_and_model_gpu_import(self):
        self.prepare_answering_rows()
        resources = types.ModuleType('code.hdepic_qwen38_baseline.resources')
        resources.require_idle_pair = Mock(return_value={'cuda_visible_devices': 'authorized-uuids'})
        resources.require_disk_space = Mock(side_effect=ValueError('Insufficient disk for pinned weights'))
        before = dict(os.environ)
        with patch.dict(sys.modules, {resources.__name__: resources}), \
             patch.object(runner, 'freeze', return_value=self.config), \
             patch.object(runner, 'fetch_weights') as weights, \
             patch('builtins.__import__', side_effect=self.deny_model_imports(builtins.__import__)):
            with self.assertRaisesRegex(ValueError, 'Insufficient disk'):
                runner.run_inference(self.run)
        resources.require_idle_pair.assert_called_once_with(self.run)
        resources.require_disk_space.assert_called_once()
        weights.assert_not_called()
        self.assertEqual(dict(os.environ), before)

    def test_matching_completed_run_skips_resource_probes_and_model_download(self):
        self.records = [{'id': f'P07-video:explain:action{index}', 'row_index': index,
                         'prompt_sha256': f'prompt-{index}'} for index in range(228)]
        self.prepare_answering_rows()
        predictions = [self.prediction(index) for index in range(228)]
        path = self.run / 'predictions/generation.jsonl'; self.write_rows(path, predictions)
        marker = {'passed': True, 'count': 228, 'configuration_sha256': self.config['configuration_sha256'],
                  'generation_sha256': runner.sha256(path)}
        (self.run / 'predictions/completed.json').write_text(json.dumps(marker))
        resources = types.ModuleType('code.hdepic_qwen38_baseline.resources')
        resources.require_idle_pair = Mock(side_effect=AssertionError('Completed cache must not probe GPUs'))
        resources.require_disk_space = Mock(side_effect=AssertionError('Completed cache must not download'))
        with patch.dict(sys.modules, {resources.__name__: resources}), \
             patch.object(runner, 'freeze', return_value=self.config), \
             patch.object(runner, 'fetch_weights') as weights, \
             patch('builtins.__import__', side_effect=self.deny_model_imports(builtins.__import__)):
            self.assertEqual(runner.run_inference(self.run), marker)
        resources.require_idle_pair.assert_not_called()
        resources.require_disk_space.assert_not_called()
        weights.assert_not_called()

    def test_failed_or_wrong_count_completion_marker_is_rejected_before_model_load(self):
        self.records = [{'id': f'P07-video:explain:action{index}', 'row_index': index,
                         'prompt_sha256': f'prompt-{index}'} for index in range(228)]
        self.prepare_answering_rows()
        path = self.run / 'predictions/generation.jsonl'
        self.write_rows(path, [self.prediction(index) for index in range(228)])
        for passed, count in ((False, 228), (True, 227)):
            with self.subTest(passed=passed, count=count):
                marker = {'passed': passed, 'count': count,
                          'configuration_sha256': self.config['configuration_sha256'],
                          'generation_sha256': runner.sha256(path)}
                (self.run / 'predictions/completed.json').write_text(json.dumps(marker))
                resources = types.ModuleType('code.hdepic_qwen38_baseline.resources')
                resources.require_idle_pair = Mock(side_effect=AssertionError('Invalid marker must not probe GPUs'))
                resources.require_disk_space = Mock()
                with patch.dict(sys.modules, {resources.__name__: resources}), \
                     patch.object(runner, 'freeze', return_value=self.config), \
                     patch.object(runner, 'fetch_weights') as weights, \
                     patch('builtins.__import__', side_effect=self.deny_model_imports(builtins.__import__)):
                    with self.assertRaisesRegex(ValueError, 'Completed run changed'):
                        runner.run_inference(self.run)
                resources.require_idle_pair.assert_not_called()
                resources.require_disk_space.assert_not_called()
                weights.assert_not_called()


if __name__ == '__main__':
    unittest.main()
