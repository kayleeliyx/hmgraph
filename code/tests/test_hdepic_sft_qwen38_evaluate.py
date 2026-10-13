"""CPU-only evaluation contract tests; no checkpoint/model/GPU loading."""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from code.hdepic_sft_qwen38 import evaluate


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_stage_paths_cannot_access_test_or_other_checkpoint(self):
        expected = {'smoke-reload': 2, 'epoch1': 909, 'epoch2': 1818}
        for stage, step in expected.items():
            checkpoint, output = evaluate.stage_paths(stage, self.path, 'attempt-002')
            self.assertEqual(checkpoint.name, f'checkpoint-{step}')
            self.assertTrue(output.is_relative_to(self.path))
        for stage in ('test', 'second_stage', 'base', 'track'):
            with self.assertRaises(ValueError):
                evaluate.stage_paths(stage, self.path)
        with self.assertRaises(ValueError):
            evaluate.stage_paths('smoke-reload', self.path, '../other')

    def test_saved_cache_requires_every_adapter_model_and_prompt_identity(self):
        record = {'id': 'P07-x:explain:x', 'row_index': 12, 'prompt_sha256': 'prompt'}
        config = {'configuration_sha256': 'config', 'checkpoint': {'adapter_sha256': 'adapter', 'adapter_config_sha256': 'adapter-config'}}
        row = {**record, 'configuration_sha256': 'config', 'model_repo': evaluate.MODEL_REPO,
               'model_revision': evaluate.MODEL_REVISION, 'adapter_sha256': 'adapter',
               'adapter_config_sha256': 'adapter-config', 'status': 'completed'}
        destination = self.path / 'generation.jsonl'
        destination.write_text(json.dumps(row) + '\n')
        self.assertEqual(evaluate.load_cached(destination, [record], config), {record['id']: row})
        for field in ('configuration_sha256', 'model_repo', 'model_revision', 'adapter_sha256',
                      'adapter_config_sha256', 'prompt_sha256', 'row_index'):
            changed = {**row, field: 'changed'}
            destination.write_text(json.dumps(changed) + '\n')
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity changed'):
                evaluate.load_cached(destination, [record], config)
        destination.write_text((json.dumps(row) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'Unknown/duplicate'):
            evaluate.load_cached(destination, [record], config)

    def test_prompt_contract_rejects_assistant_or_reference_fields(self):
        record = {'id': 'x', 'row_index': 0,
                  'messages': [{'role': 'system', 'content': 'System'},
                               {'role': 'user', 'content': '<image>' * 16 + 'Question'}],
                  'images': [f'/fake/frame{i}.jpg' for i in range(16)]}
        record['prompt_sha256'] = evaluate.prompt_digest(record['messages'], record['images'])
        evaluate._validate_prompt(record)
        leaked = copy.deepcopy(record); leaked['reference_reason'] = 'Secret target'
        with self.assertRaisesRegex(ValueError, 'unapproved fields'):
            evaluate._validate_prompt(leaked)
        leaked = copy.deepcopy(record); leaked['messages'].append({'role': 'assistant', 'content': 'Secret target'})
        with self.assertRaisesRegex(ValueError, 'assistant message'):
            evaluate._validate_prompt(leaked)

    def test_reference_join_is_blocked_before_all_answers_finish(self):
        record = {'id': 'sample'}
        with patch.object(evaluate, 'read_json', side_effect=AssertionError('No reference source should be opened')):
            with self.assertRaisesRegex(ValueError, 'Generate all smoke answers'):
                evaluate.write_smoke_reference_comparison(self.path, self.path, [record], {'sample': {'status': 'failed'}})

    def test_real_frozen_validation_prompts_are_only_same_228_explain(self):
        records, metadata, sources = evaluate.load_answering_records('epoch1')
        self.assertEqual(len(records), 228)
        self.assertTrue(all(set(r) == evaluate.PROMPT_KEYS for r in records))
        self.assertTrue(all(m == {'task': 'explain', 'has_goal': True, 'partition': 'validation'} for m in metadata.values()))
        original = evaluate.rows(evaluate.BASELINE / 'inputs/validation_explain.prompts.jsonl')
        self.assertEqual(records, original)
        self.assertEqual(sources['prompts_sha256'], evaluate.sha(evaluate.BASELINE / 'inputs/validation_explain.prompts.jsonl'))

    def test_real_smoke_selection_has_four_initial_sft_categories_without_answers(self):
        opened = []
        original_rows = evaluate.rows
        def track_rows(path):
            opened.append(str(path))
            return original_rows(path)
        with patch.object(evaluate, 'rows', side_effect=track_rows):
            records, metadata, sources = evaluate.load_answering_records('smoke-reload')
        self.assertEqual(len(records), 4)
        self.assertEqual({(m['task'], m['has_goal']) for m in metadata.values()},
                         {('explain', True), ('explain', False), ('track', True), ('track', False)})
        self.assertTrue(all(m['partition'] == 'initial_sft' for m in metadata.values()))
        self.assertTrue(all(path.endswith('initial_sft.prompts.jsonl') for path in opened))
        self.assertTrue(all([m['role'] for m in r['messages']] == ['system', 'user'] for r in records))


class TensorPrecisionTests(unittest.TestCase):
    def test_every_adapter_tensor_must_reload_exactly(self):
        import torch
        saved = {'x.lora_A.weight': torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                 'x.lora_B.weight': torch.tensor([[3.0], [4.0]], dtype=torch.float32)}
        loaded = {k: v.clone() for k, v in saved.items()}
        self.assertTrue(evaluate.verify_adapter_state(loaded, saved, expected_count=2)['all_saved_tensors_verified'])
        loaded['x.lora_A.weight'][0, 0] += 0.125
        with self.assertRaisesRegex(ValueError, 'did not reload exactly'):
            evaluate.verify_adapter_state(loaded, saved, expected_count=2)
        loaded = {k: v.to(torch.bfloat16) for k, v in saved.items()}
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            evaluate.verify_adapter_state(loaded, saved, expected_count=2)
        with self.assertRaisesRegex(ValueError, 'key mismatch'):
            evaluate.verify_adapter_state({}, saved, expected_count=2)

    def test_scoped_autocast_matches_bf16_leaf_and_restores_outer_fp32(self):
        import torch
        from torch import nn
        class Unit(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({'default': nn.Linear(2, 2, bias=False)})
                self.lora_B = nn.ModuleDict({'default': nn.Linear(2, 2, bias=False)})
        model = nn.Module()
        model.units = nn.ModuleList([Unit() for _ in range(496)])
        # Exercise identical scope/restore behavior with CPU autocast; CUDA is
        # deliberately unavailable and no actual checkpoint is loaded.
        facade = SimpleNamespace(nn=nn, bfloat16=torch.bfloat16,
                                 autocast=lambda _device, dtype: torch.autocast('cpu', dtype=dtype))
        handles, names = evaluate.install_lora_autocast(model, facade)
        try:
            layer = model.units[0].lora_A['default']
            value = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
            expected = nn.functional.linear(value.bfloat16(), layer.weight.bfloat16())
            actual = layer(value)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(layer.weight.dtype, torch.float32)
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertEqual((value @ value.T).dtype, torch.float32)
            with self.assertRaises(RuntimeError):
                layer(torch.ones(1, 7))
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertEqual(len(names), 992)
        finally:
            for handle in handles:
                handle.remove()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
