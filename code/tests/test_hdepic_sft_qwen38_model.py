"""CPU/meta checks; no real checkpoint weights, inference, or GPU access."""
from __future__ import annotations

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from types import SimpleNamespace
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hdepic_sft_qwen38.common import BASELINE
from hdepic_sft_qwen38.model import (
    EXPECTED_BASE_PARAMETERS, EXPECTED_LORA_PARAMETERS, TARGET_REGEX,
    full_native_loss_parity, inject_text_lora, json_safe, training_loss,
)


class FakeCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_logits = nn.Parameter(torch.arange(42, dtype=torch.float32).reshape(1, 6, 7) / 10)
        self.last_indices = None

    def forward(self, input_ids, attention_mask=None, labels=None, logits_to_keep=0,
                use_cache=False, return_dict=True):
        assert use_cache is False
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        self.last_indices = indices
        logits = self.position_logits[:, indices, :]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 7).float(), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def sample_batch():
    return {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6]]),
            'attention_mask': torch.ones(1, 6, dtype=torch.long),
            'labels': torch.tensor([[-100, -100, -100, 4, 5, 6]])}


class LossTests(unittest.TestCase):
    def test_recursive_loading_diagnostics_remain_json_serializable(self):
        raw = {'unexpected_keys': {'mtp.b', 'mtp.a'}, 'missing_keys': set(),
               'nested': ({'groups': frozenset({'x', 'a'})}, {'shapes': {(2, 3), (1, 4)}})}
        normalized = json_safe(raw)
        self.assertEqual(normalized, {'unexpected_keys': ['mtp.a', 'mtp.b'], 'missing_keys': [],
                                     'nested': [{'groups': ['a', 'x']}, {'shapes': [[1, 4], [2, 3]]}]})
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)
        self.assertIsInstance(raw['unexpected_keys'], set)

    def test_complete_assistant_predecessors_and_masked_prompt_gradients(self):
        model = FakeCausalModel()
        batch = sample_batch()
        loss = training_loss(model, batch)
        self.assertTrue(torch.equal(model.last_indices, torch.tensor([2, 3, 4])))
        expected = F.cross_entropy(model.position_logits[0, 2:5], torch.tensor([4, 5, 6]))
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        grads = model.position_logits.grad.abs().sum(dim=-1)[0]
        self.assertTrue(torch.equal(grads > 0, torch.tensor([False, False, True, True, True, False])))

    def test_sparse_mask_and_last_termination_token_remain_supervised(self):
        model = FakeCausalModel()
        batch = sample_batch()
        batch['labels'][0, 4] = -100
        loss = training_loss(model, batch)
        self.assertTrue(torch.equal(model.last_indices, torch.tensor([2, 4])))
        expected = F.cross_entropy(model.position_logits[0, [2, 4]], torch.tensor([4, 6]))
        self.assertTrue(torch.equal(loss, expected))

    def test_full_loss_parity_and_training_mode_restored(self):
        model = FakeCausalModel()
        audit = full_native_loss_parity(model, sample_batch())
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['supervised_token_count'], 3)
        self.assertTrue(model.training)

    def test_rejects_invalid_mask_or_answer_mutation(self):
        batch = sample_batch()
        batch['labels'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'causal predecessor'):
            training_loss(FakeCausalModel(), batch)
        batch = sample_batch()
        batch['labels'][0, 3] = 2
        with self.assertRaisesRegex(ValueError, 'unchanged assistant'):
            training_loss(FakeCausalModel(), batch)
        batch = sample_batch()
        batch['labels'].fill_(-100)
        with self.assertRaisesRegex(ValueError, 'causal predecessor'):
            training_loss(FakeCausalModel(), batch)

    def test_rejects_nonunit_batch_and_scoring_metadata(self):
        batch = {key: value.repeat(2, 1) for key, value in sample_batch().items()}
        with self.assertRaisesRegex(ValueError, 'batch size'):
            training_loss(FakeCausalModel(), batch)
        batch = sample_batch()
        batch['reference_reason'] = 'must never enter model kwargs'
        with self.assertRaisesRegex(ValueError, 'Unexpected training/scoring fields'):
            training_loss(FakeCausalModel(), batch)


class NativeArchitectureTests(unittest.TestCase):
    def test_full_meta_injection_frozen_vision_and_checkpointing(self):
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
        from torch.distributed._composable.fsdp import FSDPModule, fully_shard
        self.assertTrue(callable(fully_shard))
        config = AutoConfig.from_pretrained(BASELINE / 'model_metadata', local_files_only=True)
        with torch.device('meta'):
            native = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
            model, audit = inject_text_lora(native)
        self.assertEqual(audit['target_regex'], TARGET_REGEX)
        self.assertEqual(audit['trainable_parameter_count'], EXPECTED_LORA_PARAMETERS)
        self.assertEqual(audit['frozen_parameter_count'], EXPECTED_BASE_PARAMETERS)
        self.assertEqual(len(audit['target_modules']), 496)
        self.assertEqual(audit['trainable_tensor_count'], 992)
        self.assertEqual({p['dtype'] for p in audit['trainable_parameters']}, {'torch.float32'})
        self.assertTrue(all(not getattr(m, 'gradient_checkpointing', False)
                            for m in native.model.visual.modules()))
        self.assertTrue(all(getattr(m, 'gradient_checkpointing', False)
                            for m in native.model.language_model.layers))
        self.assertTrue(all(p.device.type == 'meta' for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_tiny_native_cpu_lora_loss_parity_and_backward(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        config = Qwen3_5Config(
            text_config={
                'vocab_size': 128, 'hidden_size': 32, 'intermediate_size': 64,
                'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                'head_dim': 8, 'max_position_embeddings': 128,
                'linear_num_key_heads': 2, 'linear_num_value_heads': 2,
                'linear_key_head_dim': 8, 'linear_value_head_dim': 8,
                'layer_types': ['linear_attention', 'full_attention'],
                'rope_parameters': {'rope_type': 'default', 'rope_theta': 1000000.0,
                                    'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]},
            },
            vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                           'num_heads': 4, 'out_hidden_size': 32, 'patch_size': 16,
                           'spatial_merge_size': 2},
        )
        config._attn_implementation = 'sdpa'
        torch.manual_seed(20261003)
        native = Qwen3_5ForConditionalGeneration(config)
        model, audit = inject_text_lora(native, enforce_full_architecture=False)
        batch = sample_batch()
        parity = full_native_loss_parity(model, batch)
        self.assertTrue(parity['passed'])
        model.train()
        loss = training_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
