"""CPU-only regression for causal-position and diagnostic classification logic."""
import unittest
import torch
from code.hdepic_sft_qwen38.parity_diagnostic import analyze, label_alignment


class ParityEvidenceTests(unittest.TestCase):
    def test_native_shift_matches_selected_predecessors_including_termination(self):
        labels = torch.tensor([[-100, -100, 2, 3, 4]])
        positions, evidence = label_alignment(labels)
        self.assertEqual(positions.tolist(), [2, 3, 4])
        self.assertEqual(evidence['selected_predecessor_positions'], [1, 2, 3])
        self.assertEqual(evidence['native_nonignored_positions'], [1, 2, 3])
        self.assertEqual(evidence['target_token_ids'], [2, 3, 4])

    def captures(self):
        selected = torch.tensor([[[0.1, 1.1, -0.3], [0.3, -1.0, 1.5]]], dtype=torch.bfloat16)
        full = selected.clone(); full[0, 0, 1] += 0.125
        hidden = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        targets = torch.tensor([1, 2])
        loss = torch.nn.functional.cross_entropy(full[0].float(), targets).item()
        common = {'selected_hidden': hidden, 'weight_sample_sha256': 'same'}
        return {'selected': {**common, 'selected_logits': selected},
                'repeat': {**common, 'selected_logits': selected.clone()},
                'full': {**common, 'selected_logits': full, 'short_bf16_logits': selected.clone(),
                         'fp32_logits': selected.float(), 'masked_ce_gpu': loss}}, targets, loss

    def test_shape_rounding_classification_requires_identical_hidden_and_repeat(self):
        captures, targets, loss = self.captures()
        _, alignment = label_alignment(torch.tensor([[-100, 1, 2]]))
        report = analyze(captures, targets, loss, alignment)
        self.assertEqual(report['classification'], 'bf16_head_projection_shape_dependence_supported')
        self.assertFalse(report['original_parity_tolerance_changed'])
        captures['full']['selected_hidden'] = captures['full']['selected_hidden'] + 0.25
        report = analyze(captures, targets, loss, alignment)
        self.assertEqual(report['classification'], 'upstream_hidden_difference_requires_investigation')

    def test_native_reduction_mismatch_is_not_dismissed_as_rounding(self):
        captures, targets, loss = self.captures()
        _, alignment = label_alignment(torch.tensor([[-100, 1, 2]]))
        report = analyze(captures, targets, loss + 0.01, alignment)
        self.assertEqual(report['classification'], 'native_loss_reduction_disagreement')


if __name__ == '__main__':
    unittest.main()
