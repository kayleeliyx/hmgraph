"""Native CPU regression checks: exact answer masking and refusal on loss."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
from hdepic_sft_qwen38 import data


class NativeTrainingInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor = data.load_processor(data.METADATA)
        cls.samples = {}
        with (data.DATASET / 'swift/initial_sft.jsonl').open() as exports, \
                (data.DATASET / 'swift/initial_sft.scoring.jsonl').open() as scoring:
            for index, (raw, side) in enumerate(zip(exports, scoring)):
                item = json.loads(side)
                category = (item['task'], item['goal_status'] == 'available')
                if category not in cls.samples:
                    cls.samples[category] = (json.loads(raw), item['id'], index)
                if len(cls.samples) == 4:
                    break
        assert len(cls.samples) == 4

    def test_all_four_categories_have_complete_answer_only_loss_and_16_images(self):
        for category, (export, rid, index) in self.samples.items():
            with self.subTest(category=category):
                tensors, audit = data.build_training_inputs(self.processor, export, rid, index)
                prefix = audit['prompt_tokens_masked']
                self.assertTrue(torch.all(tensors['labels'][0, :prefix] == -100))
                supervised = tensors['labels'][0, prefix:].tolist()
                self.assertEqual(self.processor.tokenizer.decode(supervised, skip_special_tokens=False),
                                 export['messages'][-1]['content'] + '<|im_end|>')
                self.assertEqual(supervised[-1], data.EOS_IDS[0])
                self.assertEqual(audit['individual_images'], 16)
                self.assertEqual(audit['image_tokens_per_image'], [144] * 16)
                self.assertEqual(tensors['image_grid_thw'].tolist(), [[1, 24, 24]] * 16)
                self.assertTrue(torch.all(tensors['mm_token_type_ids'][0, prefix:] == 0))
                self.assertTrue(audit['native_full_conversation_verified'])
                self.assertTrue(audit['native_token_boundary_verified'])
                self.assertEqual(audit['system_user_prefix_loss_tokens'], 0)
                self.assertFalse(audit['trailing_transport_newline_included'])
                self.assertEqual('<goal>' in export['messages'][-1]['content'], category[1])

    def test_overlength_raises_instead_of_truncating_or_deleting(self):
        export, rid, index = next(iter(self.samples.values()))
        with patch.object(data, 'MAX_LENGTH', 10):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                data.build_training_inputs(self.processor, export, rid, index)

    def test_scoring_fields_cannot_enter_answering_input(self):
        export, rid, index = next(iter(self.samples.values()))
        modified = copy.deepcopy(export); modified['source_narration'] = 'Reference-only evidence'
        with self.assertRaisesRegex(ValueError, 'Unexpected training export fields'):
            data.build_training_inputs(self.processor, modified, rid, index)

    def test_native_template_cannot_silently_trim_target(self):
        export, rid, index = next(iter(self.samples.values()))
        modified = copy.deepcopy(export); modified['messages'][-1]['content'] += ' '
        with self.assertRaisesRegex(ValueError, 'edge whitespace'):
            data.build_training_inputs(self.processor, modified, rid, index)

    def test_unexpected_native_control_token_refused(self):
        export, rid, index = next(iter(self.samples.values()))
        modified = copy.deepcopy(export); modified['messages'][-1]['content'] += '<|im_end|>'
        with self.assertRaisesRegex(ValueError, 'Native control token'):
            data.build_training_inputs(self.processor, modified, rid, index)

    def test_test_and_second_stage_loads_forbidden_before_any_io(self):
        for partition in ('test', 'second_stage'):
            with self.assertRaisesRegex(ValueError, 'Only initial_sft and validation'):
                data.load_partition(partition, '/nonexistent')

    def test_no_batch_dimension_added_by_collator(self):
        value = {'input_ids': torch.tensor([[1, 2]])}
        self.assertIs(data.collate_single([value]), value)
        with self.assertRaisesRegex(ValueError, 'batch size one'):
            data.collate_single([value, value])


class CacheIdentityTest(unittest.TestCase):
    def test_stale_cache_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audits.jsonl'
            path.write_text(json.dumps({'id': 'abc', 'preflight_identity': 'old'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'cache identity'):
                data._cache_rows(path, 'new')

    def test_incomplete_record_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audits.jsonl'; path.write_bytes(b'{"id":"interrupted')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Incomplete/corrupt'):
                data._cache_rows(path, 'x')
            self.assertEqual(path.read_bytes(), before)

    def test_matching_rows_reused_and_duplicates_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audits.jsonl'
            row = {'id': 'abc', 'preflight_identity': 'same', 'status': 'passed'}
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(data._cache_rows(path, 'same'), {'abc': row})
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                data._cache_rows(path, 'same')


if __name__ == '__main__':
    unittest.main()
