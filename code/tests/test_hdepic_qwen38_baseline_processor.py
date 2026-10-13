"""CPU regression checks for structured image placement and reference exclusion."""
import copy
import tempfile
import unittest
from pathlib import Path

from code.hdepic_qwen38_baseline.data import prompt_digest
from code.hdepic_qwen38_baseline.processor import build_messages, expected_serialization, NONTHINKING_PREFIX


class NativeMessageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        images = []
        for index in range(16):
            path = Path(self.temp.name) / f'{index:02d}.jpg'
            path.write_bytes(b'cached image')
            images.append(str(path))
        messages = [
            {'role': 'system', 'content': 'Return only the requested fields.'},
            {'role': 'user', 'content': '\n'.join(f'Frame {index:02d}, source time {index / 3:.6f} s: <image>'
                                                for index in range(16)) +
             '\nHistory: take spoon\nAction: transfer dough between spoons\nReturn <why>, <goal>, <next>.'},
        ]
        self.record = {'id': 'P07-video:explain:action1', 'row_index': 17,
                       'messages': messages, 'images': images,
                       'prompt_sha256': prompt_digest(messages, images)}

    def test_timestamps_interleave_with_original_image_order_without_text_changes(self):
        before = copy.deepcopy(self.record)
        result = build_messages(self.record)
        self.assertEqual(result[0], {'role': 'system', 'content': [
            {'type': 'text', 'text': self.record['messages'][0]['content']} ]})
        items = result[1]['content']
        self.assertEqual([item['image'] for item in items if item['type'] == 'image'], self.record['images'])
        self.assertEqual(''.join(item['text'] if item['type'] == 'text' else '<image>' for item in items),
                         self.record['messages'][1]['content'])
        for index in range(16):
            self.assertEqual(items[2 * index + 1], {'type': 'image', 'image': self.record['images'][index]})
            self.assertTrue(items[2 * index]['text'].endswith(f'{index / 3:.6f} s: '))
        self.assertEqual(self.record, before)

    def test_scoring_reference_fields_cannot_enter_model_messages(self):
        self.record['original_narration'] = 'A secret source explanation.'
        with self.assertRaisesRegex(ValueError, 'Unapproved answering-input fields'):
            build_messages(self.record)

    def test_assistant_target_cannot_enter_model_messages(self):
        self.record['messages'].append({'role': 'assistant', 'content': '<why>A secret reference.</why>'})
        self.record['prompt_sha256'] = prompt_digest(self.record['messages'], self.record['images'])
        with self.assertRaisesRegex(ValueError, 'Only system and user'):
            build_messages(self.record)

    def test_changed_prompt_or_reordered_images_cannot_reuse_original_fingerprint(self):
        for change in ('text', 'images'):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.record)
                if change == 'text':
                    changed['messages'][1]['content'] += '\nExtra context.'
                else:
                    changed['images'][0], changed['images'][1] = changed['images'][1], changed['images'][0]
                with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                    build_messages(changed)

    def test_native_expected_prompt_keeps_whitespace_and_empty_thinking_prefix(self):
        self.record['messages'][1]['content'] = ' ' + self.record['messages'][1]['content'] + '\n'
        serialized = expected_serialization(self.record)
        self.assertIn('<|im_start|>user\n Frame', serialized)
        self.assertTrue(serialized.endswith('\n<|im_end|>\n' + NONTHINKING_PREFIX))
        self.assertEqual(serialized.count('<image>'), 16)


if __name__ == '__main__':
    unittest.main()
