"""CPU-only regression checks for exact prompt reuse and leakage boundaries."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from code.hdepic_qwen35_baseline.data import audit_records, immutable_write, select_smoke


class BaselinePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        frames = []
        for index in range(16):
            path = root / f'{index:02d}.jpg'
            contents = b'cached frame ' + str(index).encode()
            path.write_bytes(contents)
            frames.append({'path': str(path), 'sha256': hashlib.sha256(contents).hexdigest(),
                           'bytes': len(contents), 'source_pts': index,
                           'source_time_base': '1/3', 'timestamp_sec': index / 3})
        rid = 'P07-video:explain:P07-video-1'
        self.prompts = [{'messages': [
            {'role': 'system', 'content': 'Return requested tagged fields only.'},
            {'role': 'user', 'content': '\n'.join(f'Frame {i}: <image>' for i in range(16)) +
             '\nAction to explain: move cup\nReturn <why>, <goal>, <next>.'}],
            'images': [frame['path'] for frame in frames]}]
        self.texts = [{'id': rid, 'partition': 'validation', 'task': 'explain', 'query_time_sec': 5,
                       'input': {'history': [{'action': 'pick up cup', 'end_sec': 4}]},
                       'target': {'reason': 'so that there is space', 'goal': 'prepare tea'},
                       'scoring_metadata': {'reason_annotation': {
                           'original_narration': 'Move cup so that there is space',
                           'char_start': 9, 'char_end': 31}}}]
        self.scoring = [{'id': rid, 'row_index': 0, 'partition': 'validation',
                         'task': 'explain', 'query_time_sec': 5, 'frames': frames,
                         'current_action': {'end_sec': 5}, 'goal_status': 'available',
                         'video_id': 'P07-video', 'participant': 'P07'}]

    def run_audit(self):
        return audit_records(self.prompts, self.texts, self.scoring, expected_count=1)

    def test_answering_inputs_are_exact_and_have_no_reference_fields(self):
        before = copy.deepcopy((self.prompts, self.texts, self.scoring))
        answering, refs, sidecars, evidence, count = self.run_audit()
        self.assertEqual(set(answering[0]), {'id', 'row_index', 'messages', 'images', 'prompt_sha256'})
        self.assertEqual(answering[0]['messages'], self.prompts[0]['messages'])
        self.assertEqual(answering[0]['images'], self.prompts[0]['images'])
        self.assertEqual(refs, self.texts)
        self.assertEqual(sidecars, self.scoring)
        self.assertEqual(count, 16)
        self.assertEqual((self.prompts, self.texts, self.scoring), before)

    def test_assistant_message_is_rejected(self):
        self.prompts[0]['messages'].append({'role': 'assistant', 'content': '<why>reference</why>'})
        with self.assertRaisesRegex(ValueError, 'Assistant/reference'):
            self.run_audit()

    def test_changed_image_bytes_are_rejected(self):
        Path(self.prompts[0]['images'][0]).write_bytes(b'tampered frame')
        with self.assertRaisesRegex(ValueError, 'Cached image'):
            self.run_audit()

    def test_image_reordering_is_rejected(self):
        self.prompts[0]['images'][0:2] = self.prompts[0]['images'][1::-1]
        with self.assertRaisesRegex(ValueError, 'Image path/order'):
            self.run_audit()

    def test_future_history_is_rejected(self):
        self.texts[0]['input']['history'][0]['end_sec'] = 5.01
        with self.assertRaisesRegex(ValueError, 'Noncausal/oversized history'):
            self.run_audit()

    def test_source_pts_disagreement_is_rejected(self):
        self.scoring[0]['frames'][0]['source_pts'] = 1
        with self.assertRaisesRegex(ValueError, 'Source PTS mismatch'):
            self.run_audit()

    def test_future_frame_is_rejected(self):
        self.scoring[0]['frames'][-1]['timestamp_sec'] = 5.01
        with self.assertRaisesRegex(ValueError, 'Noncausal/out-of-window frame'):
            self.run_audit()

    def test_duplicate_source_ids_are_rejected(self):
        self.texts.append(copy.deepcopy(self.texts[0]))
        self.prompts.append(copy.deepcopy(self.prompts[0]))
        self.scoring.append(copy.deepcopy(self.scoring[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate text ID'):
            self.run_audit()

    def test_immutable_resume_never_overwrites_changed_file(self):
        path = Path(self.temp.name) / 'immutable.jsonl'
        immutable_write(path, b'original\n')
        before = path.stat().st_mtime_ns
        immutable_write(path, b'original\n')
        self.assertEqual(path.stat().st_mtime_ns, before)
        with self.assertRaisesRegex(ValueError, 'Immutable prepared output differs'):
            immutable_write(path, b'changed\n')
        self.assertEqual(path.read_bytes(), b'original\n')

    def test_smoke_selection_uses_length_and_video_diversity_no_goal_invention(self):
        evidence = [{'id': str(i), 'row_index': i, 'video_id': video,
                     'prompt_characters': length, 'history_count': 8, 'goal_status': 'available'}
                    for i, (video, length) in enumerate([
                        ('A', 200), ('A', 199), ('B', 180), ('C', 170), ('D', 160)])]
        selected = select_smoke(evidence)
        self.assertEqual(selected['ids'], ['0', '2', '3', '4'])
        self.assertEqual(selected['cohort_goal_absent_count'], 0)
        self.assertEqual(selected['goal_absent_coverage'], 'unavailable in this fixed Explain cohort')
        self.assertTrue(all(row['goal_status'] == 'available' for row in selected['selected']))


if __name__ == '__main__':
    unittest.main()
