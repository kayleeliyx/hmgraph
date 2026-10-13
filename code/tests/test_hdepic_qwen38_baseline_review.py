"""CPU-only guards for future five-answer review material preparation."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from code.hdepic_qwen38_baseline.data import json_bytes, jsonl_bytes, prompt_digest, sha256
from code.hdepic_qwen38_baseline.review import (
    HUMAN_FIELDS, PendingGeneration, STAGES, build_html, build_packet, completed_generation,
    human_rows, prepare_review, preserve_human_csv)


class ReviewPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = []
        self.source = []
        self.predictions = {stage: {} for stage in STAGES}
        for index in range(3):
            rid = f'validation-{index}'
            messages = [{'role': 'system', 'content': 'Return tagged fields.'},
                        {'role': 'user', 'content': '<image>\nExplain action in <why> and <goal>, then <next>.'}]
            images = [f'/cached/{index}.jpg']
            self.prompts.append({'id': rid, 'row_index': index, 'messages': messages, 'images': images,
                                 'prompt_sha256': prompt_digest(messages, images)})
            self.source.append({'id': rid, 'case': f'S{index + 1:03d}', 'existing_fixed50': index == 0,
                                'input_prompt': messages, 'frames': [{'path': images[0]}],
                                'action_question': 'move cup', 'original_narration': 'Move cup to clear space',
                                'reference_reason': 'to clear space', 'reason_char_start': 9, 'reason_char_end': 23,
                                'contact_sheet': f'contact_sheets/S{index + 1:03d}.jpg'})
            for stage in STAGES:
                self.predictions[stage][rid] = {
                    'id': rid, 'status': 'completed', 'prediction': f'<why>{stage} answer</why><next>badNEXT',
                    'finish_reason': 'length', 'generated_tokens': 256, 'row_index': index,
                    'prompt_sha256': self.prompts[-1]['prompt_sha256']}

    def completion_fixture(self, values=None):
        generation = self.root / 'generation.jsonl'
        generation.write_bytes(jsonl_bytes(values if values is not None else list(self.predictions['qwen38'].values())))
        manifest = self.root / 'manifest.json'
        manifest.write_bytes(b'{"fixed":true}\n')
        marker = self.root / 'completed.json'
        marker.write_bytes(json_bytes({'passed': True, 'count': len(self.prompts),
                                       'generation_sha256': sha256(generation),
                                       'inputs_manifest_sha256': sha256(manifest)}))
        return generation, marker, manifest

    def test_no_generation_does_not_create_review_directory_or_read_reference_files(self):
        (self.root / 'inputs').mkdir()
        prompts = self.root / 'inputs/validation_explain.prompts.jsonl'
        cohort = []
        for index in range(228):
            prompt = copy.deepcopy(self.prompts[0])
            prompt.update(id=f'fixed-{index}', row_index=index)
            cohort.append(prompt)
        prompts.write_bytes(jsonl_bytes(cohort))
        manifest = {'count': 228, 'partition': 'validation', 'task': 'explain',
                    'ordered_ids': [prompt['id'] for prompt in cohort],
                    'files_sha256': {'inputs/validation_explain.prompts.jsonl': sha256(prompts)}}
        (self.root / 'inputs/manifest.json').write_bytes(json_bytes(manifest))
        with self.assertRaises(PendingGeneration):
            prepare_review(self.root)
        self.assertFalse((self.root / 'review').exists())
        self.assertFalse((self.root / 'scoring').exists())

    def test_completion_gate_verifies_count_and_input_identity(self):
        generation, marker, manifest = self.completion_fixture()
        result = completed_generation(generation, marker, manifest, self.prompts)
        self.assertEqual(set(result), {row['id'] for row in self.prompts})
        self.prompts[0]['prompt_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'prompt identity'):
            completed_generation(generation, marker, manifest, self.prompts)

    def test_completion_gate_rejects_failed_and_missing_rows(self):
        bad = copy.deepcopy(list(self.predictions['qwen38'].values()))
        bad[0]['status'] = 'failed'
        paths = self.completion_fixture(bad)
        with self.assertRaisesRegex(ValueError, 'Incomplete new answer'):
            completed_generation(*paths, self.prompts)
        paths = self.completion_fixture(bad[1:])
        with self.assertRaisesRegex(ValueError, 'IDs differ'):
            completed_generation(*paths, self.prompts)

    def test_generation_bytes_changed_after_completion_are_rejected(self):
        paths = self.completion_fixture()
        with paths[0].open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'generation bytes'):
            completed_generation(*paths, self.prompts)

    def test_five_answer_packet_has_stable_new_order_exact_sources_no_ratings(self):
        source_before = copy.deepcopy(self.source)
        first, key = build_packet(self.source, self.predictions, self.prompts, 'private-test-seed')
        second, second_key = build_packet(self.source, self.predictions, self.prompts, 'private-test-seed')
        self.assertEqual(first, second)
        self.assertEqual(key, second_key)
        self.assertEqual(self.source, source_before)
        original = {row['id']: row for row in self.source}
        for case in first:
            self.assertEqual(set(case['answers']), set('ABCDE'))
            self.assertEqual(set(key[case['case']]['slots'].values()), set(STAGES))
            for field in ('original_narration', 'reference_reason', 'reason_char_start',
                          'reason_char_end', 'frames', 'input_prompt', 'existing_fixed50'):
                self.assertEqual(case[field], original[case['id']][field])
            self.assertNotIn('reference_status', case)
            for answer in case['answers'].values():
                self.assertNotIn('label', answer)
                self.assertEqual(answer['extraction_status'], 'extracted')
                self.assertFalse(answer['whole_response_format']['valid'])
                self.assertTrue(answer['finish_length'])

    def test_fifth_model_preserves_prior35_and_new38_as_distinct_saved_answers(self):
        cases, key = build_packet(self.source, self.predictions, self.prompts, 'seed')
        self.assertEqual(len(STAGES), 5)
        for case in cases:
            by_stage = {stage: case['answers'][slot]['why']
                        for slot, stage in key[case['case']]['slots'].items()}
            self.assertEqual(by_stage['qwen35'], 'qwen35 answer')
            self.assertEqual(by_stage['qwen38'], 'qwen38 answer')
        self.predictions['qwen35'].pop('validation-0')
        with self.assertRaisesRegex(ValueError, 'qwen35 answers do not cover'):
            build_packet(self.source, self.predictions, self.prompts, 'seed')

    def test_unextractable_why_retained_without_inventing_answer_or_score(self):
        self.predictions['qwen38']['validation-0']['prediction'] = '<why>unclosed'
        cases, key = build_packet(self.source, self.predictions, self.prompts, 'seed')
        case = next(row for row in cases if row['id'] == 'validation-0')
        slot = next(slot for slot, stage in key[case['case']]['slots'].items() if stage == 'qwen38')
        self.assertIsNone(case['answers'][slot]['why'])
        self.assertEqual(case['answers'][slot]['extraction_status'], 'ambiguous')
        self.assertNotIn('label', case['answers'][slot])
        self.assertEqual(len(human_rows(cases)), 15)

    def test_ambiguous_why_keeps_raw_evidence_in_csv_and_explicit_html_notice(self):
        raw = '<why>Move it away,\nbecause space is needed<why/>'
        self.predictions['qwen38']['validation-0']['prediction'] = raw
        cases, key = build_packet(self.source, self.predictions, self.prompts, 'seed')
        case = next(row for row in cases if row['id'] == 'validation-0')
        slot = next(slot for slot, stage in key[case['case']]['slots'].items() if stage == 'qwen38')
        values = human_rows(cases)
        path = self.root / 'human.csv'
        preserve_human_csv(path, values)
        with path.open(newline='') as handle:
            exported = list(csv.DictReader(handle))
        row = next(row for row in exported if row['id'] == case['id'] and row['slot'] == slot)
        self.assertEqual(row['answer_why'], '')
        self.assertEqual(row['answer_raw_response'], raw)
        self.assertEqual(row['why_extraction_status'], 'ambiguous')
        self.assertEqual(row['finish_reason'], 'length')
        self.assertTrue(all(row[field] == '' for row in exported for field in HUMAN_FIELDS))
        self.assertNotIn('model', row)
        self.assertNotIn('model_identity', row)
        case['query_time_sec'] = 5.0
        case['history'] = []
        rendered = build_html([case]).decode()
        self.assertIn('WHY could not be extracted; inspect saved raw response', rendered)
        self.assertIn('&lt;why&gt;Move it away,\nbecause space is needed&lt;why/&gt;', rendered)
        self.assertNotIn('<p>None</p>', rendered)
        self.assertEqual(case['reference_reason'], self.source[0]['reference_reason'])

    def test_initial_human_fields_blank_and_existing_entries_preserved(self):
        cases, _ = build_packet(self.source, self.predictions, self.prompts, 'seed')
        values = human_rows(cases)
        self.assertTrue(all(row[field] == '' for row in values for field in HUMAN_FIELDS))
        path = self.root / 'human.csv'
        preserve_human_csv(path, values)
        with path.open(newline='') as handle:
            actual = list(csv.DictReader(handle))
        actual[0]['human_reviewer_notes'] = 'Existing human annotation; preserve this.'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(actual[0]))
            writer.writeheader(); writer.writerows(actual)
        before = path.read_bytes()
        preserve_human_csv(path, values)
        self.assertEqual(path.read_bytes(), before)
        changed = copy.deepcopy(values); changed[0]['answer_why'] = 'reassigned answer'
        with self.assertRaisesRegex(ValueError, 'identity/content differs'):
            preserve_human_csv(path, changed)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
