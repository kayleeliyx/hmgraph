"""CPU-only archive exclusions, completion gates and byte reproducibility."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

from code.hdepic_qwen38_baseline.data import json_bytes, jsonl_bytes, prompt_digest, sha256
from code.hdepic_qwen38_baseline.package import (
    HUMAN_FIELDS, MODEL_METADATA, ROOT_FILES, PendingPackage, allowed_member,
    build_package, collect_files, deterministic_zip, validate_ready, verify_safe_content)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'runs/baseline'
        self.run.mkdir(parents=True)

    def write(self, name, value):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(value))
        return path

    def complete_fixture(self):
        prompts, generation, cases = [], [], []
        for index in range(228):
            rid = f'validation-{index}'
            messages = [{'role': 'system', 'content': 'Return fields.'}, {'role': 'user', 'content': '<why><goal><next>'}]
            images = [f'/cached/{index}/{frame}.jpg' for frame in range(16)]
            digest = prompt_digest(messages, images)
            prompts.append({'id': rid, 'row_index': index, 'messages': messages, 'images': images, 'prompt_sha256': digest})
            generation.append({'id': rid, 'row_index': index, 'prompt_sha256': digest, 'status': 'completed',
                               'prediction': '<why>saved answer</why>', 'configuration_sha256': 'v1', 'model_revision': 'revision'})
            cases.append({'id': rid, 'case': f'Q{index + 1:03d}', 'existing_fixed50': index < 50,
                          'answers': {slot: {'why': 'saved answer'} for slot in 'ABCDE'}, 'frames': [{'path': path} for path in images],
                          'contact_sheet': f'contact_sheets/S{index + 1:03d}.jpg'})
        prompt_path = self.run / 'inputs/validation_explain.prompts.jsonl'
        prompt_path.parent.mkdir(); prompt_path.write_bytes(jsonl_bytes(prompts))
        manifest = self.write('inputs/manifest.json', {'project': str(self.root), 'count': 228,
                                                     'ordered_ids': [prompt['id'] for prompt in prompts]})
        gen = self.run / 'predictions/generation.jsonl'; gen.parent.mkdir(); gen.write_bytes(jsonl_bytes(generation))
        self.write('predictions/completed.json', {'passed': True, 'count': 228, 'configuration_sha256': 'v1',
                                                'generation_sha256': sha256(gen), 'inputs_manifest_sha256': sha256(manifest)})
        metadata = {}
        for name in MODEL_METADATA:
            path = self.run / 'model_metadata' / name; path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'{}\n' if name.endswith('.json') else b'native metadata\n')
            metadata[name] = sha256(path)
        identity = self.write('model_identity.json', {'repo_id': 'Qwen/Qwen3.8-27B',
                                                     'revision': 'revision', 'metadata_sha256': metadata})
        config = {'repo_id': 'Qwen/Qwen3.8-27B', 'revision': 'revision',
                  'configuration_sha256': 'v1', 'model_identity_sha256': sha256(identity)}
        active = self.write('frozen_config.json', config)
        self.write('active_configuration.json', {'path': 'frozen_config.json', 'configuration_sha256': 'v1',
                                                'file_sha256': sha256(active), 'previous_configuration': None,
                                                'implementation_fix': None})
        casepath = self.run / 'review/blinded/cases.jsonl'; casepath.parent.mkdir(parents=True); casepath.write_bytes(jsonl_bytes(cases))
        for case in cases:
            path = self.run / 'review/blinded' / case['contact_sheet']; path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'review-only image bytes')
        self.write('review/materials_manifest.json', {'cases': 228, 'answer_slots': 1140, 'fixed50': 50,
                                                    'rubric_sha256': 'frozen-rubric', 'status': 'materials_prepared_review_pending',
                                                    'generation_sha256': sha256(gen), 'inputs_manifest_sha256': sha256(manifest),
                                                    'files_sha256': {'blinded/cases.jsonl': sha256(casepath)}})
        with (self.run / 'review/blinded/human_review.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['case', 'id', 'slot', *HUMAN_FIELDS]); writer.writeheader()
            for case in cases:
                for slot in 'ABCDE':
                    writer.writerow({'case': case['case'], 'id': case['id'], 'slot': slot,
                                     **{field: '' for field in HUMAN_FIELDS}})
        self.write('metrics.json', {'generation_completed': True, 'generation': {'denominator': 228, 'completed': 228},
                                   'configuration_sha256': 'v1', 'semantic_review': {'materials_prepared': True},
                                   'preservation': {'passed': True}, 'no_new_training': True, 'no_test_access': True,
                                   'references_or_narration_supplied_to_answering_model': False})
        (self.run / 'execution_report.md').write_text('Completed228; semantic review pending.\n')

    def test_incomplete_stages_do_not_create_archive_or_packaging_directory(self):
        with self.assertRaises(PendingPackage):
            build_package(self.run, self.root)
        self.assertFalse((self.run / 'packaging').exists())
        self.assertFalse(any(self.run.glob('*.zip')))

    def test_complete_gate_accepts_all228_and_requires_report_after_review(self):
        self.complete_fixture()
        ready = validate_ready(self.run)
        self.assertTrue(ready['human_fields_all_blank'])
        metrics = json.loads((self.run / 'metrics.json').read_text())
        metrics['semantic_review']['materials_prepared'] = False
        self.write('metrics.json', metrics)
        with self.assertRaisesRegex(ValueError, 'predates five-answer'):
            validate_ready(self.run)

    def test_gate_rejects_modified_review_materials(self):
        self.complete_fixture()
        with (self.run / 'review/blinded/cases.jsonl').open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'review artifact changed'):
            validate_ready(self.run)

    def test_weight_cache_private_keys_and_test_dataset_paths_are_forbidden(self):
        for name in ('review/private/identity_key.json', 'review/identity_key.json', 'model.safetensors',
                     'model.bin', 'weights.gguf', 'frames/image.jpg', 'datasets/test.jsonl',
                     'training/checkpoint/adapter_model.safetensors', '../escape.json', 'review/blinded/nested.zip'):
            self.assertFalse(allowed_member(name), name)
        self.assertTrue(allowed_member('model_identity.json'))
        self.assertTrue(allowed_member('model_metadata/model.safetensors.index.json'))
        self.assertTrue(allowed_member('source/code/tests/test_hdepic_qwen38_baseline_package.py'))
        self.assertTrue(allowed_member('review/blinded/contact_sheets/S001.jpg'))

    def test_actual_private_seed_or_mapping_cannot_be_smuggled_in_json(self):
        for obj in ({'nested': {'private_seed': 'secret'}}, {'items': [{'case_slots': {'A': 'model'}}]}):
            with self.assertRaisesRegex(ValueError, 'Private identity/seed'):
                verify_safe_content('audit.json', json_bytes(obj))
        verify_safe_content('audit.json', json_bytes({'private_seed_sha256': 'hash-only', 'seed': 20261006}))

    def test_zip_is_repeatable_crc_checked_and_members_byte_identical(self):
        payloads = {'metrics.json': b'{"count":228}\n', 'execution_report.md': 'Saved café\n'.encode(),
                    'review/blinded/contact_sheets/S001.jpg': b'review image bytes'}
        archive = self.run / 'review.zip'
        first = deterministic_zip(archive, 'baseline', payloads)
        mtime = archive.stat().st_mtime_ns
        second = deterministic_zip(archive, 'baseline', payloads)
        self.assertEqual(first, second)
        self.assertEqual(archive.stat().st_mtime_ns, mtime)
        with zipfile.ZipFile(archive) as handle:
            self.assertIsNone(handle.testzip())
            self.assertEqual(set(handle.namelist()), {'baseline/' + name for name in payloads})
            for name, content in payloads.items():
                self.assertEqual(handle.read('baseline/' + name), content)
        with self.assertRaisesRegex(ValueError, 'Immutable prepared output differs'):
            deterministic_zip(archive, 'baseline', {'metrics.json': b'{"count":227}\n'})

    def test_both_complete_archives_are_stable_and_blinded_zip_has_no_named_evidence(self):
        self.complete_fixture()
        for name in ROOT_FILES:
            if not (self.run / name).exists():
                self.write(name, {})
        for name in ('review/rubric.md', 'review/review_addendum.md'):
            (self.run / name).write_text('Frozen review criteria.\n')
        for name in ('code/__init__.py', 'code/tests/__init__.py', 'code/hdepic_experiment/__init__.py',
                     'code/hdepic_experiment/metrics.py', 'code/hdepic_review_followup/__init__.py',
                     'code/hdepic_review_followup/fields.py'):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# frozen source\n')
        for name in ('data.py', 'model.py', 'processor.py', 'resources.py', 'runner.py',
                     'review.py', 'report.py', 'package.py'):
            path = self.root / 'code/hdepic_qwen38_baseline' / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# new source\n')
        for name in ('run_hdepic_qwen38_baseline.sh', 'prepare_hdepic_qwen38_environment.sh'):
            path = self.root / 'scripts' / name; path.parent.mkdir(exist_ok=True); path.write_text('# script\n')
        self.write('frozen_parser.json', {'strict_parser_sha256': sha256(self.root / 'code/hdepic_experiment/metrics.py'),
                                          'supplementary_parser_sha256': sha256(self.root / 'code/hdepic_review_followup/fields.py')})
        self.write('review/private/identity_key.json', {'private_seed': 'must remain outside'})
        first = build_package(self.run, self.root)
        second = build_package(self.run, self.root)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(first['blinded_sha256'], second['blinded_sha256'])
        self.assertNotEqual(first['sha256'], first['blinded_sha256'])
        with zipfile.ZipFile(first['blinded_archive']) as handle:
            names = handle.namelist()
            self.assertEqual(len(names), first['blinded_members'])
            self.assertFalse(any('predictions/' in name or '/source/' in name or 'model_identity' in name or '/private/' in name for name in names))
            self.assertTrue(all('/review/blinded/' in name or name.endswith(('/review/rubric.md', '/review/review_addendum.md',
                             '/review/REVIEW_INSTRUCTIONS.md', '/blinded_package_manifest.json')) for name in names))
        with zipfile.ZipFile(first['archive']) as handle:
            self.assertIn('baseline/predictions/generation.jsonl', handle.namelist())
            self.assertFalse(any('identity_key' in name or '/private/' in name for name in handle.namelist()))

    def test_collector_never_opens_private_key_and_copies_no_original_training_files(self):
        for name in ROOT_FILES:
            self.write(name, {})
        for name in MODEL_METADATA:
            self.write('model_metadata/' + name, {})
        for name in ('predictions/generation.jsonl', 'predictions/completed.json',
                     'review/rubric.md', 'review/review_addendum.md', 'review/materials_manifest.json',
                     'review/blinded/cases.jsonl'):
            self.write(name, {})
        key = self.write('review/private/identity_key.json', {'private_seed': 'do not read'})
        for name in ('code/__init__.py', 'code/tests/__init__.py', 'code/hdepic_experiment/__init__.py',
                     'code/hdepic_experiment/metrics.py', 'code/hdepic_review_followup/__init__.py',
                     'code/hdepic_review_followup/fields.py'):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('pass\n')
        hidden = self.root / 'runs/original/training/checkpoint/weights.safetensors'
        hidden.parent.mkdir(parents=True); hidden.write_bytes(b'old weights')
        download_timing = self.write('weight_download_timing.json', {'elapsed_sec': 123.25})
        members = collect_files(self.run, self.root)
        self.assertEqual(members['weight_download_timing.json'], download_timing)
        self.assertNotIn(key, members.values())
        self.assertNotIn(hidden, members.values())
        self.assertFalse(any('private' in name or 'identity_key' in name or 'training' in name for name in members))


if __name__ == '__main__':
    unittest.main()
