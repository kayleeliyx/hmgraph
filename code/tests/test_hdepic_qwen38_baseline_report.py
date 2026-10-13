"""Report regressions use synthetic rows and temporary evidence, never inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code.hdepic_qwen38_baseline import report
from code.hdepic_qwen38_baseline.data import sha256


class ReportTests(unittest.TestCase):
    def test_repeated_report_is_byte_and_mtime_stable_until_evidence_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            first = {'report_created_at_utc': 'first-time', 'completed': 228, 'preservation': {'passed': True}}
            second = {**first, 'report_created_at_utc': 'later-time'}
            changed = {**first, 'report_created_at_utc': 'changed-evidence-time', 'new_evidence': True}
            with patch.object(report, 'build_metrics', side_effect=[first, second, changed]), \
                    patch.object(report, 'markdown', side_effect=lambda value: json.dumps(value, sort_keys=True)):
                report.generate_report(run)
                paths = [run / 'metrics.json', run / 'execution_report.md']
                before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
                repeated = report.generate_report(run)
                self.assertEqual(repeated['report_created_at_utc'], 'first-time')
                self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in paths], before)
                updated = report.generate_report(run)
                self.assertEqual(updated['report_created_at_utc'], 'changed-evidence-time')
                self.assertNotEqual((run / 'metrics.json').read_bytes(), before[0][0])

    def test_active_configuration_hash_and_unchanged_settings_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            original = {'dtype': 'bfloat16', 'configuration_sha256': 'old',
                        'answering_code_sha256': {'runner.py': 'old-code'}}
            active = {'dtype': 'bfloat16', 'configuration_sha256': 'new',
                      'answering_code_sha256': {'runner.py': 'fixed-code'}}
            (run / 'frozen_config.json').write_text(json.dumps(original))
            path = run / 'frozen_config_v2.json'; path.write_text(json.dumps(active))
            pointer = {'path': path.name, 'configuration_sha256': 'new', 'file_sha256': sha256(path),
                       'previous_configuration': 'frozen_config.json'}
            (run / 'active_configuration.json').write_text(json.dumps(pointer))
            actual, evidence = report.load_active_configuration(run)
            self.assertEqual(actual, active)
            self.assertTrue(evidence['experimental_settings_unchanged'])
            path.write_text(json.dumps({**active, 'dtype': 'float16'}))
            with self.assertRaisesRegex(ValueError, 'pointer does not match'):
                report.load_active_configuration(run)
            pointer['file_sha256'] = sha256(path)
            (run / 'active_configuration.json').write_text(json.dumps(pointer))
            with self.assertRaisesRegex(ValueError, 'changed frozen experimental settings'):
                report.load_active_configuration(run)

    def test_first_configuration_pointer_has_no_inherited_repair_or_previous_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder); value = {'configuration_sha256': 'first'}
            path = run / 'frozen_config.json'; path.write_text(json.dumps(value))
            pointer = {'path': path.name, 'file_sha256': sha256(path),
                       'configuration_sha256': 'first', 'previous_configuration': None,
                       'implementation_fix': None}
            (run / 'active_configuration.json').write_text(json.dumps(pointer))
            actual, evidence = report.load_active_configuration(run)
            self.assertEqual(actual, value)
            self.assertIsNone(evidence['previous_configuration_path'])
            self.assertIsNone(evidence['experimental_settings_unchanged'])
            self.assertIsNone(evidence['pointer']['implementation_fix'])

    def test_first_profile_markdown_reports_dense_model_without_invented_repair(self):
        metrics = {
            'status': 'pending',
            'generation': {'completed': 0, 'missing_prediction': 228, 'failed_or_other_status': 0,
                           'strict_whole_response_format_valid': 0, 'independent_why_extractable': 0,
                           'length_limited': 0, 'generation_latency': report.timing([])},
            'precision_and_execution': {'physical_gpus': [0, 1], 'dtype': 'bfloat16',
                'quantization': None, 'attention_implementation': 'sdpa',
                'cpu_or_disk_offload': False, 'visual_tokens_per_frame': 144,
                'max_context': 8192, 'generation': {'max_new_tokens': 256}},
            'checkpoint': {'repo_id': 'Qwen/Qwen3.8-27B', 'revision': 'pinned', 'adaptation': 'none',
                'architecture': ['Qwen3_5ForConditionalGeneration'], 'native_model_loading_verified': False,
                'cache_evidence': {'complete_weight_files_present': 0, 'expected_weight_shards': 18}},
            'active_configuration': {'path': '/new/frozen_config.json', 'file_sha256': 'frozen',
                'pointer': {'previous_configuration': None, 'implementation_fix': None},
                'previous_configuration_path': None},
            'smoke': {'criteria_passed': False}, 'native_loading_latency_sec': None,
            'gpu_memory': {'devices': {}}, 'semantic_review': {'status': 'Pending review.'},
            'environment_failures_and_fixes': [], 'failures': [],
            'postgeneration_weight_audit_failures_and_fixes': [], 'artifacts': {},
            'rerun_launcher': '/new/launch.sh',
        }
        text = report.markdown(metrics)
        self.assertIn('dense architecture', text)
        self.assertIn('frozen_config.json', text)
        self.assertNotIn('frozen_config_v2', text)
        self.assertNotIn('first failure', text)
        self.assertNotIn('MoE', text)
        self.assertNotIn('Repair:', text)

    def test_setup_issues_are_explicit_for_new_run_without_prior35_log_heuristics(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'environment_setup.log').write_text('Name or service not known')
            self.assertEqual(report.environment_issues(run), [])
            (run / 'environment').mkdir()
            issues = [{'issue': 'Actual new-run issue', 'resolution_evidence': 'Saved evidence',
                       'failure_log': str(run / 'environment_setup.log')}]
            (run / 'environment/setup_issues.json').write_text(json.dumps(issues))
            self.assertEqual(report.environment_issues(run), issues)

    def test_no_pointer_uses_preserved_v1_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder); value = {'configuration_sha256': 'first'}
            (run / 'frozen_config.json').write_text(json.dumps(value))
            actual, evidence = report.load_active_configuration(run)
            self.assertEqual(actual, value)
            self.assertIsNone(evidence['pointer'])

    def test_partial_rows_keep_fixed_denominator_and_missing_answers(self):
        ids = [f'id-{index}' for index in range(228)]
        predictions = [
            {'id': ids[0], 'status': 'completed', 'prediction': '<why>reason</why>',
             'parsed': {'valid': False}, 'independent_why': {'extractable': True},
             'finish_reason': 'length', 'latency_sec': 3.0},
            {'id': ids[1], 'status': 'failed'},
        ]
        result = report.aggregate_predictions(predictions, ids)
        self.assertEqual(result['denominator'], 228)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['missing_prediction'], 226)
        self.assertEqual(result['failed_or_other_status'], 1)
        self.assertEqual(result['strict_whole_response_format_valid'], 0)
        self.assertEqual(result['independent_why_extractable'], 1)
        self.assertEqual(result['why_missing_or_unavailable_over_all_228'], 227)
        self.assertEqual(result['length_limited'], 1)
        self.assertFalse(result['semantic_accuracy_measured'])

    def test_duplicate_or_out_of_cohort_predictions_fail(self):
        ids = [f'id-{index}' for index in range(228)]
        for values in ([{'id': 'outside'}], [{'id': ids[0]}, {'id': ids[0]}]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'Unknown/duplicate'):
                report.aggregate_predictions(values, ids)

    def test_download_duration_is_separate_explicit_and_never_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            self.assertIsNone(report.download_timing_evidence(run))
            path = run / 'weight_download_timing.json'
            saved = {'elapsed_sec': 123.25, 'started_at_utc': 'recorded-start'}
            path.write_text(json.dumps(saved))
            evidence = report.download_timing_evidence(run)
            self.assertEqual(evidence['elapsed_sec'], 123.25)
            self.assertEqual(evidence['recorded'], saved)
            self.assertEqual(evidence['sha256'], sha256(path))
            for invalid in ({'elapsed_sec': -1}, {'elapsed_sec': float('inf')},
                            {'elapsed_sec': 1, 'duration_sec': 2}):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    report.download_timing_evidence(run)
            path.write_text(json.dumps({'status': 'timing unavailable'}))
            self.assertIsNone(report.download_timing_evidence(run)['elapsed_sec'])

    def test_missing_timing_is_not_invented_and_p95_is_nearest_rank(self):
        missing = report.timing([])
        self.assertIsNone(missing['total_sec'])
        self.assertIsNone(missing['mean_sec'])
        self.assertEqual(report.timing(list(range(1, 21)))['p95_sec'], 19)

    def test_model_loading_peak_is_retained_when_generation_peak_is_lower(self):
        loading = {'gpu_memory': {'0': {'peak_allocated_bytes': 10, 'peak_reserved_bytes': 20}}}
        generated = [{'id': 'x', 'gpu_memory': {'0': {'peak_allocated_bytes': 8, 'peak_reserved_bytes': 12}}}]
        result = report.aggregate_memory(loading, [], generated)
        self.assertEqual(result['devices']['0']['peak_allocated_bytes'], 10)
        self.assertEqual(result['devices']['0']['peak_reserved_bytes'], 20)
        self.assertIsNone(result['devices']['1']['peak_allocated_bytes'])

    def test_final_preservation_hash_and_checkpoint_stat_mismatch_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root / 'new'; source = root / 'old'
            (run / 'audit').mkdir(parents=True)
            old = source / 'original.json'; old.parent.mkdir(); old.write_text('original')
            checkpoint = source / 'adapter.bin'; checkpoint.write_bytes(b'checkpoint')
            snapshot = source / 'why_semantic_review_v1/checkpoint_stat_snapshot.json'
            snapshot.parent.mkdir()
            snapshot.write_text(json.dumps([{'path': str(checkpoint), 'size': 9,
                                            'mtime_ns': checkpoint.stat().st_mtime_ns}]))
            protected = {str(old): sha256(old)}
            (run / 'audit/protected_originals.json').write_text(json.dumps(protected))
            actual = report.audit_preservation(run, {'source_run': str(source)})
            self.assertFalse(actual['passed'])
            self.assertEqual(actual['changed_originals'], [])
            self.assertEqual(len(actual['checkpoint_stat_changed']), 1)
            self.assertEqual(old.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
