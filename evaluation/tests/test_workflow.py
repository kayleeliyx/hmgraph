import copy
import json
import tempfile
import unittest
from pathlib import Path

from hdepic_eval.mapper import ActionMapper
from hdepic_eval.semantics import make_goal_profile, source_is_eligible
from hdepic_eval.workflow import (evaluate_model, import_why_judgment,
    paired_comparison, report_text, unique_rows, judge_packets)


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.mapper = ActionMapper({'take': 0, 'wash': 1, 'put': 2}, {'bowl': 7})
        self.reference = {'id': 'v:explain:1', 'task': 'explain', 'video_id': 'v',
            'participant': 'p', 'query_time_sec': 1.0, 'current_pair': [0, 7],
            'next_pairs': [[0, 7], [1, 7], [2, 7]], 'question': 'take bowl', 'history': [],
            'why': {'eligible': True, 'reference': 'to wash it',
                'original_reference': 'to wash it', 'source_status': 'valid',
                'original_narration': 'Take the bowl to wash it.'},
            'goal': {'eligible': True, 'reference': 'Washing up',
                'profile': make_goal_profile('Washing up'), 'source_status': 'available'}}
        self.prediction = {'id': 'v:explain:1', 'task': 'explain', 'status': 'completed',
            'prediction': '<why>clean the bowl</why><goal>Wash the dishes</goal><next>take bowl;;put bowl</next>',
            'prompt_sha256': 'same'}

    def evaluate(self, predictions=None, reference=None, **kwargs):
        return evaluate_model('fixture', [reference or self.reference],
            {self.prediction['id']: predictions or self.prediction}, self.mapper, **kwargs)

    def test_missing_model_row_is_failure_with_full_denominator(self):
        summary, records, packets, _ = evaluate_model('missing', [self.reference], {}, self.mapper)
        self.assertEqual(summary['all']['next']['position_1']['action']['denominator'], 1)
        self.assertEqual(summary['all']['next']['position_1']['action']['correct'], 0)
        self.assertEqual(summary['all']['why']['counts']['incorrect'], 1)
        self.assertEqual(summary['all']['goal']['correct_rate'], 0)
        self.assertEqual(packets, [])

    def test_partial_judging_never_reports_completed_accuracy(self):
        summary, records, packets, _ = self.evaluate()
        self.assertIsNone(summary['all']['goal']['correct_rate'])
        self.assertEqual(summary['all']['goal']['pending'], 1)
        self.assertEqual(records[0]['predicted_next'], [(0, 7), (None, None), (2, 7)])
        self.assertEqual(len(packets), 2)

    def test_bound_judgments_complete_semantic_metric(self):
        _, _, packets, _ = self.evaluate()
        supplied = {p['packet_id']: {'packet_id': p['packet_id'], 'prompt_sha256': p['prompt_sha256'],
            'judge': 'fixture', 'judgment': {'field': p['field'], 'label': 'correct',
                'justification': 'Equivalent specific purpose or activity.'}} for p in packets}
        summary, _, remaining, _ = self.evaluate(judgments=supplied)
        self.assertEqual(summary['all']['goal']['correct_rate'], 1)
        self.assertEqual(remaining, [])
        supplied[packets[0]['packet_id']]['prompt_sha256'] = 'stale'
        with self.assertRaises(ValueError):
            self.evaluate(judgments=supplied)

    def test_exact_reference_sufficient_but_paraphrases_stay_pending(self):
        p = dict(self.prediction, prediction='<why>to wash it</why><goal>Washing up</goal>')
        summary, _, packets, _ = self.evaluate(predictions=p)
        self.assertEqual(summary['all']['goal']['correct_rate'], 1)
        self.assertEqual(summary['all']['why']['correct_rate'], 1)
        self.assertEqual(packets, [])

    def test_common_source_exclusion_not_model_abstention(self):
        reference = copy.deepcopy(self.reference)
        reference['why']['eligible'] = False
        summary, _, packets, _ = self.evaluate(reference=reference)
        self.assertEqual(summary['all']['why']['eligible'], 0)
        self.assertEqual(summary['all']['why']['excluded_source'], 1)
        self.assertEqual([p['field'] for p in packets], ['goal'])

    def test_failed_generation_cannot_earn_credit_from_partial_text(self):
        summary, _, packets, _ = self.evaluate(predictions=dict(self.prediction, status='failed'))
        self.assertEqual(summary['all']['why']['correct_rate'], 0)
        self.assertEqual(summary['all']['next']['position_3']['action']['correct'], 0)
        self.assertEqual(packets, [])

    def test_unknown_extra_ids_and_duplicate_rows_fail(self):
        with self.assertRaises(ValueError):
            evaluate_model('fixture', [self.reference], {'wrong': self.prediction}, self.mapper)
        with self.assertRaises(ValueError):
            unique_rows([self.prediction, self.prediction])

    def test_old_why_decision_requires_identical_extraction(self):
        old = {'answer_why': 'clean the bowl', 'label': 'correct', 'evidence_note': 'Equivalent purpose.',
            'unsupported_additional_claims': [], 'reviewer': 'old assistant'}
        self.assertIsNotNone(import_why_judgment(old, self.reference, 'clean the bowl'))
        self.assertIsNone(import_why_judgment(old, self.reference, 'wash it'))

    def test_pairs_reject_changed_prompt_and_preserve_missing_rows(self):
        _, left, _, _ = self.evaluate()
        _, right, _, _ = evaluate_model('missing', [self.reference], {}, self.mapper)
        result = paired_comparison(left, right)
        self.assertEqual(result['shared_examples'], 1)
        self.assertEqual(result['by_task']['explain']['after']['next']['position_1']['action']['rate'], 0)
        right[0]['prompt_sha256'] = 'other'
        with self.assertRaises(ValueError):
            paired_comparison(left, right)

    def test_report_keeps_positions_when_sequence_gold_is_incomplete(self):
        reference = copy.deepcopy(self.reference)
        reference['next_pairs'][2][1] = None
        metrics, _, _, _ = self.evaluate(reference=reference)
        summary = {'models': {'fixture': metrics}, 'pending_packet_count': 2}
        report = report_text(summary)
        self.assertIn('100.00%', report)
        self.assertIsNone(metrics['all']['next']['normalized_damerau_osa']['action']['mean'])

    def test_optional_automated_judge_adapter_and_resume(self):
        import sys
        _, _, packets, _ = self.evaluate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'packets.jsonl'
            inputs.write_text(''.join(json.dumps(p) + '\n' for p in packets))
            script = root / 'fixture_judge.py'
            script.write_text('import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({"field":p["field"],"label":"partial","justification":"Fixture decision."}))\n')
            outputs = root / 'new' / 'judgments.jsonl'
            command = [sys.executable, str(script)]
            judge_packets(inputs, outputs, command, 'fixture')
            judge_packets(inputs, outputs, command, 'fixture')
            supplied = {r['packet_id']: r for r in map(json.loads, outputs.read_text().splitlines())}
            self.assertEqual(len(supplied), 2)
            metrics, _, remaining, _ = self.evaluate(judgments=supplied)
            self.assertEqual(metrics['all']['goal']['counts']['partial'], 1)
            self.assertEqual(remaining, [])

    def test_native_annotation_statuses_are_provisionally_eligible(self):
        self.assertTrue(source_is_eligible('goal', 'Washing up', 'available'))
        self.assertTrue(source_is_eligible('why', 'to wash it', 'annotation_derived_pending_review'))

    def test_simple_new_results_do_not_need_training_specific_status(self):
        prediction = {k: v for k, v in self.prediction.items() if k != 'status'}
        metrics, _, _, _ = self.evaluate(predictions=prediction)
        self.assertEqual(metrics['all']['next']['position_1']['action']['rate'], 1)
        self.assertEqual(metrics['diagnostics']['failed_prediction'], 0)

    def test_cached_semantic_judgment_cannot_use_another_reference(self):
        old = {'answer_why': 'clean the bowl', 'label': 'correct', 'evidence_note': 'Equivalent purpose.',
            'reviewer': 'fixture', 'reference_reason': 'another purpose'}
        with self.assertRaises(ValueError):
            import_why_judgment(old, self.reference, 'clean the bowl')

    def test_csv_contains_every_future_position_and_correct_fraction(self):
        import csv
        from hdepic_eval.workflow import export_metrics
        summary, _, _, _ = self.evaluate(predictions=dict(self.prediction,
            prediction='<why>to wash it</why><goal>Washing up</goal><next>take bowl;wash bowl;put bowl</next>'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'metrics.csv'
            export_metrics(output, {'models': {'fixture': summary}})
            with output.open() as handle:
                values = list(csv.DictReader(handle))
            positions = [r for r in values if r['field'] == 'next' and r['metric'] == 'position_accuracy' and r['component'] == 'action']
            self.assertEqual([r['position'] for r in positions], ['1', '2', '3'])
            self.assertTrue(all(float(r['value']) == 1 for r in positions))
            ordinal = next(r for r in values if r['field'] == 'goal' and r['metric'] == 'ordinal_score')
            self.assertEqual((ordinal['numerator'], ordinal['denominator']), ('2', '2'))

    def test_unrecorded_flags_are_not_reported_as_measured_negatives(self):
        old = {'answer_why': 'clean the bowl', 'label': 'correct', 'evidence_note': 'Equivalent purpose.',
            'reviewer': 'fixture', 'unsupported_additional_claims': []}
        summary, _, _, _ = self.evaluate(legacy={self.reference['id']: old})
        score = summary['all']['why']
        self.assertEqual(score['flag_judgment_counts']['unsupported_addition'], 1)
        self.assertEqual(score['flag_judgment_counts']['contradiction'], 0)


if __name__ == '__main__':
    unittest.main()
