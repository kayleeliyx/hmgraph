"""Prepare a separate blinded review only after 228 genuine answers complete.

This module is never imported by the answering runner. It may read references
and saved answers to assemble review material, but never invokes inference,
imports a training framework, or supplies an assistant semantic judgment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import secrets

from .data import (OUT, canonical_bytes, immutable_write, json_bytes, jsonl_bytes,
                   object_sha256, prompt_digest, require, rows, sha256, unique_by_id)
from code.hdepic_experiment.metrics import parse_prediction
from code.hdepic_review_followup.fields import extract_field

STAGES = ('base', 'epoch1', 'epoch2', 'qwen35')
SLOTS = 'ABCD'
HUMAN_FIELDS = ('human_reference_validity', 'human_correct_target_attachment',
                'human_answer_correctness', 'human_unsupported_claims',
                'human_visual_support', 'human_reviewer_notes')
VERSION = 'hdepic_qwen35_four_answer_review_v1'

ADDENDUM = '''# Four-answer review preparation addendum

The accompanying rubric.md is the byte-identical frozen WHY semantic rubric
from the completed three-answer review. Its source, purpose-core, correctness,
unsupported-claim and visual-evidence definitions remain unchanged. This new
packet extends only the operational comparison to four saved answers per case:
the original 8B base, epoch1, epoch2, and an untuned-on-ActGround Qwen3.5-35B-A3B.
The new model's authorized local generation happens before review preparation;
no answer generation or training occurs in the review itself. All 228 validation
Explain IDs and the original fixed50 flags are retained, with 912 answer slots.

Case order and A/B/C/D order are newly randomized with a private saved seed.
The identity key and seed are outside blinded/. Answer styles, earlier context
and this correlated assistant team still limit blinding. These materials contain
no previous assistant ratings or copied source-status labels. Original narration,
reason span and offsets remain unchanged. Proposed source corrections and old
judgments are preserved in their original completed review, not applied here.

Missing/unextractable WHY and generation length/format failures remain visible
and in the full 228 denominator. Malformed NEXT does not invalidate an otherwise
uniquely extractable WHY. Human source/answer fields are blank. No new assistant
judgments are made by this preparation script. Human and assistant review remain
pending until substantive review is separately completed and frozen.

Report all four correctness labels over all 228 and fixed50; keep source-deficient
cases visible and unjudgeable for every answer, separately report the judgeable
denominator, unsupported claims, visual support, paired ordinal transitions and
repeated-narration dependence under the original rubric. Prior selected epoch1
and original NLL/metrics remain unchanged. New-model comparisons supplement the
completed experiment; they do not choose a different training checkpoint.

Contact sheets are copied byte-for-byte review aids from the completed review.
Their printed source review numbers refer to source_case, not the new Q-case
ordering. The model input remains 16 individual cached images at/before query.
'''


class PendingGeneration(RuntimeError):
    """The real generation cohort is incomplete; no review should be created."""


def completed_generation(generation_path, marker_path, manifest_path, prompts):
    """Fail before material writes if answers/provenance are incomplete."""
    generation_path, marker_path = Path(generation_path), Path(marker_path)
    if not generation_path.is_file() or not marker_path.is_file():
        raise PendingGeneration('228 new answers are not complete; no four-answer packet or judgments created')
    marker = json.loads(marker_path.read_text())
    require(marker.get('passed') is True, 'New generation completion marker did not pass')
    require(marker['count'] == len(prompts), 'New generation completion count differs')
    require(marker['generation_sha256'] == sha256(generation_path), 'New generation bytes differ from completed marker')
    require(marker['inputs_manifest_sha256'] == sha256(manifest_path), 'New generation used a different inputs manifest')
    values = rows(generation_path)
    byid = unique_by_id(values, 'new generation')
    require(set(byid) == {prompt['id'] for prompt in prompts}, 'New generation IDs differ from fixed Explain cohort')
    for prompt in prompts:
        row = byid[prompt['id']]
        require(row['status'] == 'completed', f'Incomplete new answer: {row["id"]}')
        require(isinstance(row['prediction'], str), f'New answer is not a saved string: {row["id"]}')
        require(row['row_index'] == prompt['row_index'], f'New answer row index mismatch: {row["id"]}')
        require(row['prompt_sha256'] == prompt['prompt_sha256'] == prompt_digest(prompt['messages'], prompt['images']),
                f'New answer prompt identity mismatch: {row["id"]}')
    return byid


def shuffled_order(values, seed, namespace):
    """Stable keyed ranking; the fresh private seed is never published blinded."""
    return sorted(values, key=lambda value: hashlib.sha256(
        canonical_bytes([str(seed), namespace, value])).hexdigest())


def build_packet(source_cases, predictions, prompts, seed):
    """Reuse source evidence while building genuinely new four-slot materials."""
    source = unique_by_id(source_cases, 'source case')
    prompt_map = unique_by_id(prompts, 'answering prompt')
    require(set(source) == set(prompt_map), 'Source cases and new prompt IDs differ')
    for stage in STAGES:
        require(set(predictions[stage]) == set(source), f'{stage} answers do not cover the fixed cohort')
    cases, identities = [], {}
    order = shuffled_order(list(source), seed, 'case-order')
    for index, rid in enumerate(order, 1):
        original = source[rid]; prompt = prompt_map[rid]
        require(original['input_prompt'] == prompt['messages'], f'Original answering text changed: {rid}')
        require([frame['path'] for frame in original['frames']] == prompt['images'], f'Original image order changed: {rid}')
        case = deepcopy({key: value for key, value in original.items() if key not in ('answers', 'case')})
        case['case'] = f'Q{index:03d}'
        case['source_case'] = original['case']
        stage_order = shuffled_order(list(STAGES), seed, f'answer-order:{rid}')
        case['answers'] = {}
        identities[case['case']] = {'id': rid, 'source_case': original['case'], 'slots': dict(zip(SLOTS, stage_order))}
        for slot, stage in zip(SLOTS, stage_order):
            saved = predictions[stage][rid]
            raw = saved['prediction']; extracted = extract_field(raw, 'why')
            case['answers'][slot] = {
                'why': extracted['text'], 'extraction_status': extracted['status'],
                'field_extraction': extracted, 'raw_response': raw,
                'generation_status': saved['status'], 'finish_reason': saved.get('finish_reason'),
                'generated_tokens': saved.get('generated_tokens'),
                'finish_length': saved.get('finish_reason') == 'length',
                'zero_generated_tokens': saved.get('generated_tokens') == 0,
                'whole_response_format': parse_prediction(raw, 'explain', '<goal>' in prompt['messages'][1]['content']),
            }
        cases.append(case)
    return cases, identities


def human_rows(cases):
    return [{'case': case['case'], 'id': case['id'], 'slot': slot,
             'action_question': case['action_question'], 'original_narration': case['original_narration'],
             'frozen_reference_reason': case['reference_reason'], 'answer_why': answer['why'],
             'contact_sheet': case['contact_sheet'], **{name: '' for name in HUMAN_FIELDS}}
            for case in cases for slot, answer in case['answers'].items()]


def csv_bytes(values):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
    writer.writeheader(); writer.writerows(values)
    return handle.getvalue().encode()


def preserve_human_csv(path, values):
    """Initial fields are blank; subsequent human entries are never rewritten."""
    path = Path(path)
    if not path.exists():
        immutable_write(path, csv_bytes(values))
        return
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(values[0]), 'Existing human CSV schema differs')
        actual = list(reader)
    require(len(actual) == len(values), 'Existing human CSV row count differs')
    for old, expected in zip(actual, values):
        for key, value in expected.items():
            if key not in HUMAN_FIELDS:
                require(old[key] == ('' if value is None else str(value)), f'Existing human CSV identity/content differs: {key}')


def build_html(cases):
    escape = lambda value: html.escape(str(value))
    sections = []
    for case in cases:
        answers = ''.join('<article><h3>Answer ' + slot + '</h3><p>' + escape(answer['why']) +
                          '</p><p>WHY extraction: ' + escape(answer['extraction_status']) +
                          '; finish: ' + escape(answer['finish_reason']) +
                          '</p><details><summary>Saved raw response</summary><pre>' +
                          escape(answer['raw_response']) + '</pre></details></article>'
                          for slot, answer in case['answers'].items())
        sections.append(f"<section id='{case['case']}'><h2>{case['case']} | {escape(case['action_question'])}</h2>"
                        f"<p>Source contact-sheet number: {case['source_case']}; query {case['query_time_sec']:.6f}s</p>"
                        f"<img loading='lazy' src='{escape(case['contact_sheet'])}' alt='All16 causal frames'>"
                        f"<p>Original narration: {escape(case['original_narration'])}</p>"
                        f"<p>Frozen reason [{case['reason_char_start']}:{case['reason_char_end']}]: {escape(case['reference_reason'])}</p>"
                        f"<p>Completed history: {escape('; '.join(h['action'] for h in case['history']))}</p>"
                        f"<div class='answers'>{answers}</div></section>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Four-answer WHY review</title>"
            "<style>body{font:16px system-ui;max-width:1700px;margin:auto;padding:20px}img{max-width:100%}"
            ".answers{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}article{background:#eef2f4;padding:15px}"
            "section{border-top:2px solid #ccd;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            "</head><body><h1>Four-answer validation WHY review</h1><p>228 cases; human and new assistant review pending. "
            "Contact sheets are presentation aids; answering inputs were16 individual JPEGs. Source sheet labels retain "
            "their original review numbers. Partial blinding remains.</p><p><a href='../rubric.md'>Frozen criteria</a> · "
            "<a href='../review_addendum.md'>Operational addendum</a> · <a href='human_review.csv'>Human fields</a></p>" +
            ''.join(sections) + '</body></html>').encode()


def prepare_review(run=OUT):
    run = Path(run).resolve(); review = run / 'review'
    manifest_path = run / 'inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['count'] == 228 and manifest['partition'] == 'validation' and manifest['task'] == 'explain',
            'Review requires the immutable228 validation Explain inputs')
    prompt_path = run / 'inputs/validation_explain.prompts.jsonl'
    require(sha256(prompt_path) == manifest['files_sha256']['inputs/validation_explain.prompts.jsonl'],
            'Prepared answering input bytes changed')
    prompts = rows(prompt_path)
    require(len(prompts) == 228 and [prompt['id'] for prompt in prompts] == manifest['ordered_ids'],
            'Actual answering inputs differ from fixed228 ordered IDs')
    # Crucially, this gate precedes creation of review/ and reading references.
    new = completed_generation(run / 'predictions/generation.jsonl', run / 'predictions/completed.json',
                               manifest_path, prompts)
    source_run = Path(manifest['source_run']).resolve()
    original_review = source_run / 'why_semantic_review_v1'
    old_manifest = json.loads((original_review / 'materials_manifest.json').read_text())
    require(sha256(original_review / 'rubric.md') == old_manifest['rubric_sha256'] == manifest['rubric_sha256'],
            'Original frozen rubric changed')
    require(sha256(original_review / 'blinded/cases.jsonl') == old_manifest['cases_sha256'] == manifest['source_review_cases_sha256'],
            'Original frozen cases changed')
    protected = json.loads((run / 'audit/protected_originals.json').read_text())
    before = {path: sha256(path) for path in protected}
    require(before == protected, 'Protected original data/predictions/reviews changed')
    source_cases = rows(original_review / 'blinded/cases.jsonl')
    require(len(source_cases) == 228 and sum(case['existing_fixed50'] for case in source_cases) == 50,
            'Original228/fixed50 cohort changed')
    expected_ids = {prompt['id'] for prompt in prompts}
    predictions = {'qwen35': new}
    old_generation_hashes = {}
    for stage in STAGES[:-1]:
        path = source_run / 'predictions' / stage / 'generation.jsonl'
        all_rows = unique_by_id(rows(path), stage + ' saved generation')
        require(expected_ids <= set(all_rows), f'Original {stage} missing Explain answers')
        selected = {rid: all_rows[rid] for rid in expected_ids}
        require(all(value['status'] == 'completed' and isinstance(value['prediction'], str) for value in selected.values()),
                f'Original {stage} contains incomplete saved answers')
        predictions[stage] = selected; old_generation_hashes[stage] = sha256(path)
    key_path = review / 'private/identity_key.json'
    if key_path.exists():
        seed = json.loads(key_path.read_text())['private_seed']
    else:
        seed = str(secrets.randbits(256))
    cases, identity = build_packet(source_cases, predictions, prompts, seed)
    unique_frames = {}
    for case in cases:
        for frame in case['frames']:
            require(frame['path'] not in unique_frames or unique_frames[frame['path']] == frame['sha256'], 'Conflicting cached frame SHA')
            unique_frames[frame['path']] = frame['sha256']
    for path, digest in unique_frames.items():
        require(sha256(path) == digest, f'Causal cached frame changed: {path}')
    immutable_write(key_path, json_bytes({'purpose': 'Private randomization seed and model identity map; exclude from blinded material and archives',
                                        'private_seed': seed, 'case_slots': identity}))
    rubric = (original_review / 'rubric.md').read_bytes()
    payloads = {'rubric.md': rubric, 'review_addendum.md': ADDENDUM.encode(),
                'blinded/cases.jsonl': jsonl_bytes(cases), 'blinded/index.html': build_html(cases)}
    sheet_hashes = {}
    for case in cases:
        relative = case['contact_sheet']
        sheet = (original_review / 'blinded' / relative).resolve()
        require(sheet.is_relative_to(original_review / 'blinded'), 'Contact sheet escaped original blinded material')
        content = sheet.read_bytes(); payloads['blinded/' + relative] = content
        sheet_hashes[relative] = hashlib.sha256(content).hexdigest()
    for batch in range(3):
        subset = cases[batch * 76:(batch + 1) * 76]
        payloads[f'blinded/batch{batch + 1}.jsonl'] = jsonl_bytes(subset)
        text = []
        for case in subset:
            text.extend([f"{case['case']} | source {case['source_case']} | {case['narration_id']} | question: {case['action_question']}",
                         'NARRATION: ' + case['original_narration'], 'REFERENCE: ' + case['reference_reason'],
                         'HISTORY: ' + '; '.join(h['action'] for h in case['history']),
                         *[f"{slot}: {answer['why']}" for slot, answer in case['answers'].items()], ''])
        payloads[f'blinded/batch{batch + 1}.txt'] = '\n'.join(text).encode()
    for name, content in payloads.items():
        immutable_write(review / name, content)
    preserve_human_csv(review / 'blinded/human_review.csv', human_rows(cases))
    after = {path: sha256(path) for path in protected}
    require(before == after, 'Original artifacts changed during review preparation')
    material = {'version': VERSION, 'status': 'materials_prepared_review_pending', 'cases': 228, 'answer_slots': 912,
                'fixed50': 50, 'assistant_judgments_created': 0, 'human_review_complete': False,
                'generation_sha256': sha256(run / 'predictions/generation.jsonl'),
                'inputs_manifest_sha256': sha256(manifest_path), 'rubric_sha256': hashlib.sha256(rubric).hexdigest(),
                'old_generation_sha256': old_generation_hashes, 'private_seed_sha256': object_sha256(seed),
                'case_order_and_slots_newly_randomized': True, 'identity_key_outside_blinded': True,
                'original_source_contexts_references_offsets_and_fixed50_exact': True,
                'old_assistant_ratings_copied': False, 'old_identity_key_read': False,
                'human_entries_preserved_on_rerun': True, 'contact_sheets_sha256': sheet_hashes,
                'files_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
                'originals_preserved': before == after, 'protected_original_files': len(protected),
                'inference_run_by_review': False, 'gpu_accessed': False, 'training_run': False, 'test_accessed': False,
                'blinding_limitations': ['Prior experiment context', 'Answer wording/styles may reveal identity',
                                        'One correlated assistant team; no completed independent human review']}
    immutable_write(review / 'materials_manifest.json', json_bytes(material))
    return material


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        result = prepare_review(args.run)
    except PendingGeneration as exc:
        print(json.dumps({'status': 'pending_generation', 'review_prepared': False, 'assistant_judgments_created': 0,
                          'reason': str(exc)}, indent=2))
        return 2
    print(json.dumps({key: result[key] for key in ('status', 'cases', 'answer_slots', 'assistant_judgments_created')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
