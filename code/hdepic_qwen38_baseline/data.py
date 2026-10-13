"""Prepare immutable, prompt-only validation inputs without model inference.

Only the completed experiment's validation Explain cohort is read. References
and source narrations are saved separately from the answering inputs. Importing
this module does not import Torch, inspect GPUs, or load model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

PROJECT = Path('/data/yaxuanli/ActGround')
SOURCE_RUN = PROJECT / 'runs/hdepic_sft_8b_v1'
OUT = PROJECT / 'runs/hdepic_qwen38_27b_baseline_v1'
VERSION = 'hdepic_qwen38_prompt_inputs_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def object_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_by_id(values, description):
    result = {}
    for value in values:
        require(value['id'] not in result, f'Duplicate {description} ID: {value["id"]}')
        result[value['id']] = value
    return result


def prompt_digest(messages, images):
    """Fingerprint the exact text and image-path order passed to the processor."""
    return object_sha256({'messages': messages, 'images': images})


def audit_records(prompts, texts, scoring, *, expected_count=228):
    """Join by original row index and ID; verify source bytes and causal inputs."""
    require(len(prompts) == len(texts) == len(scoring), 'Validation exports differ in length')
    texts_by_id = unique_by_id(texts, 'text')
    unique_by_id(scoring, 'scoring')
    selected = [sidecar for sidecar in scoring if sidecar['task'] == 'explain']
    require(len(selected) == expected_count, f'Expected {expected_count} Explain IDs, got {len(selected)}')
    answering, references, sidecars, evidence = [], [], [], []
    files = {}
    for sidecar in selected:
        rid = sidecar['id']; index = sidecar['row_index']
        require(sidecar['partition'] == 'validation', f'Non-validation sidecar: {rid}')
        require(isinstance(index, int) and 0 <= index < len(prompts), f'Invalid row index: {rid}')
        require(scoring[index]['id'] == rid and texts[index]['id'] == rid,
                f'Original validation order mismatch: {rid}')
        text = texts_by_id[rid]; prompt = prompts[index]
        require(text['task'] == 'explain' and text['partition'] == 'validation',
                f'Wrong text task/partition: {rid}')
        require(set(prompt) == {'messages', 'images'}, f'Unexpected prompt export fields: {rid}')
        require([m['role'] for m in prompt['messages']] == ['system', 'user'],
                f'Assistant/reference supplied to answering model: {rid}')
        require(all(set(m) == {'role', 'content'} and isinstance(m['content'], str)
                    for m in prompt['messages']), f'Unexpected message fields: {rid}')
        require(prompt['messages'][1]['content'].count('<image>') == 16,
                f'Expected 16 individual image markers: {rid}')
        frames = sidecar['frames']; query = text['query_time_sec']
        require(query == sidecar['query_time_sec'], f'Query time mismatch: {rid}')
        require(len(frames) == len(prompt['images']) == 16, f'Image count mismatch: {rid}')
        require(prompt['images'] == [f['path'] for f in frames], f'Image path/order mismatch: {rid}')
        times = [f['timestamp_sec'] for f in frames]
        require(times == sorted(times), f'Frames out of chronological order: {rid}')
        require(all(max(0, query - 5) - 1e-8 <= t <= query for t in times),
                f'Noncausal/out-of-window frame: {rid}')
        history = text['input']['history']
        require(len(history) <= 8 and all(h['end_sec'] <= query for h in history),
                f'Noncausal/oversized history: {rid}')
        current = sidecar['current_action']
        require(current['end_sec'] == query, f'Explain query differs from target endpoint: {rid}')
        reason = text['scoring_metadata']['reason_annotation']
        require(reason['original_narration'][reason['char_start']:reason['char_end']] == text['target']['reason'],
                f'Original reason span mismatch: {rid}')
        for frame in frames:
            actual_pts = float(Fraction(frame['source_pts']) * Fraction(frame['source_time_base']))
            require(actual_pts == frame['timestamp_sec'], f'Source PTS mismatch: {rid}')
            path = frame['path']
            signature = (frame['sha256'], frame['bytes'])
            require(path not in files or files[path] == signature, f'Conflicting frame provenance: {path}')
            files[path] = signature
        # Keep exactly these keys. Source annotation text and targets must never
        # be spread into this object or passed through an inference request.
        answering.append({'id': rid, 'row_index': index,
                          'messages': prompt['messages'], 'images': prompt['images'],
                          'prompt_sha256': prompt_digest(prompt['messages'], prompt['images'])})
        references.append(text)
        sidecars.append(sidecar)
        evidence.append({'id': rid, 'row_index': index, 'video_id': sidecar['video_id'],
                         'participant': sidecar['participant'], 'query_time_sec': query,
                         'history_count': len(history), 'goal_status': sidecar['goal_status'],
                         'prompt_characters': sum(len(m['content']) for m in prompt['messages']),
                         'frames': frames})
    for path, (digest, size) in files.items():
        image = Path(path)
        require(image.is_file() and image.stat().st_size == size, f'Cached image missing/size changed: {path}')
        require(sha256(image) == digest, f'Cached image SHA256 mismatch: {path}')
    return answering, references, sidecars, evidence, len(files)


def select_smoke(evidence, count=4):
    """Choose by prompt length and video diversity, never by target content."""
    require(len(evidence) >= count, 'Not enough validation Explain inputs for smoke selection')
    ranked = sorted(evidence, key=lambda row: (-row['prompt_characters'], row['id']))
    selected = []; videos = set()
    for row in ranked:
        if row['video_id'] not in videos:
            selected.append(row); videos.add(row['video_id'])
            if len(selected) == count:
                break
    for row in ranked:
        if len(selected) == count:
            break
        if row['id'] not in {item['id'] for item in selected}:
            selected.append(row)
    goal_counts = {status: sum(row['goal_status'] == status for row in evidence)
                   for status in sorted({row['goal_status'] for row in evidence})}
    absent = sum(row['goal_status'] != 'available' for row in evidence)
    return {'count': len(selected), 'ids': [row['id'] for row in selected],
            'selection': 'Descending prompt character length, ID tie-break; take distinct videos first, then fill. No reference-label content used.',
            'selected': [{key: row[key] for key in ('id', 'row_index', 'video_id', 'prompt_characters', 'history_count', 'goal_status')}
                         for row in selected],
            'cohort_goal_status_counts': goal_counts, 'cohort_goal_absent_count': absent,
            'goal_absent_coverage': 'unavailable in this fixed Explain cohort' if absent == 0 else 'see selected goal_status',
            'selected_input_role': 'Processor/generation smoke subset only; all 228 IDs remain in the full baseline denominator'}


def protected_paths(project, source_run, config):
    """Enumerate only validation files, saved predictions and completed reviews."""
    paths = {Path(config['data']['validation'][key]) for key in ('text', 'export', 'prompts', 'scoring')}
    paths.update(source_run / name for name in
                 ('frozen_config.json', 'frozen_evaluation.json', 'metrics.json',
                  'validation_report.md', 'execution_report.md'))
    for stage in ('base', 'epoch1', 'epoch2'):
        paths.update(source_run / 'predictions' / stage / name for name in
                     ('generation.jsonl', 'loss.jsonl', 'completed.json'))
    for name in ('why_semantic_review_v1', 'review_followup_v1'):
        directory = source_run / name
        paths.update(path for path in directory.rglob('*') if path.is_file())
        for suffix in ('.zip', '.blinded.zip', '.package.json'):
            path = source_run / (name + suffix)
            if path.is_file():
                paths.add(path)
    # The completed 35B comparison run is immutable; omit its private identity key.
    baseline35 = project / 'runs/hdepic_qwen35_35b_baseline_v1'
    if baseline35.is_dir():
        paths.update(path for path in baseline35.rglob('*')
                     if path.is_file() and 'private' not in path.relative_to(baseline35).parts
                     and path.name != 'pipeline.lock')
    # Identity keys are outside these directories and intentionally not loaded
    # or hashed. Do not enumerate checkpoints, training data, or test data.
    return sorted(path.resolve() for path in paths if path.is_file())


def immutable_write(path, content):
    """Reuse identical output; fail instead of overwriting an incompatible file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f'Immutable prepared output differs: {path}')
        return
    with path.open('xb') as handle:
        handle.write(content)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'


def jsonl_bytes(values):
    return b''.join(canonical_bytes(value) + b'\n' for value in values)


def inherit_frozen_parser(source, out, protected):
    """Inherit the prior baseline's already frozen parser, never redefine it."""
    source = Path(source).resolve(); out = Path(out)
    require(str(source) in protected and sha256(source) == protected[str(source)],
            'Prior parser manifest differs from preparation preservation snapshot')
    value = json.loads(source.read_text())
    for name in ('strict', 'supplementary'):
        require(sha256(value[name + '_parser_source']) == value[name + '_parser_sha256'],
                f'Prior frozen {name} parser implementation changed')
    immutable_write(out / 'frozen_parser.json', source.read_bytes())


def prepare_data(project=PROJECT, source_run=SOURCE_RUN, out=OUT):
    project, source_run, out = map(lambda path: Path(path).resolve(), (project, source_run, out))
    require(source_run != out and not out.is_relative_to(source_run), 'Use a separate new baseline run')
    require(out.is_relative_to(project / 'runs'), 'Prepared outputs must be inside project/runs')
    config_path = source_run / 'frozen_config.json'
    config = json.loads(config_path.read_text())
    review = source_run / 'why_semantic_review_v1'
    completed = json.loads((review / 'completed.json').read_text())
    material = json.loads((review / 'materials_manifest.json').read_text())
    require(completed['cases'] == material['cases'] == 228, 'Completed review cohort is not 228 cases')
    require(sha256(review / 'rubric.md') == material['rubric_sha256'] == completed['rubric_sha256'],
            'Completed semantic rubric changed')
    require(sha256(review / 'blinded/cases.jsonl') == material['cases_sha256'], 'Frozen review cases changed')
    for key in ('text', 'export', 'prompts', 'scoring'):
        path = Path(config['data']['validation'][key]).resolve()
        require(path.is_relative_to(project), f'Validation source outside project: {path}')
        require('test' not in path.name.lower(), f'Forbidden test source: {path}')
        require(sha256(path) == config['data']['validation'][key + '_sha256'], f'Frozen validation source changed: {path}')
    old_paths = protected_paths(project, source_run, config)
    before = {str(path): sha256(path) for path in old_paths}
    immutable_write(out / 'audit/protected_originals.json', json_bytes(before))
    inherit_frozen_parser(project / 'runs/hdepic_qwen35_35b_baseline_v1/frozen_parser.json', out, before)
    prompts = rows(config['data']['validation']['prompts'])
    texts = rows(config['data']['validation']['text'])
    scoring = rows(config['data']['validation']['scoring'])
    answering, references, sidecars, evidence, unique_frames = audit_records(prompts, texts, scoring)
    comparison_prompts = project / 'runs/hdepic_qwen35_35b_baseline_v1/inputs/validation_explain.prompts.jsonl'
    require(jsonl_bytes(answering) == comparison_prompts.read_bytes(),
            'New baseline does not use identical prior 35B model answering inputs')
    review_ids = {case['id'] for case in rows(review / 'blinded/cases.jsonl')}
    require(review_ids == {row['id'] for row in answering}, 'Baseline IDs differ from completed semantic review')
    payloads = {
        'inputs/validation_explain.prompts.jsonl': jsonl_bytes(answering),
        'inputs/smoke_selection.json': json_bytes(select_smoke(evidence)),
        'scoring/validation_explain.references.jsonl': jsonl_bytes(references),
        'scoring/validation_explain.scoring.jsonl': jsonl_bytes(sidecars),
        'scoring/input_alignment.jsonl': jsonl_bytes(evidence),
    }
    source_hashes = {key: config['data']['validation'][key + '_sha256']
                     for key in ('text', 'export', 'prompts', 'scoring')}
    manifest = {
        'version': VERSION, 'project': str(project), 'source_run': str(source_run), 'out': str(out),
        'partition': 'validation', 'task': 'explain', 'count': 228,
        'ordered_ids': [row['id'] for row in answering],
        'ordered_ids_sha256': object_sha256([row['id'] for row in answering]),
        'source_validation_sha256': source_hashes, 'source_frozen_config_sha256': sha256(config_path),
        'source_review_cases_sha256': material['cases_sha256'], 'rubric_sha256': material['rubric_sha256'],
        'frame_references': len(answering) * 16, 'unique_cached_image_files': unique_frames,
        'prompt_only_roles': ['system', 'user'], 'answering_input_schema': ['id', 'row_index', 'messages', 'images', 'prompt_sha256'],
        'answering_fields_passed_to_model': ['messages', 'images'],
        'original_messages_and_image_order_exact': True, 'all_cached_image_bytes_and_sha256_match': True,
        'all_actual_frame_timestamps_causal': True, 'source_pts_time_base_exact': True,
        'completed_histories_causal_and_at_most_eight': True,
        'source_narrations_reasons_targets_only_in_separate_scoring_files': True,
        'goal_absent_records': sum(row['goal_status'] != 'available' for row in evidence),
        'files_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
        'model_loaded': False, 'gpu_accessed': False, 'inference_run': False, 'training_run': False, 'test_accessed': False,
    }
    payloads['inputs/manifest.json'] = json_bytes(manifest)
    for name, content in payloads.items():
        immutable_write(out / name, content)
    after = {str(path): sha256(path) for path in old_paths}
    require(before == after, 'Protected original files changed during preparation')
    preservation = {'passed': True, 'files_checked': len(before), 'before_sha256': before, 'after_sha256': after,
                    'changed_originals': [], 'identity_key_read': False, 'test_accessed': False,
                    'original_validation_predictions_reviews_preserved': True}
    immutable_write(out / 'audit/preparation_preservation.json', json_bytes(preservation))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN)
    parser.add_argument('--out', '--run', type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = prepare_data(args.project, args.source_run, args.out)
    print(json.dumps({key: result[key] for key in
                      ('count', 'frame_references', 'unique_cached_image_files', 'goal_absent_records', 'out')}, indent=2))


if __name__ == '__main__':
    main()
