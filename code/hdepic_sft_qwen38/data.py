"""Frozen HD-EPIC exports and native Qwen3.8 assistant-only SFT inputs.

The answering prompt is built separately from the unchanged assistant target.
This module reads initial-SFT and validation only; it never loads model weights.
CPU preflight caches audits, not tensors, and refuses stale cache identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

from .common import BASELINE, DATASET, MODEL_REVISION, PROJECT, RUN, read_json, sha, write_json
from hdepic_qwen38_baseline.data import canonical_bytes, object_sha256, prompt_digest, require, rows
from hdepic_qwen38_baseline.processor import (
    EOS_IDS, MAX_LENGTH, NONTHINKING_PREFIX, VISION_PATTERN, build_messages,
    expected_serialization, load_processor, preprocess, tensor_sha256,
    verify_individual_image_order,
)

VERSION = 'hdepic_qwen38_native_sft_data_v1'
OLD_CONFIG = PROJECT / 'runs/hdepic_sft_8b_v1/frozen_config.json'
OLD_CONFIG_SHA = 'c9aace48d02e88c46fe79fc422e0953e32ec39138327571cd1fc937a34b14752'
PARTITIONS = {'initial_sft': {'explain': 1403, 'track': 13140},
              'validation': {'explain': 228, 'track': 1981}}
METADATA = BASELINE / 'model_metadata'


def _original_config():
    require(sha(OLD_CONFIG) == OLD_CONFIG_SHA, 'Original experiment configuration changed')
    return read_json(OLD_CONFIG)


def _verify_source_files(config):
    for partition in PARTITIONS:
        item = config['data'][partition]
        for kind in ('export', 'scoring', 'prompts', 'text'):
            require(sha(item[kind]) == item[kind + '_sha256'], f'Frozen {partition} {kind} changed')
    require(sha(DATASET / 'split_manifest.json') == config['split_sha256'], 'Original split changed')
    for item in config['existing_audits'].values():
        require(sha(item['path']) == item['sha256'], f'Original audit changed: {item["path"]}')


def prompt_record(export, rowid, index):
    """Explicit allowlist prevents any scoring field from entering native input."""
    require(set(export) == {'messages', 'images'}, 'Unexpected training export fields')
    messages = export['messages']
    require([m.get('role') for m in messages] == ['system', 'user', 'assistant'], 'Unexpected training roles')
    require(all(set(m) == {'role', 'content'} and isinstance(m['content'], str) for m in messages),
            'Unexpected training message fields')
    require(messages[-1]['content'] and messages[-1]['content'] == messages[-1]['content'].strip(),
            'Empty target or target edge whitespace would be changed by native template')
    require('<|im_' not in messages[-1]['content'] and '<|image_pad|>' not in messages[-1]['content'],
            'Native control token in assistant target')
    prompt = {'id': rowid, 'row_index': index, 'messages': messages[:2], 'images': export['images']}
    prompt['prompt_sha256'] = prompt_digest(prompt['messages'], prompt['images'])
    return prompt


def build_training_inputs(processor, export, rowid, index):
    """Return (batch-of-one CPU model tensors including labels, exact audit).

    Native prompt preprocessing is shared with the completed baseline. The
    native empty thinking prefix is entirely masked. Only the exact assistant
    target and a single im_end are supervised; the template's final transport
    newline is deliberately absent from both inputs and labels.
    """
    import torch
    prompt = prompt_record(export, rowid, index)
    inputs, audit = preprocess(processor, prompt)
    target = export['messages'][-1]['content']
    suffix = target + '<|im_end|>'
    target_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
    require(target_ids and target_ids[-1] == EOS_IDS[0], 'Assistant termination token missing')
    require(target_ids.count(EOS_IDS[0]) == 1, 'Unexpected termination token inside target')
    decode = lambda ids: processor.tokenizer.decode(ids, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
    require(decode(target_ids) == suffix, 'Complete original target did not survive tokenization')
    native_messages = build_messages(prompt) + [
        {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]}]
    full_native = processor.apply_chat_template(native_messages, tokenize=False,
                                               add_generation_prompt=False,
                                               enable_thinking=False, add_vision_id=False)
    expected_full = expected_serialization(prompt) + suffix + '\n'
    require(VISION_PATTERN.sub('<image>', full_native) == expected_full,
            'Native full conversation differs from prompt + complete target + im_end + newline')
    original_ids = inputs['input_ids'][0].tolist()
    expanded_prompt = decode(original_ids)
    require(expanded_prompt.endswith(NONTHINKING_PREFIX), 'Nonthinking prefix incomplete')
    combined_ids = original_ids + target_ids
    # Tests the BPE boundary too: independently tokenizing target must not change
    # any token at the prompt/answer boundary in the complete native serialization.
    require(processor.tokenizer.encode(expanded_prompt + suffix, add_special_tokens=False) == combined_ids,
            'Concatenated training IDs differ from complete native serialized conversation')
    require(len(combined_ids) <= MAX_LENGTH, f'{rowid}: sequence length {len(combined_ids)} exceeds {MAX_LENGTH}')
    prefix_len = len(original_ids)
    result = dict(inputs)
    target_tensor = torch.tensor([target_ids], dtype=inputs['input_ids'].dtype)
    result['input_ids'] = torch.cat([inputs['input_ids'], target_tensor], dim=1)
    result['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(target_tensor)], dim=1)
    result['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(target_tensor)], dim=1)
    result['labels'] = torch.cat([torch.full_like(inputs['input_ids'], -100), target_tensor], dim=1)
    labels = result['labels'][0]
    require(bool((labels[:prefix_len] == -100).all()) and torch.equal(labels[prefix_len:], target_tensor[0]),
            'Loss mask includes prompt/prefix or loses target tokens')
    require(decode(result['input_ids'][0].tolist()) == expanded_prompt + suffix, 'Training input round trip changed')
    require(decode(labels[labels != -100].tolist()) == suffix, 'Supervised text differs from complete target and termination')
    audit.update({
        'training_sequence_tokens': len(combined_ids), 'prompt_tokens_masked': prefix_len,
        'supervised_tokens': len(target_ids), 'target_text_sha256': hashlib.sha256(target.encode()).hexdigest(),
        'target_ids_sha256': tensor_sha256(target_tensor), 'training_ids_sha256': tensor_sha256(result['input_ids']),
        'labels_sha256': tensor_sha256(result['labels']), 'exact_target_survives': True,
        'native_full_conversation_verified': True, 'native_token_boundary_verified': True,
        'loss_only_assistant_target_and_im_end': True, 'system_user_prefix_loss_tokens': 0,
        'required_termination_token': EOS_IDS[0], 'supervised_termination_tokens': 1,
        'trailing_transport_newline_included': False, 'truncation': False,
        'dropped': False, 'mask_policy': 'Original assistant target plus one im_end; all prompt/empty-thinking-prefix tokens -100',
    })
    return result, audit


def load_partition(partition, run=RUN):
    """Aligned (exports, scoring, prompts); only the two authorized partitions."""
    require(partition in PARTITIONS, 'Only initial_sft and validation are permitted')
    manifest = read_json(Path(run) / 'inputs/manifest.json')
    spec = manifest['partitions'][partition]
    result = []
    for kind in ('export', 'scoring', 'prompts'):
        require(sha(spec[kind]) == spec[kind + '_sha256'], f'Prepared source changed: {partition} {kind}')
        result.append(rows(spec[kind]))
    require(all(len(value) == spec['count'] for value in result), 'Prepared source record count changed')
    require([s['id'] for s in result[1]] == spec['ordered_ids'], 'Prepared source ID ordering changed')
    return tuple(result)


class TrainingDataset:
    """Batch-size-one native dataset; use collate_single to avoid extra stacking."""
    def __init__(self, partition, processor, run=RUN):
        self.exports, self.scoring, _ = load_partition(partition, run)
        self.processor = processor

    def __len__(self):
        return len(self.exports)

    def __getitem__(self, index):
        return build_training_inputs(self.processor, self.exports[index], self.scoring[index]['id'], index)[0]


def collate_single(items):
    require(len(items) == 1, 'Native training inputs require per-device batch size one')
    return items[0]


def _audit_alignment(export, sidecar, prompt, index, partition, allowed_videos):
    rid = sidecar['id']; wrapped = prompt_record(export, rid, index)
    require(sidecar['row_index'] == index and sidecar['partition'] == partition, f'Index/partition mismatch: {rid}')
    require(sidecar['video_id'] in allowed_videos, f'Video not in fixed {partition} split: {rid}')
    require(set(prompt) == {'messages', 'images'} and prompt['messages'] == export['messages'][:2]
            and prompt['images'] == export['images'], f'Inference prompt differs from export: {rid}')
    require(len(wrapped['images']) == 16 and wrapped['messages'][1]['content'].count('<image>') == 16,
            f'Input image count changed: {rid}')
    frames = sidecar['frames']; query = sidecar['query_time_sec']
    require(len(frames) == 16 and [f['path'] for f in frames] == export['images'], f'Sidecar image order changed: {rid}')
    times = [f['timestamp_sec'] for f in frames]
    require(times == sorted(times) and all(max(0, query - 5) - 1e-8 <= t <= query for t in times),
            f'Noncausal input frames: {rid}')
    require(all(abs(float(f['source_pts'] * Fraction(f['source_time_base'])) - f['timestamp_sec']) < 1e-8
                and f['width'] == f['height'] == 384 for f in frames), f'Frame timestamp/resolution mismatch: {rid}')
    history = sidecar['history_actions']; current = sidecar['current_action']
    require(len(history) <= 8 and all(h['end_sec'] <= query and h['video_id'] == sidecar['video_id'] for h in history),
            f'Noncausal history: {rid}')
    require(current['narration_id'] not in {h['narration_id'] for h in history}, f'Target occurs in history: {rid}')
    future = sidecar['future_actions']
    require(len(future) == 3 and all(a['start_sec'] > query and a['video_id'] == sidecar['video_id'] for a in future),
            f'Future target timing mismatch: {rid}')
    if sidecar['task'] == 'explain':
        require(current['end_sec'] == query, f'Explain endpoint mismatch: {rid}')
    else:
        require(sidecar['task'] == 'track' and current['start_sec'] <= query < current['end_sec'], f'Track interval mismatch: {rid}')
    target = export['messages'][-1]['content']
    primary = 'why' if sidecar['task'] == 'explain' else 'action'
    fields = [primary] + (['goal'] if sidecar['goal_status'] == 'available' else []) + ['next']
    expected_last_line = 'Return only these fields, in this order: ' + ', '.join(f'<{tag}>' for tag in fields) + '.'
    require(wrapped['messages'][1]['content'].endswith(expected_last_line), f'Requested output fields mismatch: {rid}')
    import re
    require(re.fullmatch('\n'.join(f'<{tag}>[^<>]+</{tag}>' for tag in fields), target) is not None,
            f'Assistant target tags mismatch: {rid}')
    return wrapped


def prepare(run=RUN):
    run = Path(run).resolve(); config = _original_config(); _verify_source_files(config)
    split = read_json(DATASET / 'split_manifest.json')
    manifest = {'version': VERSION, 'dataset': str(DATASET), 'original_config': str(OLD_CONFIG),
                'original_config_sha256': OLD_CONFIG_SHA, 'split_sha256': config['split_sha256'],
                'partitions': {}, 'existing_audits': config['existing_audits'],
                'human_reason_label_review': 'pending', 'labels': 'provisional',
                'natural_task_mixture': True, 'second_stage_exports_read': False, 'test_exports_read': False,
                'original_questions_targets_and_order_unchanged': True, 'source_videos_decoded': False}
    samples = {}; all_ids = set(); frame_inventory = {}
    for partition, task_counts in PARTITIONS.items():
        spec = dict(config['data'][partition]); exports = rows(spec['export'])
        scoring = rows(spec['scoring']); prompts = rows(spec['prompts'])
        require(len(exports) == len(scoring) == len(prompts) == sum(task_counts.values()), f'{partition} count changed')
        ids = [s['id'] for s in scoring]
        require(len(set(ids)) == len(ids) and not (all_ids & set(ids)), 'Duplicate/cross-partition record ID')
        all_ids.update(ids)
        require(object_sha256(ids) == spec['ordered_ids_sha256'], f'{partition} original ID order changed')
        require(dict(Counter(s['task'] for s in scoring)) == task_counts, f'{partition} task counts changed')
        allowed = set(split['partitions'][partition]['video_ids'])
        categories = Counter(); participants = Counter()
        for index, (export, sidecar, prompt) in enumerate(zip(exports, scoring, prompts)):
            wrapped = _audit_alignment(export, sidecar, prompt, index, partition, allowed)
            category = sidecar['task'] + ':' + ('goal_present' if sidecar['goal_status'] == 'available' else 'goal_absent')
            categories[category] += 1; participants[sidecar['participant']] += 1
            if partition == 'initial_sft' and category not in samples:
                samples[category] = {'id': sidecar['id'], 'row_index': index, 'category': category,
                                     'partition': partition, 'prompt_sha256': wrapped['prompt_sha256']}
            for frame in sidecar['frames']:
                identity = (frame['sha256'], frame['bytes'])
                require(frame['path'] not in frame_inventory or frame_inventory[frame['path']] == identity,
                        'Conflicting cached frame identities')
                frame_inventory[frame['path']] = identity
        spec.update(ordered_ids=ids, category_counts=dict(categories), participant_counts=dict(participants))
        manifest['partitions'][partition] = spec
        print(json.dumps({'prepared': partition, 'count': len(ids), 'categories': dict(categories)}), flush=True)
    require(len(samples) == 4, 'Training lacks a requested task/goal category')
    # Hash existing JPEGs, never decode source videos or rebuild the cache.
    for path, (digest, size) in frame_inventory.items():
        require(Path(path).is_absolute() and Path(path).stat().st_size == size and sha(path) == digest,
                f'Frozen cached frame changed: {path}')
    manifest.update(total_records=len(all_ids), unique_cached_images=len(frame_inventory),
                    cached_image_inventory_sha256=object_sha256(frame_inventory),
                    cached_frame_hashes_verified=True)
    write_json(run / 'inputs/manifest.json', manifest, immutable=True)
    write_json(run / 'inputs/smoke_selection.json', {'selection': 'First original-order initial-SFT row per task/goal category',
              'ids': [x['id'] for x in samples.values()], 'records': list(samples.values()),
              'no_validation_or_second_stage_examples': True}, immutable=True)
    return manifest


def _configure_cpu():
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)


def _worker_init(modeldir):
    _configure_cpu()
    global _PROCESSOR
    _PROCESSOR = load_processor(modeldir)


def _process_job(job):
    partition, index, rid, task, goal_status, export = job
    started = time.monotonic()
    try:
        _, audit = build_training_inputs(_PROCESSOR, export, rid, index)
        audit.update(partition=partition, task=task, goal_status=goal_status,
                     elapsed_sec=time.monotonic() - started, status='passed')
        return audit
    except Exception as exc:
        return {'id': rid, 'row_index': index, 'partition': partition, 'task': task,
                'goal_status': goal_status, 'status': 'failed', 'error_type': type(exc).__name__,
                'error': str(exc), 'elapsed_sec': time.monotonic() - started}


def _cache_rows(path, identity):
    """Never repair/truncate a malformed partial cache silently."""
    found = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Incomplete/corrupt audit line {number} at {path}; preserve and investigate') from exc
            require(row.get('preflight_identity') == identity, 'Audit cache identity differs from current code/data/processor')
            require(row['id'] not in found, 'Duplicate cached preflight ID')
            found[row['id']] = row
    return found


def preflight(run=RUN, workers=4, modeldir=METADATA):
    run = Path(run).resolve(); modeldir = Path(modeldir).resolve()
    require(1 <= workers <= 8, 'CPU worker count must be between one and eight')
    _configure_cpu(); started = time.monotonic()
    manifest_path = run / 'inputs/manifest.json'; manifest = read_json(manifest_path)
    require(manifest['version'] == VERSION and manifest['total_records'] == 16752, 'Unexpected prepared cohort')
    _verify_source_files(_original_config())
    package_versions = {name: importlib.metadata.version(name) for name in
                        ('torch', 'torchvision', 'transformers', 'tokenizers', 'Pillow', 'numpy')}
    source_files = [Path(__file__).resolve(), PROJECT / 'code/hdepic_qwen38_baseline/processor.py',
                    PROJECT / 'code/hdepic_qwen38_baseline/data.py']
    identity_config = {'version': VERSION, 'manifest_sha256': sha(manifest_path), 'model_revision': MODEL_REVISION,
                       'packages': package_versions, 'source_sha256': {str(p): sha(p) for p in source_files},
                       'processor_metadata_sha256': {p.name: sha(p) for p in sorted(modeldir.iterdir()) if p.is_file()},
                       'max_length': MAX_LENGTH, 'mask_policy': 'Original assistant target plus im_end; entire native prompt masked'}
    identity = object_sha256(identity_config)
    folder = run / 'preflight'; folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'identity.json', dict(identity_config, preflight_identity=identity), immutable=True)
    audits_path = folder / 'records.jsonl'; cached = _cache_rows(audits_path, identity)
    require(all(v['status'] == 'passed' for v in cached.values()), 'Cached failed records require investigation before resume')
    cohort = {part: load_partition(part, run) for part in PARTITIONS}
    processor = load_processor(modeldir)
    sample_audits = []
    for sample in read_json(run / 'inputs/smoke_selection.json')['records']:
        index = sample['row_index']; export = cohort['initial_sft'][0][index]
        tensors, audit = build_training_inputs(processor, export, sample['id'], index)
        audit['individual_image_order_check'] = verify_individual_image_order(
            processor, prompt_record(export, sample['id'], index), tensors)
        audit['category'] = sample['category']; sample_audits.append(audit)
    write_json(folder / 'samples.json', {'preflight_identity': identity, 'passed': True, 'records': sample_audits}, immutable=True)
    del processor
    print(json.dumps({'sample_preflight': 'passed', 'categories': [s['category'] for s in sample_audits],
                      'native_full_conversation_verified': True, 'bulk_cpu_workers': workers}), flush=True)
    jobs = []; expected_ids = []
    for partition, (exports, scoring, _) in cohort.items():
        for index, (export, sidecar) in enumerate(zip(exports, scoring)):
            rid = sidecar['id']; expected_ids.append(rid)
            if rid not in cached:
                jobs.append((partition, index, rid, sidecar['task'], sidecar['goal_status'], export))
    require(set(cached) <= set(expected_ids), 'Cached records outside frozen cohort')
    initial_cached_count = len(cached)
    # A bounded submission window avoids retaining native image tensors or
    # thousands of export copies in the multiprocessing result queue.
    if jobs:
        from concurrent.futures import wait, FIRST_COMPLETED
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init, initargs=(str(modeldir),)) as pool:
            iterator = iter(jobs); pending = set()
            for _ in range(min(workers * 2, len(jobs))):
                pending.add(pool.submit(_process_job, next(iterator)))
            with audits_path.open('a') as handle:
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        audit = future.result(); audit['preflight_identity'] = identity
                        handle.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush()
                        cached[audit['id']] = audit
                        if audit['status'] != 'passed':
                            for other in pending:
                                other.cancel()
                            write_json(folder / 'failure.json', audit)
                            raise ValueError(f'Preprocessing failed; no record discarded: {audit["id"]}: {audit["error"]}')
                        if len(cached) % 100 == 0:
                            print(json.dumps({'processed_retained': len(cached), 'total': len(expected_ids),
                                              'elapsed_sec': round(time.monotonic() - started, 2)}), flush=True)
                        job = next(iterator, None)
                        if job is not None:
                            pending.add(pool.submit(_process_job, job))
    require(set(cached) == set(expected_ids) and len(cached) == 16752, 'Incomplete full-data native preprocessing audit')
    summary = {'preflight_identity': identity, 'passed': True, 'processed': len(cached), 'retained': len(cached),
               'failed': 0, 'dropped': 0, 'truncated': 0, 'offending_ids': [], 'partitions': {},
               'max_sequence_length': max(x['training_sequence_tokens'] for x in cached.values()),
               'min_sequence_length': min(x['training_sequence_tokens'] for x in cached.values()),
               'all_16_individual_images_retained': True, 'all_images_384px_and_144_tokens': True,
               'all_targets_complete': True, 'all_native_full_conversations_verified': True,
               'all_loss_masks_assistant_target_and_im_end_only': True,
               'supervised_system_user_prefix_tokens': 0, 'record_audits_sha256': sha(audits_path),
               'human_reason_label_review': 'pending', 'gpu_used': False}
    for partition, tasks in PARTITIONS.items():
        values = [x for x in cached.values() if x['partition'] == partition]
        require(dict(Counter(x['task'] for x in values)) == tasks, 'Audited task count mismatch')
        summary['partitions'][partition] = {'processed': len(values), 'retained': len(values), 'task_counts': tasks,
                                          'max_sequence_length': max(x['training_sequence_tokens'] for x in values),
                                          'max_supervised_tokens': max(x['supervised_tokens'] for x in values)}
    write_json(folder / 'summary.json', summary, immutable=True)
    invocation = {'preflight_identity': identity, 'started_unix': time.time() - (time.monotonic() - started),
                  'elapsed_sec': time.monotonic() - started, 'workers': workers,
                  'reused_records': initial_cached_count, 'new_records': len(jobs), 'passed': True}
    write_json(folder / f'invocation_{time.time_ns()}.json', invocation)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'preflight'), required=True)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--modeldir', type=Path, default=METADATA)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run)
    else:
        preflight(args.run, args.workers, args.modeldir)


if __name__ == '__main__':
    main()
