"""Strict, resumable CPU audit of every frozen SFT and validation record.

Runs the installed SwiftSft loader, LazyLLMDataset encoder and training collator.
No model tensors are loaded; a meta architecture supplies multimodal positions.
Every original record is checked. An error stops the preflight and training gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import time
import traceback
from unittest.mock import patch

PARTITIONS = ('initial_sft', 'validation')
EXPECTED_COUNTS = {'initial_sft': 14543, 'validation': 2209}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def index_lines(path):
    offsets = []
    with Path(path).open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            assert line.strip(), f'Blank dataset line: {path}:{len(offsets) + 1}'
            offsets.append(offset)
    return offsets


def load_at(handle, offsets, index):
    handle.seek(offsets[index])
    line = handle.readline()
    return json.loads(line), hashlib.sha256(line).hexdigest()


def configure_cpu(config):
    project = Path(config['paths']['project'])
    processor = config['processor']
    values = {
        'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'USE_HF': '1',
        'HF_HOME': str(project / '.cache/huggingface'),
        'HF_DATASETS_CACHE': str(project / '.cache/hdepic_swift_dataset'),
        'MODELSCOPE_CACHE': str(project / '.cache/modelscope'),
        'SWIFT_CACHE_DIR': str(project / '.cache/swift'),
        'IMAGE_MAX_TOKEN_NUM': str(processor['image_max_token_num']),
        'IMAGE_MIN_TOKEN_NUM': str(processor['image_min_token_num']),
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
    }
    os.environ.update(values)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'NPROC_PER_NODE'):
        os.environ.pop(key, None)


def audit_validation_batch(batch, row, scoring, template):
    """Same smoke-auditor checks, extended truthfully to validation/P07."""
    import torch
    assert scoring['partition'] == 'validation' and scoring['participant'] == 'P07'
    tokenizer = template.tokenizer
    ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    assert len(ids) == len(labels) <= template.max_length
    assert batch['attention_mask'][0].bool().all().item()
    grid = batch['image_grid_thw']
    assert tuple(grid.shape) == (16, 3)
    image_tokens = grid.prod(dim=1) // template.processor.image_processor.merge_size ** 2
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    starts = (ids == start_id).nonzero().reshape(-1).tolist()
    ends = (ids == end_id).nonzero().reshape(-1).tolist()
    assert len(starts) == len(ends) == 16
    assert int((ids == image_id).sum()) == int(image_tokens.sum())
    assert batch['pixel_values'].shape[0] == int(grid.prod(dim=1).sum())
    for start, end, count in zip(starts, ends, image_tokens.tolist()):
        assert end - start - 1 == count
        assert (ids[start + 1:end] == image_id).all().item()
        assert (labels[start:end + 1] == -100).all().item()
    supervised = labels != -100
    positions = supervised.nonzero().reshape(-1)
    assert positions.numel() > 0
    first = int(positions[0])
    assert torch.equal(positions, torch.arange(first, len(ids)))
    assert torch.equal(labels[supervised], ids[supervised])
    assert (labels[:first] == -100).all().item()
    target = row['messages'][-1]['content']
    decoded = tokenizer.decode(labels[supervised].tolist(), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    terminator = decoded[len(target):] if decoded.startswith(target) else None
    assert terminator in ('<|im_end|>', '<|im_end|>\n'), (scoring['id'], decoded, target)
    prompt = ''.join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                     for message in row['messages'][:2]) + '<|im_start|>assistant\n'
    full = tokenizer.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    masked = tokenizer.decode(ids[:first].tolist(), skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    pattern = r'<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>'
    assert re.sub(pattern, '<image>', full) == prompt + target + terminator
    assert re.sub(pattern, '<image>', masked) == prompt
    assert ('<goal>' in target) == (scoring['goal_status'] == 'available')
    primary = 'why' if scoring['task'] == 'explain' else 'action'
    assert target.startswith(f'<{primary}>') and target.endswith('</next>')
    return {'input_tokens': len(ids), 'image_tokens': int(image_tokens.sum()),
            'image_tokens_per_image': image_tokens.tolist(), 'pixel_values_shape': list(batch['pixel_values'].shape),
            'masked_prompt_tokens_including_assistant_header': first,
            'supervised_tokens_including_assistant_terminator': int(supervised.sum()),
            'permitted_supervised_assistant_terminator': terminator,
            'complete_target': True, 'assistant_only_loss': True,
            'system_user_image_and_assistant_header_masked': True}


def audit_record(batch, row, scoring, prompt_only, template, partition, index, video_ids):
    import torch
    assert set(row) == {'messages', 'images'}, (scoring['id'], 'Scoring data entered training input')
    assert set(prompt_only) == {'messages', 'images'}
    assert prompt_only == {'messages': row['messages'][:2], 'images': row['images']}
    assert [message['role'] for message in row['messages']] == ['system', 'user', 'assistant']
    assert all(isinstance(message['content'], str) for message in row['messages'])
    assert scoring['partition'] == partition and scoring['row_index'] == index
    assert scoring['video_id'] in video_ids[partition]
    images = row['images']
    assert len(images) == len(set(images)) == row['messages'][1]['content'].count('<image>') == 16
    frames = scoring['frames']
    assert len(frames) == 16 and [frame['path'] for frame in frames] == images
    assert all(Path(path).is_file() for path in images)
    times = [frame['timestamp_sec'] for frame in frames]
    assert times == sorted(times) and all(t <= scoring['query_time_sec'] for t in times)
    history, future = scoring['history_actions'], scoring['future_actions']
    assert len(history) <= 8 and len(future) == 3
    assert all(action['end_sec'] <= scoring['query_time_sec'] for action in history)
    assert all(action['start_sec'] > scoring['query_time_sec'] for action in future)
    assert all(action['video_id'] == scoring['video_id'] for action in history + future)
    if scoring['task'] == 'explain':
        assert scoring['current_action']['narration_id'] not in {action['narration_id'] for action in history}
        assert scoring['current_action']['end_sec'] == scoring['query_time_sec']
    else:
        assert scoring['current_action']['start_sec'] <= scoring['query_time_sec'] < scoring['current_action']['end_sec']
    assert all(not isinstance(value, torch.Tensor) or value.device.type == 'cpu' for value in batch.values())
    if partition == 'initial_sft':
        from code.hdepic_preparation.swift_smoke_audit import audit_collated_record
        result = audit_collated_record(batch, row, scoring, template)
    else:
        result = audit_validation_batch(batch, row, scoring, template)
    answer_ids = template.tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False)
    suffix_ids = []
    for part in template.template_meta.suffix:
        suffix_ids += template.tokenizer.encode(part, add_special_tokens=False) if isinstance(part, str) else part
    labels = batch['labels'][0]
    assert labels[labels != -100].tolist() == answer_ids + suffix_ids
    assert batch['image_grid_thw'].shape[0] == 16
    return {key: value for key, value in result.items() if key in {
        'input_tokens', 'image_tokens', 'image_tokens_per_image', 'pixel_values_shape',
        'masked_prompt_tokens_including_assistant_header', 'supervised_tokens_including_assistant_terminator',
        'permitted_supervised_assistant_terminator', 'complete_target', 'assistant_only_loss',
        'system_user_image_and_assistant_header_masked'}}


def read_cache(path, cache_key):
    completed = {}
    if not path.exists():
        return completed
    with path.open('rb+') as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(position)
                break
            record = json.loads(line)
            assert record['cache_key'] == cache_key
            assert record['global_index'] not in completed
            completed[record['global_index']] = record
    return completed


def worker(worker_id, workers, config, indexed, cache_key, cache_dir, video_ids, events):
    log_path = Path(cache_dir) / f'worker_{worker_id}.swift.log'
    try:
        with log_path.open('a') as log, redirect_stdout(log), redirect_stderr(log):
            worker_body(worker_id, workers, config, indexed, cache_key, cache_dir, video_ids, events)
    except BaseException as error:
        events.put({'worker_id': worker_id, 'event': 'failed', 'error': str(error),
                    'traceback': traceback.format_exc(), 'log': str(log_path)})
        raise


def worker_body(worker_id, workers, config, indexed, cache_key, cache_dir, video_ids, events):
    configure_cpu(config)
    import torch
    import safetensors
    import safetensors.torch
    import transformers.modeling_utils
    from swift.llm import TrainArguments
    from swift.llm.dataset import LazyLLMDataset
    from swift.llm.train import SwiftSft
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    settings = config['processor']
    path = Path(cache_dir) / f'worker_{worker_id}.jsonl'
    completed = read_cache(path, cache_key)
    assigned = list(range(worker_id, sum(EXPECTED_COUNTS.values()), workers))
    assert set(completed) <= set(assigned)
    events.put({'worker_id': worker_id, 'event': 'progress', 'completed': len(completed), 'reused': len(completed)})
    if len(completed) == len(assigned):
        events.put({'worker_id': worker_id, 'event': 'done', 'completed': len(completed), 'reused': len(completed)})
        return

    class ProcessorOnlySft(SwiftSft):
        def _prepare_model_tokenizer(self, **kwargs):
            super()._prepare_model_tokenizer(load_model=False, **kwargs)
            assert self.model is None

    arguments = TrainArguments(
        model=config['model']['path'], model_type='qwen3_vl', torch_dtype='bfloat16',
        dataset=[config['data']['initial_sft']['export']],
        val_dataset=[config['data']['validation']['export']], split_dataset_ratio=0.0,
        dataset_shuffle=False, val_dataset_shuffle=False, lazy_tokenize=True, strict=True,
        max_length=settings['max_length'], max_pixels=settings['max_pixels'],
        truncation_strategy='delete', loss_scale='default',
        train_type='lora', lora_rank=16, lora_alpha=32, freeze_vit=True, freeze_aligner=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        gradient_checkpointing=True, packing=False, padding_free=False,
        dataset_num_proc=1, dataloader_num_workers=0, output_dir=str(Path(cache_dir) / f'unused_{worker_id}'),
        report_to=[], use_cpu=True, bf16=False, fp16=False, use_hf=True, attn_impl='sdpa', add_version=False,
    )

    def refuse_weights(*unused_args, **unused_kwargs):
        raise RuntimeError('CPU preflight forbids checkpoint tensor loading')

    with ExitStack() as stack:
        for module, attribute in [(torch, 'load'), (safetensors, 'safe_open'),
                                  (safetensors.torch, 'load_file'), (safetensors.torch, 'safe_open'),
                                  (transformers.modeling_utils, 'safe_open')]:
            stack.enter_context(patch.object(module, attribute, side_effect=refuse_weights))
        handles = {partition: {key: stack.enter_context(Path(config['data'][partition][key]).open('rb'))
                               for key in ('export', 'scoring', 'prompts')} for partition in PARTITIONS}
        pipeline = ProcessorOnlySft(arguments)
        train_dataset, val_dataset = pipeline._prepare_dataset()
        datasets = {'initial_sft': train_dataset, 'validation': val_dataset}
        assert all(isinstance(datasets[partition], LazyLLMDataset) and
                   len(datasets[partition]) == EXPECTED_COUNTS[partition] for partition in PARTITIONS)
        assert pipeline.template.mode == 'train' and pipeline.template.loss_scale.base_strategy == 'default'
        assert pipeline.processor.global_vars['image_min_token_num'] == settings['image_min_token_num']
        assert pipeline.processor.global_vars['image_max_token_num'] == settings['image_max_token_num']
        collator = pipeline._get_data_collator()
        count = len(completed)
        with path.open('a') as output:
            for global_index in assigned:
                if global_index in completed:
                    continue
                partition = 'initial_sft' if global_index < EXPECTED_COUNTS['initial_sft'] else 'validation'
                index = global_index if partition == 'initial_sft' else global_index - EXPECTED_COUNTS['initial_sft']
                row, row_hash = load_at(handles[partition]['export'], indexed[partition]['export'], index)
                scoring, scoring_hash = load_at(handles[partition]['scoring'], indexed[partition]['scoring'], index)
                prompt, prompt_hash = load_at(handles[partition]['prompts'], indexed[partition]['prompts'], index)
                try:
                    batch = collator([datasets[partition][index]])
                    result = audit_record(batch, row, scoring, prompt, pipeline.template, partition, index, video_ids)
                except BaseException:
                    events.put({'worker_id': worker_id, 'event': 'offending_record', 'id': scoring['id'],
                                'partition': partition, 'row_index': index, 'traceback': traceback.format_exc()})
                    raise
                record = {'cache_key': cache_key, 'global_index': global_index, 'row_index': index,
                          'id': scoring['id'], 'partition': partition, 'participant': scoring['participant'],
                          'task': scoring['task'], 'goal_status': scoring['goal_status'],
                          'source_narration_id': scoring['current_action']['narration_id'],
                          'history_source_ids': [action['narration_id'] for action in scoring['history_actions']],
                          'future_source_ids': [action['narration_id'] for action in scoring['future_actions']],
                          'source_export_row_sha256': row_hash, 'source_scoring_row_sha256': scoring_hash,
                          'source_prompt_row_sha256': prompt_hash, 'image_count': 16, **result}
                output.write(json.dumps(record, separators=(',', ':')) + '\n')
                output.flush()
                count += 1
                if count % 100 == 0:
                    events.put({'worker_id': worker_id, 'event': 'progress', 'completed': count,
                                'reused': len(completed), 'last_id': scoring['id']})
        assert pipeline.template.dummy_model is not None
        assert all(parameter.device.type == 'meta' for parameter in pipeline.template.dummy_model.parameters())
        assert not torch.cuda.is_initialized()
        events.put({'worker_id': worker_id, 'event': 'done', 'completed': count, 'reused': len(completed),
                    'weights_loaded': False, 'cuda_initialized': False})


def fingerprint(config, config_path, workers):
    import importlib.util
    swift = Path(importlib.util.find_spec('swift').origin).parent
    helper = Path(__file__).parents[1] / 'hdepic_preparation/swift_smoke_audit.py'
    # Evaluation-parser/training-schedule changes do not invalidate the CPU
    # image/token audit. Its key includes only inputs, preprocessing and code.
    files = [Path(__file__), helper]
    for partition in PARTITIONS:
        files.extend(Path(config['data'][partition][key]) for key in ('export', 'scoring', 'prompts'))
    files.append(Path(config['paths']['dataset']) / 'split_manifest.json')
    model = Path(config['model']['path'])
    files.extend(model / name for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                                          'tokenizer.json', 'chat_template.json'))
    files.extend(swift / name for name in ('llm/train/sft.py', 'llm/dataset/utils.py', 'llm/dataset/loader.py',
                                          'llm/template/base.py', 'llm/template/template/qwen.py',
                                          'llm/model/register.py', 'llm/model/model/qwen.py'))
    identities = {str(path.resolve()): digest(path) for path in files}
    versions = {name: importlib.metadata.version(name) for name in
                ('ms-swift', 'transformers', 'torch', 'torchvision', 'peft', 'accelerate', 'datasets', 'qwen-vl-utils')}
    manifest = {'files_sha256': identities, 'versions': versions, 'workers': workers,
                'processor': {key: config['processor'][key] for key in
                              ('image_min_token_num', 'image_max_token_num', 'max_pixels', 'max_length')},
                'model_revision': config['model']['revision'],
                'expected_counts': EXPECTED_COUNTS}
    return json_digest(manifest), manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/data/yaxuanli/ActGround/runs/hdepic_sft_8b_v1'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    assert 1 <= args.workers <= 32
    config_path = args.config or args.run / 'frozen_config.json'
    from .common import load_config
    config = load_config(config_path, verify=True)
    assert all(config['data'][partition]['count'] == EXPECTED_COUNTS[partition] for partition in PARTITIONS)
    configure_cpu(config)
    assert config['processor']['image_min_token_num'] == 64
    assert config['processor']['image_max_token_num'] == 144
    assert config['processor']['max_pixels'] == 147456 and config['processor']['max_length'] == 8192
    output = args.run / 'preflight'
    output.mkdir(parents=True, exist_ok=True)
    cache_key, provenance = fingerprint(config, config_path, args.workers)
    cache_dir = output / 'cache' / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(cache_dir / 'provenance.json', provenance)
    indexed = {partition: {key: index_lines(config['data'][partition][key])
                           for key in ('export', 'scoring', 'prompts')} for partition in PARTITIONS}
    assert all(len(offsets) == EXPECTED_COUNTS[partition] for partition, files in indexed.items()
               for offsets in files.values())
    split = json.loads((Path(config['paths']['dataset']) / 'split_manifest.json').read_text())
    video_ids = {partition: set(split['partitions'][partition]['video_ids']) for partition in PARTITIONS}
    context = mp.get_context('spawn')
    events = context.Queue()
    processes = [context.Process(target=worker, args=(index, args.workers, config, indexed, cache_key,
                                                     str(cache_dir), video_ids, events)) for index in range(args.workers)]
    started = time.perf_counter()
    for process in processes:
        process.start()
    progress = {index: {'completed': 0, 'reused': 0} for index in range(args.workers)}
    done, failures = set(), []
    try:
        while len(done) < args.workers:
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                dead = [index for index, process in enumerate(processes)
                        if process.exitcode is not None and process.exitcode != 0]
                if dead:
                    raise RuntimeError(f'Preflight workers exited unsuccessfully: {dead}')
                continue
            worker_id = event['worker_id']
            if event['event'] in ('failed', 'offending_record'):
                failures.append(event)
                raise RuntimeError(f'Preflight stopped on worker {worker_id}: {event.get("id", event.get("error"))}')
            progress[worker_id] = event
            if event['event'] == 'done':
                done.add(worker_id)
            status = {'cache_key': cache_key, 'completed': sum(row['completed'] for row in progress.values()),
                      'reused': sum(row.get('reused', 0) for row in progress.values()),
                      'expected': sum(EXPECTED_COUNTS.values()), 'workers_finished': len(done),
                      'elapsed_sec': time.perf_counter() - started}
            atomic_json(output / 'progress.json', status)
            print(json.dumps(status), flush=True)
        for process in processes:
            process.join()
            assert process.exitcode == 0
    except BaseException as error:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        failure_report = {'passed': False, 'error': str(error), 'offending_records': failures,
                          'cache_key': cache_key, 'training_must_not_start': True,
                          'elapsed_sec': time.perf_counter() - started}
        atomic_json(output / 'preflight_audit.json', failure_report)
        raise

    records = {}
    for index in range(args.workers):
        for global_index, record in read_cache(cache_dir / f'worker_{index}.jsonl', cache_key).items():
            assert global_index not in records
            records[global_index] = record
    assert set(records) == set(range(sum(EXPECTED_COUNTS.values())))
    ordered = [records[index] for index in sorted(records)]
    assert len({row['id'] for row in ordered}) == len(ordered)
    groups = Counter((row['partition'], row['task'], row['goal_status'] == 'available') for row in ordered)
    # A partition may naturally contain only supervised goals. Compare with
    # frozen source records instead of requiring nonexistent task/goal cells.
    source_groups, source_statuses = Counter(), Counter()
    for partition in PARTITIONS:
        with Path(config['data'][partition]['scoring']).open() as handle:
            for line in handle:
                scoring = json.loads(line)
                source_groups[(partition, scoring['task'], scoring['goal_status'] == 'available')] += 1
                source_statuses[(partition, scoring['task'], scoring['goal_status'])] += 1
    assert groups == source_groups, 'Audited task/goal categories differ from frozen source records'
    assert Counter((row['partition'], row['task'], row['goal_status']) for row in ordered) == source_statuses
    counts = Counter(row['partition'] for row in ordered)
    assert dict(counts) == EXPECTED_COUNTS
    records_path = output / 'preflight_records.jsonl'
    temporary = records_path.with_name(records_path.name + '.tmp')
    with temporary.open('w') as handle:
        for record in ordered:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    temporary.replace(records_path)
    summary = {'passed': True, 'cache_key': cache_key, 'records_verified': len(ordered),
               'partition_counts': dict(counts), 'images_verified': len(ordered) * 16,
               'task_goal_counts': {f'{partition}:{task}:goal_{goal}': count
                                    for (partition, task, goal), count in sorted(groups.items())},
               'task_goal_annotation_status_counts': {f'{partition}:{task}:{status}': count
                                                      for (partition, task, status), count in sorted(source_statuses.items())},
               'sequence_token_range': [min(row['input_tokens'] for row in ordered),
                                        max(row['input_tokens'] for row in ordered)],
               'max_length': 8192, 'all_images_preserved': True, 'all_targets_complete': True,
               'assistant_only_loss': True, 'prompt_only_exports_exact': True,
               'scoring_metadata_not_in_input': True, 'original_order_and_source_ids_preserved': True,
               'skipped_or_substituted_records': 0, 'weights_loaded': False, 'cuda_initialized': False,
               'position_model': 'meta architecture only', 'workers': args.workers,
               'cpu_threads_per_worker': 1, 'elapsed_sec': time.perf_counter() - started,
               'reused_records': sum(row.get('reused', 0) for row in progress.values()),
               'record_audit': str(records_path), 'record_audit_sha256': digest(records_path),
               'provenance': provenance,
               'frozen_config_sha256_at_preflight': digest(config_path),
               'no_test_or_second_stage_records_processed': True}
    atomic_json(output / 'preflight_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
