"""Frozen, resumable local Qwen3-VL validation generation and target NLL.

The installed SwiftInfer/PtEngine runs prompt-only greedy generation. A separate
installed training template supplies complete assistant-only target masks for
teacher-forced NLL on the same loaded model. No test/second-stage input is read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from .common import (GPU_UUIDS, RUN, atomic_json, atomic_jsonl, configure_environment,
                     load_config, load_records, read_json, sha256, stable_hash)
from .metrics import parse_prediction, protocol


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_line_hashes(path):
    with Path(path).open('rb') as handle:
        return [hashlib.sha256(line).hexdigest() for line in handle if line.strip()]


def journal_rows(path, identity, record_keys, retry_errors=False):
    """Reuse exact successful rows; repair only an incomplete final write."""
    path = Path(path)
    rows, errors = {}, []
    if not path.exists():
        return rows
    with path.open('rb+') as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(position)
                break
            row = json.loads(line)
            assert row['run_key'] == identity, f'Incompatible cached result: {path}'
            key = row['id']
            assert key in record_keys and row['record_key'] == record_keys[key], key
            assert key not in rows, f'Duplicate result ID: {key}'
            rows[key] = row
            if row['status'] != 'completed':
                errors.append(row)
    if errors:
        if not retry_errors:
            raise RuntimeError(f'Journal has {len(errors)} failed rows; inspect errors before explicit --retry-errors: {path}')
        archive = path.with_name(path.name + '.errors.' + str(time.time_ns()) + '.jsonl')
        atomic_jsonl(archive, errors)
        rows = {key: row for key, row in rows.items() if row['status'] == 'completed'}
        atomic_jsonl(path, sorted(rows.values(), key=lambda row: row['row_index']))
    return rows


def append_row(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_identity(config, stage, checkpoint):
    """Verify original checkpoint bytes and identify a full-run saved adapter."""
    model_dir = Path(config['model']['path']).resolve()
    original = read_json(config['model']['identity_source'])
    assert original['revision'] == config['model']['revision']
    files = {}
    for info in original['files']:
        path = model_dir / info['filename']
        assert path.is_file() and path.stat().st_size == info['bytes']
        actual = sha256(path)
        assert actual == info['sha256'], f'Original model file changed: {path}'
        files[info['filename']] = actual
    assert len([name for name in files if name.startswith('model-') and name.endswith('.safetensors')]) == 4
    result = {'repo_id': config['model']['repo_id'], 'revision': config['model']['revision'],
              'original_path': str(model_dir), 'original_files_sha256': files,
              'original_checkpoint_identity_sha256': sha256(config['model']['identity_source']),
              'stage': stage, 'adapter': None}
    if stage == 'base':
        assert checkpoint is None, 'Base predictions must use the original unadapted 8B checkpoint'
    else:
        assert checkpoint is not None, f'{stage} requires its saved full-run checkpoint'
        checkpoint = Path(checkpoint).resolve()
        assert checkpoint.is_relative_to(Path(config['paths']['run']).resolve())
        assert 'smoke' not in str(checkpoint).lower(), 'Disposable smoke adapters are forbidden'
        adapter_config = read_json(checkpoint / 'adapter_config.json')
        assert Path(adapter_config['base_model_name_or_path']).resolve() == model_dir
        assert adapter_config['r'] == 16 and adapter_config['lora_alpha'] == 32
        assert adapter_config['peft_type'] == 'LORA'
        names = ['adapter_config.json', 'adapter_model.safetensors']
        if (checkpoint / 'trainer_state.json').is_file():
            names.append('trainer_state.json')
        result['adapter'] = {'path': str(checkpoint),
                             'files_sha256': {name: sha256(checkpoint / name) for name in names}}
    result['identity_sha256'] = stable_hash(result)
    return result


def checked_preflight(run, config):
    preflight = read_json(run / 'preflight/preflight_audit.json')
    assert preflight['passed'] is True
    assert preflight['partition_counts'] == {'initial_sft': 14543, 'validation': 2209}
    assert preflight['all_targets_complete'] and preflight['all_images_preserved']
    assert preflight['assistant_only_loss'] and preflight['skipped_or_substituted_records'] == 0
    inputs = preflight['provenance']['files_sha256']
    for partition in ('initial_sft', 'validation'):
        for key in ('export', 'scoring', 'prompts'):
            assert inputs[str(Path(config['data'][partition][key]).resolve())] == config['data'][partition][key + '_sha256']
    return preflight


def audit_generation_inputs(inputs, prompt, template, generation_config, request_config, frozen):
    """Audit actual model inputs at the installed engine's pre-inference hook."""
    import torch
    assert [message['role'] for message in prompt['messages']] == ['system', 'user']
    assert len(prompt['images']) == 16
    assert 'labels' not in inputs or inputs['labels'] is None
    ids = inputs['input_ids'][0]
    attention = inputs['attention_mask'][0]
    assert len(ids) <= 8192 and attention.bool().all().item()
    grid = inputs['image_grid_thw']
    assert tuple(grid.shape) == (16, 3)
    merge_size = template.processor.image_processor.merge_size
    counts = grid.prod(dim=1) // merge_size ** 2
    assert all(64 <= count <= 144 for count in counts.tolist())
    tokenizer = template.tokenizer
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    starts = (ids == start_id).nonzero().reshape(-1).tolist()
    ends = (ids == end_id).nonzero().reshape(-1).tolist()
    assert len(starts) == len(ends) == 16
    assert int((ids == image_id).sum()) == int(counts.sum())
    assert inputs['pixel_values'].shape[0] == int(grid.prod(dim=1).sum())
    for start, end, count in zip(starts, ends, counts.tolist()):
        assert end - start - 1 == count and torch.all(ids[start + 1:end] == image_id).item()
    decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    compact = re.sub(r'<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>', '<image>', decoded)
    expected = ''.join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                       for message in prompt['messages']) + '<|im_start|>assistant\n'
    assert compact == expected, 'Generated prompt differs from the frozen prompt-only export'
    assert generation_config.max_new_tokens == request_config.max_tokens == 256
    assert generation_config.do_sample is False and generation_config.num_beams == 1
    assert request_config.temperature == 0 and request_config.stream is False
    original = frozen['original_generation_config']
    assert generation_config.eos_token_id == original['eos_token_id'] == [151645, 151643]
    assert generation_config.pad_token_id == original['pad_token_id'] == 151643
    assert len(ids) + 256 <= 8192, 'Prompt plus frozen output budget exceeds context'
    return {'image_count': 16, 'image_grid_thw': grid.detach().cpu().tolist(),
            'image_tokens_per_image': counts.detach().cpu().tolist(),
            'image_tokens': int(counts.sum()), 'prompt_tokens': len(ids),
            'pixel_values_shape': list(inputs['pixel_values'].shape),
            'complete_prompt_exact': True, 'assistant_target_in_generation_input': False,
            'actual_generation_config': {'do_sample': generation_config.do_sample,
                'num_beams': generation_config.num_beams, 'max_new_tokens': generation_config.max_new_tokens,
                'eos_token_id': generation_config.eos_token_id, 'pad_token_id': generation_config.pad_token_id,
                'stop_words': generation_config.stop_words}}


def generate_one(engine, request_config, prompt, scoring, frozen):
    import torch
    from swift.llm import InferRequest
    captured = []
    request = InferRequest(messages=deepcopy(prompt['messages']), images=list(prompt['images']))
    assert [message['role'] for message in request.messages] == ['system', 'user']

    def hook(kwargs):
        captured.append(audit_generation_inputs(kwargs['inputs'], prompt, kwargs['template'],
                                               kwargs['generation_config'], kwargs['request_config'], frozen))
        return kwargs

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # _infer is the installed non-streaming PtEngine path, whose hook is not
    # exposed by the public batched infer wrapper in ms-swift 3.12.6.
    response = engine._infer([request], request_config, pre_infer_hook=hook)[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert len(captured) == 1 and len(response.choices) == 1
    choice = response.choices[0]
    assert choice.message.role == 'assistant' and isinstance(choice.message.content, str)
    assert choice.token_ids is not None
    assert response.usage.completion_tokens == len(choice.token_ids)
    assert len(choice.token_ids) <= 256
    assert response.usage.prompt_tokens == captured[0]['prompt_tokens']
    parsed = parse_prediction(choice.message.content, scoring['task'], scoring['goal_status'] == 'available')
    return {'prediction': choice.message.content, 'raw_generated_token_ids': choice.token_ids,
            'raw_generated_tokens_text': engine.tokenizer.decode(choice.token_ids, skip_special_tokens=False,
                                                                clean_up_tokenization_spaces=False),
            'finish_reason': choice.finish_reason, 'generated_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens, 'parsed': parsed,
            'parsed_fields': parsed['fields'], 'parser_errors': parsed['errors'],
            'strict_format_valid': parsed['valid'], 'preprocessing_audit': captured[0],
            'elapsed_sec': elapsed,
            'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
            'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20}


def teacher_forced_one(model, template, row, scoring, prompt, index, video_ids, verify_native_loss=False):
    """FP32 CE at p-1 for each supervised assistant label position p."""
    import torch
    import torch.nn.functional as F
    from swift.llm import to_device
    from .preflight import audit_record
    template.set_mode('train')
    encoded = template.encode(deepcopy(row))
    batch = template.data_collator([encoded])
    audited = audit_record(batch, row, scoring, prompt, template, 'validation', index, video_ids)
    # The auditor checks CPU collator tensors. Only then transfer to the model.
    labels = batch.pop('labels')
    assert labels.shape[0] == 1
    positions = (labels[0] != -100).nonzero().reshape(-1)
    assert positions.numel() > 0 and positions.min().item() > 0
    targets = labels[0, positions].to(model.device)
    # These collator bookkeeping fields are only needed by training/packing.
    # We use one ordinary nonpacked SDPA example, so preserve model-forward
    # inputs and explicitly reject unreviewed additions rather than pass them.
    allowed = {'input_ids', 'attention_mask', 'position_ids', 'pixel_values', 'image_grid_thw'}
    bookkeeping = {'loss_scale', 'text_position_ids', 'cu_seq_lens_q', 'cu_seq_lens_k',
                   'max_length_q', 'max_length_k'}
    assert set(batch) <= allowed | bookkeeping, f'Unreviewed collator keys: {set(batch) - allowed - bookkeeping}'
    inputs = to_device({key: value for key, value in batch.items() if key in allowed}, model.device)
    assert tuple(inputs['image_grid_thw'].shape) == (16, 3)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, logits_to_keep=(positions - 1).to(model.device),
                       use_cache=False, return_dict=True)
        assert output.logits.shape[:2] == (1, positions.numel())
        assert torch.isfinite(output.logits).all().item()
        token_losses = F.cross_entropy(output.logits[0].float(), targets, reduction='none')
        assert token_losses.numel() == positions.numel() and torch.isfinite(token_losses).all().item()
        total = token_losses.double().sum().item()
        mean = total / positions.numel()
        native_check = None
        if verify_native_loss:
            # One first-row cross-check against the architecture's ordinary
            # full-logit shifted masked loss. Subsequent rows use the proven
            # selective projection, avoiding an unnecessary context×vocab
            # allocation. Both CE paths evaluate the same complete labels.
            native = model(**inputs, labels=labels.to(model.device), use_cache=False, return_dict=True)
            native_mean = native.loss.float().item()
            assert math.isfinite(native_mean)
            delta = abs(native_mean - mean)
            assert math.isclose(native_mean, mean, rel_tol=2e-4, abs_tol=2e-4), (native_mean, mean, delta)
            native_check = {'passed': True, 'ordinary_model_loss': native_mean,
                            'selective_fp32_cross_entropy_mean': mean, 'absolute_difference': delta,
                            'relative_tolerance': 2e-4, 'absolute_tolerance': 2e-4,
                            'ordinary_full_logits_shape': list(native.logits.shape)}
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert math.isfinite(mean) and mean >= 0
    return {'mean_nll': mean, 'nll_sum': total, 'answer_tokens': positions.numel(),
            'target_complete': True, 'assistant_only_loss': True,
            'target_includes_exact_assistant_suffix': True,
            'first_record_native_model_loss_check': native_check,
            'loss_implementation': 'FP32 cross_entropy(logits[p-1], labels[p]) for all labels[p] != -100',
            'preprocessing_audit': audited, 'elapsed_sec': elapsed,
            'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
            'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--stage', choices=('base', 'epoch1', 'epoch2'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--retry-errors', action='store_true',
                        help='Preserve failed rows in an error archive and retry them; reuse exact successful rows')
    args = parser.parse_args(argv)
    run = args.run.resolve()
    config_path, frozen_path = run / 'frozen_config.json', run / 'frozen_evaluation.json'
    config = load_config(config_path, verify=True)
    frozen = read_json(frozen_path)
    assert frozen['frozen_config_sha256'] == sha256(config_path)
    assert frozen['metric_protocol'] == protocol()
    assert frozen['parser_sha256'] == sha256(Path(__file__).with_name('metrics.py'))
    assert frozen['validation_prompts_sha256'] == config['data']['validation']['prompts_sha256']
    configure_environment()
    # Allocation/idle checks run in the parent immediately before launching
    # this process. Visibility is checked again before any model load.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert visible in GPU_UUIDS.values(), 'Evaluation uses one allocated physical GPU 0 or 1'
    physical_gpu_index = next(index for index, uuid in GPU_UUIDS.items() if uuid == visible)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'NPROC_PER_NODE'):
        os.environ.pop(name, None)
    preflight = checked_preflight(run, config)
    rows, scoring, prompts = load_records(config, 'validation')
    assert len(rows) == 2209 and [score['id'] for score in scoring] == frozen['ordered_validation_ids']
    assert all(score['participant'] == 'P07' for score in scoring)
    model_identity = checkpoint_identity(config, args.stage, args.checkpoint)
    code_paths = [Path(__file__), Path(__file__).with_name('common.py'),
                  Path(__file__).with_name('preflight.py'), Path(__file__).with_name('metrics.py')]
    code_hashes = {str(path.resolve()): sha256(path) for path in code_paths}
    versions = {name: importlib.metadata.version(name) for name in
                ('ms-swift', 'transformers', 'torch', 'torchvision', 'peft', 'accelerate', 'qwen-vl-utils')}
    identity = {'stage': args.stage, 'frozen_config_sha256': sha256(config_path),
                'frozen_evaluation_sha256': sha256(frozen_path), 'model': model_identity,
                'processor': config['processor'], 'evaluation': config['evaluation'],
                'data': config['data']['validation'], 'code_sha256': code_hashes, 'versions': versions,
                'preflight_cache_key': preflight['cache_key']}
    run_key = stable_hash(identity)
    prompt_hashes = source_line_hashes(config['data']['validation']['prompts'])
    export_hashes = source_line_hashes(config['data']['validation']['export'])
    scoring_hashes = source_line_hashes(config['data']['validation']['scoring'])
    record_keys = {score['id']: stable_hash({'id': score['id'], 'row_index': index,
                  'prompt_sha256': prompt_hashes[index], 'export_sha256': export_hashes[index],
                  'scoring_sha256': scoring_hashes[index], 'run_key': run_key}) for index, score in enumerate(scoring)}
    output_dir = run / 'predictions' / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / 'metadata.json'
    if metadata_path.exists():
        assert read_json(metadata_path)['run_key'] == run_key, 'Evaluation provenance changed; refuse stale cache'
    else:
        atomic_json(metadata_path, {'run_key': run_key, 'identity': identity, 'created_at_utc': utc_now()})
    generation_path, loss_path = output_dir / 'generation.jsonl', output_dir / 'loss.jsonl'
    generations = journal_rows(generation_path, run_key, record_keys, args.retry_errors)
    losses = journal_rows(loss_path, run_key, record_keys, args.retry_errors)
    completed_path = output_dir / 'completed.json'
    if completed_path.exists():
        completed = read_json(completed_path)
        assert completed['run_key'] == run_key and completed['passed'] is True
        assert completed['generation_sha256'] == sha256(generation_path)
        assert completed['loss_sha256'] == sha256(loss_path)
        assert len(generations) == len(losses) == 2209
        print(json.dumps({'stage': args.stage, 'already_complete': True, 'records': 2209}), flush=True)
        return

    import torch
    from transformers import GenerationConfig
    from swift.llm import InferArguments, RequestConfig, get_template
    from swift.llm.infer.infer import SwiftInfer
    torch.set_num_threads(4)
    torch.manual_seed(config['training']['seed'])
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).total_memory >= 47 * 2 ** 30
    model_started = time.perf_counter()
    inference_arguments = InferArguments(
        model=config['model']['path'], model_type='qwen3_vl',
        adapters=[] if args.stage == 'base' else [str(args.checkpoint.resolve())],
        infer_backend='pt', torch_dtype='bfloat16', attn_impl='sdpa',
        use_hf=True, load_args=False, max_batch_size=1, max_length=8192,
        max_pixels=147456, truncation_strategy='delete',
        device_map={'': 'cuda:0'}, seed=config['training']['seed'],
    )
    inference = SwiftInfer(inference_arguments)
    engine = inference.infer_engine
    engine.strict = True
    assert engine.default_template.truncation_strategy == 'raise'
    assert engine.default_template.max_length == 8192
    assert engine.processor.global_vars['image_min_token_num'] == 64
    assert engine.processor.global_vars['image_max_token_num'] == 144
    canonical = GenerationConfig.from_dict(frozen['original_generation_config'])
    assert canonical.eos_token_id == [151645, 151643] and canonical.pad_token_id == 151643
    engine.generation_config = deepcopy(canonical)
    engine.model.generation_config = deepcopy(canonical)
    model = engine.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            permitted = (torch.bfloat16, torch.float32) if 'lora_' in name else (torch.bfloat16,)
            assert parameter.dtype in permitted, (name, str(parameter.dtype))
    train_template = get_template('qwen3_vl', engine.processor, max_length=8192,
                                  max_pixels=147456, truncation_strategy='raise',
                                  loss_scale='default', padding_free=False)
    train_template.set_mode('train')
    train_template.model = model
    request_config = RequestConfig(max_tokens=256, temperature=0, num_beams=1, stream=False,
                                   seed=config['training']['seed'], return_details=True)
    split = read_json(Path(config['paths']['dataset']) / 'split_manifest.json')
    video_ids = {'validation': set(split['partitions']['validation']['video_ids'])}
    actual_model = {'model_class': type(model).__name__,
                    'physical_gpu_index': physical_gpu_index, 'physical_gpu_uuid': visible,
                    'parameter_dtype_counts': {str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
                                               for dtype in {p.dtype for p in model.parameters()}},
                    'device': str(model.device), 'parameters_require_grad': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'model_load_elapsed_sec': time.perf_counter() - model_started,
                    'request_config': vars(request_config),
                    'canonical_generation_config': canonical.to_dict(),
                    'generation_template_class': type(engine.default_template).__name__,
                    'loss_template_class': type(train_template).__name__}
    atomic_json(output_dir / 'actual_pipeline.json', actual_model)
    started = time.perf_counter()
    reused = {'generation': len(generations), 'loss': len(losses)}
    for index, (row, score, prompt) in enumerate(zip(rows, scoring, prompts)):
        shared = {'id': score['id'], 'row_index': index, 'partition': 'validation', 'participant': 'P07',
                  'task': score['task'], 'goal_status': score['goal_status'], 'stage': args.stage,
                  'run_key': run_key, 'record_key': record_keys[score['id']],
                  'prompt_sha256': prompt_hashes[index], 'model_identity_sha256': model_identity['identity_sha256'],
                  'frozen_config_sha256': identity['frozen_config_sha256'],
                  'frozen_evaluation_sha256': identity['frozen_evaluation_sha256'],
                  'eval_code_sha256': code_hashes[str(Path(__file__).resolve())]}
        for kind, journal, results, computation in (
            ('generation', generation_path, generations, lambda: generate_one(engine, request_config, prompt, score, frozen)),
            ('loss', loss_path, losses, lambda: teacher_forced_one(model, train_template, row, score, prompt, index, video_ids,
                                                                 verify_native_loss=index == 0)),
        ):
            if score['id'] in results:
                continue
            try:
                result = {**shared, 'status': 'completed', **computation()}
            except BaseException as error:
                failed = {**shared, 'status': 'error', 'error': str(error), 'traceback': traceback.format_exc(),
                          'failed_at_utc': utc_now(), 'kind': kind}
                append_row(journal, failed)
                atomic_json(output_dir / 'failure.json', {**failed, 'passed': False,
                    'completed_generation': len(generations), 'completed_loss': len(losses),
                    'error_rows_excluded_from_denominator': False, 'explicit_retry_required': True})
                raise
            append_row(journal, result)
            results[score['id']] = result
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            progress = {'stage': args.stage, 'record_index_completed': index + 1,
                        'generation_completed': len(generations), 'loss_completed': len(losses),
                        'expected': len(rows), 'elapsed_sec': time.perf_counter() - started,
                        'reused': reused, 'last_id': score['id']}
            atomic_json(output_dir / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    ordered_ids = [score['id'] for score in scoring]
    assert set(generations) == set(losses) == set(ordered_ids)
    assert all(generations[key]['status'] == losses[key]['status'] == 'completed' for key in ordered_ids)
    assert all(math.isfinite(losses[key]['mean_nll']) and losses[key]['answer_tokens'] > 0 for key in ordered_ids)
    atomic_jsonl(generation_path, [generations[key] for key in ordered_ids])
    atomic_jsonl(loss_path, [losses[key] for key in ordered_ids])
    result = {'passed': True, 'stage': args.stage, 'run_key': run_key,
              'partition': 'validation', 'records': 2209,
              'frozen_config_sha256': identity['frozen_config_sha256'],
              'frozen_evaluation_sha256': identity['frozen_evaluation_sha256'],
              'model_identity_sha256': model_identity['identity_sha256'],
              'generation_records': len(generations), 'loss_records': len(losses), 'expected_records': 2209,
              'all_16_images_preserved': True, 'complete_assistant_targets': True, 'assistant_only_target_loss': True,
              'no_targets_in_generation_inputs': True, 'original_order_preserved': True,
              'skipped_or_substituted_records': 0, 'generation': str(generation_path), 'loss': str(loss_path),
              'generation_sha256': sha256(generation_path), 'loss_sha256': sha256(loss_path),
              'elapsed_sec': time.perf_counter() - started, 'model_load_elapsed_sec': actual_model['model_load_elapsed_sec'],
              'reused': reused, 'finished_at_utc': utc_now(),
              'peak_gpu_memory_mib': max(value['peak_gpu_memory_mib'] for value in [*generations.values(), *losses.values()]),
              'semantic_quality_claim': 'none; generation is scored separately under the frozen lexical/format protocol',
              'human_reason_label_review': 'pending', 'test_predictions': 0, 'second_stage_predictions': 0}
    atomic_json(completed_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
