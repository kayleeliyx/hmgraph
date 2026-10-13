"""Fresh base+adapter reload and resumable prompt-only local generation.

The answering path reads no reference targets or source narration. Only after
all four smoke answers finish does a separate reporting join read their targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .common import (BASELINE, DATASET, GPU_UUIDS, MODEL, MODEL_REPO, MODEL_REVISION,
                     RUN, canonical, configure, gpu_probe, read_json, rows, sha, write_json)
from code.hdepic_qwen38_baseline.data import prompt_digest
from code.hdepic_qwen38_baseline.runner import gpu_memory, json_safe

VERSION = 'hdepic_qwen38_fresh_peft_generation_v1'
STAGES = ('smoke-reload', 'epoch1', 'epoch2')
PROMPT_KEYS = {'id', 'row_index', 'messages', 'images', 'prompt_sha256'}
GENERATION = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256,
              'use_cache': True, 'eos_token_id': [248046, 248044], 'pad_token_id': 248044,
              'temperature': None, 'top_p': None, 'top_k': None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def stage_paths(stage, run=RUN, attempt='attempt-001'):
    require(stage in STAGES, 'Only smoke-reload, epoch1 and epoch2 are allowed')
    require(attempt and Path(attempt).name == attempt and attempt not in ('.', '..'), 'Unsafe attempt name')
    run = Path(run)
    if stage == 'smoke-reload':
        root = run / 'smoke' / attempt
        return root / 'checkpoints/checkpoint-2', root / 'reload'
    step = 909 if stage == 'epoch1' else 1818
    return run / 'training/main/checkpoints' / f'checkpoint-{step}', run / 'predictions' / stage


def _validate_prompt(record):
    require(set(record) == PROMPT_KEYS, 'Answering record contains unapproved fields')
    require([m.get('role') for m in record['messages']] == ['system', 'user'], 'Reference assistant message in input')
    require(all(set(m) == {'role', 'content'} and isinstance(m['content'], str) for m in record['messages']),
            'Unapproved message fields')
    require(len(record['images']) == 16 and record['messages'][1]['content'].count('<image>') == 16,
            'Sixteen individual images are required')
    require(record['prompt_sha256'] == prompt_digest(record['messages'], record['images']), 'Prompt hash changed')


def load_answering_records(stage, run=RUN):
    """Read frozen prompt-only artifacts, never exports/scoring/source narration."""
    require(stage in STAGES, 'Forbidden evaluation partition/stage')
    run = Path(run)
    if stage == 'smoke-reload':
        manifest = read_json(run / 'inputs/manifest.json')
        spec = manifest['partitions']['initial_sft']
        source = Path(spec['prompts'])
        require(source == DATASET / 'swift/initial_sft.prompts.jsonl', 'Unexpected smoke prompt source')
        require(sha(source) == spec['prompts_sha256'], 'Frozen initial-SFT prompts changed')
        selection = read_json(run / 'inputs/smoke_selection.json')
        selected = selection['records']
        require(len(selected) == 4 and len({r['id'] for r in selected}) == 4, 'Wrong smoke cohort')
        require({r['category'] for r in selected} == {
            'explain:goal_present', 'explain:goal_absent', 'track:goal_present', 'track:goal_absent'},
            'Smoke must cover both tasks with and without goals')
        prompts = rows(source)
        records, metadata = [], {}
        for item in selected:
            require(item['partition'] == 'initial_sft', 'Smoke may only use initial-SFT records')
            index = item['row_index']
            require(spec['ordered_ids'][index] == item['id'], 'Smoke selected ID/index changed')
            prompt = prompts[index]
            require(set(prompt) == {'messages', 'images'}, 'Unexpected prompt export columns')
            record = {'id': item['id'], 'row_index': index, **prompt,
                      'prompt_sha256': prompt_digest(prompt['messages'], prompt['images'])}
            require(record['prompt_sha256'] == item['prompt_sha256'], 'Smoke prompt fingerprint changed')
            task, goal = item['category'].split(':')
            metadata[item['id']] = {'task': task, 'has_goal': goal == 'goal_present', 'partition': 'initial_sft'}
            records.append(record)
        sources = {'prompts_sha256': sha(source), 'selection_sha256': sha(run / 'inputs/smoke_selection.json')}
    else:
        baseline = read_json(BASELINE / 'frozen_config.json')
        source = BASELINE / 'inputs/validation_explain.prompts.jsonl'
        require(sha(source) == baseline['prompts_sha256'], 'Original baseline prompts changed')
        require(baseline['generation'] == GENERATION, 'Original baseline generation policy changed')
        records = rows(source)
        require(len(records) == len({r['id'] for r in records}) == 228, 'Wrong validation Explain cohort')
        require(all(':explain:' in r['id'] and r['id'].startswith('P07-') for r in records),
                'Only the fixed P07 validation Explain cohort is permitted')
        metadata = {r['id']: {'task': 'explain', 'has_goal': True, 'partition': 'validation'} for r in records}
        sources = {'prompts_sha256': sha(source), 'baseline_config_sha256': sha(BASELINE / 'frozen_config.json'),
                   'baseline_processor_preflight_sha256': sha(BASELINE / 'processor_preflight/audit.json')}
    for record in records:
        _validate_prompt(record)
    return records, metadata, sources


def checkpoint_identity(path, stage, run=RUN):
    from .checkpoints import validate_checkpoint
    path, run = Path(path), Path(run)
    marker = read_json(path / 'completed.json')
    identity = marker['identity']
    require(identity['config_sha256'] == sha(run / 'frozen_config.json')
            and identity['model_revision'] == MODEL_REVISION, 'Checkpoint belongs to another experiment')
    expected_step = {'smoke-reload': 2, 'epoch1': 909, 'epoch2': 1818}[stage]
    require(marker['state']['step'] == expected_step, 'Wrong stage checkpoint')
    validate_checkpoint(path, identity)
    adapter_config = read_json(path / 'adapter_config.json')
    require(adapter_config['r'] == 16 and adapter_config['lora_alpha'] == 32
            and adapter_config['lora_dropout'] == 0 and adapter_config['bias'] == 'none', 'LoRA configuration changed')
    return {'path': str(path), 'step': expected_step, 'training_identity': identity,
            'completed_sha256': sha(path / 'completed.json'),
            'adapter_sha256': sha(path / 'adapter_model.safetensors'),
            'adapter_config_sha256': sha(path / 'adapter_config.json')}


def freeze(stage, run=RUN, attempt='attempt-001'):
    checkpoint, output = stage_paths(stage, run, attempt)
    records, metadata, sources = load_answering_records(stage, run)
    checkpoint_info = checkpoint_identity(checkpoint, stage, run)
    code_paths = [Path(__file__), Path(__file__).with_name('common.py'),
                  Path(__file__).parents[1] / 'hdepic_qwen38_baseline/processor.py',
                  Path(__file__).parents[1] / 'hdepic_experiment/metrics.py',
                  Path(__file__).parents[1] / 'hdepic_review_followup/fields.py']
    config = {'version': VERSION, 'stage': stage, 'attempt': attempt if stage == 'smoke-reload' else None,
              'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'base_path': str(MODEL),
              'checkpoint': checkpoint_info, 'input_sources': sources,
              'ordered_ids': [r['id'] for r in records], 'count': len(records),
              'ordered_prompt_sha256': digest([r['prompt_sha256'] for r in records]),
              'requested_fields_metadata': metadata, 'generation': GENERATION,
              'physical_gpus': [0, 1], 'gpu_uuids': GPU_UUIDS,
              'parallelism': 'one native BF16 model sharded across GPUs 0 and 1',
              'device_map': 'balanced', 'max_memory_gib_per_gpu': 42,
              'attention_implementation': 'sdpa', 'individual_images': 16,
              'width': 384, 'height': 384, 'visual_tokens_per_frame': 144,
              'min_pixels': 65536, 'max_pixels': 147456, 'max_context': 8192,
              'enable_thinking': False, 'add_vision_id': False, 'seed': 20261006,
              'precision': {'base': 'bfloat16', 'adapter_storage': 'float32',
                            'adapter_compute': 'BF16 autocast scoped to LoRA A/B Linear leaves',
                            'native_math': 'No global autocast; preserve native FP32 recurrence operations'},
              'packages_sha256': sha(Path(run) / 'environment/packages.txt'),
              'evaluation_code_sha256': {str(p.relative_to(Path(__file__).parents[1])): sha(p) for p in code_paths},
              'references_or_narration_supplied': False, 'training': False, 'test_access': False,
              'semantic_metrics': 'Not evaluated here; human judgments remain separate and assistant judgments provisional'}
    config['configuration_sha256'] = digest(config)
    write_json(output / 'frozen_evaluation.json', config, immutable=True)
    return config, records, metadata, checkpoint, output


def load_cached(path, records, config):
    allowed = {record['id']: record for record in records}
    result = {}
    if not Path(path).exists():
        return result
    for row in rows(path):
        require(row['id'] in allowed and row['id'] not in result, 'Unknown/duplicate prediction ID')
        record = allowed[row['id']]
        require(row['configuration_sha256'] == config['configuration_sha256']
                and row['model_repo'] == MODEL_REPO and row['model_revision'] == MODEL_REVISION
                and row['adapter_sha256'] == config['checkpoint']['adapter_sha256']
                and row['adapter_config_sha256'] == config['checkpoint']['adapter_config_sha256']
                and row['prompt_sha256'] == record['prompt_sha256'] and row['row_index'] == record['row_index'],
                'Prediction cache identity changed')
        require(row['status'] in ('completed', 'failed'), 'Unrecognized prediction status')
        result[row['id']] = row
    return result


def append_prediction(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as handle:
        handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())


def verify_adapter_state(loaded, saved, *, expected_count=992):
    """Compare every saved tensor exactly; no tolerance, dtype conversion or omissions."""
    import torch
    require(set(loaded) == set(saved) and len(saved) == expected_count, 'Saved/reloaded adapter key mismatch')
    counts = Counter()
    for name in sorted(saved):
        require('lora_' in name and 'visual' not in name and 'lm_head' not in name, 'Forbidden adapter tensor')
        actual, expected = loaded[name].detach().cpu(), saved[name]
        require(actual.dtype == expected.dtype == torch.float32 and actual.shape == expected.shape,
                f'Adapter dtype/shape changed: {name}')
        require(torch.equal(actual, expected), f'Adapter tensor did not reload exactly: {name}')
        counts[str(actual.dtype)] += actual.numel()
    return {'all_saved_tensors_verified': True, 'tensor_count': len(saved),
            'parameter_count_by_dtype': dict(counts), 'equality': 'exact tensor equality, including dtype and shape'}


def install_lora_autocast(model, torch):
    """Match FSDP BF16 leaf computation without downcasting native recurrence."""
    handles, names = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not ('.lora_A.' in name or '.lora_B.' in name):
            continue
        stack = []
        def before(_module, _args, _stack=stack):
            context = torch.autocast('cuda', dtype=torch.bfloat16)
            context.__enter__(); _stack.append(context)
        def after(_module, _args, _output, _stack=stack):
            if _stack:
                _stack.pop().__exit__(None, None, None)
        handles.extend([module.register_forward_pre_hook(before),
                        module.register_forward_hook(after, always_call=True)])
        names.append(name)
    require(len(names) == 992, 'Expected exactly 992 LoRA A/B autocast leaves')
    return handles, names


def load_reloaded_model(checkpoint, config):
    """Caller performs the idle/allocation gate before calling this GPU function."""
    import torch, peft, transformers
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration
    require(torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.3.0'
            and peft.__version__ == '0.18.1', 'Unexpected inference environment')
    require(MODEL.name == MODEL_REVISION and MODEL.is_dir(), 'Pinned original base is unavailable')
    for filename in ('config.json', 'model.safetensors.index.json'):
        require(sha(MODEL / filename) == sha(BASELINE / 'model_metadata' / filename),
                f'Pinned base metadata changed: {filename}')
    torch.manual_seed(config['seed'])
    for index in (0, 1):
        # The allocator statistics API does not lazily initialize CUDA.
        torch.cuda.set_device(index)
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    native, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='balanced', max_memory={0: '42GiB', 1: '42GiB'},
        output_loading_info=True)
    require(not loading.get('missing_keys') and not loading.get('mismatched_keys')
            and not loading.get('error_msgs'), f'Incomplete native base loading: {json_safe(loading)}')
    require(all(k.startswith('mtp.') for k in loading.get('unexpected_keys', [])),
            f'Unexpected non-MTP weight: {json_safe(loading)}')
    model = PeftModel.from_pretrained(native, str(checkpoint), is_trainable=False,
                                     autocast_adapter_dtype=True, local_files_only=True)
    saved = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
    # Supplying the LoRA-only mapping avoids collecting or copying frozen weights.
    raw = {name: value for name, value in model.named_parameters() if 'lora_' in name}
    loaded = get_peft_model_state_dict(model, state_dict=raw, save_embedding_layers=False)
    verification = verify_adapter_state(loaded, saved)
    del saved, raw, loaded
    model.eval(); model.requires_grad_(False)
    devices = {str(p.device) for p in model.parameters()}
    require(devices == {'cuda:0', 'cuda:1'}, 'CPU/disk/unauthorized-device offload or missing second GPU')
    require(all(not p.requires_grad for p in model.parameters()), 'Inference parameter remained trainable')
    handles, leaf_names = install_lora_autocast(model, torch)
    audit = {'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
             'adapter_sha256': config['checkpoint']['adapter_sha256'],
             'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
             'configuration_sha256': config['configuration_sha256'],
             'native_loading_info': json_safe(loading), 'device_map': native.hf_device_map,
             'parameter_devices': sorted(devices), 'adapter_verification': verification,
             'all_parameters_frozen': True, 'adapter_compute_policy': config['precision'],
             'autocast_leaf_names': leaf_names, 'load_latency_sec': time.perf_counter() - started,
             'gpu_memory': gpu_memory(torch)}
    return model, audit, handles


def generate_one(model, processor, record, metadata, config):
    import torch
    from transformers import GenerationConfig
    from code.hdepic_qwen38_baseline.processor import preprocess
    from code.hdepic_experiment.metrics import parse_prediction
    from code.hdepic_review_followup.fields import extract_field
    _validate_prompt(record)
    inputs, audit = preprocess(processor, record)
    for index in (0, 1):
        torch.cuda.synchronize(index); torch.cuda.reset_peak_memory_stats(index)
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    observed = {'prefill_calls': 0, 'raw_logits_checks': 0}
    # PEFT.generate delegates to the native model's generate, bypassing the
    # PeftModel.forward wrapper; hooks must observe that native model.
    native = model.get_base_model()
    def prefill(_module, _args, kwargs):
        if kwargs.get('pixel_values') is not None:
            require(observed['prefill_calls'] == 0, 'Images were unexpectedly reprocessed')
            require(kwargs['image_grid_thw'].shape == (16, 3)
                    and kwargs['pixel_values'].shape == (9216, 1536), 'Image count changed at actual prefill')
            require(torch.equal(kwargs['input_ids'], inputs['input_ids'])
                    and torch.equal(kwargs['mm_token_type_ids'], inputs['mm_token_type_ids']), 'Actual prefill prompt changed')
            observed['prefill_calls'] += 1
    def finite_logits(_module, _args, output):
        require(bool(torch.isfinite(output.logits).all()), 'Nonfinite raw logits')
        observed['raw_logits_checks'] += 1
    hooks = [native.register_forward_pre_hook(prefill, with_kwargs=True),
             native.register_forward_hook(finite_logits)]
    started = time.perf_counter()
    try:
        generation = GenerationConfig(**config['generation'])
        with torch.inference_mode():
            sequences = model.generate(**inputs, generation_config=generation, logits_to_keep=1)
        for index in (0, 1):
            torch.cuda.synchronize(index)
        elapsed = time.perf_counter() - started
        require(observed['prefill_calls'] == 1 and observed['raw_logits_checks'] > 0, 'Actual model checks not observed')
        prompt_length = inputs['input_ids'].shape[-1]
        require(torch.equal(sequences[0, :prompt_length], inputs['input_ids'][0]), 'Generated prompt prefix changed')
        token_ids = sequences[0, prompt_length:].detach().cpu().tolist()
        require(0 < len(token_ids) <= 256, 'Empty/excess generation')
        raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(bool(answer.strip()), 'Empty generated answer')
        stopped = token_ids[-1] in config['generation']['eos_token_id']
        task, has_goal = metadata['task'], metadata['has_goal']
        fields = ['why' if task == 'explain' else 'action'] + (['goal'] if has_goal else []) + ['next']
        result = {'id': record['id'], 'row_index': record['row_index'], 'task': task,
                  'partition': metadata['partition'], 'goal_supervised': has_goal, 'status': 'completed',
                  'prediction': answer, 'raw_with_special_tokens': raw, 'generated_token_ids': token_ids,
                  'generated_tokens': len(token_ids), 'finish_reason': 'eos' if stopped else 'length',
                  'latency_sec': elapsed, 'prompt_sha256': record['prompt_sha256'],
                  'configuration_sha256': config['configuration_sha256'],
                  'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                  'adapter_sha256': config['checkpoint']['adapter_sha256'],
                  'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
                  'generation_config': generation.to_dict(), 'parsed': parse_prediction(answer, task, has_goal),
                  'independent_fields': {field: extract_field(answer, field) for field in fields},
                  'preprocessing': audit, 'actual_model_checks': observed,
                  'gpu_memory': gpu_memory(torch), 'created_utc': utc()}
        if task == 'explain':
            result['independent_why'] = result['independent_fields']['why']
        return result
    finally:
        for hook in hooks:
            hook.remove()


def write_smoke_reference_comparison(run, output, records, predictions):
    """Post-generation reporting only; these references never enter answering inputs."""
    require(all(predictions[r['id']]['status'] == 'completed' for r in records), 'Generate all smoke answers before reference join')
    manifest = read_json(Path(run) / 'inputs/manifest.json')
    spec = manifest['partitions']['initial_sft']
    for kind in ('export', 'scoring'):
        require(sha(spec[kind]) == spec[kind + '_sha256'], 'Frozen smoke reference source changed')
    exports, scoring = rows(spec['export']), rows(spec['scoring'])
    comparisons = []
    lines = ['# Fresh smoke adapter reload: training examples', '',
             'These four initial-SFT examples verify execution and formatting, not model accuracy.', '']
    for record in records:
        index = record['row_index']; sidecar = scoring[index]
        require(sidecar['id'] == record['id'], 'Smoke reference ID mismatch')
        target = exports[index]['messages'][-1]['content']; prediction = predictions[record['id']]
        comparisons.append({'id': record['id'], 'task': prediction['task'], 'goal_supervised': prediction['goal_supervised'],
                            'reference_target': target, 'prediction': prediction['prediction'],
                            'strict_format_valid': prediction['parsed']['valid'], 'prompt_sha256': record['prompt_sha256']})
        lines.extend([f"## {record['id']}", '', 'Reference:', '', '```text', target, '```', '',
                      'Generated:', '', '```text', prediction['prediction'], '```', '',
                      f"Requested tags valid: {prediction['parsed']['valid']}; finish: {prediction['finish_reason']}.", ''])
    write_json(Path(output) / 'reference_comparison.json', comparisons, immutable=True)
    destination = Path(output) / 'reference_comparison.md'
    content = '\n'.join(lines) + '\n'
    if destination.exists():
        require(destination.read_text() == content, 'Smoke reference comparison changed')
    else:
        destination.write_text(content)


def run_evaluation(stage, run=RUN, attempt='attempt-001'):
    configure()
    run = Path(run)
    config, records, metadata, checkpoint, output = freeze(stage, run, attempt)
    prediction_path = output / 'generation.jsonl'
    cached = load_cached(prediction_path, records, config)
    completed = output / 'completed.json'
    if completed.exists():
        marker = read_json(completed)
        require(marker['passed'] and marker['configuration_sha256'] == config['configuration_sha256']
                and marker['generation_sha256'] == sha(prediction_path)
                and len(cached) == len(records) and all(r['status'] == 'completed' for r in cached.values()),
                'Completed evaluation was modified')
        return marker
    require(all(row['status'] == 'completed' for row in cached.values()),
            'A saved generation failure requires diagnosis; no silent retry or replacement')
    missing = [r for r in records if r['id'] not in cached]
    loading = None
    if missing:
        resources = gpu_probe(require_idle=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_UUIDS)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['OMP_NUM_THREADS'] = '1'
        from code.hdepic_qwen38_baseline.processor import load_processor
        model, loading, autocast_handles = load_reloaded_model(checkpoint, config)
        loading['resources'] = resources
        write_json(output / 'loading_attempts' / f'{time.time_ns()}.json', loading)
        processor = load_processor(MODEL)
        try:
            for record in missing:
                try:
                    row = generate_one(model, processor, record, metadata[record['id']], config)
                except Exception as error:
                    import torch
                    row = {'id': record['id'], 'row_index': record['row_index'], 'status': 'failed',
                           'task': metadata[record['id']]['task'], 'prompt_sha256': record['prompt_sha256'],
                           'configuration_sha256': config['configuration_sha256'],
                           'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                           'adapter_sha256': config['checkpoint']['adapter_sha256'],
                           'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
                           'error': repr(error), 'traceback': traceback.format_exc(),
                           'gpu_memory': gpu_memory(torch), 'created_utc': utc()}
                    append_prediction(prediction_path, row)
                    raise
                append_prediction(prediction_path, row); cached[record['id']] = row
                print(json.dumps({'stage': stage, 'completed': len(cached), 'expected': len(records),
                                  'id': record['id'], 'latency_sec': row['latency_sec']}), flush=True)
        finally:
            for hook in autocast_handles:
                hook.remove()
            del model
    require(len(cached) == len(records), 'Incomplete evaluation')
    if stage == 'smoke-reload':
        write_smoke_reference_comparison(run, output, records, cached)
    loading_attempts = [read_json(path) for path in sorted((output / 'loading_attempts').glob('*.json'))]
    require(loading_attempts and all(a['configuration_sha256'] == config['configuration_sha256']
                                    and a['adapter_verification']['all_saved_tensors_verified']
                                    for a in loading_attempts), 'Fresh-reload audit missing or changed')
    memory_records = [r['gpu_memory'] for r in cached.values()] + [a['gpu_memory'] for a in loading_attempts]
    marker = {'passed': True, 'stage': stage, 'count': len(records),
              'configuration_sha256': config['configuration_sha256'],
              'adapter_sha256': config['checkpoint']['adapter_sha256'],
              'generation_sha256': sha(prediction_path), 'created_utc': utc(),
              'strict_format_valid': sum(r['parsed']['valid'] for r in cached.values()),
              'length_limited': sum(r['finish_reason'] == 'length' for r in cached.values()),
              'total_saved_generation_seconds': sum(r['latency_sec'] for r in cached.values()),
              'total_successful_loading_seconds': sum(a['load_latency_sec'] for a in loading_attempts),
              'per_gpu_peak_allocated_bytes': {str(i): max(r[str(i)]['peak_allocated_bytes'] for r in memory_records) for i in (0, 1)},
              'per_gpu_peak_reserved_bytes': {str(i): max(r[str(i)]['peak_reserved_bytes'] for r in memory_records) for i in (0, 1)},
              'semantic_accuracy_evaluated': False, 'references_supplied_to_model': False}
    write_json(completed, marker, immutable=True)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--attempt', default='attempt-001')
    args = parser.parse_args()
    try:
        print(json.dumps(run_evaluation(args.stage, args.run, args.attempt), indent=2))
    except Exception as error:
        _, output = stage_paths(args.stage, args.run, args.attempt)
        failure = {'stage': args.stage, 'status': 'failed_or_blocked', 'error': repr(error),
                   'traceback': traceback.format_exc(), 'created_utc': utc(), 'training': False, 'test_access': False}
        write_json(output / 'failures' / f'{time.time_ns()}.json', failure)
        print(json.dumps(failure, indent=2)); raise SystemExit(2)


if __name__ == '__main__':
    main()
