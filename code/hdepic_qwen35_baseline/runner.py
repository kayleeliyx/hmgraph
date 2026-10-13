"""Gated native local inference. This answering module never reads references."""
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

from .data import canonical_bytes, immutable_write, json_bytes, require, rows, sha256
from .model import RUN, REPO, REVISION, CACHE, fetch_weights

CONFIG_NAME = 'frozen_config_v2.json'


def json_safe(value):
    """Native Transformers 5.3 loading diagnostics contain sets."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze(run=RUN):
    """Freeze only answering inputs, native metadata, code and environment."""
    run = Path(run)
    manifest = json.loads((run / 'inputs/manifest.json').read_text())
    require(manifest['count'] == 228 and manifest['partition'] == 'validation', 'Wrong cohort')
    preflight = json.loads((run / 'processor_preflight/audit.json').read_text())
    require(preflight['passed'] and preflight['records'] == 228, 'Complete CPU preflight required')
    identity = json.loads((run / 'model_identity.json').read_text())
    require(identity['repo_id'] == REPO and identity['revision'] == REVISION, 'Wrong checkpoint')
    compatibility = json.loads((run / 'environment/compatibility.json').read_text())
    require(compatibility['qwen35_moe_native_import'], 'Native class unavailable')
    package_dir = Path(__file__).parent
    source_names = ('runner.py', 'processor.py', 'model.py', 'resources.py')
    sources = {name: sha256(package_dir / name) for name in source_names}
    config = {
        'version': 'hdepic_qwen35_35b_native_baseline_v1',
        'repo_id': REPO, 'revision': REVISION, 'adaptation': 'none; official released checkpoint',
        'dataset': 'hdepic_sft_v2_repaired', 'partition': 'validation', 'task': 'explain', 'count': 228,
        'input_manifest_sha256': sha256(run / 'inputs/manifest.json'),
        'prompts_sha256': sha256(run / 'inputs/validation_explain.prompts.jsonl'),
        'processor_preflight_sha256': sha256(run / 'processor_preflight/audit.json'),
        'model_identity_sha256': sha256(run / 'model_identity.json'),
        'environment_sha256': sha256(run / 'environment/packages.txt'),
        'packages': compatibility['packages'], 'answering_code_sha256': sources,
        'physical_gpus': [0, 1], 'parallelism': 'one model sharded across two authorized idle GPUs',
        'dtype': 'bfloat16', 'quantization': None, 'cpu_or_disk_offload': False,
        'device_map': 'balanced', 'max_memory_gib_per_gpu': 42,
        'attention_implementation': 'sdpa', 'experts_implementation': 'eager',
        'native_torch_linear_attention_fallback': True,
        'individual_images': 16, 'min_pixels': 65536, 'max_pixels': 147456,
        'visual_tokens_per_frame': 144, 'max_context': 8192,
        'enable_thinking': False, 'add_vision_id': False, 'prompt_editing': False,
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256,
                       'use_cache': True, 'eos_token_id': [248046, 248044], 'pad_token_id': 248044,
                       'temperature': None, 'top_p': None, 'top_k': None},
        'generation_forward_kwargs': {'logits_to_keep': 1}, 'seed': 20261006,
        'smoke': {'records': 4, 'selection_file': 'inputs/smoke_selection.json',
                  'criteria': ['native loading has no missing/mismatched parameters',
                               '16 original images and complete prompt reach first prefill',
                               'finite raw logits; nonempty generated answer; no OOM',
                               'at least 1 GiB free per GPU after each smoke generation'],
                  'format_policy': 'Malformed or length-limited answers are retained diagnostics; custom-tag compliance is not a model-substitution or semantic-success gate.'},
        'resume': 'Reuse only completed rows with matching model/configuration/prompt hashes; failures are explicit and block completion.',
        'references_or_narration_supplied': False, 'training': False, 'test_access': False,
        'original_epoch1_selection_preserved': 'checkpoint-909 under original Explain NLL',
        'semantic_scores': 'Not produced by answering pipeline; future assistant judgments provisional, human fields separate.',
    }
    config['configuration_sha256'] = digest(config)
    immutable_write(run / CONFIG_NAME, json_bytes(config))
    immutable_write(run / 'active_configuration.json', json_bytes({
        'path': CONFIG_NAME, 'configuration_sha256': config['configuration_sha256'],
        'file_sha256': sha256(run / CONFIG_NAME),
        'previous_configuration': 'frozen_config.json',
        'implementation_fix': 'Serialize native loading-info sets as sorted lists; no experimental settings changed.'}))
    return config


def load_cached(path, records, config):
    allowed = {record['id']: record for record in records}
    result = {}
    if not Path(path).exists():
        return result
    for row in rows(path):
        require(row['id'] in allowed and row['id'] not in result, 'Unknown/duplicate cached ID')
        require(row['configuration_sha256'] == config['configuration_sha256']
                and row['model_revision'] == REVISION
                and row.get('model_repo', REPO) == REPO
                and row['row_index'] == allowed[row['id']]['row_index']
                and row['prompt_sha256'] == allowed[row['id']]['prompt_sha256'], 'Prediction cache identity changed')
        result[row['id']] = row
    return result


def append(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as handle:
        handle.write(canonical_bytes(row) + b'\n'); handle.flush(); os.fsync(handle.fileno())


def gpu_memory(torch):
    return {str(index): {'allocated_bytes': torch.cuda.memory_allocated(index),
                        'reserved_bytes': torch.cuda.memory_reserved(index),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(index),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(index),
                        'free_bytes': torch.cuda.mem_get_info(index)[0],
                        'total_bytes': torch.cuda.mem_get_info(index)[1]} for index in (0, 1)}


def generate_one(torch, model, processor, record, config):
    from .processor import preprocess
    from transformers import GenerationConfig
    from code.hdepic_experiment.metrics import parse_prediction
    from code.hdepic_review_followup.fields import extract_field
    inputs, audit = preprocess(processor, record)
    for index in (0, 1):
        torch.cuda.synchronize(index); torch.cuda.reset_peak_memory_stats(index)
    embedding_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(embedding_device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    observed = {'prefill_calls': 0, 'raw_logits_checks': 0}

    def prefill_hook(module, args, kwargs):
        if kwargs.get('pixel_values') is not None:
            require(observed['prefill_calls'] == 0, 'Images were unexpectedly reprocessed')
            require(kwargs['image_grid_thw'].shape == (16, 3)
                    and kwargs['pixel_values'].shape == (9216, 1536), 'Image count changed at actual model prefill')
            require(torch.equal(kwargs['input_ids'], inputs['input_ids'])
                    and torch.equal(kwargs['mm_token_type_ids'], inputs['mm_token_type_ids']), 'Prompt changed at actual model prefill')
            observed['prefill_calls'] += 1

    def logits_hook(module, args, output):
        logits = output.logits
        require(bool(torch.isfinite(logits).all()), 'Nonfinite raw model logits')
        observed['raw_logits_checks'] += 1

    hooks = [model.register_forward_pre_hook(prefill_hook, with_kwargs=True),
             model.register_forward_hook(logits_hook)]
    started = time.perf_counter()
    try:
        generation_config = GenerationConfig(**config['generation'])
        with torch.inference_mode():
            sequences = model.generate(**inputs, generation_config=generation_config, logits_to_keep=1)
        for index in (0, 1):
            torch.cuda.synchronize(index)
        latency = time.perf_counter() - started
        require(observed['prefill_calls'] == 1 and observed['raw_logits_checks'] > 0, 'Model prefill/logit checks were not observed')
        prompt_length = inputs['input_ids'].shape[-1]
        require(torch.equal(sequences[0, :prompt_length], inputs['input_ids'][0]), 'Returned prompt prefix changed')
        token_ids = sequences[0, prompt_length:].detach().cpu().tolist()
        require(0 < len(token_ids) <= 256, 'Empty/excess generation')
        raw_with_special = processor.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prediction = processor.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(bool(prediction.strip()), 'Empty generated answer')
        stopped = token_ids[-1] in config['generation']['eos_token_id']
        # All 228 frozen Explain prompts explicitly request WHY, GOAL, NEXT.
        return {'id': record['id'], 'row_index': record['row_index'], 'task': 'explain',
                'status': 'completed', 'prediction': prediction, 'raw_with_special_tokens': raw_with_special,
                'generated_token_ids': token_ids, 'generated_tokens': len(token_ids),
                'finish_reason': 'eos' if stopped else 'length', 'latency_sec': latency,
                'prompt_sha256': record['prompt_sha256'], 'configuration_sha256': config['configuration_sha256'],
                'model_repo': REPO, 'model_revision': REVISION, 'generation_config': generation_config.to_dict(),
                'parsed': parse_prediction(prediction, 'explain', True),
                'independent_why': extract_field(prediction, 'why'),
                'preprocessing': audit, 'actual_model_checks': observed,
                'gpu_memory': gpu_memory(torch), 'created_utc': utc()}
    finally:
        for hook in hooks:
            hook.remove()
        del inputs


def run_inference(run=RUN):
    """Nothing GPU-related is imported until the idle allocation gate passes."""
    from .resources import require_idle_pair, require_disk_space
    run = Path(run)
    config = freeze(run)
    records = rows(run / 'inputs/validation_explain.prompts.jsonl')
    prediction_path = run / 'predictions/generation.jsonl'
    cached = load_cached(prediction_path, records, config)
    completed_path = run / 'predictions/completed.json'
    if completed_path.exists():
        marker = json.loads(completed_path.read_text())
        require(marker['passed'] is True and marker['count'] == 228
                and len(cached) == 228 and all(row['status'] == 'completed' for row in cached.values())
                and marker['generation_sha256'] == sha256(prediction_path)
                and marker['configuration_sha256'] == config['configuration_sha256'], 'Completed run changed')
        print('All 228 predictions already complete; no model/GPU load.'); return marker
    require(not any(row['status'] != 'completed' for row in cached.values()), 'Saved failure requires explicit diagnosis; no silent retry')
    resources = require_idle_pair(run)
    require_disk_space(run, CACHE)
    # Logical 0/1 map exactly to the authorized physical 0/1, recorded by UUID.
    os.environ.update({'CUDA_VISIBLE_DEVICES': resources['cuda_visible_devices'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'})
    snapshot = fetch_weights()
    require(snapshot.name == REVISION, 'Downloaded snapshot differs from pinned model')
    require_idle_pair(run)  # Download may take time; recheck before CUDA allocation.
    import torch
    from transformers import Qwen3_5MoeForConditionalGeneration
    from .processor import load_processor
    torch.manual_seed(config['seed']); torch.set_num_threads(1)
    require(torch.cuda.device_count() == 2, 'Expected exactly two visible authorized GPUs')
    started = time.perf_counter()
    model, info = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='eager', device_map='balanced',
        max_memory={0: '42GiB', 1: '42GiB'}, output_loading_info=True)
    devices = {str(p.device) for p in model.parameters()}
    model.eval(); model.requires_grad_(False)
    loading = {'loading_info': json_safe(info), 'device_map': model.hf_device_map, 'parameter_devices': sorted(devices),
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'dtype_device_parameter_counts': dict(Counter({
                   key: sum(p.numel() for p in model.parameters() if f'{p.dtype}:{p.device}' == key)
                   for key in {f'{p.dtype}:{p.device}' for p in model.parameters()}})),
               'load_latency_sec': time.perf_counter() - started, 'resource_probe': resources,
               'gpu_memory': gpu_memory(torch), 'native_unused_mtp_policy': 'Only official mtp.* extra tensors may be unused.'}
    immutable_write(run / f'smoke/loading_attempts/{time.time_ns()}.json', json_bytes(loading))
    if not (run / 'smoke/model_loading.json').exists():
        immutable_write(run / 'smoke/model_loading.json', json_bytes(loading))
    require(devices <= {'cuda:0', 'cuda:1'} and devices == {'cuda:0', 'cuda:1'}, 'CPU/disk/unauthorized device offload detected')
    require(not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'), 'Incomplete native weight loading')
    require(all(key.startswith('mtp.') for key in info.get('unexpected_keys', [])), 'Unexpected non-MTP checkpoint tensors')
    processor = load_processor(run / 'model_metadata')
    smoke_ids = json.loads((run / 'inputs/smoke_selection.json').read_text())['ids']
    smoke_path = run / 'smoke/generation.jsonl'
    smoke_cached = load_cached(smoke_path, records, config)
    for rid in smoke_ids:
        if rid not in smoke_cached:
            row = generate_one(torch, model, processor, next(r for r in records if r['id'] == rid), config)
            append(smoke_path, row); smoke_cached[rid] = row
        require(smoke_cached[rid]['status'] == 'completed', 'Failed smoke generation')
        require(all(item['free_bytes'] >= 1024**3 for item in gpu_memory(torch).values()), 'Smoke has less than 1GiB GPU headroom')
    immutable_write(run / 'smoke/passed.json', json_bytes({'passed': True, 'count': 4,
                    'configuration_sha256': config['configuration_sha256'], 'generation_sha256': sha256(smoke_path),
                    'criteria': config['smoke']['criteria'], 'custom_tag_compliance': sum(smoke_cached[rid]['parsed']['valid'] for rid in smoke_ids)}))
    for record in records:
        if record['id'] in cached:
            continue
        try:
            row = smoke_cached.get(record['id']) or generate_one(torch, model, processor, record, config)
        except Exception as error:
            row = {'id': record['id'], 'row_index': record['row_index'], 'status': 'failed',
                   'model_revision': REVISION, 'configuration_sha256': config['configuration_sha256'],
                   'prompt_sha256': record['prompt_sha256'], 'error': repr(error), 'traceback': traceback.format_exc(),
                   'created_utc': utc(), 'gpu_memory': gpu_memory(torch)}
            append(prediction_path, row)
            raise
        append(prediction_path, row); cached[record['id']] = row
        print(json.dumps({'completed': len(cached), 'expected': 228, 'id': record['id'], 'latency_sec': row['latency_sec']}), flush=True)
    require(len(cached) == 228, 'Incomplete generation')
    marker = {'passed': True, 'count': 228, 'generation_sha256': sha256(prediction_path),
              'inputs_manifest_sha256': sha256(run / 'inputs/manifest.json'),
              'configuration_sha256': config['configuration_sha256'], 'created_utc': utc()}
    immutable_write(completed_path, json_bytes(marker))
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'infer'])
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    try:
        result = freeze(args.run) if args.stage == 'freeze' else run_inference(args.run)
        print(json.dumps(result, indent=2))
    except Exception as error:
        failure = {'status': 'blocked_or_failed', 'stage': args.stage, 'utc': utc(),
                   'error': repr(error), 'traceback': traceback.format_exc(),
                   'training': False, 'test_access': False}
        destination = args.run / 'failures' / f'{time.time_ns()}.json'
        immutable_write(destination, json_bytes(failure))
        print(json.dumps(failure, indent=2)); raise SystemExit(2)


if __name__ == '__main__':
    main()
