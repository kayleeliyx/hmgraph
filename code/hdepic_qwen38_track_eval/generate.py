"""Prompt-only generation for the 1,981 P07 validation Track queries.

Reuses the frozen Explain generation path: native BF16 loading balanced over
authorized GPUs 0/1, exact LoRA reload for epoch checkpoints, 16 cached images,
greedy decoding with at most 256 new tokens and native EOS. Answering reads only
prompt files; references are joined later by the standalone evaluator.
Each worker uses one user-authorized pair of identical RTX 6000 Ada GPUs and
waits until both are idle; other jobs are never terminated. Per-model locks let
two pairs generate different models concurrently.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from code.hdepic_sft_qwen38.common import (MODEL, MODEL_REPO, MODEL_REVISION, PROJECT,
                                           RUN as SFT_RUN, configure, read_json, rows, sha, write_json)
from code.hdepic_sft_qwen38.evaluate import (GENERATION, _validate_prompt, append_prediction,
                                             checkpoint_identity, digest, load_cached,
                                             load_reloaded_model, require, stage_paths, utc)
from code.hdepic_qwen38_baseline.data import prompt_digest
from code.hdepic_qwen38_baseline.runner import gpu_memory, json_safe

VERSION = 'hdepic_qwen38_track_generation_v1'
OUT = PROJECT / 'runs/hdepic_qwen38_27b_track_v1'
MODELS = ('epoch1', 'base', 'epoch2')
SEED = 20261006


def load_records():
    """All validation Track prompts, in frozen dataset order."""
    spec = read_json(SFT_RUN / 'inputs/manifest.json')['partitions']['validation']
    require(sha(spec['prompts']) == spec['prompts_sha256'], 'Frozen validation prompts changed')
    require(spec['category_counts'] == {'track:goal_present': 1981, 'explain:goal_present': 228},
            'Validation goal coverage changed')
    prompts, ids = rows(spec['prompts']), spec['ordered_ids']
    require(len(prompts) == len(ids) == 2209, 'Validation prompt/ID alignment changed')
    records = []
    for index, (rid, prompt) in enumerate(zip(ids, prompts)):
        if ':track:' not in rid:
            continue
        require(set(prompt) == {'messages', 'images'}, 'Unexpected prompt export columns')
        record = {'id': rid, 'row_index': index, **prompt,
                  'prompt_sha256': prompt_digest(prompt['messages'], prompt['images'])}
        _validate_prompt(record)
        user = record['messages'][1]['content']
        require(rid.startswith('P07-') and user.startswith('Task: Track\n')
                and '<action>' in user and '<goal>' in user and '<next>' in user, 'Not a Track prompt: ' + rid)
        records.append(record)
    require(len(records) == len({r['id'] for r in records}) == 1981, 'Wrong validation Track cohort')
    metadata = {r['id']: {'task': 'track', 'has_goal': True, 'partition': 'validation'} for r in records}
    sources = {'prompts': spec['prompts'], 'prompts_sha256': spec['prompts_sha256'],
               'sft_input_manifest_sha256': sha(SFT_RUN / 'inputs/manifest.json')}
    return records, metadata, sources


def freeze(name, write=True):
    require(name in MODELS, 'Unknown model')
    records, metadata, sources = load_records()
    if name == 'base':
        checkpoint = None
        checkpoint_info = {'path': None, 'step': None, 'adapter_sha256': None, 'adapter_config_sha256': None}
    else:
        checkpoint, _ = stage_paths(name, SFT_RUN)
        checkpoint_info = checkpoint_identity(checkpoint, name, SFT_RUN)
    here = Path(__file__)
    code_paths = [here, here.parents[1] / 'hdepic_sft_qwen38/evaluate.py',
                  here.parents[1] / 'hdepic_sft_qwen38/common.py',
                  here.parents[1] / 'hdepic_qwen38_baseline/processor.py',
                  here.parents[1] / 'hdepic_experiment/metrics.py',
                  here.parents[1] / 'hdepic_review_followup/fields.py']
    config = {'version': VERSION, 'model': name, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
              'base_path': str(MODEL), 'checkpoint': checkpoint_info, 'input_sources': sources,
              'partition': 'validation', 'task': 'track', 'count': len(records),
              'ordered_ids': [r['id'] for r in records],
              'ordered_prompt_sha256': digest([r['prompt_sha256'] for r in records]),
              'requested_fields': ['action', 'goal', 'next'], 'generation': GENERATION,
              'gpu_policy': 'one idle pair of identical RTX 6000 Ada GPUs authorized by the user; '
                            'physical indices and UUIDs recorded in each loading attempt',
              'parallelism': 'one native BF16 model sharded across one GPU pair',
              'device_map': 'balanced', 'max_memory_gib_per_gpu': 42,
              'attention_implementation': 'sdpa', 'individual_images': 16,
              'width': 384, 'height': 384, 'visual_tokens_per_frame': 144,
              'min_pixels': 65536, 'max_pixels': 147456, 'max_context': 8192,
              'enable_thinking': False, 'add_vision_id': False, 'seed': SEED,
              'precision': {'base': 'bfloat16', 'adapter_storage': 'float32' if checkpoint else None,
                            'adapter_compute': 'BF16 autocast scoped to LoRA A/B Linear leaves' if checkpoint else None,
                            'native_math': 'No global autocast; preserve native FP32 recurrence operations'},
              'same_settings_as': 'Explain generation in runs/hdepic_qwen38_27b_baseline_v1 and '
                                  'runs/hdepic_sft_qwen38_27b_v1/predictions',
              'packages_sha256': sha(SFT_RUN / 'environment/packages.txt'),
              'generation_code_sha256': {str(p.relative_to(here.parents[1])): sha(p) for p in code_paths},
              'references_or_narration_supplied': False, 'training': False, 'test_access': False}
    config['configuration_sha256'] = digest(config)
    output = OUT / 'predictions' / name
    if write:
        write_json(output / 'frozen_evaluation.json', config, immutable=True)
    return config, records, metadata, checkpoint, output


def load_base_model(config):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    for filename in ('config.json', 'model.safetensors.index.json'):
        require(sha(MODEL / filename) == sha(PROJECT / 'runs/hdepic_qwen38_27b_baseline_v1/model_metadata' / filename),
                f'Pinned base metadata changed: {filename}')
    torch.manual_seed(config['seed'])
    for index in (0, 1):
        torch.cuda.set_device(index); torch.cuda.synchronize(index); torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='balanced', max_memory={0: '42GiB', 1: '42GiB'},
        output_loading_info=True)
    require(not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),
            'Incomplete native weight loading')
    require(all(k.startswith('mtp.') for k in info.get('unexpected_keys', [])), 'Unexpected non-MTP weight')
    model.eval(); model.requires_grad_(False)
    devices = {str(p.device) for p in model.parameters()}
    require(devices == {'cuda:0', 'cuda:1'}, 'CPU/disk/unauthorized-device offload or missing second GPU')
    return model, {'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'adapter_sha256': None,
                   'configuration_sha256': config['configuration_sha256'], 'native_loading_info': json_safe(info),
                   'device_map': model.hf_device_map, 'parameter_devices': sorted(devices),
                   'load_latency_sec': time.perf_counter() - started, 'gpu_memory': gpu_memory(torch)}


def generate_one(model, processor, record, metadata, config):
    """Same checks and decoding as the Explain path, generalized to base or PEFT models."""
    import torch
    from peft import PeftModel
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
    # PEFT.generate delegates to the native model's generate; hook the native model.
    native = model.get_base_model() if isinstance(model, PeftModel) else model
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
        task, has_goal = metadata['task'], metadata['has_goal']
        fields = ['action'] + (['goal'] if has_goal else []) + ['next']
        return {'id': record['id'], 'row_index': record['row_index'], 'task': task,
                'partition': metadata['partition'], 'goal_supervised': has_goal, 'status': 'completed',
                'prediction': answer, 'raw_with_special_tokens': raw, 'generated_token_ids': token_ids,
                'generated_tokens': len(token_ids),
                'finish_reason': 'eos' if token_ids[-1] in config['generation']['eos_token_id'] else 'length',
                'latency_sec': elapsed, 'prompt_sha256': record['prompt_sha256'],
                'configuration_sha256': config['configuration_sha256'],
                'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'adapter_sha256': config['checkpoint']['adapter_sha256'],
                'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
                'generation_config': generation.to_dict(), 'parsed': parse_prediction(answer, task, has_goal),
                'independent_fields': {field: extract_field(answer, field) for field in fields},
                'preprocessing': audit, 'actual_model_checks': observed,
                'gpu_memory': gpu_memory(torch), 'created_utc': utc()}
    finally:
        for hook in hooks:
            hook.remove()


def probe_pair(indices):
    """Same idle criteria as the SFT workflow: <1 GiB used, <5% utilization, no compute process."""
    fields = 'index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu'
    query = lambda *a: subprocess.run(['nvidia-smi', *a], capture_output=True, text=True, check=True).stdout
    gpus = [v.strip().split(', ') for v in query('--query-gpu=' + fields, '--format=csv,noheader,nounits').splitlines()]
    selected = [v for v in gpus if int(v[0]) in indices]
    require(len(selected) == 2 and all(v[2] == 'NVIDIA RTX 6000 Ada Generation' for v in selected),
            'Expected two RTX 6000 Ada GPUs')
    processes = query('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits')
    busy = [v for v in selected if int(v[4]) >= 1024 or int(v[6]) >= 5 or v[1] in processes]
    return {'at_unix': time.time(), 'physical_gpus': selected, 'uuids': [v[1] for v in selected],
            'processes': processes, 'idle': not busy}


def wait_for_idle(poll_seconds, indices, confirmations=2):
    """Both GPUs of the pair idle on consecutive probes; other jobs are never touched."""
    streak, waited, last_note = 0, 0.0, 0.0
    while True:
        probe = probe_pair(indices)
        streak = streak + 1 if probe['idle'] else 0
        if not probe['idle'] and (waited - last_note >= 3600 or last_note == 0):
            print(json.dumps({'waiting_for_gpus': indices, 'waited_sec': round(waited),
                              'state': [v[:1] + v[4:] for v in probe['physical_gpus']], 'at_utc': utc()}), flush=True)
            last_note = waited or 1
        if streak >= confirmations:
            return probe
        time.sleep(poll_seconds); waited += poll_seconds


def run(name, poll_seconds, indices):
    configure()
    config, records, metadata, checkpoint, output = freeze(name)
    prediction_path = output / 'generation.jsonl'
    cached = load_cached(prediction_path, records, config)
    completed = output / 'completed.json'
    if completed.exists():
        marker = read_json(completed)
        require(marker['passed'] and marker['configuration_sha256'] == config['configuration_sha256']
                and marker['generation_sha256'] == sha(prediction_path) and len(cached) == len(records),
                'Completed generation was modified')
        print(json.dumps({'model': name, 'already_complete': True}), flush=True)
        return marker
    require(all(r['status'] == 'completed' for r in cached.values()),
            'A saved generation failure requires diagnosis; no silent retry or replacement')
    missing = [r for r in records if r['id'] not in cached]
    if missing:
        from code.hdepic_sft_qwen38.storage import validate_storage
        validate_storage(require_space=True, run=OUT)
        resources = wait_for_idle(poll_seconds, indices)
        os.environ.update(CUDA_VISIBLE_DEVICES=','.join(resources['uuids']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          OMP_NUM_THREADS='1')
        import torch
        from code.hdepic_qwen38_baseline.processor import load_processor
        require(torch.cuda.device_count() == 2, 'Expected exactly the two authorized GPUs')
        handles = []
        if checkpoint is None:
            model, loading = load_base_model(config)
        else:
            model, loading, handles = load_reloaded_model(checkpoint, config)
        loading['resources'] = resources
        write_json(output / 'loading_attempts' / f'{time.time_ns()}.json', loading)
        processor = load_processor(MODEL)
        try:
            for record in missing:
                try:
                    row = generate_one(model, processor, record, metadata[record['id']], config)
                except Exception as error:
                    row = {'id': record['id'], 'row_index': record['row_index'], 'status': 'failed', 'task': 'track',
                           'prompt_sha256': record['prompt_sha256'], 'configuration_sha256': config['configuration_sha256'],
                           'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                           'adapter_sha256': config['checkpoint']['adapter_sha256'],
                           'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
                           'error': repr(error), 'traceback': traceback.format_exc(),
                           'gpu_memory': gpu_memory(torch), 'created_utc': utc()}
                    append_prediction(prediction_path, row)
                    raise
                append_prediction(prediction_path, row); cached[record['id']] = row
                if len(cached) % 25 == 0 or len(cached) == len(records):
                    print(json.dumps({'model': name, 'completed': len(cached), 'expected': len(records),
                                      'latency_sec': round(row['latency_sec'], 2), 'at_utc': utc()}), flush=True)
        finally:
            for hook in handles:
                hook.remove()
            del model
            torch.cuda.empty_cache()
    require(len(cached) == len(records), 'Incomplete generation')
    attempts = [read_json(p) for p in sorted((output / 'loading_attempts').glob('*.json'))]
    require(attempts and all(a['configuration_sha256'] == config['configuration_sha256'] for a in attempts),
            'Loading audit missing or changed')
    if checkpoint is not None:
        require(all(a['adapter_verification']['all_saved_tensors_verified'] for a in attempts), 'Adapter reload unverified')
    marker = {'passed': True, 'model': name, 'count': len(records), 'configuration_sha256': config['configuration_sha256'],
              'adapter_sha256': config['checkpoint']['adapter_sha256'], 'generation_sha256': sha(prediction_path),
              'strict_format_valid': sum(r['parsed']['valid'] for r in cached.values()),
              'length_limited': sum(r['finish_reason'] == 'length' for r in cached.values()),
              'finish_reasons': dict(Counter(r['finish_reason'] for r in cached.values())),
              'total_saved_generation_seconds': sum(r['latency_sec'] for r in cached.values()),
              'created_utc': utc(), 'semantic_accuracy_evaluated': False, 'references_supplied_to_model': False}
    write_json(completed, marker, immutable=True)
    print(json.dumps(marker), flush=True)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', choices=MODELS, help='Default: epoch1 base epoch2')
    parser.add_argument('--gpus', default='0,1', help='One physical GPU pair authorized by the user, e.g. 2,3')
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true', help='CPU-only input/checkpoint validation; writes nothing')
    args = parser.parse_args()
    args.models = args.models or list(MODELS)
    if args.dry_run:
        for name in args.models:
            config, records, *_ = freeze(name, write=False)
            print(json.dumps({'model': name, 'records': len(records), 'checkpoint': config['checkpoint']['path'],
                              'configuration_sha256': config['configuration_sha256']}), flush=True)
        return
    indices = [int(x) for x in args.gpus.split(',')]
    require(len(indices) == 2 and len(set(indices)) == 2, 'Use exactly one GPU pair')
    for name in args.models:
        lock_path = OUT / 'predictions' / name / 'generation.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({'model': name, 'skipped': 'another worker holds this model'}), flush=True)
                continue
            try:
                run(name, args.poll_seconds, indices)
            except Exception as error:
                write_json(OUT / 'predictions' / name / 'failures' / f'{time.time_ns()}.json',
                           {'model': name, 'gpus': indices, 'error': repr(error),
                            'traceback': traceback.format_exc(), 'created_utc': utc()})
                raise


if __name__ == '__main__':
    main()
