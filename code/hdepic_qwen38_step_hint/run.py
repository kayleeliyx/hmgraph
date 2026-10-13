"""GPU generation for the step-hint experiment; one worker per model and GPU pair.

Order per model: step_prediction, true_step, shuffled_step, predicted_step (built
from that model's own step_prediction answers). The model is loaded once and uses
exactly the Track run's loading, adapter verification and greedy decoding.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from code.hdepic_sft_qwen38.common import MODEL, MODEL_REPO, MODEL_REVISION, RUN as SFT_RUN, configure, read_json, rows, sha, write_json
from code.hdepic_sft_qwen38.evaluate import (GENERATION, append_prediction, checkpoint_identity, digest,
                                             load_cached, load_reloaded_model, require, stage_paths, utc)
from code.hdepic_qwen38_track_eval.generate import generate_one, load_base_model, wait_for_idle
from .prepare import OUT, predicted_step_prompts

VERSION = 'hdepic_qwen38_step_hint_generation_v1'
MODELS = ('base', 'epoch1')
CONDITIONS = ('step_prediction', 'true_step', 'shuffled_step', 'predicted_step')
METADATA = {'task': 'track', 'has_goal': True, 'partition': 'validation'}


def prompts(model, condition):
    if condition != 'predicted_step':
        return rows(OUT / 'inputs/prompts' / f'{condition}.jsonl')
    step = OUT / 'predictions' / model / 'step_prediction'
    require((step / 'completed.json').exists(), 'step_prediction must complete before predicted_step')
    return predicted_step_prompts(model, {r['id']: r for r in rows(step / 'generation.jsonl')})


def freeze(model, condition, records):
    checkpoint = None if model == 'base' else stage_paths(model, SFT_RUN)[0]
    info = ({'path': None, 'step': None, 'adapter_sha256': None, 'adapter_config_sha256': None}
            if checkpoint is None else checkpoint_identity(checkpoint, model, SFT_RUN))
    here = Path(__file__)
    code = [here, here.with_name('prepare.py'), here.parents[1] / 'hdepic_qwen38_track_eval/generate.py',
            here.parents[1] / 'hdepic_sft_qwen38/evaluate.py', here.parents[1] / 'hdepic_qwen38_baseline/processor.py']
    config = {'version': VERSION, 'model': model, 'condition': condition, 'model_repo': MODEL_REPO,
              'model_revision': MODEL_REVISION, 'base_path': str(MODEL), 'checkpoint': info,
              'count': len(records), 'ordered_ids': [r['id'] for r in records],
              'ordered_prompt_sha256': digest([r['prompt_sha256'] for r in records]),
              'inputs_manifest_sha256': sha(OUT / 'inputs/manifest.json'), 'generation': GENERATION,
              'settings': 'identical to runs/hdepic_qwen38_27b_track_v1: native BF16 on one GPU pair, '
                          'balanced device map, SDPA, 16 images, no thinking, greedy, 256 new tokens',
              'seed': 20261006,
              'precision': {'base': 'bfloat16', 'adapter_storage': 'float32' if checkpoint else None,
                            'adapter_compute': 'BF16 autocast scoped to LoRA A/B Linear leaves' if checkpoint else None,
                            'native_math': 'No global autocast; preserve native FP32 recurrence operations'},
              'packages_sha256': sha(SFT_RUN / 'environment/packages.txt'),
              'code_sha256': {str(p.relative_to(here.parents[1])): sha(p) for p in code},
              'privileged_input': {'step_prediction': 'recipe name and step list',
                                   'true_step': 'annotated current recipe step', 'shuffled_step': 'another query step',
                                   'predicted_step': "recipe step chosen by this model's step_prediction"}[condition],
              'training': False, 'test_access': False}
    config['configuration_sha256'] = digest(config)
    output = OUT / 'predictions' / model / condition
    write_json(output / 'frozen_evaluation.json', config, immutable=True)
    return config, checkpoint, output


def run(model, indices, poll_seconds):
    configure()
    loaded = None
    for condition in CONDITIONS:
        records = prompts(model, condition)
        config, checkpoint, output = freeze(model, condition, records)
        path, done = output / 'generation.jsonl', output / 'completed.json'
        cached = load_cached(path, records, config)
        if done.exists():
            marker = read_json(done)
            require(marker['configuration_sha256'] == config['configuration_sha256']
                    and marker['generation_sha256'] == sha(path) and len(cached) == len(records), 'Completed condition changed')
            continue
        require(all(r['status'] == 'completed' for r in cached.values()), 'Saved failure requires diagnosis')
        if loaded is None:
            from code.hdepic_sft_qwen38.storage import validate_storage
            validate_storage(require_space=True, run=OUT)
            resources = wait_for_idle(poll_seconds, indices)
            os.environ.update(CUDA_VISIBLE_DEVICES=','.join(resources['uuids']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                              OMP_NUM_THREADS='1')
            import torch
            from code.hdepic_qwen38_baseline.processor import load_processor
            require(torch.cuda.device_count() == 2, 'Expected exactly one GPU pair')
            if checkpoint is None:
                net, audit = load_base_model(config); handles = []
            else:
                net, audit, handles = load_reloaded_model(checkpoint, config)
                require(audit['adapter_verification']['all_saved_tensors_verified'], 'Adapter reload unverified')
            audit['resources'] = resources
            audit_path = OUT / 'predictions' / model / 'loading_attempts' / f'{time.time_ns()}.json'
            write_json(audit_path, audit)
            loaded = (net, load_processor(MODEL), handles, audit_path.name)
        net, processor, _, audit_name = loaded
        for record in (r for r in records if r['id'] not in cached):
            try:
                row = generate_one(net, processor, record, METADATA, config)
            except Exception as error:
                append_prediction(path, {'id': record['id'], 'row_index': record['row_index'], 'status': 'failed',
                    'prompt_sha256': record['prompt_sha256'], 'configuration_sha256': config['configuration_sha256'],
                    'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                    'adapter_sha256': config['checkpoint']['adapter_sha256'],
                    'adapter_config_sha256': config['checkpoint']['adapter_config_sha256'],
                    'error': repr(error), 'traceback': traceback.format_exc(), 'created_utc': utc()})
                raise
            append_prediction(path, row); cached[record['id']] = row
            if len(cached) % 50 == 0 or len(cached) == len(records):
                print(json.dumps({'model': model, 'condition': condition, 'completed': len(cached),
                                  'expected': len(records), 'at_utc': utc()}), flush=True)
        write_json(done, {'passed': True, 'model': model, 'condition': condition, 'count': len(records),
                          'configuration_sha256': config['configuration_sha256'], 'generation_sha256': sha(path),
                          'loading_attempt': audit_name,
                          'length_limited': sum(r['finish_reason'] == 'length' for r in cached.values()),
                          'total_saved_generation_seconds': sum(r['latency_sec'] for r in cached.values()),
                          'created_utc': utc()}, immutable=True)
    if loaded is not None:
        for hook in loaded[2]:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--gpus', required=True, help='One physical GPU pair authorized by the user, e.g. 2,3')
    parser.add_argument('--poll-seconds', type=int, default=300)
    args = parser.parse_args()
    indices = [int(x) for x in args.gpus.split(',')]
    require(len(indices) == 2 and len(set(indices)) == 2, 'Use exactly one GPU pair')
    lock_path = OUT / 'predictions' / args.model / 'generation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.model, indices, args.poll_seconds)
        except Exception as error:
            write_json(OUT / 'predictions' / args.model / 'failures' / f'{time.time_ns()}.json',
                       {'model': args.model, 'gpus': indices, 'error': repr(error),
                        'traceback': traceback.format_exc(), 'created_utc': utc()})
            raise


if __name__ == '__main__':
    main()
