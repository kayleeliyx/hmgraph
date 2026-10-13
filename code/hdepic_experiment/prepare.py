"""Freeze experiment identities and evaluation rules without changing data."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import subprocess
import time
import zipfile
from pathlib import Path

from .common import (PROJECT, RUN, MODEL, CONFIG_PATH, atomic_json, load_config,
                     load_records, read_json, read_jsonl, sha256, stable_hash)
from .metrics import protocol, blind_review_ids


def freeze(run=RUN, freeze_code=True):
    run = Path(run).resolve()
    config_path = run / 'frozen_config.json'
    config = load_config(config_path)
    images = {}
    for part in ('initial_sft', 'validation'):
        text = Path(config['paths']['dataset']) / 'text' / f'{part}.jsonl'
        info = config['data'][part]
        if 'text' not in info:
            assert not (run / 'predictions').exists(), 'Cannot alter frozen inputs after evaluation starts'
            info.update(text=str(text), text_sha256=sha256(text))
        assert info['text_sha256'] == sha256(text)
        records = read_jsonl(text)
        rows, scores, prompts = load_records(config, part)
        assert len(records) == info['count']
        assert [r['id'] for r in records] == [s['id'] for s in scores]
        assert dict(Counter(r['task'] for r in records)) == info['task_counts']
        for sidecar in scores:
            for image in sidecar['frames']:
                identity = (image['bytes'], image['sha256'])
                assert image['path'] not in images or images[image['path']] == identity
                images[image['path']] = identity
    image_started = time.perf_counter()
    def verify_image(item):
        name, (size, expected) = item
        path = Path(name)
        assert path.stat().st_size == size and sha256(path) == expected, f'Frozen source image changed: {path}'
        return size
    with ThreadPoolExecutor(max_workers=16) as pool:
        image_bytes = sum(pool.map(verify_image, images.items()))
    atomic_json(run / 'source_images_verified.json', {
        'passed': True, 'unique_images': len(images), 'image_bytes': image_bytes,
        'scope': 'initial_sft and validation only', 'identity_source': 'frozen scoring-sidecar bytes and SHA256',
        'elapsed_sec': time.perf_counter() - image_started})
    # Add only explicit resolved defaults already used by installed Swift.
    defaults = {'dataset_num_proc': 1, 'dataloader_num_workers': 2,
                'dataloader_persistent_workers': False, 'dataloader_prefetch_factor': 2,
                'weight_decay': 0.1, 'adam_beta1': 0.9, 'adam_beta2': 0.95,
                'adam_epsilon': 1e-8, 'max_grad_norm': 1.0,
                'ddp_find_unused_parameters': False, 'fp16': False}
    for key, value in defaults.items():
        if key in config['training']:
            assert config['training'][key] == value
        else:
            assert not (run / 'predictions').exists(), 'Cannot change resolved defaults after baseline starts'
            config['training'][key] = value
    atomic_json(config_path, config)
    validation = read_jsonl(config['data']['validation']['text'])
    exports, scoring, prompts = load_records(config, 'validation')
    frozen_eval = {
        'metric_protocol': protocol(), 'blind_review_ids': blind_review_ids(validation),
        'ordered_validation_ids': [s['id'] for s in scoring],
        'validation_prompts_sha256': config['data']['validation']['prompts_sha256'],
        'validation_scoring_sha256': config['data']['validation']['scoring_sha256'],
        'ordered_prompt_identities': [stable_hash({'id': s['id'], 'prompt': p,
                                                  'source_frames': s['frames']})
                                      for s, p in zip(scoring, prompts)],
        'frozen_config_sha256': sha256(config_path),
        'parser_sha256': sha256(Path(__file__).with_name('metrics.py')),
        'settings': config['evaluation'],
        'original_generation_config': read_json(MODEL / 'generation_config.json'),
        'request_config': {'max_tokens': 256, 'temperature': 0,
                           'num_beams': 1, 'stream': False},
        'end_of_response': {'eos_token_id': [151645, 151643], 'pad_token_id': 151643,
                            'template_stop_words': [],
                            'handling': 'installed Swift Qwen3-VL decoder; retain raw returned text and finish_reason'},
        'teacher_forced_loss': {
            'dtype': 'float32 cross entropy', 'shift': 'logit at p-1 predicts supervised label at p',
            'token_scope': 'all assistant target and exact ChatML termination suffix labels; ignore -100',
            'aggregation': 'arithmetic mean of per-example token mean NLL, separately Explain/Track',
            'selection': 'lowest complete mean per-example Explain NLL among epoch1/epoch2; tie epoch1'},
        'human_reason_review': 'pending', 'test_predictions': 0,
    }
    eval_path = run / 'frozen_evaluation.json'
    if eval_path.exists():
        assert read_json(eval_path) == frozen_eval, 'Frozen evaluation or parser changed'
    else:
        atomic_json(eval_path, frozen_eval)
    provenance = run / 'provenance'; provenance.mkdir(exist_ok=True)
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    required = {'ms-swift': '3.12.6', 'transformers': '4.57.1', 'torch': '2.6.0+cu124',
                'torchvision': '0.21.0+cu124', 'peft': '0.17.1',
                'accelerate': '1.10.1', 'qwen-vl-utils': '0.0.14'}
    assert all(packages.get(k) == v for k, v in required.items()), 'Installed verified environment changed'
    atomic_json(provenance / 'package_versions.json', packages)
    freeze_text = subprocess.check_output([config['paths']['python'], '-m', 'pip', 'freeze', '--all'], text=True)
    (provenance / 'pip-freeze.txt').write_text(freeze_text)
    identity_path = run / 'original_model_identity.json'
    identity = read_json(config['model']['identity_source'])
    assert identity['snapshot_path'] == config['model']['path'] and identity['revision'] == config['model']['revision']
    if identity_path.exists():
        assert read_json(identity_path) == identity
    for item in identity['files']:
        path = MODEL / item['filename']
        assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']
    if not identity_path.exists():
        atomic_json(identity_path, identity)
    if freeze_code:
        names = ('common.py', 'prepare.py', 'preflight.py', 'metrics.py', 'report.py',
                 'eval.py', 'train.py', 'train_audit.py', 'workflow.py')
        paths = [Path(__file__).with_name(n) for n in names]
        paths += [PROJECT / 'scripts/run_hdepic_sft_8b_v1.sh']
        paths += sorted((PROJECT / 'code/tests').glob('test_hdepic_experiment*.py'))
        assert all(p.is_file() for p in paths), 'Experiment implementation is incomplete'
        code_hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in paths}
        snapshot = provenance / 'code_snapshot.json'
        if snapshot.exists():
            assert read_json(snapshot)['source_sha256'] == code_hashes, 'Frozen experiment code changed; report the deviation before resume'
        else:
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
            diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=PROJECT)
            (provenance / 'existing_worktree.diff').write_bytes(diff)
            status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT)
            (provenance / 'existing_worktree_status.txt').write_bytes(status)
            with zipfile.ZipFile(provenance / 'experiment_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in paths:
                    archive.write(path, str(path.relative_to(PROJECT)))
            atomic_json(snapshot, {'git_head': head, 'source_sha256': code_hashes,
                'tracked_diff_sha256': sha256(provenance / 'existing_worktree.diff'),
                'source_archive_sha256': sha256(provenance / 'experiment_source.zip'),
                'existing_user_changes_preserved': True})
        atomic_json(run / 'preparation_completed.json', {
            'passed': True, 'frozen_config_sha256': sha256(config_path),
            'frozen_evaluation_sha256': sha256(eval_path), 'model_identity_sha256': sha256(identity_path),
            'input_counts': {p: config['data'][p]['count'] for p in ('initial_sft', 'validation')},
            'dataset_rebuilt': False, 'frames_decoded': 0, 'human_reason_label_review': 'pending'})
    print(json.dumps({'inputs_frozen': True, 'evaluation_frozen': True,
                      'code_frozen': freeze_code, 'run': str(run)}, indent=2), flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    freeze(args.run, freeze_code=not args.inputs_only)


if __name__ == '__main__':
    main()
