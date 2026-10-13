"""Sequential, allocation-checked experiment stages with identity-safe resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from .common import (PROJECT, RUN, PYTHON, GPU_UUIDS, atomic_json, configure_environment,
                     load_config, read_json, read_jsonl, require_idle_gpus, sha256, stable_hash)
from .prepare import freeze


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluation_complete(run, stage):
    marker = run / 'predictions' / stage / 'completed.json'
    if not marker.exists():
        return False
    saved = read_json(marker)
    assert saved['stage'] == stage
    assert saved['passed'] is True and saved['partition'] == 'validation'
    assert saved['records'] == saved['generation_records'] == saved['loss_records'] == 2209
    assert saved['frozen_config_sha256'] == sha256(run / 'frozen_config.json')
    assert saved['frozen_evaluation_sha256'] == sha256(run / 'frozen_evaluation.json')
    metadata = read_json(run / 'predictions' / stage / 'metadata.json')
    identity = metadata['identity']
    assert metadata['run_key'] == saved['run_key'] == stable_hash(identity)
    assert identity['stage'] == stage
    assert identity['frozen_config_sha256'] == saved['frozen_config_sha256']
    assert identity['frozen_evaluation_sha256'] == saved['frozen_evaluation_sha256']
    config = load_config(run)
    assert identity['data'] == config['data']['validation']
    model = identity['model']
    assert model['stage'] == stage and model['revision'] == config['model']['revision']
    assert model['original_path'] == config['model']['path']
    assert model['identity_sha256'] == saved['model_identity_sha256']
    assert model['identity_sha256'] == stable_hash({k: v for k, v in model.items() if k != 'identity_sha256'})
    if stage == 'base':
        assert model['adapter'] is None
    else:
        index = read_json(run / 'training/checkpoint_index.json')
        assert model['adapter']['path'] == index[stage]
        for name, digest in model['adapter']['files_sha256'].items():
            assert sha256(Path(index[stage]) / name) == digest
    for path, digest in identity['code_sha256'].items():
        assert sha256(path) == digest
    frozen = read_json(run / 'frozen_evaluation.json')
    ordered_ids = frozen['ordered_validation_ids']
    from .eval import source_line_hashes
    source_hashes = {k: source_line_hashes(config['data']['validation'][k])
                     for k in ('prompts', 'export', 'scoring')}
    for kind in ('generation', 'loss'):
        path = run / 'predictions' / stage / f'{kind}.jsonl'
        assert saved[f'{kind}_sha256'] == sha256(path)
        rows = read_jsonl(path)
        assert len(rows) == len({r['id'] for r in rows}) == 2209
        assert all(r['status'] == 'completed' for r in rows)
        assert [r['id'] for r in rows] == ordered_ids
        for i, row in enumerate(rows):
            assert row['row_index'] == i and row['partition'] == 'validation' and row['stage'] == stage
            assert row['run_key'] == saved['run_key']
            assert row['model_identity_sha256'] == saved['model_identity_sha256']
            assert row['frozen_config_sha256'] == saved['frozen_config_sha256']
            assert row['frozen_evaluation_sha256'] == saved['frozen_evaluation_sha256']
            assert row['prompt_sha256'] == source_hashes['prompts'][i]
            expected_key = stable_hash({'id': row['id'], 'row_index': i,
                'prompt_sha256': source_hashes['prompts'][i],
                'export_sha256': source_hashes['export'][i],
                'scoring_sha256': source_hashes['scoring'][i], 'run_key': saved['run_key']})
            assert row['record_key'] == expected_key
    return True


def training_complete(run):
    markers = [run / 'training' / f'training_complete_rank{r}.json' for r in (0, 1)]
    if not all(path.exists() for path in markers):
        return False
    from .train_audit import validate_checkpoint
    config_sha = sha256(run / 'frozen_config.json')
    index = read_json(run / 'training/checkpoint_index.json')
    assert index['frozen_config_sha256'] == config_sha
    for rank, path in enumerate(markers):
        saved = read_json(path)
        assert saved['passed'] and saved['global_step'] == 1818 and saved['epoch'] == 2
        assert saved['rank'] == rank and saved['frozen_config_sha256'] == config_sha
        assert saved['final_checkpoint'] == index['epoch2']
    for epoch in (1, 2):
        saved = validate_checkpoint(index[f'epoch{epoch}'], config_sha)
        assert saved['global_step'] == epoch * 909
    return True


def child(run, name, command, devices=None):
    logs = run / 'logs'; logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log = logs / f'{name}-{stamp}.log'
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(PROJECT)
    if devices is not None:
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_UUIDS[i] for i in devices)
    else:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    entry = {'stage': name, 'command_argv': command, 'exact_command': shlex.join(command),
             'CUDA_VISIBLE_DEVICES': environment['CUDA_VISIBLE_DEVICES'],
             'physical_indices': devices or [], 'log': str(log),
             'started_at_utc': now(), 'frozen_config_sha256': sha256(run / 'frozen_config.json')}
    manifest = logs / f'{name}-{stamp}.json'
    atomic_json(manifest, entry)
    print(json.dumps({'starting': name, 'log': str(log), 'physical_gpu_indices': devices or []}), flush=True)
    started = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=PROJECT, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode, elapsed_sec=time.perf_counter() - started,
                 finished_at_utc=now())
    atomic_json(manifest, entry)
    if result.returncode:
        print(log.read_text()[-12000:], flush=True)
        raise RuntimeError(f'{name} failed with exit{result.returncode}; preserved log: {log}')
    print(json.dumps({'finished': name, 'elapsed_sec': entry['elapsed_sec']}), flush=True)
    return entry


def idle_pair(run, phase):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_UUIDS[i] for i in (0, 1))
    return require_idle_gpus((0, 1), run, phase)


def idle_evaluation_device(run, phase, require_pair=False):
    # Baseline is allowed only when the allocated training pair is available.
    if require_pair:
        idle_pair(run, phase)
        return 0
    failures = []
    for index in (0, 1):
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUIDS[index]
        try:
            require_idle_gpus((index,), run, f'{phase}_{index}')
            return index
        except RuntimeError as error:
            failures.append(str(error))
    raise RuntimeError('; '.join(failures))


def prepare(run):
    freeze(run, freeze_code=True)
    audit_path = run / 'preflight/preflight_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        assert audit['passed'] and audit['records_verified'] == 16752
        assert audit['frozen_config_sha256_at_preflight'] == sha256(run / 'frozen_config.json')
        assert audit['provenance']['files_sha256'][str(PROJECT / 'code/hdepic_experiment/preflight.py')] == sha256(PROJECT / 'code/hdepic_experiment/preflight.py')
        assert audit['record_audit_sha256'] == sha256(audit['record_audit'])
        print('Reusing complete16752-record processor audit', flush=True)
    else:
        child(run, 'preflight', [str(PYTHON), '-m', 'code.hdepic_experiment.preflight',
                                '--run', str(run), '--workers', '16'])


def evaluate(run, stage, retry_errors=False):
    command = [str(PYTHON), '-m', 'code.hdepic_experiment.eval', '--run', str(run), '--stage', stage]
    if stage != 'base':
        assert training_complete(run), 'Both training epochs must finish before checkpoint evaluation'
        index = read_json(run / 'training/checkpoint_index.json')
        command += ['--checkpoint', index[stage]]
    if retry_errors:
        command += ['--retry-errors']
    # The evaluator validates the entire prediction identity even when complete.
    if evaluation_complete(run, stage):
        child(run, f'{stage}_verify_cache', command, [0])
        return
    device = idle_evaluation_device(run, stage, require_pair=stage == 'base')
    child(run, stage, command, [device])
    assert evaluation_complete(run, stage)


def train(run):
    assert evaluation_complete(run, 'base'), 'The original-model baseline must be complete'
    if training_complete(run):
        print('Both epochs already complete at1818 optimizer steps; no training repeated', flush=True)
        return
    idle_pair(run, 'training')
    command = [str(PYTHON), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
               '-m', '--', 'code.hdepic_experiment.train', '--run', str(run),
               '--config', str(run / 'frozen_config.json')]
    execution = child(run, 'training', command, [0, 1])
    assert training_complete(run)
    ranks = [read_json(run / 'training' / f'training_complete_rank{i}.json') for i in (0, 1)]
    atomic_json(run / 'training/runtime.json', {'execution': execution, 'rank_audits': ranks,
        'physical_indices': [0, 1], 'physical_uuids': GPU_UUIDS,
        'optimizer_steps': 1818, 'epochs': 2, 'effective_batch_size': 16})


def report(run):
    assert training_complete(run) and all(evaluation_complete(run, s) for s in ('base', 'epoch1', 'epoch2'))
    from .report import build_report
    config = load_config(run)
    result = build_report(run, Path(config['paths']['dataset']))
    selected = result['checkpoint_selection']['selected_stage']
    assert selected in ('epoch1', 'epoch2')
    index = read_json(run / 'training/checkpoint_index.json')
    execution = {
        'passed': True, 'execution_success': True, 'quality_validated': False,
        'human_reason_label_review': 'pending', 'selected_stage': selected,
        'selected_checkpoint': index[selected], 'selection': result['checkpoint_selection'],
        'optimizer_steps': 1818, 'initial_sft_records': 14543, 'validation_records': 2209,
        'skipped_records': 0, 'test_predictions': 0, 'extra_sft_or_grpo_started': False,
        'smoke_adapter_loaded': False, 'full_start': config['model'],
        'frozen_config_sha256': sha256(run / 'frozen_config.json'),
        'frozen_evaluation_sha256': sha256(run / 'frozen_evaluation.json'),
        'metrics_sha256': sha256(run / 'metrics.json'), 'finished_at_utc': now()}
    atomic_json(run / 'experiment_completed.json', execution)
    lines = ['# HD-EPIC initial SFT experiment', '',
        'Execution completed: untouched pinned base validation, two SFT epochs, both checkpoint validations, and a blinded review CSV. Human reason-label review is pending; semantic why correctness is unmeasured.', '',
        f'Selected checkpoint: `{index[selected]}` by the lowest mean per-example Explain validation target NLL among the two epoch checkpoints. Track is reported separately. See `validation_report.md` and `metrics.json` for the complete comparison.', '',
        'Training: physical GPUs0/1, BF16, text LoRA16/32/0, frozen vision and aligner, batch1 per device, accumulation8, effective batch16,1818 optimizer steps. Every initial-SFT record is covered; standard distributed sampling pads one explicitly logged shuffled record per epoch. Dataset files, labels, prompts and split are unchanged.', '',
        'The launch and resume command is identical; completed stages are verified and skipped:', '',
        '```bash', f'bash {PROJECT}/scripts/run_hdepic_sft_8b_v1.sh all', '```', '',
        'Stage commands: `prepare`, `baseline`, `train`, `evaluate`, `report`, `status`. Evaluation errors require an explicit `--retry-evaluation-errors` flag; failures and output identities are preserved. Training resumes only from a checkpoint with complete, matching optimizer/scheduler/RNG/trainer manifests.', '',
        'Exact subprocess commands and wall times are saved in `logs/*.json`; training memory/finite gradient/update audits are in `training/`, evaluation raw text/NLL and model identities are in `predictions/`. Original checkpoint hashes, packages and source archive/diff are under `provenance/`. No test-set inference, extra-SFT or GRPO was run.']
    (run / 'experiment_results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(execution, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='all', choices=('all', 'prepare', 'baseline', 'train', 'evaluate', 'report', 'status'))
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--retry-evaluation-errors', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    assert run == RUN, 'This launcher is bound to the frozen v1 experiment'
    configure_environment()
    with (run / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'status':
            print(json.dumps({'run': str(run), 'preflight': (run / 'preflight/preflight_audit.json').exists(),
                'evaluation_complete': {s: evaluation_complete(run, s) for s in ('base', 'epoch1', 'epoch2')},
                'training_complete': training_complete(run)}, indent=2), flush=True)
            return
        prepare(run)
        if args.stage in ('all', 'baseline'):
            evaluate(run, 'base', args.retry_evaluation_errors)
        if args.stage in ('all', 'train'):
            train(run)
        if args.stage in ('all', 'evaluate'):
            for stage in ('epoch1', 'epoch2'):
                evaluate(run, stage, args.retry_evaluation_errors)
        if args.stage in ('all', 'report'):
            report(run)


if __name__ == '__main__':
    main()
