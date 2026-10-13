"""CPU-only, evidence-gated reporting for the isolated Qwen3.8 SFT run.

No model is imported or loaded. Saved validation outputs are joined to frozen
references only here, after generation. Existing artifacts remain read-only;
derived reports and content-addressed provenance snapshots belong to this run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

from .common import BASELINE, ENV, GPU_UUIDS, MODEL, MODEL_REPO, MODEL_REVISION, PROJECT, RUN, read_json, rows, sha, write_json
from code.hdepic_experiment.metrics import aggregate_losses, score_predictions, select_checkpoint
from code.hdepic_review_followup.fields import score_fields

VERSION = 'hdepic_qwen38_sft_execution_report_v1'
STAGES = ('base', 'epoch1', 'epoch2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optional(path, default=None):
    return read_json(path) if Path(path).is_file() else default


def read_rows(path):
    return rows(path) if Path(path).is_file() else []


def file_info(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} if path.is_file() else None


def without_examples(value):
    return {k: v for k, v in value.items() if k != 'per_example'}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _immutable_bytes(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f'Provenance snapshot changed: {path}')
    else:
        path.write_bytes(content)


def snapshot_provenance(run):
    """Snapshot current code/diff/packages; distinguish this from execution identity."""
    run = Path(run)
    paths = list(sorted((PROJECT / 'code/hdepic_sft_qwen38').glob('*.py')))
    paths += list(sorted((PROJECT / 'code/tests').glob('test_hdepic_sft_qwen38*.py')))
    paths += [PROJECT / 'code/hdepic_qwen38_baseline/processor.py',
              PROJECT / 'code/hdepic_qwen38_baseline/data.py',
              PROJECT / 'code/hdepic_experiment/metrics.py',
              PROJECT / 'code/hdepic_review_followup/fields.py',
              PROJECT / 'scripts/run_hdepic_sft_qwen38_27b_v1.sh',
              PROJECT / 'scripts/prepare_hdepic_qwen38_sft_environment.sh']
    content = {str(p.relative_to(PROJECT)): p.read_bytes() for p in paths if p.is_file()}
    if (run / 'README.md').is_file():
        content['run_README.md'] = (run / 'README.md').read_bytes()
    commands = {'git_head.txt': ['git', 'rev-parse', 'HEAD'],
                'git_status.txt': ['git', 'status', '--porcelain=v1', '--untracked-files=normal'],
                'tracked_code_diff.patch': ['git', 'diff', '--binary', 'HEAD', '--', 'code', 'scripts']}
    git_status = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, check=False)
        content[name] = result.stdout
        git_status[name] = {'command': command, 'returncode': result.returncode,
                            'stderr': result.stderr.decode(errors='replace')}
    for name in ('packages.txt', 'compatibility.json', 'peft_wheel_manifest.json'):
        path = run / 'environment' / name
        if path.is_file():
            content['environment/' + name] = path.read_bytes()
    recovery_path = run / 'recovery/finalization_recovery.json'
    if recovery_path.is_file():
        content['recovery/finalization_recovery.json'] = recovery_path.read_bytes()
    for name in ('config.json', 'generation_config.json', 'preprocessor_config.json',
                 'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors.index.json'):
        path = BASELINE / 'model_metadata' / name
        if path.is_file():
            content['model_metadata/' + name] = path.read_bytes()
    for relative in ('model_identity.json', 'audit/model_weight_integrity.json'):
        path = BASELINE / relative
        if path.is_file():
            content['original_base_evidence/' + relative] = path.read_bytes()
    inventory = {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                 for name, value in sorted(content.items())}
    identity = hashlib.sha256(_json_bytes(inventory)).hexdigest()
    directory = run / 'report/provenance' / identity
    for name, value in content.items():
        _immutable_bytes(directory / name, value)
    manifest = {'snapshot_sha256': identity, 'files': inventory, 'git_commands': git_status,
                'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'model_path': str(MODEL),
                'snapshot_scope': 'Current reporting-time source and package evidence; actual executing identities are independently saved in training/checkpoint markers',
                'weights_copied': False, 'git_state_modified': False}
    write_json(directory / 'manifest.json', manifest, immutable=True)
    return {'directory': str(directory), 'manifest': file_info(directory / 'manifest.json'),
            'snapshot_sha256': identity, 'git_head': content['git_head.txt'].decode().strip(),
            'package_versions': file_info(run / 'environment/packages.txt'),
            'base_weight_hash_evidence': file_info(BASELINE / 'audit/model_weight_integrity.json')}


def _check_identity(identity, config, run):
    require(identity['config_sha256'] == sha(Path(run) / 'frozen_config.json'), 'Completed artifact configuration identity changed')
    require(identity['model_revision'] == MODEL_REVISION, 'Completed artifact model revision changed')
    require(identity['data_sha256'] == {k: v['export_sha256'] for k, v in config['data'].items()},
            'Completed artifact dataset identity changed')


def loss_summary(stage, records, config, run):
    folder = Path(run) / 'validation' / stage
    marker = optional(folder / 'completed.json')
    saved = read_rows(folder / 'loss.jsonl')
    normalized = [{'id': row['id'], 'task': row['task'], 'mean_nll': row['nll'],
                   'answer_tokens': row['supervised_tokens'],
                   'nll_sum': row['nll'] * row['supervised_tokens']} for row in saved]
    scores = aggregate_losses(records, normalized)
    result = {'completion_marker_present': marker is not None, 'completed': False,
              'loss_file': file_info(folder / 'loss.jsonl'), 'saved_records': len(saved),
              **without_examples(scores)}
    if marker is not None:
        _check_identity(marker['identity'], config, run)
        require(marker['stage'] == stage and marker['records'] == 2209, 'Validation completion cohort changed')
        require(sha(folder / 'loss.jsonl') == marker['loss_sha256'], 'Completed validation loss file changed')
        require([r['id'] for r in saved] == [r['id'] for r in records], 'Validation loss IDs/order changed')
        require(scores['scopes']['all']['complete'], 'Invalid/incomplete completed validation losses')
        for task in ('explain', 'track'):
            require(marker['tasks'][task]['count'] == scores['scopes'][task]['expected_examples']
                    and math.isclose(marker['tasks'][task]['mean_per_example_nll'],
                                     scores['scopes'][task]['mean_per_example_nll'], rel_tol=1e-12),
                    'Saved validation aggregate differs from recomputation')
        result.update(completed=True, marker=marker, completion_marker=file_info(folder / 'completed.json'))
    return result


def generation_summary(stage, references, prompt_records, run):
    folder = BASELINE / 'predictions' if stage == 'base' else Path(run) / 'predictions' / stage
    path = folder / 'generation.jsonl'; predictions = read_rows(path)
    marker = optional(folder / 'completed.json')
    frozen = optional(BASELINE / 'frozen_config.json' if stage == 'base' else folder / 'frozen_evaluation.json')
    by_id = {row['id']: row for row in predictions}
    require(len(by_id) == len(predictions), 'Duplicate saved generation ID')
    expected = {p['id']: p for p in prompt_records}
    require(set(by_id) <= set(expected), 'Saved output outside fixed validation Explain IDs')
    for rid, row in by_id.items():
        require(row['model_repo'] == MODEL_REPO and row['model_revision'] == MODEL_REVISION,
                'Saved output model identity differs')
        require(row['prompt_sha256'] == expected[rid]['prompt_sha256'], 'Saved output prompt differs from fixed baseline')
        require(frozen is not None and row['configuration_sha256'] == frozen['configuration_sha256'],
                'Saved output frozen generation identity differs')
    completed = False
    if marker is not None:
        require(marker['passed'] and marker['count'] == 228 and len(predictions) == 228
                and all(row['status'] == 'completed' for row in predictions), 'Invalid completed generation cohort')
        require(marker['generation_sha256'] == sha(path), 'Completed raw predictions changed')
        require(marker['configuration_sha256'] == frozen['configuration_sha256'], 'Completed generation config changed')
        completed = True
    strict = score_predictions(references, predictions)
    independent = score_fields(references, predictions)
    saved_latencies = [r['latency_sec'] for r in predictions if r.get('status') == 'completed']
    return {'completed': completed, 'completion_marker_present': marker is not None,
            'prediction_file': file_info(path), 'saved_records': len(predictions), 'expected_records': 228,
            'strict': without_examples(strict), 'independent_fields': without_examples(independent),
            'saved_successful_generation_seconds': sum(saved_latencies),
            'median_saved_generation_seconds': statistics.median(saved_latencies) if saved_latencies else None,
            'completion': marker, 'semantic_accuracy_measured': False}, strict['per_example'], independent['per_example']


def _log_summary(folder):
    result = {}
    for rank in (0, 1):
        path = Path(folder) / f'optimizer_steps_rank{rank}.jsonl'; values = read_rows(path)
        require([v['step'] for v in values] == list(range(1, len(values) + 1)), 'Optimizer log has duplicate/missing/reordered steps')
        for row in values:
            require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) for k in
                        ('loss', 'grad_norm_before_clip', 'seconds', 'lr_used', 'lr_next')),
                    'Nonfinite recorded training loss/gradient/time/LR')
            require(row['finite_grad_tensors'] == 992, 'Training gradient coverage changed')
        result[str(rank)] = {'log': file_info(path), 'recorded_steps': len(values),
                            'last_optimizer_step': values[-1] if values else None,
                            'finite_loss_and_gradients_in_all_saved_steps': bool(values),
                            'sum_optimizer_step_seconds': sum(v['seconds'] for v in values),
                            'median_optimizer_step_seconds': statistics.median(v['seconds'] for v in values) if values else None,
                            'peak_allocated_bytes': max((v['peak_allocated_bytes'] for v in values), default=None),
                            'peak_reserved_bytes': max((v['peak_reserved_bytes'] for v in values), default=None),
                            'loss_range': [min(v['loss'] for v in values), max(v['loss'] for v in values)] if values else None,
                            'grad_norm_range': [min(v['grad_norm_before_clip'] for v in values), max(v['grad_norm_before_clip'] for v in values)] if values else None,
                            'resolved': optional(Path(folder) / f'resolved_rank{rank}.json'),
                            'parameter_updates': optional(Path(folder) / f'parameter_updates_rank{rank}.json')}
        updates = result[str(rank)]['parameter_updates']
        if updates:
            result[str(rank)]['parameter_updates'] = {k: v for k, v in updates.items() if k != 'tensors'}
            result[str(rank)]['parameter_update_file'] = file_info(Path(folder) / f'parameter_updates_rank{rank}.json')
    return result


def _training_summary(config, run):
    folder = Path(run) / 'training/main'; marker = optional(folder / 'completed.json')
    result = {'completed': False, 'completion_marker_present': marker is not None,
              'expected_optimizer_steps': 1818, 'steps_per_epoch': 909,
              'microbatches_per_rank_per_epoch': 7272, 'ranks': _log_summary(folder),
              'completion': marker, 'resumable_checkpoints': []}
    for path in sorted((folder / 'checkpoints').glob('checkpoint-*')):
        item = optional(path / 'completed.json')
        if item:
            _check_identity(item['identity'], config, run)
            result['resumable_checkpoints'].append({'path': str(path), 'real_storage_path': str(path.resolve()),
                'step': item['state']['step'], 'completion_marker': file_info(path / 'completed.json'),
                'checkpoint_file_inventory': item['files']})
    if marker:
        _check_identity(marker['identity'], config, run)
        require(marker['passed'] and marker['smoke'] is False and marker['optimizer_steps'] == 1818,
                'Full training completion marker is invalid')
        require(len(marker['ranks']) == 2 and all(r['optimizer_steps'] == 1818 and r['passed'] for r in marker['ranks']),
                'Both training ranks must complete')
        require(all(result['ranks'][str(r)]['recorded_steps'] == 1818 for r in (0, 1)), 'Full optimizer log incomplete')
        for rank in (0, 1):
            completed_rank = next(r for r in marker['ranks'] if r['rank'] == rank)
            for name in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                result['ranks'][str(rank)][name] = max(result['ranks'][str(rank)][name], completed_rank[name])
            proof = result['ranks'][str(rank)]['parameter_updates']
            require(proof and proof['all_992_local_lora_tensors_changed']
                    and proof['frozen_sample_hashes_unchanged'] and proof['frozen_optimizer_membership'] == 0,
                    'Completed training parameter proof missing')
        result['completed'] = True
    return result


def training_runtime(training, results, recovery=None):
    """Keep the real optimizer-run duration separate from finalization resume.

    A completion marker written by a zero-update resume describes that short
    process only. It must never replace the elapsed original training launch.
    """
    launches = sorted((x for x in results if x.get('stage') == 'train'),
                      key=lambda x: x['started_unix'])
    recovery = recovery or {}
    original = recovery.get('original_launch_result')
    if original is None and launches:
        original = launches[0]
    resumed = recovery.get('zero_update_resume', {})
    finalization = resumed.get('launch_result')
    marker = training.get('completion') or {}
    rank_markers = marker.get('ranks', [])
    marker_times = [x['training_and_validation_seconds'] for x in rank_markers
                    if 'training_and_validation_seconds' in x]
    if resumed.get('status') == 'completed':
        require(resumed.get('additional_optimizer_steps') == 0
                and resumed.get('before_steps') == resumed.get('after_steps') == 1818,
                'Finalization resume changed the optimizer-step count')
        require(finalization is not None and finalization.get('returncode') == 0,
                'Completed zero-update resume lacks a successful launch result')
        integrity = recovery.get('logs_and_checkpoint_integrity', {})
        require(integrity.get('unchanged') is True and integrity.get('before') == integrity.get('after'),
                'Zero-update resume changed original optimizer/checkpoint evidence')
    return {
        'original_optimizer_run_launch': original,
        'original_optimizer_run_wall_seconds': original.get('elapsed_seconds') if original else None,
        'original_optimizer_run_returncode': original.get('returncode') if original else None,
        'zero_update_finalization_status': resumed.get('status', 'not_applicable'),
        'zero_update_finalization_launch': finalization,
        'zero_update_finalization_wall_seconds': finalization.get('elapsed_seconds') if finalization else None,
        'all_saved_training_launches': launches,
        'all_saved_training_launch_wall_seconds': sum(x['elapsed_seconds'] for x in launches),
        'completion_marker_attempt_seconds': max(marker_times) if marker_times else None,
        'completion_marker_duration_is_full_original_run': not bool(recovery),
        'optimizer_step_seconds_rank0': training['ranks']['0']['sum_optimizer_step_seconds'],
        'optimizer_step_seconds_rank1': training['ranks']['1']['sum_optimizer_step_seconds'],
        'duration_definition': 'Original launch includes loading, all optimizer steps, validation, checkpointing and final audit; zero-update recovery is reported separately. Optimizer step sums exclude validation/checkpoint overhead.',
    }


def _smoke_summary(run):
    result = {'passed_gate': optional(Path(run) / 'smoke/passed.json'), 'attempts': []}
    for folder in sorted((Path(run) / 'smoke').glob('attempt-*')):
        diagnostics = []
        for diagnostic_path in sorted((folder / 'parity_diagnostic').glob('record0_rank*.json')):
            item = read_json(diagnostic_path)
            diagnostics.append({'file': file_info(diagnostic_path),
                                **{k: item[k] for k in ('classification', 'diagnosis', 'selected_training_loss',
                                   'native_full_loss', 'mean_token_nll', 'selected_vs_full_hidden',
                                   'selected_vs_repeat_logits', 'selected_vs_short_bf16_on_full_hidden',
                                   'native_full_vs_same_logits_masked_ce_absolute_difference',
                                   'all_992_local_adapter_hashes_unchanged', 'no_optimizer_updates') if k in item}})
        result['attempts'].append({'attempt': folder.name, 'completion': optional(folder / 'completed.json'),
                                  'ranks': _log_summary(folder),
                                  'fresh_fsdp_reload': optional(folder / 'fresh_fsdp_reload.json'),
                                  'generation_reload': optional(folder / 'reload/completed.json'),
                                  'loss_objective_diagnostics': diagnostics,
                                  'four_training_examples': file_info(folder / 'reload/reference_comparison.md')})
    return result


def _selection(losses, training, config, run):
    eligible = {stage: value for stage, value in losses.items() if value['completed']}
    result = select_checkpoint(eligible)
    if result['selected_stage'] is not None and not training['completed']:
        result.update(selected_stage=None, status='pending_full_training_completion')
        result.pop('selected_metric', None)
        return result
    if result['selected_stage'] is not None:
        from .checkpoints import validate_checkpoint
        for stage, step in (('epoch1', 909), ('epoch2', 1818)):
            path = Path(run) / 'training/main/checkpoints' / f'checkpoint-{step}'
            marker = read_json(path / 'completed.json'); _check_identity(marker['identity'], config, run)
            require(marker['state']['step'] == step, 'Epoch checkpoint step changed')
            validate_checkpoint(path, marker['identity'])
        step = 909 if result['selected_stage'] == 'epoch1' else 1818
        result['checkpoint'] = str(Path(run) / 'training/main/checkpoints' / f'checkpoint-{step}')
        result['meaning'] = 'Lowest complete Explain assistant-target NLL, including goal/next/termination; not semantic WHY accuracy'
    return result


def _nll(value, task):
    scope = value['scopes'][task]
    return f"{scope['mean_per_example_nll']:.6f}" if value['completed'] else 'pending'


def _ratio(item, numerator='exact_matches', denominator='denominator'):
    return f"{item[numerator]}/{item[denominator]}"


def _render(summary):
    training = summary['training']; selected = summary['selection']; pf = summary['preflight'] or {}
    lines = ['# HD-EPIC Qwen3.8-27B SFT execution report', '',
             f"Status: **{summary['status']}**. Full training completion marker: **{training['completion_marker_present']}**. "
             f"Saved optimizer steps: rank0={training['ranks']['0']['recorded_steps']}, rank1={training['ranks']['1']['recorded_steps']}, expected=1,818 each.", '',
             'Independent human reason-label review remains pending. Labels are provisional. Execution, format, and lexical diagnostics do not establish semantic WHY correctness.', '',
             f"Model: `{MODEL_REPO}`, revision `{MODEL_REVISION}`. Untuned base weights are the starting point; the disposable smoke adapter is excluded from full training.", '',
             f"CPU preprocessing: passed={pf.get('passed', False)}, processed/retained={pf.get('processed', 'pending')}/{pf.get('retained', 'pending')}, "
             f"failures={pf.get('failed', 'pending')}, truncations={pf.get('truncated', 'pending')}, complete sequence length "
             f"{pf.get('min_sequence_length', 'pending')}–{pf.get('max_sequence_length', 'pending')} tokens. "
             'All successful records retain 16 separate 384×384 images at 144 visual tokens each. Only the complete original assistant target plus one im_end are supervised; the native nonthinking prefix is masked.', '',
             '## Validation and selection', '',
             'Teacher-forced NLL covers all 2,209 validation rows (228 Explain / 1,981 Track). Saved-answer generation covers the same 228 validation Explain IDs as the completed untuned 27B baseline; this run does not generate Track or test answers.', '',
             '| Model | Explain mean per-example NLL | Track mean per-example NLL | Saved answers | Strict format | Length-limited |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for stage in STAGES:
        gen = summary['generation'][stage]; all_scope = gen['strict']['scopes']['all']
        format_value = f"{all_scope['strict_format_valid']}/228" if gen['saved_records'] else 'pending'
        length_value = f"{all_scope['finish_length']}/228" if gen['saved_records'] else 'pending'
        lines.append(f"| {stage} | {_nll(summary['validation_loss'][stage], 'explain')} | {_nll(summary['validation_loss'][stage], 'track')} | "
                     f"{gen['saved_records']}/228 | {format_value} | {length_value} |")
    lines += ['', (f"Selected checkpoint: `{selected['checkpoint']}`; Explain NLL={selected['selected_metric']:.6f}."
                  if selected['selected_stage'] else 'Checkpoint selection is pending both complete, finite epoch validation-loss runs and full training completion.'), '',
              'Each example receives the mean next-token cross entropy over its assistant target and im_end; those means are averaged equally within task. Explain targets include WHY, supervised goal, and next actions. Lowest Explain NLL selects between epoch1 and epoch2; exact ties select epoch1. The base is ineligible. Token-weighted NLL is saved only as a diagnostic.', '',
              '| Model | Independently extractable WHY | Independent WHY normalized exact match | Independent goal normalized exact match | Strict next-triple exact match |',
              '| --- | ---: | ---: | ---: | ---: |']
    for stage in STAGES:
        gen = summary['generation'][stage]; fields = gen['independent_fields']['scopes']['all']
        next_match = gen['strict']['scopes']['all']['lexical_exact_match']['next_sequence']
        if gen['saved_records']:
            lines.append(f"| {stage} | {fields['why']['extractable']}/228 | {_ratio(fields['why'])} | {_ratio(fields['goal'])} | {_ratio(next_match)} |")
        else:
            lines.append(f"| {stage} | pending | pending | pending | pending |")
    lines += ['', 'These normalized exact matches measure wording, not semantic correctness. The supplementary parser extracts WHY/goal independently of whole-response formatting. Strict NEXT requires exactly three semicolon-delimited actions. Missing, failed, malformed, empty, and length-limited outputs remain in all 228 applicable denominators. All validation Explain rows have goal supervision; current-action generation is unmeasured here.', '',
              '## Training, memory, and timing', '',
              'BF16 native model with FP32 text-LoRA masters: rank 16 / alpha 32 / dropout 0; 496 native text Linear targets, 992 adapter tensors, 116,727,808 trainable parameters. Vision, aligner, embeddings, and output head are frozen. SDPA, non-reentrant text checkpointing, no packing or quantization; context 8192. Two processes, per-device batch 1, accumulation 8, effective batch 16. Two epochs, 1,818 expected optimizer steps; LR 1e-4, cosine, 55 warmup steps, AdamW (0.9, 0.95), epsilon 1e-8, weight decay 0.1, clip 1, seed 20261003.', '',
              'The shuffled distributed sampler preserves all 14,543 initial-SFT rows and pads one shuffled draw per epoch. Each rank receives 7,272 microbatches per epoch, yielding 909 optimizer steps. Full state is saved every 250 steps and at 909/1818, including LoRA, optimizer, scheduler, progress, and both ranks’ RNG.', '',
              '| Physical GPU | UUID | Saved steps | Peak allocated GiB | Peak reserved GiB | Median step seconds |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for rank in (0, 1):
        item = training['ranks'][str(rank)]
        fmt = lambda key, divisor=1: f"{item[key]/divisor:.3f}" if item[key] is not None else 'pending'
        lines.append(f"| {rank} | {GPU_UUIDS[rank]} | {item['recorded_steps']} | {fmt('peak_allocated_bytes', 2**30)} | {fmt('peak_reserved_bytes', 2**30)} | {fmt('median_optimizer_step_seconds')} |")
    lines += ['', 'Per-step logs contain loss, learning rate, unclipped global gradient norm, and memory. Full completion requires finite losses/gradients, all 992 local LoRA tensors changing on both ranks, exhaustive frozen-parameter exclusion from optimizer/gradients, and unchanged sampled frozen parameter bytes. The sampled byte check is not an exhaustive full-base byte comparison.', '',
              f"Smoke training attempts: {len(summary['smoke']['attempts'])}; complete smoke/reload gate recorded: {bool(summary['smoke']['passed_gate'])}. "
              'The smoke is disposable; its initial-SFT generated answers test execution and tagged output, not accuracy. Detailed losses, gradients, reload proofs and four references/answers are under smoke/attempt-*/.', '',
              'Actual completed launch runtimes and commands are retained in execution_details.json. Step-time sums exclude checkpoint/validation/model-loading time. A two-step smoke cannot establish full-run runtime; do not interpret an extrapolation as measured completion.', '',
              '## Documented implementation changes', '']
    timing_section = []
    runtime = summary['training_runtime']
    if runtime['original_optimizer_run_wall_seconds'] is not None:
        seconds = runtime['original_optimizer_run_wall_seconds']
        timing_section += [f"Original optimizer-run launch: **{seconds:,.3f} seconds ({seconds/3600:.3f} hours)**, "
                           f"recorded process return code {runtime['original_optimizer_run_returncode']}. "
                           'This duration includes model loading, training, all validation-loss passes, checkpointing and the final audit.', '']
    if runtime['zero_update_finalization_wall_seconds'] is not None:
        timing_section += [f"Zero-update finalization launch: **{runtime['zero_update_finalization_wall_seconds']:.3f} seconds**, "
                           'reported separately from the original optimizer run. '
                           f"The new completion marker's {runtime['completion_marker_attempt_seconds']:.3f}-second duration describes the recovery process only; "
                           'it is not the full SFT runtime.', '']
    elif summary['finalization_recovery'] is not None:
        timing_section += [f"Zero-update finalization status: `{runtime['zero_update_finalization_status']}`. "
                           'The original optimizer-run wall time above remains authoritative; no short recovery duration is substituted for it.', '']
    for attempt in summary['smoke']['attempts']:
        if not attempt['completion']:
            continue
        timing_section += [f"Smoke `{attempt['attempt']}` measured evidence:", '',
                           '| Rank | Optimizer steps | Median seconds/step | Peak allocated GiB | Peak reserved GiB |',
                           '| --- | ---: | ---: | ---: | ---: |']
        for rank in (0, 1):
            item = attempt['ranks'][str(rank)]
            timing_section.append(f"| {rank} | {item['recorded_steps']} | {item['median_optimizer_step_seconds']:.3f} | "
                                  f"{item['peak_allocated_bytes']/2**30:.3f} | {item['peak_reserved_bytes']/2**30:.3f} |")
        last = attempt['ranks']['0']['last_optimizer_step']['seconds']
        timing_section += ['', f"Extrapolating only the final smoke step ({last:.3f} seconds) to 1,818 updates gives "
                           f"approximately {last*1818/3600:.1f} hours ({last*1818/86400:.1f} days), before validation, checkpoints and inference. "
                           'This is a two-step pilot estimate, not a measured completed runtime or a guarantee.', '',
                           f"Fresh FSDP reload proof: {bool(attempt['fresh_fsdp_reload'])}; fresh PEFT generation proof: {bool(attempt['generation_reload'])}. "
                           f"Four training-example references/answers: `{(attempt['four_training_examples'] or {}).get('path', 'pending')}`.", '']
        if attempt['loss_objective_diagnostics']:
            d = attempt['loss_objective_diagnostics'][0]
            timing_section += [f"Loss-objective diagnosis: `{d['classification']}`. The saved selected-head loss was "
                               f"{d['selected_training_loss']:.9f}, versus native full-head loss {d['native_full_loss']:.9f} on the same record. "
                               'The diagnostic found exact hidden states and causal label positions, identical repeated selected logits, '
                               'an exact short-GEMM reprojection from full-forward hidden states, and agreement between native loss and '
                               'masked cross entropy on those same full logits. BF16 head projections with different row counts round '
                               'differently; these two logits/losses are not claimed to agree at FP32 tolerance. The training computation '
                               'and original failed assertion remain unchanged, with the numerical evidence used for the reload gate. '
                               f"Evidence: `{d['file']['path']}`. No extra optimizer update was made by this diagnostic.", '']
    if timing_section:
        position = lines.index('## Documented implementation changes')
        lines[position:position] = timing_section
    for name, value in summary['configuration']['deviations_from_8b'].items():
        lines.append(f'- {name}: {value}')
    lines += ['', 'The 27B frozen BF16 model alone exceeds one 48GB GPU, so training uses native Torch 2.6 FSDP2 FULL_SHARD across authorized GPUs 0/1. Isolated Transformers 5.3.0 / PEFT 0.18.1 replaces the incompatible ms-swift 3.12.6 model interface. Native nonthinking serialization is used without modifying official model math. Generation reload uses the native model balanced across the same two GPUs, sequentially after training.', '',
              '## Failures, fixes, and limitations', '']
    recovery = summary['finalization_recovery']
    if recovery is not None:
        lines += ['The original process saved both epoch checkpoints, all 1,818 optimizer steps on both ranks, '
                  'and all three complete validation-loss files, then failed the final `Frozen parameter sample changed` assertion. '
                  'That failure is preserved and is not described as a successful original process exit.', '',
                  f"Recovery status: `{recovery.get('zero_update_resume', {}).get('status', 'not_started')}`. "
                  'The recovery procedure retains the original training implementation and identity, resumes checkpoint-1818 '
                  'with zero additional optimizer updates, and reuses validated loss caches. Original metadata is archived '
                  'before the normal finalization path writes its new completion proofs.', '']
        if recovery.get('diagnostic_proofs'):
            lines += ['The separately saved no-update diagnostic investigates FSDP parameter representation after a forward '
                      'pass and after explicit resharding. Its hash-bound evidence is referenced in '
                      '`recovery/finalization_recovery.json`; it must not be confused with serialized historical frozen-base tensors.', '']
        lines += ['The original failed GPU process did not serialize the frozen base tensors or its before/after sample hashes. '
                  'Consequently those exact historical bytes cannot be retrospectively compared. The restored-base sample check '
                  'verifies the fresh recovery process; it does not establish a historical full-base byte comparison. '
                  'The original per-step optimizer/gradient exclusion checks and retained checkpoint/log hashes provide separate evidence. '
                  'Recorded memory peaks use saved optimizer logs and later completion markers; the failed process’s final instantaneous '
                  'memory state was not serialized.', '']
        for limitation in recovery.get('limitations', []):
            lines.append(f'- Recovery limitation: {limitation}')
        lines.append('')
    if summary['failures_and_fixes']:
        for item in summary['failures_and_fixes']:
            lines.append(f"- {item.get('phase', 'Recorded event')}: {item.get('failure', item.get('error', 'See saved record')).rstrip('.')}. Fix: {item.get('fix', 'not recorded').rstrip('.')}.")
    else:
        lines.append('No failure/fix entries are recorded at this reporting snapshot.')
    lines += ['', 'No split/question/label changes, oversampling, second-stage training, GRPO, external inference, or test access are part of this run. Existing experiment and semantic-review artifacts remain separate. Semantic WHY/goal/action correctness and visual grounding of newly trained answers are unmeasured; no LLM judge or unverified action mapper is added.', '',
              '## Commands and provenance', '',
              '```bash', 'bash /data/yaxuanli/ActGround/scripts/run_hdepic_sft_qwen38_27b_v1.sh all',
              'bash /data/yaxuanli/ActGround/scripts/run_hdepic_sft_qwen38_27b_v1.sh train',
              'bash /data/yaxuanli/ActGround/scripts/run_hdepic_sft_qwen38_27b_v1.sh evaluate',
              'bash /data/yaxuanli/ActGround/scripts/run_hdepic_sft_qwen38_27b_v1.sh report', '```', '',
              'The same train command resumes matching complete state; full completion skips further updates. Prediction reuse requires exact model/adapter/configuration/prompt hashes. GPU stages are exclusive and do not terminate other jobs. See README.md for stage prerequisites and ephemeral-storage retention.', '',
              f"Current report provenance snapshot: `{summary['provenance']['directory']}`. Git HEAD: `{summary['provenance']['git_head']}`. "
              'It contains source files, relevant tracked diff, package versions, model metadata and existing base-weight integrity evidence. Executing checkpoint/code identities remain in their original markers; reporting does not rewrite them.', '']
    if summary['saved_commands']:
        lines.extend(['Exact recorded subprocess commands (with launch stage):', ''])
        for item in summary['saved_commands']:
            lines.extend([f"{item['stage']} — log `{item['log']}`:", '', '```bash',
                          'CUDA_VISIBLE_DEVICES=' + shlex.quote(item['cuda_visible_devices']) + ' ' + shlex.join(item['command']), '```', ''])
    return '\n'.join(lines)


def generate_report(run=RUN):
    """Refresh derived reports; safe during preparation, training, or completion."""
    run = Path(run).resolve(); config = read_json(run / 'frozen_config.json')
    require(config['model']['revision'] == MODEL_REVISION and config['model']['repo_id'] == MODEL_REPO, 'Wrong report model')
    validation_spec = config['data']['validation']; require(sha(validation_spec['text']) == validation_spec['text_sha256'], 'Frozen validation references changed')
    records = rows(validation_spec['text'])
    require(len(records) == 2209 and all(r['partition'] == 'validation' and r['participant'] == 'P07' for r in records),
            'Report is limited to fixed validation references')
    references = [r for r in records if r['task'] == 'explain']
    prompt_records = rows(BASELINE / 'inputs/validation_explain.prompts.jsonl')
    baseline_config = read_json(BASELINE / 'frozen_config.json')
    require(sha(BASELINE / 'inputs/validation_explain.prompts.jsonl') == baseline_config['prompts_sha256']
            and [r['id'] for r in references] == [r['id'] for r in prompt_records] and len(references) == 228,
            'Frozen baseline and SFT validation Explain cohorts differ')
    training = _training_summary(config, run)
    losses = {stage: loss_summary(stage, records, config, run) for stage in STAGES}
    generation = {}
    for stage in STAGES:
        value, strict_examples, independent_examples = generation_summary(stage, references, prompt_records, run)
        generation[stage] = value
        write_json(run / 'report' / f'{stage}_strict_per_example.json', strict_examples)
        write_json(run / 'report' / f'{stage}_independent_fields_per_example.json', independent_examples)
    selection = _selection(losses, training, config, run)
    completed = training['completed'] and all(x['completed'] for x in losses.values()) and all(x['completed'] for x in generation.values())
    provenance = snapshot_provenance(run)
    commands = [read_json(path) for path in sorted((run / 'logs').glob('*.command.json'))]
    results = [read_json(path) for path in sorted((run / 'logs').glob('*.result.json'))]
    recovery = optional(run / 'recovery/finalization_recovery.json')
    runtime = training_runtime(training, results, recovery)
    summary = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'completed_execution_semantic_review_pending' if completed else 'in_progress_or_blocked',
               'execution_complete': completed, 'configuration': config, 'configuration_file': file_info(run / 'frozen_config.json'),
               'preflight': optional(run / 'preflight/summary.json'), 'preflight_invocations': [read_json(p) for p in sorted((run / 'preflight').glob('invocation_*.json'))],
               'training': training, 'smoke': _smoke_summary(run), 'validation_loss': losses,
               'training_runtime': runtime, 'finalization_recovery': recovery,
               'finalization_recovery_file': file_info(run / 'recovery/finalization_recovery.json'),
               'generation': generation, 'selection': selection, 'provenance': provenance,
               'failures_and_fixes': read_rows(run / 'failures_and_fixes.jsonl'),
               'saved_commands': commands, 'saved_launch_results': results,
               'background_workflow': optional(run / 'background_workflow.json'),
               'last_saved_process_status': optional(run / 'active_process.json'),
               'successful_logged_launch_wall_seconds': sum(r['elapsed_seconds'] for r in results if r['returncode'] == 0),
               'failed_logged_launch_wall_seconds': sum(r['elapsed_seconds'] for r in results if r['returncode'] != 0),
               'checkpoint_storage_plan': optional(run / 'checkpoint_storage_plan.json'),
               'human_reason_label_review': 'pending', 'semantic_accuracy_measured': False,
               'no_test_or_second_stage_exports_read': True, 'gpu_accessed_by_reporting': False,
               'loss_metric_definition': 'Arithmetic mean by task of per-example mean assistant target plus im_end NLL; Explain includes why, goal and next, not why-only tokens',
               'lexical_metric_definition': 'Unchanged strict scorer plus independent WHY/goal extraction; NFKC/casefold/whitespace normalized exact match only, not semantic accuracy',
               'generation_scope': '228 fixed validation Explain IDs only; Track has teacher-forced loss but no new generated answers'}
    write_json(run / 'metrics.json', summary)
    write_json(run / 'selected_checkpoint.json', selection)
    write_json(run / 'execution_details.json', {'commands': commands, 'results': results,
               'training': training, 'smoke': summary['smoke'], 'preflight_invocations': summary['preflight_invocations'],
               'training_runtime': runtime, 'finalization_recovery': recovery,
               'generation': {s: {k: v for k, v in x.items() if k not in ('strict', 'independent_fields')} for s, x in generation.items()}})
    report = _render(summary)
    (run / 'execution_report.md').write_text(report)
    validation_section = report.split('## Validation and selection', 1)[1].split('## Training, memory, and timing', 1)[0]
    (run / 'validation_report.md').write_text('# Validation report\n\n' + validation_section +
        '\nSemantic scores are not computed here. Full denominators and per-example diagnostics are saved in metrics.json and report/.\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args(); result = generate_report(args.run)
    print(json.dumps({'status': result['status'], 'execution_complete': result['execution_complete'],
                      'selection': result['selection'], 'report': str(args.run / 'execution_report.md')}, indent=2))


if __name__ == '__main__':
    main()
