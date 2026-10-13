"""Aggregate completed disposable 8B smoke artifacts without loading a model.

This module reads existing audit files and never starts training or inference.
It writes results only after ten audited steps and the fresh reload/generation
artifacts exist. Generation-format failures are reported without extra steps.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT / 'runs/hdepic_sft_smoke_8b'
REQUIRED = (
    'gpu_allocation.json', 'gpu_before_train_load.json', 'gpu_before_reload_load.json', 'smoke_train_invocation.json',
    'smoke_dataset_manifest.json', 'smoke_initial8.jsonl', 'smoke_initial8.scoring.jsonl',
    'preprocessing_audit.json', 'trainable_parameters.json', 'training_audit_summary.json',
    'gradient_audit.json', 'loss_audit.json', 'optimizer_step_timing.json',
    'lora_parameter_delta_audit.json', 'checkpoint_save_audit.json',
    'saved_checkpoint_identity.json', 'checkpoint_reload_audit.json',
    'reload_initial_generations.jsonl', 'generation_audit.json', 'swift_train_result.json',
    'setup/checkpoint_identity.json', 'setup/environment_identity.json', 'setup/pip-freeze.txt',
    'cpu_preflight/preprocessing_audit.json',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def parse_argv(argv):
    values = {}
    key = None
    for item in argv:
        if item.startswith('--'):
            key = item[2:]
            values[key] = []
        elif key is not None:
            values[key].append(item)
    return {key: (value[0] if len(value) == 1 else value) for key, value in values.items()}


def aggregate(run):
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    if missing:
        raise RuntimeError('Smoke is incomplete; results were not written. Missing: ' + ', '.join(missing))
    training = read_json(run / 'training_audit_summary.json')
    preprocessing = read_json(run / 'preprocessing_audit.json')
    cpu = read_json(run / 'cpu_preflight/preprocessing_audit.json')
    losses = read_json(run / 'loss_audit.json')
    gradients = read_json(run / 'gradient_audit.json')
    timing = read_json(run / 'optimizer_step_timing.json')
    deltas = read_json(run / 'lora_parameter_delta_audit.json')
    trainable = read_json(run / 'trainable_parameters.json')
    reload = read_json(run / 'checkpoint_reload_audit.json')
    generation_audit = read_json(run / 'generation_audit.json')
    checkpoint = read_json(run / 'saved_checkpoint_identity.json')
    invocation = read_json(run / 'smoke_train_invocation.json')
    manifest = read_json(run / 'smoke_dataset_manifest.json')
    model = read_json(run / 'setup/checkpoint_identity.json')
    environment = read_json(run / 'setup/environment_identity.json')
    allocation = read_json(run / 'gpu_allocation.json')
    gpu_before = read_json(run / 'gpu_before_train_load.json')
    gpu_reload = read_json(run / 'gpu_before_reload_load.json')
    source_rows = read_jsonl(run / 'smoke_initial8.jsonl')
    scoring = read_jsonl(run / 'smoke_initial8.scoring.jsonl')
    generations = read_jsonl(run / 'reload_initial_generations.jsonl')
    fixes_path = run / 'smoke_failures_and_fixes.json'
    failures_and_fixes = read_json(fixes_path) if fixes_path.is_file() else []
    argv = parse_argv(invocation['argv'])

    assert training['passed'] and training['optimizer_steps'] == 10
    assert preprocessing['passed'] and preprocessing['records_verified'] == 8
    assert cpu['status'] == 'passed' and cpu['records'] == 20
    assert len(losses) == len(gradients) == len(timing) == 10
    steps = set(range(1, 11))
    assert {row['optimizer_step'] for row in losses} == steps
    assert {row['optimizer_step'] for row in gradients} == steps
    assert {row['optimizer_step'] for row in timing} == steps
    assert all(math.isfinite(row['loss']) for row in losses)
    assert all(not row['nonfinite_gradient_names'] and row['finite_gradient_tensors'] > 0
               and row['nonzero_gradient_tensors'] > 0 for row in gradients)
    assert any(row['changed_elements'] > 0 for row in deltas)
    assert training['text_lora_only'] and training['vision_and_aligner_frozen']
    assert trainable['lora_rank'] == 16 and trainable['lora_alpha'] == 32
    assert trainable['bf16_training'] and trainable['gradient_checkpointing']
    assert reload['passed'] and checkpoint['optimizer_steps'] == 10
    assert checkpoint['disposable'] and checkpoint['full_run_must_start_from'] == model['snapshot_path']
    assert invocation['original_checkpoint'] == model['snapshot_path']
    assert len(source_rows) == len(scoring) == len(set(row['id'] for row in scoring)) == 8
    assert all(row['partition'] == 'initial_sft' for row in scoring)
    assert len(generations) == generation_audit['records'] == 4
    assert all(row['partition'] == 'initial_sft' for row in generations)
    assert generation_audit['test_predictions'] == invocation['test_predictions'] == 0
    assert not invocation['full_training_launched'] and not invocation['grpo_launched']
    assert allocation['authorized_by_user'] and gpu_before['gpu_uuid'] == allocation['gpu_uuid']

    by_id = {metadata['id']: (row, metadata) for row, metadata in zip(source_rows, scoring)}
    comparisons = []
    for generated in generations:
        row, metadata = by_id[generated['id']]
        assert metadata['task'] == generated['task']
        comparisons.append({**generated, 'reference_target': row['messages'][-1]['content'],
                            'reference_is_supervision_not_model_output': True})
    packages = environment['packages']
    stack_names = ['ms_swift', 'transformers', 'torch', 'torchvision', 'peft', 'accelerate',
                   'trl', 'datasets', 'qwen-vl-utils', 'pillow', 'numpy', 'pandas']
    versions = {name: packages[name] for name in stack_names if name in packages}
    seconds = [row['seconds_including_gradient_audit'] for row in timing]
    gradient_missing = {name for row in gradients for name in row['missing_gradient_names']}
    frozen_config = {
        'precision': argv['torch_dtype'], 'bf16': argv['bf16'],
        'train_type': argv['train_type'], 'lora_rank': int(argv['lora_rank']),
        'lora_alpha': int(argv['lora_alpha']), 'freeze_vit': argv['freeze_vit'],
        'freeze_aligner': argv['freeze_aligner'], 'gradient_checkpointing': argv['gradient_checkpointing'],
        'per_device_batch_size': int(argv['per_device_train_batch_size']),
        'gradient_accumulation_steps': int(argv['gradient_accumulation_steps']),
        'world_size': 1,
        'effective_batch_examples': int(argv['per_device_train_batch_size']) * int(argv['gradient_accumulation_steps']),
        'max_optimizer_steps': int(argv['max_steps']), 'context_tokens': int(argv['max_length']),
        'images_per_example': 16, 'image_min_token_num': cpu['image_min_token_num'],
        'image_max_token_num': cpu['image_max_token_num'],
        'max_pixels': int(argv['max_pixels']), 'attention_implementation': argv['attn_impl'],
        'learning_rate': float(argv['learning_rate']), 'packing': argv['packing'],
        'loss_scale': argv['loss_scale'], 'truncate_policy': 'raise on overlength, strict dataset encoding',
    }
    evidence = {str(run / name): {'sha256': sha256(run / name), 'bytes': (run / name).stat().st_size}
                for name in REQUIRED}
    for name in ['smoke_failures_and_fixes.json',
                 'train.log', 'reload.log', 'setup/ready_environment.json']:
        if (run / name).is_file():
            evidence[str(run / name)] = {'sha256': sha256(run / name), 'bytes': (run / name).stat().st_size}
    code_paths = [PROJECT / 'code/hdepic_preparation' / name for name in
                  ['smoke_8b.py', 'swift_smoke_audit.py', 'swift_preflight_8b.py', 'smoke_report_8b.py']]
    code_paths.append(PROJECT / 'scripts/run_hdepic_sft_smoke_8b.sh')
    for path in code_paths:
        evidence[str(path)] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    executed_commands = None
    for filename in ['exact_commands.json', 'executed_commands.json', 'smoke_commands.json', 'smoke_command_manifest.json']:
        if (run / filename).is_file():
            executed_commands = read_json(run / filename)
            evidence[str(run / filename)] = {'sha256': sha256(run / filename),
                                             'bytes': (run / filename).stat().st_size}
            break
    if executed_commands is not None:
        assert executed_commands['total_optimizer_steps'] == 10
        assert not executed_commands['full_sft_launched'] and not executed_commands['grpo_launched']
        assert executed_commands['test_predictions'] == 0
    result = {
        'status': '10_optimizer_step_smoke_completed', 'pipeline_checks_passed': True,
        'model': {'repo_id': model['repo_id'], 'revision': model['revision'],
                  'original_snapshot': model['snapshot_path'], 'architecture': model['architecture'],
                  'files_verified_against_official_hashes': len(model['files'])},
        'environment': {'python': environment['python'], 'executable': environment['executable'],
                        'versions': versions, 'pip_freeze': str(run / 'setup/pip-freeze.txt')},
        'gpu': {'allocation': allocation, 'before_training_or_last_load': gpu_before,
                'before_reload': gpu_reload, 'gpu_count_used': 1},
        'training_configuration': frozen_config,
        'invocation': {'launcher': f'bash {PROJECT / "scripts/run_hdepic_sft_smoke_8b.sh"}',
                       'executed_commands': executed_commands,
                       'exact_swift_arguments': invocation['argv'],
                       'equivalent_swift_command_for_review': shlex.join(['swift', 'sft'] + invocation['argv']),
                       'repeated_run_guard': 'An existing checkpoint-10 prevents starting again in this run directory'},
        'dataset': manifest, 'cpu_preprocessing': {key: value for key, value in cpu.items() if key != 'rows'},
        'actual_training_preprocessing': {key: value for key, value in preprocessing.items() if key != 'rows'},
        'loss_and_gradients': {'finite_losses': True, 'finite_gradients_all_steps': True,
                              'nonzero_gradients_all_steps': True,
                              'first_loss': losses[0]['loss'], 'last_loss': losses[-1]['loss'],
                              'loss_range': [min(row['loss'] for row in losses), max(row['loss'] for row in losses)],
                              'missing_gradient_tensor_names': sorted(gradient_missing),
                              'loss_log': losses, 'gradient_log': gradients},
        'lora_updates': {'tensor_count': len(deltas), 'trainable_parameter_count': trainable['parameter_count'],
                         'changed_tensor_count': sum(row['changed_elements'] > 0 for row in deltas),
                         'changed_tensor_groups': dict(Counter(row['group'] for row in deltas if row['changed_elements'])),
                         'changed_elements': sum(row['changed_elements'] for row in deltas),
                         'vision_and_aligner_trainable_parameters': trainable['vision_and_aligner_trainable_parameters'],
                         'delta_audit': str(run / 'lora_parameter_delta_audit.json')},
        'performance': {'peak_cuda_allocated_gib': training['peak_cuda_allocated_bytes'] / 1024 ** 3,
                        'peak_cuda_reserved_gib': training['peak_cuda_reserved_bytes'] / 1024 ** 3,
                        'memory_measurement_scope': 'PyTorch CUDA allocator peaks during training; driver/context allocations are separate',
                        'start_cuda_allocated_gib': training['start_cuda_allocated_bytes'] / 1024 ** 3,
                        'mean_seconds_per_optimizer_step_including_gradient_audit': statistics.mean(seconds),
                        'median_seconds_per_optimizer_step_including_gradient_audit': statistics.median(seconds),
                        'optimizer_step_seconds': seconds,
                        'gradient_audit_overhead_seconds': training['gradient_audit_overhead_seconds'],
                        'training_seconds_including_audits_logging_and_checkpoint': training['training_seconds_including_audits_logging_and_checkpoint'],
                        'timing_limit': 'Includes instrumentation; ten steps are insufficient to establish full-run throughput'},
        'checkpoint': checkpoint, 'reload': reload,
        'generation': {'records': 4, 'tag_format_passed': sum(row['tag_format_valid'] for row in comparisons),
                       'three_next_actions_by_semicolon_passed': sum(row['semicolon_next_action_count'] == 3 for row in comparisons),
                       'comparisons': comparisons, 'only_initial_sft_examples': True,
                       'answer_prefix_forcing': False, 'semantic_accuracy_claim': None},
        'failures_and_fixes': failures_and_fixes,
        'boundaries': {'full_sft_started': False, 'grpo_started': False, 'test_predictions': 0,
                       'reason_labels_pending_human_review': True,
                       'smoke_adapter_disposable': True,
                       'eventual_full_sft_initialization': model['snapshot_path'],
                       'resume_smoke_adapter_for_full_sft': False,
                       'accuracy_conclusion': 'None; this smoke test establishes pipeline behavior only'},
        'evidence_artifacts': evidence,
    }
    return result


def inline(value):
    return '`' + str(value).replace('`', '\\`').replace('\n', ' ').replace('|', '\\|') + '`'


def markdown(result, run):
    config = result['training_configuration']
    performance = result['performance']
    model = result['model']
    before = result['gpu']['before_training_or_last_load']
    generation = result['generation']
    updates = result['lora_updates']
    rows = [
        'The disposable Qwen3-VL-8B-Instruct smoke completed exactly **10 optimizer steps** and a fresh checkpoint reload. '
        'Finite losses and gradients, actual text LoRA updates, complete assistant targets, and all 16 individual images passed the audits. '
        f'Tag formatting passed for **{generation["tag_format_passed"]}/4** diagnostic training-example generations. '
        'No model-accuracy conclusion follows from this smoke test.',
        '',
        f'Original model: {inline(model["repo_id"])} at revision {inline(model["revision"])}. '
        f'The original snapshot is [{model["original_snapshot"]}]({model["original_snapshot"]}); '
        f'{model["files_verified_against_official_hashes"]} downloaded files matched official hashes. '
        f'The isolated environment is {inline(result["environment"]["executable"])}. '
        f'Exact dependencies are in [pip-freeze.txt]({run / "setup/pip-freeze.txt"}).',
        '',
        '| Package | Version |', '| --- | --- |',
    ]
    rows += [f'| {name} | {inline(version)} |' for name, version in result['environment']['versions'].items()]
    rows += [
        '',
        f'Only allocated GPU **{before["gpu_index"]}**, {before["name"]}, was used: {inline(before["gpu_uuid"])}. '
        f'Total VRAM {before["total_mib"] / 1024:.2f} GiB; before model loading {before["used_mib"]} MiB was used and '
        f'{before["free_mib"]} MiB was free. Allocation covered indices '
        f'{result["gpu"]["allocation"].get("authorized_gpu_indices", [before["gpu_index"]])}; this run used one GPU.',
        '',
        f'Training used BF16, text-model LoRA rank {config["lora_rank"]}/alpha {config["lora_alpha"]}, '
        'frozen vision encoder and aligner, gradient checkpointing, SDPA, batch size 1, and one accumulation step '
        f'(effective batch {config["effective_batch_examples"]}). Context limit {config["context_tokens"]}; '
        f'16 images, image token range {config["image_min_token_num"]}–{config["image_max_token_num"]}, '
        f'maximum {config["max_pixels"]} pixels per image. Learning rate {config["learning_rate"]:g}. '
        'Eight unique initial-SFT examples cover both tasks and goal-present/absent targets; the short run revisits some examples. '
        'No second-stage or held-out examples entered training.',
        '',
        f'The CPU pipeline audit verified 20 initial-SFT records and 320 images. '
        f'The actual trainer audit verified {result["actual_training_preprocessing"]["records_verified"]} records, '
        'all individual images, exact prompts and complete targets, and masked system/user/image/assistant-header tokens. '
        'Only assistant target tokens and their exact stop suffix contribute to loss.',
        '',
        '| Measurement | Result |', '| --- | --- |',
        f'| Peak PyTorch allocated GPU memory | {performance["peak_cuda_allocated_gib"]:.3f} GiB |',
        f'| Peak PyTorch reserved GPU memory | {performance["peak_cuda_reserved_gib"]:.3f} GiB |',
        f'| Mean seconds per optimizer step, including gradient audit | {performance["mean_seconds_per_optimizer_step_including_gradient_audit"]:.3f} |',
        f'| Median seconds per optimizer step, including gradient audit | {performance["median_seconds_per_optimizer_step_including_gradient_audit"]:.3f} |',
        f'| Gradient-audit overhead, total | {performance["gradient_audit_overhead_seconds"]:.3f} seconds |',
        f'| Training wall time including audits/logging/checkpoint | {performance["training_seconds_including_audits_logging_and_checkpoint"]:.3f} seconds |',
        f'| LoRA tensors changed | {updates["changed_tensor_count"]}/{updates["tensor_count"]} |',
        f'| LoRA elements changed | {updates["changed_elements"]:,} |',
        f'| Vision/aligner trainable parameters | {updates["vision_and_aligner_trainable_parameters"]} |',
        '',
        'Timing includes instrumentation. Ten steps do not establish sustained full-run throughput.',
        '',
        '| Optimizer step | Loss | Reported gradient norm before clipping | Seconds including gradient audit |',
        '| --- | --- | --- | --- |',
    ]
    for loss, seconds in zip(result['loss_and_gradients']['loss_log'], performance['optimizer_step_seconds']):
        norm = loss['reported_gradient_norm_before_clipping']
        rows.append(f'| {loss["optimizer_step"]} | {loss["loss"]:.6f} | {norm if norm is not None else "unavailable"} | {seconds:.3f} |')
    rows += [
        '',
        f'The disposable adapter was saved at [{result["checkpoint"]["checkpoint"]}]({result["checkpoint"]["checkpoint"]}). '
        f'A fresh process reloaded {result["reload"]["exact_saved_adapter_tensors_reloaded"]} saved adapter tensors exactly. '
        'The four responses below use initial-SFT inputs only; reference targets are supervision, and responses are unconstrained diagnostic generations.',
        '',
        '| Task / goal | ID | Reference target | Generation after reload | Tags valid / three next actions |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in generation['comparisons']:
        rows.append(f'| {row["task"]} / {row["goal_status"]} | {inline(row["id"])} | '
                    f'{inline(row["reference_target"])} | {inline(row["generation"])} | '
                    f'{row["tag_format_valid"]} / {row["semicolon_next_action_count"] == 3} |')
    rows += ['', 'Recorded failures and fixes:']
    if result['failures_and_fixes']:
        rows += ['', '| Phase | Failure | Fix | Effect on steps | Log |', '| --- | --- | --- | --- | --- |']
        for entry in result['failures_and_fixes']:
            rows.append('| ' + ' | '.join(inline(entry.get(key, '')) for key in
                                          ['phase', 'failure', 'fix', 'effect_on_steps', 'log_path']) + ' |')
    else:
        rows += ['', 'No retry/fix entries were recorded.']
    commands = result['invocation']['executed_commands']
    if commands is not None:
        rows += ['', f'Exact commands are recorded in [exact_commands.json]({run / "exact_commands.json"}). '
                 'They ran in this order; the first attempt stopped before any optimizer step.']
        for entry in commands['commands']:
            rows += ['', f'Attempt {entry["attempt"]}: {entry["optimizer_steps"]} optimizer steps.',
                     '', '```bash', entry['command'], '```']
    else:
        rows += ['', 'Launcher command:', '', '```bash', result['invocation']['launcher'], '```']
    rows += [
        '',
        'Exact underlying Swift arguments are in '
        f'[smoke_train_invocation.json]({run / "smoke_train_invocation.json"}). '
        'The launcher refuses to resume an existing smoke checkpoint. Running the report helper reads existing results only:',
        '', '```bash',
        f'PYTHONPATH={PROJECT} PYTHONDONTWRITEBYTECODE=1 {result["environment"]["executable"]} '
        f'-m code.hdepic_preparation.smoke_report_8b --run {run}', '```', '',
        '**Full SFT and GRPO were not launched; test-set predictions were not generated.** '
        'Reason labels remain pending human review. The smoke adapter is disposable. '
        'Any eventual full SFT must start from the original pinned 8B checkpoint, with a fresh adapter and optimizer. '
        'Full-run configuration selection remains separate from this completed smoke.', '',
        f'All source audit paths and SHA256 hashes are in [smoke_results.json]({run / "smoke_results.json"}).', '',
    ]
    return '\n'.join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.check_ready:
        missing = [name for name in REQUIRED if not (run / name).is_file()]
        print(json.dumps({'ready': not missing, 'missing': missing}, indent=2))
        return
    result = aggregate(run)
    write_atomic(run / 'smoke_results.json', json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    write_atomic(run / 'smoke_results.md', markdown(result, run))
    print(json.dumps({'status': result['status'], 'results': str(run / 'smoke_results.md'),
                      'optimizer_steps': result['training_configuration']['max_optimizer_steps'],
                      'tag_format_passed': result['generation']['tag_format_passed']}, indent=2))


if __name__ == '__main__':
    main()
