"""Read-only evidence aggregation for the separate native Qwen3.5 baseline.

Only this new run's report files are written. No model, Torch, inference,
training, external API, identity key, or test-set records are accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import tempfile
import os

from .data import OUT, immutable_write, json_bytes, require, rows, sha256

EXPECTED = 228
VERSION = 'hdepic_qwen35_execution_report_v1'


def read_optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def load_active_configuration(run):
    """Follow the saved implementation revision without overwriting v1."""
    run = Path(run).resolve()
    pointer_path = run / 'active_configuration.json'
    pointer = read_optional(pointer_path)
    selected = (run / pointer['path']).resolve() if pointer else run / 'frozen_config.json'
    require(selected.is_relative_to(run) and selected.is_file(), 'Active configuration path is missing or outside this run')
    config = json.loads(selected.read_text())
    evidence = {'path': str(selected), 'file_sha256': sha256(selected), 'pointer': pointer,
                'pointer_file_sha256': sha256(pointer_path) if pointer else None,
                'previous_configuration_path': None, 'previous_configuration_sha256': None,
                'experimental_settings_unchanged': None}
    if pointer:
        require(pointer['file_sha256'] == evidence['file_sha256']
                and pointer['configuration_sha256'] == config['configuration_sha256'], 'Active configuration pointer does not match its file')
        if pointer.get('previous_configuration'):
            previous = (run / pointer['previous_configuration']).resolve()
            require(previous.is_relative_to(run) and previous.is_file(), 'Previous configuration evidence is missing or outside this run')
            original = json.loads(previous.read_text())
            settings = lambda value: {key: item for key, item in value.items()
                                      if key not in ('configuration_sha256', 'answering_code_sha256')}
            evidence.update({'previous_configuration_path': str(previous),
                             'previous_configuration_sha256': sha256(previous),
                             'experimental_settings_unchanged': settings(original) == settings(config)})
            require(evidence['experimental_settings_unchanged'], 'Implementation repair changed frozen experimental settings')
    return config, evidence


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timing(values):
    require(all(finite_number(value) and value >= 0 for value in values), 'Invalid recorded latency')
    ordered = sorted(values)
    return {'observations': len(values), 'total_sec': sum(values) if values else None,
            'mean_sec': statistics.mean(values) if values else None,
            'median_sec': statistics.median(values) if values else None,
            'p95_sec': ordered[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None,
            'p95_definition': 'nearest-rank 95th percentile',
            'scope': 'Recorded model.generate latency with device synchronization; excludes CPU preprocessing and model/weight download.'}


def aggregate_predictions(predictions, ordered_ids, config=None, prompts=None):
    """Count every fixed ID, including absent, failed and malformed responses."""
    require(len(ordered_ids) == EXPECTED and len(set(ordered_ids)) == EXPECTED, 'Wrong fixed Explain denominator')
    allowed = set(ordered_ids); byid = {}; prompt_map = {p['id']: p for p in (prompts or [])}
    for row in predictions:
        rid = row['id']
        require(rid in allowed and rid not in byid, f'Unknown/duplicate generated ID: {rid}')
        if config is not None:
            require(row.get('configuration_sha256') == config['configuration_sha256']
                    and row.get('model_revision') == config['revision']
                    and row.get('model_repo', config['repo_id']) == config['repo_id'], f'Wrong generation identity: {rid}')
        if prompts is not None:
            require(row.get('prompt_sha256') == prompt_map[rid]['prompt_sha256']
                    and row.get('row_index') == prompt_map[rid]['row_index'], f'Wrong input identity: {rid}')
        byid[rid] = row
    statuses = Counter(row.get('status', 'unspecified') for row in predictions)
    complete = [byid[rid] for rid in ordered_ids if rid in byid and byid[rid].get('status') == 'completed']
    missing_ids = [rid for rid in ordered_ids if rid not in byid]
    failed_ids = [rid for rid in ordered_ids if rid in byid and byid[rid].get('status') != 'completed']
    counts = Counter(); flags = Counter(); finishes = Counter(); latencies = []; invalid_latency_ids = []
    pipeline = Counter()
    for row in complete:
        require(isinstance(row.get('prediction'), str), f'Completed response lacks raw answer: {row["id"]}')
        counts['nonempty_raw_answer'] += bool(row['prediction'].strip())
        counts['strict_whole_response_format_valid'] += row.get('parsed', {}).get('valid') is True
        field = row.get('independent_why', {})
        counts['independent_why_extractable'] += field.get('extractable') is True
        for flag in ('missing', 'empty', 'duplicate', 'ambiguous'):
            flags[flag] += field.get(flag) is True
        finishes[row.get('finish_reason', 'unspecified')] += 1
        actual = row.get('actual_model_checks', {}); preprocessing = row.get('preprocessing', {})
        pipeline['exactly_one_image_prefill'] += actual.get('prefill_calls') == 1
        pipeline['finite_raw_logits_observed'] += actual.get('raw_logits_checks', 0) > 0
        pipeline['sixteen_complete_image_grids'] += len(preprocessing.get('image_grid_thw', [])) == 16
        pipeline['all_144_visual_tokens_per_image'] += preprocessing.get('image_tokens_per_image') == [144] * 16
        pipeline['pixel_patch_shape_complete'] += preprocessing.get('pixel_values_shape') == [9216, 1536]
        pipeline['complete_original_prompt'] += (preprocessing.get('complete_prompt') is True
            and preprocessing.get('original_text_exact') is True and preprocessing.get('truncation') is False)
        pipeline['references_and_targets_absent'] += preprocessing.get('no_reference_fields_or_assistant_target') is True
        if finite_number(row.get('latency_sec')) and row['latency_sec'] >= 0:
            latencies.append(row['latency_sec'])
        else:
            invalid_latency_ids.append(row['id'])
    result = {
        'denominator': EXPECTED, 'partition': 'validation', 'task': 'explain',
        'recorded_rows': len(predictions), 'completed': len(complete),
        'missing_prediction': len(missing_ids), 'failed_or_other_status': len(failed_ids),
        'statuses': dict(statuses), 'missing_prediction_ids': missing_ids, 'failed_or_other_ids': failed_ids,
        'nonempty_raw_answers': counts['nonempty_raw_answer'],
        'strict_whole_response_format_valid': counts['strict_whole_response_format_valid'],
        'strict_whole_response_format_invalid_or_unavailable': EXPECTED - counts['strict_whole_response_format_valid'],
        'independent_why_extractable': counts['independent_why_extractable'],
        'independent_why_unextractable_or_unavailable': EXPECTED - counts['independent_why_extractable'],
        'independent_why_flags_among_completed': {flag: flags[flag] for flag in ('missing', 'empty', 'duplicate', 'ambiguous')},
        'why_missing_or_unavailable_over_all_228': flags['missing'] + len(missing_ids) + len(failed_ids),
        'why_flags_not_mutually_exclusive': True,
        'finish_reason_counts_among_completed': dict(finishes), 'length_limited': finishes['length'],
        'generation_latency': timing(latencies), 'latency_missing_or_invalid_ids': invalid_latency_ids,
        'actual_pipeline_checks_among_completed': dict(pipeline),
        'semantic_accuracy_measured': False,
        'interpretation': 'Formatting and WHY extractability diagnostics; no semantic correctness or visual-support score.',
    }
    for name in ('completed', 'strict_whole_response_format_valid', 'independent_why_extractable', 'length_limited'):
        result[name + '_rate_over_all_228'] = result[name] / EXPECTED
    return result


def aggregate_memory(loading, smoke_rows, generation_rows, loading_attempts=()):
    """Allocator peaks include the separately captured native loading phase."""
    observations = []
    if loading:
        observations.append(('native_model_loading', loading.get('gpu_memory', {})))
    observations.extend(('loading_attempt:' + str(index), item.get('gpu_memory', {}))
                        for index, item in enumerate(loading_attempts, 1))
    observations.extend(('smoke:' + row['id'], row.get('gpu_memory', {})) for row in smoke_rows)
    observations.extend(('generation:' + row['id'], row.get('gpu_memory', {})) for row in generation_rows)
    result = {}
    for logical in ('0', '1'):
        observed = [(phase, record[logical]) for phase, record in observations if logical in record]
        peaks = {}
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [(phase, item[key]) for phase, item in observed if finite_number(item.get(key))]
            peaks[key] = max(value for _, value in values) if values else None
            peaks[key.replace('_bytes', '_gib')] = peaks[key] / 2 ** 30 if values else None
            peaks[key + '_observed_in'] = [phase for phase, value in values if value == peaks[key]]
        free = [item['free_bytes'] for _, item in observed if finite_number(item.get('free_bytes'))]
        result[logical] = {'physical_gpu_index': int(logical), 'observations': len(observed), **peaks,
                          'minimum_observed_free_bytes': min(free) if free else None}
    return {'devices': result,
            'scope': 'Maximum of saved PyTorch allocator peaks from loading, smoke, and per-record generation. NVML/process memory is a separate resource probe; unrecorded failures have no invented peak.',
            'loading_memory': loading.get('gpu_memory') if loading else None}


def audit_preservation(run, manifest):
    """One fresh final SHA audit; checkpoint weights are compared by saved stats."""
    run = Path(run); destination = run / 'audit/final_preservation.json'
    protected_path = run / 'audit/protected_originals.json'
    protected_sha = sha256(protected_path)
    if destination.is_file():
        old = json.loads(destination.read_text())
        require(old['protected_manifest_sha256'] == protected_sha, 'Original protected manifest changed after final audit')
        return old
    protected = json.loads(protected_path.read_text()); changed = []; missing = []; actual = {}
    for name, expected in protected.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name); continue
        actual[name] = sha256(path)
        if actual[name] != expected:
            changed.append(name)
    snapshot = Path(manifest['source_run']) / 'why_semantic_review_v1/checkpoint_stat_snapshot.json'
    require(snapshot.is_file(), 'Saved original checkpoint stat snapshot is missing')
    snapshot_records = json.loads(snapshot.read_text())
    require(snapshot_records and len({row['path'] for row in snapshot_records}) == len(snapshot_records),
            'Saved original checkpoint stat snapshot is empty or has duplicate paths')
    stat_changed = []; stat_missing = []
    for row in snapshot_records:
        path = Path(row['path'])
        if not path.is_file():
            stat_missing.append(str(path)); continue
        value = path.stat()
        if value.st_size != row['size'] or value.st_mtime_ns != row['mtime_ns']:
            stat_changed.append({'path': str(path), 'expected_size': row['size'], 'actual_size': value.st_size,
                                 'expected_mtime_ns': row['mtime_ns'], 'actual_mtime_ns': value.st_mtime_ns})
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'passed': not changed and not missing and not stat_changed and not stat_missing,
              'protected_manifest_sha256': protected_sha, 'protected_original_files': len(protected),
              'sha256_files_checked': len(actual), 'changed_originals': changed, 'missing_originals': missing,
              'after_sha256': actual, 'checkpoint_stat_snapshot': str(snapshot),
              'checkpoint_stat_snapshot_sha256': sha256(snapshot) if snapshot.is_file() else None,
              'checkpoint_stat_files_checked': len(snapshot_records),
              'checkpoint_stat_changed': stat_changed, 'checkpoint_stat_missing': stat_missing,
              'checkpoint_verification_scope': 'Path, byte size and mtime_ns against the completed prior semantic review snapshot. This is not a weight-content hash and the snapshot predates the new baseline preparation.',
              'identity_key_read': False, 'test_accessed': False, 'weight_contents_read_for_checkpoint_stats': False}
    immutable_write(destination, json_bytes(result))
    return result


def environment_issues(run):
    """List fixes only where retained failure logs establish the original issue."""
    issues = []
    definitions = [
        ('environment_setup.log', 'Name or service not known', 'Sandbox package-download DNS failure',
         'The retained host setup log and compatibility report establish successful isolated installation.'),
        ('environment_setup.host.log', 'libcusparseLt.so.0', 'Copied Torch runtime initially lacked the CUDA cusparseLt shared library',
         'The isolated CUDA runtime dependencies were repaired; the final compatibility report records a successful native import.'),
        ('environment_setup.repaired.log', "No module named 'torchgen'", 'Copied Torch runtime initially omitted torchgen',
         'The isolated runtime copy was completed; environment_setup.torchgen.log and the compatibility report establish successful native import.'),
    ]
    for name, needle, issue, correction in definitions:
        path = Path(run) / name
        if path.is_file() and needle in path.read_text(errors='replace'):
            issues.append({'issue': issue, 'resolution_evidence': correction, 'failure_log': str(path),
                           'failure_log_sha256': sha256(path)})
    return issues


def weight_cache_evidence(identity):
    from .model import CACHE
    snapshot = CACHE / 'models--Qwen--Qwen3.5-35B-A3B/snapshots' / identity['revision']
    files = []
    for name in identity['weight_shards']:
        path = snapshot / name
        if path.is_file():
            files.append({'filename': name, 'bytes': path.stat().st_size, 'resolved_blob_name': path.resolve().name})
    return {'checkpoint_snapshot': str(snapshot), 'expected_weight_shards': len(identity['weight_shards']),
            'complete_weight_files_present': len(files), 'bytes_present': sum(row['bytes'] for row in files),
            'all_expected_weight_files_present': len(files) == len(identity['weight_shards']),
            'observed_files': files, 'reporter_rehashed_weight_contents': False,
            'initial_metadata_weights_downloaded_flag': identity.get('weights_downloaded'),
            'initial_flag_interpretation': 'model_identity.json is immutable preparation metadata. Its false download flag does not describe later download/loading completion.'}


def weight_audit_issues(run):
    issues = []
    for path in sorted((Path(run) / 'audit/weight_integrity_failures').glob('*.json')):
        result = json.loads(path.read_text())
        all_sha_passed = bool(result.get('weight_shards')) and all(row.get('passed') is True for row in result['weight_shards'])
        old_expected = result.get('expected_weight_bytes'); old_actual = result.get('actual_weight_bytes')
        accounting_failure = all_sha_passed and isinstance(old_expected, int) and isinstance(old_actual, int) and old_actual > old_expected
        issues.append({'artifact': str(path), 'artifact_sha256': sha256(path),
                       'all_content_sha256_passed': all_sha_passed,
                       'initial_expected_tensor_bytes': old_expected, 'initial_actual_file_bytes': old_actual,
                       'issue': 'The initial CPU integrity audit compared tensor-only index size with complete safetensors file size, including header overhead.' if accounting_failure else 'See retained failed integrity audit.',
                       'repair': 'Parse each 8-byte little-endian length prefix and JSON header, validate tensor offsets and exact file lengths, and compare tensor payload alone with the index total. Reuse prior hashes only when blob paths, byte sizes, mtimes, hashes, configuration, and completed generation identity all match.' if accounting_failure else None,
                       'generation_repeated': False, 'frozen_answering_code_changed': False})
    return issues


def build_metrics(run=OUT, *, preservation=True):
    run = Path(run).resolve()
    require(run != Path('/data/yaxuanli/ActGround/runs/hdepic_sft_8b_v1'), 'Do not write original experiment reports')
    manifest = read_optional(run / 'inputs/manifest.json'); config, configuration_evidence = load_active_configuration(run)
    require(manifest and config and manifest['count'] == EXPECTED and config['count'] == EXPECTED, 'Frozen 228-row run is required')
    require(manifest['partition'] == config['partition'] == 'validation' and config['task'] == 'explain', 'Wrong report cohort')
    frozen_parser = read_optional(run / 'frozen_parser.json')
    require(frozen_parser, 'The parser audit must be frozen before report generation')
    parser_current_hashes = {}
    for name in ('strict', 'supplementary'):
        actual_hash = sha256(frozen_parser[name + '_parser_source'])
        require(actual_hash == frozen_parser[name + '_parser_sha256'], f'Frozen {name} parser source changed')
        parser_current_hashes[name] = actual_hash
    prompt_path = run / 'inputs/validation_explain.prompts.jsonl'; prompts = rows(prompt_path)
    require(sha256(prompt_path) == config['prompts_sha256'] and [row['id'] for row in prompts] == manifest['ordered_ids'], 'Frozen answering inputs changed')
    path = run / 'predictions/generation.jsonl'; generated = rows(path) if path.is_file() else []
    diagnostics = aggregate_predictions(generated, manifest['ordered_ids'], config, prompts)
    marker = read_optional(run / 'predictions/completed.json')
    marker_valid = bool(marker and marker.get('passed') is True and marker.get('count') == EXPECTED
                        and diagnostics['completed'] == EXPECTED
                        and marker.get('configuration_sha256') == config['configuration_sha256']
                        and marker.get('generation_sha256') == sha256(path)
                        and marker.get('inputs_manifest_sha256') == sha256(run / 'inputs/manifest.json'))
    if marker is not None:
        require(marker_valid, 'Generation completion marker conflicts with recorded outputs')
    loading = read_optional(run / 'smoke/model_loading.json')
    loading_attempts = [json.loads(path.read_text()) for path in sorted((run / 'smoke/loading_attempts').glob('*.json'))]
    if loading_attempts:
        loading = loading_attempts[-1]
    smoke_marker = read_optional(run / 'smoke/passed.json')
    smoke_path = run / 'smoke/generation.jsonl'; smoke = rows(smoke_path) if smoke_path.is_file() else []
    cpu = read_optional(run / 'processor_preflight/audit.json'); env = read_optional(run / 'environment/compatibility.json')
    require(cpu and sha256(run / 'processor_preflight/audit.json') == config['processor_preflight_sha256'],
            'Frozen CPU preprocessing audit changed')
    smoke_valid = bool(smoke_marker and smoke_marker.get('passed') is True
                       and smoke_marker.get('count') == len(smoke) == 4
                       and len({row['id'] for row in smoke}) == 4
                       and smoke_marker.get('configuration_sha256') == config['configuration_sha256']
                       and smoke_marker.get('generation_sha256') == sha256(smoke_path))
    if smoke_marker is not None:
        require(smoke_valid, 'Smoke completion marker conflicts with recorded outputs')
    if marker_valid:
        require(smoke_valid and all(value == EXPECTED for value in diagnostics['actual_pipeline_checks_among_completed'].values()),
                'Actual model input/logit checks are incomplete despite a completion marker')
    identity = read_optional(run / 'model_identity.json'); require(identity, 'Model identity required')
    failures = [{**json.loads(p.read_text()), 'artifact': str(p), 'sha256': sha256(p)}
                for p in sorted((run / 'failures').glob('*.json'))]
    resources = [{'artifact': str(p), **json.loads(p.read_text())} for p in sorted((run / 'resources').glob('*.json'))]
    review_materials = read_optional(run / 'review/materials_manifest.json')
    review_complete = read_optional(run / 'review/completed.json')
    preserved = audit_preservation(run, manifest) if preservation else None
    loading_info = loading.get('loading_info', {}) if loading else {}
    native_loading_passed = bool(loading and not loading_info.get('missing_keys')
                                 and not loading_info.get('mismatched_keys') and not loading_info.get('error_msgs')
                                 and all(k.startswith('mtp.') for k in loading_info.get('unexpected_keys', []))
                                 and set(loading.get('parameter_devices', [])) == {'cuda:0', 'cuda:1'})
    status = ('completed_local_generation_review_pending' if marker_valid else
              'failed_or_blocked' if failures else 'pending_or_partial_local_generation')
    result = {'version': VERSION, 'status': status, 'run': str(run),
              'report_created_at_utc': datetime.now(timezone.utc).isoformat(),
              'generation_completed': marker_valid, 'configuration_sha256': config['configuration_sha256'],
              'frozen_config_file_sha256': configuration_evidence['file_sha256'],
              'active_configuration': configuration_evidence,
              'frozen_parser': {'artifact': str(run / 'frozen_parser.json'),
                                'artifact_sha256': sha256(run / 'frozen_parser.json'),
                                'current_source_sha256': parser_current_hashes, 'sources_match_frozen': True,
                                'protocol': frozen_parser},
              'checkpoint': {'repo_id': identity['repo_id'], 'revision': identity['revision'],
                             'adaptation': config['adaptation'], 'native_model_loading_verified': native_loading_passed,
                             'cache_evidence': weight_cache_evidence(identity),
                             'weight_content_integrity': read_optional(run / 'audit/model_weight_integrity.json')},
              'environment': env, 'precision_and_execution': {k: config.get(k) for k in
                  ('dtype', 'quantization', 'cpu_or_disk_offload', 'attention_implementation', 'experts_implementation',
                   'native_torch_linear_attention_fallback', 'device_map', 'max_memory_gib_per_gpu', 'physical_gpus',
                   'parallelism', 'max_context', 'individual_images', 'min_pixels', 'max_pixels', 'visual_tokens_per_frame',
                   'enable_thinking', 'generation', 'generation_forward_kwargs', 'seed')},
              'resources': resources, 'cpu_preflight': cpu,
              'native_loading': loading, 'native_loading_latency_sec': loading.get('load_latency_sec') if loading else None,
              'native_loading_attempts': loading_attempts,
              'smoke': {'marker': smoke_marker, 'recorded_answers': len(smoke), 'criteria_passed': smoke_valid},
              'generation': diagnostics, 'gpu_memory': aggregate_memory(loading, smoke, generated, loading_attempts),
              'output_format_audit': read_optional(run / 'audit/output_format_diagnostics.json'),
              'failures': failures, 'environment_failures_and_fixes': environment_issues(run),
              'postgeneration_weight_audit_failures_and_fixes': weight_audit_issues(run),
              'semantic_review': {'materials_prepared': review_materials is not None,
                                  'materials_status': review_materials.get('status') if review_materials else None,
                                  'completion_marker_present': review_complete is not None,
                                  'semantic_accuracy_measured_by_this_report': False,
                                  'status': 'Review completion marker exists; semantic results must be read in the separately audited review report.' if review_complete else 'Pending substantive assistant semantic review and independent human review.',
                                  'human_review_complete': False},
              'preservation': preserved,
              'original_selected_checkpoint': config.get('original_epoch1_selection_preserved'),
              'no_new_training': config.get('training') is False, 'no_test_access': config.get('test_access') is False,
              'references_or_narration_supplied_to_answering_model': config.get('references_or_narration_supplied'),
              'reporter_model_inference': False, 'external_inference_api': False,
              'exact_execution_commands': read_optional(run / 'exact_commands.json'),
              'rerun_launcher': '/data/yaxuanli/ActGround/scripts/run_hdepic_qwen35_baseline.sh',
              'artifacts': {name: str(run / name) for name in ('frozen_config.json', 'environment/packages.txt',
                  'processor_preflight/audit.json', 'predictions/generation.jsonl', 'predictions/completed.json',
                  'smoke/model_loading.json', 'smoke/passed.json', 'review/blinded/index.html', 'review/blinded/human_review.csv',
                  'frozen_parser.json', 'exact_commands.json', 'active_configuration.json', 'frozen_config_v2.json',
                  'audit/active_smoke_input_verification.json', 'audit/model_weight_integrity.json')
                  if (run / name).is_file()}}
    return result


def fmt(value, suffix=''):
    return 'unavailable' if value is None else f'{value:.3f}{suffix}' if isinstance(value, float) else f'{value}{suffix}'


def markdown(metrics):
    gen = metrics['generation']; precision = metrics['precision_and_execution']; model = metrics['checkpoint']
    lines = [f"Status: **{metrics['status']}**.", '',
             f"The fixed denominator is all {EXPECTED} P07 validation Explain records. {gen['completed']}/{EXPECTED} have completed raw local answers; {gen['missing_prediction']} are absent and {gen['failed_or_other_status']} have failed or another status. No new model training, test predictions, or external inference API was used.", '',
             f"Checkpoint: {model['repo_id']} at {model['revision']}, {model['adaptation']}. Native loading verified: {model['native_model_loading_verified']}. The immutable initial metadata flag weights_downloaded:false records preparation time; the current cache contains {model['cache_evidence']['complete_weight_files_present']}/{model['cache_evidence']['expected_weight_shards']} expected weight shards, and actual loading/generation evidence is reported separately.", '',
             f"Execution: physical GPUs {precision['physical_gpus']}, one sharded model, {precision['dtype']}, quantization {precision['quantization']}, attention {precision['attention_implementation']}, experts {precision['experts_implementation']}, native Torch linear-attention fallback. CPU/disk offload: {precision['cpu_or_disk_offload']}. The configured native nonthinking template retains the exact frozen system/user prompt, 16 individual images at {precision['visual_tokens_per_frame']} visual tokens each, context limit {precision['max_context']}, greedy generation, and a {precision['generation']['max_new_tokens']}-token answer budget.", '',
             f"Whole-response strict format valid: {gen['strict_whole_response_format_valid']}/{EXPECTED}; independently extractable WHY: {gen['independent_why_extractable']}/{EXPECTED}; length-limited answers: {gen['length_limited']}/{EXPECTED}. Malformed responses and missing WHYs remain in the denominator. These are formatting diagnostics, not semantic accuracy.", '']
    env = metrics.get('environment') or {}
    format_audit = metrics.get('output_format_audit')
    if format_audit:
        lines.extend([f"Output-format investigation: parser errors {format_audit['parser_error_counts']}; independent field extraction {format_audit['independent_field_status_counts']}. Complete WHY/GOAL/NEXT fields can coexist with an invalid NEXT triple. The frozen strict parser requires three nonempty semicolon-delimited actions; its failure does not score WHY semantics.", ''])
    lines.extend([f"Isolated Python: {env.get('executable', 'unavailable')}; Python {env.get('python', 'unavailable')}. Exact versions: " + ', '.join(f"{name} {version}" for name, version in env.get('packages', {}).items()) + '.', ''])
    active = metrics['active_configuration']
    if active['pointer']:
        lines.extend([f"Active configuration: [{Path(active['path']).name}]({active['path']}), file SHA256 {active['file_sha256']}. The earlier [{Path(active['previous_configuration_path']).name}]({active['previous_configuration_path']}) is preserved. Repair: {active['pointer'].get('implementation_fix', 'see active pointer')}. Experimental settings unchanged: {active['experimental_settings_unchanged']}. The recorded first failure occurred while serializing native loading diagnostics, before any answer generation; current outputs use the active configuration hash.", ''])
    integrity = model.get('weight_content_integrity')
    if integrity:
        lines.extend([f"Post-generation pinned weight-content verification: passed={integrity['passed']}, {integrity['checked_shards']}/{integrity['expected_shards']} shards. {integrity['actual_tensor_bytes']} tensor-payload bytes plus {integrity['actual_header_and_prefix_bytes']} header/prefix bytes equal {integrity['actual_file_bytes']} file bytes. Each SHA256 matches its Hugging Face blob key in the exact revision snapshot; exact header lengths and contiguous tensor offsets are checked. This CPU-only audit ran after the 228-answer completion gate. Prior fresh checksum proof reused: {integrity['checksum_proof_reused']}.", ''])
    cpu = metrics.get('cpu_preflight') or {}
    lines.extend([f"CPU native-processor preflight: passed={cpu.get('passed', False)}, records={cpu.get('records', 'unavailable')}, all 16 frames complete={cpu.get('all_16_frames_complete', False)}, input tokens {cpu.get('input_tokens_min', 'unavailable')}–{cpu.get('input_tokens_max', 'unavailable')}, frozen text exact={cpu.get('all_original_text_exact', False)}, targets/references absent={cpu.get('all_targets_references_absent', False)}. Separate-image pixel-order comparison covered {cpu.get('individual_order_checked_records', 'unavailable')} records / {cpu.get('individual_order_checked_images', 'unavailable')} images. Smoke criteria passed: {metrics['smoke']['criteria_passed']}.", ''])
    latency = gen['generation_latency']
    lines.extend([f"Recorded generation latency ({latency['observations']} observations): total {fmt(latency['total_sec'], ' s')}, mean {fmt(latency['mean_sec'], ' s')}, median {fmt(latency['median_sec'], ' s')}, nearest-rank p95 {fmt(latency['p95_sec'], ' s')}. These synchronized generation timings exclude preprocessing and download. Model loading took {fmt(metrics['native_loading_latency_sec'], ' s')}. Download and end-to-end wall timing are unavailable when not separately recorded.", ''])
    for logical, record in metrics['gpu_memory']['devices'].items():
        uuids = {item['index']: item['uuid'] for probe in metrics['resources'] for item in probe.get('selected_gpus', [])}
        lines.append(f"GPU {record['physical_gpu_index']} ({uuids.get(int(logical), 'UUID unavailable')}): observed peak allocated {fmt(record['peak_allocated_gib'], ' GiB')}, reserved {fmt(record['peak_reserved_gib'], ' GiB')}, including native-loading and generation allocator records.")
    lines.extend(['', metrics['semantic_review']['status'] + ' Human review is not claimed complete. No semantic accuracy is assigned by this reporter. The existing 8B epoch1 selection and original reports remain unchanged.', ''])
    preserve = metrics.get('preservation')
    if preserve:
        lines.extend([f"Final preservation: passed={preserve['passed']}; {preserve['sha256_files_checked']}/{preserve['protected_original_files']} protected original artifacts rehashed, {len(preserve['changed_originals'])} changed and {len(preserve['missing_originals'])} missing. {preserve['checkpoint_stat_files_checked']} original checkpoint files match the saved size/mtime snapshot except {len(preserve['checkpoint_stat_changed'])} changed and {len(preserve['checkpoint_stat_missing'])} missing. Checkpoint verification uses the completed prior semantic-review snapshot and does not hash weight contents.", ''])
    if metrics['environment_failures_and_fixes']:
        lines.extend(['Retained setup failures and resolutions:', ''])
        lines.extend(f"- {item['issue']}. {item['resolution_evidence']} [{Path(item['failure_log']).name}]({item['failure_log']})" for item in metrics['environment_failures_and_fixes'])
        lines.append('')
    if metrics['failures']:
        lines.extend(['Retained execution failures:', ''])
        lines.extend(f"- {item.get('utc', 'time unavailable')}: {item.get('error', 'error unavailable')}. [Evidence]({item['artifact']})" for item in metrics['failures'])
        lines.append('')
    if metrics['postgeneration_weight_audit_failures_and_fixes']:
        lines.extend(['Retained CPU weight-audit failure and repair:', ''])
        lines.extend(f"- {item['issue']} All content SHA256 checks passed: {item['all_content_sha256_passed']}. {item['repair']} No generation was repeated. [Original failed audit]({item['artifact']})" for item in metrics['postgeneration_weight_audit_failures_and_fixes'])
        lines.append('')
    if metrics['artifacts']:
        lines.extend(['Saved evidence and review paths:', ''])
        lines.extend(f"- [{name}]({path})" for name, path in metrics['artifacts'].items())
    lines.extend(['', f"Rerun launcher: [run_hdepic_qwen35_baseline.sh]({metrics['rerun_launcher']}). Exact executed commands are included only when a saved exact_commands.json manifest exists."])
    return '\n'.join(lines) + '\n'


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == value:
        return
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(value)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def generate_report(run=OUT):
    run = Path(run).resolve(); metrics = build_metrics(run)
    previous = read_optional(run / 'metrics.json')
    without_timestamp = lambda value: {key: item for key, item in value.items()
                                      if key != 'report_created_at_utc'}
    if previous and without_timestamp(previous) == without_timestamp(metrics):
        metrics['report_created_at_utc'] = previous['report_created_at_utc']
    atomic_text(run / 'metrics.json', json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    atomic_text(run / 'execution_report.md', markdown(metrics))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=OUT)
    args = parser.parse_args(argv)
    metrics = generate_report(args.run)
    print(json.dumps({'status': metrics['status'], 'completed': metrics['generation']['completed'],
                      'denominator': EXPECTED, 'preservation_passed': metrics['preservation']['passed']}, indent=2))
    return 0 if metrics['preservation']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
