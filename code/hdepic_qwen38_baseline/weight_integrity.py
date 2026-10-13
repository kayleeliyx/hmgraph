"""Post-generation, CPU-only SHA256 verification of the pinned HF weight blobs.

The 228-answer completion gate is checked before reading any weight content.
This module never imports Torch, loads a model, or invokes inference/downloads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from .data import immutable_write, json_bytes, require, rows, sha256
from .model import CACHE, REPO, REVISION, RUN
from .report import load_active_configuration
from .resources import WEIGHT_BYTES


def require_completed_generation(run):
    run = Path(run)
    config, evidence = load_active_configuration(run)
    marker_path = run / 'predictions/completed.json'
    require(marker_path.is_file(), 'Refuse weight hashing while generation is incomplete')
    marker = json.loads(marker_path.read_text())
    path = run / 'predictions/generation.jsonl'
    require(marker.get('passed') is True and marker.get('count') == 228
            and marker.get('configuration_sha256') == config['configuration_sha256']
            and marker.get('generation_sha256') == sha256(path),
            'Refuse weight hashing without a valid 228-answer completion marker')
    generated = rows(path); manifest = json.loads((run / 'inputs/manifest.json').read_text())
    require(len(generated) == 228 and len({row['id'] for row in generated}) == 228
            and {row['id'] for row in generated} == set(manifest['ordered_ids'])
            and all(row['status'] == 'completed' for row in generated),
            'Refuse weight hashing: full frozen answer cohort is not complete')
    return config, evidence, marker


def verify_blob(path, blob_directory):
    """Compare a complete pinned-snapshot link with its HF SHA256 blob key."""
    path = Path(path); blob_directory = Path(blob_directory).resolve()
    require(path.is_file(), f'Expected complete weight shard is missing: {path}')
    resolved = path.resolve()
    require(resolved.is_relative_to(blob_directory)
            and re.fullmatch(r'[a-f0-9]{64}', resolved.name),
            f'Weight shard does not resolve to a SHA256-keyed HF blob: {path}')
    before = resolved.stat(); digest = hashlib.sha256(); started = time.perf_counter()
    with resolved.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = resolved.stat(); actual = digest.hexdigest()
    require(before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
            f'Weight blob changed during hashing: {resolved}')
    return {'filename': path.name, 'snapshot_path': str(path), 'resolved_blob_path': str(resolved),
            'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
            'expected_sha256_hf_blob_key': resolved.name, 'actual_sha256': actual,
            'passed': actual == resolved.name, 'hash_latency_sec': time.perf_counter() - started}


def inspect_safetensors_header(path):
    """Validate tensor offsets and separate payload from format overhead."""
    path = Path(path); stat = path.stat()
    with path.open('rb') as handle:
        length_bytes = handle.read(8)
        require(len(length_bytes) == 8, f'Safetensors length prefix is incomplete: {path}')
        header_length = int.from_bytes(length_bytes, byteorder='little', signed=False)
        require(2 <= header_length <= min(100 * 1024 ** 2, stat.st_size - 8),
                f'Safetensors header length is invalid: {path}')
        raw = handle.read(header_length)
    require(len(raw) == header_length, f'Safetensors JSON header is incomplete: {path}')
    header = json.loads(raw)
    require(isinstance(header, dict), f'Safetensors header is not an object: {path}')
    intervals = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        require(isinstance(tensor, dict) and 'data_offsets' in tensor,
                f'Safetensors tensor entry lacks data offsets: {name}')
        offsets = tensor['data_offsets']
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in offsets)
                and offsets[0] <= offsets[1], f'Safetensors tensor offsets are invalid: {name}')
        intervals.append((offsets[0], offsets[1], name))
    require(intervals, f'Safetensors shard has no tensor entries: {path}')
    cursor = 0
    for start, end, name in sorted(intervals):
        require(start == cursor, f'Safetensors tensor offsets overlap or leave a gap: {name}')
        cursor = end
    tensor_bytes = sum(end - start for start, end, _ in intervals)
    header_bytes = 8 + header_length
    require(stat.st_size == header_bytes + cursor and tensor_bytes == cursor,
            f'Safetensors file length differs from its header and tensor offsets: {path}')
    return {'file_bytes': stat.st_size, 'length_prefix_bytes': 8, 'header_length_bytes': header_length,
            'header_and_prefix_bytes': header_bytes, 'tensor_bytes': tensor_bytes,
            'tensor_count': len(intervals), 'maximum_tensor_data_end_offset': cursor,
            'header_json_sha256': hashlib.sha256(raw).hexdigest(),
            'file_length_matches_header_and_tensor_offsets': True}


def reuse_failed_checksum_proof(run, config, marker, shard_names, snapshot, blobs):
    """Reuse only complete passing hashes whose source identity/stats still match."""
    paths = sorted((Path(run) / 'audit/weight_integrity_failures').glob('*.json'), reverse=True)
    for path in paths:
        failed = json.loads(path.read_text())
        if (failed.get('checkpoint_repo') != REPO or failed.get('checkpoint_revision') != REVISION
                or failed.get('configuration_sha256') != config['configuration_sha256']
                or failed.get('generation_sha256') != marker['generation_sha256']):
            continue
        records = failed.get('weight_shards', [])
        if len(records) != len(shard_names) or {row['filename'] for row in records} != set(shard_names):
            continue
        compatible = True
        for row in records:
            source = snapshot / row['filename']; resolved = source.resolve()
            if not source.is_file() or not resolved.is_relative_to(blobs.resolve()):
                compatible = False; break
            stat = resolved.stat()
            if (row.get('passed') is not True or str(source) != row['snapshot_path']
                    or str(resolved) != row['resolved_blob_path'] or stat.st_size != row['bytes']
                    or stat.st_mtime_ns != row['mtime_ns']
                    or row.get('expected_sha256_hf_blob_key') != resolved.name
                    or row.get('actual_sha256') != resolved.name
                    or not re.fullmatch(r'[a-f0-9]{64}', resolved.name)):
                compatible = False; break
        if compatible:
            return {row['filename']: row for row in records}, {
                'artifact': str(path), 'artifact_sha256': sha256(path),
                'all_18_content_sha256_previously_passed': True, 'all_paths_sizes_mtimes_and_hash_keys_unchanged': True,
                'original_full_hash_elapsed_sec': failed.get('elapsed_sec'),
                'original_failure': 'Compared complete safetensors file sizes against tensor payload bytes from the index; SHA256 comparisons themselves all passed.'}
    return {}, None


def validate_cached_inventory(cached, shard_names, snapshot, blobs, expected_tensor_bytes):
    """Reuse a complete pinned inventory only when every prior proof still fits.

    A successful audit is a proof for all18 official shards. Checking only the
    rows remaining in an edited/truncated cache would silently weaken that
    claim. Reuse preserves the original full hashes and checks current links,
    stats, hash keys and format totals without rehashing the weight payload.
    """
    snapshot, blobs = Path(snapshot), Path(blobs).resolve()
    records = cached.get('weight_shards', [])
    require(cached.get('checkpoint_repo') == REPO
            and cached.get('checkpoint_snapshot') == str(snapshot)
            and cached.get('expected_shards') == cached.get('checked_shards') == len(shard_names) == 18
            and isinstance(records, list) and len(records) == 18
            and all(isinstance(row, dict) for row in records),
            'Cached weight-integrity inventory is incomplete or uses a different checkpoint')
    names = [row.get('filename') for row in records]
    require(len(set(names)) == 18 and set(names) == set(shard_names),
            'Cached weight-integrity filenames differ from the official18-shard inventory')
    for row in records:
        path = snapshot / row['filename']
        require(row.get('snapshot_path') == str(path) and path.is_file(),
                'Cached weight shard has a missing or different pinned snapshot path')
        resolved = path.resolve()
        require(resolved.is_relative_to(blobs) and re.fullmatch(r'[a-f0-9]{64}', resolved.name)
                and row.get('resolved_blob_path') == str(resolved)
                and row.get('expected_sha256_hf_blob_key') == row.get('actual_sha256') == resolved.name
                and row.get('passed') is True,
                'Cached weight shard no longer matches its original passing SHA256 blob proof')
        stat = resolved.stat()
        require(stat.st_size == row.get('bytes') == row.get('file_bytes')
                and stat.st_mtime_ns == row.get('mtime_ns'), 'A previously hashed weight shard changed')
        for field in ('file_bytes', 'header_and_prefix_bytes', 'header_length_bytes', 'tensor_bytes'):
            value = row.get(field)
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    'Cached weight shard format byte counts are invalid')
        require(row.get('length_prefix_bytes') == 8
                and row['header_and_prefix_bytes'] == 8 + row['header_length_bytes']
                and row['file_bytes'] == row['header_and_prefix_bytes'] + row['tensor_bytes']
                and row.get('file_length_matches_header_and_tensor_offsets') is True,
                'Cached weight shard format accounting is inconsistent')
    file_bytes = sum(row['file_bytes'] for row in records)
    header_bytes = sum(row['header_and_prefix_bytes'] for row in records)
    tensor_bytes = sum(row['tensor_bytes'] for row in records)
    require(cached.get('expected_tensor_bytes') == cached.get('actual_tensor_bytes') == tensor_bytes == expected_tensor_bytes
            and cached.get('actual_file_bytes') == file_bytes
            and cached.get('actual_header_and_prefix_bytes') == header_bytes
            and file_bytes == header_bytes + tensor_bytes
            and cached.get('all_file_lengths_match_header_and_tensor_offsets') is True,
            'Cached weight-integrity totals differ from the complete pinned tensor inventory')


def verify_weights(run=RUN, cache=CACHE):
    run = Path(run).resolve(); cache = Path(cache).resolve()
    config, evidence, marker = require_completed_generation(run)
    identity_path = run / 'model_identity.json'; identity = json.loads(identity_path.read_text())
    require(identity['repo_id'] == config['repo_id'] == REPO
            and identity['revision'] == config['revision'] == REVISION,
            'Wrong pinned checkpoint identity')
    require(sha256(identity_path) == config['model_identity_sha256'], 'Frozen model identity metadata changed')
    index_path = run / 'model_metadata/model.safetensors.index.json'
    require(sha256(index_path) == identity['metadata_sha256']['model.safetensors.index.json'],
            'Pinned model weight index changed')
    weight_index = json.loads(index_path.read_text())
    expected_tensor_bytes = weight_index['metadata']['total_size']
    require(expected_tensor_bytes == WEIGHT_BYTES, 'Pinned tensor payload byte count changed')
    shard_names = sorted(set(weight_index['weight_map'].values()))
    require(shard_names == sorted(identity['weight_shards']) and len(shard_names) == 18,
            'Expected official 18-shard weight inventory changed')
    repository = cache / 'models--Qwen--Qwen3.8-27B'
    snapshot = repository / 'snapshots' / REVISION
    destination = run / 'audit/model_weight_integrity.json'
    if destination.exists():
        cached = json.loads(destination.read_text())
        require(cached['passed'] is True and cached['checkpoint_revision'] == REVISION
                and cached['generation_sha256'] == marker['generation_sha256']
                and cached['configuration_sha256'] == config['configuration_sha256'],
                'Existing weight-integrity result has a different completed run identity')
        validate_cached_inventory(cached, shard_names, snapshot, repository / 'blobs', expected_tensor_bytes)
        return cached
    started = time.perf_counter(); verified = []
    reused, checksum_proof = reuse_failed_checksum_proof(run, config, marker, shard_names, snapshot, repository / 'blobs')
    for index, name in enumerate(shard_names, 1):
        record = dict(reused[name]) if name in reused else verify_blob(snapshot / name, repository / 'blobs')
        record.update(inspect_safetensors_header(snapshot / name))
        record['checksum_reused_from_fresh_failed_audit'] = name in reused
        verified.append(record)
        print(json.dumps({'verified_weight_shards': index, 'expected': 18,
                          'filename': name, 'passed': record['passed']}), flush=True)
    total_files = sum(record['file_bytes'] for record in verified)
    total_headers = sum(record['header_and_prefix_bytes'] for record in verified)
    total_tensors = sum(record['tensor_bytes'] for record in verified)
    result = {'passed': all(record['passed'] for record in verified)
                        and total_tensors == expected_tensor_bytes and total_files == total_headers + total_tensors,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_repo': REPO, 'checkpoint_revision': REVISION,
              'checkpoint_snapshot': str(snapshot), 'weight_shards': verified,
              'expected_shards': 18, 'checked_shards': len(verified),
              'expected_tensor_bytes': expected_tensor_bytes, 'actual_tensor_bytes': total_tensors,
              'actual_file_bytes': total_files, 'actual_header_and_prefix_bytes': total_headers,
              'all_file_lengths_match_header_and_tensor_offsets': True,
              'checksum_proof_reused': bool(reused), 'checksum_proof_source': checksum_proof,
              'newly_hashed_weight_bytes': sum(row['bytes'] for row in verified if not row['checksum_reused_from_fresh_failed_audit']),
              'format_accounting': 'Index metadata.total_size is tensor payload only. Complete safetensors files also contain an 8-byte little-endian header-length prefix and padded JSON header per shard.',
              'generation_sha256': marker['generation_sha256'],
              'configuration_sha256': config['configuration_sha256'],
              'configuration_evidence': evidence, 'elapsed_sec': time.perf_counter() - started,
              'comparison': 'SHA256 of each complete shard equals its SHA256-keyed Hugging Face LFS blob filename in the exact repository/revision snapshot; the frozen metadata/index inventory is independently checked.',
              'gpu_accessed': False, 'model_loaded': False, 'inference_run': False,
              'download_run': False, 'test_accessed': False, 'training_run': False}
    if result['passed']:
        immutable_write(destination, json_bytes(result))
    else:
        immutable_write(run / f'audit/weight_integrity_failures/{time.time_ns()}.json', json_bytes(result))
        raise ValueError('Pinned model weight content SHA256 or total byte count failed verification')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--cache', type=Path, default=CACHE)
    args = parser.parse_args(argv)
    result = verify_weights(args.run, args.cache)
    print(json.dumps({key: result[key] for key in ('passed', 'checked_shards', 'actual_tensor_bytes',
                                                  'actual_header_and_prefix_bytes', 'actual_file_bytes',
                                                  'checksum_proof_reused', 'elapsed_sec')}, indent=2))


if __name__ == '__main__':
    main()
