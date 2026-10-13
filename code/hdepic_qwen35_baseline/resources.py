"""Read-only, fail-closed resource gates before weights or model imports.

Only physical GPUs 0 and 1 are authorized. This module never imports Torch,
loads a model, downloads weights, changes device state, or stops other jobs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

PROJECT = Path('/data/yaxuanli/ActGround')
RUN = PROJECT / 'runs/hdepic_qwen35_35b_baseline_v1'
CACHE = PROJECT / '.cache/huggingface-qwen35'
REVISION = '59d61f3ce65a6d9863b86d2e96597125219dc754'
AUTHORIZED_GPUS = {0: 'GPU-f63ffaff-cdd7-e954-cea4-b051c47fbb20',
                   1: 'GPU-1ac69f34-45b7-8b31-05a6-98cce9530168'}
WEIGHT_BYTES = 71903655008
DISK_HEADROOM_BYTES = 10 * 2 ** 30
MIN_FREE_MIB = 42 * 1024
MAX_IDLE_USED_MIB = 256
GPU_COMMAND = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
PROCESS_COMMAND = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                   '--format=csv,noheader,nounits']


class ResourceBlocked(RuntimeError):
    def __init__(self, reasons, report=None):
        self.reasons = list(reasons)
        self.report = report or {}
        super().__init__('Resource gate blocked: ' + '; '.join(self.reasons))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_audit(run, phase, report):
    """Unique append-only artifact names; preserve previous resource checks."""
    folder = Path(run) / 'resources'
    folder.mkdir(parents=True, exist_ok=True)
    phase = re.sub(r'[^A-Za-z0-9_-]', '_', phase)
    path = folder / f'{time.time_ns()}_{phase}.json'
    report['audit_path'] = str(path.resolve())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    return path


def parse_gpu_csv(raw):
    rows = []
    for fields in csv.reader(io.StringIO(raw)):
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f'Expected seven GPU CSV fields, received {len(fields)}')
        index, uuid, name, total, used, free, utilization = [x.strip() for x in fields]
        rows.append({'index': int(index), 'uuid': uuid, 'name': name,
                     'memory_total_mib': int(total), 'memory_used_mib': int(used),
                     'memory_free_mib': int(free), 'utilization_percent': int(utilization)})
    if not rows or len({g['index'] for g in rows}) != len(rows):
        raise ValueError('GPU query returned no devices or duplicate indices')
    if len({g['uuid'] for g in rows}) != len(rows):
        raise ValueError('GPU query returned duplicate UUIDs')
    return rows


def parse_process_csv(raw):
    rows = []
    for fields in csv.reader(io.StringIO(raw)):
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(f'Expected four process CSV fields, received {len(fields)}')
        uuid, pid, name, memory = [x.strip() for x in fields]
        rows.append({'gpu_uuid': uuid, 'pid': int(pid), 'process_name': name,
                     'used_gpu_memory_mib': int(memory) if memory.isdecimal() else None})
    return rows


def query_resources():
    report = {'query_started_at_utc': utc_now(), 'commands': [], 'raw': {}}
    for name, command in [('gpu_csv', GPU_COMMAND), ('compute_process_csv', PROCESS_COMMAND)]:
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            report['query_error'] = str(error)
            raise ResourceBlocked([f'Read-only {name} query failed: {error}'], report) from error
        report['commands'].append({'argv': command, 'returncode': result.returncode,
                                   'stderr': result.stderr})
        report['raw'][name] = result.stdout
        if result.returncode:
            raise ResourceBlocked([f'Read-only {name} query exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}'], report)
    report['query_finished_at_utc'] = utc_now()
    try:
        report['gpus'] = parse_gpu_csv(report['raw']['gpu_csv'])
        report['compute_processes'] = parse_process_csv(report['raw']['compute_process_csv'])
    except ValueError as error:
        raise ResourceBlocked([f'Unreliable NVIDIA resource-query output: {error}'], report) from error
    return report


def allocation_indices(value, gpus):
    by_uuid = {g['uuid']: g['index'] for g in gpus}
    if value.strip().lower() in {'', '-1', 'none', 'void', 'nodevfiles'}:
        return set()
    if value.strip().lower() == 'all':
        return {g['index'] for g in gpus}
    selected = set()
    for token in value.split(','):
        token = token.strip()
        if token in by_uuid:
            selected.add(by_uuid[token])
        elif token.isdecimal():
            selected.add(int(token))
        elif re.fullmatch(r'\d+-\d+', token):
            first, last = map(int, token.split('-'))
            if last < first:
                raise ValueError(f'Invalid allocation range {token!r}')
            selected.update(range(first, last + 1))
        else:
            raise ValueError(f'Unsupported allocation token {token!r}; require physical indices or full GPU UUIDs')
    known = {g['index'] for g in gpus}
    if not selected <= known:
        raise ValueError(f'Allocation names unknown physical indices: {sorted(selected - known)}')
    return selected


def assess_idle_pair(snapshot, *, selected_indices=(0, 1), environ=None):
    """Pure gate for regression tests; PID occupancy overrides low utilization."""
    environ = os.environ if environ is None else environ
    reasons = []
    if tuple(sorted(selected_indices)) != (0, 1):
        reasons.append(f'Only the complete authorized physical pair [0, 1] may be selected; received {list(selected_indices)}')
    by_index = {g['index']: g for g in snapshot['gpus']}
    allocations = {}
    for variable in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'NVIDIA_VISIBLE_DEVICES'):
        if variable not in environ:
            continue
        value = environ[variable]
        try:
            indices = allocation_indices(value, snapshot['gpus'])
            allocations[variable] = {'value': value, 'physical_indices': sorted(indices)}
            if not {0, 1} <= indices:
                reasons.append(f'{variable}={value!r} does not allocate both authorized GPUs 0 and 1')
        except ValueError as error:
            reasons.append(f'{variable}: {error}')
    selected = []
    for index, expected_uuid in AUTHORIZED_GPUS.items():
        gpu = by_index.get(index)
        if gpu is None:
            reasons.append(f'Authorized physical GPU {index} is absent from NVIDIA query')
            continue
        selected.append(gpu)
        if gpu['uuid'] != expected_uuid:
            reasons.append(f'GPU {index} UUID mismatch: {gpu["uuid"]}, expected {expected_uuid}')
        processes = [p for p in snapshot['compute_processes'] if p['gpu_uuid'] == gpu['uuid']]
        if processes:
            reasons.append(f'GPU {index} occupied by compute PID(s) ' + ', '.join(f'{p["pid"]} ({p["process_name"]})' for p in processes))
        if gpu['utilization_percent'] != 0:
            reasons.append(f'GPU {index} utilization is {gpu["utilization_percent"]}%, require idle 0%')
        if gpu['memory_used_mib'] > MAX_IDLE_USED_MIB:
            reasons.append(f'GPU {index} uses {gpu["memory_used_mib"]} MiB despite idle requirement (limit {MAX_IDLE_USED_MIB} MiB)')
        if gpu['memory_free_mib'] < MIN_FREE_MIB:
            reasons.append(f'GPU {index} has {gpu["memory_free_mib"]} MiB free, require at least {MIN_FREE_MIB} MiB (42 GiB)')
    result = {**snapshot, 'checked_at_utc': utc_now(), 'passed': not reasons,
              'authorized_physical_indices': [0, 1], 'authorized_gpu_uuids': AUTHORIZED_GPUS,
              'selected_gpus': selected, 'allocation_environment': allocations,
              'required_free_mib_per_gpu': MIN_FREE_MIB, 'blocking_reasons': reasons,
              'cuda_visible_devices': ','.join(AUTHORIZED_GPUS.values()),
              'policy': 'Reject every compute PID, nonzero utilization, excess used memory, wrong UUID, or partial allocation. Never choose any other physical GPU.'}
    if reasons:
        raise ResourceBlocked(reasons, result)
    return result


def require_idle_pair(run=RUN, phase='before_weight_download', *, snapshot=None, environ=None):
    try:
        result = assess_idle_pair(query_resources() if snapshot is None else snapshot, environ=environ)
    except ResourceBlocked as error:
        error.report.update({'phase': phase, 'passed': False, 'blocking_reasons': error.reasons})
        save_audit(run, phase, error.report)
        raise
    result['phase'] = phase
    save_audit(run, phase, result)
    return result


def verified_cached_weight_bytes(cache=CACHE, run=RUN):
    """Deduct complete SHA256-verified HF blobs only; partial files never count."""
    index = Path(run) / 'model_metadata/model.safetensors.index.json'
    if not index.is_file():
        return 0, []
    names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    snapshot = Path(cache) / 'models--Qwen--Qwen3.5-35B-A3B/snapshots' / REVISION
    verified = []
    for name in names:
        path = snapshot / name
        if not path.is_file():
            continue
        target = path.resolve()
        if not re.fullmatch(r'[a-f0-9]{64}', target.name):
            continue
        digest = hashlib.sha256()
        with target.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() == target.name:
            verified.append({'filename': name, 'path': str(path), 'sha256': target.name,
                             'bytes': target.stat().st_size})
    return min(sum(v['bytes'] for v in verified), WEIGHT_BYTES), verified


def require_disk_space(run=RUN, cache=CACHE, phase='before_weight_download', *, free_bytes=None):
    cached, files = verified_cached_weight_bytes(cache, run)
    available = shutil.disk_usage(PROJECT).free if free_bytes is None else free_bytes
    remaining = WEIGHT_BYTES - cached
    required = remaining + DISK_HEADROOM_BYTES
    reasons = [] if available >= required else [f'Disk has {available} bytes free; need {required} bytes ({remaining} remaining weights + {DISK_HEADROOM_BYTES} headroom)']
    result = {'phase': phase, 'checked_at_utc': utc_now(), 'passed': not reasons,
              'filesystem_path': str(PROJECT), 'checkpoint_revision': REVISION,
              'planned_weight_bytes': WEIGHT_BYTES, 'verified_cached_weight_bytes': cached,
              'verified_cached_files': files, 'remaining_weight_bytes': remaining,
              'headroom_bytes': DISK_HEADROOM_BYTES, 'required_free_bytes': required,
              'actual_free_bytes': available, 'blocking_reasons': reasons,
              'weights_downloaded_by_gate': False}
    save_audit(run, 'disk_' + phase, result)
    if reasons:
        raise ResourceBlocked(reasons, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--phase', default='read_only_availability_check')
    parser.add_argument('--check-disk', action='store_true')
    parser.add_argument('--cache', type=Path, default=CACHE)
    args = parser.parse_args(argv)
    try:
        result = {'gpu': require_idle_pair(args.run, args.phase)}
        if args.check_disk:
            result['disk'] = require_disk_space(args.run, args.cache, args.phase)
        print(json.dumps(result, indent=2))
    except ResourceBlocked as error:
        print(json.dumps({'passed': False, 'error': str(error), 'audit_path': error.report.get('audit_path')}, indent=2))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
