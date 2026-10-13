"""Shared paths, identities and allocation checks; no model imports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/data/yaxuanli/ActGround')
RUN = PROJECT / 'runs/hdepic_sft_8b_v1'
DATASET = PROJECT / 'datasets/hdepic_sft_v2_repaired'
PYTHON = PROJECT / '.venvs/hdepic-swift-8b/bin/python'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
MODEL = PROJECT / '.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots' / REVISION
CONFIG_PATH = RUN / 'frozen_config.json'
GPU_UUIDS = {0: 'GPU-f63ffaff-cdd7-e954-cea4-b051c47fbb20',
             1: 'GPU-1ac69f34-45b7-8b31-05a6-98cce9530168'}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


sha256_file = sha256


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def atomic_jsonl(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.replace(path)


def configure_environment(cpu=False):
    values = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
              'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_HF': '1',
              'HF_HOME': str(PROJECT / '.cache/huggingface'),
              'HF_DATASETS_CACHE': str(PROJECT / '.cache/hdepic_swift_dataset'),
              'MODELSCOPE_CACHE': str(PROJECT / '.cache/modelscope'),
              'SWIFT_CACHE_DIR': str(PROJECT / '.cache/swift'),
              'IMAGE_MIN_TOKEN_NUM': '64', 'IMAGE_MAX_TOKEN_NUM': '144',
              'OMP_NUM_THREADS': '1' if cpu else '4', 'TOKENIZERS_PARALLELISM': 'false'}
    os.environ.update(values)
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_config(path=CONFIG_PATH, verify=True):
    path = Path(path)
    if path.is_dir():
        path = path / 'frozen_config.json'
    config = read_json(path)
    assert config['model']['path'] == str(MODEL)
    assert config['model']['revision'] == REVISION
    assert config['gpus']['authorized_indices'] == [0, 1]
    if verify:
        for part in ('initial_sft', 'validation'):
            for key in ('export', 'scoring', 'prompts'):
                info = config['data'][part]
                assert sha256(info[key]) == info[key + '_sha256'], f'Frozen input changed: {part}/{key}'
    return config


def load_records(config, partition):
    assert partition in ('initial_sft', 'validation'), 'Experiment never loads test or second-stage records'
    info = config['data'][partition]
    rows, sidecars, prompts = [read_jsonl(info[key]) for key in ('export', 'scoring', 'prompts')]
    assert len(rows) == len(sidecars) == len(prompts) == info['count']
    assert len({s['id'] for s in sidecars}) == info['count']
    for i, (row, sidecar, prompt) in enumerate(zip(rows, sidecars, prompts)):
        assert sidecar['row_index'] == i and sidecar['partition'] == partition
        assert set(row) == {'messages', 'images'} and set(prompt) == {'messages', 'images'}
        assert prompt == {'messages': row['messages'][:2], 'images': row['images']}
        assert [m['role'] for m in prompt['messages']] == ['system', 'user']
        assert len(prompt['images']) == 16
    return rows, sidecars, prompts


def gpu_inventory(indices=(0, 1)):
    devices = []
    for index in indices:
        assert index in GPU_UUIDS, 'Only GPUs 0 and 1 are authorized for this run'
        result = subprocess.run(['nvidia-smi', '-i', str(index),
            '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        fields = next(csv.reader(result.stdout.splitlines()))
        processes = subprocess.run(['nvidia-smi', '-i', str(index),
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True).stdout.strip()
        assert fields[1].strip() == GPU_UUIDS[index], 'Physical GPU identity changed'
        used, free, utilization = map(int, (fields[4], fields[5], fields[6]))
        devices.append({'index': index, 'uuid': fields[1].strip(), 'name': fields[2].strip(),
                        'total_mib': int(fields[3]), 'used_mib': used, 'free_mib': free,
                        'utilization_percent': utilization, 'compute_processes': processes,
                        'idle': used < 1024 and free >= 47000 and utilization == 0 and not processes})
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'devices': devices,
            'scheduler': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_GPUS',
                'SLURM_STEP_GPUS', 'PBS_JOBID', 'LSB_JOBID') if os.environ.get(k)}}


def require_idle_gpus(indices, run=RUN, phase='load'):
    observation = gpu_inventory(indices)
    # nvidia-smi utilization can briefly lag after our preceding child exits.
    # Only wait for a device with no compute process and no model allocation;
    # an occupied device is never treated as our job or made available by force.
    for _ in range(15):
        if all(d['idle'] for d in observation['devices']):
            break
        if not all(not d['compute_processes'] and d['used_mib'] < 1024
                   and d['free_mib'] >= 47000 for d in observation['devices']):
            break
        time.sleep(1)
        observation = gpu_inventory(indices)
    atomic_json(Path(run) / f'gpu_before_{phase}.json', observation)
    unavailable = [d['index'] for d in observation['devices'] if not d['idle']]
    if unavailable:
        raise RuntimeError(f'Authorized GPUs are occupied: {unavailable}; no process will be terminated')
    if os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'):
        allocation = os.environ.get('SLURM_STEP_GPUS') or os.environ['SLURM_JOB_GPUS']
        allocated = set(allocation.split(','))
        assert all(str(i) in allocated or GPU_UUIDS[i] in allocated for i in indices), 'Scheduler allocation excludes a requested GPU'
    expected = ','.join(GPU_UUIDS[i] for i in indices)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected, 'CUDA visibility must match physical authorized UUIDs'
    return observation
