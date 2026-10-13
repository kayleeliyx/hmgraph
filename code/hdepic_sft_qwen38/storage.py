"""Physical /data-only storage checks without changing frozen model identities."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

DATA = Path('/data')
STORAGE = DATA / 'yaxuanli/qwen38-storage'
MIN_FREE_BYTES = 3 * 2**30
ALIASES = {
    Path('/tmp') / name: STORAGE / name
    for name in ('hdepic-qwen38-27b-env', 'hdepic-qwen38-27b-hf',
                 'hdepic-qwen38-sft-env', 'hdepic-qwen38-sft-checkpoints')
}
REQUIRED_ENV = ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'HF_HOME',
                'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'HF_MODULES_CACHE',
                'PIP_CACHE_DIR', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR')


def data_path(path):
    path = Path(path)
    if not path.is_absolute():
        raise RuntimeError(f'Storage path must be absolute: {path}')
    real = path.resolve()
    if not real.is_relative_to(DATA):
        raise RuntimeError(f'Storage escapes /data: {path} -> {real}')
    return real


def validate_storage(*, require_space=False, run=None):
    """Read-only guard: no allocation, migration, deletion, or GPU access."""
    paths = {}
    for name in REQUIRED_ENV:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError('Source scripts/hdepic_data_storage.sh first; missing ' + name)
        paths[name] = str(data_path(value))
    # Pin the parent interpreter instead of tempfile's fallback search, which
    # can otherwise try /tmp when a configured directory is unavailable.
    tempfile.tempdir = paths['TMPDIR']
    paths['python_tempdir'] = paths['TMPDIR']
    aliases = {}
    for alias, expected in ALIASES.items():
        if not alias.is_symlink() or not alias.is_dir():
            raise RuntimeError(f'Missing relocated legacy alias; never recreate it in /tmp: {alias}')
        real = data_path(alias)
        if real != expected or data_path(expected) != expected:
            raise RuntimeError(f'Unexpected legacy alias destination: {alias} -> {real}')
        aliases[str(alias)] = str(real)
    targets = [STORAGE, Path(paths['TMPDIR'])]
    if run is not None:
        targets.append(data_path(run))
    available = {}
    for target in targets:
        parent = data_path(target)
        while not parent.exists():
            parent = parent.parent
        available[str(target)] = shutil.disk_usage(parent).free
    if require_space and min(available.values()) < MIN_FREE_BYTES:
        raise RuntimeError('Storage blocker: /data has less than 3 GiB free; '
                           'GPU work remains stopped. No fallback outside /data. '
                           f'Available bytes: {available}')
    return {'environment': paths, 'legacy_aliases': aliases,
            'available_bytes': available, 'minimum_free_bytes': MIN_FREE_BYTES,
            'space_checked': require_space}
