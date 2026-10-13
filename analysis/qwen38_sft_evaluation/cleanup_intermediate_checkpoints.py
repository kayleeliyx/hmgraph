"""Delete only the seven intermediate checkpoints approved on 2026-10-11."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/yaxuanli/ActGround')
RUN = ROOT / 'runs/hdepic_sft_qwen38_27b_v1'
STORAGE = Path('/data/yaxuanli/qwen38-storage/hdepic-qwen38-sft-checkpoints')
STEPS = (250, 500, 750, 1000, 1250, 1500, 1750)
FILES = {'adapter_config.json', 'adapter_model.safetensors', 'rank-00000.pt', 'rank-00001.pt', 'completed.json'}
AUDIT = ROOT / 'analysis/qwen38_sft_evaluation/checkpoint_cleanup_20261011'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def protected_hashes():
    hashes = {}
    for step in (909, 1818):
        path = RUN / 'training/main/checkpoints' / f'checkpoint-{step}'
        assert not path.is_symlink() and path.resolve() == path
        marker = json.loads((path / 'completed.json').read_text())
        assert marker['passed'] and marker['state']['step'] == step
        assert {x.name for x in path.iterdir()} == FILES
        for name in sorted(FILES):
            file = path / name
            assert file.is_file() and not file.is_symlink()
            hashes[str(file)] = sha(file)
            if name != 'completed.json':
                assert hashes[str(file)] == marker['files'][name]['sha256']
                assert file.stat().st_size == marker['files'][name]['bytes']
    for path in (RUN / 'selected_checkpoint.json', RUN / 'training/main/completed.json',
                 RUN / 'predictions/epoch1/generation.jsonl'):
        hashes[str(path)] = sha(path)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert STORAGE.resolve() == STORAGE and STORAGE.is_dir()
    alias = Path('/tmp/hdepic-qwen38-sft-checkpoints')
    assert alias.is_symlink() and alias.resolve() == STORAGE
    assert not (AUDIT / 'completed.json').exists(), 'Cleanup already completed'
    with (RUN / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep = protected_hashes()
        candidates = []
        for step in STEPS:
            path = STORAGE / 'training' / f'checkpoint-{step}'
            leaf = RUN / 'training/main/checkpoints' / path.name
            assert path.is_dir() and not path.is_symlink() and path.resolve() == path
            assert leaf.is_symlink() and leaf.resolve() == path
            assert {x.name for x in path.iterdir()} == FILES
            marker = json.loads((path / 'completed.json').read_text())
            assert marker['passed'] and marker['state']['step'] == step
            assert marker['identity']['config_sha256'] == sha(RUN / 'frozen_config.json')
            allocated = path.stat().st_blocks * 512
            for file in path.iterdir():
                assert file.is_file() and not file.is_symlink() and file.stat().st_nlink == 1
                if file.name != 'completed.json':
                    assert file.stat().st_size == marker['files'][file.name]['bytes']
                allocated += file.stat().st_blocks * 512
            candidates.append({'step': step, 'physical_directory': str(path), 'leaf_symlink': str(leaf),
                               'allocated_bytes': allocated, 'marker': marker})
        overview = {'checkpoints': list(STEPS), 'allocated_gib': sum(x['allocated_bytes'] for x in candidates) / 2**30,
                    'protected_steps': [909, 1818], 'execute': args.execute}
        print(json.dumps(overview), flush=True)
        if not args.execute:
            return
        assert AUDIT.resolve().is_relative_to(Path('/data'))
        AUDIT.mkdir(exist_ok=True)
        before = shutil.disk_usage('/data').free
        dump(AUDIT / 'plan.json', {'authorization': 'User: okay. lets delete and finish the evaluation',
                                  'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'free_bytes_before': before, 'candidates': candidates, 'protected_sha256': keep})
        deleted = []
        for entry in candidates:
            path, leaf = Path(entry['physical_directory']), Path(entry['leaf_symlink'])
            for name in sorted(FILES):
                (path / name).unlink()
            path.rmdir()
            leaf.unlink()
            deleted.append(entry['step'])
            dump(AUDIT / 'progress.json', {'deleted_steps': deleted})
        assert alias.is_symlink() and alias.resolve() == STORAGE
        assert protected_hashes() == keep, 'Protected artifact changed'
        after = shutil.disk_usage('/data').free
        result = {**overview, 'deleted_steps': deleted, 'free_bytes_before': before, 'free_bytes_after': after,
                  'free_gib_after': after / 2**30, 'protected_hashes_unchanged': True,
                  'completed_utc': datetime.now(timezone.utc).isoformat()}
        dump(AUDIT / 'completed.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
