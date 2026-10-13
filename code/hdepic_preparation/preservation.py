"""Snapshot and verify v1, E0/E1, and pre-existing tracked user changes."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

from code.e0.dataset import atomic_write_json, sha256_file, stable_hash
from .revise import DEFAULT_OUTPUT, DEFAULT_SOURCE, PROJECT


def tree(root, hash_non_jpeg=False):
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            s = p.stat()
            result[str(p.relative_to(root))] = {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}
            if hash_non_jpeg and p.suffix != ".jpg":
                result[str(p.relative_to(root))]["sha256"] = sha256_file(p)
    return result


def baseline(out, source):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preservation_before.json"
    if path.exists():
        verify(out, source)
        return
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", "HEAD"], cwd=PROJECT).decode().split("\0")
    user = {name: sha256_file(PROJECT / name) if (PROJECT / name).is_file() else None for name in changed if name}
    saved = {"v1": tree(source, True), "historical": {name: tree(PROJECT / "runs" / name) for name in ("e0", "e1_online_why", "e1_local_why")}, "tracked_user_changes": user, "v1_root": str(source)}
    atomic_write_json(path, saved)
    legacy = [PROJECT / "code/hdepic_preparation" / name for name in ("__init__.py", "build.py", "frames.py", "export.py", "audit_annotations.py", "report.py")]
    legacy += [PROJECT / "scripts" / name for name in ("build_hdepic_supervision.py", "extract_frames.py", "make_swift_sft.py", "prepare_hdepic_sft.sh")]
    legacy += [PROJECT / "code" / name for name in ("__init__.py", "e0/__init__.py", "e0/dataset.py", "e0/frames.py", "tests/test_hdepic_preparation.py")]
    with zipfile.ZipFile(out / "v1_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in legacy:
            archive.write(p, str(p.relative_to(PROJECT)))
    print(f"Saved preservation baseline: {len(saved['v1'])} v1 files", flush=True)


def verify(out, source):
    saved = json.loads((out / "preservation_before.json").read_text())
    assert saved["v1_root"] == str(source)
    assert tree(source, True) == saved["v1"], "v1 outputs changed"
    for name, before in saved["historical"].items():
        assert tree(PROJECT / "runs" / name) == before, f"Historical {name} changed"
    for name, digest in saved["tracked_user_changes"].items():
        p = PROJECT / name
        assert (sha256_file(p) if p.is_file() else None) == digest, f"Pre-existing user change modified: {name}"
    result = {"passed": True, "v1_outputs_unchanged": True, "fixed_split_sha256": sha256_file(source / "split_manifest.json"), "historical_e0_e1_unchanged": True, "tracked_user_changes_unchanged": True, "v1_files": len(saved["v1"]), "historical_files": sum(len(x) for x in saved["historical"].values()), "v1_tree_hash": stable_hash(saved["v1"]), "method": "all file paths, byte sizes and nanosecond mtimes; SHA256 for all non-JPEG v1 files and existing tracked changes; JPEG SHA256 additionally verified by export audit"}
    atomic_write_json(out / "preservation_audit.json", result)
    print("Preservation audit passed: v1, E0/E1, existing user changes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    (verify if a.verify else baseline)(a.output.resolve(), a.source.resolve())
