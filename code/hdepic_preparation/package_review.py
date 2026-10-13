"""Package actual source, complete exports, and a portable 20-row review set."""
import argparse
import copy
import json
import zipfile
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from .build import read_jsonl
from .revise import DEFAULT_OUTPUT, PROJECT


def package(out):
    audit_path = out / "modifier_repair_audit.json"
    if not audit_path.is_file():
        audit_path = out / "revision_audit.json"
    assert json.loads(audit_path.read_text())["passed"]
    assert json.loads((out / "preservation_audit.json").read_text())["passed"]
    exports = list(read_jsonl(out / "sample/training20.jsonl"))
    sidecars = list(read_jsonl(out / "sample/training20.scoring.jsonl"))
    text = list(read_jsonl(out / "sample/training20.text.jsonl"))
    assert len(exports) == len(sidecars) == len(text) == 20
    members = {}
    source = list((PROJECT / "code/hdepic_preparation").glob("*.py"))
    source += [PROJECT / "code" / p for p in ("__init__.py", "e0/__init__.py", "e0/dataset.py", "e0/frames.py", "tests/__init__.py", "tests/test_hdepic_preparation.py", "tests/test_hdepic_revision.py")]
    source += [PROJECT / "scripts" / p for p in ("build_hdepic_supervision.py", "extract_frames.py", "make_swift_sft.py", "prepare_hdepic_sft.sh", "prepare_hdepic_sft_v2.sh", "run_hdepic_sft_smoke_8b.sh")]
    for path in source:
        members[str(path.relative_to(PROJECT))] = path
    smoke = PROJECT / "runs/hdepic_sft_smoke_8b"
    report_paths = [smoke / name for name in ("README.md", "readiness_status.json", "smoke_dataset_manifest.json", "setup/pip-freeze.txt", "setup/requirements-direct.txt", "setup/environment_identity.json", "setup/checkpoint_identity.json", "setup/checkpoint_source_manifest.json", "setup/ready_environment.json", "cpu_preflight/preprocessing_audit.json")]
    for path in report_paths:
        if path.is_file():
            members[str(path.relative_to(PROJECT))] = path
    for path in out.rglob("*"):
        if path.is_file() and path.suffix not in (".log",) and path.name not in ("preservation_before.json", "review_bundle_manifest.json", "review_package_audit.json") and "review_bundle" not in path.parts:
            members[str(path.relative_to(PROJECT))] = path
    portable, portable_scoring, portable_text, mapping = [], [], [], {}
    for row_index, (item, s, r) in enumerate(zip(exports, sidecars, text)):
        assert s["id"] == r["id"] and s["row_index"] == row_index
        data, score, record = copy.deepcopy(item), copy.deepcopy(s), copy.deepcopy(r)
        new_images = []
        for image in item["images"]:
            path = Path(image)
            rel = "review/training20/images/" + "/".join(path.parts[-3:])
            members[rel] = path
            mapping[image] = rel
            new_images.append(rel)
        data["images"] = new_images
        for f, rel in zip(score["frames"], new_images):
            f["original_absolute_path"] = f["path"]; f["path"] = rel
        for f, rel in zip(record["frames"], new_images):
            f["original_absolute_path"] = f["path"]; f["path"] = rel
        assert [f["path"] for f in score["frames"]] == data["images"]
        portable.append(data); portable_scoring.append(score); portable_text.append(record)
    review = out / "review_bundle"
    atomic_write_jsonl(review / "training20.portable.jsonl", portable)
    atomic_write_jsonl(review / "training20.portable.scoring.jsonl", portable_scoring)
    atomic_write_jsonl(review / "training20.portable.text.jsonl", portable_text)
    atomic_write_json(review / "image_path_mapping.json", mapping)
    dataset_member = str(out.relative_to(PROJECT))
    atomic_write_text(review / "README.md", f"# HD-EPIC source/export review bundle\n\nPending human reason-label review. This package operation does not train or infer; smoke status is recorded separately under runs/hdepic_sft_smoke_8b.\n\nComplete corrected exports: {dataset_member}/text and swift. Full exports reference the preserved server v1 frame cache; the full cache is not embedded. Actual Python implementations, wrappers, dependencies and tests are under code and scripts.\n\nResolve portable sample images relative to the extracted ZIP root: review/training20/images/... . The portable model rows, aligned scoring sidecars and rich text records are in review/training20. image_path_mapping.json maps original paths to bundled JPEGs. These 320 JPEGs are individual model-input images. Contact sheets are presentation aids; readable previews are at {dataset_member}/previews/examples.html.\n\nEach task × goal-present/absent category contains five initial-SFT rows. System prompt: {dataset_member}/system_prompt.txt. Changed questions and exclusions are recorded in the dataset audits and CSV/JSONL sidecars. Human reviewer fields remain blank.\n\nDataset-only server rerun: bash /data/yaxuanli/ActGround/scripts/prepare_hdepic_sft_v2.sh . Immutable v1/v2 records and cached frames are required. The separate smoke launcher requires a confirmed GPU allocation; it permits ten disposable steps only. Full SFT and GRPO are outside this task.\n")
    for path in review.iterdir():
        members["review/training20/" + path.name] = path
    # Include every real file with its digest; no source placeholders or hashes
    # substituting for implementation files. The manifest is stored once.
    manifest = {"status": "pending_human_reason_label_review", "full_cache_embedded": False, "training_sample_records": 20, "individual_sample_jpegs": len(mapping), "files": {name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for name, p in sorted(members.items())}}
    atomic_write_json(out / "review_bundle_manifest.json", manifest)
    members["review/bundle_manifest.json"] = out / "review_bundle_manifest.json"
    dest = out.parent / f"{out.name}_review.zip"
    temp = dest.with_name("." + dest.name + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in (".jpg", ".zip") else zipfile.ZIP_DEFLATED)
    temp.replace(dest)
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        assert all(name in archive.namelist() for name in members)
        bundled = [json.loads(l) for l in archive.read("review/training20/training20.portable.jsonl").decode().splitlines()]
        for item in bundled:
            assert len(item["images"]) == 16 and all(p in members for p in item["images"])
    atomic_write_json(out / "review_package_audit.json", {"passed": True, "zip": str(dest), "zip_bytes": dest.stat().st_size, "zip_sha256": sha256_file(dest), "members": len(members), "full_text_and_swift_exports_included": True, "python_implementations_and_tests_included": True, "portable_training_rows": 20, "individual_jpegs": len(mapping), "zip_crc_verified": True, "status": "pending_human_reason_label_review"})
    print(f"Review ZIP verified: {dest} ({dest.stat().st_size / 1024**2:.2f} MiB), {len(members)} real files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    package(p.parse_args().output.resolve())
