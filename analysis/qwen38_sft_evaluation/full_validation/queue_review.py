#!/usr/bin/env python3
"""Queue immutable WHY/GOAL review batches from completed model generations.

Source scripts/hdepic_data_storage.sh first. Reads only the approved base/batch
review locations; independent QA ratings are never imported as benchmark labels.
No GPU use. --collect-only prepares the same verified ratings for final scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from hdepic_eval.semantics import validate_judgment

FIELDS = ("why", "goal")
BATCHES = HERE / "review_batches"
COLLECTED = HERE / "completed_review_judgments.jsonl"
COLLECTION_AUDIT = HERE / "review_collection.json"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def parse_rows(content, path):
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = row["packet_id"]
        if key in indexed:
            raise ValueError(f"Duplicate packet_id {key} in {path}")
        indexed[key] = row
    return indexed


def batch_directories():
    if not BATCHES.exists():
        return []
    return sorted((p for p in BATCHES.iterdir() if p.is_dir() and re.fullmatch(r"batch\d+", p.name)),
                  key=lambda p: int(p.name[5:]))


def review_sources():
    return [HERE / "base_only", *batch_directories()]


def inspect_review(folder, field):
    packets_path = folder / f"{field}_packets.jsonl"
    judgments_path = folder / f"{field}_judgments.jsonl"
    if not packets_path.exists():
        raise ValueError(f"Missing assigned packet file: {packets_path}")
    packet_bytes = packets_path.read_bytes()
    packets = parse_rows(packet_bytes, packets_path)
    for packet in packets.values():
        if packet["field"] != field or sha(packet["prompt"].encode()) != packet["prompt_sha256"]:
            raise ValueError(f"Invalid assigned packet field/hash in {packets_path}")
    audit = {"packets": str(packets_path), "packets_sha256": sha(packet_bytes),
             "judgments": str(judgments_path), "expected_packets": len(packets)}
    if not packets:
        return packets, {}, {**audit, "state": "empty_no_review_needed"}
    if not judgments_path.exists():
        return packets, {}, {**audit, "state": "pending_file", "missing_packets": len(packets)}
    content = judgments_path.read_bytes()
    audit.update(judgments_sha256=sha(content), judgment_bytes=len(content))
    if content and not content.endswith(b"\n"):
        return packets, {}, {**audit, "state": "pending_unterminated_file"}
    try:
        judgments = parse_rows(content, judgments_path)
    except json.JSONDecodeError:
        return packets, {}, {**audit, "state": "pending_invalid_json"}
    extras = set(judgments) - set(packets)
    if extras:
        raise ValueError(f"Unexpected judgment packet IDs in {judgments_path}: {sorted(extras)}")
    for key, row in judgments.items():
        if row["prompt_sha256"] != packets[key]["prompt_sha256"]:
            raise ValueError(f"Judgment prompt hash mismatch in {judgments_path}: {key}")
        validate_judgment(row["judgment"], field)
        if not isinstance(row.get("judge"), str) or not row["judge"].strip():
            raise ValueError(f"Missing judge identity in {judgments_path}: {key}")
    missing = set(packets) - set(judgments)
    if missing:
        return packets, {}, {**audit, "state": "pending_incomplete_coverage", "present_packets": len(judgments),
                             "missing_packets": len(missing)}
    return packets, judgments, {**audit, "state": "complete_exact_id_and_hash_coverage", "accepted_judgments": len(judgments)}


def write_rows(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def collect_reviews():
    accepted, assigned, audits = {}, {}, []
    previous_sources = json.loads(COLLECTION_AUDIT.read_text()).get("sources", []) if COLLECTION_AUDIT.exists() else []
    assigned_hashes = {entry["packets"]: entry["packets_sha256"] for entry in previous_sources}
    original_path = HERE.parent / "semantic_review/judgments.jsonl"
    original_bytes = original_path.read_bytes()
    original = parse_rows(original_bytes, original_path)
    for folder in review_sources():
        if not folder.resolve().is_relative_to(HERE):
            raise ValueError(f"Review folder escaped full_validation: {folder}")
        for field in FIELDS:
            packets, judgments, audit = inspect_review(folder, field)
            previous_hash = assigned_hashes.get(audit["packets"])
            if previous_hash and previous_hash != audit["packets_sha256"]:
                raise ValueError(f"Previously assigned packet file changed: {audit['packets']}")
            sidecar_path = folder / "source_status.json"
            if sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text())
                expected_hash = sidecar.get("packet_file_sha256", {}).get(field)
                if expected_hash and expected_hash != audit["packets_sha256"]:
                    raise ValueError(f"Immutable batch packet hash changed: {audit['packets']}")
            audits.append(audit)
            for key, packet in packets.items():
                if key in assigned and assigned[key]["prompt_sha256"] != packet["prompt_sha256"]:
                    raise ValueError(f"Assigned packet hash conflict: {key}")
                assigned[key] = packet
            for key, row in judgments.items():
                previous = accepted.get(key, original.get(key))
                if previous and previous != row:
                    raise ValueError(f"Conflicting accepted judgment: {key}")
                accepted[key] = row
    write_rows(COLLECTED, [accepted[key] for key in sorted(accepted)])
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "original_judgments_automatically_loaded_by_prepare": {"path": str(original_path), "sha256": sha(original_bytes), "count": len(original)},
             "additional_completed_judgments": len(accepted), "assigned_packet_count": len(assigned),
             "merged_additional_file": str(COLLECTED), "merged_additional_sha256": sha(COLLECTED.read_bytes()),
             "sources": audits, "source_states": dict(Counter(a["state"] for a in audits)),
             "independent_qa_imported": False}
    write_json(COLLECTION_AUDIT, audit)
    return assigned, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("epoch1", "epoch2"))
    parser.add_argument("--collect-only", action="store_true", help="Collect complete verified review files, without scoring or creating a batch")
    parser.add_argument("--max-per-field", type=int, default=40, help="At most this many packets for each field in one new batch (default 40)")
    args = parser.parse_args()
    if not args.collect_only and not args.model:
        parser.error("--model is required unless --collect-only")
    if args.max_per_field < 1:
        parser.error("--max-per-field must be positive")
    if not HERE.resolve().is_relative_to(Path("/data")):
        raise SystemExit("Refusing writes outside /data")

    # Serializes mutable collection/in_progress snapshots and next batch number.
    # Reviewers can continue writing their separately owned judgment files.
    with (HERE / ".queue_review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assigned, collection = collect_reviews()
        if args.collect_only:
            print(json.dumps({"completed_judgments": str(COLLECTED), "additional_judgment_count": collection["additional_completed_judgments"],
                              "source_states": collection["source_states"], "independent_qa_imported": False}, indent=2))
            return
        command = [sys.executable, str(HERE / "prepare_and_score.py"), "--models", args.model,
                   "--completed-intersection", "--name", "in_progress", "--judgments", str(COLLECTED)]
        result = subprocess.run(command, check=False, text=True, capture_output=True, cwd=ROOT)
        if result.returncode:
            if "No shared completed examples" in result.stderr or "No shared completed examples" in result.stdout:
                print(json.dumps({"model": args.model, "batch": None, "reason": "no_completed_generations", "additional_judgments": collection["additional_completed_judgments"]}))
                return
            raise SystemExit(f"Snapshot scoring failed ({result.returncode}):\n{result.stdout}{result.stderr}")
        progress = HERE / "in_progress"
        status = json.loads((progress / "status.json").read_text())
        selected, pending_counts, already_assigned = {}, {}, {}
        for field in FIELDS:
            pending = parse_rows((progress / f"{field}_packets.jsonl").read_bytes(), progress / f"{field}_packets.jsonl")
            pending_counts[field] = len(pending)
            already_assigned[field] = sum(key in assigned for key in pending)
            for key, packet in pending.items():
                if key in assigned and assigned[key]["prompt_sha256"] != packet["prompt_sha256"]:
                    raise ValueError(f"Pending packet conflicts with an assigned packet: {key}")
            selected[field] = [pending[key] for key in sorted(pending) if key not in assigned][:args.max_per_field]
        counts = {field: len(values) for field, values in selected.items()}
        if not any(counts.values()):
            print(json.dumps({"model": args.model, "batch": None, "reason": "no_unassigned_pending_packets",
                              "source_completion": status["source_completion"], "pending_packets": pending_counts,
                              "already_assigned_pending": already_assigned, "additional_judgments": collection["additional_completed_judgments"]}, indent=2))
            return

        BATCHES.mkdir(exist_ok=True)
        existing_numbers = [int(p.name[5:]) for p in batch_directories()]
        batch_name = f"batch{max(existing_numbers, default=0) + 1:03d}"
        dest = BATCHES / batch_name
        staging = BATCHES / f".{batch_name}.creating.{os.getpid()}"
        staging.mkdir(exist_ok=False)
        for field, values in selected.items():
            write_rows(staging / f"{field}_packets.jsonl", values)
        sidecar = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
                   "batch": batch_name, "source_status": status,
                   "snapshot_cohort_sha256": sha((progress / "cohort.json").read_bytes()),
                   "review_collection": collection, "packet_counts": counts,
                   "packet_file_sha256": {field: sha((staging / f"{field}_packets.jsonl").read_bytes()) for field in FIELDS},
                   "pending_before_assignment": pending_counts, "already_assigned_pending": already_assigned,
                   "max_per_field": args.max_per_field, "immutable_packet_assignment": True,
                   "assistant_review_status": "provisional; identical frozen rubric and packets omit model names; not a fully blinded study"}
        write_json(staging / "source_status.json", sidecar)
        if dest.exists():
            raise ValueError(f"Refusing to overwrite existing batch: {dest}")
        staging.rename(dest)
        print(json.dumps({"model": args.model, "batch": str(dest), "packet_counts": counts,
                          "completed_generation_examples": status["evaluated_examples_per_model"],
                          "additional_judgments_collected": collection["additional_completed_judgments"],
                          "already_assigned_pending": already_assigned,
                          "unassigned_remaining_after_batch": {field: pending_counts[field] - already_assigned[field] - counts[field] for field in FIELDS}}, indent=2))


if __name__ == "__main__":
    main()
