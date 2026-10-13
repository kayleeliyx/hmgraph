#!/usr/bin/env python3
"""Snapshot completed outputs and run the unchanged standalone CPU evaluator.

Source scripts/hdepic_data_storage.sh first. Never starts model inference.
The default refuses incomplete generation. --completed-intersection explicitly
evaluates only IDs completed for every requested model, with the subset recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from hdepic_eval.workflow import run, unique_rows
from hdepic_eval.semantics import validate_judgment

SOURCES = {
    "base": ROOT / "runs/hdepic_qwen38_27b_baseline_v1/predictions/generation.jsonl",
    "epoch1": ROOT / "runs/hdepic_sft_qwen38_27b_v1/predictions/epoch1/generation.jsonl",
    "epoch2": ROOT / "runs/hdepic_sft_qwen38_27b_v1/predictions/epoch2/generation.jsonl",
}
REFERENCE = ROOT / "evaluation/assets/references.jsonl"
PREVIOUS_JUDGMENTS = HERE.parent / "semantic_review/judgments.jsonl"


def read_snapshot(path, allow_incomplete_tail=False):
    """Read once so appended records cannot produce mixed provenance hashes."""
    if not path.exists():
        return [], {"path": str(path), "exists": False, "complete_lines": 0}
    content = path.read_bytes()
    consumed = content
    if content and not content.endswith(b"\n"):
        if not allow_incomplete_tail:
            raise ValueError(f"Unterminated JSONL record: {path}")
        consumed = content[: content.rfind(b"\n") + 1]
    values = [json.loads(line) for line in consumed.splitlines() if line.strip()]
    unique_rows(values)
    return values, {
        "path": str(path), "exists": True,
        "read_sha256": hashlib.sha256(content).hexdigest(),
        "consumed_sha256": hashlib.sha256(consumed).hexdigest(),
        "read_bytes": len(content), "consumed_bytes": len(consumed),
        "ignored_in_progress_tail_bytes": len(content) - len(consumed),
        "complete_lines": len(values),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, data):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--completed-intersection", action="store_true")
    parser.add_argument("--name", required=True, help="Owned child folder, e.g. base_only or full_comparison")
    parser.add_argument("--judgments", type=Path, nargs="*", default=[])
    parser.add_argument("--require-complete-semantics", action="store_true")
    args = parser.parse_args()
    if len(args.models) != len(set(args.models)):
        parser.error("Repeated model")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.name):
        parser.error("--name must be a simple lowercase folder name")

    references, reference_audit = read_snapshot(REFERENCE)
    references = [r for r in references if r["task"] == "explain"]
    expected = {r["id"] for r in references}
    assert len(expected) == len(references) == 228, "Frozen full Explain cohort changed"
    by_model, source_audit, counts = {}, {}, {}
    for model in args.models:
        values, source_audit[model] = read_snapshot(SOURCES[model], allow_incomplete_tail=True)
        observed = {r["id"] for r in values}
        if observed - expected:
            raise ValueError(f"{model} contains an unexpected ID")
        status_counts = Counter(r.get("status", "completed") for r in values)
        unexpected_statuses = set(status_counts) - {"completed", "failed", "error", "pending", "running", "in_progress"}
        if unexpected_statuses:
            raise ValueError(f"{model} contains unrecognized generation status: {sorted(unexpected_statuses)}")
        completed = {r["id"]: r for r in values if r.get("status", "completed") == "completed"}
        by_model[model] = completed
        failures = sum(status_counts[state] for state in ("failed", "error"))
        in_progress = sum(status_counts[state] for state in ("pending", "running", "in_progress"))
        counts[model] = {"completed": len(completed), "expected": len(expected), "failed": failures,
                         "in_progress": in_progress, "missing_prediction_records": len(expected - observed),
                         "pending_generation": len(expected - observed) + in_progress,
                         "incomplete_total": len(expected - set(completed)),
                         "recorded_status_counts": dict(status_counts)}
    shared = set.intersection(*(set(values) for values in by_model.values()))
    if not args.completed_intersection and shared != expected:
        raise SystemExit("Generation incomplete; no full-cohort scores written. " + json.dumps(counts))
    if not shared:
        raise SystemExit("No shared completed examples; no scores written.")
    references = [r for r in references if r["id"] in shared]
    ids = [r["id"] for r in references]

    baseline_values = by_model.get("base")
    if baseline_values is None:
        baseline_rows, _ = read_snapshot(SOURCES["base"])
        baseline_values = unique_rows(baseline_rows)
    paired_input_fields = ("prompt_sha256", "preprocessing", "generation_config", "model_repo", "model_revision")
    for model, values in by_model.items():
        for identity in ids:
            for key in paired_input_fields:
                if key not in values[identity] or values[identity][key] != baseline_values[identity][key]:
                    raise ValueError(f"{model}: {key} differs from baseline for {identity}")

    # A reused decision must be hash-bound and schema-valid. Packet matching is
    # performed again by the canonical evaluator against the current answer.
    judgments = {}
    judgment_sources = []
    for path in [PREVIOUS_JUDGMENTS, *args.judgments]:
        path = path.resolve()
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        unique_rows(values, "packet_id")
        for row in values:
            validate_judgment(row["judgment"])
            if not row.get("prompt_sha256") or not row.get("judge"):
                raise ValueError(f"Unbound judgment in {path}")
            previous = judgments.get(row["packet_id"])
            if previous and previous != row:
                raise ValueError(f"Conflicting judgment for {row['packet_id']}")
            judgments[row["packet_id"]] = row
        judgment_sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "records": len(values)})

    dest = HERE / args.name
    if not dest.resolve().is_relative_to(HERE):
        raise ValueError("Output escaped analysis folder")
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "references.jsonl", references)
    for model, values in by_model.items():
        write_rows(dest / f"{model}.jsonl", [values[i] for i in ids])
    write_rows(dest / "judgments.jsonl", sorted(judgments.values(), key=lambda r: r["packet_id"]))
    config = {
        "references": "references.jsonl", "reference_format": "prepared",
        "mapper": str(ROOT / "evaluation/assets/mapper.json"), "output": "scores",
        "models": [{"name": m, "tasks": ["explain"], "predictions": f"{m}.jsonl"} for m in args.models],
        "comparisons": [[args.models[0], m] for m in args.models[1:]],
    }
    write_json(dest / "config.json", config)
    cohort = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full_228_Explain_validation" if shared == expected else "explicit_completed_intersection_only",
        "full_generation_complete_for_requested_models": shared == expected,
        "expected_examples": len(expected), "evaluated_examples": len(ids),
        "selected_query_ids": ids, "source_completion": counts,
        "source_snapshots": source_audit, "reference_source": reference_audit,
        "judgment_sources": judgment_sources,
        "paired_inputs_verified_against_base": {"fields": list(paired_input_fields), "examples_per_model": len(ids)},
        "no_pending_generation_scored_as_incorrect": True,
        "scope_warning": "P07 Explain only; no Track/current-action or test generation accuracy. Semantic ratings are provisional assistant assessments.",
    }
    write_json(dest / "cohort.json", cohort)
    summary = run(dest / "config.json", judgments_path=dest / "judgments.jsonl")
    packets = [json.loads(line) for line in (dest / "scores/semantic_packets.jsonl").read_text().splitlines() if line.strip()]
    for field in ("why", "goal"):
        write_rows(dest / f"{field}_packets.jsonl", [p for p in packets if p["field"] == field])
    status = {"folder": str(dest), "source_completion": counts, "evaluated_examples_per_model": len(ids),
              "pending_semantic_packets": len(packets), "pending_by_field": dict(Counter(p["field"] for p in packets)),
              "judgments_available": len(judgments), "config": str(dest / "config.json")}
    write_json(dest / "status.json", status)
    print(json.dumps(status, indent=2))
    if args.require_complete_semantics and packets:
        raise SystemExit("Semantic review remains incomplete; numerical semantic rates are not final.")


if __name__ == "__main__":
    main()
