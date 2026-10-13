"""Reprocess cached W and rebuild a separate L1 baseline without model calls.

Inputs: --run-dir with w_outputs.jsonl and optional alias/baseline L1 files.
Outputs: --output-dir (must not exist), derived W, L1, support context, ledger,
before/after diagnostics, and a manifest. Original files are never overwritten.
Uses only the standard library; no GPU, frame extraction, or GT access.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from code.l1.from_w import GROUPING_VERSION, build_l1_from_w
from code.w.prompt import ACTION_GATE_VERSION, PARSER_VERSION
from code.w.records import reprocess_record


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(value):
    return value.get("name", "") if isinstance(value, dict) else str(value)


def _validate_windows(records):
    seen = set()
    for row in records:
        if row["window_idx"] in seen:
            raise ValueError(f"Duplicate W window_idx: {row['window_idx']}")
        seen.add(row["window_idx"])
        start, end = float(row["t_start"]), float(row["t_end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid W target interval in window {row['window_idx']}")


def reprocess_run(
    run_dir: Path,
    output_dir: Path,
    baseline_l1: Path | None = None,
    *,
    w_jsonl: Path | None = None,
    l1_filename: str = "l1_candidates_from_w_v2.jsonl",
) -> dict:
    """Create a new output directory; fail before writing if it already exists.

    Optional input/output filenames let the project-root build_l1_wgroup.py
    retain its --w-jsonl/--out-jsonl interface while using this same pipeline.
    The module CLI retains its existing defaults and output filenames.
    """
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new path: {output_dir}")
    reserved_outputs = {"w_reprocessed.jsonl", "w_support_observations.jsonl",
                        "w_to_l1_ledger.jsonl", "w_processing_changes.jsonl",
                        "reprocess_summary.json", "reprocess_manifest.json"}
    if (not l1_filename or Path(l1_filename).name != l1_filename
            or Path(l1_filename).suffix != ".jsonl" or l1_filename in reserved_outputs):
        raise ValueError("L1 filename must be a .jsonl basename distinct from the diagnostic filenames")
    source = w_jsonl.resolve() if w_jsonl is not None else run_dir / "w_outputs.jsonl"
    alias_path = run_dir / "alias_table.json"
    inputs = [source] + ([alias_path] if alias_path.exists() else [])
    if baseline_l1 is None:
        candidate = run_dir / "l1_candidates_wgroup_qwen.jsonl"
        baseline_l1 = candidate if candidate.exists() else None
    if baseline_l1 is not None:
        baseline_l1 = baseline_l1.resolve()
        inputs.append(baseline_l1)
    hashes = {str(path): _hash(path) for path in inputs}
    original = _read(source)
    if not original:
        raise ValueError(f"No W records in input: {source}")
    _validate_windows(original)
    for row in original:
        if not isinstance(row.get("raw_output"), str):
            raise ValueError(f"W window {row['window_idx']} needs cached raw_output text for reprocessing")
    aliases = json.loads(alias_path.read_text()) if alias_path.exists() else {}
    repaired = [reprocess_record(row) for row in original]
    nodes, supports, ledger = build_l1_from_w(repaired, aliases)
    old_nodes = _read(baseline_l1) if baseline_l1 else []
    old_actions = {(row["window_idx"], a["action_idx"]): a for row in original for a in row.get("valid_actions", [])}
    new_actions = {(row["window_idx"], a["action_idx"]): a for row in repaired for a in row["valid_actions"]}
    refs = [(ref["window_idx"], ref["action_idx"]) for node in nodes for ref in node["source_action_refs"]]
    expected = {key for key, a in new_actions.items() if a["interaction"]["role"] in {"primary_manipulation", "preparatory"}}
    if set(refs) != expected or len(refs) != len(set(refs)):
        raise AssertionError("Each accepted primary/preparatory action must have exactly one L1 owner")
    old_counterparts = {(int(w), n.get("verb"), _name(n.get("direct_object")), n.get("hand"))
                        for n in old_nodes for w in n.get("source_window_idxs", [])}
    changes = []
    for key in sorted(set(old_actions) | set(new_actions)):
        before, after = old_actions.get(key), new_actions.get(key)
        change = "newly_accepted" if before is None else "newly_rejected" if after is None else "retained"
        if change != "retained" or any(before.get(k) != after.get(k) for k in ("event_t_start", "event_t_end")):
            changes.append({"window_idx": key[0], "action_idx": key[1], "change": change, "before": before, "after": after})
    summary = {
        "w_windows": len(original), "model_calls": 0, "frame_extractions": 0,
        "accepted_w_actions_before": len(old_actions), "accepted_w_actions_after": len(new_actions),
        "newly_accepted_actions": len(set(new_actions) - set(old_actions)),
        "newly_rejected_actions": len(set(old_actions) - set(new_actions)),
        "nonpositive_accepted_w_durations_before": sum(a["event_t_end"] <= a["event_t_start"] for a in old_actions.values()),
        "nonpositive_accepted_w_durations_after": sum(a["event_t_end"] <= a["event_t_start"] for a in new_actions.values()),
        "timing_status_after": dict(Counter(a["timing_status"] for a in new_actions.values())),
        "json_failure_windows_after": [r["window_idx"] for r in repaired if not r["parse_diagnostics"]["json_parse_ok"]],
        "schema_warning_windows_after": [r["window_idx"] for r in repaired if r["parse_diagnostics"]["json_parse_ok"] and not r["parse_diagnostics"]["schema_ok"]],
        "baseline_l1_nodes": len(old_nodes) if baseline_l1 else None,
        "new_l1_candidates": len(nodes), "support_observations": len(supports),
        "baseline_primary_actions_without_exact_counterpart": sum(
            a.get("interaction", {}).get("role") == "primary_manipulation" and
            (w, a["verb"], _name(a["direct_object"]), a.get("hand")) not in old_counterparts for (w, _), a in old_actions.items()) if baseline_l1 else None,
        "new_primary_preparatory_actions_without_exact_source_ref": len(expected - set(refs)),
        "primary_preparatory_source_refs": len(refs),
        "interpretation": "Structural preservation diagnostics, not recognition accuracy. New acceptances and candidate merges require visual review.",
    }
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
                "processing_code_sha256": {name: _hash(Path(__file__).parent / name) for name in
                                           ("reprocess_w_l1.py", "w/prompt.py", "w/records.py", "l1/from_w.py", "utils/alias_table.py")},
                "parser_version": PARSER_VERSION, "action_gate_version": ACTION_GATE_VERSION,
                "grouping_version": GROUPING_VERSION, "model_inference": False,
                "prompt_unchanged": True, "ground_truth_used_in_rebuild": False,
                "l1_output_filename": l1_filename,
                "baseline_note": "Uses the shared W-grouping v2 implementation. Prior L1 files are read only for diagnostic comparisons, never to construct predictions."}
    for path in inputs:
        if _hash(path) != hashes[str(path)]:
            raise RuntimeError(f"Input changed during reprocessing: {path}")
    output_dir.mkdir(parents=True)
    _write(output_dir / "w_reprocessed.jsonl", repaired)
    _write(output_dir / l1_filename, nodes)
    _write(output_dir / "w_support_observations.jsonl", supports)
    _write(output_dir / "w_to_l1_ledger.jsonl", ledger)
    _write(output_dir / "w_processing_changes.jsonl", changes)
    for filename, data in (("reprocess_summary.json", summary), ("reprocess_manifest.json", manifest)):
        (output_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-l1", type=Path)
    args = parser.parse_args()
    print(json.dumps(reprocess_run(args.run_dir, args.output_dir, args.baseline_l1), indent=2))


if __name__ == "__main__":
    main()
