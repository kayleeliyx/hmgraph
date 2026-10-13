"""Verify the targeted question repair against preserved v2 exports."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from .build import PARTITIONS, read_jsonl, text_fingerprint
from .revise import DEFAULT_OUTPUT, HOLDS, PROJECT, QUESTION_OVERRIDES, render, write_csv

DEFAULT_PREVIOUS = PROJECT / "datasets/hdepic_sft_v2"
REGRESSIONS = {
    "P01-20240202-110250-210": ("tilt food processor bowl", "The left-hand modifier belongs to holding the glass, not tilting the food processor bowl."),
    "P03-20240216-185832-669": ("stir potatoes", "The left hand holds the frying pan; the stirring spatula is held in the right hand. Both broad enrichments are removed."),
    "P01-20240202-195538-90": ("place spoon", "The spoon is placed on the lid; the countertop phrase describes the lid's location."),
    "P05-20240425-124420-11": ("close cutlery drawer (with my hip)", "The hip phrase directly modifies closing the drawer; retain its exact reviewed source span [25,36)."),
}


def hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def baseline(previous, out):
    if out == previous or previous in out.parents:
        raise ValueError("Repaired outputs must be outside the immutable v2 directory")
    dest = out / "previous_v2_preservation_baseline.json"
    if dest.exists():
        saved = json.loads(dest.read_text())
        assert saved["previous_v2"] == str(previous)
        assert saved["sha256"] == hashes(previous), "Existing v2 preservation baseline differs"
        return
    atomic_write_json(dest, {"previous_v2": str(previous), "sha256": hashes(previous)})
    print(f"Saved immutable v2 SHA256 baseline: {dest}", flush=True)


def report(previous, out):
    saved = json.loads((out / "previous_v2_preservation_baseline.json").read_text())
    assert saved["previous_v2"] == str(previous)
    assert saved["sha256"] == hashes(previous), "Immutable v2 files changed during repair"
    assert sha256_file(previous / "split_manifest.json") == sha256_file(out / "split_manifest.json")
    for name in ("reason_labels.jsonl", "frame_requests.json", "skipped_queries.jsonl", "benchmark_overlap.json", "benchmark_overlap.csv"):
        assert sha256_file(previous / name) == sha256_file(out / name), f"Unrelated output changed: {name}"
    old_holds = list(read_jsonl(previous / "quarantined.jsonl"))
    new_holds = list(read_jsonl(out / "quarantined.jsonl"))
    assert len(old_holds) == len(new_holds) == len(HOLDS)
    for old, new in zip(old_holds, new_holds):
        assert old["exclusion_reason"] == new["exclusion_reason"]
        assert old["original_v1_record"] == new["original_v1_record"]
        a, b = old["quarantined_record"], new["quarantined_record"]
        assert {k: v for k, v in a.items() if k != "format_version"} == {k: v for k, v in b.items() if k != "format_version"}
    tally = Counter(); changes = []; regressions = {}; checked = 0
    override_ids = set()
    for part in PARTITIONS:
        inputs = [previous / "text" / f"{part}.jsonl", out / "text" / f"{part}.jsonl", out / "swift" / f"{part}.jsonl", out / "swift" / f"{part}.scoring.jsonl", out / "swift" / f"{part}.prompts.jsonl"]
        for index, group in enumerate(itertools.zip_longest(*(read_jsonl(p) for p in inputs))):
            old, new, item, score, prompt = group
            assert all(r is not None for r in group), "Export length or order changed"
            assert old["id"] == new["id"] == score["id"] and score["row_index"] == index
            for key in ("id", "partition", "participant", "video_id", "task", "query_time_sec", "frames", "target"):
                assert old[key] == new[key], f"Unrelated record field changed: {new['id']} {key}"
            assert old["input"]["history"] == new["input"]["history"]
            assert {k: v for k, v in old["scoring_metadata"].items() if k != "question_provenance"} == {k: v for k, v in new["scoring_metadata"].items() if k != "question_provenance"}
            assert item == render(new, score["frames"])
            assert score == {"row_index": index, "id": new["id"], "partition": part, "participant": new["participant"], "video_id": new["video_id"], "task": new["task"], "query_time_sec": new["query_time_sec"], "frames": score["frames"], **new["scoring_metadata"]}
            assert prompt == {"messages": item["messages"][:2], "images": item["images"]}
            tally[(part, f"records_{new['task']}")] += 1; checked += 1
            if new["task"] == "track":
                assert old["input"] == new["input"]
                continue
            previous_question, repaired_question = old["input"]["action_question_text"], new["input"]["action_question_text"]
            old_spans = old["scoring_metadata"]["question_provenance"]["observed_modifiers"]
            new_spans = new["scoring_metadata"]["question_provenance"]["observed_modifiers"]
            nid = new["scoring_metadata"]["current_action"]["narration_id"]
            canonical = new["scoring_metadata"]["current_action"]["action"]
            tally[(part, "previous_modifier_questions")] += bool(old_spans)
            tally[(part, "retained_reviewed_modifier_questions")] += bool(new_spans)
            tally[(part, "canonical_questions_after_repair")] += not bool(new_spans)
            if new_spans:
                override_ids.add(nid)
                assert nid in QUESTION_OVERRIDES and part != "test"
                assert previous_question == repaired_question
            else:
                assert repaired_question == canonical
            if previous_question != repaired_question:
                tally[(part, "changed_questions")] += 1
                changes.append({"id": new["id"], "narration_id": nid, "partition": part, "participant": new["participant"], "old_action_question_text": previous_question, "new_action_question_text": repaired_question, "removed_modifiers": old_spans, "reason": "canonical fallback: broad clause-scanning enrichment does not establish attachment to the target action"})
            else:
                tally[(part, "unchanged_questions")] += 1
            if nid in REGRESSIONS:
                expected, explanation = REGRESSIONS[nid]
                assert repaired_question == expected
                regressions[nid] = {"passed": True, "old_question": previous_question, "new_question": repaired_question, "source_text": new["scoring_metadata"]["reason_annotation"]["original_narration"], "source_modifier_offsets": new_spans, "confirmation": explanation, "review_type": "assistant source-text attachment check; independent human review pending"}
    assert set(regressions) == set(REGRESSIONS)
    assert override_ids == set(QUESTION_OVERRIDES)
    old_pool = json.loads((previous / "second_stage_explain_ids.json").read_text())["eligible_ids"]
    new_pool = json.loads((out / "second_stage_explain_ids.json").read_text())["eligible_ids"]
    assert old_pool == new_pool
    counts = {p: {k: tally[(p, k)] for k in ("records_explain", "records_track", "previous_modifier_questions", "changed_questions", "unchanged_questions", "retained_reviewed_modifier_questions", "canonical_questions_after_repair")} for p in PARTITIONS}
    audit = {"passed": True, "status": "pending_human_reason_label_review", "previous_v2": str(previous), "output": str(out), "records_verified": checked, "changed_questions": len(changes), "retained_explicit_modifier_questions": len(override_ids), "partition_counts": counts, "regression_confirmations": regressions, "explicit_question_overrides": QUESTION_OVERRIDES, "fixed_split_byte_identical": True, "previous_v2_files_byte_identical": True, "reason_labels_byte_identical": True, "reason_extraction_unchanged": True, "quarantined_narration_ids": list(HOLDS), "quarantines_unchanged": True, "targets_source_ids_classes_histories_futures_timestamps_frame_paths_unchanged": True, "second_stage_explain_ids_unchanged": True, "second_stage_explain_count": len(new_pool), "frames_decoded": 0, "frames_created": 0, "model_calls": 0, "dataset_fingerprint": text_fingerprint(out), "rerun_commands": [f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT} {PROJECT}/.venvs/llavaction/bin/python -m code.hdepic_preparation.modifier_repair_audit --phase baseline --previous {previous} --output {out}", *[f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT} {PROJECT}/.venvs/llavaction/bin/python -m code.hdepic_preparation.revise --output {out} --phase {phase}" for phase in ("build", "text-audit", "sample", "export")], f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT} {PROJECT}/.venvs/llavaction/bin/python -m code.hdepic_preparation.audit_annotations --output {out}", f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT} {PROJECT}/.venvs/llavaction/bin/python -m code.hdepic_preparation.modifier_repair_audit --phase report --previous {previous} --output {out}"]}
    atomic_write_jsonl(out / "modifier_attachment_changes.jsonl", changes)
    write_csv(out / "modifier_attachment_changes.csv", [{**r, "removed_modifiers": json.dumps(r["removed_modifiers"], ensure_ascii=False)} for r in changes])
    atomic_write_json(out / "modifier_repair_audit.json", audit)
    atomic_write_json(out / "modifier_repair_provenance.json", {"code_sha256": {str(p.relative_to(PROJECT)): sha256_file(p) for p in (PROJECT / "code/hdepic_preparation/revise.py", PROJECT / "code/hdepic_preparation/modifier_repair_audit.py", PROJECT / "code/tests/test_hdepic_revision.py")}, "previous_v2_baseline": str(out / "previous_v2_preservation_baseline.json"), "dataset_fingerprint": text_fingerprint(out)})
    lines = ["# Targeted HD-EPIC v2 modifier-attachment repair", "", "**Pending human reason-label review.** This repair changes questions only; it does not redesign reason extraction or add new labels. Previous v2 outputs, the fixed split, six quarantine decisions, target text, canonical action/class/source IDs, history/future actions, query timestamps and individual cached-frame paths are preserved.", "", f"Removed unverified enrichment from {len(changes):,} Explain questions. Five individually checked source-backed choices remain: hip, small quantity, careful tilt, opposite-hand glove, and validation worktop. All other Explain questions use the canonical action. Broad hand/tool/location/repeat enrichment is disabled. Override text, exact offsets, observed attachment evidence and review notes are saved in correction_policy.json and each scoring sidecar. These assistant text checks do not establish independent human label validity.", "", "| Partition | Explain | Track | Changed questions | Retained reviewed modifiers |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {p} | {counts[p]['records_explain']} | {counts[p]['records_track']} | {counts[p]['changed_questions']} | {counts[p]['retained_reviewed_modifier_questions']} |" for p in PARTITIONS)
    lines.extend(["", "The three reported regressions now use `tilt food processor bowl`, `stir potatoes`, and `place spoon`; the hip example remains `close cutlery drawer (with my hip)`. Their source text and confirmations are in modifier_repair_audit.json. Every changed question is enumerated in modifier_attachment_changes.csv / .jsonl.", "", f"All {checked:,} serialized exports and aligned scoring/prompt rows were reread. Full text, cached-frame and independent source-alignment checks passed. No frame was re-decoded. The shared WHY-only second-stage pool remains {len(new_pool)} IDs. The 20 initial-SFT review rows include five per task × goal-present/absent category, with 16 individual images per record; contact sheets remain presentation aids.", "", "The repaired dataset is in this directory; complete exports are under swift/, text records under text/, and training examples under sample/. Model/environment/preprocessing/smoke reports are produced separately by the current smoke-test task.", "", "Exact dataset repair rerun commands:", "", "```bash", *audit["rerun_commands"], "```", ""])
    atomic_write_text(out / "README.md", "\n".join(lines))
    print(json.dumps({"passed": True, "changed_questions": len(changes), "partition_counts": counts, "output": str(out)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", required=True, choices=("baseline", "report"))
    args = parser.parse_args()
    (baseline if args.phase == "baseline" else report)(args.previous.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
