"""Conservative v2 correction pass over immutable v1 records and cached JPEGs.

No model calls, recall expansion, frame decoding, split changes, or training.
Questions default to the canonical action. Modifier attachment is accepted only
for the explicit source-backed review overrides below; a modifier elsewhere in
the narration's binding clause is insufficient. There are no test overrides.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import io
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, stable_hash
from code.e0.frames import video_fingerprint
from .build import PARTITIONS, read_jsonl, stratified, text_fingerprint, write_review
from .export import SYSTEM, audit_record as audit_v1, load_cache, previews

VERSION = "hdepic_causal_sft_v2_modifier_attachment_repair"
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT / "datasets/hdepic_sft_v1"
DEFAULT_OUTPUT = PROJECT / "datasets/hdepic_sft_v2_repaired"

# Explicit holds are training/validation review flags, never test-label tuning.
# Suggested trims are not confirmed replacements: keep the original span and
# quarantine rather than assert a newly edited ground truth.
HOLDS = {
    "P04-20240415-181212-279": "source_conflict: narration says lid off but purpose says close; no correction invented",
    "P01-20240204-121042-232": "reason_scope_uncertain: pouring dates and support-hand explanation share an overly broad span",
    "P02-20240211-120927-670": "temporal_scope_uncertain: apparent tray rotation precedes annotated placement; cached lookback does not establish the reason for this interval",
    "P05-20240427-143353-65": "span_cleanup_pending_review: suggested removal of right-hand manner clause is not a confirmed replacement",
    "P06-20240510-100047-3": "span_cleanup_pending_review: suggested removal of simultaneous holding clause is not a confirmed replacement",
    "P07-20240531-101357-535": "span_cleanup_pending_review: balance and subsequent wiping share the extracted span; suggested trim requires review",
}

# Text-reviewed action choices previously flagged in the training review. The
# worktop validation example checks the same attachment issue; it is explicitly
# recorded rather than inferred by scanning every binding clause. These are
# assistant text checks, not completed independent human annotation review.
# Exact evidence and offsets make each override auditable and fail closed if its
# source or canonical action changes. No generic hand/tool/location enrichment.
QUESTION_OVERRIDES = {
    "P05-20240425-124420-11": {
        "canonical_action": "close cutlery drawer", "rule": "body_part_choice",
        "text": "with my hip", "char_start": 25, "char_end": 36,
        "attachment_evidence": "Close the cutlery drawer with my hip ",
        "evidence_char_start": 0,
        "review_note": "Hip directly modifies closing the named drawer; retain the relevant body-part choice without the full-hands reason.",
    },
    "P02-20240209-194141-455": {
        "canonical_action": "pour squash", "rule": "quantity_choice",
        "text": "only a small amount", "char_start": 44, "char_end": 63,
        "attachment_evidence": "Pour squash into the glass making sure that only a small amount is used ",
        "evidence_char_start": 0,
        "review_note": "The small amount refers directly to the named pouring action; strength remains exclusively in the reason target.",
    },
    "P01-20240204-142301-40": {
        "canonical_action": "tilt bowl", "rule": "careful_manner",
        "text": "only carefully", "char_start": 66, "char_end": 80,
        "attachment_evidence": "Tilt the bowl on top of the sink to drop the water in it, tilting only carefully ",
        "evidence_char_start": 0,
        "review_note": "The repeated tilting clause specifies the manner of the same bowl tilt; neither the embedded purpose nor the reason is copied.",
    },
    "P01-20240204-152537-15": {
        "canonical_action": "wear glove", "rule": "opposite_glove_choice",
        "text": "left hand glove in the right hand", "char_start": 117, "char_end": 150,
        "attachment_evidence": "I'm wearing what should be the left hand glove in the right hand ",
        "evidence_char_start": 86,
        "review_note": "The first-person wearing clause directly specifies the glove and hand choice; the wet-glove explanation stays in the target.",
    },
    "P07-20240529-191007-380": {
        "canonical_action": "put colander", "rule": "placement_destination",
        "text": "on the worktop instead", "char_start": 17, "char_end": 39,
        "attachment_evidence": "put the colander on the worktop instead ",
        "evidence_char_start": 0,
        "review_note": "The worktop prepositional phrase directly modifies putting the colander; it does not describe a supporting object's location.",
    },
}


def question_text(r):
    """Canonical by default; exact reviewed overrides establish attachment."""
    canonical = r["scoring_metadata"]["current_action"]["action"]
    a = r["scoring_metadata"]["reason_annotation"]
    narration_id = a.get("narration_id", r["scoring_metadata"]["current_action"].get("narration_id"))
    override = QUESTION_OVERRIDES.get(narration_id)
    spans = []
    if override is not None:
        assert not str(narration_id).startswith(("P08-", "P09-")), "No test-specific modifier overrides"
        assert r.get("partition") != "test", "No test-specific modifier overrides"
        assert canonical == override["canonical_action"], "Reviewed override action no longer matches"
        narration = a["original_narration"]
        evidence_start = override["evidence_char_start"]
        evidence_end = evidence_start + len(override["attachment_evidence"])
        assert narration[evidence_start:evidence_end] == override["attachment_evidence"], "Reviewed attachment evidence no longer matches"
        assert evidence_start <= override["char_start"] < override["char_end"] <= evidence_end <= a["char_start"]
        assert narration[override["char_start"]:override["char_end"]] == override["text"], "Reviewed modifier offsets no longer match"
        spans.append({**override, "override_narration_id": narration_id, "attachment_check": "explicit_source_text_review", "human_review_status": "pending"})
    text = canonical + (" (" + "; ".join(s["text"] for s in spans) + ")" if spans else "")
    return text, {"rule": "canonical_with_explicit_reviewed_override_v2_repair" if spans else "canonical_action_no_unverified_modifier_v2_repair", "canonical_action": canonical, "observed_modifiers": spans, "action_question_text": text, "broad_modifier_enrichment": "disabled", "override_narration_id": narration_id if spans else None, "human_review_status": "pending"}


def render(r, frames):
    lines = [f"Task: {r['task'].title()}", f"Query time: {r['query_time_sec']:.6f} s", "Frames in chronological order:"]
    lines.extend(f"Frame {i+1:02d}, source time {f['timestamp_sec']:.6f} s: <image>" for i, f in enumerate(frames))
    lines.append("Previous completed actions (oldest to newest):")
    lines.extend(f"[{a['start_sec']:.3f}, {a['end_sec']:.3f}] {a['action']}" for a in r["input"]["history"])
    if not r["input"]["history"]:
        lines.append("(none)")
    if r["task"] == "explain":
        lines.append(f"Action to explain: {r['input']['action_question_text']}")
        lines.append("Based on the supplied observations, explain the likely immediate reason for this action, including any specified choice or manner, in <why>.")
        answer = f"<why>{html.escape(r['target']['reason'], quote=False)}</why>"
        primary = "<why>"
    else:
        lines.append("Based on the supplied observations, identify the ongoing action in <action>.")
        answer = f"<action>{html.escape(r['target']['current_action'], quote=False)}</action>"
        primary = "<action>"
    fields = [primary]
    if "goal" in r["target"]:
        lines.append("Identify the associated activity in <goal>.")
        answer += f"\n<goal>{html.escape(r['target']['goal'], quote=False)}</goal>"
        fields.append("<goal>")
    lines.append("Forecast exactly three next actions in temporal order in <next>.")
    fields.append("<next>")
    lines.append("Return only these fields, in this order: " + ", ".join(fields) + ".")
    answer += "\n<next>" + "; ".join(html.escape(a, quote=False) for a in r["target"]["next_actions"]) + "</next>"
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}, {"role": "assistant", "content": answer}], "images": [f["path"] for f in frames]}


def audit_record(r, frames, partition_of, check_paths=True):
    # Reuse the working v1 timing/history/class/span checks, with its canonical
    # input schema, then check the v2 question and renderer independently.
    compatible = {**r, "input": dict(r["input"])}
    if r["task"] == "explain":
        assert set(r["input"]) == {"history", "action_question_text"}
        compatible["input"] = {"history": r["input"]["history"], "action_to_explain": r["scoring_metadata"]["current_action"]["action"]}
        text, provenance = question_text(r)
        assert text == r["input"]["action_question_text"]
        assert provenance == r["scoring_metadata"]["question_provenance"]
        a = r["scoring_metadata"]["reason_annotation"]
        for s in provenance["observed_modifiers"]:
            assert a["original_narration"][s["char_start"]:s["char_end"]] == s["text"]
            assert s["char_end"] <= a["char_start"]
            assert not re.search(r"\b(?:because|so|in order to|to \w+|for \w+ing)\b", s["text"], re.I)
    else:
        assert set(r["input"]) == {"history"}
    audit_v1(compatible, frames, partition_of, check_paths)
    item = render(r, frames)
    user, answer = (item["messages"][i]["content"] for i in (1, 2))
    assert user.count("<image>") == len(item["images"]) == 16
    assert ("<goal>" in user) == ("<goal>" in answer) == ("goal" in r["target"])
    assert "when supervised" not in user and "stated reason" not in user
    assert [m["role"] for m in item["messages"]] == ["system", "user", "assistant"]
    assert set(item) == {"messages", "images"}
    return item


def write_csv(path, rows, fields=None):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def revise(source, out):
    if source == out or source in out.parents:
        raise ValueError("v2 must be outside the immutable v1 output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("split_manifest.json", "video_inventory.json", "benchmark_overlap.json", "benchmark_overlap.csv", "source_annotation_issues.json", "skipped_queries.jsonl"):
        src, dst = source / name, out / name
        if name == "split_manifest.json" and dst.exists() and sha256_file(dst) != sha256_file(src):
            raise ValueError("Existing v2 fixed manifest differs from v1")
        if not dst.exists() or sha256_file(dst) != sha256_file(src):
            shutil.copyfile(src, dst)
    old_counts = json.loads((source / "counts.json").read_text())
    for p, digest in old_counts["source_hashes"].items():
        assert sha256_file(Path(p)) == digest, f"Source annotation changed: {p}"
    records, excluded, changes, used, requested = {}, [], [], set(), defaultdict(set)
    tally = Counter()
    old_dim = old_counts["dimensions"]
    # Carry forward the original grid skips and reason rejection counts.
    for x in old_dim:
        if x["kind"] not in ("accepted", "goal", "reason_rule"):
            tally[tuple(x[k] for k in ("partition", "participant", "task", "kind", "category"))] += x["count"]
    reason_updates = {}
    for part in PARTITIONS:
        rows = []
        for original in read_jsonl(source / "text" / f"{part}.jsonl"):
            r = copy.deepcopy(original)
            nid = r["scoring_metadata"]["current_action"]["narration_id"]
            change = {"id": r["id"], "narration_id": nid, "partition": part, "participant": r["participant"], "task": r["task"], "status": "changed", "changes": ["explicit_requested_fields_and_observation_based_prompt_v2"], "old_action_question_text": original["input"].get("action_to_explain"), "new_action_question_text": None, "reason": "request goal only when a target annotation exists; no narrator-access wording"}
            r["format_version"] = VERSION
            if r["task"] == "explain":
                a = r["scoring_metadata"]["reason_annotation"]
                span = {"text": a["reason_text"], "char_start": a["char_start"], "char_end": a["char_end"]}
                a["v1_span"] = dict(span)
                a["revised_span"] = dict(span)
                a["revision_reason"] = "span unchanged; no unconfirmed trim promoted to ground truth"
                if nid in HOLDS:
                    assert part != "test", "No test-specific hold is permitted"
                    a.update(status="quarantined_pending_human_review", revised_span=None, revision_reason=HOLDS[nid])
                    r["scoring_metadata"]["review_status"] = "quarantined_pending_human_review"
                    reason_updates[nid] = a
                    excluded.append({"exclusion_reason": HOLDS[nid], "original_v1_record": original, "quarantined_record": r})
                    change.update(status="excluded", changes=["quarantine_pending_source_or_reason_scope_review"], reason=HOLDS[nid])
                    changes.append(change)
                    tally[(part, r["participant"], "explain", "quarantined", HOLDS[nid].split(":")[0])] += 1
                    continue
                text, provenance = question_text(r)
                r["input"].pop("action_to_explain")
                r["input"]["action_question_text"] = text
                r["scoring_metadata"]["question_provenance"] = provenance
                change["new_action_question_text"] = text
                if provenance["observed_modifiers"]:
                    change["changes"].append("preserve_observed_modifier_with_exact_source_offsets")
                reason_updates[nid] = a
                used.add(nid)
                tally[(part, r["participant"], "explain", "reason_rule", a["extraction_rule"])] += 1
            rows.append(r); changes.append(change)
            tally[(part, r["participant"], r["task"], "accepted", "total")] += 1
            tally[(part, r["participant"], r["task"], "goal", r["scoring_metadata"]["goal_status"])] += 1
            requested[r["video_id"]].update(f["source_frame_index"] for f in r["frames"])
        records[part] = rows
        atomic_write_jsonl(out / "text" / f"{part}.jsonl", rows)
    reasons = [reason_updates.get(a["narration_id"], a) for a in read_jsonl(source / "reason_labels.jsonl")]
    assert len(excluded) == len(HOLDS) and len({c["id"] for c in changes}) == len(changes)
    atomic_write_jsonl(out / "reason_labels.jsonl", reasons)
    atomic_write_jsonl(out / "quarantined.jsonl", excluded)
    atomic_write_jsonl(out / "changes.jsonl", changes)
    write_csv(out / "changes.csv", [{**c, "changes": "; ".join(c["changes"])} for c in changes])
    write_csv(out / "quarantine_review.csv", [{"narration_id": e["quarantined_record"]["scoring_metadata"]["current_action"]["narration_id"], "partition": e["quarantined_record"]["partition"], "reason_for_hold": e["exclusion_reason"], "original_narration": e["quarantined_record"]["scoring_metadata"]["reason_annotation"]["original_narration"], "original_span": e["quarantined_record"]["scoring_metadata"]["reason_annotation"]["reason_text"], "valid purpose?": "", "correct target action?": "", "reviewer notes": ""} for e in excluded])
    write_review(out, reasons, used)
    question_of = {r["scoring_metadata"]["current_action"]["narration_id"]: r["input"]["action_question_text"] for rows in records.values() for r in rows if r["task"] == "explain"}
    review = list(csv.DictReader((out / "reason_review.csv").open()))
    for row in review:
        row["action_question_text"] = question_of.get(row["narration_id"], "")
    write_csv(out / "reason_review.csv", review)
    counts = {**old_counts, "protocol": VERSION, "partition_records": {p: len(rows) for p, rows in records.items()}, "explain_records": len(used), "quarantined_explain_records": len(excluded), "v1_accepted_candidates_before_protocol_filter": old_counts["reason_labels_accepted_before_protocol_filter"], "reason_labels_accepted_before_protocol_filter": sum(a["status"] == "accepted_pending_human_review" for a in reasons), "dimensions": [{"partition": k[0], "participant": k[1], "task": k[2], "kind": k[3], "category": k[4], "count": n} for k, n in sorted(tally.items())]}
    # Reason-label dimension records the v1 extraction decision; quarantine is a
    # separate final eligibility gate, not a change to the extraction rules.
    counts["reason_label_dimension_stage"] = "v1_extraction_before_v2_quarantine"
    atomic_write_json(out / "counts.json", counts)
    write_csv(out / "counts.csv", counts["dimensions"])
    atomic_write_json(out / "frame_requests.json", {vid: sorted(indices) for vid, indices in sorted(requested.items())})
    atomic_write_text(out / "system_prompt.txt", SYSTEM + "\n")
    atomic_write_json(out / "correction_policy.json", {"version": VERSION, "rules_frozen_before_test_export": True, "rule_design_source": "explicit source-backed assistant text checks of previously flagged training action choices", "validation_use": "one individually checked worktop attachment; no generic validation-derived rules", "test_specific_overrides": 0, "rejected_reason_recovery": False, "unconfirmed_trims": "quarantine", "holds": HOLDS, "default_question": "canonical_action", "broad_modifier_enrichment": "disabled", "explicit_question_overrides": QUESTION_OVERRIDES, "source_v1": str(source), "source_split_sha256": sha256_file(source / "split_manifest.json"), "source_text_fingerprint": text_fingerprint(source)})
    print(json.dumps({"version": VERSION, "counts": counts["partition_records"], "Explain": len(used), "quarantined": len(excluded)}, indent=2), flush=True)


def training_sample(out):
    rows = list(read_jsonl(out / "text/initial_sft.jsonl"))
    chosen = []
    for task in ("explain", "track"):
        for has_goal in (True, False):
            candidates = [r for r in rows if r["task"] == task and ("goal" in r["target"]) == has_goal]
            for r in candidates:
                r["narration_id"] = r["id"]  # sampling helper uses this only
            selected = stratified(candidates, 5, lambda r: (r["participant"],))
            assert len(selected) == 5
            for r in selected:
                r.pop("narration_id")
            chosen.extend(selected)
    assert len(chosen) == len({r["id"] for r in chosen}) == 20
    return chosen


def export(out, sample_only=False):
    manifest = json.loads((out / "split_manifest.json").read_text())
    partition_of = {v: p for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]}
    inventory = json.loads((out / "video_inventory.json").read_text())
    source_frames = Path(json.loads((out / "correction_policy.json").read_text())["source_v1"]) / "frames"
    frame_settings = json.loads((out / "counts.json").read_text())["frame_settings"]
    selected = training_sample(out)
    sample_ids = {r["id"] for r in selected}
    groups = [("training20", selected)] if sample_only else [(p, list(read_jsonl(out / "text" / f"{p}.jsonl"))) for p in PARTITIONS]
    cache, unique_frames, ids, tally, sampled = {}, {}, set(), Counter(), {}
    for group, rows in groups:
        swift, scoring, rich, prompts, why, why_prompts, why_scoring = [], [], [], [], [], [], []
        for r in rows:
            assert r["id"] not in ids; ids.add(r["id"])
            for f in r["frames"]:
                assert Path(f["path"]).is_relative_to(source_frames)
            root = str(Path(r["frames"][0]["path"]).parent)
            if root not in cache:
                entry = json.loads((Path(root) / "record.json").read_text())
                v = inventory[r["video_id"]]
                assert entry["cache_id"] == v["cache_id"]
                assert entry["preprocessing"] == frame_settings
                assert entry["source_video_fingerprint"] == v["fingerprint"] == video_fingerprint(Path(v["path"]))
            frames = load_cache(out, r, cache)
            for f in frames:
                assert f["source_pts"] == f["source_frame_index"] * 512
                assert f["source_time_base"] == "1/15360" and f["timestamp_sec"] == f["source_frame_index"] / 30
            unique_frames.update({f["path"]: f for f in frames})
            item = audit_record(r, frames, partition_of)
            s = {"row_index": len(swift), "id": r["id"], "partition": r["partition"], "participant": r["participant"], "video_id": r["video_id"], "task": r["task"], "query_time_sec": r["query_time_sec"], "frames": frames, **r["scoring_metadata"]}
            swift.append(item); scoring.append(s)
            prompt = {"messages": item["messages"][:2], "images": item["images"]}
            prompts.append(prompt)
            rich.append({"id": r["id"], "partition": r["partition"], "task": r["task"], "query_time_sec": r["query_time_sec"], **item})
            if r["task"] == "explain":
                why_scoring.append({**s, "row_index": len(why), "full_partition_row_index": s["row_index"]})
                why.append(item); why_prompts.append(prompt)
            tally[f"{r['partition']}:{r['task']}"] += 1
            if r["id"] in sample_ids:
                sampled[r["id"]] = (r, item, frames, s)
        dest = out / ("sample" if sample_only else "swift")
        for suffix, data in [("", swift), (".scoring", scoring), (".prompts", prompts), (".records", rich)]:
            atomic_write_jsonl(dest / f"{group}{suffix}.jsonl", data)
        if group == "second_stage":
            for suffix, data in [("", why), (".scoring", why_scoring), (".prompts", why_prompts)]:
                atomic_write_jsonl(dest / f"second_stage.explain{suffix}.jsonl", data)
            atomic_write_json(out / "second_stage_explain_ids.json", {"status": "pending_human_reason_label_review", "eligible_ids": [r["id"] for r in why_scoring], "shared_by": ["extra_sft", "grpo"], "v1_count": 147, "v2_count": len(why), "policy": "Both WHY-only methods use precisely these same IDs; Track replay requires a later matched design."})
        print(f"Verified/exported {group}: {len(swift)} records; reused cached JPEGs", flush=True)
    byte_count = sum(f["bytes"] for f in unique_frames.values())
    audit = {"passed": True, "status": "pending_human_reason_label_review", "records_verified": len(ids), "counts": dict(tally), "dataset_fingerprint": text_fingerprint(out), "individual_frame_references": 16 * len(ids), "unique_frames_verified": len(unique_frames), "frame_cache_bytes": byte_count, "frames_created": 0, "frames_decoded": 0, "model_calls": 0, "training_started": False, "checks": ["16 chronological distinct individual JPEGs in causal five-second lookback", "actual cached decoder PTS and time base, source fingerprint, bytes and SHA256", "exact completed history excludes target and narrated methods/purposes/activity labels", "strict future starts within video", "canonical class/source IDs unchanged", "question modifiers are exact observed pre-reason source spans", "explicit requested tags match target tags including missing-goal omission", "Track input has no current ground-truth action name", "scoring sidecars aligned one-to-one with model rows"], "semantic_label_correctness": "not established; human review pending"}
    atomic_write_json(out / ("sample_audit.json" if sample_only else "timestamp_leakage_audit.json"), audit)
    ordered = [sampled[r["id"]] for r in selected]
    if sample_only:
        previews(out, [x[0] for x in ordered], [x[1] for x in ordered], [x[2] for x in ordered])
        # The reused preview writer's legacy title said ten; only change v2.
        path = out / "previews/examples.html"
        atomic_write_text(path, path.read_text().replace("HD-EPIC: 10 examples", "HD-EPIC: 20 training examples"))
        atomic_write_jsonl(out / "sample/training20.text.jsonl", [x[0] for x in ordered])
    print(json.dumps(audit, indent=2), flush=True)


def audit_text(out):
    manifest = json.loads((out / "split_manifest.json").read_text())
    partition_of = {v: p for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]}
    n = 0
    for p in PARTITIONS:
        for r in read_jsonl(out / "text" / f"{p}.jsonl"):
            frames = [{**f, "timestamp_sec": f["planned_timestamp_sec"]} for f in r["frames"]]
            audit_record(r, frames, partition_of, check_paths=False); n += 1
    atomic_write_json(out / "text_audit.json", {"passed": True, "records_verified": n, "dataset_fingerprint": text_fingerprint(out), "status": "pending_actual_cached_frame_audit_and_human_reason_review"})
    print(f"All {n} corrected text records passed checks before cached-frame export", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("build", "text-audit", "sample", "export"), required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if args.phase == "build":
        revise(args.source.resolve(), out)
    elif args.phase == "text-audit":
        audit_text(out)
    else:
        export(out, sample_only=args.phase == "sample")


if __name__ == "__main__":
    main()
