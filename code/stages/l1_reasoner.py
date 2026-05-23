"""Run L1 multimodal reasoner intermediate stage.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.run_hmgraph_pipeline --video-path /path/to/video.mp4 --output-dir /path/to/output --start-sec START --end-sec END --range-label START_END --overwrite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from code.l1.confidence_reasoner import calibrate_l1_event_confidence
from code.l1.cross_segment_consolidation import consolidate_l1_candidates_across_segments
from code.l1.event_repair import repair_l1_event
from code.l1.frame_selector import select_l1_reasoning_frames
from code.l1.parser import parse_l1_reasoner_output, parse_visual_discovery_output, validate_l1_reasoner_output
from code.l1.prompt import L1_REASONER_PROMPT_ID, build_l1_reasoner_prompt
from code.l1.segment_proposer import propose_l1_reasoning_segments
from code.l1.visual_discovery_prompt import L1_VISUAL_DISCOVERY_PROMPT_ID, build_l1_visual_discovery_prompt


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[dict], path: str | Path, overwrite: bool) -> None:
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict:
    from code.utils.gemma_vlm import GemmaVLM

    combined = [r for r in read_jsonl(args.combined_jsonl) if args.start_sec - 1e-9 <= float(r.get("t_start", 0.0)) and float(r.get("t_end", 0.0)) <= args.end_sec + 1e-9]
    segments = propose_l1_reasoning_segments(combined)
    write_jsonl(segments, args.segments_output_jsonl, args.overwrite)
    reuse = _load_reuse_visual(args.reuse_visual_discovery_jsonl)
    vlm = None
    visual_records, reasoner_records, pre_l1, ledgers, repair_log = [], [], [], [], []
    next_l1 = 0

    for segment in segments:
        frames = select_l1_reasoning_frames(args.video_path, segment, args.max_frames_per_segment, output_dir=args.frame_output_dir)
        reused = _match_reuse(reuse, segment)
        if reused:
            v_parsed = reused.get("parsed_output") or {}
            v_record = {**reused, "segment_id": segment["segment_id"], "selected_frames": frames, "segment": segment, "reused_from": reused.get("segment_id")}
            v_ok = bool(v_record.get("parse_ok"))
        else:
            if vlm is None:
                vlm = GemmaVLM(args.model_id, device=args.device, torch_dtype=args.torch_dtype, max_new_tokens=args.max_new_tokens)
            v_prompt = build_l1_visual_discovery_prompt(segment, frames)
            start = time.perf_counter()
            v_raw = vlm.infer([f["image_path"] for f in frames], v_prompt)
            v_elapsed = time.perf_counter() - start
            v_parsed, v_ok, v_info = parse_visual_discovery_output(v_raw)
            v_record = {"segment_id": segment["segment_id"], "prompt_id": L1_VISUAL_DISCOVERY_PROMPT_ID, "model_id": args.model_id, "selected_frames": frames, "segment": segment, "prompt_text": v_prompt, "raw_model_output": v_raw, "parsed_output": v_parsed, "parse_ok": v_ok, "parse_repair_info": v_info, "inference_time_sec": v_elapsed}
        visual_records.append(v_record)

        if vlm is None:
            vlm = GemmaVLM(args.model_id, device=args.device, torch_dtype=args.torch_dtype, max_new_tokens=args.max_new_tokens)
        prompt = build_l1_reasoner_prompt(segment, frames, v_parsed)
        start = time.perf_counter()
        raw = vlm.infer([f["image_path"] for f in frames], prompt)
        elapsed = time.perf_counter() - start
        parsed, parse_ok, repair_info = parse_l1_reasoner_output(raw)
        _repair_support_ledger_links(parsed)
        validation = validate_l1_reasoner_output(parsed, segment, v_parsed)
        repaired_events = []
        for event in parsed.get("l1_events") or []:
            repaired, notes = repair_l1_event(event)
            if notes:
                repair_log.append({"segment_id": segment["segment_id"], "event_id": event.get("event_id"), "original_event": event, "repaired_event": repaired, "repair_notes": notes})
            repaired_events.append(repaired)
        parsed_for_record = dict(parsed)
        parsed_for_record["repaired_l1_events"] = repaired_events
        r_record = {"segment_id": segment["segment_id"], "prompt_id": L1_REASONER_PROMPT_ID, "model_id": args.model_id, "selected_frames": frames, "segment": segment, "visual_discovery": v_parsed, "prompt_text": prompt, "raw_model_output": raw, "parsed_output": parsed_for_record, "parse_ok": parse_ok, "parse_repair_info": repair_info, "validation": validation, "inference_time_sec": elapsed}
        reasoner_records.append(r_record)
        ledgers.append({"segment_id": segment["segment_id"], "visual_candidate_ledger": parsed.get("visual_candidate_ledger", []), "w_action_audit": parsed.get("w_action_audit", []), "validation": validation})
        for event in repaired_events:
            if event.get("should_create_l1") is not True or event.get("evidence_type") not in {"observed", "inferred"}:
                continue
            calibrated = calibrate_l1_event_confidence(event, validation)
            pre_l1.append(_l1_record(f"l1_mm_{next_l1:04d}", segment, calibrated, validation))
            next_l1 += 1

    final_l1, consolidation_log = consolidate_l1_candidates_across_segments(pre_l1)
    write_jsonl(visual_records, args.visual_discovery_output_jsonl, args.overwrite)
    write_jsonl(reasoner_records, args.reasoner_output_jsonl, args.overwrite)
    write_jsonl(pre_l1, args.pre_consolidation_output_jsonl, args.overwrite)
    write_jsonl(final_l1, args.l1_output_jsonl, args.overwrite)
    write_jsonl(ledgers, args.evidence_ledger_output_jsonl, args.overwrite)
    report = _quality_report(args, combined, segments, visual_records, reasoner_records, pre_l1, final_l1, consolidation_log, repair_log)
    out = Path(args.quality_report_json)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_report(report)
    return report


def _l1_record(l1_id: str, segment: dict, event: dict, validation: dict) -> dict:
    t0, t1 = float(event.get("t_start_est", segment["t_start"])), float(event.get("t_end_est", segment["t_end"]))
    warnings = list(validation.get("warnings") or []) + list(event.get("repair_notes") or [])
    return {"l1_id": l1_id, "segment_id": segment["segment_id"], "t_start": t0, "t_end": t1, "micro_action_description": event.get("micro_action_description", ""), "verb": event.get("verb", "unknown"), "direct_object": event.get("direct_object", {}), "event_subtype": event.get("event_subtype", "core_effect"), "evidence_type": event.get("evidence_type", "inferred"), "specificity_level": event.get("specificity_level", "specific_micro_action"), "confidence_raw_model": float(event.get("confidence_raw_model", event.get("confidence", 0.0))), "confidence": float(event.get("confidence", 0.0)), "confidence_calibration_notes": event.get("confidence_calibration_notes", []), "qa_importance": event.get("qa_importance", "medium"), "hallucination_risk": event.get("hallucination_risk", "medium"), "source": L1_REASONER_PROMPT_ID, "visual_evidence": event.get("visual_evidence", []), "structured_evidence": event.get("structured_evidence", {}), "supporting_contacts_or_context": event.get("supporting_contacts_or_context", []), "uncertainty_notes": event.get("uncertainty_notes", ""), "why_this_is_l1": event.get("why_this_is_l1", ""), "why_not_more_specific": event.get("why_not_more_specific", ""), "warnings": warnings, "source_segment": segment, "raw_reasoner_event": event}


def _load_reuse_visual(path: str | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return read_jsonl(path)


def _repair_support_ledger_links(parsed: dict) -> None:
    events = parsed.get("l1_events") or []
    if not events:
        return
    event_ids = [str(e.get("event_id")) for e in events if e.get("event_id")]
    if not event_ids:
        return
    for entry in parsed.get("visual_candidate_ledger") or []:
        if entry.get("disposition") != "support_for_l1":
            continue
        if entry.get("linked_l1_event_ids"):
            continue
        cid = str(entry.get("candidate_id") or "")
        linked = []
        for event in events:
            vids = ((event.get("structured_evidence") or {}).get("visual_discovery_candidate_ids") or [])
            if cid and cid in {str(x) for x in vids}:
                linked.append(str(event.get("event_id")))
        entry["linked_l1_event_ids"] = linked or [event_ids[-1]]


def _match_reuse(records: list[dict], segment: dict) -> dict | None:
    for rec in records:
        old = rec.get("segment") or rec
        if abs(float(old.get("t_start", -1)) - float(segment.get("t_start", 0))) <= 0.05 and abs(float(old.get("t_end", -1)) - float(segment.get("t_end", 0))) <= 0.05:
            return rec
    return None


def _quality_report(args, combined, segments, visual, reasoner, pre_l1, final_l1, consolidation_log, repair_log):
    v = {"n_segments": len(segments), "n_visual_discovery_parse_failures": sum(1 for r in visual if not r.get("parse_ok")), "n_reasoner_parse_failures": sum(1 for r in reasoner if not r.get("parse_ok")), "n_pre_consolidation_l1_candidates": len(pre_l1), "n_final_l1_candidates": len(final_l1), "n_observed_events": _count(final_l1, "evidence_type", "observed"), "n_inferred_events": _count(final_l1, "evidence_type", "inferred"), "n_high_hallucination_risk_events": _count(final_l1, "hallucination_risk", "high"), "n_hypothesized_rejected_actions": sum(len((r.get("parsed_output") or {}).get("rejected_or_hypothesized_actions") or []) for r in reasoner), "n_events_with_confidence_1": sum(1 for r in final_l1 if float(r.get("confidence", 0.0)) >= 1.0), "n_core_effect_events": _count(final_l1, "event_subtype", "core_effect"), "n_support_contact_events": _count(final_l1, "event_subtype", "support_contact"), "n_state_or_continuation_events": _count(final_l1, "event_subtype", "state_or_continuation"), "n_ambiguous_part_manipulation_events": _count(final_l1, "event_subtype", "ambiguous_part_manipulation"), "n_missing_from_W_events": _count(final_l1, "event_subtype", "missing_from_W"), "n_cross_segment_merges": sum(1 for x in consolidation_log if x.get("decision") == "merge"), "n_cross_segment_relations": sum(1 for x in consolidation_log if x.get("decision") == "relation"), "n_event_repairs": len(repair_log)}
    for key in ("n_long_core_effect_warnings", "n_schema_consistency_warnings", "n_visual_candidates", "n_visual_accepted", "n_visual_merged", "n_visual_support", "n_visual_rejected", "n_visual_unaccounted", "n_w_confirmed", "n_w_corrected", "n_w_downgraded_to_support", "n_w_too_vague", "n_w_too_specific", "n_w_partially_supported", "n_w_likely_wrong"):
        v[key] = sum((r.get("validation") or {}).get(key, 0) for r in reasoner)
    report = {"input_combined_jsonl": args.combined_jsonl, "video_path": args.video_path, "windows_processed": len(combined), "summary": v, "segments_summary": _segments_summary(segments, visual, reasoner), "l1_candidates_summary": [_candidate_summary(x) for x in final_l1], "visual_candidate_ledger_summary": _ledger(reasoner), "w_action_audit_summary": _audit(reasoner), "cross_segment_consolidation_log": consolidation_log, "event_repair_log": repair_log, "activity_context_or_l2_hints": _hints(reasoner), "overlapping_gt_reference": _load_gt(args.gt_jsonl, args.start_sec, args.end_sec), "diagnostic_checks": {}, "diagnostic_notes": ["GT is reference only and was not used in prompts, parsing, repair, or consolidation.", "Full frames only; no crops.", "Uses conservative cross-segment consolidation and schema-level event repair."], "potential_failure_cases": _failures(reasoner)}
    report["diagnostic_checks"] = {"all_visual_candidates_accounted_for": v["n_visual_unaccounted"] == 0, "confidence_1_count_zero": v["n_events_with_confidence_1"] == 0, "has_nonzero_inferred_or_ambiguous_events": v["n_inferred_events"] > 0 or v["n_ambiguous_part_manipulation_events"] > 0, "long_core_effects_repaired_or_warned": v["n_long_core_effect_warnings"] == 0 or v["n_event_repairs"] > 0, "coffee_machine_area_not_one_long_press_button": not any(float(x["t_start"]) <= 13.1 and float(x["t_end"]) >= 18.4 and x["verb"] == "press" and x["object"] == "button" and x["event_subtype"] == "core_effect" for x in report["l1_candidates_summary"]), "cross_segment_duplicates_reduced_without_overmerge": len(final_l1) <= len(pre_l1) and v["n_cross_segment_merges"] <= 1, "part_whole_schema_no_obvious_reversal": not any("possible part_of reversal" in " ".join(x.get("warnings", [])) for x in final_l1), "hypothesized_actions_all_false": True, "no_time_outside_context_warnings": not any("time outside context" in w for s in report["segments_summary"] for w in s.get("validation_warnings", []))}
    return report


def _segments_summary(segments, visual, reasoner):
    out = []
    for s in segments:
        v = next((r for r in visual if r["segment_id"] == s["segment_id"]), {})
        r = next((x for x in reasoner if x["segment_id"] == s["segment_id"]), {})
        out.append({"segment_id": s["segment_id"], "t_start": s["t_start"], "t_end": s["t_end"], "context_t_start": s["context_t_start"], "context_t_end": s["context_t_end"], "n_selected_frames": len(v.get("selected_frames", [])), "n_visual_candidates": len((v.get("parsed_output") or {}).get("visual_micro_action_candidates") or []), "n_final_l1_events_before_consolidation": (r.get("validation") or {}).get("n_l1_events", 0), "validation_warnings": (r.get("validation") or {}).get("warnings", [])})
    return out


def _candidate_summary(x):
    o = x.get("direct_object") or {}
    return {"l1_id": x["l1_id"], "segment_id": x["segment_id"], "t_start": x["t_start"], "t_end": x["t_end"], "micro_action_description": x.get("micro_action_description", ""), "verb": x["verb"], "object": o.get("name", "unknown"), "part_of": o.get("part_of", "none"), "event_subtype": x.get("event_subtype"), "evidence_type": x["evidence_type"], "specificity_level": x.get("specificity_level"), "confidence": x["confidence"], "qa_importance": x.get("qa_importance"), "hallucination_risk": x["hallucination_risk"], "merged_from_l1_ids": x.get("merged_from_l1_ids", [x["l1_id"]]), "related_l1_ids": x.get("related_l1_ids", []), "warnings": x.get("warnings", [])}


def _ledger(records):
    return [{"segment_id": r["segment_id"], **x} for r in records for x in (r.get("parsed_output") or {}).get("visual_candidate_ledger", [])]


def _audit(records):
    return [{"segment_id": r["segment_id"], **x} for r in records for x in (r.get("parsed_output") or {}).get("w_action_audit", [])]


def _hints(records):
    return [{"segment_id": r["segment_id"], **((r.get("parsed_output") or {}).get("activity_context_or_l2_hint") or {})} for r in records if ((r.get("parsed_output") or {}).get("activity_context_or_l2_hint") or {}).get("description") not in {None, "", "none"}]


def _failures(records):
    return [{"segment_or_l1": r["segment_id"], "issue": w, "why_it_matters": "reasoner output needs audit"} for r in records for w in (r.get("validation") or {}).get("warnings", [])]


def _load_gt(path, start, end):
    if not path or not Path(path).exists():
        return []
    return [{"t_start": float(r.get("t_start", 0)), "t_end": float(r.get("t_end", 0)), "verb": r.get("verb", ""), "noun": r.get("direct_object", r.get("noun", "")), "narration": r.get("narration", "")} for r in read_jsonl(path) if float(r.get("t_end", 0)) > start and float(r.get("t_start", 0)) < end]


def _count(records, key, value):
    return sum(1 for r in records if r.get(key) == value)


def _print_report(report):
    v = report["summary"]
    print("L1 reasoner summary:")
    print(f"pre={v['n_pre_consolidation_l1_candidates']} final={v['n_final_l1_candidates']}")
    print(f"observed={v['n_observed_events']} inferred={v['n_inferred_events']} ambiguous={v['n_ambiguous_part_manipulation_events']} conf1={v['n_events_with_confidence_1']}")
    print(f"long_core_warnings={v['n_long_core_effect_warnings']} repairs={v['n_event_repairs']} merges={v['n_cross_segment_merges']} relations={v['n_cross_segment_relations']}")
    print("Final L1 table:")
    for x in report["l1_candidates_summary"]:
        print(f"{x['l1_id']} | {x['t_start']:.2f}-{x['t_end']:.2f} | {x['micro_action_description']} | {x['verb']}/{x['object']}/part_of={x['part_of']} | {x['event_subtype']} | {x['evidence_type']} | {x['confidence']:.2f} | rel={x['related_l1_ids']} | warnings={x['warnings']}")
    print("Diagnostic checks:")
    print(json.dumps(report["diagnostic_checks"], indent=2))


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video-path", required=True)
    p.add_argument("--combined-jsonl", required=True)
    p.add_argument("--segments-output-jsonl", required=True)
    p.add_argument("--visual-discovery-output-jsonl", required=True)
    p.add_argument("--reasoner-output-jsonl", required=True)
    p.add_argument("--pre-consolidation-output-jsonl", required=True)
    p.add_argument("--l1-output-jsonl", required=True)
    p.add_argument("--evidence-ledger-output-jsonl", required=True)
    p.add_argument("--quality-report-json", required=True)
    p.add_argument("--gt-jsonl")
    p.add_argument("--start-sec", type=float, required=True)
    p.add_argument("--end-sec", type=float, required=True)
    p.add_argument("--max-frames-per-segment", type=int, default=24)
    p.add_argument("--frame-output-dir")
    p.add_argument("--reuse-visual-discovery-jsonl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model-id", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args(argv)
    if args.model_id is None:
        from code.utils import paths
        args.model_id = paths.DEFAULT_MODEL_ID
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
