"""Run final L1 postprocess.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.run_hmgraph_pipeline --video-path /path/to/video.mp4 --output-dir /path/to/output --start-sec START --end-sec END --range-label START_END --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from code.l1.confidence import recalibrate_l1_confidence
from code.l1.consolidation import consolidate_l1_candidates
from code.l1.ledger import normalize_visual_candidate_ledger
from code.l1.w_audit import normalize_w_audit
from code.l1.warning_cleanup import cleanup_l1_warnings


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
    raw_candidates = read_jsonl(args.raw_candidates_jsonl)
    _ = read_jsonl(args.raw_preconsolidation_jsonl)
    ledgers = read_jsonl(args.raw_evidence_ledger_jsonl)
    raw_report = json.loads(Path(args.raw_quality_report_json).read_text(encoding="utf-8"))

    working = [dict(c) for c in raw_candidates]
    post_log: list[dict] = []
    final_candidates, consolidation_log = consolidate_l1_candidates(working)
    post_log.extend({"stage": "consolidation", **x} for x in consolidation_log)

    flat_ledger = [dict(x, segment_id=rec.get("segment_id")) for rec in ledgers for x in rec.get("visual_candidate_ledger", [])]
    flat_audit = [dict(x, segment_id=rec.get("segment_id")) for rec in ledgers for x in rec.get("w_action_audit", [])]
    norm_audit, audit_log = normalize_w_audit(flat_audit, final_candidates)
    norm_ledger, ledger_log = normalize_visual_candidate_ledger(flat_ledger, final_candidates)
    post_log.extend(audit_log)
    post_log.extend(ledger_log)

    final_candidates = _remap_candidate_ids(final_candidates, consolidation_log)
    evidence_by_l1 = _evidence_context_by_l1(norm_ledger, norm_audit)
    recal_log, cleanup_log = [], []
    repaired = []
    for cand in final_candidates:
        recalibrated, notes = recalibrate_l1_confidence(cand, evidence_by_l1.get(cand.get("l1_id"), {}))
        recal_log.append({"stage": "confidence_recalibration", "l1_id": cand.get("l1_id"), "old_confidence": cand.get("confidence"), "new_confidence": recalibrated.get("confidence"), "notes": notes})
        cleaned, cnotes = cleanup_l1_warnings(recalibrated)
        if cnotes:
            cleanup_log.append({"stage": "warning_cleanup", "l1_id": cand.get("l1_id"), "notes": cnotes})
        repaired.append(cleaned)
    post_log.extend(recal_log)
    post_log.extend(cleanup_log)

    norm_ledger = _remap_links_in_entries(norm_ledger, final_candidates)
    norm_audit = _remap_links_in_entries(norm_audit, final_candidates)
    coverage_entries, coverage_log = _ensure_final_l1_ledger_coverage(final_candidates, norm_ledger, norm_audit)
    if coverage_entries:
        norm_ledger.extend(coverage_entries)
        post_log.extend(coverage_log)
    ledger_records = _group_ledger_records(ledgers, norm_ledger, norm_audit)

    write_jsonl(repaired, args.output_candidates_jsonl, args.overwrite)
    write_jsonl(ledger_records, args.output_evidence_ledger_jsonl, args.overwrite)
    write_jsonl(post_log, args.output_postprocess_log_jsonl, args.overwrite)
    report = _quality_report(args, raw_report, repaired, norm_ledger, norm_audit, consolidation_log, recal_log, cleanup_log)
    out = Path(args.output_quality_report_json)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_report(report)
    return report


def _remap_candidate_ids(candidates: list[dict], consolidation_log: list[dict]) -> list[dict]:
    for cand in candidates:
        cand["source"] = "l1_postprocess"
    return candidates


def _evidence_context_by_l1(ledger: list[dict], audit: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entry in ledger:
        for lid in entry.get("linked_l1_event_ids") or []:
            out.setdefault(lid, {"visual_dispositions": [], "w_audit_decisions": []})["visual_dispositions"].append(entry.get("disposition"))
    for entry in audit:
        for lid in entry.get("linked_l1_event_ids") or []:
            out.setdefault(lid, {"visual_dispositions": [], "w_audit_decisions": []})["w_audit_decisions"].append(entry.get("decision"))
    return out


def _remap_links_in_entries(entries: list[dict], candidates: list[dict]) -> list[dict]:
    valid = {c["l1_id"] for c in candidates}
    merged = {}
    for c in candidates:
        for mid in c.get("merged_from_l1_ids") or []:
            merged[str(mid)] = c["l1_id"]
        raw = (c.get("raw_reasoner_event") or {}).get("event_id")
        if raw:
            merged[str(raw)] = c["l1_id"]
    out = []
    for entry in entries:
        item = dict(entry)
        links = []
        for link in item.get("linked_l1_event_ids") or []:
            lid = merged.get(str(link), str(link))
            if lid in valid and lid not in links:
                links.append(lid)
        item["linked_l1_event_ids"] = links
        out.append(item)
    return out




def _ensure_final_l1_ledger_coverage(candidates: list[dict], ledger: list[dict], audit: list[dict]) -> tuple[list[dict], list[dict]]:
    linked = set()
    for entry in ledger + audit:
        linked.update(str(x) for x in (entry.get("linked_l1_event_ids") or []))
    additions, log = [], []
    for cand in candidates:
        lid = str(cand.get("l1_id"))
        if not lid or lid in linked:
            continue
        raw = cand.get("raw_reasoner_event") or {}
        structured = cand.get("structured_evidence") or {}
        visual_ids = structured.get("visual_discovery_candidate_ids") or raw.get("structured_evidence", {}).get("visual_discovery_candidate_ids") or []
        description = cand.get("micro_action_description") or raw.get("micro_action_description") or "unlinked final L1 event"
        entry = {
            "candidate_id": ",".join(str(x) for x in visual_ids) if visual_ids else str(raw.get("event_id") or lid),
            "candidate_description": description,
            "disposition": "accepted_as_l1",
            "linked_l1_event_ids": [lid],
            "reason": "Added during postprocess bookkeeping so the final evidence ledger explicitly covers this retained final L1. Source evidence remains in the final candidate and raw_reasoner_event.",
            "segment_id": cand.get("segment_id"),
            "source_reasoner_event_id": raw.get("event_id"),
            "source_visual_discovery_candidate_ids": visual_ids,
            "bookkeeping_repair": True,
        }
        additions.append(entry)
        log.append({
            "stage": "evidence_ledger_coverage",
            "l1_id": lid,
            "segment_id": cand.get("segment_id"),
            "source_reasoner_event_id": raw.get("event_id"),
            "reason": "final L1 had no explicit normalized visual/W ledger link; added provenance-preserving ledger coverage entry",
        })
    return additions, log


def _group_ledger_records(old_records: list[dict], ledger: list[dict], audit: list[dict]) -> list[dict]:
    segs = [r.get("segment_id") for r in old_records]
    return [
        {
            "segment_id": sid,
            "visual_candidate_ledger": [x for x in ledger if x.get("segment_id") == sid],
            "w_action_audit": [x for x in audit if x.get("segment_id") == sid],
        }
        for sid in segs
    ]


def _quality_report(args, raw_report, candidates, ledger, audit, consolidation_log, recal_log, cleanup_log):
    v = {
        "n_final_l1_candidates": len(candidates),
        "n_observed_events": _count(candidates, "evidence_type", "observed"),
        "n_inferred_events": _count(candidates, "evidence_type", "inferred"),
        "n_high_hallucination_risk_events": _count(candidates, "hallucination_risk", "high"),
        "n_events_with_confidence_1": sum(1 for c in candidates if float(c.get("confidence", 0.0)) >= 1.0),
        "n_core_effect_events": _count(candidates, "event_subtype", "core_effect"),
        "n_support_contact_events": _count(candidates, "event_subtype", "support_contact"),
        "n_state_or_continuation_events": _count(candidates, "event_subtype", "state_or_continuation"),
        "n_ambiguous_part_manipulation_events": _count(candidates, "event_subtype", "ambiguous_part_manipulation"),
        "n_visual_candidates": len(ledger),
        "n_visual_accepted": _count(ledger, "disposition", "accepted_as_l1"),
        "n_visual_merged": _count(ledger, "disposition", "merged_into_l1"),
        "n_visual_support": _count(ledger, "disposition", "support_for_l1"),
        "n_visual_rejected": sum(1 for x in ledger if str(x.get("disposition", "")).startswith("rejected")),
        "n_visual_unaccounted": sum(1 for x in ledger if not x.get("disposition")),
        "n_w_confirmed": _count(audit, "decision", "confirmed"),
        "n_w_corrected": _count(audit, "decision", "corrected"),
        "n_w_downgraded_to_support": _count(audit, "decision", "downgraded_to_support"),
        "n_w_too_vague": _count(audit, "decision", "too_vague"),
        "n_w_too_specific": _count(audit, "decision", "too_specific"),
        "n_w_partially_supported": _count(audit, "decision", "partially_supported"),
        "n_w_likely_wrong": _count(audit, "decision", "likely_wrong"),
        "n_w_not_applicable": _count(audit, "decision", "not_applicable"),
        "n_cross_segment_merges": sum(1 for x in consolidation_log if x.get("decision") == "merge"),
        "n_cross_segment_relations": sum(1 for x in consolidation_log if x.get("decision") == "relation"),
        "n_confidence_recalibrated": len(recal_log),
        "n_warning_cleanup": len(cleanup_log),
    }
    summaries = [_summary(c) for c in candidates]
    checks = {
        "confidence_not_overpenalized": min(float(c.get("confidence", 0.0)) for c in candidates) >= 0.45,
        "kept_events_have_nonzero_confidence": all(float(c.get("confidence", 0.0)) > 0 for c in candidates),
        "clear_core_events_have_reasonable_confidence": all(float(c.get("confidence", 0.0)) >= 0.65 for c in candidates if c.get("event_subtype") == "core_effect"),
        "ambiguous_events_have_medium_not_zero_confidence": all(0.45 <= float(c.get("confidence", 0.0)) <= 0.65 for c in candidates if c.get("event_subtype") == "ambiguous_part_manipulation"),
        "w_audit_not_applicable_used_only_for_none_or_irrelevant": all(x.get("decision") != "not_applicable" or not x.get("linked_l1_event_ids") or str(x.get("w_action", "")).lower() == "none" for x in audit),
        "visual_ledger_dispositions_interpretable": v["n_visual_accepted"] > 0 and v["n_visual_unaccounted"] == 0,
        "duplicate_mug_pickup_reduced": sum(1 for c in candidates if c.get("verb") == "pick up" and (c.get("direct_object") or {}).get("name") == "mug") <= 1,
        "coffee_machine_events_remain_separate": sum(1 for c in candidates if (c.get("direct_object") or {}).get("part_of") == "coffee machine") >= 2,
        "confidence_1_count_zero": v["n_events_with_confidence_1"] == 0,
        "part_whole_schema_no_obvious_reversal": not any("part_of reversal" in " ".join(c.get("warnings", [])).lower() for c in candidates),
    }
    return {
        "input_raw_candidates_jsonl": args.raw_candidates_jsonl,
        "windows_processed": raw_report.get("windows_processed"),
        "raw_reasoner_reference": {"n_final_l1_candidates": (raw_report.get("summary") or {}).get("n_final_l1_candidates"), "n_observed_events": (raw_report.get("summary") or {}).get("n_observed_events"), "n_inferred_events": (raw_report.get("summary") or {}).get("n_inferred_events"), "n_ambiguous_part_manipulation_events": (raw_report.get("summary") or {}).get("n_ambiguous_part_manipulation_events"), "n_events_with_confidence_1": (raw_report.get("summary") or {}).get("n_events_with_confidence_1"), "candidate_summaries": raw_report.get("l1_candidates_summary", [])},
        "summary": v,
        "l1_candidates_summary": summaries,
        "w_audit_normalization_summary": [x for x in audit],
        "visual_ledger_normalization_summary": [x for x in ledger],
        "consolidation_log": consolidation_log,
        "confidence_recalibration_log": recal_log,
        "warning_cleanup_log": cleanup_log,
        "overlapping_gt_reference": _load_gt(args.gt_jsonl),
        "diagnostic_checks": checks,
        "diagnostic_notes": ["Postprocess-only: no VLM, frame extraction, crops, or GT-based construction.", "Confidence was recomputed from evidence bands rather than serial penalties."],
        "potential_failure_cases": [],
    }


def _summary(c):
    o = c.get("direct_object") or {}
    return {"l1_id": c.get("l1_id"), "t_start": c.get("t_start"), "t_end": c.get("t_end"), "micro_action_description": c.get("micro_action_description"), "verb": c.get("verb"), "object": o.get("name"), "part_of": o.get("part_of"), "event_subtype": c.get("event_subtype"), "evidence_type": c.get("evidence_type"), "specificity_level": c.get("specificity_level"), "confidence_before_postprocess": c.get("confidence_before_postprocess"), "confidence": c.get("confidence"), "qa_importance": c.get("qa_importance"), "hallucination_risk": c.get("hallucination_risk"), "merged_from_l1_ids": c.get("merged_from_l1_ids", []), "related_l1_ids": c.get("related_l1_ids", []), "warnings": c.get("warnings", [])}


def _load_gt(path):
    if not path or not Path(path).exists():
        return []
    return read_jsonl(path)


def _count(records, key, value):
    return sum(1 for r in records if r.get(key) == value)


def _print_report(report):
    v = report["summary"]
    print("L1 postprocess summary:")
    print(f"final={v['n_final_l1_candidates']} observed={v['n_observed_events']} inferred={v['n_inferred_events']} ambiguous={v['n_ambiguous_part_manipulation_events']} conf1={v['n_events_with_confidence_1']}")
    print(f"visual accepted/merged/support/rejected={v['n_visual_accepted']}/{v['n_visual_merged']}/{v['n_visual_support']}/{v['n_visual_rejected']}")
    print(f"W audit confirmed/corrected/support/vague/specific/partial/wrong/not_applicable={v['n_w_confirmed']}/{v['n_w_corrected']}/{v['n_w_downgraded_to_support']}/{v['n_w_too_vague']}/{v['n_w_too_specific']}/{v['n_w_partially_supported']}/{v['n_w_likely_wrong']}/{v['n_w_not_applicable']}")
    print(f"merges={v['n_cross_segment_merges']} relations={v['n_cross_segment_relations']} recalibrated={v['n_confidence_recalibrated']} warning_cleanup={v['n_warning_cleanup']}")
    for c in report["l1_candidates_summary"]:
        print(f"{c['l1_id']} | {c['t_start']:.2f}-{c['t_end']:.2f} | {c['verb']}/{c['object']}/part_of={c['part_of']} | {c['event_subtype']} | {c['evidence_type']} | conf={c['confidence']:.2f} | merged={c['merged_from_l1_ids']} | warnings={c['warnings']}")
    print("Diagnostic checks:")
    print(json.dumps(report["diagnostic_checks"], indent=2))


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-candidates-jsonl", required=True)
    p.add_argument("--raw-preconsolidation-jsonl", required=True)
    p.add_argument("--raw-evidence-ledger-jsonl", required=True)
    p.add_argument("--raw-quality-report-json", required=True)
    p.add_argument("--output-candidates-jsonl", required=True)
    p.add_argument("--output-evidence-ledger-jsonl", required=True)
    p.add_argument("--output-postprocess-log-jsonl", required=True)
    p.add_argument("--output-quality-report-json", required=True)
    p.add_argument("--gt-jsonl")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
