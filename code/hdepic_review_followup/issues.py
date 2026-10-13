"""Source-label advisories for the fixed 50 validation WHY review examples.

This module writes only new review artifacts. It never changes a question,
reason label, prediction, parser, NLL, split, checkpoint, or extraction rule.
All proposals require human adjudication and preserve exact source offsets.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


VERSION = "hdepic_source_label_advisories_v1"
REQUIRED = {
    "P07-20240529-191007-94",
    "P07-20240529-131737-170",
    "P07-20240529-102652-196",
}

# Assistant observations of the existing cached contact sheets, not human
# reason-validity judgments or confirmations of the narrated intention.
VISUAL_CONTEXT = {
    8: "Two utensils are manipulated above the mixing bowl; a spoon then moves toward the tray and dough portions in the final frames. Both spoon manipulation and placement occur in the input window. Check temporal alignment before narrowing the question to shaping transfers.",
    23: "Two utensils are manipulated above the mixing bowl, followed by a spoon moving to the tray and dough placement in the final frames. The contact sheet does not establish that the full target interval concerns only shaping transfers.",
    25: "A hand presses on the knife while cutting on the board. This supports inspecting the omitted knife object, but does not establish the reason label's validity or the actor's intention.",
    32: "Two utensils are manipulated above the mixing bowl, followed by movement toward the tray and dough placement. The window contains multiple related operations; verify the annotated transfer interval before adopting a narrower question.",
    33: "Onion manipulation and knife cutting are both visible within the input window. The contact sheet does not conclusively establish the onion's flat-side orientation or isolate the placement interval from cutting.",
    41: "The under-counter storage area and hand interaction are partly dark or occluded. The sheet does not resolve the object or relation denoted by 'it fits'; no replacement noun is proposed.",
    42: "A spoon is held over the tray while a jar is held in the other hand; jar placement is visible later in the input window. This is visual context for the additional-action flag, not proof of purpose or a new temporal label.",
    46: "Two spoons move together over the bowl. The source's spoon-to-spoon relation can be reviewed in these inputs; the sheet alone does not validate the stated purpose.",
}

# These are individually inspected fixed-review cases, not generic extraction
# or modifier-enrichment rules. Exact source phrases serve as proposal evidence.
SPECS = {
    "P07-20240529-191007-94": {
        "categories": ["reason_span_contains_another_action", "question_scope_omits_orientation"],
        "concern": "The saved reason includes using the knife to cut off the onion end. That is an additional action rather than the stated reason for placing the half flat-side down. The question also omits the flat-side-down orientation that the no-rolling reason explains.",
        "proposed_reason": "so that it doesn't roll",
        "proposed_question_source": "Place the other half flat side down",
        "proposed_action": "Consider an exact-span reason trim and a question identifying the observed orientation, only after checking action alignment. Keep canonical IDs unchanged.",
    },
    "P07-20240529-131737-170": {
        "categories": ["reason_span_contains_another_action"],
        "concern": "The saved reason continues with putting the spice jar down using the left hand. This is another action rather than the reason for tipping the spoon.",
        "proposed_reason": "so that all of the spices go onto the chickpeas on the tray",
        "proposed_action": "Consider retaining only the exact purpose clause before the jar-placement action, after human scope review.",
    },
    "P07-20240529-102652-196": {
        "categories": ["unclear_reason_referent", "question_scope_underspecified_object"],
        "concern": "The source says 'it fits' without naming the referent, and the action question says only 'move things'. The supplied history mentions bottle placement, but it does not settle which object or fit relation the pronoun denotes.",
        "proposed_action": "Review the original source, cached inputs and completed-action history. Do not replace 'it' with a guessed noun. If the referent or target remains uncertain, consider an explicit ambiguity flag in a future adjudicated dataset version.",
    },
    "P07-20240530-102543-534": {
        "categories": ["question_scope_omits_spoon_to_spoon_transfer"],
        "concern": "The source specifies repeatedly transferring cookie dough from one spoon to the other. The frozen question 'transfer cookie dough' loses the transfer endpoints and repetition that the shaping reason explains.",
        "proposed_question_source": "transfer the cookie dough from one spoon to the other spoon repeatedly",
        "proposed_action": "Consider a purpose-free question retaining the explicitly observed spoon-to-spoon transfer and repetition, after alignment review. Do not add the shaping purpose to the question.",
    },
    "P07-20240530-102543-536": {
        "categories": ["question_scope_omits_spoon_to_spoon_transfer"],
        "concern": "The source specifies repeatedly transferring cookie dough from one spoon to the other. The frozen question 'transfer cookie dough' loses the transfer endpoints and repetition that the shaping reason explains.",
        "proposed_question_source": "transfer the cookie dough from one spoon to the other spoon repeatedly",
        "proposed_action": "Consider a purpose-free question retaining the explicitly observed spoon-to-spoon transfer and repetition, after alignment review. Do not add the shaping purpose to the question.",
    },
    "P07-20240530-102543-538": {
        "categories": ["question_scope_omits_spoon_to_spoon_transfer"],
        "concern": "The source specifies repeatedly transferring cookie dough from one spoon to the other. The frozen question 'transfer cookie dough' loses the transfer endpoints and repetition that the shaping reason explains.",
        "proposed_question_source": "transfer the cookie dough from one spoon to the other spoon repeatedly",
        "proposed_action": "Consider a purpose-free question retaining the explicitly observed spoon-to-spoon transfer and repetition, after alignment review. Do not add the shaping purpose to the question.",
    },
    "P07-20240530-102543-529": {
        "categories": ["question_scope_omits_spoon_to_spoon_scraping"],
        "concern": "The frozen question 'scrape one spoon' omits scraping it onto the other spoon. The destination matters to the stated reason for collecting the dough on the left spoon.",
        "proposed_question_source": "Scrape one spoon onto the other spoon",
        "proposed_action": "Consider a question retaining the source's observed spoon-to-spoon relation, after alignment review. Do not infer a different hand or destination from the reason alone.",
    },
    "P07-20240531-201703-39": {
        "categories": ["question_scope_omits_object_of_pressure"],
        "concern": "The frozen question 'add pressure' does not identify the knife. The source explicitly attaches the pressure to the knife; this matters to the stated cutting reason.",
        "proposed_question_source": "add pressure to the knife",
        "proposed_action": "Consider a question identifying the pressure on the knife after checking the interval. This proposal does not attach a hand modifier to a different action.",
    },
    "P07-20240529-191007-108": {
        "categories": ["reason_attachment_joint_action_uncertain", "question_scope_omits_rotation_detail"],
        "concern": "The source links 'turn the onion 45 degrees and make thin horizontal slices' to producing cubes, while the annotated action and question identify only turning the onion. The reason may attach to the combined cutting sequence rather than turning alone.",
        "proposed_action": "Review attachment and temporal alignment. Do not automatically add the separate slicing action to the target, or infer that the reason explains turning alone. No replacement reason or question is proposed before adjudication.",
    },
    "P07-20240529-194518-519": {
        "categories": ["question_scope_omits_scraping_surface"],
        "concern": "The source specifies scraping the spatula against the frying-pan edge. The frozen question omits that surface, which distinguishes this transfer of remaining butter from a generic scrape or cleaning action.",
        "proposed_question_source": "Scrape the spatula against the edge of the frying pan",
        "proposed_action": "Consider preserving the observed scraping surface in an individually reviewed question. Keep the reason unchanged unless a reviewer finds a separate label issue.",
    },
    "P07-20240531-101357-663": {
        "categories": ["question_scope_omits_tapping_destination"],
        "concern": "The source specifies tapping the spoon against the food-waste bin. The frozen question omits that location, which is relevant to the reference's disposal purpose.",
        "proposed_question_source": "Tap the spoon against the side of the food waste bin",
        "proposed_action": "Consider preserving the observed tapping surface in an individually reviewed question. Do not add the subsequent bin-lid action.",
    },
    "P07-20240531-201703-352": {
        "categories": ["question_scope_omits_manner_choice"],
        "concern": "The source says 'carefully pour the can', and the reason distinguishes draining liquid from retaining beans. A bare 'pour can' question can be broader than that manner-specific reason.",
        "proposed_question_source": "carefully pour the can",
        "proposed_action": "Review whether the supplied frames establish the careful draining choice. The source-grounded candidate retains only the observed manner; it does not copy the liquid/bean outcome into the question.",
    },
    "P07-20240529-194518-440": {
        "categories": ["question_scope_omits_wiping_surface"],
        "concern": "The source describes wiping the knife on the small-bowl side, with butter transferred into the bowl. The frozen 'wipe knife' question can also describe cleaning or drying and omits the observed contact surface.",
        "proposed_question_source": "Wipe the knife on the side of the small bowl",
        "proposed_action": "Consider preserving the observed vessel-edge wiping relation after action-alignment review. Do not put the butter-transfer purpose into the question.",
    },
    "P07-20240530-102543-107": {
        "categories": ["question_scope_omits_wiping_surface"],
        "concern": "The source describes wiping the knife blade along the jug edge, with butter transferred into the jug. The frozen 'wipe blade of knife' question omits that relation and can be broader than the reference purpose.",
        "proposed_question_source": "wipe the blade of the knife along the edge of the jug",
        "proposed_action": "Consider preserving the observed vessel-edge wiping relation after action-alignment review. Do not put the butter-transfer purpose into the question.",
    },
}


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _source_span(narration, text, start=None):
    """Return one exact source span; offsets are Python Unicode character offsets."""
    if start is None:
        if narration.count(text) != 1:
            raise ValueError(f"Proposed source phrase is absent or nonunique: {text}")
        start = narration.index(text)
    end = start + len(text)
    if narration[start:end] != text:
        raise ValueError("Saved or proposed source span does not match exact narration offsets")
    return {"text": text, "char_start": start, "char_end": end}


def _span_from_saved(narration, span):
    result = _source_span(narration, span["text"], span["char_start"])
    if result["char_end"] != span["char_end"]:
        raise ValueError("Saved span end offset disagrees with exact source length")
    return result


def _build_issue(record, sidecar, blind, spec):
    annotation = record["scoring_metadata"]["reason_annotation"]
    narration = annotation["original_narration"]
    current = _source_span(narration, annotation["reason_text"], annotation["char_start"])
    assert current["char_end"] == annotation["char_end"]
    assert record["target"]["reason"] == current["text"] == blind["source_stated_reason"]
    assert annotation == sidecar["reason_annotation"]
    assert blind["original_narration"] == narration
    assert int(blind["reason_char_start"]) == current["char_start"]
    assert int(blind["reason_char_end"]) == current["char_end"]
    assert record["input"]["action_question_text"] == blind["action_question_text"]
    proposed_reason = _source_span(narration, spec["proposed_reason"]) if spec.get("proposed_reason") else None
    if proposed_reason:
        assert current["char_start"] == proposed_reason["char_start"] and proposed_reason["char_end"] < current["char_end"]
    proposed_question = _source_span(narration, spec["proposed_question_source"]) if spec.get("proposed_question_source") else None
    if proposed_question:
        assert proposed_question["char_end"] <= current["char_start"], "Question proposal includes reason text"
    target = sidecar["current_action"]
    assert target == annotation["target"] and target == record["scoring_metadata"]["current_action"]
    assert target["end_sec"] == record["query_time_sec"]
    assert all(frame["timestamp_sec"] <= record["query_time_sec"] + 1e-9 for frame in sidecar["frames"])
    assert all(action["end_sec"] <= record["query_time_sec"] for action in record["input"]["history"])
    row_number = int(blind["blind_row"])
    return {
        "issue_id": "source-advisory:" + annotation["narration_id"],
        "version": VERSION, "status": "pending_human_adjudication",
        "record_id": record["id"], "narration_id": annotation["narration_id"],
        "blind_row": row_number, "participant": record["participant"], "partition": "validation",
        "video_id": record["video_id"], "query_time_sec": record["query_time_sec"],
        "action_question_text": record["input"]["action_question_text"],
        "target_action": target, "frozen_target": record["target"],
        "original_narration": narration,
        "frozen_v1_reason_span": _span_from_saved(narration, annotation.get("v1_span", current)),
        "frozen_current_reason_span": current,
        "frozen_revised_reason_span": _span_from_saved(narration, annotation.get("revised_span", current)),
        "frozen_revision_reason": annotation.get("revision_reason"),
        "extraction_rule": annotation["extraction_rule"], "binding_rule": annotation.get("binding_rule"),
        "binding_clause": annotation.get("binding_clause"),
        "concern_categories": spec["categories"], "concern": spec["concern"],
        "proposed_reason_span": proposed_reason,
        "proposed_removed_suffix_span": (_source_span(narration, narration[proposed_reason["char_end"]:current["char_end"]], proposed_reason["char_end"])
                                         if proposed_reason else None),
        "proposed_question_text": proposed_question["text"] if proposed_question else None,
        "proposed_question_source_span": proposed_question,
        "proposal_rationale_and_next_check": spec["proposed_action"],
        "proposal_applied": False, "original_nll_recomputed_against_proposal": False,
        "input_history": record["input"]["history"], "actual_input_frames": sidecar["frames"],
        "review_contact_sheet": f"../visual/contact_sheets/{row_number:02d}.jpg",
        "assistant_cached_visual_observation": VISUAL_CONTEXT.get(row_number),
        "evidence_status": "assistant source/context triage only; human label and visual-purpose review pending",
        "offset_convention": "zero-based Unicode character offsets; half-open [char_start,char_end); exact original narration, no normalization",
        "human_adjudication": "", "approved_reason_span": "", "approved_question_text": "", "reviewer_notes": "",
    }


def build_issues(project, run, dataset, out):
    """Write advisory-only JSONL/CSV/Markdown; return an artifact/count summary."""
    project, run, dataset, out = (Path(path).resolve() for path in (project, run, dataset, out))
    assert run.is_relative_to(project) and dataset.is_relative_to(project)
    assert out.is_relative_to(run) and out != run
    paths = {"blind": run / "review/blind_why_review.csv",
             "text": dataset / "text/validation.jsonl",
             "scoring": dataset / "swift/validation.scoring.jsonl"}
    input_hashes = {str(path): _sha256(path) for path in paths.values()}
    with paths["blind"].open(newline="") as handle:
        blind_rows = list(csv.DictReader(handle))
    assert len(blind_rows) == 50 and len({row["id"] for row in blind_rows}) == 50
    records = _read_jsonl(paths["text"])
    sidecars = _read_jsonl(paths["scoring"])
    assert len(records) == len(sidecars) == 2209
    assert all(record["id"] == sidecar["id"] and record["partition"] == sidecar["partition"] == "validation"
               for record, sidecar in zip(records, sidecars))
    by_id = {record["id"]: (record, sidecar) for record, sidecar in zip(records, sidecars)}
    issues, inspected = [], []
    for blind in blind_rows:
        record, sidecar = by_id[blind["id"]]
        assert record["task"] == "explain" and record["participant"] == "P07"
        annotation = record["scoring_metadata"]["reason_annotation"]
        assert annotation["narration_id"] == blind["source_narration_id"]
        inspected.append({"blind_row": int(blind["blind_row"]), "record_id": record["id"],
                          "narration_id": annotation["narration_id"],
                          "question": record["input"]["action_question_text"],
                          "source_narration": annotation["original_narration"],
                          "advisory_flagged": annotation["narration_id"] in SPECS,
                          "unflagged_does_not_mean_human_validated": True})
        if annotation["narration_id"] in SPECS:
            issues.append(_build_issue(record, sidecar, blind, SPECS[annotation["narration_id"]]))
    assert {issue["narration_id"] for issue in issues} == set(SPECS)
    assert REQUIRED <= {issue["narration_id"] for issue in issues}
    issues_dir = out / "issues"
    jsonl_path = issues_dir / "source_label_issues.jsonl"
    csv_path = issues_dir / "source_label_issues.csv"
    markdown_path = issues_dir / "README.md"
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            saved = list(csv.DictReader(handle))
        review_fields = ["human adjudication", "approved reason span", "approved question text", "reviewer notes"]
        if any((row.get(field) or "").strip() for row in saved for field in review_fields):
            raise ValueError("Existing advisory CSV contains reviewer entries; preserve it and use a new output version")
    _write_text(jsonl_path, "".join(_json(issue) + "\n" for issue in issues))
    csv_rows = []
    for issue in issues:
        csv_rows.append({
            "blind_row": issue["blind_row"], "record_id": issue["record_id"], "narration_id": issue["narration_id"],
            "status": issue["status"], "query_time_sec": issue["query_time_sec"],
            "action_question_text": issue["action_question_text"], "target_action_json": _json(issue["target_action"]),
            "original_narration": issue["original_narration"],
            "frozen_v1_reason_span_json": _json(issue["frozen_v1_reason_span"]),
            "frozen_current_reason_span_json": _json(issue["frozen_current_reason_span"]),
            "frozen_revised_reason_span_json": _json(issue["frozen_revised_reason_span"]),
            "extraction_rule": issue["extraction_rule"], "binding_rule": issue["binding_rule"],
            "concern_categories": ";".join(issue["concern_categories"]), "concern": issue["concern"],
            "proposed_reason_span_json": _json(issue["proposed_reason_span"]),
            "proposed_removed_suffix_span_json": _json(issue["proposed_removed_suffix_span"]),
            "proposed_question_text": issue["proposed_question_text"] or "",
            "proposed_question_source_span_json": _json(issue["proposed_question_source_span"]),
            "proposal_rationale_and_next_check": issue["proposal_rationale_and_next_check"],
            "input_history_json": _json(issue["input_history"]), "frozen_target_json": _json(issue["frozen_target"]),
            "review_contact_sheet": issue["review_contact_sheet"],
            "assistant_cached_visual_observation": issue["assistant_cached_visual_observation"] or "",
            "human adjudication": "", "approved reason span": "", "approved question text": "", "reviewer notes": "",
        })
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    _write_text(csv_path, buffer.getvalue())
    paragraphs = ["# Source-label advisory issues", "",
        "These are assistant source/context review flags for the existing fixed 50 validation WHY examples. Human adjudication remains pending. An unflagged row is not a validated label. No dataset, question, prediction, frozen metric, or original NLL has been changed. Do not apply these proposals to the completed experiment's scores.", "",
        "All offsets are zero-based Unicode character offsets into the saved original narration, with an exclusive end. The JSONL and CSV retain the v1, current and existing revised spans separately from every proposed correction. Canonical action and annotation IDs remain unchanged. Question proposals are individually reviewed source phrases, not restored automatic modifier enrichment.", "",
        "The three cookie-dough transfer cases share narration text but have different action intervals and record IDs; all remain separate review cases.", ""]
    for issue in issues:
        span = issue["frozen_current_reason_span"]
        paragraphs += [f"## Blind row {issue['blind_row']}: {issue['narration_id']}", "",
                       f"Status: **pending human adjudication**. Query: {issue['query_time_sec']:.3f}s. Question: `{issue['action_question_text']}`.", "",
                       f"Canonical target: `{issue['target_action']['action']}` [{issue['target_action']['start_sec']:.3f}, {issue['target_action']['end_sec']:.3f}]s; verb IDs {issue['target_action']['verb_ids']}; noun IDs {issue['target_action']['noun_ids']}.", "",
                       "Original narration: " + issue["original_narration"], "",
                       f"Frozen reason [{span['char_start']}, {span['char_end']}): {span['text']}", "",
                       "Concern: " + issue["concern"], ""]
        if issue["proposed_reason_span"]:
            proposed = issue["proposed_reason_span"]
            paragraphs += [f"Proposed exact reason span [{proposed['char_start']}, {proposed['char_end']}), pending: {proposed['text']}", ""]
        else:
            paragraphs += ["Proposed reason replacement: none. The frozen reason remains unchanged.", ""]
        if issue["proposed_question_text"]:
            proposed = issue["proposed_question_source_span"]
            paragraphs += [f"Proposed purpose-free question source [{proposed['char_start']}, {proposed['char_end']}), pending: {proposed['text']}", ""]
        if issue["assistant_cached_visual_observation"]:
            paragraphs += ["Assistant cached-frame observation (human review pending): " + issue["assistant_cached_visual_observation"], ""]
        paragraphs += ["Next review: " + issue["proposal_rationale_and_next_check"], "",
                       f"[Same cached input frames]({issue['review_contact_sheet']}) · [Visual review](../visual/index.html#record-{issue['blind_row']:02d})", ""]
    _write_text(markdown_path, "\n".join(paragraphs))
    categories = Counter(category for issue in issues for category in issue["concern_categories"])
    audit = {"version": VERSION, "status": "pending_human_adjudication", "fixed_review_text_rows_inspected": 50,
             "cached_contact_sheets_visually_inspected": sorted(VISUAL_CONTEXT),
             "assistant_visual_observations": len(VISUAL_CONTEXT), "human_visual_adjudications": 0,
             "issues": len(issues), "known_required_issues": len(REQUIRED), "category_counts": dict(categories),
             "proposed_reason_trims": sum(issue["proposed_reason_span"] is not None for issue in issues),
             "proposed_question_contexts": sum(issue["proposed_question_text"] is not None for issue in issues),
             "offset_checks_passed": True, "question_proposals_exclude_reason_clauses": True,
             "human_review_columns_blank": True, "frozen_labels_modified": False,
             "original_nll_recomputed_against_proposals": False, "automatic_modifier_enrichment_restored": False,
             "source_video_decodes": 0, "test_records_read": 0, "model_inference_calls": 0,
             "input_hashes": input_hashes, "inspected_fixed_review_rows": inspected,
             "paths": {"jsonl": str(jsonl_path), "csv": str(csv_path), "markdown": str(markdown_path)}}
    assert input_hashes == {str(path): _sha256(path) for path in paths.values()}, "An original input changed during advisory preparation"
    _write_text(issues_dir / "audit.json", json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit
