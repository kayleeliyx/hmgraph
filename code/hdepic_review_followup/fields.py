"""Independent field lexical diagnostics from saved validation predictions.

This supplementary protocol never changes the original parser, metric report,
predictions, prompts, labels, or selected checkpoint. It measures wording only.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from code.hdepic_experiment.metrics import normalized, score_predictions

VERSION = "hdepic_independent_field_lexical_v1"
STAGES = ("base", "epoch1", "epoch2")
CONTENT_FIELDS = ("why", "action", "goal")


def protocol():
    return {
        "version": VERSION,
        "extraction": "independently inspect each requested literal lowercase field tag; require exactly one opening and one closing occurrence enclosing one complete nonempty plain-text pair; no nested markup, guessing, reference-dependent choice, or whole-response validity gate",
        "flags": {
            "missing": "no complete plain-text opening/closing pair",
            "empty": "a complete plain-text pair is whitespace-only after decoding HTML entities once",
            "duplicate": "more than one opening occurrence, closing occurrence, or complete plain-text pair",
            "ambiguous": "unmatched/reversed occurrences, nested/crossed markup, or duplicate occurrences prevent a unique field boundary",
        },
        "flags_not_mutually_exclusive": True,
        "normalization": "decode extracted generated field HTML entities exactly once, then use the unchanged original normalized(): Unicode NFKC, casefold, collapse whitespace; retain punctuation and wording; reference text is not HTML-decoded",
        "denominators": "all fixed validation IDs eligible for the requested field; WHY Explain only, action Track only, goal annotated targets only; missing/error/empty/unextractable predictions remain",
        "generation_error": "field content may be shown, but a non-completed or absent saved generation never receives a lexical match",
        "other_fields": "malformed NEXT, unknown tags and prose outside this field do not invalidate a unique requested field; markup inside this field does",
        "original_strict_metric": "retained separately, recomputed with unchanged frozen scorer and verified against original metrics.json; not replaced by supplementary field scores",
        "interpretation": "normalized lexical exact-match diagnostics, not semantic accuracy, action-class accuracy, visual support, or verified intent",
        "checkpoint_selection": "epoch1 remains selected under the existing mean-per-example Explain assistant-target NLL criterion",
    }


def extract_field(raw, field):
    """Extract a unique literal pair without using surrounding-field validity.

    Additional unmatched occurrences of the same field are uncertain and are
    rejected. A malformed different field outside this pair has no effect.
    """
    if field not in (*CONTENT_FIELDS, "next"):
        raise ValueError(f"Unsupported field: {field}")
    text = raw if isinstance(raw, str) else ""
    opening, closing = f"<{field}>", f"</{field}>"
    opens, closes = text.count(opening), text.count(closing)
    matches = list(re.finditer(re.escape(opening) + r"([^<>]*)" + re.escape(closing), text, re.DOTALL))
    decoded = [html.unescape(m.group(1)).strip() for m in matches]
    duplicate = opens > 1 or closes > 1 or len(matches) > 1
    missing = not matches
    empty = any(not value for value in decoded)
    # A missing field without any tag occurrence has no ambiguous boundary.
    # Tags with no complete plain-text pair include unclosed, reversed, nested
    # or crossed fields, all of which require review rather than a guess.
    ambiguous = duplicate or (bool(opens or closes) and (opens != 1 or closes != 1 or len(matches) != 1))
    extractable = opens == closes == len(matches) == 1 and not empty
    if extractable:
        status = "extracted"
    elif duplicate:
        status = "duplicate"
    elif ambiguous:
        status = "ambiguous"
    elif empty:
        status = "empty"
    else:
        status = "missing"
    match = matches[0] if extractable else None
    return {
        "status": status, "extractable": extractable,
        "missing": missing, "empty": empty, "duplicate": duplicate, "ambiguous": ambiguous,
        "opening_count": opens, "closing_count": closes, "complete_pair_count": len(matches),
        "text": decoded[0] if extractable else None,
        "raw_field_text": match.group(1) if match else None,
        "raw_char_start": match.start(1) if match else None,
        "raw_char_end": match.end(1) if match else None,
    }


def reference_fields(record):
    if record["task"] == "explain":
        fields = {"why": record["target"]["reason"]}
    elif record["task"] == "track":
        fields = {"action": record["target"]["current_action"]}
    else:
        raise ValueError(f"Unexpected validation task: {record['task']}")
    if "goal" in record["target"]:
        fields["goal"] = record["target"]["goal"]
    return fields


def _rows_by_id(rows, expected_ids):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        if row["id"] not in expected_ids:
            raise ValueError(f"Prediction outside frozen validation: {row['id']}")
        result[row["id"]] = row
    return result


def score_fields(records, rows):
    """Score all requested content fields over their complete denominators."""
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate validation IDs")
    by_id = _rows_by_id(rows, set(ids))
    original = score_predictions(records, rows)
    strict_rows = {r["id"]: r for r in original["per_example"]}
    counters = Counter()
    per_example = []
    for index, record in enumerate(records):
        row = by_id.get(record["id"])
        raw = row.get("prediction", "") if row else ""
        successful = row is not None and row.get("status", "completed") == "completed"
        references = reference_fields(record)
        extracted = {field: extract_field(raw, field) for field in references}
        exact = {}
        for field, reference in references.items():
            item = extracted[field]
            exact[field] = bool(successful and item["extractable"] and normalized(item["text"]) == normalized(reference))
            for scope in ("all", record["task"]):
                counters[(scope, field, "denominator")] += 1
                counters[(scope, field, "exact_matches")] += exact[field]
                for flag in ("extractable", "missing", "empty", "duplicate", "ambiguous"):
                    counters[(scope, field, flag)] += item[flag]
                counters[(scope, field, "missing_prediction")] += row is None
                counters[(scope, field, "generation_error")] += row is not None and not successful
        per_example.append({
            "protocol_version": VERSION, "id": record["id"], "row_index": index,
            "task": record["task"], "goal_supervised": "goal" in record["target"],
            "saved_generation_status": row.get("status", "completed") if row else "missing",
            "finish_reason": row.get("finish_reason") if row else None,
            "generated_tokens": row.get("generated_tokens") if row else None,
            "raw_prediction": raw, "raw_generated_tokens_text": row.get("raw_generated_tokens_text") if row else None,
            "reference_fields": references,
            "normalized_reference_fields": {field: normalized(value) for field, value in references.items()},
            "field_extraction": extracted,
            "extracted_fields": {field: item["text"] for field, item in extracted.items()},
            "normalized_extracted_fields": {field: normalized(item["text"]) if item["extractable"] else None for field, item in extracted.items()},
            "lexical_exact_match": exact,
            "original_strict_metric": strict_rows[record["id"]],
        })
    scopes = {}
    for scope in ("all", "explain", "track"):
        scopes[scope] = {}
        for field in CONTENT_FIELDS:
            denominator = counters[(scope, field, "denominator")]
            matches = counters[(scope, field, "exact_matches")]
            scopes[scope][field] = {
                "denominator": denominator, "exact_matches": matches,
                "rate": matches / denominator if denominator else None,
                **{flag: counters[(scope, field, flag)] for flag in ("extractable", "missing", "empty", "duplicate", "ambiguous", "missing_prediction", "generation_error")},
            }
    return {"protocol_version": VERSION, "scopes": scopes, "per_example": per_example,
            "original_strict_metric": {k: v for k, v in original.items() if k != "per_example"},
            "semantic_accuracy_measured": False}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path, rows):
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows))


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_validation(run, dataset):
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    records = _read_jsonl(dataset / "text/validation.jsonl")
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or not records:
        raise ValueError("Missing or duplicate validation records")
    if any(r["partition"] != "validation" or r["participant"] != "P07" for r in records):
        raise ValueError("Only frozen P07 validation records are allowed")
    config = json.loads((run / "frozen_config.json").read_text())
    frozen = json.loads((run / "frozen_evaluation.json").read_text())
    if ids != frozen["ordered_validation_ids"]:
        raise ValueError("Validation order differs from the frozen evaluation")
    for key, relative in (("text", "text/validation.jsonl"), ("prompts", "swift/validation.prompts.jsonl")):
        path = dataset / relative
        if Path(config["data"]["validation"][key]).resolve() != path or _sha256(path) != config["data"]["validation"][key + "_sha256"]:
            raise ValueError(f"Changed frozen validation {key}")
    if len(ids) != 2209 or Counter(r["task"] for r in records) != {"explain": 228, "track": 1981}:
        raise ValueError("Unexpected frozen validation denominators")
    return records


def _output_dir(run, out, subdirectory):
    run, out = Path(run).resolve(), Path(out).resolve()
    # This new follow-up may never overwrite an original run artifact.
    if out != run / "review_followup_v1":
        raise ValueError("Supplementary outputs must use run/review_followup_v1")
    return out / subdirectory


def audit_semicolon_prompts(records, prompts):
    """Inspect saved model inputs, not targets or rewritten prompt templates."""
    if len(prompts) != len(records):
        raise ValueError("Prompt/validation row counts differ")
    findings = []
    for record, prompt in zip(records, prompts):
        messages = prompt["messages"]
        if [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("Expected actual prompt-only system and user messages")
        complete = "\n".join(m["content"] for m in messages)
        # Require an actual delimiter instruction, not a semicolon in history.
        instruction = bool(re.search(r"semicolon|semi-colon", complete, re.IGNORECASE))
        next_lines = [line for line in complete.splitlines() if "<next>" in line or "Forecast exactly three next actions" in line]
        findings.append({"id": record["id"], "task": record["task"],
                         "semicolon_instruction_present": instruction,
                         "next_request_lines": next_lines})
    count = sum(row["semicolon_instruction_present"] for row in findings)
    return {
        "validation_denominator": len(records), "prompts_explicitly_requesting_semicolon": count,
        "prompts_without_explicit_semicolon_instruction": len(records) - count,
        "strict_parser_requirement": "exactly three nonempty next-action strings in one <next> field separated by semicolons",
        "documentary_mismatch": count != len(records),
        "prompts_changed": False, "generation_rerun": False,
        "evidence_source": "saved swift/validation.prompts.jsonl system/user messages only",
        "per_example": findings,
    }


def build_fields(project, run, dataset, out):
    """Write new diagnostics only under out/scoring using saved predictions."""
    project, run, dataset = Path(project).resolve(), Path(run).resolve(), Path(dataset).resolve()
    destination = _output_dir(run, out, "scoring")
    records = _load_validation(run, dataset)
    original_report = json.loads((run / "metrics.json").read_text())
    if original_report["checkpoint_selection"]["selected_stage"] != "epoch1":
        raise ValueError("The completed run must retain its original epoch1 selection")
    stages = {}
    sources = {
        "validation_text": {"path": str(dataset / "text/validation.jsonl"), "sha256": _sha256(dataset / "text/validation.jsonl")},
        "validation_prompts": {"path": str(dataset / "swift/validation.prompts.jsonl"), "sha256": _sha256(dataset / "swift/validation.prompts.jsonl")},
        "original_metrics": {"path": str(run / "metrics.json"), "sha256": _sha256(run / "metrics.json")},
        "frozen_config": {"path": str(run / "frozen_config.json"), "sha256": _sha256(run / "frozen_config.json")},
        "frozen_evaluation": {"path": str(run / "frozen_evaluation.json"), "sha256": _sha256(run / "frozen_evaluation.json")},
    }
    for stage in STAGES:
        source = run / "predictions" / stage / "generation.jsonl"
        result = score_fields(records, _read_jsonl(source))
        if result["original_strict_metric"] != original_report["stages"][stage]["generation"]:
            raise ValueError(f"Original strict metrics failed reproduction for {stage}")
        path = destination / f"{stage}.fields.jsonl"
        _write_jsonl(path, ({"stage": stage, **row} for row in result["per_example"]))
        stages[stage] = {k: v for k, v in result.items() if k != "per_example"}
        stages[stage]["per_example_file"] = str(path.relative_to(Path(out)))
        sources[stage + "_generation"] = {"path": str(source), "sha256": _sha256(source)}
    prompt_audit = audit_semicolon_prompts(records, _read_jsonl(dataset / "swift/validation.prompts.jsonl"))
    _write_jsonl(destination / "semicolon_prompts.jsonl", prompt_audit.pop("per_example"))
    _write_json(destination / "semicolon_mismatch_audit.json", prompt_audit)
    summary = {
        "protocol": protocol(), "validation_records": len(records),
        "fixed_denominators": {"why": sum(r["task"] == "explain" for r in records),
                               "action": sum(r["task"] == "track" for r in records),
                               "goal": sum("goal" in r["target"] for r in records)},
        "stages": stages, "semicolon_prompt_audit": prompt_audit,
        "sources": sources, "original_strict_scores_verified": True,
        "selected_stage_unchanged": "epoch1", "human_review_status": "pending",
        "semantic_accuracy_measured": False, "inference_or_training_performed": False,
    }
    _write_json(destination / "protocol.json", protocol())
    _write_json(destination / "field_scoring_summary.json", summary)
    return summary
