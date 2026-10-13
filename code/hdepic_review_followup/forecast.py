"""Saved future-action repetition diagnostics with fixed validation counts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from code.hdepic_experiment.metrics import normalized
from .fields import _load_validation, _output_dir, _read_jsonl, _rows_by_id, _sha256, _write_json, _write_jsonl, extract_field

VERSION = "hdepic_saved_forecast_repetition_v1"


def parse_triple(raw):
    """Require a unique complete <next> block and three semicolon items.

    Extraction does not require valid WHY/action/goal or whole-response format.
    No alternative delimiter, action name or repeated block is guessed.
    """
    extraction = extract_field(raw, "next")
    actions = [item.strip() for item in extraction["text"].split(";")] if extraction["extractable"] else []
    valid = extraction["extractable"] and len(actions) == 3 and all(actions)
    return {"valid": bool(valid), "extraction": extraction,
            "actions": actions, "normalized_actions": [normalized(x) for x in actions],
            "error": None if valid else "unextractable_next_field" if not extraction["extractable"] else "next_action_count_or_empty_action"}


def _reference_triple(record):
    actions = record["target"]["next_actions"]
    if not isinstance(actions, list) or len(actions) != 3 or not all(isinstance(a, str) and normalized(a) for a in actions):
        raise ValueError(f"Malformed frozen reference triple: {record['id']}")
    return {"valid": True, "actions": actions, "normalized_actions": [normalized(a) for a in actions]}


def repetition_rows(records, predictions=None):
    """Include every record, retaining malformed triples in the denominator."""
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate validation IDs")
    by_id = _rows_by_id(predictions, set(ids)) if predictions is not None else None
    output = []
    for index, record in enumerate(records):
        reference = _reference_triple(record)
        row = by_id.get(record["id"]) if by_id is not None else None
        raw = row.get("prediction", "") if row else None
        parsed = parse_triple(raw) if by_id is not None else reference
        successful = by_id is None or row is not None and row.get("status", "completed") == "completed"
        valid = successful and parsed["valid"]
        triple = parsed["normalized_actions"]
        repeated = bool(valid and len(set(triple)) == 1)
        output.append({
            "protocol_version": VERSION, "id": record["id"], "row_index": index,
            "task": record["task"], "raw_prediction": raw,
            "saved_generation_status": "reference" if by_id is None else row.get("status", "completed") if row else "missing",
            "triple_valid": bool(valid), "malformed_triple": not valid,
            "error": None if valid else "missing_prediction" if row is None else "generation_error" if not successful else parsed["error"],
            "predicted_triple": parsed["actions"] if by_id is not None else None,
            "normalized_predicted_triple": triple if by_id is not None else None,
            "next_field_extraction": parsed.get("extraction"),
            "reference_triple": reference["actions"], "normalized_reference_triple": reference["normalized_actions"],
            "all_three_identical": repeated,
            "reference_all_three_identical": len(set(reference["normalized_actions"])) == 1,
        })
    return output


def repetition_summary(rows):
    denominator = len(rows)
    repeat = sum(r["all_three_identical"] for r in rows)
    malformed = sum(r["malformed_triple"] for r in rows)
    return {"denominator": denominator, "all_three_identical": repeat,
            "rate": repeat / denominator if denominator else None,
            "valid_triples": denominator - malformed, "malformed_triples": malformed,
            "malformed_triple_rate": malformed / denominator if denominator else None,
            "malformed_error_counts": dict(Counter(r["error"] for r in rows if r["malformed_triple"]))}


def build_forecast(project, run, dataset, out):
    """Write repetition counts, every triple, affected IDs and malformed cases."""
    project, run, dataset = Path(project).resolve(), Path(run).resolve(), Path(dataset).resolve()
    destination = _output_dir(run, out, "forecast")
    records = _load_validation(run, dataset)
    stages, sources = {}, {}
    for stage in ("epoch1", "epoch2", "references"):
        source = run / "predictions" / stage / "generation.jsonl" if stage != "references" else dataset / "text/validation.jsonl"
        rows = repetition_rows(records, _read_jsonl(source) if stage != "references" else None)
        stages[stage] = repetition_summary(rows)
        _write_jsonl(destination / f"{stage}.all_triples.jsonl", ({"stage": stage, **r} for r in rows))
        _write_jsonl(destination / f"{stage}.repeated_triples.jsonl", ({"stage": stage, **r} for r in rows if r["all_three_identical"]))
        _write_jsonl(destination / f"{stage}.malformed_triples.jsonl", ({"stage": stage, **r} for r in rows if r["malformed_triple"]))
        stages[stage]["all_triples_file"] = f"forecast/{stage}.all_triples.jsonl"
        stages[stage]["repeated_triples_file"] = f"forecast/{stage}.repeated_triples.jsonl"
        stages[stage]["malformed_triples_file"] = f"forecast/{stage}.malformed_triples.jsonl"
        sources[stage] = {"path": str(source), "sha256": _sha256(source)}
    summary = {
        "protocol_version": VERSION,
        "definition": "three nonempty semicolon-delimited strings in one independently extractable literal <next> field, identical after the unchanged original normalization and a single generated-field HTML unescape; reference strings use original normalization without HTML decoding",
        "denominator": "all frozen validation IDs, including unextractable/malformed/missing/error triples; none excluded",
        "repetition_is_automatically_an_error": False, "semantic_accuracy_measured": False,
        "whole_response_format_is_not_a_gate": True,
        "stages": stages, "sources": sources,
        "labels_or_predictions_changed": False, "inference_or_training_performed": False,
    }
    _write_json(destination / "repetition_summary.json", summary)
    return summary
