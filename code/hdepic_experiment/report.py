"""Aggregate frozen validation diagnostics and prepare a blinded WHY review.

Reads validation records and saved predictions only. It never runs inference,
changes the dataset, reads test labels, or maps generated text to action IDs.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import statistics
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from code.hdepic_preparation.build import read_jsonl
from .metrics import BLIND_REVIEW_SEED, STAGES, aggregate_losses, blind_order, blind_review_ids, parse_prediction, protocol, rows_by_id, score_predictions, select_checkpoint

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT / "runs/hdepic_sft_8b_v1"
DEFAULT_DATASET = PROJECT / "datasets/hdepic_sft_v2_repaired"
REVIEW_FIELDS = ("valid source reason?", "correct target action?", "answer A valid reason?", "answer A supported by observations?", "answer B valid reason?", "answer B supported by observations?", "preferred answer (A/B/tie)?", "reviewer notes")


def optional_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_validation(dataset):
    records = list(read_jsonl(dataset / "text/validation.jsonl"))
    if not records or any(r["partition"] != "validation" or r["participant"] != "P07" for r in records):
        raise ValueError("Expected only the fixed P07 validation records")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate validation IDs")
    return records


def runtime_summary(rows):
    def finite_numbers(key):
        import math
        return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r[key], bool) and math.isfinite(r[key])]
    seconds = finite_numbers("elapsed_sec")
    memory = finite_numbers("peak_gpu_memory_mib")
    return {"rows_with_elapsed_time": len(seconds), "sum_row_elapsed_sec": sum(seconds), "mean_row_elapsed_sec": statistics.fmean(seconds) if seconds else None, "rows_with_peak_memory": len(memory), "peak_reported_gpu_memory_mib": max(memory) if memory else None, "elapsed_semantics": "sum of saved per-example times, excludes model loading or other runtime unless separately recorded"}


def _display_why(row):
    if row is None:
        return "[missing saved prediction]"
    raw = row.get("prediction", "")
    matches = re.findall(r"<why>([^<>]*)</why>", raw, re.DOTALL) if isinstance(raw, str) else []
    if len(matches) == 1 and matches[0].strip():
        return html.unescape(matches[0]).strip()
    return "[no complete unique <why> field; saved raw output follows]\n" + str(raw)


def write_blind_review(run, dataset, records, review_ids, base_rows, selected_rows, selected_stage):
    expected_ids = blind_review_ids(records)
    if review_ids != expected_ids:
        raise ValueError("Blind review IDs differ from the pre-baseline fixed sample")
    by_id = {r["id"]: r for r in records}
    predictions = {"base": rows_by_id(base_rows), "selected": rows_by_id(selected_rows)}
    model_rows = list(read_jsonl(dataset / "swift/validation.jsonl"))
    if len(model_rows) != len(records):
        raise ValueError("Validation inputs are not aligned with text records")
    source_rows = {r["id"]: (i, model_rows[i]) for i, r in enumerate(records)}
    rows, keys = [], []
    for number, identity in enumerate(review_ids, 1):
        record = by_id[identity]
        row_index, model_row = source_rows[identity]
        order = blind_order(identity)
        a, b = (predictions[stage].get(identity) for stage in order)
        reason = record["scoring_metadata"]["reason_annotation"]
        row = {
            "blind_row": number, "id": identity, "participant": record["participant"],
            "source_video_id": record["video_id"], "query_time_sec": record["query_time_sec"],
            "validation_row_index": row_index, "validation_text_path": str(dataset / "text/validation.jsonl"),
            "source_narration_id": reason["narration_id"],
            "action_question_text": record["input"]["action_question_text"],
            "history_json": json.dumps(record["input"]["history"], ensure_ascii=False),
            "individual_images_json": json.dumps(model_row["images"], ensure_ascii=False),
            "input_prompt": model_row["messages"][1]["content"],
            "source_stated_reason": record["target"]["reason"],
            "reason_char_start": reason["char_start"], "reason_char_end": reason["char_end"],
            "original_narration": reason["original_narration"],
            "answer_A_why": _display_why(a), "answer_B_why": _display_why(b),
            "answer_A_raw": a.get("prediction", "") if a else "", "answer_B_raw": b.get("prediction", "") if b else "",
            "answer_A_saved_status": a.get("status", "completed") if a else "missing", "answer_B_saved_status": b.get("status", "completed") if b else "missing",
            **{field: "" for field in REVIEW_FIELDS},
        }
        rows.append(row)
        keys.append({"blind_row": number, "id": identity, "answer_A_stage": "base" if order[0] == "base" else selected_stage, "answer_B_stage": "base" if order[1] == "base" else selected_stage})
    path = run / "review/blind_why_review.csv"
    if path.exists():
        with path.open(newline="") as f:
            old_rows = {r["id"]: r for r in csv.DictReader(f)}
        for row in rows:
            previous = old_rows.get(row["id"], {})
            annotated = any(previous.get(k, "") for k in REVIEW_FIELDS)
            if annotated:
                content_changed = any(str(row[k]) != previous.get(k, "") for k in row if k not in REVIEW_FIELDS)
                if content_changed:
                    raise ValueError(f"Refusing to replace a reviewed row with changed content: {row['id']}")
                for field in REVIEW_FIELDS:
                    row[field] = previous.get(field, "")
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    atomic_write_json(run / "review/blind_why_review_identity_key.json", {"seed": BLIND_REVIEW_SEED, "selected_stage": selected_stage, "warning": "Keep this identity key separate from the blinded reviewer CSV.", "rows": keys})
    atomic_write_json(run / "review/blind_why_review_manifest.json", {"seed": BLIND_REVIEW_SEED, "sample_size": len(rows), "frozen_ids": review_ids, "review_file": str(path), "human_review_status": "pending", "model_identity_key_separate": True, "missing_or_malformed_predictions_excluded": False})
    return path


def _metric_cell(block):
    if block["rate"] is None:
        return "— (no supervised denominator)"
    return f"{block['exact_matches']}/{block['denominator']} ({100*block['rate']:.2f}%)"


def build_report(run, dataset, frozen_evaluation=None):
    run, dataset = run.resolve(), dataset.resolve()
    records = load_validation(dataset)
    frozen_path = frozen_evaluation or run / "frozen_evaluation.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("metric_protocol") != protocol():
        raise ValueError("Metric implementation differs from the frozen pre-baseline protocol")
    review_ids = frozen.get("blind_review_ids")
    if review_ids != blind_review_ids(records):
        raise ValueError("Blind WHY review IDs differ from the frozen sample")
    scopes, loss_summaries, generation_rows, runtime = {}, {}, {}, {}
    for stage in STAGES:
        path = run / "predictions" / stage
        predictions = list(read_jsonl(path / "generation.jsonl")) if (path / "generation.jsonl").exists() else []
        losses = list(read_jsonl(path / "loss.jsonl")) if (path / "loss.jsonl").exists() else []
        generation = score_predictions(records, predictions)
        loss = aggregate_losses(records, losses)
        generation_rows[stage] = predictions
        loss_summaries[stage] = loss
        scopes[stage] = {"generation": {k: v for k, v in generation.items() if k != "per_example"}, "loss": {k: v for k, v in loss.items() if k != "per_example"}}
        runtime[stage] = {"generation": runtime_summary(predictions), "teacher_forced_loss": runtime_summary(losses), "metadata": optional_json(path / "metadata.json")}
        atomic_write_jsonl(path / "metrics.per_example.jsonl", generation["per_example"])
        atomic_write_jsonl(path / "loss.metrics.per_example.jsonl", loss["per_example"])
    selection = select_checkpoint(loss_summaries)
    checkpoints = optional_json(run / "training/checkpoint_index.json")
    if checkpoints is not None and selection["selected_stage"]:
        selection["checkpoint_index_entry"] = checkpoints.get(selection["selected_stage"])
    blind_path = None
    selected = selection["selected_stage"]
    if selected:
        blind_path = write_blind_review(run, dataset, records, review_ids, generation_rows["base"], generation_rows[selected], selected)
    training_runtime = optional_json(run / "training/runtime.json")
    report = {"metric_protocol": protocol(), "frozen_evaluation_sha256": sha256_file(frozen_path), "validation_dataset": str(dataset), "validation_records": len(records), "validation_explain": sum(r["task"] == "explain" for r in records), "validation_track": sum(r["task"] == "track" for r in records), "stages": scopes, "checkpoint_selection": selection, "runtime": runtime, "training_runtime": training_runtime, "blind_review_csv": str(blind_path) if blind_path else None, "human_reason_label_review": "pending", "semantic_quality_measured": False, "test_predictions_generated_by_report": False}
    atomic_write_json(run / "metrics.json", report)
    lines = ["# Qwen3-VL-8B validation comparison", "", "All generation metrics below are structural or normalized lexical exact-match diagnostics. Semantic reason correctness and action-class accuracy are unmeasured. Missing, error, empty, malformed, length-finished and zero-token outputs remain in the fixed validation denominators. Human reason-label review remains pending.", "", f"Frozen validation: {report['validation_records']} records, including {report['validation_explain']} Explain and {report['validation_track']} Track. Goal diagnostics include only annotated goal targets. No test labels or predictions are read by this report.", "", "| Stage | Explain mean per-example target NLL | Track mean per-example target NLL | All loss rows valid | Strict output format |", "| --- | ---: | ---: | ---: | ---: |"]
    for stage in STAGES:
        loss = scopes[stage]["loss"]["scopes"]
        gen = scopes[stage]["generation"]["scopes"]["all"]
        def nll(scope):
            value = loss[scope]["mean_per_example_nll"]
            return f"{value:.6f}" if value is not None else "incomplete / unavailable"
        lines.append(f"| {stage} | {nll('explain')} | {nll('track')} | {loss['all']['valid_examples']}/{loss['all']['expected_examples']} | {gen['strict_format_valid']}/{gen['expected']} |")
    if selected:
        lines.extend(["", f"Selected **{selected}** by the lower complete-validation mean per-example Explain NLL ({selection['selected_metric']:.6f}); an exact tie selects epoch1. This averages complete assistant targets, including next actions and any supervised goal, rather than why-only tokens. Base is excluded from checkpoint selection; it remains the comparison baseline.", ""])
    else:
        lines.extend(["", "Checkpoint selection is pending complete finite validation loss results for both epoch1 and epoch2.", ""])
    lines.extend(["| Stage | WHY lexical exact match | Action lexical exact match | Goal lexical exact match | Ordered next triple exact match | Next position 1 | Next position 2 | Next position 3 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for stage in STAGES:
        lexical = scopes[stage]["generation"]["scopes"]["all"]["lexical_exact_match"]
        lines.append("| " + stage + " | " + " | ".join(_metric_cell(lexical[k]) for k in ("why", "action", "goal", "next_sequence", "next_position_1", "next_position_2", "next_position_3")) + " |")
    lines.extend(["", "A valid paraphrase can receive zero lexical exact match. Matching the annotation's words does not establish visual support or semantic quality. Strict format failure makes every requested-field lexical score zero, while the raw response and parser errors remain available for review. No LLM judge or verb/noun class mapping was applied.", "", "| Stage | Missing | Generation errors | Empty | Malformed | Finish reason length | Zero generated tokens |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for stage in STAGES:
        gen = scopes[stage]["generation"]["scopes"]["all"]
        lines.append("| " + stage + " | " + " | ".join(str(gen[k]) for k in ("missing_prediction", "generation_error", "empty_output", "malformed_output", "finish_length", "zero_generated_tokens")) + " |")
    lines.extend(["", "| Stage | WHY missing / empty | Action missing / empty | Goal missing / empty | Next missing / empty |", "| --- | ---: | ---: | ---: | ---: |"])
    for stage in STAGES:
        presence = scopes[stage]["generation"]["scopes"]["all"]["requested_field_diagnostics"]
        cells = []
        for field in ("why", "action", "goal", "next"):
            d = presence[field]
            cells.append(f"{d['missing']}/{d['denominator']} missing; {d['empty']}/{d['denominator']} empty" if d["denominator"] else "— (not supervised)")
        lines.append("| " + stage + " | " + " | ".join(cells) + " |")
    lines.extend(["", "A requested field is missing when no complete literal opening/closing tag pair exists. A complete whitespace-only pair is empty. Partial and malformed outputs remain in these counts, as do missing whole predictions. Goal field denominators contain only supervised goals. Duplicate complete fields are recorded in metrics.json and per-example diagnostics; they fail strict parsing."])
    lines.extend(["", "Diagnostic counts can overlap; they are not exclusions or a partition of failures. Raw saved outputs are under predictions/{base,epoch1,epoch2}/generation.jsonl. Per-example teacher-forced NLL, answer-token counts and NLL sums are under the corresponding loss.jsonl; normalized aggregation and parser checks are saved beside them.", "", "| Stage | Generation seconds (sum of rows) | Loss seconds (sum of rows) | Peak reported GPU MiB |", "| --- | ---: | ---: | ---: |"])
    for stage in STAGES:
        gen, loss = runtime[stage]["generation"], runtime[stage]["teacher_forced_loss"]
        peaks = [x["peak_reported_gpu_memory_mib"] for x in (gen, loss) if x["peak_reported_gpu_memory_mib"] is not None]
        peak = f"{max(peaks):.1f}" if peaks else "unrecorded"
        lines.append(f"| {stage} | {gen['sum_row_elapsed_sec']:.3f} | {loss['sum_row_elapsed_sec']:.3f} | {peak} |")
    lines.extend(["", "The timing table sums saved per-example times. Model loading, training and evaluation wall times belong in the separate runtime metadata when available; these sums alone do not measure end-to-end runtime.", ""])
    if blind_path:
        lines.extend([f"The fixed seed {BLIND_REVIEW_SEED} review CSV contains 50 validation Explain IDs chosen before predictions, paired base and selected-checkpoint WHY answers in independently randomized A/B order, source reasons and input references. Reviewer columns are blank unless prior human entries already exist. Missing or malformed answers are retained. Keep review/blind_why_review_identity_key.json separate from the blinded reviewer CSV.", ""])
    else:
        lines.extend(["The frozen 50-ID blind review sample is saved in frozen_evaluation.json. The paired review CSV will be written after checkpoint selection; no human comparison score is claimed.", ""])
    atomic_write_text(run / "validation_report.md", "\n".join(lines))
    print(json.dumps({"report": str(run / "validation_report.md"), "selected_stage": selected, "validation_records": len(records), "blind_review_csv": str(blind_path) if blind_path else None}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    build_report(args.run, args.dataset, args.frozen_evaluation)


if __name__ == "__main__":
    main()
