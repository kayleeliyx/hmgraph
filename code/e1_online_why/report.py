"""Metrics and concise artifacts for E1."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from code.e0.dataset import atomic_write_json, atomic_write_text


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _usage_totals(traces: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for trace in traces:
        for step in trace["steps"]:
            usage = step.get("token_usage") or {}
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] += value
    return dict(sorted(totals.items()))


def generate_report(
    run_root: Path,
    queries: Sequence[Mapping[str, Any]],
    scoring_labels: Mapping[str, Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    reference_c1: Mapping[str, Any],
) -> dict[str, Any]:
    report_root = run_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    trace_by_id = {str(trace["question_id"]): trace for trace in traces}
    rows: list[dict[str, Any]] = []
    for query in queries:
        question_id = str(query["question_id"])
        trace = trace_by_id.get(question_id)
        if trace is None:
            continue
        gold = str(scoring_labels[question_id]["gold_letter"])
        requested_intervals = [
            [batch["start_sec"], batch["end_sec"]] for batch in trace["frame_batches"][1:]
        ]
        row = {
            "question_id": question_id,
            "question": query["question"],
            "gold_answer": gold,
            "gold_answer_text": query["displayed_choices"]["ABCDE".index(gold)],
            "initial_answer": trace["initial_answer"],
            "final_answer": trace["final_answer"],
            "initial_correct": trace["initial_answer"] == gold,
            "final_correct": trace["final_answer"] == gold,
            "changed": trace["initial_answer"] != trace["final_answer"],
            "additional_reads": trace["additional_reads"],
            "requested_intervals": requested_intervals,
            "trace_path": str((run_root / "traces" / f"{question_id}.json").resolve()),
        }
        rows.append(row)

    completed = len(rows)
    incomplete_question_ids = [
        str(query["question_id"])
        for query in queries
        if str(query["question_id"]) not in trace_by_id
    ]
    initial_correct = sum(int(row["initial_correct"]) for row in rows)
    final_correct = sum(int(row["final_correct"]) for row in rows)
    wrong_to_correct = sum(
        int(not row["initial_correct"] and row["final_correct"]) for row in rows
    )
    correct_to_wrong = sum(
        int(row["initial_correct"] and not row["final_correct"]) for row in rows
    )
    read_histogram = Counter(int(row["additional_reads"]) for row in rows)
    all_frames = [
        (trace["source_video_id"], float(frame["timestamp_sec"]), frame["sha256"])
        for trace in traces
        for batch in trace["frame_batches"]
        for frame in batch["frames"]
    ]
    step_latencies = [
        float(step["latency_seconds"])
        for trace in traces
        for step in trace["steps"]
        if step.get("latency_seconds") is not None
    ]
    query_model_latencies = [
        sum(float(step.get("latency_seconds") or 0.0) for step in trace["steps"])
        for trace in traces
    ]
    execution_wall_latencies = [
        float(trace["execution_wall_latency_seconds"])
        for trace in traces
        if trace.get("execution_wall_latency_seconds") is not None
    ]
    unique_response_hashes = {
        str(step["request_hash"]) for trace in traces for step in trace["steps"]
    }
    metrics = {
        "protocol_description": (
            "Evaluation at annotated action-end times, with model-directed access to past video."
        ),
        "denominator": len(queries),
        "completed": completed,
        "failed_or_incomplete": len(queries) - completed,
        "incomplete_question_ids": incomplete_question_ids,
        "initial": {
            "correct": initial_correct,
            "accuracy": initial_correct / completed if completed else None,
        },
        "final": {
            "correct": final_correct,
            "accuracy": final_correct / completed if completed else None,
        },
        "paired_changes": {
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "answer_changed": sum(int(row["changed"]) for row in rows),
        },
        "additional_reads_histogram": {
            str(index): read_histogram.get(index, 0) for index in range(4)
        },
        "frames": {
            "total_presentations": len(all_frames),
            "unique_source_frames": len(set(all_frames)),
            "frame_batches": sum(len(trace["frame_batches"]) for trace in traces),
            "frame_batch_cache_hits_this_execution": sum(
                int(trace.get("frame_cache_hits_this_execution", 0)) for trace in traces
            ),
            "contact_sheet_images_presented": sum(
                len(batch.get("contact_sheets", []))
                for trace in traces
                for batch in trace["frame_batches"]
            ),
            "contact_sheet_cache_hits_this_execution": sum(
                int(trace.get("contact_sheet_cache_hits_this_execution", 0))
                for trace in traces
            ),
        },
        "model": {
            "decision_steps": sum(len(trace["steps"]) for trace in traces),
            "unique_canonical_provider_responses": len(unique_response_hashes),
            "provider_calls_this_execution": sum(
                int(trace.get("provider_calls_this_execution", 0)) for trace in traces
            ),
            "response_cache_hits_this_execution": sum(
                int(trace.get("response_cache_hits_this_execution", 0)) for trace in traces
            ),
            "token_usage": _usage_totals(traces),
        },
        "latency_seconds": {
            "per_model_call_mean": statistics.mean(step_latencies) if step_latencies else None,
            "per_model_call_median": statistics.median(step_latencies) if step_latencies else None,
            "per_query_model_mean": statistics.mean(query_model_latencies)
            if query_model_latencies
            else None,
            "per_query_model_median": statistics.median(query_model_latencies)
            if query_model_latencies
            else None,
            "per_query_model_p90": _percentile(query_model_latencies, 0.9),
            "current_execution_wall_mean": statistics.mean(execution_wall_latencies)
            if execution_wall_latencies
            else None,
            "current_execution_wall_median": statistics.median(execution_wall_latencies)
            if execution_wall_latencies
            else None,
        },
        "invalid_request_terminations": sum(
            int(trace.get("terminated_by_invalid_request", False)) for trace in traces
        ),
        "failures": list(failures),
        "e0_c1_reference": dict(reference_c1),
    }
    atomic_write_json(report_root / "metrics.json", metrics)

    csv_path = report_root / "item_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "question_id",
            "gold_answer",
            "gold_answer_text",
            "initial_answer",
            "final_answer",
            "initial_correct",
            "final_correct",
            "changed",
            "additional_reads",
            "requested_intervals",
            "trace_path",
            "question",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["requested_intervals"] = json.dumps(output["requested_intervals"])
            writer.writerow(output)

    changed = [row for row in rows if row["changed"]][:5]
    examples: list[str] = ["# Changed-answer examples", ""]
    if not changed:
        examples.append("No initial-to-final answers changed.")
    for row in changed:
        trace = trace_by_id[row["question_id"]]
        examples.extend(
            [
                f"## {row['question_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Gold: {row['gold_answer']} — {row['gold_answer_text']}",
                f"- Initial → final: {row['initial_answer']} → {row['final_answer']}",
                f"- Trace: `{row['trace_path']}`",
                "- Requested intervals and frames:",
            ]
        )
        for batch in trace["frame_batches"][1:]:
            frames = ", ".join(
                f"{float(frame['timestamp_sec']):.3f} (`{frame['path']}`)"
                for frame in batch["frames"]
            )
            examples.append(
                f"  - [{float(batch['start_sec']):.3f}, {float(batch['end_sec']):.3f}]: {frames}"
            )
        examples.append("")
    atomic_write_text(report_root / "changed_examples.md", "\n".join(examples).rstrip() + "\n")

    def pct(block: Mapping[str, Any]) -> str:
        return "unavailable" if block["accuracy"] is None else f"{100*block['accuracy']:.1f}%"

    summary = f"""# E1 online why-recognition at annotated action end

E1 evaluates why-recognition at each annotated action-end time, with model-directed access to any past video in `[0, t]`. The initial view is always 16 uniformly sampled frames from the fixed five seconds ending at `t`; it is independent of the annotated action start.

## Results

- Completed: {completed}/{len(queries)}; failed or incomplete: {len(queries) - completed}.
- Initial fixed-five-second accuracy: {initial_correct}/{completed} ({pct(metrics['initial'])}).
- Final accuracy after optional retrieval: {final_correct}/{completed} ({pct(metrics['final'])}).
- Wrong → correct: {wrong_to_correct}; correct → wrong: {correct_to_wrong}.
- Additional reads (0/1/2/3): {metrics['additional_reads_histogram']}.
- Frames: {metrics['frames']['total_presentations']} presentations, {metrics['frames']['unique_source_frames']} unique source frames, transported in {metrics['frames']['contact_sheet_images_presented']} timestamp-labeled 2×2 contact-sheet images.
- Model decision steps / canonical responses: {metrics['model']['decision_steps']} / {metrics['model']['unique_canonical_provider_responses']}; provider calls this execution: {metrics['model']['provider_calls_this_execution']}; response-cache hits: {metrics['model']['response_cache_hits_this_execution']}.
- API usage: `{json.dumps(metrics['model']['token_usage'], sort_keys=True)}`.
- Stored provider-model latency per query — mean: {metrics['latency_seconds']['per_query_model_mean']} seconds; median: {metrics['latency_seconds']['per_query_model_median']} seconds; p90: {metrics['latency_seconds']['per_query_model_p90']} seconds. Current execution wall latency is separately recorded in `metrics.json` because cache hits change it.
- Failed/incomplete IDs: `{json.dumps(metrics['incomplete_question_ids'])}`.

## Reference

E0 C1 scored 83/100 (83.0%). Unlike E1, E0 C1 was explicitly given frames sampled from the annotation-selected action clip, including its hidden start boundary.

## Scope

This is a fixed-budget development pilot at annotated action-end times. Up to three additional reads are allowed; it is not an estimate of the minimum history required. Gold answers and annotated action starts remain outside all model inputs.
"""
    atomic_write_text(report_root / "summary.md", summary)
    return metrics
