"""Scoring and concise reporting for local action-clip and E1 conditions."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from code.e0.dataset import atomic_write_json, atomic_write_text


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def generate_report(
    run_root: Path,
    queries: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    action_traces: Mapping[str, Mapping[str, Any]],
    e1_traces: Mapping[str, Mapping[str, Any]],
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report_root = run_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for query in queries:
        question_id = str(query["question_id"])
        gold = str(labels[question_id]["gold_letter"])
        action = action_traces.get(question_id)
        e1 = e1_traces.get(question_id)
        action_answer = action.get("answer") if action else None
        initial = e1.get("initial_answer") if e1 else None
        final = e1.get("final_answer") if e1 else None
        batches = list(e1.get("frame_batches", [])) if e1 else []
        rows.append(
            {
                "question_id": question_id,
                "question": query["question"],
                "displayed_choices": list(query["displayed_choices"]),
                "option_order": list(query["option_order"]),
                "gold_answer": gold,
                "gold_answer_text": query["displayed_choices"]["ABCDE".index(gold)],
                "local_action_clip_answer": action_answer,
                "local_action_clip_correct": action_answer == gold,
                "local_e1_initial_answer": initial,
                "local_e1_initial_correct": initial == gold,
                "local_e1_final_answer": final,
                "local_e1_final_correct": final == gold,
                "changed": initial is not None and final is not None and initial != final,
                "additional_reads": int(e1.get("additional_reads", 0)) if e1 else 0,
                "requested_intervals": [
                    [float(batch["start_sec"]), float(batch["end_sec"])]
                    for batch in batches[1:]
                ],
                "action_error": action.get("parse_error") if action else "missing_trace",
                "e1_error": e1.get("terminal_error") if e1 else "missing_trace",
                "action_trace_path": str((run_root / "traces" / "local_action_clip" / f"{question_id}.json").resolve()),
                "e1_trace_path": str((run_root / "traces" / "local_e1" / f"{question_id}.json").resolve()),
            }
        )

    denominator = len(queries)
    action_correct = sum(int(row["local_action_clip_correct"]) for row in rows)
    initial_correct = sum(int(row["local_e1_initial_correct"]) for row in rows)
    final_correct = sum(int(row["local_e1_final_correct"]) for row in rows)
    wrong_to_correct = sum(
        int(not row["local_e1_initial_correct"] and row["local_e1_final_correct"])
        for row in rows
    )
    correct_to_wrong = sum(
        int(row["local_e1_initial_correct"] and not row["local_e1_final_correct"])
        for row in rows
    )
    histogram = Counter(int(row["additional_reads"]) for row in rows)
    action_failures = [row["question_id"] for row in rows if row["local_action_clip_answer"] is None]
    e1_failures = [row["question_id"] for row in rows if row["local_e1_initial_answer"] is None]
    e1_errors = [row["question_id"] for row in rows if row["e1_error"]]
    action_latencies = [
        float(trace["execution_wall_latency_seconds"])
        for trace in action_traces.values()
        if trace.get("execution_wall_latency_seconds") is not None
    ]
    e1_latencies = [
        float(trace["execution_wall_latency_seconds"])
        for trace in e1_traces.values()
        if trace.get("execution_wall_latency_seconds") is not None
    ]
    all_responses = [
        trace["response"] for trace in action_traces.values() if trace.get("response")
    ] + [
        step for trace in e1_traces.values() for step in trace.get("steps", [])
    ]
    peak_allocated = max(
        [int(response.get("peak_gpu_memory_allocated_bytes") or 0) for response in all_responses]
        + [
            int(item.get("peak_gpu_memory_allocated_bytes") or 0)
            for item in (preflight or {}).get("capacity", [])
        ]
        + [0]
    )
    peak_reserved = max(
        [int(response.get("peak_gpu_memory_reserved_bytes") or 0) for response in all_responses]
        + [
            int(item.get("peak_gpu_memory_reserved_bytes") or 0)
            for item in (preflight or {}).get("capacity", [])
        ]
        + [0]
    )
    metrics = {
        "protocol_description": "Evaluation at annotated action-end times, with model-directed access to past video.",
        "denominator": denominator,
        "local_action_clip": {
            "correct": action_correct,
            "accuracy": action_correct / denominator,
            "context": "annotation-selected action-clip reference using exact saved E0 C1 frames",
        },
        "local_e1_initial": {
            "correct": initial_correct,
            "accuracy": initial_correct / denominator,
            "context": "16 individual frames uniformly sampled from the fixed five seconds ending at t",
        },
        "local_e1_final": {
            "correct": final_correct,
            "accuracy": final_correct / denominator,
            "context": "same run after zero to three model-directed past-video reads",
        },
        "paired_changes": {
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "answer_changed": sum(int(row["changed"]) for row in rows),
            "net_correct_change": final_correct - initial_correct,
        },
        "additional_reads_histogram": {str(index): histogram.get(index, 0) for index in range(4)},
        "failures": {
            "local_action_clip_failed_count": len(action_failures),
            "local_action_clip_failed_ids": action_failures,
            "local_e1_failed_count": len(e1_failures),
            "local_e1_failed_ids": e1_failures,
            "local_e1_terminal_error_count": len(e1_errors),
            "local_e1_terminal_error_ids": e1_errors,
        },
        "latency_seconds": {
            "local_action_clip_end_to_end_mean": _mean(action_latencies),
            "local_action_clip_end_to_end_median": _median(action_latencies),
            "local_e1_end_to_end_mean": _mean(e1_latencies),
            "local_e1_end_to_end_median": _median(e1_latencies),
        },
        "gpu_memory": {
            "measurement": "peak observed per-GPU NVML device-used memory",
            "note": "the two assigned GPUs were idle before model launch",
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_allocated / (1024**3),
            "peak_reserved_gib": peak_reserved / (1024**3),
        },
        "work": {
            "e1_model_decisions": sum(len(trace.get("steps", [])) for trace in e1_traces.values()),
            "action_clip_model_decisions": len(action_traces),
            "preflight_model_decisions": len((preflight or {}).get("capacity", [])),
            "external_inference_api_calls": 0,
            "e1_frame_presentations": sum(
                len(batch.get("frames", []))
                for trace in e1_traces.values()
                for batch in trace.get("frame_batches", [])
            ),
            "action_clip_frame_presentations": sum(
                len(trace.get("frame_batch", {}).get("frames", []))
                for trace in action_traces.values()
            ),
        },
    }
    metrics["work"]["evaluation_model_decisions"] = (
        metrics["work"]["e1_model_decisions"]
        + metrics["work"]["action_clip_model_decisions"]
    )
    metrics["work"]["evaluation_frames_inspected"] = (
        metrics["work"]["e1_frame_presentations"]
        + metrics["work"]["action_clip_frame_presentations"]
    )
    atomic_write_json(report_root / "metrics.json", metrics)

    csv_path = report_root / "item_results.csv"
    fields = [
        "question_id",
        "gold_answer",
        "gold_answer_text",
        "local_action_clip_answer",
        "local_action_clip_correct",
        "local_e1_initial_answer",
        "local_e1_initial_correct",
        "local_e1_final_answer",
        "local_e1_final_correct",
        "changed",
        "additional_reads",
        "requested_intervals",
        "action_error",
        "e1_error",
        "option_order",
        "displayed_choices",
        "question",
        "action_trace_path",
        "e1_trace_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("requested_intervals", "action_error", "e1_error", "option_order", "displayed_choices"):
                output[key] = json.dumps(output[key], ensure_ascii=False)
            writer.writerow(output)

    changed = [row for row in rows if row["changed"]][:5]
    example_lines = ["# Changed-answer examples", ""]
    if not changed:
        example_lines.append("No local E1 initial-to-final answers changed.")
    for row in changed:
        trace = e1_traces[row["question_id"]]
        example_lines.extend(
            [
                f"## {row['question_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Gold: {row['gold_answer']} — {row['gold_answer_text']}",
                f"- Initial → final: {row['local_e1_initial_answer']} → {row['local_e1_final_answer']}",
                "- Requested intervals with individual source frames:",
            ]
        )
        for batch in trace["frame_batches"][1:]:
            frames = ", ".join(
                f"{float(frame['timestamp_sec']):.3f}s ({frame['path']})"
                for frame in batch["frames"]
            )
            example_lines.append(
                f"  - [{float(batch['start_sec']):.3f}, {float(batch['end_sec']):.3f}]: {frames}"
            )
        example_lines.append("")
    atomic_write_text(report_root / "changed_examples.md", "\n".join(example_lines).rstrip() + "\n")

    helped = final_correct > initial_correct
    table = f"""| Condition | Correct / 100 | Accuracy |
|---|---:|---:|
| Local annotation-selected action clip | {action_correct}/100 | {action_correct:.1f}% |
| Local E1 initial fixed five-second view | {initial_correct}/100 | {initial_correct:.1f}% |
| Local E1 final after optional retrieval | {final_correct}/100 | {final_correct:.1f}% |
"""
    summary = f"""# Local HD-EPIC why-recognition

This is an evaluation at annotated action-end times, with model-directed access to past video. The local E1 model initially sees 16 individual frames from `[max(0, t-5), t]` and may request at most three additional intervals from `[0, t]`.

## Results

{table}
- Wrong → correct: {wrong_to_correct}; correct → wrong: {correct_to_wrong}; net: {final_correct - initial_correct:+d}.
- Additional reads (0/1/2/3): `{json.dumps(metrics['additional_reads_histogram'], sort_keys=True)}`.
- Failed initial E1 items: {len(e1_failures)}; E1 traces terminated by malformed output or invalid request: {len(e1_errors)}.
- Local E1 end-to-end latency per question — mean: {metrics['latency_seconds']['local_e1_end_to_end_mean']} s; median: {metrics['latency_seconds']['local_e1_end_to_end_median']} s.
- Peak observed per-GPU device memory used: {metrics['gpu_memory']['peak_reserved_gib']:.2f} GiB.
- Evaluation work: {metrics['work']['evaluation_model_decisions']} local model decisions and {metrics['work']['evaluation_frames_inspected']} frames inspected; external inference API calls: 0. The capacity preflight used {metrics['work']['preflight_model_decisions']} additional local decisions.
- Changed-answer examples available: {sum(int(row['changed']) for row in rows)}. The report includes all available changed examples, up to five.

Retrieval {'helped' if helped else 'did not improve'} this local model by the measured accuracy criterion ({initial_correct}/100 → {final_correct}/100). The latency is measured offline end-to-end latency on this server; it does not by itself establish real-time operation.

## Historical GPT reference (not a controlled comparison)

E0 C1 with `azure/gpt-5.6-sol` scored 83/100 when given the annotation-selected action clip. That historical model, image presentation, and inference stack differ from this local run, so it is a reference only.

## Artifacts

- Per-item CSV: `{csv_path.resolve()}`
- Metrics: `{(report_root / 'metrics.json').resolve()}`
- Changed examples: `{(report_root / 'changed_examples.md').resolve()}`
- Traces: `{(run_root / 'traces').resolve()}`
- Exact prompts and raw outputs: `{(run_root / 'cache' / 'responses').resolve()}`
- Resolved configuration: `{(run_root / 'config.resolved.json').resolve()}`
- Preflight: `{(run_root / 'preflight').resolve()}`

Exact resume/rerun command:

```bash
cd /data/yaxuanli/ActGround && scripts/run_e1_local_why.sh --stage all
```
"""
    atomic_write_text(report_root / "summary.md", summary)
    return metrics
