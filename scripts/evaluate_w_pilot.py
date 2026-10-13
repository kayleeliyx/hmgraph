#!/usr/bin/env python3
"""Compare two W JSONL files against overlapping GT actions by 1s window."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: object) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def object_name(action: dict) -> str:
    rich = action.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name"):
        return norm(rich["name"])
    value = action.get("direct_object")
    if isinstance(value, dict):
        value = value.get("name")
    return norm(value)


def aliases(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("verb_aliases", {}), payload.get("noun_aliases", {})


def canonical(value: str, table: dict[str, str]) -> str:
    return norm(table.get(norm(value), norm(value)))


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) > max(a_start, b_start)


def score_model(
    label: str,
    records: list[dict],
    gt: list[dict],
    verb_aliases: dict[str, str],
    noun_aliases: dict[str, str],
    start: float,
    end: float,
) -> tuple[dict, list[dict]]:
    selected = [
        record
        for record in records
        if float(record.get("t_start", -1)) >= start
        and float(record.get("t_end", -1)) <= end
    ]
    rows = []
    for record in selected:
        t_start, t_end = float(record["t_start"]), float(record["t_end"])
        gt_here = [
            event
            for event in gt
            if overlap(t_start, t_end, float(event["t_start"]), float(event["t_end"]))
        ]
        predictions = record.get("valid_actions") or []
        gt_pairs = {
            (
                canonical(event.get("verb", ""), verb_aliases),
                canonical(event.get("direct_object", ""), noun_aliases),
            )
            for event in gt_here
        }
        pred_pairs = {
            (
                canonical(action.get("verb", ""), verb_aliases),
                canonical(object_name(action), noun_aliases),
            )
            for action in predictions
        }
        verb_match = bool({pair[0] for pair in gt_pairs} & {pair[0] for pair in pred_pairs})
        object_match = bool({pair[1] for pair in gt_pairs} & {pair[1] for pair in pred_pairs})
        pair_match = bool(gt_pairs & pred_pairs)
        rows.append(
            {
                "model": label,
                "window_idx": int(record["window_idx"]),
                "t_start": t_start,
                "gt": "; ".join(f"{v} {o}" for v, o in sorted(gt_pairs)),
                "prediction": "; ".join(f"{v} {o}" for v, o in sorted(pred_pairs)),
                "gt_present": bool(gt_here),
                "prediction_present": bool(predictions),
                "verb_match": verb_match,
                "object_match": object_match,
                "pair_match": pair_match,
                "parse_ok": bool(record.get("parse_ok")),
            }
        )

    n = len(rows)
    gt_rows = [row for row in rows if row["gt_present"]]
    pred_rows = [row for row in rows if row["prediction_present"]]
    both_presence = [
        row for row in rows if row["gt_present"] and row["prediction_present"]
    ]
    true_presence = len(both_presence)
    precision = true_presence / len(pred_rows) if pred_rows else 0.0
    recall = true_presence / len(gt_rows) if gt_rows else 0.0
    presence_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    summary = {
        "model": label,
        "windows": n,
        "gt_positive_windows": len(gt_rows),
        "predicted_positive_windows": len(pred_rows),
        "parse_json_rate": round(sum(row["parse_ok"] for row in rows) / n, 4) if n else 0.0,
        "presence_precision": round(precision, 4),
        "presence_recall": round(recall, 4),
        "presence_f1": round(presence_f1, 4),
        "verb_window_recall": round(sum(row["verb_match"] for row in gt_rows) / len(gt_rows), 4) if gt_rows else 0.0,
        "object_window_recall": round(sum(row["object_match"] for row in gt_rows) / len(gt_rows), 4) if gt_rows else 0.0,
        "verb_object_window_recall": round(sum(row["pair_match"] for row in gt_rows) / len(gt_rows), 4) if gt_rows else 0.0,
        "mean_inference_time_sec": round(
            sum(float(record.get("inference_time_sec", 0.0)) for record in selected) / n,
            4,
        ) if n else 0.0,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--alias-table", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    verb_aliases, noun_aliases = aliases(args.alias_table)
    gt = read_jsonl(args.gt)
    candidate_summary, candidate_rows = score_model(
        "LLaVAction-7B",
        read_jsonl(args.candidate),
        gt,
        verb_aliases,
        noun_aliases,
        args.start_sec,
        args.end_sec,
    )
    baseline_summary, baseline_rows = score_model(
        "Qwen3-VL-30B-A3B-Instruct",
        read_jsonl(args.baseline),
        gt,
        verb_aliases,
        noun_aliases,
        args.start_sec,
        args.end_sec,
    )
    payload = {
        "range": {"start_sec": args.start_sec, "end_sec": args.end_sec},
        "metric_note": (
            "Strict per-window canonical HD-EPIC alias matching against every GT action "
            "that temporally overlaps the 1-second W window."
        ),
        "models": [candidate_summary, baseline_summary],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = list((candidate_rows or baseline_rows)[0])
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows + baseline_rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
