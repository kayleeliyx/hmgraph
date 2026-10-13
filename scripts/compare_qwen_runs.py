#!/usr/bin/env python3
"""Compare W and L1 predictions from two runs against HD-EPIC ground truth.

The comparison is diagnostic only: predictions are never changed using GT.
Labels are canonicalized with the run's HD-EPIC alias table, and a prediction
matches a GT event only when their time intervals overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def norm(value: object) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def object_name(item: dict) -> str:
    rich = item.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name"):
        return norm(rich["name"])
    value = item.get("direct_object")
    if isinstance(value, dict):
        value = value.get("name")
    return norm(value)


def canonical(value: object, table: dict[str, str]) -> str:
    normalized = norm(value)
    return norm(table.get(normalized, normalized))


def pair(item: dict, aliases: dict) -> tuple[str, str]:
    return (
        canonical(item.get("verb"), aliases.get("verb_aliases", {})),
        canonical(object_name(item), aliases.get("noun_aliases", {})),
    )


def overlaps(left: dict, right: dict) -> bool:
    return min(float(left["t_end"]), float(right["t_end"])) > max(
        float(left["t_start"]), float(right["t_start"])
    )


def iou(left: dict, right: dict) -> float:
    intersection = max(
        0.0,
        min(float(left["t_end"]), float(right["t_end"]))
        - max(float(left["t_start"]), float(right["t_start"])),
    )
    union = max(float(left["t_end"]), float(right["t_end"])) - min(
        float(left["t_start"]), float(right["t_start"])
    )
    return intersection / union if union else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def round_metrics(metrics: dict) -> dict:
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def w_actions(records: list[dict]) -> list[dict]:
    actions = []
    for record in records:
        for action in record.get("valid_actions") or []:
            actions.append(
                {
                    **action,
                    "t_start": float(action.get("event_t_start", record["t_start"])),
                    "t_end": float(action.get("event_t_end", record["t_end"])),
                    "window_idx": int(record["window_idx"]),
                }
            )
    return actions


def event_metrics(predictions: list[dict], gt: list[dict], aliases: dict) -> tuple[dict, list[dict]]:
    pred_pairs = [pair(item, aliases) for item in predictions]
    gt_pairs = [pair(item, aliases) for item in gt]
    gt_rows = []
    exact_gt_matches = 0
    verb_gt_matches = 0
    object_gt_matches = 0
    best_exact_ious = []
    for index, event in enumerate(gt):
        candidates = [
            pred_index
            for pred_index, prediction in enumerate(predictions)
            if overlaps(prediction, event)
        ]
        exact = [pred_index for pred_index in candidates if pred_pairs[pred_index] == gt_pairs[index]]
        verb = [pred_index for pred_index in candidates if pred_pairs[pred_index][0] == gt_pairs[index][0]]
        obj = [pred_index for pred_index in candidates if pred_pairs[pred_index][1] == gt_pairs[index][1]]
        exact_gt_matches += bool(exact)
        verb_gt_matches += bool(verb)
        object_gt_matches += bool(obj)
        best_iou = max((iou(predictions[pred_index], event) for pred_index in exact), default=0.0)
        if exact:
            best_exact_ious.append(best_iou)
        gt_rows.append(
            {
                "gt_id": event.get("gt_id", index),
                "t_start": event["t_start"],
                "t_end": event["t_end"],
                "gt_verb": gt_pairs[index][0],
                "gt_object": gt_pairs[index][1],
                "exact_pair_match": bool(exact),
                "verb_match": bool(verb),
                "object_match": bool(obj),
                "best_exact_pair_iou": round(best_iou, 4),
            }
        )

    exact_pred_matches = sum(
        any(overlaps(prediction, event) and pred_pairs[index] == gt_pairs[gt_index]
            for gt_index, event in enumerate(gt))
        for index, prediction in enumerate(predictions)
    )
    precision = exact_pred_matches / len(predictions) if predictions else 0.0
    recall = exact_gt_matches / len(gt) if gt else 0.0
    metrics = {
        "predictions": len(predictions),
        "gt_events": len(gt),
        "exact_pair_matched_predictions": exact_pred_matches,
        "exact_pair_matched_gt_events": exact_gt_matches,
        "exact_pair_precision": precision,
        "exact_pair_recall": recall,
        "exact_pair_f1": f1(precision, recall),
        "verb_recall": verb_gt_matches / len(gt) if gt else 0.0,
        "object_recall": object_gt_matches / len(gt) if gt else 0.0,
        "mean_best_iou_for_exact_matches": (
            sum(best_exact_ious) / len(best_exact_ious) if best_exact_ious else 0.0
        ),
    }
    return round_metrics(metrics), gt_rows


def window_presence_metrics(records: list[dict], gt: list[dict]) -> dict:
    gt_positive = 0
    pred_positive = 0
    both = 0
    for record in records:
        gt_here = any(overlaps(record, event) for event in gt)
        pred_here = bool(record.get("valid_actions"))
        gt_positive += gt_here
        pred_positive += pred_here
        both += gt_here and pred_here
    precision = both / pred_positive if pred_positive else 0.0
    recall = both / gt_positive if gt_positive else 0.0
    return round_metrics(
        {
            "windows": len(records),
            "gt_positive_windows": gt_positive,
            "predicted_positive_windows": pred_positive,
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": f1(precision, recall),
        }
    )


def evaluate_run(
    name: str,
    w_path: Path,
    l1_path: Path,
    gt: list[dict],
    aliases: dict,
) -> tuple[dict, list[dict]]:
    w_records = read_jsonl(w_path)
    l1_events = read_jsonl(l1_path)
    w_result, w_rows = event_metrics(w_actions(w_records), gt, aliases)
    w_result.update(window_presence_metrics(w_records, gt))
    l1_result, l1_rows = event_metrics(l1_events, gt, aliases)
    rows = []
    for level, source_rows in (("W", w_rows), ("L1", l1_rows)):
        rows.extend({"run": name, "level": level, **row} for row in source_rows)
    return {
        "name": name,
        "w_path": str(w_path.resolve()),
        "l1_path": str(l1_path.resolve()),
        "W": w_result,
        "L1": l1_result,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-w", type=Path, required=True)
    parser.add_argument("--baseline-l1", type=Path, required=True)
    parser.add_argument("--candidate-w", type=Path, required=True)
    parser.add_argument("--candidate-l1", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--alias-table", type=Path, required=True)
    parser.add_argument("--baseline-name", default="previous_qwen")
    parser.add_argument("--candidate-name", default="rerun_qwen")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    aliases = json.loads(args.alias_table.read_text(encoding="utf-8"))
    gt = read_jsonl(args.gt)
    baseline, baseline_rows = evaluate_run(
        args.baseline_name, args.baseline_w, args.baseline_l1, gt, aliases
    )
    candidate, candidate_rows = evaluate_run(
        args.candidate_name, args.candidate_w, args.candidate_l1, gt, aliases
    )
    payload = {
        "metric_note": (
            "Strict canonical verb+object matching with positive temporal overlap. "
            "These automatic GT metrics are diagnostic and do not measure synonyms "
            "missing from the HD-EPIC alias table."
        ),
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            level: {
                metric: round(candidate[level][metric] - baseline[level][metric], 4)
                for metric in ("exact_pair_precision", "exact_pair_recall", "exact_pair_f1")
            }
            for level in ("W", "L1")
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = baseline_rows + candidate_rows
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
