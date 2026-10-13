"""Metrics, concise summary, CSV, and static review page for E0."""

from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import LETTERS
from .dataset import atomic_write_json, atomic_write_text

CONDITIONS = ("C0", "C1", "C2", "C3")


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["question_id"])
        if question_id in rows:
            raise ValueError(f"duplicate prediction {question_id} at {path}:{line_number}")
        rows[question_id] = row
    return rows


def _correct(row: Mapping[str, Any] | None, gold: str) -> bool:
    return bool(row and row.get("request_status") == "completed" and row.get("parsed_answer") == gold)


def _condition_metrics(
    records: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]], predictions: Mapping[str, Any]
) -> dict[str, Any]:
    total = len(records)
    completed = 0
    parsed = 0
    correct = 0
    for record in records:
        question_id = str(record["question_id"])
        row = predictions.get(question_id)
        if row and row.get("request_status") == "completed":
            completed += 1
            parsed += int(row.get("parsed_answer") in LETTERS)
            correct += int(_correct(row, str(labels[question_id]["gold_letter"])))
    parse_failures = completed - parsed
    missing = total - completed
    return {
        "expected": total,
        "completed": completed,
        "parsed": parsed,
        "parse_failures": parse_failures,
        "missing": missing,
        "correct": correct,
        "accuracy": correct / total if completed == total else None,
        "status": "complete" if completed == total else "incomplete",
    }


def _paired(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, int]:
    counts = Counter()
    for record in records:
        question_id = str(record["question_id"])
        left = baseline.get(question_id)
        right = comparison.get(question_id)
        if not left or not right or left.get("request_status") != "completed" or right.get("request_status") != "completed":
            counts["missing_pair"] += 1
            continue
        gold = str(labels[question_id]["gold_letter"])
        left_correct = _correct(left, gold)
        right_correct = _correct(right, gold)
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif not left_correct and right_correct:
            counts["baseline_wrong_comparison_correct"] += 1
        elif left_correct and not right_correct:
            counts["baseline_correct_comparison_wrong"] += 1
        else:
            counts["both_wrong"] += 1
        counts["paired_denominator"] += 1
    return {
        key: counts[key]
        for key in (
            "paired_denominator",
            "both_correct",
            "baseline_wrong_comparison_correct",
            "baseline_correct_comparison_wrong",
            "both_wrong",
            "missing_pair",
        )
    }


def _facts_text(record: Mapping[str, Any] | None) -> str:
    return " | ".join(str(value) for value in (record or {}).get("facts", []))


def _source_refs(record: Mapping[str, Any] | None) -> str:
    references: list[str] = []
    seen: set[str] = set()
    for source in (record or {}).get("fact_provenance", []):
        url = str(source.get("url") or "")
        title = str(source.get("title") or url)
        identity = f"{title} <{url}>"
        if url and identity not in seen:
            seen.add(identity)
            references.append(identity)
    return " | ".join(references)


def _build_item_rows(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    frames: Mapping[str, Mapping[str, Any]],
    c2: Mapping[str, Mapping[str, Any]],
    c3: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record["question_id"])
        gold = str(labels[question_id]["gold_letter"])
        row: dict[str, Any] = {
            "question_id": question_id,
            "participant": record["participant"],
            "source_video_id": record["source_video_id"],
            "source_video_path": record["source_video_path"],
            "start_sec": record["start_sec"],
            "end_sec": record["end_sec"],
            "target_action": record["action_phrase"],
            "question": record["question"],
            "displayed_choices": " | ".join(
                f"{letter}. {choice}" for letter, choice in zip(LETTERS, record["displayed_choices"])
            ),
            "gold_letter": gold,
        }
        for condition in CONDITIONS:
            prediction = predictions[condition].get(question_id)
            row[f"{condition}_prediction"] = prediction.get("parsed_answer") if prediction else ""
            row[f"{condition}_correct"] = _correct(prediction, gold) if prediction else ""
            row[f"{condition}_parse_failure"] = bool(
                prediction and prediction.get("error_category") == "parse_failure"
            )
            row[f"{condition}_request_status"] = prediction.get("request_status", "missing") if prediction else "missing"
        c2_record = c2.get(question_id)
        c3_record = c3.get(question_id)
        frame_record = frames.get(question_id)
        row.update(
            {
                "C2_facts": _facts_text(c2_record),
                "C3_facts": _facts_text(c3_record),
                "C3_retrieval_status": (c3_record or {}).get("status", "missing"),
                "C3_external_sources": _source_refs(c3_record),
                "frame_record_path": (frame_record or {}).get("_record_path", ""),
                "ordered_frame_payload_hash": (frame_record or {}).get("ordered_frame_payload_hash", ""),
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_review_annotations(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "question_id",
        "clip_boundary_issue",
        "answer_ambiguity",
        "clip_appears_sufficient",
        "C2_fact_accuracy",
        "C2_fact_relevance_applicability",
        "C3_fact_accuracy",
        "C3_fact_relevance_applicability",
        "notes",
    ]
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("question_id"):
                    existing[row["question_id"]] = row
    rows: list[dict[str, str]] = []
    for record in records:
        question_id = str(record["question_id"])
        previous = existing.get(question_id, {})
        rows.append({field: question_id if field == "question_id" else previous.get(field, "") for field in fields})
    _write_csv(path, rows)


def _review_html(
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    frames: Mapping[str, Mapping[str, Any]],
    c2: Mapping[str, Mapping[str, Any]],
    c3: Mapping[str, Mapping[str, Any]],
) -> None:
    sections: list[str] = []
    for number, record in enumerate(records, 1):
        question_id = str(record["question_id"])
        gold = str(labels[question_id]["gold_letter"])
        choices = "".join(
            f"<li><b>{letter}.</b> {html.escape(str(choice))}</li>"
            for letter, choice in zip(LETTERS, record["displayed_choices"])
        )
        prediction_rows = []
        for condition in CONDITIONS:
            prediction = predictions[condition].get(question_id)
            parsed = prediction.get("parsed_answer") if prediction else None
            status = prediction.get("request_status") if prediction else "missing"
            result_class = "correct" if _correct(prediction, gold) else "incorrect"
            prediction_rows.append(
                f"<tr><th>{condition}</th><td class='{result_class}'>{html.escape(str(parsed or '—'))}</td>"
                f"<td>{html.escape(str(status))}</td><td>{html.escape(str((prediction or {}).get('error_category') or ''))}</td></tr>"
            )
        frame_cards = []
        for frame in (frames.get(question_id) or {}).get("frames", []):
            relative = os.path.relpath(frame["path"], output_path.parent)
            frame_cards.append(
                f"<figure><img loading='lazy' src='{html.escape(relative)}' alt='frame at {frame['timestamp_sec']:.6f}s'>"
                f"<figcaption>{frame['timestamp_sec']:.6f}s</figcaption></figure>"
            )
        c2_items = "".join(f"<li>{html.escape(str(fact))}</li>" for fact in (c2.get(question_id) or {}).get("facts", [])) or "<li>Unavailable</li>"
        c3_record = c3.get(question_id) or {}
        c3_items = []
        provenance = c3_record.get("fact_provenance", [])
        for index, fact in enumerate(c3_record.get("facts", [])):
            source = provenance[index] if index < len(provenance) else {}
            link = (
                f" <a href='{html.escape(str(source.get('url')))}'>[{html.escape(str(source.get('title') or 'source'))}]</a>"
                if source.get("url")
                else ""
            )
            c3_items.append(f"<li>{html.escape(str(fact))}{link}</li>")
        c3_html = "".join(c3_items) or f"<li>Unavailable ({html.escape(str(c3_record.get('status', 'missing')))})</li>"
        frame_record_path = (frames.get(question_id) or {}).get("_record_path", "")
        sections.append(
            f"""<section id="{html.escape(question_id)}">
<h2>{number}. {html.escape(question_id)} — {html.escape(str(record['action_phrase']))}</h2>
<p><b>Participant/video:</b> {html.escape(str(record['participant']))} / {html.escape(str(record['source_video_id']))}<br>
<b>Source:</b> <code>{html.escape(str(record['source_video_path']))}</code><br>
<b>Official interval:</b> {record['start_sec']:.6f}–{record['end_sec']:.6f}s<br>
<b>Frame record:</b> <code>{html.escape(str(frame_record_path))}</code></p>
<p><b>Question:</b> {html.escape(str(record['question']))}</p>
<ol class="choices">{choices}</ol><p><b>Correct answer:</b> {gold}</p>
<div class="frames">{''.join(frame_cards) or '<p>No frames cached.</p>'}</div>
<table><thead><tr><th>Condition</th><th>Prediction</th><th>Status</th><th>Flag</th></tr></thead><tbody>{''.join(prediction_rows)}</tbody></table>
<div class="knowledge"><div><h3>C2 background facts</h3><ul>{c2_items}</ul></div><div><h3>C3 background facts</h3><p>Status: {html.escape(str(c3_record.get('status', 'missing')))}</p><ul>{c3_html}</ul></div></div>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HD-EPIC Why Recognition E0 Review</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:auto;padding:20px;color:#1b1b1b}}section{{border-top:3px solid #333;padding:20px 0}}code{{overflow-wrap:anywhere}}.choices{{columns:2}}.frames{{display:flex;gap:8px;overflow-x:auto;background:#eee;padding:8px}}figure{{margin:0;min-width:180px}}img{{width:180px;height:180px;object-fit:contain;background:#111}}figcaption{{text-align:center;font-size:12px}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:5px 9px}}.correct{{background:#d7f5dc}}.incorrect{{background:#ffe1e1}}.knowledge{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}@media(max-width:800px){{.knowledge{{grid-template-columns:1fr}}.choices{{columns:1}}}}
</style></head><body><h1>HD-EPIC Why Recognition E0 Review</h1>
<p>Automated experiment output only. No human audit is claimed. Fill <code>review_annotations.csv</code> separately.</p>
{''.join(sections)}</body></html>"""
    atomic_write_text(output_path, document)


def generate_reports(
    analysis_root: Path,
    run_root: Path,
    split: Mapping[str, Any],
    manifest: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    c2: Mapping[str, Mapping[str, Any]],
    c3: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    analysis_root.mkdir(parents=True, exist_ok=True)
    records = split["records"]
    labels = split["scoring_labels"]
    predictions = {
        condition: read_predictions(run_root / "predictions" / f"{condition}.jsonl")
        for condition in CONDITIONS
    }
    condition_metrics = {
        condition: _condition_metrics(records, labels, predictions[condition]) for condition in CONDITIONS
    }
    comparisons = {
        "C2_vs_C1": _paired(records, labels, predictions["C1"], predictions["C2"]),
        "C3_vs_C1": _paired(records, labels, predictions["C1"], predictions["C3"]),
        "C3_vs_C2": _paired(records, labels, predictions["C2"], predictions["C3"]),
    }
    c3_statuses = Counter((c3.get(str(record["question_id"])) or {}).get("status", "missing") for record in records)
    metrics = {
        "experiment": "HD-EPIC Why Recognition pilot",
        "expected_items": len(records),
        "condition_metrics": condition_metrics,
        "paired_comparisons": comparisons,
        "C3_retrieval_status_counts": dict(sorted(c3_statuses.items())),
        "participant_distribution": split["participant_distribution"],
        "annotation": manifest["annotation"],
    }
    atomic_write_json(analysis_root / "metrics.json", metrics)
    item_rows = _build_item_rows(records, labels, predictions, frames, c2, c3)
    _write_csv(analysis_root / "item_results.csv", item_rows)
    _write_review_annotations(analysis_root / "review_annotations.csv", records)
    _review_html(analysis_root / "review.html", records, labels, predictions, frames, c2, c3)

    changed: list[str] = []
    for record in records:
        question_id = str(record["question_id"])
        values = {condition: predictions[condition].get(question_id, {}).get("parsed_answer") for condition in CONDITIONS}
        if len({value for value in values.values() if value is not None}) > 1:
            gold = labels[question_id]["gold_letter"]
            changed.append(
                f"- `{question_id}` ({record['action_phrase']}), gold {gold}: "
                + ", ".join(f"{condition}={values[condition] or 'missing'}" for condition in CONDITIONS)
            )
        if len(changed) >= 8:
            break
    if not changed:
        changed = ["- No changed predictions are available in the completed records."]

    model = resolved_config["model"]
    frame = resolved_config["frames"]
    sample_size = len(records)
    participant_count = len(split["participant_distribution"])
    base_per_participant, remainder = divmod(sample_size, participant_count)
    lines = [
        "# HD-EPIC Why Recognition Pilot (E0)",
        "",
        "## Experimental question",
        "",
        "For the official HD-EPIC action clip, does supplying background knowledge change multiple-choice why-recognition accuracy? This is a participant-balanced exploratory development pilot, not an estimate under the benchmark's natural participant distribution.",
        "",
        "## Data and protocol",
        "",
        f"- Official annotation: `{manifest['annotation']['path']}` ({manifest['annotation']['item_count']} items; SHA-256 `{manifest['annotation']['sha256']}`).",
        f"- Saved development sample: {sample_size} IDs, seed {split['seed']}; participant distribution: `{json.dumps(split['participant_distribution'], sort_keys=True)}`.",
        f"- Participant-balanced allocation: {base_per_participant} per participant, with {remainder} deterministic remainder slot(s) assigned to: {', '.join(split['extra_participants']) or 'none'}.",
        f"- Duplicate selected source clips flagged: {len(manifest.get('duplicate_selected_clips', {}))}.",
        f"- Automated integrity checks: `{json.dumps(manifest.get('automated_integrity_checks', {}), sort_keys=True)}`. These are artifact checks, not a human audit.",
        f"- Answer and C2-generation model/provider: `{model['provider']}` / `{model['identifier']}`.",
        f"- Visual input: {frame['count']} uniformly selected distinct source PTS within each official interval; max edge {frame['preprocessing']['max_edge']} px; JPEG quality {frame['preprocessing']['jpeg_quality']}; API image detail `{frame['image_detail']}`.",
        f"- C2: 3–5 general kitchen facts generated from only the target action and fixed kitchen context (prompt `{resolved_config['generation']['prompt_version']}`).",
        f"- C3: public Wikipedia reference pages retrieved with answer-blind fixed queries and deterministic TF-IDF sentence selection (`{resolved_config['retrieval']['version']}`); every supplied fact retains its URL, revision, query, and supporting span.",
        "",
        "## Completion and accuracy",
        "",
        "Invalid model answers count as incorrect when a condition is complete and are also listed as parse failures. Incomplete conditions are not reported as zero accuracy.",
        "",
        "| Condition | Completed | Correct | Accuracy | Parse failures | Missing | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in CONDITIONS:
        values = condition_metrics[condition]
        accuracy = f"{100 * values['accuracy']:.1f}%" if values["accuracy"] is not None else "unavailable/incomplete"
        correct = f"{values['correct']}/{values['completed']}"
        lines.append(
            f"| {condition} | {values['completed']}/{sample_size} | {correct} | {accuracy} | {values['parse_failures']} | {values['missing']} | {values['status']} |"
        )
    lines += [
        "",
        f"C3 retrieval availability/status across the {sample_size} selected items: `{json.dumps(dict(sorted(c3_statuses.items())), sort_keys=True)}`.",
        "",
        "## Paired outcomes",
        "",
        "The paired denominator includes items with completed model responses in both named conditions; malformed answers remain genuine wrong answers, while missing predictions are excluded and shown separately.",
        "",
        "| Comparison | Denominator | Both correct | Baseline wrong → comparison correct | Baseline correct → comparison wrong | Both wrong | Missing pair |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in comparisons.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {values['paired_denominator']} | {values['both_correct']} | {values['baseline_wrong_comparison_correct']} | {values['baseline_correct_comparison_wrong']} | {values['both_wrong']} | {values['missing_pair']} |"
        )
    lines += ["", "## Representative changed predictions", "", *changed]
    lines += [
        "",
        "## Limitations and unresolved issues",
        "",
        f"- This {sample_size}-question participant-balanced development sample is exploratory and is not statistically conclusive.",
        "- It does not test whether knowledge helps before purpose becomes visible, and it contains no reveal-time or future-video experiment.",
        "- It does not establish that hierarchy or memory is necessary, nor that external knowledge is generally useful or useless.",
        "- Retrieved facts are general reference statements; weak or empty retrievals remain visible rather than being replaced with self-generated knowledge.",
        "- No recipe plan, participant-specific recipe, narration, gold label, or unselected annotation metadata enters knowledge generation/retrieval.",
        "- No human audit has been completed. `review_annotations.csv` intentionally preserves initially empty reviewer fields.",
    ]
    blockers = manifest.get("blockers", [])
    if blockers:
        lines += ["", "Current execution blockers:", "", *[f"- {value}" for value in blockers]]
    atomic_write_text(analysis_root / "summary.md", "\n".join(lines) + "\n")
    return metrics


# C3-v2 reporting is deliberately separate from the original report writer: it
# combines immutable C0-C2/C3-v1 reference predictions with only the revised C3
# artifact and never copies baseline predictions into the revision run.


def _paired_for_ids(
    question_ids: Sequence[str],
    labels: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    corrections: list[str] = []
    harms: list[str] = []
    for question_id in question_ids:
        left = baseline.get(question_id)
        right = comparison.get(question_id)
        if not left or not right or left.get("request_status") != "completed" or right.get("request_status") != "completed":
            counts["missing_pair"] += 1
            continue
        gold = str(labels[question_id]["gold_letter"])
        left_correct = _correct(left, gold)
        right_correct = _correct(right, gold)
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif not left_correct and right_correct:
            counts["baseline_wrong_comparison_correct"] += 1
            corrections.append(question_id)
        elif left_correct and not right_correct:
            counts["baseline_correct_comparison_wrong"] += 1
            harms.append(question_id)
        else:
            counts["both_wrong"] += 1
        counts["paired_denominator"] += 1
    return {
        "requested_denominator": len(question_ids),
        "paired_denominator": counts["paired_denominator"],
        "both_correct": counts["both_correct"],
        "baseline_wrong_comparison_correct": counts["baseline_wrong_comparison_correct"],
        "baseline_correct_comparison_wrong": counts["baseline_correct_comparison_wrong"],
        "both_wrong": counts["both_wrong"],
        "missing_pair": counts["missing_pair"],
        "correction_ids": corrections,
        "harm_ids": harms,
    }


def _prediction_metrics_for_ids(
    question_ids: Sequence[str],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    completed = sum(
        predictions.get(question_id, {}).get("request_status") == "completed"
        for question_id in question_ids
    )
    parsed = sum(
        predictions.get(question_id, {}).get("parsed_answer") in LETTERS
        for question_id in question_ids
        if predictions.get(question_id, {}).get("request_status") == "completed"
    )
    correct = sum(
        _correct(predictions.get(question_id), str(labels[question_id]["gold_letter"]))
        for question_id in question_ids
    )
    return {
        "denominator": len(question_ids),
        "completed": completed,
        "parsed": parsed,
        "parse_failures": completed - parsed,
        "missing": len(question_ids) - completed,
        "correct": correct,
        "accuracy_on_denominator": correct / len(question_ids) if question_ids else None,
        "accuracy_among_completed": correct / completed if completed else None,
    }


def _load_knowledge_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(question_id): dict(value.get("record") or value)
        for question_id, value in payload.get("items", {}).items()
    }


def _token_totals(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for row in rows.values():
        usage = row.get("token_usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return dict(totals)


def write_retrieval_review(
    analysis_root: Path,
    split: Mapping[str, Any],
    retrieval: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write an answer-blind fact audit before answer inference."""
    analysis_root.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for index, item in enumerate(split["records"], 1):
        question_id = str(item["question_id"])
        record = retrieval[question_id]
        facts = []
        for fact, source in zip(record.get("facts", []), record.get("fact_provenance", [])):
            facts.append(
                "<li>"
                + html.escape(str(fact))
                + f" <a href='{html.escape(str(source.get('url') or ''))}'>"
                + html.escape(str(source.get("title") or "source"))
                + "</a><details><summary>support</summary><blockquote>"
                + html.escape(str(source.get("supporting_span") or ""))
                + "</blockquote></details></li>"
            )
        queries = "".join(
            f"<li><code>{html.escape(str(row.get('query')))}</code> — {html.escape(str(row.get('status')))}</li>"
            for row in record.get("search_audit", [])
        )
        pages = "".join(
            f"<li>{html.escape(str(row.get('title') or row.get('requested_title') or row.get('requested_page_id')))} — "
            f"{html.escape(str(row.get('status')))}</li>"
            for row in record.get("page_acquisitions", [])
        )
        sections.append(
            f"<section><h2>{index}. {html.escape(question_id)} — {html.escape(str(item['action_phrase']))}</h2>"
            f"<p>Status: <b>{html.escape(str(record.get('status')))}</b>; reason: "
            f"{html.escape(str(record.get('abstention_reason') or '—'))}; cache: <code>{html.escape(str(record.get('cache_id')))}</code></p>"
            f"<h3>Facts and captured support</h3><ol>{''.join(facts) or '<li>No usable fact retained.</li>'}</ol>"
            f"<details><summary>Executed queries</summary><ul>{queries}</ul></details>"
            f"<details><summary>Page acquisition outcomes</summary><ul>{pages}</ul></details></section>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>C3-v2 answer-blind retrieval review</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:auto;padding:20px}}section{{border-top:2px solid #444;padding:18px 0}}code{{overflow-wrap:anywhere}}blockquote{{background:#eee;padding:10px}}
</style></head><body><h1>C3-v2 answer-blind retrieval review</h1>
<p>This view intentionally omits choices, scoring labels, and predictions. Checks are automated; no human audit is claimed.</p>
{''.join(sections)}</body></html>"""
    atomic_write_text(analysis_root / "retrieval_review.html", document)
    annotation_fields = [
        "question_id", "source_identity_correct", "fact_source_support", "fact_accuracy",
        "fact_relevance_applicability", "notes",
    ]
    annotation_path = analysis_root / "retrieval_review_annotations.csv"
    existing: dict[str, dict[str, str]] = {}
    if annotation_path.is_file():
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("question_id"):
                    existing[str(row["question_id"])] = row
    rows = []
    for item in split["records"]:
        question_id = str(item["question_id"])
        old = existing.get(question_id, {})
        rows.append(
            {field: question_id if field == "question_id" else old.get(field, "") for field in annotation_fields}
        )
    _write_csv(annotation_path, rows)


def _write_c3_v2_review(
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Mapping[str, Any]]],
    revised: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Mapping[str, Any]],
    c2: Mapping[str, Mapping[str, Any]],
    retrieval: Mapping[str, Mapping[str, Any]],
) -> None:
    sections: list[str] = []
    for number, item in enumerate(records, 1):
        question_id = str(item["question_id"])
        gold = str(labels[question_id]["gold_letter"])
        choices = "".join(
            f"<li><b>{letter}.</b> {html.escape(str(choice))}</li>"
            for letter, choice in zip(LETTERS, item["displayed_choices"])
        )
        pred_rows = []
        sources = {"C0": baseline["C0"], "C1": baseline["C1"], "C2": baseline["C2"], "C3-v1": baseline["C3"], "C3-v2": revised}
        for name, predictions in sources.items():
            prediction = predictions.get(question_id)
            answer = prediction.get("parsed_answer") if prediction else None
            pred_rows.append(
                f"<tr><th>{name}</th><td class='{'correct' if _correct(prediction, gold) else 'incorrect'}'>{html.escape(str(answer or '—'))}</td>"
                f"<td>{html.escape(str((prediction or {}).get('request_status', 'missing')))}</td></tr>"
            )
        frame_cards = []
        for frame in frames[question_id]["frames"]:
            relative = os.path.relpath(frame["path"], output_path.parent)
            frame_cards.append(
                f"<figure><img loading='lazy' src='{html.escape(relative)}'><figcaption>{float(frame['timestamp_sec']):.6f}s</figcaption></figure>"
            )
        knowledge = retrieval[question_id]
        facts = []
        for fact, source in zip(knowledge.get("facts", []), knowledge.get("fact_provenance", [])):
            facts.append(
                f"<li>{html.escape(str(fact))} <a href='{html.escape(str(source.get('url') or ''))}'>"
                f"[{html.escape(str(source.get('title') or 'source'))}]</a></li>"
            )
        c2_facts = "".join(f"<li>{html.escape(str(fact))}</li>" for fact in c2.get(question_id, {}).get("facts", []))
        sections.append(
            f"<section><h2>{number}. {html.escape(question_id)} — {html.escape(str(item['action_phrase']))}</h2>"
            f"<p><b>Video:</b> <code>{html.escape(str(item['source_video_path']))}</code>; interval "
            f"{float(item['start_sec']):.6f}–{float(item['end_sec']):.6f}s</p>"
            f"<p><b>Question:</b> {html.escape(str(item['question']))}</p><ol>{choices}</ol><p><b>Gold:</b> {gold}</p>"
            f"<div class='frames'>{''.join(frame_cards)}</div><table><tbody>{''.join(pred_rows)}</tbody></table>"
            f"<div class='knowledge'><div><h3>C2 facts (unchanged)</h3><ul>{c2_facts}</ul></div>"
            f"<div><h3>C3-v2 facts</h3><p>Status: {html.escape(str(knowledge['status']))}; "
            f"{html.escape(str(knowledge.get('abstention_reason') or ''))}</p><ul>{''.join(facts) or '<li>Retrieval abstained.</li>'}</ul></div></div></section>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>C3-v2 review</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:auto;padding:20px}}section{{border-top:3px solid #333;padding:20px 0}}code{{overflow-wrap:anywhere}}.frames{{display:flex;gap:7px;overflow-x:auto;background:#eee;padding:7px}}figure{{margin:0;min-width:170px}}img{{width:170px;height:170px;object-fit:contain;background:#111}}figcaption{{text-align:center}}table{{border-collapse:collapse;margin:10px 0}}th,td{{border:1px solid #aaa;padding:5px 10px}}.correct{{background:#d8f3dc}}.incorrect{{background:#ffe0e0}}.knowledge{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
</style></head><body><h1>HD-EPIC Why Recognition C3-v2 review</h1><p>Development output. Official exact-choice scoring is preserved. Automated checks are not a human evidence audit.</p>{''.join(sections)}</body></html>"""
    atomic_write_text(output_path, document)


def generate_c3_v2_reports(
    analysis_root: Path,
    run_root: Path,
    reference_root: Path,
    split: Mapping[str, Any],
    manifest: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    retrieval: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    analysis_root.mkdir(parents=True, exist_ok=True)
    records = split["records"]
    labels = split["scoring_labels"]
    all_ids = [str(row["question_id"]) for row in records]
    baseline = {
        condition: read_predictions(reference_root / "predictions" / f"{condition}.jsonl")
        for condition in CONDITIONS
    }
    revised = read_predictions(run_root / "predictions" / "C3.jsonl")
    eligible_ids = [question_id for question_id in all_ids if retrieval[question_id]["status"] == "usable"]
    abstention_ids = [question_id for question_id in all_ids if retrieval[question_id]["status"] != "usable"]
    v1_ids = [question_id for question_id in all_ids if question_id in baseline["C3"]]
    v2_answered_ids = [question_id for question_id in all_ids if question_id in revised]
    common_ids = [question_id for question_id in all_ids if question_id in baseline["C3"] and question_id in revised]

    condition_metrics = {
        condition: _prediction_metrics_for_ids(all_ids, labels, baseline[condition])
        for condition in ("C0", "C1", "C2")
    }
    c3_v1 = _prediction_metrics_for_ids(v1_ids, labels, baseline["C3"])
    c3_v1.update({"total_coverage": len(v1_ids) / 100, "answered_of_total": len(v1_ids)})
    c3_v2 = _prediction_metrics_for_ids(eligible_ids, labels, revised)
    c3_v2.update(
        {
            "retrieval_eligible": len(eligible_ids),
            "retrieval_abstentions": len(abstention_ids),
            "answer_coverage_of_eligible": c3_v2["completed"] / len(eligible_ids) if eligible_ids else None,
            "answer_coverage_of_total": c3_v2["completed"] / 100,
        }
    )

    fallback_rows: list[dict[str, Any]] = []
    fallback_correct = 0
    for question_id in all_ids:
        c3_row = revised.get(question_id)
        if c3_row and c3_row.get("request_status") == "completed":
            selected = c3_row
            source = "C3-v2"
            reason = "usable_retrieved_facts"
        else:
            selected = baseline["C1"][question_id]
            source = "C1_fallback"
            reason = retrieval[question_id].get("abstention_reason") or "missing_C3_v2_answer"
        is_correct = _correct(selected, str(labels[question_id]["gold_letter"]))
        fallback_correct += int(is_correct)
        fallback_rows.append(
            {
                "question_id": question_id,
                "answer": selected.get("parsed_answer"),
                "answer_source": source,
                "fallback_reason": reason,
                "correct": is_correct,
            }
        )

    comparisons = {
        "C2_vs_C1_all_100": _paired_for_ids(all_ids, labels, baseline["C1"], baseline["C2"]),
        "C3_v2_vs_C1_eligible": _paired_for_ids(eligible_ids, labels, baseline["C1"], revised),
        "C3_v2_vs_C2_eligible": _paired_for_ids(eligible_ids, labels, baseline["C2"], revised),
        "C3_v1_vs_C1_answered": _paired_for_ids(v1_ids, labels, baseline["C1"], baseline["C3"]),
        "C3_v2_vs_C3_v1_common": _paired_for_ids(common_ids, labels, baseline["C3"], revised),
    }
    unique_retrieval = {
        record["cache_id"]: record for record in retrieval.values()
    }
    retrieval_operations = {
        "unique_action_packs": len(unique_retrieval),
        "operational_error_records": sum(
            len(record.get("operational_errors", [])) for record in unique_retrieval.values()
        ),
        "api_warning_search_records": sum(
            bool(search.get("warnings"))
            for record in unique_retrieval.values()
            for search in record.get("search_audit", [])
        ),
        "bounded_search_continuation_records": sum(
            bool(search.get("continuation"))
            for record in unique_retrieval.values()
            for search in record.get("search_audit", [])
        ),
        "continuation_disposition": "recorded but not followed beyond the fixed top-result budget",
    }
    metrics = {
        "experiment": "HD-EPIC Why Recognition C3 external-retrieval development revision",
        "total_denominator": 100,
        "annotation": manifest["annotation"],
        "participant_distribution": split["participant_distribution"],
        "reference_run": str(reference_root),
        "condition_metrics": condition_metrics,
        "C3_v1": c3_v1,
        "C3_v2": c3_v2,
        "C1_on_C3_v2_eligible": _prediction_metrics_for_ids(eligible_ids, labels, baseline["C1"]),
        "C2_on_C3_v2_eligible": _prediction_metrics_for_ids(eligible_ids, labels, baseline["C2"]),
        "paired_comparisons": comparisons,
        "v1_v2_common_answered_count": len(common_ids),
        "coverage_change_items": len(v2_answered_ids) - len(v1_ids),
        "retrieval_status_counts": dict(sorted(Counter(retrieval[q]["status"] for q in all_ids).items())),
        "retrieval_operations": retrieval_operations,
        "abstention_ids_and_reasons": {
            question_id: {
                "status": retrieval[question_id]["status"],
                "reason": retrieval[question_id].get("abstention_reason"),
            }
            for question_id in abstention_ids
        },
        "C3_v2_with_C1_fallback_policy": {
            "denominator": 100,
            "correct": fallback_correct,
            "accuracy": fallback_correct / 100,
            "C3_v2_answer_sources": sum(row["answer_source"] == "C3-v2" for row in fallback_rows),
            "C1_fallback_sources": sum(row["answer_source"] == "C1_fallback" for row in fallback_rows),
        },
        "C3_v2_token_usage": _token_totals(revised),
        "execution_lineage": manifest.get("automated_integrity_checks", {}).get("answer_lineage", {}),
        "development_diagnostic": True,
    }
    atomic_write_json(analysis_root / "metrics.json", metrics)
    _write_csv(analysis_root / "fallback_policy.csv", fallback_rows)

    c2 = _load_knowledge_index(reference_root / "cache" / "knowledge" / "C2_index.json")
    item_rows: list[dict[str, Any]] = []
    fallback_by_id = {row["question_id"]: row for row in fallback_rows}
    for item in records:
        question_id = str(item["question_id"])
        gold = str(labels[question_id]["gold_letter"])
        knowledge = retrieval[question_id]
        row: dict[str, Any] = {
            "question_id": question_id,
            "participant": item["participant"],
            "source_video_id": item["source_video_id"],
            "source_video_path": item["source_video_path"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            "target_action": item["action_phrase"],
            "question": item["question"],
            "displayed_choices": " | ".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["displayed_choices"])),
            "gold_letter": gold,
        }
        for name, predictions in {**baseline, "C3_v2": revised}.items():
            prediction = predictions.get(question_id)
            row[f"{name}_prediction"] = (prediction or {}).get("parsed_answer", "")
            row[f"{name}_correct"] = _correct(prediction, gold) if prediction else ""
            row[f"{name}_request_status"] = (prediction or {}).get("request_status", "missing")
            row[f"{name}_parse_failure"] = bool(prediction and prediction.get("error_category") == "parse_failure")
        row.update(
            {
                "C3_v2_retrieval_status": knowledge["status"],
                "C3_v2_abstention_reason": knowledge.get("abstention_reason") or "",
                "C2_facts": _facts_text(c2.get(question_id)),
                "C3_v2_facts": _facts_text(knowledge),
                "C3_v2_external_sources": _source_refs(knowledge),
                "frame_record_path": frames[question_id]["_record_path"],
                "ordered_frame_payload_hash": frames[question_id]["ordered_frame_payload_hash"],
                "fallback_answer": fallback_by_id[question_id]["answer"],
                "fallback_answer_source": fallback_by_id[question_id]["answer_source"],
                "fallback_reason": fallback_by_id[question_id]["fallback_reason"],
                "fallback_correct": fallback_by_id[question_id]["correct"],
            }
        )
        item_rows.append(row)
    _write_csv(analysis_root / "item_results.csv", item_rows)
    _write_c3_v2_review(
        analysis_root / "review.html", records, labels, baseline, revised, frames, c2, retrieval
    )
    _write_review_annotations(analysis_root / "review_annotations.csv", records)
    write_retrieval_review(analysis_root, split, retrieval)

    def accuracy_cell(values: Mapping[str, Any]) -> str:
        accuracy = values.get("accuracy_on_denominator")
        return f"{int(values['correct'])}/{int(values['denominator'])} ({100 * accuracy:.1f}%)" if accuracy is not None else "unavailable"

    lines = [
        "# HD-EPIC Why Recognition: C3 Retrieval Revision (development)",
        "",
        "This revision asks whether repaired, source-supported external background facts change why-recognition on the existing participant-balanced 100-question development sample. It preserves official exact-choice scoring and does not establish a general knowledge, pre-reveal, hierarchy, or memory claim.",
        "",
        "## Data, lineage, and change log",
        "",
        f"- Official annotation: `{manifest['annotation']['path']}`; 500 questions; SHA-256 `{manifest['annotation']['sha256']}`.",
        f"- Exact saved dev100 and 400 reserved IDs were loaded from `{reference_root}`; no resampling and no reserved-item evaluation occurred.",
        f"- Frames: {manifest['reference_artifacts']['frames']['jpeg_count']} existing JPEGs referenced directly; no video decoding or image duplication. Reference JPEG-set hash: `{manifest['reference_artifacts']['frames']['jpeg_set_hash']}`.",
        f"- C0-C2 and C3-v1 are original predictions loaded read-only from `{reference_root}`. Their request IDs and timestamps were not rewritten.",
        f"- Answer model: `{resolved_config['model']['provider']}` / `{resolved_config['model']['identifier']}` at `{resolved_config['model']['base_url']}`, reasoning effort `{resolved_config['answer']['reasoning_effort']}`, max completion tokens {resolved_config['answer']['max_output_tokens']}.",
        f"- C3-v2 retrieval: Wikipedia MediaWiki API, context-first queries, bounded multi-page `exintro` plaintext extraction with each extract capped at {resolved_config['retrieval']['extract_chars']} characters, explicit page outcomes, disambiguation exclusion, and deterministic selector `{retrieval[all_ids[0]]['selector_version']}`. The selector retains 1-3 checked source spans (the inference gate permits 1-5); zero facts abstains.",
        "- No LLM or general-web fallback produced retrieval facts. Retrieval queries received only action phrase plus fixed general kitchen context.",
        "",
        "## Accuracy and coverage",
        "",
        "| View | Coverage | Exact-choice accuracy | Parse failures |",
        "|---|---:|---:|---:|",
        f"| C0 original | 100/100 | {accuracy_cell(condition_metrics['C0'])} | {condition_metrics['C0']['parse_failures']} |",
        f"| C1 original | 100/100 | {accuracy_cell(condition_metrics['C1'])} | {condition_metrics['C1']['parse_failures']} |",
        f"| C2 original | 100/100 | {accuracy_cell(condition_metrics['C2'])} | {condition_metrics['C2']['parse_failures']} |",
        f"| C3-v1 original, answered subset | {len(v1_ids)}/100 | {c3_v1['correct']}/{len(v1_ids)} ({100*c3_v1['accuracy_on_denominator']:.1f}%) | {c3_v1['parse_failures']} |",
        f"| C1 on C3-v2 eligible subset | {len(eligible_ids)}/100 | {metrics['C1_on_C3_v2_eligible']['correct']}/{len(eligible_ids)} ({100*metrics['C1_on_C3_v2_eligible']['accuracy_on_denominator']:.1f}%) | {metrics['C1_on_C3_v2_eligible']['parse_failures']} |",
        f"| C2 on C3-v2 eligible subset | {len(eligible_ids)}/100 | {metrics['C2_on_C3_v2_eligible']['correct']}/{len(eligible_ids)} ({100*metrics['C2_on_C3_v2_eligible']['accuracy_on_denominator']:.1f}%) | {metrics['C2_on_C3_v2_eligible']['parse_failures']} |",
        f"| C3-v2 pure retrieval condition, eligible subset | {c3_v2['completed']}/{len(eligible_ids)} eligible ({c3_v2['completed']}/100 total) | {c3_v2['correct']}/{len(eligible_ids)} ({100*c3_v2['accuracy_on_denominator']:.1f}%) | {c3_v2['parse_failures']} |",
        f"| C3-v2 policy with explicit C1 fallback | 100/100 | {fallback_correct}/100 ({fallback_correct:.1f}%) | n/a |",
        "",
        f"Retrieval statuses over the 100-item denominator: `{json.dumps(metrics['retrieval_status_counts'], sort_keys=True)}`. Operational retrieval errors: {retrieval_operations['operational_error_records']}; API warning records: {retrieval_operations['api_warning_search_records']}. Search continuation was retained for {retrieval_operations['bounded_search_continuation_records']} bounded queries and deliberately not followed beyond the configured top-result budget. Abstentions are not C3 predictions; the fallback policy labels their answers as C1-derived in `fallback_policy.csv`.",
        "",
        "Abstention IDs and reasons:",
        "",
        *[
            f"- `{question_id}`: `{metrics['abstention_ids_and_reasons'][question_id]['reason']}`"
            for question_id in abstention_ids
        ],
        "",
        "## Paired development diagnostics",
        "",
        "| Comparison | Paired n | Both correct | Correction | Harm | Both wrong | Missing pair |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("C2 vs C1, original 100", "C2_vs_C1_all_100"),
        ("C3-v2 vs C1, v2 eligible", "C3_v2_vs_C1_eligible"),
        ("C3-v2 vs C2, v2 eligible", "C3_v2_vs_C2_eligible"),
        ("C3-v1 vs C1, v1 answered", "C3_v1_vs_C1_answered"),
        ("C3-v2 vs C3-v1, common answered", "C3_v2_vs_C3_v1_common"),
    ):
        value = comparisons[key]
        lines.append(
            f"| {label} | {value['paired_denominator']} | {value['both_correct']} | {value['baseline_wrong_comparison_correct']} | {value['baseline_correct_comparison_wrong']} | {value['both_wrong']} | {value['missing_pair']} |"
        )
    lines += [
        "",
        f"- C3-v2 versus C1 correction IDs: `{json.dumps(comparisons['C3_v2_vs_C1_eligible']['correction_ids'])}`.",
        f"- C3-v2 versus C1 harm IDs: `{json.dumps(comparisons['C3_v2_vs_C1_eligible']['harm_ids'])}`.",
        f"- V1→v2 answer-coverage change: {len(v2_answered_ids) - len(v1_ids):+d} items; common answered subset: {len(common_ids)}.",
        "",
        "## Usage and limitations",
        "",
        f"- Revised-C3 token usage reported by the endpoint: `{json.dumps(metrics['C3_v2_token_usage'], sort_keys=True)}`.",
        f"- Revised answer lineage: {len(revised)} canonical provider-response records with original request IDs; none imported from C3-v1. Latest verified resume: {metrics['execution_lineage'].get('latest_resume_reused', 0)} reused, {metrics['execution_lineage'].get('latest_resume_created', 0)} newly generated.",
        "- This is one development revision, not a statistically conclusive or preregistered test. Items share participants and videos, and no independence assumption is justified here.",
        "- Automated source/span checks allowed execution, but source identity, factual accuracy, applicability, clip sufficiency, and answer ambiguity still require human review. No human audit is claimed.",
        "- Wikipedia introductions can omit useful procedural detail; underspecified actions may appropriately yield no usable facts. Official labels were not changed for ambiguous or closely related choices.",
        "- Existing frame records were rehashed and bounds-checked, but the revision did not independently re-probe source-video presentation timestamps.",
        "",
        "## Commands",
        "",
        "```bash",
        "bash scripts/run_e0.sh --config configs/e0_c3_v2.yaml --stage all",
        "bash scripts/run_e0.sh --config configs/e0_c3_v2.yaml --stage retrieve",
        "bash scripts/run_e0.sh --config configs/e0_c3_v2.yaml --stage run --conditions C3",
        "bash scripts/run_e0.sh --config configs/e0_c3_v2.yaml --stage report",
        "```",
    ]
    atomic_write_text(analysis_root / "summary.md", "\n".join(lines) + "\n")
    atomic_write_text(
        analysis_root / "change_log.md",
        "# C3-v2 change log\n\n"
        "- Preserved the original E0 run and loaded its exact dev100, reserved IDs, frames, C0-C2, and C3-v1 artifacts read-only.\n"
        "- Replaced title-first/unsupported extraction with context-first searches and documented multi-page `exintro` MediaWiki extraction (`exchars` capped at 1,200), recording every returned page outcome.\n"
        "- Added generic action structure and kitchen word-sense resolution, explicit API/acquisition outcomes, disambiguation exclusion, and exact source-span lineage.\n"
        "- Separated the C2 3-5-fact gate from the C3-v2 1-5-fact gate and added explicit retrieval abstention plus C1-fallback policy reporting.\n"
        "- No baseline model call, video decode, hierarchy input, reserved-question evaluation, or generated-knowledge fallback was performed.\n",
    )
    return metrics
