"""Persistent human labels with revision history; no evaluator files are changed."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIELDS = {"why", "goal", "current", "next_1", "next_2", "next_3"}
FLAGS = {"unsupported_addition", "contradiction", "too_broad", "too_narrow", "reference_issue", "mapper_issue"}


class ConflictError(ValueError):
    pass


def validate_label(payload, catalog):
    if not isinstance(payload, dict):
        raise ValueError("Label must be a JSON object.")
    reviewer = payload.get("reviewer", "")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
        raise ValueError("Enter a reviewer name of 1–100 characters.")
    identity, model, field = (payload.get(k) for k in ("id", "model", "field"))
    if not all(isinstance(x, str) for x in (identity, model, field)):
        raise ValueError("Example, model and field are required.")
    if identity not in catalog.references or model not in catalog.models:
        raise ValueError("Unknown example or model.")
    detail = catalog.get_example(identity)
    prediction = next((p for p in detail["predictions"] if p["model"] == model), None)
    if prediction is None:
        raise ValueError("This model has no evaluated record for this example.")
    if payload.get("answer_sha256") != prediction["answer_sha256"]:
        raise ConflictError("Prediction or reference changed. Reload this example before labeling.")
    if field not in FIELDS:
        raise ValueError("Unknown review field.")
    if field in {"why", "goal", "current"} and not detail["reference"][field]["eligible"]:
        raise ValueError("This field has no eligible reference.")
    if field.startswith("next_"):
        pos = int(field[-1]) - 1
        pair = detail["reference"]["next"][pos]["pair"]
        if not any(x is not None for x in pair):
            raise ValueError("This future action has no reference class labels.")
    label = payload.get("label")
    allowed = {"correct", "partial", "incorrect", "uncertain"} if field in {"why", "goal"} else {"correct", "incorrect", "uncertain"}
    if label not in allowed:
        raise ValueError("Invalid label for this field.")
    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 12000:
        raise ValueError("Notes must be text, up to 12,000 characters.")
    flags = payload.get("flags", {})
    if not isinstance(flags, dict) or set(flags) - FLAGS or any(type(v) is not bool for v in flags.values()):
        raise ValueError("Flags must be named boolean values.")
    if flags.get("contradiction") and label != "incorrect":
        raise ValueError("A central contradiction requires an incorrect label.")
    verb, noun = payload.get("verb_correct"), payload.get("noun_correct")
    if any(v is not None and type(v) is not bool for v in (verb, noun)):
        raise ValueError("Verb/noun labels must be true, false, or null.")
    if field not in {"why", "goal"}:
        pair = detail["reference"]["current"]["pair"] if field == "current" else detail["reference"]["next"][int(field[-1]) - 1]["pair"]
        if None in pair and label == "correct":
            raise ValueError("A correct joint action requires both reference components. Label the available component and leave the joint decision uncertain.")
        if pair[0] is None and verb is not None or pair[1] is None and noun is not None:
            raise ValueError("Do not score a component without a reference class.")
        if (verb is False or noun is False) and label != "incorrect" and not (None in pair and label == "uncertain"):
            raise ValueError("A wrong verb or noun makes the joint action incorrect.")
        if verb is True and noun is True and label != "correct":
            raise ValueError("Two correct components require a correct joint label.")
    evidence = payload.get("evidence", "observed")
    if evidence not in {"observed", "future"}:
        raise ValueError("Evidence must be observed or future.")
    return {"reviewer": reviewer.strip(), "id": identity, "model": model,
            "answer_sha256": prediction["answer_sha256"], "field": field, "label": label,
            "verb_correct": verb, "noun_correct": noun, "notes": notes.strip(),
            "flags": flags, "evidence": evidence,
            "semantic_packet": prediction.get("semantic_packets", {}).get(field)}


class LabelStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS labels (
                    reviewer TEXT NOT NULL, example_id TEXT NOT NULL, model TEXT NOT NULL,
                    answer_sha256 TEXT NOT NULL, field TEXT NOT NULL, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (reviewer, example_id, model, answer_sha256, field)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save(self, payload, catalog, expected_revision=None):
        label = validate_label(payload, catalog)
        key = tuple(label[k] for k in ("reviewer", "id", "model", "answer_sha256", "field"))
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError("Expected revision must be a nonnegative integer.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT revision FROM labels WHERE reviewer=? AND example_id=? AND model=? AND answer_sha256=? AND field=?", key).fetchone()
            revision = previous[0] if previous else 0
            if expected_revision is not None and expected_revision != revision:
                raise ConflictError("This label was edited in another tab. Reload before saving.")
            label.update(revision=revision + 1, updated_at=datetime.now(timezone.utc).isoformat())
            encoded = json.dumps(label, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)", (*key, revision + 1, encoded))
            db.execute("INSERT INTO events(payload) VALUES (?)", (encoded,))
        return label

    def all(self, reviewer=None, example_id=None):
        with self.connect() as db:
            clauses, params = [], []
            for column, value in (("reviewer", reviewer), ("example_id", example_id)):
                if value:
                    clauses.append(column + "=?")
                    params.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = db.execute("SELECT payload FROM labels" + where + " ORDER BY reviewer,example_id,model,field", params)
            return [json.loads(row[0]) for row in rows]

    def current(self, catalog, reviewer=None, example_id=None):
        labels = self.all(reviewer, example_id)
        hashes = {}
        selected = []
        for label in labels:
            identity = label["id"]
            if identity not in catalog.references:
                continue
            if identity not in hashes:
                hashes[identity] = {p["model"]: p["answer_sha256"] for p in catalog.get_example(identity)["predictions"]}
            if hashes[identity].get(label["model"]) == label["answer_sha256"]:
                selected.append(label)
        return selected


def required_fields(reference):
    fields = [f for f in ("why", "goal") if reference[f]["eligible"]]
    if reference["task"] == "track" and any(x is not None for x in reference["current_pair"]):
        fields.append("current")
    fields += [f"next_{i+1}" for i, pair in enumerate(reference["next_pairs"]) if any(x is not None for x in pair)]
    return fields


def progress(catalog, labels):
    index = {}
    for label in labels:
        index.setdefault((label["id"], label["model"]), {})[label["field"]] = label
    reviewed, total, example_states = 0, 0, {}
    for identity, reference in catalog.references.items():
        needed = required_fields(reference)
        state = {"label_count": 0, "required_count": 0, "complete": False,
                 "uncertain": False, "reference_issue": False, "reviewed_models": [], "by_model": {}}
        for model in catalog.models:
            if identity not in catalog.records[model]:
                continue
            total += 1
            saved = index.get((identity, model), {})
            complete = all(f in saved for f in needed)
            state["by_model"][model] = {"label_count": len(saved), "required_count": len(needed),
                                       "complete": complete,
                                       "uncertain": any(x["label"] == "uncertain" for x in saved.values()),
                                       "reference_issue": any(x["flags"].get("reference_issue", False) for x in saved.values())}
            reviewed += int(complete)
            if complete:
                state["reviewed_models"].append(model)
            state["label_count"] += len(saved)
            state["required_count"] += len(needed)
            state["uncertain"] |= any(x["label"] == "uncertain" for x in saved.values())
            state["reference_issue"] |= any(x["flags"].get("reference_issue", False) for x in saved.values())
        state["complete"] = bool(state["required_count"] and state["label_count"] >= state["required_count"])
        example_states[identity] = state
    return {"reviewed": reviewed, "total_predictions": total, "labels": len(labels),
            "uncertain": sum(x["label"] == "uncertain" for x in labels),
            "by_field": dict(Counter(x["field"] for x in labels)), "examples": example_states}


def export_labels(labels, format_name):
    if format_name == "jsonl":
        return "application/x-ndjson", "human_labels.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels)
    if format_name == "csv":
        output = io.StringIO(newline="")
        fields = ["reviewer", "id", "model", "field", "label", "verb_correct", "noun_correct", "notes", "flags", "evidence", "answer_sha256", "revision", "updated_at"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            row = {k: label.get(k, "") for k in fields}
            row["flags"] = json.dumps(row["flags"], sort_keys=True)
            # Opening CSV in spreadsheet software must not execute model/user text.
            for k, value in row.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    row[k] = "'" + value
            writer.writerow(row)
        return "text/csv; charset=utf-8", "human_labels.csv", output.getvalue()
    if format_name == "judgments":
        decisions = {}
        for label in labels:
            packet = label.get("semantic_packet")
            if label["field"] not in {"why", "goal"} or label["label"] == "uncertain" or label["flags"].get("reference_issue") or not packet:
                continue
            decision = {"packet_id": packet["packet_id"], "prompt_sha256": packet["prompt_sha256"],
                        "judge": "human:" + label["reviewer"], "assessment_status": "human_review",
                        "judgment": {"field": label["field"], "label": label["label"],
                                     "justification": label["notes"] or "Human compared the prediction with the frozen reference and selected " + label["label"] + ".",
                                     "flags": {k: v for k, v in label["flags"].items() if k in {"unsupported_addition", "contradiction", "too_broad", "too_narrow"}}}}
            existing = decisions.get(packet["packet_id"])
            if existing and any(existing["judgment"][key] != decision["judgment"][key] for key in ("label", "flags")):
                raise ConflictError("The same semantic answer received conflicting labels across models. Resolve them before exporting evaluator judgments.")
            if not existing:
                decisions[packet["packet_id"]] = decision
        return "application/x-ndjson", "human_judgments.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in decisions.values())
    raise ValueError("Export format must be jsonl, csv, or judgments.")
