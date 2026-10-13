"""Evaluate saved predictions and prepare hash-bound semantic judge inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import unittest
from collections import Counter
from pathlib import Path

from . import VERSION
from .actions import score_action_pairs, score_next
from .extraction import extract_field, extract_next
from .mapper import ActionMapper, build_mapper, sha256
from .references import load_references, load_reference_bundle
from .semantics import (GOAL_RUBRIC, WHY_RUBRIC, aggregate_semantic,
                        make_judge_prompt, validate_judgment)

BUNDLE_ROOT = Path(__file__).resolve().parent.parent


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def unique_rows(values, key="id"):
    result = {}
    for value in values:
        identity = value[key]
        if identity in result:
            raise ValueError(f"Duplicate {key}: {identity}")
        result[identity] = value
    return result


def packet_for(reference, field, text):
    source = reference[field]
    context = {"question": reference["question"], "completed_history": reference["history"]}
    if field == "why":
        context.update(original_narration=source["original_narration"],
                       original_reference=source["original_reference"],
                       frozen_source_status=source["source_status"])
    prompt = make_judge_prompt(field, source["reference"], text, context=context,
                               goal_profile=source.get("profile") if field == "goal" else None)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    # Include query ID so repeated contexts cannot accidentally share decisions.
    packet_id = digest({"version": VERSION, "id": reference["id"], "field": field,
                        "prompt_sha256": prompt_hash})
    return {"packet_id": packet_id, "prompt_sha256": prompt_hash, "field": field,
            "prompt": prompt}


def import_why_judgment(record, reference, prediction):
    """Reuse an old decision only for exactly the same extracted answer/source."""
    if not record or not reference["why"]["eligible"] or prediction is None:
        return None
    if record.get("answer_why") != prediction or record.get("label") == "unjudgeable":
        return None
    source = reference["why"]
    for key, expected in (("id", reference["id"]), ("reference_reason", source["original_reference"]),
                          ("purpose_core", source["reference"]), ("reference_status", source["source_status"]),
                          ("action_question", reference["question"])):
        if key in record and record[key] != expected:
            raise ValueError("Cached WHY judgment is bound to another source or question")
    return validate_judgment({"field": "why", "label": record["label"],
        "justification": record["evidence_note"],
        "flags": {"unsupported_addition": bool(record.get("unsupported_additional_claims"))},
        "provenance": {"method": "imported_frozen_Codex_review", "reviewer": record["reviewer"],
                       "assessment_status": "provisional_assistant_assessment"}}, "why")


def _cohort_summary(records):
    track = [r for r in records if r["task"] == "track"]
    why = [r["semantic"]["why"] for r in records if r["task"] == "explain"]
    return {
        "examples": len(records),
        "current_action": score_action_pairs([r["predicted_current"] for r in track],
                                              [r["current_pair"] for r in track]),
        "why": aggregate_semantic(why, "why"),
        "goal": aggregate_semantic([r["semantic"]["goal"] for r in records], "goal"),
        "next": score_next([r["predicted_next"] for r in records],
                            [r["next_pairs"] for r in records]),
    }


def evaluate_model(name, references, predictions, mapper, legacy=None, judgments=None):
    """The intended reference cohort sets denominators, including absent rows."""
    expected = {r["id"] for r in references}
    if set(predictions) - expected:
        raise ValueError(f"{name}: predictions contain IDs outside the intended cohort")
    records, packets, links = [], {}, []
    diagnostics = Counter()
    legacy, judgments = legacy or {}, judgments or {}
    for reference in references:
        identity = reference["id"]
        prediction = predictions.get(identity)
        if prediction and prediction.get("task", reference["task"]) != reference["task"]:
            raise ValueError(f"{name}: task mismatch for {identity}")
        completed = bool(prediction and prediction.get("status", "completed") == "completed")
        raw = prediction.get("prediction", "") if completed else ""
        if not isinstance(raw, str):
            raise ValueError(f"{name}: prediction must be text for {identity}")
        diagnostics["absent_prediction"] += prediction is None
        diagnostics["failed_prediction"] += prediction is not None and not completed
        fields = {field: extract_field(raw, field) for field in ("action", "why", "goal")}
        nxt = extract_next(raw, lambda text: mapper.map_text(text)[0] is not None)
        diagnostics["repeated_next_blocks"] += nxt["block_count"] > 1
        diagnostics["explicit_empty_next_slots"] += sum(x is None for x in nxt["actions"][:3])
        rec = {"id": identity, "task": reference["task"], "video_id": reference["video_id"],
               "participant": reference["participant"], "current_pair": reference["current_pair"],
               "next_pairs": reference["next_pairs"], "generation_completed": completed,
               "raw_prediction_sha256": hashlib.sha256(raw.encode()).hexdigest(),
               "prompt_sha256": prediction.get("prompt_sha256") if prediction else None,
               "fields": fields, "next_extraction": nxt,
               "predicted_current": mapper.map_text(fields["action"]["text"]),
               "predicted_next": [mapper.map_text(text) for text in nxt["actions"]],
               "semantic": {}}
        for field in ("why", "goal"):
            source, text = reference[field], fields[field]["text"]
            entry = {"id": identity, "eligible": source["eligible"], "prediction": text,
                     "judgment": None}
            if source["eligible"] and text:
                packet = packet_for(reference, field, text)
                supplied = judgments.get(packet["packet_id"])
                if supplied:
                    if supplied["prompt_sha256"] != packet["prompt_sha256"]:
                        raise ValueError("Judgment prompt hash mismatch")
                    entry["judgment"] = validate_judgment(supplied["judgment"], field)
                    if not isinstance(supplied.get("judge"), str) or not supplied["judge"].strip():
                        raise ValueError("Supplied judgment requires judge identity")
                    entry["judgment"]["provenance"] = {"method": "supplied_semantic_judgment",
                        "judge": supplied["judge"], "assessment_status": supplied.get(
                            "assessment_status", "provisional_automated_assessment")}
                elif field == "why":
                    entry["judgment"] = import_why_judgment(legacy.get(identity), reference, text)
                    if legacy.get(identity) and entry["judgment"] is None:
                        diagnostics["legacy_why_not_reused_changed_extraction"] += 1
                # Exact source agreement is a sufficient match, not a rule that
                # nonidentical wording is wrong. Paraphrases still need a judge.
                if entry["judgment"] is None and text.strip() == source["reference"].strip():
                    entry["judgment"] = {"field": field, "label": "correct",
                        "justification": "Answer is identical to the eligible frozen reference.",
                        "flags": {}, "provenance": {"method": "identical_reference_sufficient_match"}}
                entry["packet_id"] = packet["packet_id"]
                if entry["judgment"] is None:
                    packets[packet["packet_id"]] = packet
                    links.append({"model": name, "id": identity, "field": field,
                                  "packet_id": packet["packet_id"]})
            rec["semantic"][field] = entry
        records.append(rec)
    summary = {"by_task": {task: _cohort_summary([r for r in records if r["task"] == task])
                            for task in ("explain", "track")
                            if any(r["task"] == task for r in records)},
               "all": _cohort_summary(records), "diagnostics": dict(diagnostics)}
    summary["by_video"] = {video: _cohort_summary([r for r in records if r["video_id"] == video])
                           for video in sorted({r["video_id"] for r in records})}
    return summary, records, list(packets.values()), links


def paired_comparison(before, after):
    """Compare shared intended IDs; display per-task cohort and prompt equality."""
    left, right = unique_rows(before), unique_rows(after)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("No shared IDs for comparison")
    for identity in shared:
        if left[identity]["task"] != right[identity]["task"]:
            raise ValueError("Paired task mismatch")
        a, b = left[identity]["prompt_sha256"], right[identity]["prompt_sha256"]
        if a and b and a != b:
            raise ValueError(f"Paired prompts differ for {identity}")
    result = {"shared_examples": len(shared), "by_task": {}}
    for task in ("explain", "track"):
        ids = [identity for identity in shared if left[identity]["task"] == task]
        if not ids:
            continue
        result["by_task"][task] = {
            "examples": len(ids), "before": _cohort_summary([left[i] for i in ids]),
            "after": _cohort_summary([right[i] for i in ids]),
            "prompt_hashes_verified": sum(bool(left[i]["prompt_sha256"] and right[i]["prompt_sha256"])
                                          for i in ids),
        }
    return result


def _percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def _semantic_cell(metric):
    return "pending" if metric["pending"] else _percent(metric["correct_rate"])


def report_text(summary):
    lines = ["# HD-EPIC field evaluation v1", "",
        "Saved outputs only. Action scores use a train-only heuristic mapper. Semantic scores are provisional; pending judgments produce no completed rate. No human review or visualization was performed.", "",
        "| Model / task | N | WHY correct | GOAL correct | Current pair | NEXT 1 pair | NEXT 2 pair | NEXT 3 pair | NEXT mean pair | NEXT OSA distance ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, model in summary["models"].items():
        for task, metrics in model["by_task"].items():
            current = _percent(metrics["current_action"]["action"]["rate"]) if task == "track" else "—"
            why = _semantic_cell(metrics["why"]) if task == "explain" else "—"
            goal = _semantic_cell(metrics["goal"])
            next_score = metrics["next"]
            distance = next_score["normalized_damerau_osa"]["action"]["mean"]
            distance_text = f"{distance:.4f}" if distance is not None else "—"
            lines.append(f"| {name} / {task} | {metrics['examples']} | {why} | {goal} | {current} | "
                f"{_percent(next_score['position_1']['action']['rate'])} | "
                f"{_percent(next_score['position_2']['action']['rate'])} | "
                f"{_percent(next_score['position_3']['action']['rate'])} | "
                f"{_percent(next_score['mean_position_accuracy']['action'])} | {distance_text} |")
    lines += ["", "WHY retains the Codex narrated-purpose rubric; Claude contributed action-mapping heuristics. GOAL measures active annotated activity at its original scope. Identical references can be resolved directly; nonidentical wording is never automatically marked wrong.", "",
        "NEXT positions 1–3 are primary. Full ordered sequence, verb/noun scores, classic Levenshtein and restricted Damerau/OSA are in summary.json. One forecast of three class actions is not INSIGHT's best-of-five twenty-action benchmark.", "",
        "Every model's intended cohort retains absent/failed predictions. Missing gold components affect only their own denominators. WHY source exclusions are shared. GOAL annotation absence is not a negative label.", "",
        f"Semantic decisions still needed: {summary['pending_packet_count']} unique packets. Supply hash-bound automated judgments and rerun; visualization and human calibration remain deferred."]
    return "\n".join(lines) + "\n"


def export_metrics(path, summary):
    """One flat, named row per metric for comparing future result files."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "task", "field", "metric", "component",
                                                  "position", "numerator", "denominator", "value", "status"))
        writer.writeheader()
        for name, model in summary["models"].items():
            for task, scores in model["by_task"].items():
                def emit(field, metric, value, component="", position="", numerator="", denominator="", status="computed"):
                    writer.writerow(dict(model=name, task=task, field=field, metric=metric, value=value,
                        component=component, position=position, numerator=numerator, denominator=denominator, status=status))
                for field in ("why", "goal"):
                    if field == "why" and task != "explain":
                        continue
                    score = scores[field]
                    for metric in ("correct_rate", "ordinal_score", "judgment_completion_rate"):
                        numerator = (score["counts"]["correct"] if metric == "correct_rate" else
                                     2 * score["counts"]["correct"] + score["counts"]["partial"] if metric == "ordinal_score" else
                                     score["completed_judgments"])
                        denominator = score["eligible"] * (2 if metric == "ordinal_score" else 1)
                        emit(field, metric, score[metric], numerator=numerator, denominator=denominator,
                             status=score["status"] if denominator else "unmeasured")
                    for label, count in score["counts"].items():
                        emit(field, label + "_count", count, denominator=score["eligible"], status=score["status"])
                    for flag, count in score["flags"].items():
                        coverage = score["flag_judgment_counts"][flag]
                        emit(field, flag + "_count", count, denominator=coverage,
                             status=score["status"] if coverage else "unmeasured")
                    for metric in ("eligible", "excluded_source", "missing_prediction", "pending"):
                        emit(field, metric + "_count", score[metric], status=score["status"])
                def accuracy(field, metric, score, component, position=""):
                    status = "computed" if score["denominator"] else "unmeasured"
                    emit(field, metric, score["rate"], component, position, score["correct"], score["denominator"], status)
                    coverage_name = "sequence_mapping_coverage" if metric == "exact_ordered_sequence_accuracy" else "mapping_coverage"
                    emit(field, coverage_name, score["mapped_predictions"] / score["denominator"] if score["denominator"] else None,
                         component, position, score["mapped_predictions"], score["denominator"], status)
                if task == "track":
                    for component, score in scores["current_action"].items():
                        accuracy("current_action", "accuracy", score, component)
                forecast = scores["next"]
                for position in (1, 2, 3):
                    for component, score in forecast[f"position_{position}"].items():
                        accuracy("next", "position_accuracy", score, component, position)
                for component, rate in forecast["mean_position_accuracy"].items():
                    emit("next", "mean_position_accuracy", rate, component,
                         status="computed" if rate is not None else "unmeasured")
                for component, score in forecast["exact_ordered_sequence"].items():
                    accuracy("next", "exact_ordered_sequence_accuracy", score, component)
                for metric in ("normalized_edit_distance", "normalized_damerau_osa"):
                    for component, score in forecast[metric].items():
                        emit("next", metric, score["mean"], component,
                             numerator=score["normalized_distance_sum"], denominator=score["denominator"],
                             status="computed" if score["denominator"] else "unmeasured")
                for metric, count in forecast["forecast_diagnostics"].items():
                    emit("next", metric, count)
            for metric, count in model.get("diagnostics", {}).items():
                writer.writerow(dict(model=name, task="all", field="generation", metric=metric,
                    component="", position="", numerator="", denominator="", value=count, status="computed"))


def run(config_path, output=None, judgments_path=None, mapper_path=None, config_override=None):
    config_path = Path(config_path).resolve()
    config = config_override if config_override is not None else json.loads(config_path.read_text())
    # Portable configurations resolve their assets and input files beside them.
    root = (config_path.parent / config.get("project_root", ".")).resolve()
    path = lambda value: (root / value).resolve()
    output = Path(output).resolve() if output is not None else path(config.get("output", "results"))
    source_path = path(config["references"])
    legacy_path = path(config["legacy_why_judgments"]) if config.get("legacy_why_judgments") else None
    reference_format = config.get("reference_format", "prepared")
    if reference_format not in {"prepared", "scoring"}:
        raise ValueError("reference_format must be prepared or scoring")
    references = load_reference_bundle(source_path) if reference_format == "prepared" else load_references(source_path, legacy_path)
    if {r["task"] for r in references} - {"explain", "track"}:
        raise ValueError("Unexpected task in references")
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else {}
    resolved_mapper = Path(mapper_path).resolve() if mapper_path else path(config["mapper"]) if config.get("mapper") else None
    mapper = ActionMapper.from_dict(json.loads(resolved_mapper.read_text())) if resolved_mapper else build_mapper(
        path(config["annotations_root"]), path(config["split_manifest"]))
    legacy_rows = rows(legacy_path) if legacy_path else []
    legacy_by_model = {}
    for row in legacy_rows:
        model = legacy_by_model.setdefault(row["model"], {})
        if row["id"] in model:
            raise ValueError("Duplicate legacy WHY judgment")
        model[row["id"]] = row
    judgments = unique_rows(rows(judgments_path), "packet_id") if judgments_path else {}
    summary = {"version": VERSION, "models": {}, "comparisons": {}}
    all_records, pending, links = {}, {}, []
    input_hashes = {str(config_path): sha256(config_path), str(source_path): sha256(source_path)}
    if legacy_path:
        input_hashes[str(legacy_path)] = sha256(legacy_path)
    if resolved_mapper:
        input_hashes[str(resolved_mapper)] = sha256(resolved_mapper)
    names = set()
    for model in config["models"]:
        name = model["name"]
        if name in names or not re_safe_name(name):
            raise ValueError("Model names must be unique simple identifiers")
        names.add(name)
        if (not isinstance(model["tasks"], list) or not model["tasks"] or
                set(model["tasks"]) - {"explain", "track"} or
                len(model["tasks"]) != len(set(model["tasks"]))):
            raise ValueError("Model tasks must be a nonempty unique Explain/Track list")
        intended = [r for r in references if r["task"] in model["tasks"]]
        if not intended:
            raise ValueError("Empty intended model cohort")
        prediction_path = path(model["predictions"])
        predictions = unique_rows(rows(prediction_path))
        input_hashes[str(prediction_path)] = sha256(prediction_path)
        metrics, records, packets, model_links = evaluate_model(name, intended, predictions, mapper,
            legacy=legacy_by_model.get(model.get("legacy_model"), {}), judgments=judgments)
        summary["models"][name] = metrics
        all_records[name] = records
        pending.update({packet["packet_id"]: packet for packet in packets})
        links.extend(model_links)
    for before, after in config.get("comparisons", []):
        summary["comparisons"][f"{before}_to_{after}"] = paired_comparison(all_records[before], all_records[after])
    summary["pending_packet_count"] = len(pending)
    summary["completed_semantic_evaluation"] = not pending
    summary["limitations"] = ["Action mapper not yet validated on generated paraphrases.",
        "WHY source decisions and semantic ratings are provisional assistant assessments.",
        "GOAL uses heterogeneous annotated activity scopes, not a uniform session-intent hierarchy.",
        "Validation covers P07 only; annotated action history is supplied.",
        "Visualization and human checks deferred at user request."]
    code_paths = sorted(Path(__file__).parent.glob("*.py"))
    manifest = {"version": VERSION, "input_sha256": input_hashes,
        "code_sha256": {str(p): sha256(p) for p in code_paths},
        "mapper_sha256": digest(mapper.to_dict()), "reference_count": len(references),
        "semantic_rubric_sha256": digest({"why": WHY_RUBRIC, "goal": GOAL_RUBRIC}),
        "judge_input": "blind text agreement; no visual-support claims or future frames",
        "model_inference_calls": 0, "human_review_performed": False,
        "existing_reports_modified": False, "pending_packet_count": len(pending)}
    manifest["standalone_bundle"] = True
    manifest["config"] = config
    manifest["config_sha256"] = digest(config)
    if judgments_path:
        manifest["input_sha256"][str(Path(judgments_path).resolve())] = sha256(judgments_path)
    write_json(output / "manifest.json", manifest)
    write_json(output / "mapper.json", mapper.to_dict())
    write_rows(output / "references.jsonl", references)
    profiles = {}
    for reference in references:
        goal = reference["goal"]
        if not goal["eligible"]:
            continue
        profile = profiles.setdefault(goal["reference"], dict(goal["profile"]))
        profile["source_annotation_ids"] = sorted(set(profile["source_annotation_ids"]) |
                                                  set(goal["profile"]["source_annotation_ids"]))
    write_json(output / "goal_profiles.json", profiles)
    # Only prune files owned by the previous evaluation's model list.
    if previous.get("version") == VERSION:
        for old_name in set(previous.get("models", {})) - set(all_records):
            if re_safe_name(old_name):
                (output / "per_example" / f"{old_name}.jsonl").unlink(missing_ok=True)
    for name, records in all_records.items():
        write_rows(output / "per_example" / f"{name}.jsonl", records)
    write_rows(output / "semantic_packets.jsonl", sorted(pending.values(), key=lambda x: x["packet_id"]))
    write_rows(output / "semantic_packet_links.jsonl", links)
    write_json(output / "summary.json", summary)
    export_metrics(output / "metrics.csv", summary)
    (output / "report.md").write_text(report_text(summary))
    return summary


def re_safe_name(value):
    return isinstance(value, str) and value and all(c.isalnum() or c in "_-" for c in value)


def judge_packets(packet_path, output_path, command, judge_name):
    """Optional generic judge adapter: one JSON packet in, one decision out.

    No default command/provider is executed. An explicitly supplied executable
    can call an automated judge; the subprocess receives no model identity.
    """
    if not command or not judge_name:
        raise ValueError("Judge executable and identity are required")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    existing = unique_rows(rows(output_path), "packet_id") if Path(output_path).exists() else {}
    with Path(output_path).open("a") as handle:
        for packet in unique_rows(rows(packet_path), "packet_id").values():
            if packet["packet_id"] in existing:
                if existing[packet["packet_id"]]["prompt_sha256"] != packet["prompt_sha256"]:
                    raise ValueError("Existing decision is bound to another prompt")
                validate_judgment(existing[packet["packet_id"]]["judgment"], packet["field"])
                continue
            result = subprocess.run(command, input=json.dumps(packet, ensure_ascii=False),
                                    text=True, capture_output=True, check=True, timeout=300)
            decision = validate_judgment(json.loads(result.stdout), packet["field"])
            record = {"packet_id": packet["packet_id"], "prompt_sha256": packet["prompt_sha256"],
                      "judge": judge_name, "assessment_status": "provisional_automated_assessment",
                      "judgment": decision}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score archived outputs; prepare pending semantic packets")
    mode = score.add_mutually_exclusive_group()
    mode.add_argument("--config", type=Path, help="Configuration with paths relative to itself")
    mode.add_argument("--predictions", type=Path, help="Evaluate one new result JSONL with bundled references")
    score.add_argument("--name", default="new_results", help="Name for a single predictions file")
    score.add_argument("--tasks", nargs="+", choices=("explain", "track"),
                       help="Intended tasks for single-file evaluation; required with --predictions")
    score.add_argument("--references", type=Path, help="Prepared reference JSONL for single-file evaluation")
    score.add_argument("--output", type=Path, help="Defaults to bundle/results")
    score.add_argument("--judgments", type=Path)
    score.add_argument("--mapper", type=Path, help="Reuse a frozen mapper snapshot")
    score.add_argument("--require-complete", action="store_true")
    judge = commands.add_parser("judge", help="Run an explicitly supplied automated judge executable")
    judge.add_argument("--packets", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-name", required=True)
    judge.add_argument("executable", nargs=argparse.REMAINDER)
    commands.add_parser("test", help="Run the bundled CPU tests")
    args = parser.parse_args()
    if args.command == "test":
        suite = unittest.defaultTestLoader.discover(str(BUNDLE_ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    elif args.command == "judge":
        command = args.executable[1:] if args.executable[:1] == ["--"] else args.executable
        judge_packets(args.packets, args.output, command, args.judge_name)
    else:
        config_path = args.config or BUNDLE_ROOT / "config.json"
        override = None
        if args.predictions:
            if not args.tasks:
                parser.error("--predictions requires --tasks to keep the intended cohort fixed")
            config_path = BUNDLE_ROOT / "config.json"
            override = {"references": str(args.references.resolve()) if args.references else "assets/references.jsonl",
                "reference_format": "prepared", "mapper": "assets/mapper.json", "output": "results",
                "models": [{"name": args.name, "tasks": args.tasks,
                            "predictions": str(args.predictions.resolve())}], "comparisons": []}
        elif args.tasks or args.references:
            parser.error("--tasks and --references apply to --predictions mode")
        summary = run(config_path, args.output, args.judgments, args.mapper, override)
        resolved_output = args.output.resolve() if args.output else (config_path.parent / (override or json.loads(config_path.read_text())).get("output", "results")).resolve()
        print(json.dumps({"report": str(resolved_output / "report.md"),
                          "models": list(summary["models"]),
                          "pending_semantic_packets": summary["pending_packet_count"]}))
        if args.require_complete and not summary["completed_semantic_evaluation"]:
            raise SystemExit(2)
