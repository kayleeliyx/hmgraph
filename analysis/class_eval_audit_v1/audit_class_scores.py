"""Reproduce Claude's scorer and isolate extraction, gold, and lexicon effects.

CPU-only diagnostic. Predictions and the frozen evaluation are never changed.
The retained mapper is heuristic; category agreement is not validated semantics.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import re
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recover_field(raw, field):
    """Recover a unique field ending at its close tag or the next field boundary."""
    raw = raw or ""
    if field == "action":
        # ACTION tags nested inside NEXT are future actions, not duplicates of
        # the current-action field.
        raw = raw.split("<next>", 1)[0]
    opening, closing = f"<{field}>", f"</{field}>"
    if raw.count(opening) != 1 or raw.count(closing) > 1:
        return {"text": None, "status": "missing_or_ambiguous"}
    start = raw.index(opening) + len(opening)
    tail = raw[start:]
    boundary = re.search(r"</?[A-Za-z][A-Za-z0-9_]*>", tail)
    body = tail[:boundary.start()] if boundary else tail
    if not body.strip():
        return {"text": None, "status": "empty"}
    if closing in raw:
        if not boundary or boundary.group() != closing:
            return {"text": None, "status": "crossed_or_ambiguous"}
        status = "closed"
    else:
        status = "recovered_unclosed"
    return {"text": body.strip(), "status": status}


def recover_next(raw, action_detector=None):
    """Read all NEXT blocks; preserve chronological position and missing slots.

    Recover semicolon, newline, numbered-list, literal-list, timestamped-list,
    and nested ACTION representations.
    Do not infer actions from references. Split a comma list only when all three
    parts have an action verb under the fixed training/vocabulary mapper.
    """
    raw = raw or ""
    actions = []
    statuses = []
    for match in re.finditer(r"<next>", raw):
        tail = raw[match.end():]
        boundary = re.search(r"</?next>|<(?:why|goal)>", tail)
        body = tail[:boundary.start()] if boundary else tail
        statuses.append("closed" if boundary and boundary.group() == "</next>" else "unclosed")
        # Some untuned models put explicit action/time children inside NEXT.
        # Recover these values rather than treating the child tags as a stop.
        children = re.findall(r"<action>\s*([^<>]+?)\s*</action>", body, re.DOTALL)
        if children and body.count("<action>") == body.count("</action>") == len(children):
            actions.extend(child.strip() for child in children)
            continue
        body = re.sub(r"<(?:time|timestamp)>[^<>]*</(?:time|timestamp)>", "", body)
        try:
            value = ast.literal_eval(body.strip())
        except (ValueError, TypeError, SyntaxError):
            value = None
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            parts = value
        else:
            times = list(re.finditer(r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]", body))
            if len(times) >= 2:
                parts = [body[m.start():times[i+1].start() if i+1 < len(times) else len(body)].strip(" ,;\n]")
                         for i, m in enumerate(times)]
                # Preserve each timestamp's opening '['; norm() removes the
                # balanced timestamp before category mapping.
                parts = ["[" + part if not part.startswith("[") else part for part in parts]
            else:
                body = re.sub(r"(?<![\d.])\b\d+[.)]\s+(?=[A-Za-z])", "\n", body)
                parts = re.split(r"[;\n]", body)
                nonempty = [part.strip() for part in parts if part.strip()]
                if len(nonempty) == 1 and action_detector:
                    comma_parts = [part.strip() for part in nonempty[0].split(",")]
                    if len(comma_parts) == 3 and all(action_detector(part) for part in comma_parts):
                        parts = comma_parts
        for part in parts:
            part = re.sub(r"^\s*[-*•]\s*", "", part).strip()
            if part:
                actions.append(part)
    return {"actions": actions[:3], "total_actions": len(actions),
            "block_count": len(statuses), "block_statuses": statuses}


def class_id(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return int(value) if math.isfinite(value) and value.is_integer() else None


def metadata_pair(action):
    pairs = action.get("main_action_class_ids", []) if action else []
    p = pairs[0] if len(pairs) == 1 else []
    return (class_id(p[0]), class_id(p[1]) if len(p) > 1 else None) if p else (None, None)


def normalize_pair(pair):
    return tuple(class_id(x) for x in pair) if pair else (None, None)


def metric(items):
    result = {}
    for name, positions in (("verb", (0,)), ("noun", (1,)), ("action", (0, 1))):
        eligible = [(p, g) for p, g in items if all(g[i] is not None for i in positions)]
        correct = sum(all(p[i] is not None and p[i] == g[i] for i in positions) for p, g in eligible)
        mapped = sum(all(p[i] is not None for i in positions) for p, _ in eligible)
        result[name] = {"correct": correct, "denominator": len(eligible),
                        "rate": correct / len(eligible) if eligible else None,
                        "mapped_predictions": mapped}
    return result


def score(records, stages):
    out = {}
    for stage in stages:
        track = [r for r in records if r["task"] == "track" and stage in r["predictions"]]
        stages_result = {"current_action_track": metric([
            (r["predictions"][stage]["current"], r["gold_current"]) for r in track])}
        next_result = {}
        for task in ("all", "track", "explain"):
            cohort = [r for r in records if stage in r["predictions"] and
                      (task == "all" or r["task"] == task)]
            next_result[task] = {f"position_{i+1}": metric([
                (r["predictions"][stage]["next"][i], r["gold_next"][i]) for r in cohort])
                                for i in range(3)}
            eligible = [r for r in cohort if all(None not in p for p in r["gold_next"])]
            correct = sum(r["predictions"][stage]["next"] == r["gold_next"] for r in eligible)
            next_result[task]["ordered_triple"] = {"correct": correct,
                "denominator": len(eligible), "rate": correct / len(eligible) if eligible else None}
        stages_result["next_actions"] = next_result
        stages_result["current_action_by_video"] = {
            vid: metric([(r["predictions"][stage]["current"], r["gold_current"])
                         for r in track if r["video_id"] == vid])["action"]
            for vid in sorted({r["video_id"] for r in track})}
        out[stage] = stages_result
    return out


def training_lexicons(scope, annotations, video_ids):
    lexicons = []
    fit = scope["df"][scope["df"]["video_id"].isin(video_ids)]
    for kind, phrase_col, class_col in (("verb", "verb_s", "vc"), ("noun", "noun_s", "nc")):
        lex = defaultdict(Counter)
        for phrase, value in zip(fit[phrase_col], fit[class_col]):
            value = class_id(value)
            if value is not None:
                lex[phrase][value] += 1
        import pandas as pd
        table = pd.read_csv(annotations / "narrations-and-action-segments" / f"HD_EPIC_{kind}_classes.csv")
        for _, row in table.iterrows():
            for instance in ast.literal_eval(row["instances"]):
                if ":" in instance:
                    continue
                phrase = scope["norm"](instance)
                if phrase and phrase not in lex:
                    lex[phrase][int(row["id"])] += 1
        lexicons.append({s: c.most_common(1)[0][0] for s, c in lex.items()})
    return lexicons, {"fit_videos": len(video_ids), "fit_annotation_rows": len(fit),
                      "verb_aliases": len(lexicons[0]), "noun_aliases": len(lexicons[1]),
                      "policy": "initial-SFT videos plus official class definitions; no held-out narration mining"}


def self_checks():
    assert recover_field("<action>place mug\n<goal>put dishes away", "action")["text"] == "place mug"
    assert recover_field("<action>open drawer</action><next>one</next>", "action")["text"] == "open drawer"
    assert recover_field("<action>open<action>close", "action")["text"] is None
    assert recover_field("<action>open<goal>x</action>", "action")["text"] is None
    assert recover_next("<next>a</next><next>b</next><next>c</next>")["actions"] == ["a", "b", "c"]
    assert recover_next("<next>a\n<next>b\n<next>c")["actions"] == ["a", "b", "c"]
    assert recover_next("<next>[12.350, 13.400] pick up mug\nclose cupboard</next>")["actions"] == ["[12.350, 13.400] pick up mug", "close cupboard"]
    assert recover_next("<next>1. pick up mug 2. place mug 3. close cupboard</next>")["actions"] == ["pick up mug", "place mug", "close cupboard"]
    assert recover_next('<next>["a", "b", "c"]</next>')["actions"] == ["a", "b", "c"]
    assert recover_next("<next>['a', 'b', 'c']</next>")["actions"] == ["a", "b", "c"]
    assert recover_next("<next><action>a</action><action>b</action><action>c</action></next>")["actions"] == ["a", "b", "c"]
    assert recover_next("<next><time>1.00</time>a\n<time>2.00</time>b\n<time>3.00</time>c</next>")["actions"] == ["a", "b", "c"]
    assert len(recover_next("<next>[1.00, 2.00] a, [2.00, 3.00] b, [3.00, 4.00] c</next>")["actions"]) == 3
    assert recover_field("<action>a</action><next><action>b</action></next>", "action")["text"] == "a"
    assert recover_next("<next>open cupboard, pick up mug, close cupboard</next>",
                        lambda phrase: phrase.split()[0] in {"open", "pick", "close"})["total_actions"] == 3
    assert recover_next("<next>mix flour, sugar, eggs</next>",
                        lambda phrase: phrase.split()[0] == "mix")["total_actions"] == 1
    assert metric([((None, None), (1, None))])["action"]["denominator"] == 0
    assert metric([((None, None), (1, 2))])["action"]["correct"] == 0
    assert metadata_pair({"main_action_class_ids": [[15]]}) == (15, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--claude-scorer", type=Path, required=True)
    parser.add_argument("--supplied-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_checks()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.claude_scorer.read_bytes()
    archived = args.output / "score_hdepic_classes.original.py"
    archived.write_bytes(source)
    follow = args.project / "runs/hdepic_sft_8b_v1/review_followup_v1"
    reproduced = args.output / "claude_reproduced"
    previous_argv = sys.argv
    try:
        sys.argv = [str(archived), str(args.annotations), str(follow), str(reproduced)]
        with (args.output / "original_execution.log").open("w") as log, contextlib.redirect_stdout(log):
            loaded = runpy.run_path(str(archived))
    finally:
        sys.argv = previous_argv
    scope = loaded["parse_action"].__globals__
    supplied = json.loads(args.supplied_summary.read_text())
    original_summary = json.loads((reproduced / "class_level_summary.json").read_text())
    headline_equal = all(supplied["current_action_track"][s] == original_summary["current_action_track"][s]
                         and supplied["next_actions"][t][s] == original_summary["next_actions"][t][s]
                         for s in ("base", "epoch1", "epoch2") for t in ("all", "track", "explain"))
    assert headline_equal
    scoring = rows(args.project / "datasets/hdepic_sft_v2_repaired/swift/validation.scoring.jsonl")
    assert len(scoring) == len({r["id"] for r in scoring}) == 2209
    by_id = {r["id"]: r for r in scoring}
    raw_predictions = {s: {r["id"]: r for r in rows(args.project / f"runs/hdepic_sft_8b_v1/predictions/{s}/generation.jsonl")}
                       for s in ("base", "epoch1", "epoch2")}
    for stage, run in (("qwen35", "hdepic_qwen35_35b_baseline_v1"),
                       ("qwen38", "hdepic_qwen38_27b_baseline_v1")):
        raw_predictions[stage] = {r["id"]: r for r in rows(args.project / f"runs/{run}/predictions/generation.jsonl")}
    for s in raw_predictions:
        expected = {r["id"] for r in scoring if r["task"] == "explain"} if s in ("qwen35", "qwen38") else set(by_id)
        assert set(raw_predictions[s]) == expected
    all_model_maps = (scope["verb_map"].copy(), scope["noun_map"].copy())
    manifest = json.loads((args.project / "datasets/hdepic_sft_v2_repaired/split_manifest.json").read_text())
    videos = set(manifest["partitions"]["initial_sft"]["video_ids"])
    restricted_maps, fit_info = training_lexicons(scope, args.annotations, videos)
    scope["verb_map"], scope["noun_map"] = restricted_maps
    examples = []
    extracted = {}
    diagnostics = defaultdict(Counter)
    for stage, predictions in raw_predictions.items():
        for identity, row in predictions.items():
            assert row["status"] == "completed"
            current = recover_field(row["prediction"], "action") if by_id[identity]["task"] == "track" else None
            nxt = recover_next(row["prediction"],
                               lambda phrase: class_id(scope["parse_action"](phrase)[0]) is not None)
            extracted[(stage, identity)] = {"current": current, "next": nxt}
            if current:
                diagnostics[stage]["current_" + current["status"]] += 1
            if nxt["block_count"] > 1:
                diagnostics[stage]["multiple_next_blocks"] += 1
            diagnostics[stage]["fewer_than_three_recovered_next_actions"] += nxt["total_actions"] < 3
            diagnostics[stage]["more_than_three_recovered_next_actions"] += nxt["total_actions"] > 3
            if stage == "base" and current and current["status"] == "recovered_unclosed" and len(examples) < 5:
                examples.append({"id": identity, "raw": row["prediction"], "recovered_action": current["text"]})
    variants = {}
    per_examples = {}
    for name, maps in (("recovered_fields_with_original_all_participant_mapper", all_model_maps),
                       ("recovered_fields_with_initial_sft_and_official_mapper", restricted_maps)):
        scope["verb_map"], scope["noun_map"] = maps
        records = []
        for metadata in scoring:
            identity = metadata["id"]
            rec = {"id": identity, "task": metadata["task"], "video_id": metadata["video_id"],
                   "gold_current": metadata_pair(metadata["current_action"]),
                   "gold_next": [metadata_pair(a) for a in metadata["future_actions"]], "predictions": {}}
            assert len(rec["gold_next"]) == 3
            for stage in raw_predictions:
                if identity not in raw_predictions[stage]:
                    continue
                extract = extracted[(stage, identity)]
                phrases = extract["next"]["actions"]
                current = extract["current"]
                rec["predictions"][stage] = {
                    "current": normalize_pair(scope["parse_action"](current["text"] if current else None)),
                    "next": [normalize_pair(scope["parse_action"](x)) for x in phrases]
                            + [(None, None)] * (3 - len(phrases)),
                    "current_text": current["text"] if current else None, "next_texts": phrases}
            records.append(rec)
        variants[name] = score(records, list(raw_predictions))
        per_examples[name] = records
        with (args.output / f"{name}.per_example.jsonl").open("w") as handle:
            for rec in records:
                handle.write(json.dumps(rec, allow_nan=False) + "\n")
    gold_diffs = []
    for old in scope["records"]:
        metadata = by_id[old["id"]]
        frozen_next = [metadata_pair(a) for a in metadata["future_actions"]]
        if [normalize_pair(p) for p in old["gold_next"]] != frozen_next:
            gold_diffs.append({"id": old["id"], "claude_next": [normalize_pair(p) for p in old["gold_next"]],
                               "frozen_next": frozen_next})
    report = {"scope": "CPU-only action-class audit; no WHY/goal semantic judgments, training, or inference",
              "claude_source_sha256": hashlib.sha256(source).hexdigest(),
              "supplied_all_model_headline_metrics_reproduced": headline_equal,
              "entire_supplied_summary_equal": supplied == original_summary,
              "entire_summary_difference": "one auxiliary oracle-bigram set-recall rounds to .084 here vs .085 supplied" if supplied != original_summary else None,
              "original_current_action": original_summary["current_action_track"],
              "original_next_actions": original_summary["next_actions"],
              "recovered_field_counts": dict(diagnostics), "examples": examples,
              "initial_sft_mapper_fit": fit_info, "gold_class_differences": gold_diffs,
              "variants": variants,
              "limitations": ["The retained free-text mapper is heuristic and unvalidated on model paraphrases.",
                              "Category agreement does not isolate video understanding from action history.",
                              "All validation cases are P07; related examples share nine videos.",
                              "Original row-wise bootstrap intervals should not be treated as video-cluster uncertainty.",
                              "The audit does not change frozen predictions, checkpoint selection, or original reports."],
              "main_evaluation_replaced": False, "self_checks_passed": True}
    (args.output / "audit_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Claude headline scores reproduced:", headline_equal)
    for variant, result in variants.items():
        print(variant)
        for stage, metrics in result.items():
            print(stage, "current", metrics["current_action_track"]["action"],
                  "next1", metrics["next_actions"]["all"]["position_1"]["action"])


if __name__ == "__main__":
    main()
