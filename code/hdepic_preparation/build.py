"""Build the fixed protocol and all text records before decoding any frames."""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pandas as pd

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, stable_hash
from code.e0.frames import uniform_indices, video_fingerprint

VERSION = "hdepic_causal_sft_v1"
PARTITIONS = ("initial_sft", "second_stage", "validation", "test")
DEFAULT_DATA = Path("/data/kailaic/HDEPIC/data")
DEFAULT_OUT = Path("/data/yaxuanli/ActGround/datasets/hdepic_sft_v1")
FRAME_SETTINGS = {"max_edge": 384, "jpeg_quality": 85, "colorspace": "RGB", "resize": "preserve_aspect_ratio_lanczos_if_needed", "jpeg_subsampling": 0}
MARKERS = re.compile(r"\b(?P<so_that>so that)\b|\b(?P<in_order_to>in order to)\b|\b(?P<because>because)\b|\b(?P<so_can>so (?:I|we|you) (?:can|could|may))\b(?!['’]t)", re.I)
CLAUSES = re.compile(r"\b(?:so that|in order to|because|using|by (?:holding|using|grasping|pulling|pushing|turning|moving)|while|in preparation for)\b", re.I)
STOP = {"a", "an", "the", "of", "in", "on", "to", "up", "down", "for", "with", "my", "left", "right", "both", "hand", "hands", "some", "it", "this", "that", "other", "another"}


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def text_fingerprint(out):
    paths = [out / "split_manifest.json", out / "frame_requests.json"] + [out / "text" / f"{p}.jsonl" for p in PARTITIONS]
    return stable_hash({str(p.relative_to(out)): sha256_file(p) for p in paths})


def stem(word):
    word = word.lower()
    irregular = {"held": "hold", "holding": "hold", "putting": "put", "taking": "take", "taken": "take", "shaking": "shake", "moving": "move", "placing": "place", "using": "use", "raising": "raise", "squeezing": "squeeze", "releasing": "release", "stirring": "stir", "cutting": "cut", "tapping": "tap", "flipping": "flip", "scrubbing": "scrub", "transferring": "transfer", "rinsing": "rinse", "wiping": "wipe", "sliding": "slide", "closing": "close", "pulling": "pull", "rolling": "roll", "filling": "fill", "spilling": "spill", "falling": "fall", "peeling": "peel", "handling": "handle", "tasting": "taste"}
    if word in irregular:
        return irregular[word]
    if word.endswith("ing") and len(word) > 5:
        root = word[:-3]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] else root
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-zA-Z]+", text) if w.lower() not in STOP]


def clean_action(row):
    """Use the structured main verb/object, never full narration or activity."""
    pairs = row["main_actions"]
    if not pairs:
        return ""
    verb, noun = pairs[0]
    noun = CLAUSES.split(str(noun), maxsplit=1)[0].strip(" ,.;")
    verb = str(verb).strip()
    return f"{verb} {noun}".strip()


def action_metadata(row):
    classes = row["main_action_classes"]
    return {"narration_id": row["unique_narration_id"], "video_id": row["video_id"], "action": clean_action(row), "start_sec": float(row["start_timestamp"]), "end_sec": float(row["end_timestamp"]), "main_actions": row["main_actions"], "verb_ids": [int(pair[0]) if pair else None for pair in classes], "noun_ids": [int(pair[1]) if len(pair) > 1 else None for pair in classes], "main_action_class_ids": classes, "pair_class_ids": row["pair_classes"], "class_label_status": "missing_main_action" if not classes else "missing_noun_class" if any(len(pair) < 2 for pair in classes) else "available"}


def extract_reason(row):
    text = row["narration"]
    hits = list(MARKERS.finditer(text))
    base = {"narration_id": row["unique_narration_id"], "participant": row["participant_id"], "video_id": row["video_id"], "original_narration": text, "target": action_metadata(row), "reason_text": None, "char_start": None, "char_end": None, "extraction_rule": "none", "status": "rejected", "rejection_category": None}
    if not hits:
        base["rejection_category"] = "no_explicit_marker"
        return base
    hit = hits[0]
    base["extraction_rule"] = hit.lastgroup
    start = hit.start()
    suffix = text[hit.end():]
    stop = re.search(r"[.!?]|\b(?:and then|then|however)\b", suffix, re.I)
    end = hit.end() + (stop.start() if stop else len(suffix))
    while end > start and text[end - 1].isspace():
        end -= 1
    base.update(reason_text=text[start:end], char_start=start, char_end=end)
    if len(hits) != 1:
        base["rejection_category"] = "multiple_reason_markers"
        return base
    if len(row["main_actions"]) != 1 or not clean_action(row):
        base["rejection_category"] = "missing_or_ambiguous_target_action"
        return base
    if not tokens(text[hit.end():end]):
        base["rejection_category"] = "empty_or_short_reason"
        return base
    # Bind the marker to an annotated verb/object in its own preceding sentence.
    # A separate sentence is allowed only with an explicit anaphoric bridge.
    prefix = text[:start]
    sentence_start = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?")) + 1
    clause = prefix[sentence_start:]
    # Standalone 'So you can hear ...' often describes an incidental observable
    # result. Do not treat an unlinked sentence as the action's purpose.
    if sentence_start and not clause.strip():
        base["rejection_category"] = "unlinked_reason_sentence"
        return base
    bridge = re.fullmatch(r"\s*(?:I do this|This is|I did this|This is done)\s*", clause, re.I)
    if bridge and sentence_start:
        prior = prefix[:sentence_start - 1]
        prior_start = max(prior.rfind("."), prior.rfind("!"), prior.rfind("?")) + 1
        clause = prior[prior_start:]
        sentence_start = prior_start
    words = list(re.finditer(r"[a-zA-Z]+", clause))
    wt = [stem(w.group()) for w in words]
    main_verb, main_noun = row["main_actions"][0]
    verb_token = stem(re.findall(r"[a-zA-Z]+", str(main_verb))[0])
    noun_tokens = set(tokens(str(main_noun)))
    verb_positions = [i for i, w in enumerate(wt) if w == verb_token]
    if not verb_positions or (noun_tokens and not noun_tokens.intersection(wt)):
        base["rejection_category"] = "target_not_in_marker_clause"
        return base
    # Reject markers following another parsed action: they might explain that
    # action instead. This is conservative and deliberately sacrifices recall.
    anchor = verb_positions[-1]
    tail = clause[words[anchor].end():]
    if hit.lastgroup == "because" and re.search(r"\b(?:fail\w*|but|however|although|when)\b|\b(?:won't|can't|cannot|doesn't|didn't)\b", clause, re.I):
        base["rejection_category"] = "because_explains_failure_or_other_event"
        return base
    others = []
    for verb, noun in row["pairs"]:
        if [str(verb), str(noun)] == [str(main_verb), str(main_noun)]:
            continue
        parts = re.findall(r"[a-zA-Z]+", str(verb))
        if not parts:
            continue
        v = stem(parts[0])
        if v == verb_token:
            continue
        positions = [i for i, w in enumerate(wt) if w == v and i > anchor]
        if positions:
            others.append([str(verb), str(noun)])
    if others:
        base["rejection_category"] = "intervening_other_action"
        base["intervening_pairs"] = others
        return base
    # Causal 'because' about a later event must not be rebound to the main action.
    if hit.lastgroup == "because" and re.search(r"\b(?:will|would|might)\b", clause[words[anchor].end():], re.I):
        base["rejection_category"] = "because_explains_other_event"
        return base
    base.update(status="accepted_pending_human_review", rejection_category=None, binding_rule="same_sentence_main_verb_and_object_no_intervening_parsed_action_v1", binding_clause=text[sentence_start:start])
    return base


def probe_video(path):
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames", "-of", "json", str(path)]))["streams"][0]
    # Planned source indices depend on this inspected CFR contract. Actual PTS
    # from the decoder are checked independently before records are exported.
    if data["r_frame_rate"] != "30/1" or data["avg_frame_rate"] != "30/1" or float(data["start_time"]) != 0 or data["time_base"] != "1/15360":
        raise ValueError(f"Unsupported time base; cannot assume frame timestamps: {path}")
    data.update(video_id=path.stem, participant=path.stem.split("-")[0], path=str(path.resolve()), duration_sec=float(data["duration"]), frame_count=int(data["nb_frames"]), fingerprint=video_fingerprint(path))
    data["cache_id"] = stable_hash({"protocol": VERSION, "source": data["fingerprint"], "settings": FRAME_SETTINGS})
    return data


def build_manifest(out, inventory, source_hash):
    path = out / "split_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["protocol"] != VERSION or manifest["narrations_sha256"] != source_hash:
            raise ValueError("Existing fixed manifest incompatible: preserve it and use a new output directory")
        assigned = [v for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(inventory):
            raise ValueError("Fixed manifest coverage changed")
        return manifest
    by_person = defaultdict(list)
    for vid in sorted(inventory):
        by_person[inventory[vid]["participant"]].append(vid)
    training = [f"P{i:02}" for i in range(1, 7)]
    # Largest-remainder allocation fixes the total nearest to one sixth, while
    # distributing reservations across every training participant.
    count = round(sum(len(by_person[p]) for p in training) / 6)
    allocations = {p: len(by_person[p]) // 6 for p in training}
    for p in sorted(training, key=lambda p: (-(len(by_person[p]) % 6), p))[:count - sum(allocations.values())]:
        allocations[p] += 1
    second = []
    for person in training:
        ranked = sorted(by_person[person], key=lambda v: stable_hash([VERSION, "second_stage_seed_20261002", v]))
        second.extend(ranked[:allocations[person]])
    partitions = {"initial_sft": [v for p in training for v in by_person[p] if v not in second], "second_stage": second, "validation": by_person["P07"], "test": by_person["P08"] + by_person["P09"]}
    manifest = {"protocol": VERSION, "status": "pending_human_reason_label_review", "narrations_sha256": source_hash, "participant_policy": {"training": training, "validation": ["P07"], "test": ["P08", "P09"]}, "second_stage_selection": {"method": "largest_remainder_participant_allocation_then_sha256_video_ranking", "seed": "20261002", "training_video_count": sum(len(by_person[p]) for p in training), "video_count": len(second), "allocations": allocations, "fraction": len(second) / sum(len(by_person[p]) for p in training), "shared_by": ["extra_sft", "grpo"]}, "partitions": {p: {"video_ids": sorted(vids), "video_count": len(vids), "participants": sorted(set(inventory[v]["participant"] for v in vids)), "duration_sec": sum(inventory[v]["duration_sec"] for v in vids), "duration_hours": sum(inventory[v]["duration_sec"] for v in vids) / 3600} for p, vids in partitions.items()}}
    atomic_write_json(path, manifest)
    return manifest


def activity_at(activities, timestamp):
    hits = [a for a in activities if a["start_sec"] <= timestamp < a["end_sec"]]
    if len(hits) == 1 and hits[0]["label"]:
        return hits[0], "available"
    return None, "ambiguous_activity" if len(hits) > 1 else "missing_activity"


def frame_plan(video, t, out):
    decimal_t = Decimal(str(t))
    lo = math.ceil(max(Decimal(0), decimal_t - Decimal(5)) * 30)
    hi = min(video["frame_count"] - 1, math.floor(decimal_t * 30))
    available = hi - lo + 1
    if available < 16:
        return None
    return [{"index": i, "source_frame_index": lo + j, "planned_timestamp_sec": (lo + j) / 30, "path": str((out / "frames" / video["video_id"] / video["cache_id"] / f"{lo + j:08d}.jpg").resolve())} for i, j in enumerate(uniform_indices(available, 16))]


def record(row, task, t, history, future, goal, goal_status, frames, partition, reason):
    current = action_metadata(row)
    target = {"current_action": current["action"]} if task == "track" else {"reason": reason["reason_text"]}
    if goal:
        target["goal"] = goal["label"]
    target["next_actions"] = [clean_action(a) for a in future]
    return {"id": f"{row['video_id']}:{task}:{row['unique_narration_id'] if task == 'explain' else format(t,'.3f')}", "partition": partition, "participant": row["participant_id"], "video_id": row["video_id"], "task": task, "query_time_sec": t, "input": {"history": [{"action": clean_action(a), "start_sec": float(a["start_timestamp"]), "end_sec": float(a["end_timestamp"])} for a in history], **({"action_to_explain": current["action"]} if task == "explain" else {})}, "frames": frames, "target": target, "scoring_metadata": {"current_action": current, "history_actions": [action_metadata(a) for a in history], "future_actions": [action_metadata(a) for a in future], "goal_status": goal_status, "goal_annotation": goal, "reason_annotation": reason if task == "explain" else None, "review_status": "pending_human_reason_label_review" if task == "explain" else "annotation_derived"}}


def stratified(rows, count, group_key):
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    for group in groups.values():
        group.sort(key=lambda r: stable_hash([VERSION, "review", r["narration_id"]]))
    selected = []
    while len(selected) < count and groups:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == count:
                break
    return selected


def write_review(out, reasons, used):
    accepted = [r for r in reasons if r["narration_id"] in used]
    rejected_markers = [r for r in reasons if r["status"] == "rejected" and r["extraction_rule"] != "none"]
    rejected_none = [r for r in reasons if r["extraction_rule"] == "none"]
    rows = stratified(accepted, 50, lambda r: (r["extraction_rule"], r["participant"])) + stratified(rejected_markers, 15, lambda r: (r["extraction_rule"], r["participant"])) + stratified(rejected_none, 5, lambda r: (r["participant"],))
    fields = ["status", "partition", "participant", "video_id", "narration_id", "original_narration", "target_action", "start_sec", "end_sec", "extracted_span", "char_start", "char_end", "extraction_rule", "binding_clause", "rejection_category", "valid purpose?", "correct target action?", "reviewer notes"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({"status": r["status"], "partition": r["partition"], "participant": r["participant"], "video_id": r["video_id"], "narration_id": r["narration_id"], "original_narration": r["original_narration"], "target_action": r["target"]["action"], "start_sec": r["target"]["start_sec"], "end_sec": r["target"]["end_sec"], "extracted_span": r["reason_text"], "char_start": r["char_start"], "char_end": r["char_end"], "extraction_rule": r["extraction_rule"], "binding_clause": r.get("binding_clause", ""), "rejection_category": r["rejection_category"]})
    if len(rows) != 70 or len(rows[:50]) != 50:
        raise ValueError("Not enough examples for the requested review")
    atomic_write_text(out / "reason_review.csv", buffer.getvalue())


def benchmark_audit(project, out, partition_of):
    reference = project / "runs/e0/hd_epic_why_pilot/splits"
    dev = json.loads((reference / "dev100.json").read_text())
    reserved = json.loads((reference / "reserved_ids.json").read_text())
    rows = []
    for role, items in (("dev", dev["records"]), ("historically_reserved", [{"question_id": q, **c} for q, c in reserved["source_clips"].items()])):
        for item in items:
            p = partition_of[item["source_video_id"]]
            rows.append({"question_id": item["question_id"], "video_id": item["source_video_id"], "historical_role": role, "partition": p, "video_disjoint_from_training": p in ("validation", "test"), "eligible_for_fresh_heldout_evaluation": p in ("validation", "test") and role == "historically_reserved", "prior_development_use": role == "dev"})
    audit = {"policy": "Only historical reserved questions on validation/test videos remain eligible for fresh held-out evaluation. Previously used dev questions on those participants are video-disjoint, but remain development-exposed.", "questions": rows, "counts": dict(Counter(f"{r['historical_role']}:{r['partition']}" for r in rows)), "eligible_question_ids": {p: sorted(r["question_id"] for r in rows if r["partition"] == p and r["eligible_for_fresh_heldout_evaluation"]) for p in ("validation", "test")}, "historical_source_videos_by_partition": {p: sorted(set(r["video_id"] for r in rows if r["partition"] == p)) for p in PARTITIONS}, "reference_hashes": {str(f): sha256_file(f) for f in (reference / "dev100.json", reference / "reserved_ids.json")}}
    atomic_write_json(out / "benchmark_overlap.json", audit)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(out / "benchmark_overlap.csv", buffer.getvalue())


def build(data=DEFAULT_DATA, out=DEFAULT_OUT):
    out.mkdir(parents=True, exist_ok=True)
    anno = data / "hd-epic-annotations"
    narration_path = anno / "narrations-and-action-segments/HD_EPIC_Narrations.pkl"
    df = pd.read_pickle(narration_path)
    inventory = {r["video_id"]: r for r in ThreadPoolExecutor(8).map(probe_video, sorted((data / "HD-EPIC/Videos").glob("*/*.mp4")))}
    if set(df.video_id) != set(inventory) or not df.unique_narration_id.is_unique:
        raise ValueError("Video coverage or narration uniqueness failed")
    atomic_write_json(out / "video_inventory.json", inventory)
    manifest = build_manifest(out, inventory, sha256_file(narration_path))
    partition_of = {v: p for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]}
    activities = defaultdict(list)
    source_issues = []
    for row in df.to_dict("records"):
        label_status = action_metadata(row)["class_label_status"]
        if label_status != "available":
            source_issues.append({"source": str(narration_path), "narration_id": row["unique_narration_id"], "category": label_status})
    source_hashes = {str(narration_path): sha256_file(narration_path)}
    for path in sorted((anno / "high-level/activities").glob("*.csv")):
        source_hashes[str(path)] = sha256_file(path)
        for line, a in enumerate(csv.DictReader(path.open()), 2):
            original_video_id = a["video_id"]
            # Inspected release typo: the date omits the zero in June. Keep the
            # source untouched and retain both identifiers in the scoring data.
            if original_video_id == "P09-2024623-154547":
                a["video_id"] = "P09-20240623-154547"
                source_issues.append({"source": str(path), "line": line, "category": "activity_video_id_typo", "original_video_id": original_video_id, "resolved_video_id": a["video_id"]})
            if a["video_id"] not in inventory:
                source_issues.append({"source": str(path), "line": line, "category": "unmatched_activity_video_id", "original_video_id": original_video_id})
                continue
            end = inventory[a["video_id"]]["duration_sec"] if a["end_time"].lower() == "end" else float(a["end_time"])
            activities[a["video_id"]].append({"annotation_id": f"{path.name}:{line}", "original_video_id": original_video_id, "resolved_video_id": a["video_id"], "label": a["high_level_activity_label"].strip() or None, "recipe_id": a["recipe_id"] or None, "start_sec": float(a["start_time"]), "end_sec": end})
    atomic_write_json(out / "source_annotation_issues.json", source_issues)
    records = {p: [] for p in PARTITIONS}
    reasons, used = [], set()
    counts = Counter()
    skipped = []
    def skip(p, person, task, category, vid, identifier):
        counts[(p, person, task, "skipped", category)] += 1
        skipped.append({"partition": p, "participant": person, "task": task, "category": category, "video_id": vid, "source_id_or_time": identifier})
    for vid, frame in df.groupby("video_id", sort=True):
        p, person = partition_of[vid], inventory[vid]["participant"]
        actions = sorted(frame.to_dict("records"), key=lambda a: (a["start_timestamp"], a["end_timestamp"], a["unique_narration_id"]))
        starts = [float(a["start_timestamp"]) for a in actions]
        completed = sorted(actions, key=lambda a: (a["end_timestamp"], a["start_timestamp"], a["unique_narration_id"]))
        ends = [float(a["end_timestamp"]) for a in completed]
        def context(row, task, t, reason=None):
            future = actions[bisect.bisect_right(starts, t):][:3]
            if len(future) < 3:
                skip(p, person, task, "fewer_than_three_future_actions", vid, row["unique_narration_id"] if task == "explain" else t)
                return
            if not clean_action(row) or any(not clean_action(a) for a in future):
                skip(p, person, task, "missing_action_label", vid, row["unique_narration_id"] if task == "explain" else t)
                return
            frames = frame_plan(inventory[vid], t, out)
            if frames is None:
                skip(p, person, task, "fewer_than_16_source_frames", vid, row["unique_narration_id"] if task == "explain" else t)
                return
            history = [a for a in completed[:bisect.bisect_right(ends, t)] if a["unique_narration_id"] != row["unique_narration_id"] and clean_action(a)][-8:]
            goal, gs = activity_at(activities[vid], t)
            r = record(row, task, t, history, future, goal, gs, frames, p, reason)
            records[p].append(r)
            counts[(p, person, task, "accepted", "total")] += 1
            counts[(p, person, task, "goal", gs)] += 1
            if reason:
                counts[(p, person, task, "reason_rule", reason["extraction_rule"])] += 1
                used.add(reason["narration_id"])
        for row in actions:
            reason = extract_reason(row)
            reason["partition"] = p
            reasons.append(reason)
            counts[(p, person, "explain", "reason_labels", reason["status"] if reason["status"] != "rejected" else reason["rejection_category"])] += 1
            if reason["status"] == "rejected":
                continue
            t = float(row["end_timestamp"])
            if t > inventory[vid]["duration_sec"]:
                skip(p, person, "explain", "action_endpoint_after_video", vid, row["unique_narration_id"])
                continue
            context(row, "explain", t, reason)
        for t in range(0, math.ceil(inventory[vid]["duration_sec"]), 5):
            ongoing = [a for a in actions if float(a["start_timestamp"]) <= t < float(a["end_timestamp"])]
            if len(ongoing) != 1:
                skip(p, person, "track", "gap" if not ongoing else "overlapping_actions", vid, t)
                continue
            context(ongoing[0], "track", float(t))
    for p in PARTITIONS:
        records[p].sort(key=lambda r: (r["video_id"], r["query_time_sec"], r["task"], r["id"]))
        atomic_write_jsonl(out / "text" / f"{p}.jsonl", records[p])
    atomic_write_jsonl(out / "reason_labels.jsonl", reasons)
    atomic_write_jsonl(out / "skipped_queries.jsonl", skipped)
    write_review(out, reasons, used)
    benchmark_audit(Path(__file__).resolve().parents[2], out, partition_of)
    atomic_write_json(out / "counts.json", {"protocol": VERSION, "annotation_actions": len(df), "videos": len(inventory), "video_hours": sum(v["duration_sec"] for v in inventory.values()) / 3600, "partition_records": {p: len(records[p]) for p in PARTITIONS}, "reason_labels_accepted_before_protocol_filter": sum(r["status"] != "rejected" for r in reasons), "explain_records": len(used), "dimensions": [{"partition": k[0], "participant": k[1], "task": k[2], "kind": k[3], "category": k[4], "count": n} for k, n in sorted(counts.items())], "source_hashes": source_hashes, "frame_settings": FRAME_SETTINGS, "status": "pending_human_reason_label_review"})
    planned_frames = defaultdict(set)
    for rows in records.values():
        for r in rows:
            planned_frames[r["video_id"]].update(f["source_frame_index"] for f in r["frames"])
    atomic_write_json(out / "frame_requests.json", {vid: sorted(indices) for vid, indices in sorted(planned_frames.items())})
    print(json.dumps({"text_records": {p: len(records[p]) for p in PARTITIONS}, "accepted_reason_labels": sum(r["status"] != "rejected" for r in reasons), "unique_requested_frames": sum(len(s) for s in planned_frames.values()), "status": "pending_human_reason_label_review"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
