"""Audit causal records and export ms-swift messages/images, without training."""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from .build import CLAUSES, DEFAULT_OUT, PARTITIONS, read_jsonl, text_fingerprint
from .frames import sample_records

SYSTEM = "Use the timestamped frames and completed action history to answer the task. Return only the requested tagged fields. Forecast exactly three next actions in temporal order."


def render(r, frames):
    lines = [f"Task: {r['task'].title()}", f"Query time: {r['query_time_sec']:.6f} s", "Frames in chronological order:"]
    lines.extend(f"Frame {i + 1:02d}, source time {f['timestamp_sec']:.6f} s: <image>" for i, f in enumerate(frames))
    lines.append("Previous completed actions (oldest to newest):")
    lines.extend(f"[{a['start_sec']:.3f}, {a['end_sec']:.3f}] {a['action']}" for a in r["input"]["history"])
    if not r["input"]["history"]:
        lines.append("(none)")
    if r["task"] == "explain":
        lines.append(f"Action to explain: {r['input']['action_to_explain']}")
        lines.append("Explain this action's stated reason in <why>. Include <goal> when supervised, followed by <next> with exactly three actions.")
        answer = f"<why>{html.escape(r['target']['reason'], quote=False)}</why>"
    else:
        lines.append("Identify the ongoing action in <action>. Include <goal> when supervised, followed by <next> with exactly three actions.")
        answer = f"<action>{html.escape(r['target']['current_action'], quote=False)}</action>"
    if "goal" in r["target"]:
        answer += f"\n<goal>{html.escape(r['target']['goal'], quote=False)}</goal>"
    answer += "\n<next>" + "; ".join(html.escape(a, quote=False) for a in r["target"]["next_actions"]) + "</next>"
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}, {"role": "assistant", "content": answer}], "images": [f["path"] for f in frames]}


def audit_record(r, frames, partition_of, check_paths=True):
    t = r["query_time_sec"]
    assert partition_of[r["video_id"]] == r["partition"]
    assert len(frames) == 16
    times = [f["timestamp_sec"] for f in frames]
    assert times == sorted(set(times)), r["id"]
    lower_bound = float(max(Decimal(0), Decimal(str(t)) - Decimal(5)))
    assert all(lower_bound <= ts <= t for ts in times), r["id"]
    assert len(r["input"]["history"]) <= 8
    assert all(a["end_sec"] <= t for a in r["input"]["history"])
    assert all(a["narration_id"] != r["scoring_metadata"]["current_action"]["narration_id"] for a in r["scoring_metadata"]["history_actions"])
    assert [a["end_sec"] for a in r["input"]["history"]] == sorted(a["end_sec"] for a in r["input"]["history"])
    assert all(not CLAUSES.search(a["action"]) for a in r["input"]["history"])
    assert len(r["target"]["next_actions"]) == 3
    assert all(a["video_id"] == r["video_id"] and a["start_sec"] > t for a in r["scoring_metadata"]["future_actions"])
    assert [a["start_sec"] for a in r["scoring_metadata"]["future_actions"]] == sorted(a["start_sec"] for a in r["scoring_metadata"]["future_actions"])
    if r["task"] == "track":
        current = r["scoring_metadata"]["current_action"]
        assert current["start_sec"] <= t < current["end_sec"]
        assert "action_to_explain" not in r["input"]
    else:
        assert r["scoring_metadata"]["current_action"]["end_sec"] == t
        a = r["scoring_metadata"]["reason_annotation"]
        assert a["original_narration"][a["char_start"]:a["char_end"]] == r["target"]["reason"]
        assert a["narration_id"] == r["scoring_metadata"]["current_action"]["narration_id"]
    assert ("goal" in r["target"]) == (r["scoring_metadata"]["goal_status"] == "available")
    user_allowed = {"history", "action_to_explain"} if r["task"] == "explain" else {"history"}
    assert set(r["input"]) == user_allowed
    rendered = render(r, frames)
    assert rendered["messages"][1]["content"].count("<image>") == len(rendered["images"]) == 16
    assert all(Path(p).is_absolute() for p in rendered["images"])
    if check_paths:
        assert all(Path(p).is_file() for p in rendered["images"])
    assert "<goal>" not in rendered["messages"][2]["content"] or "goal" in r["target"]
    return rendered


def load_cache(out, r, cache):
    root = Path(r["frames"][0]["path"]).parent
    if str(root) not in cache:
        entry = json.loads((root / "record.json").read_text())
        for frame in entry["frames"].values():
            path = Path(frame["path"])
            if not path.is_file() or path.stat().st_size != frame["bytes"] or sha256_file(path) != frame["sha256"]:
                raise ValueError(f"Missing/corrupt cached frame: {path}")
        cache[str(root)] = entry
    frames = []
    for plan in r["frames"]:
        f = cache[str(root)]["frames"][str(plan["source_frame_index"])].copy()
        assert f["timestamp_sec"] == plan["planned_timestamp_sec"]
        frames.append(f)
    return frames


def previews(out, records, rendered, frame_sets):
    destination = out / "previews"
    destination.mkdir(exist_ok=True)
    sections, markdown = [], ["# HD-EPIC dataset examples", "", "Pending human reason-label review. Contact sheets below are for review only; training receives 16 individual JPEGs.", ""]
    for i, (r, item, frames) in enumerate(zip(records, rendered, frame_sets), 1):
        sheet = Image.new("RGB", (1024, 4 * 284), "black")
        draw = ImageDraw.Draw(sheet)
        for j, f in enumerate(frames):
            x, y = (j % 4) * 256, (j // 4) * 284
            with Image.open(f["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((256, 256))
                sheet.paste(image, (x, y + 28))
            draw.text((x + 5, y + 6), f"Frame {j+1:02d} | {f['timestamp_sec']:.6f} s", fill="white")
        name = f"example_{i:02d}.jpg"
        sheet.save(destination / name, quality=90)
        user = item["messages"][1]["content"]
        answer = item["messages"][2]["content"]
        sections.append(f"<section><h2>{i}. {html.escape(r['id'])}</h2><p>Partition: {r['partition']}; goal status: {r['scoring_metadata']['goal_status']}</p><img src='{name}' width='1024'><h3>Input</h3><pre>{html.escape(user)}</pre><h3>Target</h3><pre>{html.escape(answer)}</pre></section>")
        markdown.extend([f"## {i}. {r['partition']} / {r['id']}", "", f"![16 timestamped review frames]({(destination / name).resolve()})", "", "Input:", "```text", user, "```", "Target:", "```text", answer, "```", ""])
    atomic_write_text(destination / "examples.html", "<!doctype html><meta charset='utf-8'><title>HD-EPIC examples</title><style>body{font:16px sans-serif;margin:30px;max-width:1100px}pre{white-space:pre-wrap;background:#eee;padding:16px}section{margin-bottom:60px}</style><h1>HD-EPIC: 10 examples</h1><p>Pending human reason-label review. Sheets are review previews only. Training uses individual frames.</p>" + "".join(sections))
    atomic_write_text(destination / "examples.md", "\n".join(markdown))


def export(out, sample=False):
    manifest = json.loads((out / "split_manifest.json").read_text())
    partition_of = {v: p for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]}
    cache, counts, used_frames = {}, Counter(), {}
    chosen = sample_records(out)
    chosen_ids = {r["id"] for r in chosen}
    preview_rows, preview_rendered, preview_frames = [], [], []
    if sample:
        groups = [("sample", chosen)]
    else:
        groups = [(p, read_jsonl(out / "text" / f"{p}.jsonl")) for p in PARTITIONS]
    global_ids = set()
    for group, rows in groups:
        swift, scoring, rich, prompts = [], [], [], []
        for r in rows:
            assert r["id"] not in global_ids
            global_ids.add(r["id"])
            frames = load_cache(out, r, cache)
            used_frames.update({f["path"]: f for f in frames})
            item = audit_record(r, frames, partition_of)
            swift.append(item)
            prompts.append({"messages": item["messages"][:2], "images": item["images"]})
            scoring.append({"row_index": len(swift) - 1, "id": r["id"], "partition": r["partition"], "participant": r["participant"], "video_id": r["video_id"], "task": r["task"], "query_time_sec": r["query_time_sec"], "frames": frames, **r["scoring_metadata"]})
            rich.append({"id": r["id"], "partition": r["partition"], "task": r["task"], "query_time_sec": r["query_time_sec"], **item})
            counts[f"{r['partition']}:{r['task']}"] += 1
            if r["id"] in chosen_ids:
                preview_rows.append(r); preview_rendered.append(item); preview_frames.append(frames)
        # Scoring metadata is deliberately a separate sidecar, never model input.
        target_dir = out / ("sample" if sample else "swift")
        atomic_write_jsonl(target_dir / f"{group}.jsonl", swift)
        atomic_write_jsonl(target_dir / f"{group}.prompts.jsonl", prompts)
        atomic_write_jsonl(target_dir / f"{group}.scoring.jsonl", scoring)
        atomic_write_jsonl(target_dir / f"{group}.records.jsonl", rich)
        print(f"Verified and exported {group}: {len(swift)} records", flush=True)
    size = sum(f["bytes"] for f in used_frames.values())
    unique = len(used_frames)
    total_planned = sum(len(indices) for indices in json.loads((out / "frame_requests.json").read_text()).values())
    audit = {"passed": True, "status": "pending_human_reason_label_review", "training_started": False, "model_calls": 0, "records_verified": len(global_ids), "counts": dict(counts), "unique_frames_verified": unique, "frame_cache_bytes": size, "estimated_full_cache_bytes": round(size / unique * total_planned), "full_requested_unique_frames": total_planned, "checks": {"actual_decoder_pts_inside_lookback_and_at_or_before_query": True, "exactly_16_distinct_chronological_individual_frames": True, "all_frame_paths_absolute_existing_sha256_verified": True, "history_ends_at_or_before_query": True, "target_excluded_from_history": True, "history_from_main_action_pairs_without_purpose_how_activity": True, "track_current_action_not_supplied": True, "future_starts_strictly_after_query_same_video": True, "reason_exact_original_span_and_target_binding": True, "missing_goal_omits_goal_tag": True, "metadata_separate_from_model_messages": True, "no_word_overlap_rejection": True}, "limitation": "Rule-based reason/target binding remains pending human review; this audit does not establish semantic label correctness.", "swift_format_reference": "https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html"}
    audit["dataset_fingerprint"] = text_fingerprint(out)
    atomic_write_json(out / ("sample_audit.json" if sample else "timestamp_leakage_audit.json"), audit)
    if len(preview_rows) == 10:
        ordering = sorted(range(10), key=lambda i: [r["id"] for r in chosen].index(preview_rows[i]["id"]))
        previews(out, [preview_rows[i] for i in ordering], [preview_rendered[i] for i in ordering], [preview_frames[i] for i in ordering])
    print(json.dumps(audit, indent=2), flush=True)


def audit_text(out):
    manifest = json.loads((out / "split_manifest.json").read_text())
    partition_of = {v: p for p in PARTITIONS for v in manifest["partitions"][p]["video_ids"]}
    inventory = json.loads((out / "video_inventory.json").read_text())
    ids = set()
    counts = Counter()
    for p in PARTITIONS:
        for r in read_jsonl(out / "text" / f"{p}.jsonl"):
            assert r["id"] not in ids
            ids.add(r["id"])
            frames = [{**f, "timestamp_sec": f["planned_timestamp_sec"]} for f in r["frames"]]
            audit_record(r, frames, partition_of, check_paths=False)
            for a in [r["scoring_metadata"]["current_action"], *r["scoring_metadata"]["future_actions"]]:
                assert 0 <= a["start_sec"] <= a["end_sec"] <= inventory[r["video_id"]]["duration_sec"], (r["id"], a)
            counts[f"{p}:{r['task']}"] += 1
    atomic_write_json(out / "text_audit.json", {"passed": True, "records_verified": len(ids), "counts": dict(counts), "dataset_fingerprint": text_fingerprint(out), "frame_timestamps_status": "planned_CFR_source_indices_pending_independent_actual_decoder_PTS_check", "status": "pending_human_reason_label_review"})
    print(f"All {len(ids)} text records passed causal and format checks before extraction", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    if args.text_only:
        audit_text(args.output.resolve())
    else:
        export(args.output.resolve(), args.sample)


if __name__ == "__main__":
    main()
