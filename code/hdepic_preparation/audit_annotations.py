"""Independently reconcile protocol records with the complete source pickle."""
import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from code.e0.dataset import atomic_write_json, sha256_file
from .build import DEFAULT_OUT, PARTITIONS, clean_action, read_jsonl, text_fingerprint


def audit(out):
    counts = json.loads((out / "counts.json").read_text())
    source = next(Path(p) for p in counts["source_hashes"] if p.endswith("HD_EPIC_Narrations.pkl"))
    assert sha256_file(source) == counts["source_hashes"][str(source)]
    df = pd.read_pickle(source)
    inventory = json.loads((out / "video_inventory.json").read_text())
    by_video = {}
    for vid, frame in df.groupby("video_id"):
        actions = sorted(frame.to_dict("records"), key=lambda r: (r["start_timestamp"], r["end_timestamp"], r["unique_narration_id"]))
        completed = sorted(actions, key=lambda r: (r["end_timestamp"], r["start_timestamp"], r["unique_narration_id"]))
        by_video[vid] = (actions, completed, [r["start_timestamp"] for r in actions], [r["end_timestamp"] for r in completed], {r["unique_narration_id"]: r for r in actions})
    tally = Counter()
    for vid, (actions, completed, starts, ends, ids) in by_video.items():
        for t in range(0, math.ceil(inventory[vid]["duration_sec"]), 5):
            ongoing = bisect.bisect_right(starts, t) - bisect.bisect_right(ends, t)
            tally["track_grid_queries"] += 1
            tally["track_grid_gap" if ongoing == 0 else "track_grid_overlap" if ongoing > 1 else "track_grid_single_action"] += 1
    for part in PARTITIONS:
        for r in read_jsonl(out / "text" / f"{part}.jsonl"):
            actions, completed, starts, ends, ids = by_video[r["video_id"]]
            t = r["query_time_sec"]
            current_id = r["scoring_metadata"]["current_action"]["narration_id"]
            source_row = ids[current_id]
            if r["task"] == "track":
                assert t % 5 == 0
                assert bisect.bisect_right(starts, t) - bisect.bisect_right(ends, t) == 1
                assert source_row["start_timestamp"] <= t < source_row["end_timestamp"]
            else:
                assert t == source_row["end_timestamp"]
                reason = r["scoring_metadata"]["reason_annotation"]
                assert reason["original_narration"] == source_row["narration"]
                assert source_row["narration"][reason["char_start"]:reason["char_end"]] == r["target"]["reason"]
            future = actions[bisect.bisect_right(starts, t):][:3]
            assert [a["narration_id"] for a in r["scoring_metadata"]["future_actions"]] == [a["unique_narration_id"] for a in future]
            history = [a for a in completed[:bisect.bisect_right(ends, t)] if a["unique_narration_id"] != current_id and clean_action(a)][-8:]
            assert [a["narration_id"] for a in r["scoring_metadata"]["history_actions"]] == [a["unique_narration_id"] for a in history]
            for a in [r["scoring_metadata"]["current_action"], *r["scoring_metadata"]["future_actions"], *r["scoring_metadata"]["history_actions"]]:
                original = ids[a["narration_id"]]
                assert a["start_sec"] == original["start_timestamp"] and a["end_sec"] == original["end_timestamp"]
                assert a["main_action_class_ids"] == [list(pair) for pair in original["main_action_classes"]]
            tally[f"verified_{r['task']}"] += 1
    for category, key in [("gap", "track_grid_gap"), ("overlapping_actions", "track_grid_overlap")]:
        assert sum(x["count"] for x in counts["dimensions"] if x["task"] == "track" and x["kind"] == "skipped" and x["category"] == category) == tally[key]
    atomic_write_json(out / "annotation_alignment_audit.json", {"passed": True, "dataset_fingerprint": text_fingerprint(out), "source_sha256": sha256_file(source), "counts": dict(tally), "checks": ["exactly one ongoing source action for every Track record", "Explain endpoint and narration span match source", "exact first three strictly future source action IDs", "exact last eight completed source action IDs excluding target", "all timestamps and main verb/noun class IDs match source", "grid gaps and overlaps independently reconciled"], "semantic_reason_review": "pending_human_reason_label_review"})
    print(f"Source alignment audit passed: {dict(tally)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    audit(parser.parse_args().output.resolve())
