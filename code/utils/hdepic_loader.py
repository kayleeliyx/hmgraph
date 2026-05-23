"""Load HD-EPIC GT action annotations for one video.

Inputs:
  HD_EPIC_Narrations.pkl path and video_id.
Outputs:
  A list of GT dicts and optional gt.jsonl for the selected video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _first(seq, default=""):
    if isinstance(seq, (list, tuple)) and seq:
        return seq[0]
    return default


def load_video_actions(video_id: str, annotations_path: str | Path) -> List[dict]:
    """Return HD-EPIC action rows for one video_id."""
    df = pd.read_pickle(annotations_path)
    video_df = df[df["video_id"] == video_id].sort_values("start_timestamp")
    actions = []
    for _, row in video_df.iterrows():
        main_action = _first(row.get("main_actions"), ("", ""))
        hand_raw = _first(row.get("hands"), "")
        hand = str(hand_raw).replace(" hand", "").strip().lower() if hand_raw else ""
        actions.append(
            {
                "gt_id": str(row["unique_narration_id"]),
                "video_id": str(row["video_id"]),
                "narration": str(row["narration"]),
                "verb": str(main_action[0]).strip().lower(),
                "direct_object": str(main_action[1]).strip().lower(),
                "hand": hand,
                "t_start": float(row["start_timestamp"]),
                "t_end": float(row["end_timestamp"]),
            }
        )
    return actions


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--annotations-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    actions = load_video_actions(args.video_id, args.annotations_path)
    write_jsonl(actions, args.output)
    print(f"L1 GT actions written: {args.output} ({len(actions)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
