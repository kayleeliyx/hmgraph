#!/usr/bin/env python3
"""Reparse cached W responses and build L1 through the shared implementation.

Run from the ActGround project root, for example:
    python build_l1_wgroup.py \
        --w-jsonl runs/P01-20240202-161948/qwen/w_outputs.jsonl \
        --out-jsonl runs/P01-20240202-161948/qwen/wgroup_v2/l1_candidates_wgroup.jsonl

The output's parent directory must be NEW. It will contain the requested L1
file plus reprocessed W, support observations, an action ledger, and diagnostics.
No model, GPU, video, or ground-truth annotations are used. Cached raw_output
is required: earlier valid_actions are reparsed rather than trusted unchanged.

This file is only an entry point. Parsing/filtering lives in code.w.records;
grouping lives in code.l1.from_w; file handling lives in code.reprocess_w_l1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code.reprocess_w_l1 import reprocess_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w-jsonl", type=Path, required=True,
                        help="Cached W JSONL containing raw_output; input is preserved.")
    parser.add_argument("--out-jsonl", type=Path, required=True,
                        help="L1 output path inside a NEW directory (not the existing run directory).")
    parser.add_argument("--baseline-l1", type=Path,
                        help="Optional old L1 file for diagnostic counts only; never used to build L1.")
    args = parser.parse_args()
    source = args.w_jsonl.resolve()
    output = args.out_jsonl.resolve()
    try:
        summary = reprocess_run(source.parent, output.parent, args.baseline_l1,
                                w_jsonl=source, l1_filename=output.name)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({**summary, "l1_output": str(output),
                      "reprocessed_w_output": str(output.parent / "w_reprocessed.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
