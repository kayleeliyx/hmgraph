"""Central path constants for the Week 1 W/L1 verification pipeline.

Inputs:
  None at runtime; constants point to the local HD-EPIC and hmgraph roots.
Outputs:
  Path constants used by CLI scripts, which still accept argparse overrides.
"""

from pathlib import Path

HMGRAPH_ROOT = Path("/data2/yaxuanli/hmgraph_codex")
CODE_DIR = HMGRAPH_ROOT / "code"
HD_EPIC_ROOT = Path("/data2/yaxuanli/HD-EPIC")
HD_EPIC_VIDEO_ROOT = HD_EPIC_ROOT / "Videos"
HD_EPIC_ANNOTATION_ROOT = HD_EPIC_ROOT / "hd-epic-annotations"
HD_EPIC_NARRATIONS = (
    HD_EPIC_ANNOTATION_ROOT
    / "narrations-and-action-segments"
    / "HD_EPIC_Narrations.pkl"
)
HD_EPIC_VERB_CLASSES = (
    HD_EPIC_ANNOTATION_ROOT
    / "narrations-and-action-segments"
    / "HD_EPIC_verb_classes.csv"
)
HD_EPIC_NOUN_CLASSES = (
    HD_EPIC_ANNOTATION_ROOT
    / "narrations-and-action-segments"
    / "HD_EPIC_noun_classes.csv"
)
DEFAULT_MODEL_ID = (
    "/home/yaxuanli/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/"
    "snapshots/47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"
)
DEFAULT_VIDEO_PATH = HD_EPIC_VIDEO_ROOT / "P01" / "P01-20240202-110250.mp4"
DEFAULT_VIDEO_ID = "P01-20240202-110250"
DEFAULT_OUTPUT_DIR = HMGRAPH_ROOT / "week1_validation" / DEFAULT_VIDEO_ID
