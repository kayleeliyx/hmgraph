"""Central path constants for the Week 1 W/L1 verification pipeline.

Inputs:
  None at runtime; constants point to the local HD-EPIC and hmgraph roots.
Outputs:
  Path constants used by CLI scripts, which still accept argparse overrides.
"""

from pathlib import Path

HMGRAPH_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = HMGRAPH_ROOT / "code"
HD_EPIC_ROOT = Path("/data/kailaic/HDEPIC/data/HD-EPIC")
HD_EPIC_VIDEO_ROOT = HD_EPIC_ROOT / "Videos"
HD_EPIC_ANNOTATION_ROOT = Path("/data/kailaic/HDEPIC/data/hd-epic-annotations")
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
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_VIDEO_PATH = HD_EPIC_VIDEO_ROOT / "P01" / "P01-20240202-110250.mp4"
DEFAULT_VIDEO_ID = "P01-20240202-110250"
DEFAULT_OUTPUT_DIR = HMGRAPH_ROOT / "week1_validation" / DEFAULT_VIDEO_ID
