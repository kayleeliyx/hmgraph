"""Conservative object normalization helpers for scene rolling context.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.scene.object_utils import canonicalize_object_name, is_generic_object_name"
"""

from __future__ import annotations

import re
import string
from typing import Any

GENERIC_OBJECT_NAMES = {
    "unknown",
    "none",
    "object",
    "item",
    "thing",
    "stuff",
    "area",
    "region",
    "part",
    "appliance",
    "machine",
    "device",
    "container",
    "surface",
    "tool",
    "utensil",
}

PART_TERMS = {
    "button",
    "door",
    "handle",
    "knob",
    "lid",
    "cap",
    "switch",
    "lever",
    "dial",
    "rim",
    "spout",
    "slot",
    "opening",
    "edge",
    "hinge",
}

NON_PARENT_MODIFIERS = {
    "black",
    "blue",
    "brown",
    "clear",
    "dark",
    "gray",
    "green",
    "grey",
    "large",
    "left",
    "lower",
    "metal",
    "orange",
    "plastic",
    "red",
    "right",
    "round",
    "silver",
    "small",
    "top",
    "upper",
    "white",
    "wooden",
    "yellow",
}

SPECIFIC_OBJECT_NAMES = {
    "mug",
    "cup",
    "coffee maker",
    "coffee machine",
    "refrigerator",
    "refrigerator door",
    "cupboard",
    "cupboard door",
    "cupboard knob",
    "cabinet",
    "cabinet door",
    "drawer",
    "button",
    "handle",
    "knob",
    "lid",
    "pan",
    "box",
    "countertop",
    "stove",
    "microwave",
    "kettle",
    "toaster",
    "orange",
}


def _clean_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    text = text.strip(string.whitespace)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text or default


def canonicalize_object_name(name: str) -> str:
    """Return a conservative normalized name without task-specific aliasing."""
    text = _clean_text(name, "")
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article) :]
            break
    text = re.sub(r"\s+", " ", text).strip()
    text = _safe_singularize(text)
    return text or "unknown"


def _safe_singularize(text: str) -> str:
    if " " in text:
        return text
    if len(text) <= 4:
        return text
    if text.endswith("ies") and len(text) > 5:
        return text[:-3] + "y"
    if text.endswith("ses") or text.endswith("xes") or text.endswith("ches") or text.endswith("shes"):
        return text[:-2]
    if text.endswith("s") and not text.endswith(("ss", "us")):
        return text[:-1]
    return text


def is_generic_object_name(name: str) -> bool:
    return canonicalize_object_name(name) in GENERIC_OBJECT_NAMES


def specificity_score(name: str) -> float:
    canonical = canonicalize_object_name(name)
    if canonical in GENERIC_OBJECT_NAMES:
        return 0.0
    if canonical in SPECIFIC_OBJECT_NAMES or any(term in canonical.split() for term in PART_TERMS):
        return 1.0
    return 0.5


def infer_part_relation(raw_name: str, canonical_name: str | None = None) -> tuple[str, str]:
    """Return (canonical_name, part_of) while preserving raw part-whole distinctions."""
    raw = canonicalize_object_name(raw_name)
    canonical = canonicalize_object_name(canonical_name or raw)
    words = raw.split()
    if len(words) >= 2 and words[-1] in PART_TERMS:
        parent = " ".join(words[:-1]).strip()
        if parent and not is_generic_object_name(parent) and parent not in NON_PARENT_MODIFIERS:
            return words[-1], parent
    return canonical, "none"


def is_part_object_name(name: str) -> bool:
    return canonicalize_object_name(name) in PART_TERMS


def make_object_record(
    raw_name: str,
    *,
    category: str = "unknown",
    location: str = "unknown",
    state: str = "unknown",
    confidence: float = 0.0,
    source: str,
    current_time_end: float,
    part_of: str | None = None,
) -> dict:
    raw = canonicalize_object_name(raw_name)
    canonical, inferred_part_of = infer_part_relation(raw)
    return {
        "raw_name": raw,
        "canonical_name": canonical,
        "category": _clean_text(category),
        "part_of": canonicalize_object_name(part_of or inferred_part_of),
        "location": _clean_text(location),
        "state": _clean_text(state),
        "confidence": _clamp_float(confidence),
        "last_seen_t": float(current_time_end),
        "source_counts": {
            "scene_observation": 1 if source == "scene_observation" else 0,
            "W_action": 1 if source == "W_action" else 0,
        },
    }


def merge_object_record(existing: dict | None, new: dict, current_time_end: float) -> dict:
    """Merge records for the same canonical object conservatively."""
    if existing is None:
        merged = dict(new)
        merged["last_seen_t"] = float(current_time_end)
        merged.setdefault("source_counts", {"scene_observation": 0, "W_action": 0})
        return merged

    old_score = specificity_score(existing.get("canonical_name", ""))
    new_score = specificity_score(new.get("canonical_name", ""))
    merged = dict(existing)
    if new_score >= old_score and not is_generic_object_name(new.get("canonical_name", "")):
        merged["raw_name"] = new.get("raw_name", merged.get("raw_name", "unknown"))
        merged["canonical_name"] = new.get("canonical_name", merged.get("canonical_name", "unknown"))
    merged["category"] = _prefer_specific(merged.get("category"), new.get("category"))
    merged["part_of"] = _prefer_specific(merged.get("part_of"), new.get("part_of"))
    merged["location"] = _prefer_specific(merged.get("location"), new.get("location"))
    merged["state"] = _prefer_specific(merged.get("state"), new.get("state"))
    merged["confidence"] = max(_clamp_float(merged.get("confidence")), _clamp_float(new.get("confidence")))
    merged["last_seen_t"] = float(current_time_end)
    counts = dict(merged.get("source_counts") or {})
    new_counts = new.get("source_counts") or {}
    counts["scene_observation"] = int(counts.get("scene_observation", 0)) + int(new_counts.get("scene_observation", 0))
    counts["W_action"] = int(counts.get("W_action", 0)) + int(new_counts.get("W_action", 0))
    merged["source_counts"] = counts
    return merged


def _prefer_specific(old: Any, new: Any) -> str:
    old_text = _clean_text(old, "unknown")
    new_text = _clean_text(new, "unknown")
    if new_text not in {"", "unknown", "none"}:
        return new_text
    return old_text


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, 0.0), 1.0)
