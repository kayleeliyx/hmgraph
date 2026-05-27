"""W-based L1 segment proposal for the canonical pipeline.

W is the base temporal unit. This module groups consecutive one-second W
records into variable-length candidate L1 segments, then annotates each group
with diagnostics that guide downstream frame verification.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from code.scene.object_utils import canonicalize_object_name, is_generic_object_name


MAX_GROUP_SEC = 6.0
LOW_CONFIDENCE = 0.45
HIGH_CONFIDENCE = 0.65


def propose_l1_reasoning_segments(combined_records: list[dict]) -> list[dict]:
    """Return variable-length candidate L1 groups built from consecutive W units."""
    records = sorted(combined_records, key=lambda r: float(r.get("t_start", 0.0)))
    segments: list[dict] = []
    active: list[dict] = []

    def flush(reason: str) -> None:
        nonlocal active
        if not active:
            return
        segment = _make_segment(len(segments), active, reason)
        if segment["t_end"] - segment["t_start"] >= 1.0:
            segments.append(segment)
        active = []

    for record in records:
        if not _has_action_signal(record):
            if active and _can_bridge(active[-1], record):
                active.append(record)
            else:
                flush("ended by no-action gap")
            continue

        if not active:
            active = [record]
            continue

        split, reason = _should_split_w_group(active, record)
        if split:
            flush(reason)
            active = [record]
            continue

        active.append(record)

    flush("end of range")
    return segments


def _should_split_w_group(active: list[dict], record: dict) -> tuple[bool, str]:
    """Decide whether a new W record starts a new L1 candidate group."""
    duration = float(record["t_end"]) - float(active[0]["t_start"])
    if duration > MAX_GROUP_SEC:
        return True, "split by max W-group duration safety cap"
    if _no_action_gap(active[-1], record) > 1.0:
        return True, "split by no-action gap"

    prev_units = _action_units(active[-1])
    cur_units = _action_units(record)
    if prev_units and cur_units:
        if _has_compatible_action(prev_units, cur_units):
            return False, ""
        if _strong_specific_change(prev_units, cur_units):
            return True, "split by strong W action identity change"
        if _focus_changed(active[-1], record) and not _stable_overlap(active[-1], record):
            return True, "split by W action change plus focus conflict"
        if duration >= 3.0 and not _weak_or_generic(prev_units + cur_units):
            return True, "split by sustained W action identity change"
        return False, ""

    if _focus_changed(active[-1], record) and not _stable_overlap(active[-1], record):
        return True, "split by focus conflict without compatible W action"
    return False, ""


def _make_segment(idx: int, records: list[dict], reason: str) -> dict:
    t_start = float(records[0]["t_start"])
    t_end = float(records[-1]["t_end"])
    scene_types = [_scene(record).get("scene_type", "unknown") for record in records]
    focuses = [_focus(record) for record in records if _focus(record) not in {"", "unknown", "none"}]
    diagnostics = _grouping_diagnostics(records)
    return {
        "segment_id": f"seg_{idx:04d}",
        "t_start": t_start,
        "t_end": t_end,
        "context_t_start": max(0.0, t_start - 1.5),
        "context_t_end": t_end + 1.5,
        "source_window_indices": [int(r["window_idx"]) for r in records],
        "reason": reason,
        "dominant_scene_type": _mode(scene_types, "unknown"),
        "dominant_focus": _mode(focuses, "unknown"),
        "stable_objects_in_segment": _stable_objects(records),
        "w_action_summaries": [_w_summary(record) for record in records],
        "scene_summaries": [_scene_summary(record) for record in records],
        "rolling_context_summaries": [_rolling_summary(record) for record in records],
        "w_grouping_diagnostics": diagnostics,
        "recommended_frame_budget": _recommended_frame_budget(t_start, t_end, diagnostics),
        "proposal_version": "w_grouped_l1_candidates",
    }


def _grouping_diagnostics(records: list[dict]) -> dict:
    units_by_window = [
        {
            "window_idx": int(record.get("window_idx", -1)),
            "action_units": _action_units(record),
            "scene_status": _scene_status(record),
            "focus": _focus(record),
        }
        for record in records
    ]
    all_units = [unit for item in units_by_window for unit in item["action_units"]]
    signatures = [unit["signature"] for unit in all_units]
    distinct = sorted(set(signatures))
    low_conf = [unit for unit in all_units if unit["confidence"] < LOW_CONFIDENCE]
    generic = [unit for unit in all_units if unit["is_generic_object"]]
    focus_values = [item["focus"] for item in units_by_window if item["focus"] not in {"", "none", "unknown"}]
    return {
        "grouping_unit": "W_action_1sec",
        "action_signature_sequence": signatures,
        "distinct_action_signatures": distinct,
        "n_distinct_action_signatures": len(distinct),
        "n_low_confidence_w_actions": len(low_conf),
        "n_generic_or_vague_w_objects": len(generic),
        "mean_w_action_confidence": round(mean([unit["confidence"] for unit in all_units]), 3) if all_units else 0.0,
        "focus_sequence": focus_values,
        "n_focus_changes": sum(1 for a, b in zip(focus_values, focus_values[1:]) if a != b),
        "needs_dense_visual_verification": len(distinct) > 1 or bool(low_conf) or bool(generic),
        "w_units_by_window": units_by_window,
    }


def _recommended_frame_budget(t_start: float, t_end: float, diagnostics: dict) -> int:
    duration = max(1.0, float(t_end) - float(t_start))
    budget = 6 + int(round(duration * 2.0))
    if diagnostics["needs_dense_visual_verification"]:
        budget += 4
    if diagnostics["n_distinct_action_signatures"] > 1:
        budget += 2
    if diagnostics["n_focus_changes"] > 0:
        budget += 2
    return max(6, min(16, budget))


def _action_units(record: dict) -> list[dict]:
    units = []
    for action in _valid_actions(record):
        verb = _norm_action_text(action.get("verb"))
        obj = _action_object_name(action)
        part = _action_part_name(action)
        confidence = _float(action.get("confidence"), 0.0)
        units.append(
            {
                "verb": verb,
                "object": obj,
                "part": part,
                "confidence": confidence,
                "is_generic_object": obj in {"", "none", "unknown"} or is_generic_object_name(obj),
                "signature": f"{verb}|{obj}|part={part}",
            }
        )
    return units


def _has_compatible_action(a_units: list[dict], b_units: list[dict]) -> bool:
    for a in a_units:
        for b in b_units:
            if a["signature"] == b["signature"]:
                return True
            if a["verb"] == b["verb"] and _same_object_or_part(a, b):
                return True
            if _same_object_or_part(a, b) and (a["is_generic_object"] or b["is_generic_object"]):
                return True
    return False


def _same_object_or_part(a: dict, b: dict) -> bool:
    if a["object"] and a["object"] == b["object"]:
        return True
    if a["part"] not in {"none", "unknown", ""} and a["part"] == b["part"]:
        return True
    return False


def _strong_specific_change(prev_units: list[dict], cur_units: list[dict]) -> bool:
    if _weak_or_generic(prev_units + cur_units):
        return False
    return max(unit["confidence"] for unit in prev_units) >= HIGH_CONFIDENCE and max(unit["confidence"] for unit in cur_units) >= HIGH_CONFIDENCE


def _weak_or_generic(units: list[dict]) -> bool:
    return any(unit["confidence"] < LOW_CONFIDENCE or unit["is_generic_object"] for unit in units)


def _has_action_signal(record: dict) -> bool:
    w = (record.get("W_action") or {}).get("record") or {}
    status = _scene_status(record)
    return bool(w.get("action_present")) or status in {"visible_action", "uncertain_action"}


def _can_bridge(prev: dict, cur: dict) -> bool:
    return _no_action_gap(prev, cur) <= 1.0 and _same_focus_or_overlap(prev, cur)


def _same_focus_or_overlap(prev: dict, cur: dict) -> bool:
    return _focus(prev) == _focus(cur) or _stable_overlap(prev, cur)


def _no_action_gap(prev: dict, cur: dict) -> float:
    if _has_action_signal(prev) or _has_action_signal(cur):
        return max(0.0, float(cur["t_start"]) - float(prev["t_end"]))
    return float(cur["t_end"]) - float(prev["t_start"])


def _valid_actions(record: dict) -> list[dict]:
    w = (record.get("W_action") or {}).get("record") or {}
    actions = w.get("valid_actions") or []
    if actions:
        return [a for a in actions if isinstance(a, dict)]
    parsed = w.get("parsed") or {}
    actions = parsed.get("actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _action_object_name(action: dict) -> str:
    rich = action.get("direct_object_rich") or {}
    raw = rich.get("specific_type") or rich.get("name")
    obj = action.get("direct_object")
    if not raw and isinstance(obj, dict):
        raw = obj.get("name") or obj.get("canonical_name") or obj.get("raw_name")
    if not raw:
        raw = obj or action.get("object")
    return canonicalize_object_name(raw or "unknown")


def _action_part_name(action: dict) -> str:
    attrs = action.get("object_attributes") or {}
    part = attrs.get("part")
    if not part:
        obj = action.get("direct_object")
        if isinstance(obj, dict):
            part = obj.get("part") or obj.get("part_of")
    part = canonicalize_object_name(part or "none")
    return part if part not in {"whole object", "whole_object"} else "none"


def _norm_action_text(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ") or "unknown"


def _focus_changed(a: dict, b: dict) -> bool:
    fa, fb = _focus(a), _focus(b)
    return fa not in {"", "unknown", "none"} and fb not in {"", "unknown", "none"} and fa != fb


def _stable_overlap(a: dict, b: dict) -> bool:
    return bool(_stable_keys(a) & _stable_keys(b))


def _stable_keys(record: dict) -> set[str]:
    rolling = record.get("rolling_context_v1_1") or {}
    objs = ((rolling.get("after_window") or {}).get("stable_objects") or [])[:8]
    return {
        canonicalize_object_name(o.get("canonical_name") or o.get("raw_name") or "")
        for o in objs
        if o.get("canonical_name") or o.get("raw_name")
    }


def _stable_objects(records: list[dict]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for record in records:
        rolling = record.get("rolling_context_v1_1") or {}
        for obj in ((rolling.get("after_window") or {}).get("stable_objects") or []):
            key = canonicalize_object_name(obj.get("canonical_name") or obj.get("raw_name") or "")
            if key and key not in by_name:
                by_name[key] = obj
    return list(by_name.values())[:12]


def _w_summary(record: dict) -> dict:
    return {
        "window_idx": int(record.get("window_idx", -1)),
        "t_start": float(record.get("t_start", 0.0)),
        "t_end": float(record.get("t_end", 0.0)),
        "action_present": bool(((record.get("W_action") or {}).get("record") or {}).get("action_present")),
        "actions": [
            {
                "verb": unit["verb"],
                "object": unit["object"],
                "part": unit["part"],
                "confidence": unit["confidence"],
                "signature": unit["signature"],
                "is_generic_object": unit["is_generic_object"],
            }
            for unit in _action_units(record)
        ],
    }


def _scene_summary(record: dict) -> dict:
    scene = _scene(record)
    presence = scene.get("action_presence") or {}
    return {
        "window_idx": int(record.get("window_idx", -1)),
        "status": _scene_status(record),
        "confidence": presence.get("confidence", 0.0),
        "current_focus": scene.get("current_focus", "unknown"),
        "main_objects": [o.get("name", "unknown") for o in (scene.get("visible_objects") or [])[:8]],
    }


def _rolling_summary(record: dict) -> dict:
    rolling = record.get("rolling_context_v1_1") or {}
    after = rolling.get("after_window") or {}
    return {
        "window_idx": int(record.get("window_idx", -1)),
        "current_focus": after.get("current_focus", "unknown"),
        "stable_objects": [
            {
                "raw_name": o.get("raw_name", ""),
                "canonical_name": o.get("canonical_name", ""),
                "part_of": o.get("part_of", "none"),
                "location": o.get("location", "unknown"),
            }
            for o in (after.get("stable_objects") or [])[:10]
        ],
    }


def _scene(record: dict) -> dict:
    return (((record.get("scene_observation") or {}).get("record") or {}).get("scene_observation") or {})


def _scene_status(record: dict) -> str:
    block = record.get("scene_observation") or {}
    calibrated = block.get("calibrated_action_presence") or {}
    return calibrated.get("calibrated_status") or (_scene(record).get("action_presence") or {}).get("status", "unknown")


def _focus(record: dict) -> str:
    rolling = record.get("rolling_context_v1_1") or {}
    return canonicalize_object_name((rolling.get("after_window") or {}).get("current_focus") or _scene(record).get("current_focus") or "unknown")


def _mode(values: list[Any], default: str) -> str:
    vals = [str(v) for v in values if str(v)]
    return Counter(vals).most_common(1)[0][0] if vals else default


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
