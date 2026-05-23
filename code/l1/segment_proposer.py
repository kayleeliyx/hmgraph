"""L1 micro-action segment proposal for the canonical pipeline.

Canonical L1 micro-action segment proposal. Helper layers are inlined here so the active source tree does not depend on legacy wrapper modules.
"""



from collections import Counter
from typing import Any


def _propose_l1_reasoning_segments_basic(combined_records: list[dict]) -> list[dict]:
    records = sorted(combined_records, key=lambda r: float(r.get("t_start", 0.0)))
    segments: list[dict] = []
    active: list[dict] = []
    max_segment_sec = 3.5

    def flush(reason: str) -> None:
        nonlocal active
        if not active:
            return
        segment = _make_segment(len(segments), active, reason)
        if segment["t_end"] - segment["t_start"] >= 1.0:
            segments.append(segment)
        active = []

    for record in records:
        signal = _has_action_signal(record)
        if not active:
            if signal:
                active = [record]
            continue
        if _should_split(active, record, max_segment_sec):
            flush("split by focus/object/action-change/max duration")
            if signal:
                active = [record]
            continue
        if signal or _can_bridge(active[-1], record):
            active.append(record)
        else:
            flush("ended by no-action gap")
            if signal:
                active = [record]
    flush("end of range")
    return segments


def _make_segment(idx: int, records: list[dict], reason: str) -> dict:
    t_start = float(records[0]["t_start"])
    t_end = float(records[-1]["t_end"])
    scene_types = [_scene(record).get("scene_type", "unknown") for record in records]
    focuses = [_focus(record) for record in records if _focus(record) not in {"", "unknown", "none"}]
    return {
        "segment_id": f"seg_v2_{idx:04d}",
        "t_start": t_start,
        "t_end": t_end,
        "context_t_start": max(0.0, t_start - 1.0),
        "context_t_end": t_end + 1.0,
        "source_window_indices": [int(r["window_idx"]) for r in records],
        "reason": reason,
        "dominant_scene_type": _mode(scene_types, "unknown"),
        "dominant_focus": _mode(focuses, "unknown"),
        "stable_objects_in_segment": _stable_objects(records),
        "w_action_summaries": [_w_summary(record) for record in records],
        "scene_summaries": [_scene_summary(record) for record in records],
        "rolling_context_summaries": [_rolling_summary(record) for record in records],
        "proposal_version": "v2_micro_action_segments",
    }


def _should_split(active: list[dict], record: dict, max_segment_sec: float) -> bool:
    duration = float(record["t_end"]) - float(active[0]["t_start"])
    if duration > max_segment_sec:
        return True
    if _no_action_gap(active[-1], record) > 1.0:
        return True
    if _has_action_signal(record) and _has_action_signal(active[-1]):
        if _action_signature(record) and _action_signature(active[-1]) and _action_signature(record) != _action_signature(active[-1]):
            # A single-second W verb/object shift is often just sensor-rate noise or
            # a support contact before the conceptual micro-action. Let the
            # multimodal reasoner see a short local sequence before splitting.
            if duration >= 2.5:
                return True
        if _focus_changed(active[-1], record) and not _stable_overlap(active[-1], record):
            return True
    return False


def _has_action_signal(record: dict) -> bool:
    w = (record.get("W_action") or {}).get("record") or {}
    status = _scene_status(record)
    return bool(w.get("action_present")) or status in {"visible_action", "uncertain_action"}


def _can_bridge(prev: dict, cur: dict) -> bool:
    return _no_action_gap(prev, cur) <= 1.0 and (_focus(prev) == _focus(cur) or _stable_overlap(prev, cur))


def _no_action_gap(prev: dict, cur: dict) -> float:
    if _has_action_signal(prev) or _has_action_signal(cur):
        return max(0.0, float(cur["t_start"]) - float(prev["t_end"]))
    return float(cur["t_end"]) - float(prev["t_start"])


def _action_signature(record: dict) -> str:
    actions = _valid_actions(record)
    if not actions:
        return ""
    first = actions[0]
    obj = first.get("direct_object")
    if isinstance(obj, dict):
        name = obj.get("name", "")
    else:
        name = obj or first.get("object", "")
    return f"{str(first.get('verb','')).lower()}|{str(name).lower()}"


def _valid_actions(record: dict) -> list[dict]:
    w = (record.get("W_action") or {}).get("record") or {}
    actions = w.get("valid_actions") or []
    if actions:
        return [a for a in actions if isinstance(a, dict)]
    parsed = w.get("parsed") or {}
    actions = parsed.get("actions") or []
    return [a for a in actions if isinstance(a, dict)]


def _focus_changed(a: dict, b: dict) -> bool:
    fa, fb = _focus(a), _focus(b)
    return fa not in {"", "unknown", "none"} and fb not in {"", "unknown", "none"} and fa != fb


def _stable_overlap(a: dict, b: dict) -> bool:
    return bool(_stable_keys(a) & _stable_keys(b))


def _stable_keys(record: dict) -> set[str]:
    rolling = record.get("rolling_context_v1_1") or {}
    objs = ((rolling.get("after_window") or {}).get("stable_objects") or [])[:8]
    return {str(o.get("canonical_name") or o.get("raw_name") or "").lower() for o in objs if (o.get("canonical_name") or o.get("raw_name"))}


def _stable_objects(records: list[dict]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for record in records:
        rolling = record.get("rolling_context_v1_1") or {}
        for obj in ((rolling.get("after_window") or {}).get("stable_objects") or []):
            key = str(obj.get("canonical_name") or obj.get("raw_name") or "").lower()
            if key and key not in by_name:
                by_name[key] = obj
    return list(by_name.values())[:12]


def _w_summary(record: dict) -> dict:
    actions = _valid_actions(record)
    return {
        "window_idx": int(record.get("window_idx", -1)),
        "t_start": float(record.get("t_start", 0.0)),
        "t_end": float(record.get("t_end", 0.0)),
        "action_present": bool(((record.get("W_action") or {}).get("record") or {}).get("action_present")),
        "actions": [
            {
                "verb": a.get("verb", ""),
                "object": _object_name(a),
                "part": (a.get("object_attributes") or {}).get("part", "none"),
                "confidence": a.get("confidence", 0.0),
            }
            for a in actions
        ],
    }


def _object_name(action: dict) -> str:
    obj = action.get("direct_object")
    if isinstance(obj, dict):
        return str(obj.get("name") or "unknown")
    return str(obj or action.get("object") or "unknown")


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
    return str((rolling.get("after_window") or {}).get("current_focus") or _scene(record).get("current_focus") or "unknown").lower()


def _mode(values: list[Any], default: str) -> str:
    vals = [str(v) for v in values if str(v)]
    return Counter(vals).most_common(1)[0][0] if vals else default


# v3 overlap/context behavior
def _propose_l1_reasoning_segments_with_context(combined_records: list[dict]) -> list[dict]:
    records = sorted(combined_records, key=lambda r: float(r.get("t_start", 0.0)))
    segments: list[dict] = []
    active: list[dict] = []
    max_segment_sec = 3.0

    def flush(reason: str) -> None:
        nonlocal active
        if not active:
            return
        segment = _make_segment_with_context(len(segments), active, reason)
        if segment["t_end"] - segment["t_start"] >= 1.0:
            segments.append(segment)
        active = []

    for record in records:
        signal = _has_action_signal(record)
        if not active:
            if signal:
                active = [record]
            continue
        if _should_split(active, record, max_segment_sec):
            flush("split by focus/action evidence/max duration with overlap context")
            if signal:
                active = [record]
            continue
        if signal or _can_bridge(active[-1], record):
            active.append(record)
        else:
            flush("ended by no-action gap")
            if signal:
                active = [record]
    flush("end of range")
    return segments


def _make_segment_with_context(idx: int, records: list[dict], reason: str) -> dict:
    segment = _make_segment(idx, records, reason)
    segment["segment_id"] = f"seg_context_{idx:04d}"
    segment["context_t_start"] = max(0.0, float(segment["t_start"]) - 1.5)
    segment["context_t_end"] = float(segment["t_end"]) + 1.5
    segment["proposal_version"] = "visual_first_micro_action_segments"
    return segment


def _should_split(active: list[dict], record: dict, max_segment_sec: float) -> bool:
    duration = float(record["t_end"]) - float(active[0]["t_start"])
    if duration > max_segment_sec:
        return True
    if _no_action_gap(active[-1], record) > 1.0:
        return True
    if _has_action_signal(record) and _has_action_signal(active[-1]):
        if _focus(active[-1]) != _focus(record) and not _stable_overlap(active[-1], record):
            return True
    return False


def _can_bridge(prev: dict, cur: dict) -> bool:
    return _no_action_gap(prev, cur) <= 1.0 and (_focus(prev) == _focus(cur) or _stable_overlap(prev, cur))


# Evidence-ledger relabeling behavior
def _propose_l1_reasoning_segments_with_ledger(combined_records: list[dict]) -> list[dict]:
    segments = _propose_l1_reasoning_segments_with_context(combined_records)
    for idx, segment in enumerate(segments):
        segment["segment_id"] = f"seg_ledger_{idx:04d}"
        segment["proposal_version"] = "evidence_ledger_micro_action_segments"
    return segments


def propose_l1_reasoning_segments(combined_records: list[dict]) -> list[dict]:
    """Return canonical evidence-faithful micro-action segments."""
    segments = _propose_l1_reasoning_segments_with_ledger(combined_records)
    for idx, segment in enumerate(segments):
        segment["segment_id"] = f"seg_{idx:04d}"
        segment["proposal_version"] = "evidence_faithful_micro_action_segments"
    return segments
