"""Scene-aware rolling context helpers for scene_obs_v1 diagnostics.

Import check:
  python -c "from code.scene.rolling_context import update_scene_rolling_context, render_scene_context_object_only"
"""

from __future__ import annotations

from typing import Any

GENERIC_NAMES = {"object", "item", "thing", "area", "unknown", "part"}
SPECIFIC_NAMES = {"mug", "coffee machine", "refrigerator", "cupboard", "drawer", "button", "bottle"}


def _clean(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = " ".join(str(value).strip().lower().split())
    return text if text else default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _object_score(obj: dict, max_seen: float) -> float:
    name = _clean(obj.get("name"), "")
    confidence = _float(obj.get("confidence"), 0.0)
    age = max(0.0, max_seen - _float(obj.get("last_seen_t"), 0.0))
    recency = max(0.0, 0.2 - 0.02 * age)
    specificity = 0.0 if name in GENERIC_NAMES else 0.3
    if any(specific in name for specific in SPECIFIC_NAMES):
        specificity = 0.3
    return confidence + recency + specificity


def _merge_objects(previous: list[dict], observed: list[dict], current_time_end: float) -> list[dict]:
    merged: dict[str, dict] = {}
    for obj in previous:
        name = _clean(obj.get("name"), "")
        if not name:
            continue
        merged[name] = {
            "name": name,
            "category": _clean(obj.get("category")),
            "location": _clean(obj.get("location")),
            "state": _clean(obj.get("state")),
            "confidence": _float(obj.get("confidence"), 0.0),
            "last_seen_t": _float(obj.get("last_seen_t"), 0.0),
        }
    for obj in observed:
        name = _clean(obj.get("name"), "")
        if not name or name in {"unknown", "none"}:
            continue
        incoming = {
            "name": name,
            "category": _clean(obj.get("category")),
            "location": _clean(obj.get("location")),
            "state": _clean(obj.get("state")),
            "confidence": _float(obj.get("confidence"), 0.0),
            "last_seen_t": float(current_time_end),
        }
        old = merged.get(name)
        if old is None:
            merged[name] = incoming
        else:
            old["location"] = incoming["location"] if incoming["location"] != "unknown" else old["location"]
            old["state"] = incoming["state"] if incoming["state"] != "unknown" else old["state"]
            old["category"] = incoming["category"] if incoming["category"] != "unknown" else old["category"]
            old["confidence"] = max(old["confidence"], incoming["confidence"])
            old["last_seen_t"] = current_time_end
    max_seen = max((obj["last_seen_t"] for obj in merged.values()), default=current_time_end)
    return sorted(merged.values(), key=lambda obj: _object_score(obj, max_seen), reverse=True)[:12]


def _action_summary(action: dict) -> str:
    return f"{_clean(action.get('verb'), '')} {_clean(action.get('direct_object'), '')}".strip()


def update_scene_rolling_context(
    previous_context: dict | None,
    scene_obs_record: dict,
    w_action_record: dict | None,
    current_time_end: float,
) -> dict:
    previous_context = previous_context or {}
    scene_obs = scene_obs_record.get("scene_observation") or {}
    action_presence = scene_obs.get("action_presence") or {}
    t_start = float(scene_obs_record.get("t_start", 0.0))
    t_end = float(scene_obs_record.get("t_end", current_time_end))

    stable_objects = _merge_objects(
        previous_context.get("stable_objects") or [],
        scene_obs.get("visible_objects") or [],
        float(current_time_end),
    )
    recent_action_evidence = list(previous_context.get("recent_action_evidence") or [])
    if w_action_record and w_action_record.get("action_present"):
        for action in w_action_record.get("valid_actions") or []:
            summary = _action_summary(action)
            if summary:
                recent_action_evidence.append(
                    {
                        "t_start": float(w_action_record.get("t_start", t_start)),
                        "t_end": float(w_action_record.get("t_end", t_end)),
                        "summary": summary,
                        "source": "W_action",
                        "confidence": _float(action.get("confidence"), 0.0),
                    }
                )
    recent_action_evidence = recent_action_evidence[-4:]

    uncertain = list(previous_context.get("uncertain_action_windows") or [])
    if action_presence.get("status") == "uncertain_action":
        uncertain.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "region": _clean(action_presence.get("likely_action_region"), "unknown"),
                "reason": _clean(scene_obs.get("uncertainty_notes"), "uncertain target-frame action"),
            }
        )
    uncertain = uncertain[-5:]

    status = action_presence.get("status")
    previous_no_action = int(previous_context.get("no_action_windows_recent", 0))
    if status == "visible_action":
        no_action_recent = 0
    elif status == "no_action":
        no_action_recent = previous_no_action + 1
    else:
        no_action_recent = previous_no_action

    scene_type = _clean(scene_obs.get("scene_type"))
    if scene_type == "unknown":
        scene_type = _clean(previous_context.get("scene_type"))
    current_focus = _clean(scene_obs.get("current_focus"))
    if current_focus == "unknown":
        current_focus = _clean(previous_context.get("current_focus"))

    return {
        "context_id": int(previous_context.get("context_id", 0)) + 1,
        "valid_until_t": float(current_time_end),
        "scene_type": scene_type,
        "stable_objects": stable_objects,
        "current_focus": current_focus,
        "recent_action_evidence": recent_action_evidence,
        "uncertain_action_windows": uncertain,
        "no_action_windows_recent": no_action_recent,
    }


def render_scene_context_object_only(context: dict | None) -> str:
    if not context:
        return "Scene context so far: none."
    objects = context.get("stable_objects") or []
    object_bits = []
    for obj in objects[:8]:
        name = _clean(obj.get("name"), "")
        location = _clean(obj.get("location"))
        state = _clean(obj.get("state"))
        if not name:
            continue
        bit = f"{name} at {location}" if location != "unknown" else name
        if state not in {"unknown", "stationary"}:
            bit += f" ({state})"
        object_bits.append(bit)
    uncertain = context.get("uncertain_action_windows") or []
    uncertain_bits = [
        _clean(item.get("region"), "")
        for item in uncertain[-2:]
        if _clean(item.get("region"), "") not in {"", "none", "unknown"}
    ]
    lines = [
        "Scene context so far:",
        f"- scene_type: {_clean(context.get('scene_type'))}",
        "- stable_objects: " + ("; ".join(object_bits) if object_bits else "none"),
        f"- current_focus: {_clean(context.get('current_focus'))}",
    ]
    if uncertain_bits:
        lines.append("- uncertain_action_regions: " + "; ".join(uncertain_bits))
    return "\n".join(lines)
