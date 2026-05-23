"""Improved scene_obs_v1 rolling context builder using existing W and scene records.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.scene.context_builder import update_scene_rolling_context_v1_1, render_scene_context_object_only_v1_1, calibrate_action_presence"
"""

from __future__ import annotations

from typing import Any

try:
    from code.scene.object_utils import (
        canonicalize_object_name,
        is_generic_object_name,
        is_part_object_name,
        make_object_record,
        merge_object_record,
        specificity_score,
    )
except ImportError:
    from code.scene.object_utils import (
        canonicalize_object_name,
        is_generic_object_name,
        is_part_object_name,
        make_object_record,
        merge_object_record,
        specificity_score,
    )

ACTION_LIKE_FOCUS_WORDS = {
    "operating",
    "using",
    "picking",
    "pick",
    "opening",
    "closing",
    "pressing",
    "holding",
    "manipulating",
    "placing",
    "moving",
    "grabbing",
    "taking",
    "reaching",
}

VAGUE_REGIONS = {"near appliance", "near object", "unknown", "none", "object", "appliance"}


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


def extract_stable_objects_from_scene(scene_obs_record: dict) -> list[dict]:
    scene_obs = scene_obs_record.get("scene_observation") or {}
    t_end = float(scene_obs_record.get("t_end", 0.0))
    objects: list[dict] = []
    for obj in scene_obs.get("visible_objects") or []:
        raw_name = obj.get("name", "")
        canonical = canonicalize_object_name(raw_name)
        if canonical in {"", "unknown", "none"}:
            continue
        objects.append(
            make_object_record(
                raw_name,
                category=obj.get("category", "unknown"),
                location=obj.get("location", "unknown"),
                state=obj.get("state", "unknown"),
                confidence=_float(obj.get("confidence"), 0.0),
                source="scene_observation",
                current_time_end=t_end,
            )
        )
    return objects


def extract_stable_objects_from_w_action(w_action_record: dict) -> list[dict]:
    t_end = float(w_action_record.get("t_end", 0.0))
    actions = _valid_actions(w_action_record)
    objects: list[dict] = []
    for action in actions:
        confidence = _float(action.get("confidence"), 0.8)
        direct_rich = action.get("direct_object_rich") or {}
        raw_object = direct_rich.get("name") or action.get("direct_object")
        if raw_object and not is_generic_object_name(raw_object):
            spatial = action.get("spatial_context") or {}
            part_of = _infer_part_of_from_spatial(raw_object, spatial)
            objects.append(
                make_object_record(
                    raw_object,
                    category=direct_rich.get("category", "unknown"),
                    location=spatial.get("location", "unknown"),
                    state=_state_from_action(action),
                    confidence=confidence,
                    source="W_action",
                    current_time_end=t_end,
                    part_of=part_of,
                )
            )
        part = ((action.get("object_attributes") or {}).get("part") or "").strip()
        if part and canonicalize_object_name(part) not in {"whole_object", "whole object", "none", "unknown"}:
            part_of = raw_object if raw_object and canonicalize_object_name(raw_object) != canonicalize_object_name(part) else None
            objects.append(
                make_object_record(
                    part,
                    category="part",
                    location=(action.get("spatial_context") or {}).get("location", "unknown"),
                    state=_state_from_action(action),
                    confidence=min(confidence, 0.7),
                    source="W_action",
                    current_time_end=t_end,
                    part_of=part_of,
                )
            )
        spatial = action.get("spatial_context") or {}
        for near in spatial.get("near") or []:
            if near and not is_generic_object_name(near):
                objects.append(
                    make_object_record(
                        near,
                        category="unknown",
                        location=f"near {canonicalize_object_name(raw_object or 'target object')}",
                        state="unknown",
                        confidence=0.6,
                        source="W_action",
                        current_time_end=t_end,
                    )
                )
        contained_in = spatial.get("contained_in")
        if contained_in and not is_generic_object_name(contained_in) and canonicalize_object_name(contained_in) != "none":
            objects.append(
                make_object_record(
                    contained_in,
                    category="container",
                    location=f"contains {canonicalize_object_name(raw_object or 'object')}",
                    state="unknown",
                    confidence=0.7,
                    source="W_action",
                    current_time_end=t_end,
                )
            )
    return objects


def calibrate_action_presence(scene_obs_record: dict, w_action_record: dict) -> dict:
    scene_obs = scene_obs_record.get("scene_observation") or {}
    presence = scene_obs.get("action_presence") or {}
    evidence = presence.get("evidence") or {}
    raw_status = _clean(presence.get("status"), "uncertain_action")
    raw_confidence = _float(presence.get("confidence"), 0.5)
    w_present = bool(w_action_record.get("action_present"))
    scene_present = raw_status == "visible_action"
    w_generic = _w_primary_object_is_generic(w_action_record)
    region = _clean(presence.get("likely_action_region"), "unknown")
    direct_evidence = any(
        bool(evidence.get(key))
        for key in ("hand_object_contact", "object_motion", "object_state_change", "tool_effect", "material_transfer")
    )

    calibrated_status = raw_status if raw_status in {"visible_action", "uncertain_action", "no_action"} else "uncertain_action"
    reason = "kept raw scene status"
    confidence = raw_confidence

    if w_present and raw_status == "visible_action":
        calibrated_status = "visible_action"
        reason = "W action and scene observation both indicate visible_action"
    elif w_present and raw_status == "no_action":
        calibrated_status = "uncertain_action"
        confidence = min(max(raw_confidence, 0.5), 0.8)
        reason = "W action present but scene observation says no_action"
    elif (not w_present) and raw_status == "visible_action":
        if direct_evidence:
            calibrated_status = "uncertain_action"
            confidence = min(raw_confidence, 0.75)
            reason = "scene sees possible action but W validity gate found no valid action"
        else:
            calibrated_status = "no_action"
            confidence = min(raw_confidence, 0.7)
            reason = "scene visible_action has weak evidence and W has no valid action"

    only_hand_visible = bool(evidence.get("hand_visible")) and not direct_evidence
    if raw_status == "visible_action" and raw_confidence >= 0.95 and only_hand_visible:
        calibrated_status = "uncertain_action"
        confidence = min(confidence, 0.7)
        reason = "overconfident visible_action with weak evidence"

    if raw_status == "visible_action" and region in VAGUE_REGIONS and w_generic:
        calibrated_status = "uncertain_action"
        confidence = min(confidence, 0.7)
        reason = "vague visible action region and generic W action object"

    return {
        "raw_scene_status": raw_status,
        "calibrated_status": calibrated_status,
        "confidence": confidence,
        "reason": reason,
        "w_action_present": w_present,
        "scene_action_present": scene_present,
    }


def sanitize_current_focus(scene_obs_record: dict, stable_objects: list[dict]) -> str:
    scene_obs = scene_obs_record.get("scene_observation") or {}
    focus = _clean(scene_obs.get("current_focus"), "")
    view_context = _clean(scene_obs.get("view_context"), "")
    focus_words = set(focus.replace("/", " ").split())
    if focus and not (focus_words & ACTION_LIKE_FOCUS_WORDS):
        return focus
    top_objects = [
        _display_object(obj)
        for obj in stable_objects
        if not is_generic_object_name(obj.get("canonical_name") or obj.get("raw_name", ""))
    ][:3]
    if view_context and view_context != "unknown":
        if top_objects:
            return f"{view_context} near {', '.join(top_objects[:2])}"
        return view_context
    if top_objects:
        return "area near " + ", ".join(top_objects[:2])
    return "unknown"


def update_scene_rolling_context_v1_1(
    previous_context: dict | None,
    scene_obs_record: dict,
    w_action_record: dict,
    current_time_end: float,
) -> dict:
    previous_context = previous_context or {}
    scene_obs = scene_obs_record.get("scene_observation") or {}
    calibrated = calibrate_action_presence(scene_obs_record, w_action_record)
    scene_objects = extract_stable_objects_from_scene(scene_obs_record)
    w_objects = extract_stable_objects_from_w_action(w_action_record)

    merged = _merge_object_lists(previous_context.get("stable_objects") or [], scene_objects + w_objects, current_time_end)
    stable_objects = _rank_objects(merged, current_time_end)[:12]
    recent = _update_recent_action_evidence(previous_context.get("recent_action_evidence") or [], w_action_record)
    uncertain = list(previous_context.get("uncertain_action_windows") or [])
    if calibrated["calibrated_status"] == "uncertain_action":
        uncertain.append(
            {
                "t_start": float(scene_obs_record.get("t_start", 0.0)),
                "t_end": float(scene_obs_record.get("t_end", current_time_end)),
                "region": _clean((scene_obs.get("action_presence") or {}).get("likely_action_region"), "unknown"),
                "reason": calibrated["reason"],
            }
        )
    uncertain = uncertain[-5:]
    previous_no_action = int(previous_context.get("no_action_windows_recent", 0))
    if calibrated["calibrated_status"] == "visible_action":
        no_action_recent = 0
    elif calibrated["calibrated_status"] == "no_action":
        no_action_recent = previous_no_action + 1
    else:
        no_action_recent = previous_no_action
    scene_type = _clean(scene_obs.get("scene_type"))
    if scene_type == "unknown":
        scene_type = _clean(previous_context.get("scene_type"))

    return {
        "context_id": int(previous_context.get("context_id", 0)) + 1,
        "valid_until_t": float(current_time_end),
        "scene_type": scene_type,
        "stable_objects": stable_objects,
        "current_focus": sanitize_current_focus(scene_obs_record, stable_objects),
        "recent_action_evidence": recent,
        "uncertain_action_windows": uncertain,
        "no_action_windows_recent": no_action_recent,
    }


def render_scene_context_object_only_v1_1(context: dict | None) -> str:
    if not context:
        return "Scene context so far: none."
    object_bits = []
    for obj in (context.get("stable_objects") or [])[:8]:
        name = _display_object(obj)
        if not name:
            continue
        location = _clean(obj.get("location"))
        state = _clean(obj.get("state"))
        bit = f"{name} at {location}" if location not in {"unknown", "none"} else name
        if state not in {"unknown", "none", "stationary"}:
            bit += f" ({state})"
        object_bits.append(bit)
    uncertain_bits = [
        _clean(item.get("region"), "")
        for item in (context.get("uncertain_action_windows") or [])[-2:]
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


def _valid_actions(record: dict) -> list[dict]:
    for key in ("valid_actions", "parsed_valid_actions"):
        actions = record.get(key)
        if isinstance(actions, list):
            return [a for a in actions if isinstance(a, dict)]
    parsed = record.get("parsed") or {}
    actions = parsed.get("actions")
    if isinstance(actions, list):
        return [a for a in actions if isinstance(a, dict)]
    actions = record.get("actions")
    if isinstance(actions, list):
        return [a for a in actions if isinstance(a, dict)]
    return []


def _state_from_action(action: dict) -> str:
    interaction = action.get("interaction") or {}
    change = _clean(interaction.get("object_change"), "")
    if change and change not in {"unknown", "none", "no_visible_change", "no visible change"}:
        return change
    states = (action.get("object_attributes") or {}).get("state")
    if isinstance(states, list) and states:
        return _clean(states[0])
    return "unknown"


def _infer_part_of_from_spatial(raw_object: str, spatial: dict) -> str | None:
    if not is_part_object_name(raw_object):
        return None
    location = _clean(spatial.get("location"), "")
    for marker in ("on top of", "on the top of", "on", "of"):
        prefix = marker + " "
        if location.startswith(prefix):
            candidate = location[len(prefix) :]
            if candidate.startswith("the "):
                candidate = candidate[4:]
            if candidate.startswith("top of "):
                candidate = candidate[7:]
            if candidate.startswith("the "):
                candidate = candidate[4:]
            if candidate and not is_generic_object_name(candidate):
                return candidate
            return None
    near = spatial.get("near") or []
    specific_near = [item for item in near if item and not is_generic_object_name(item)]
    if len(specific_near) == 1:
        return specific_near[0]
    return None


def _merge_object_lists(previous: list[dict], observed: list[dict], current_time_end: float) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for obj in previous:
        key = _object_key(obj)
        merged[key] = dict(obj)
    for obj in observed:
        key = _object_key(obj)
        if key[0] in {"", "unknown", "none"}:
            continue
        merged[key] = merge_object_record(merged.get(key), obj, current_time_end)
    return list(merged.values())


def _object_key(obj: dict) -> tuple[str, str]:
    return (
        canonicalize_object_name(obj.get("canonical_name") or obj.get("raw_name", "")),
        canonicalize_object_name(obj.get("part_of", "none")),
    )


def _rank_objects(objects: list[dict], current_time_end: float) -> list[dict]:
    def score(obj: dict) -> tuple[float, float, float]:
        confidence = _float(obj.get("confidence"))
        age = max(0.0, current_time_end - _float(obj.get("last_seen_t"), current_time_end))
        recency = max(0.0, 1.0 - 0.05 * age)
        specificity = specificity_score(obj.get("canonical_name") or obj.get("raw_name", ""))
        return (specificity + confidence + recency, specificity, confidence)

    return sorted(objects, key=score, reverse=True)


def _update_recent_action_evidence(previous: list[dict], w_action_record: dict) -> list[dict]:
    recent = list(previous)
    for action in _valid_actions(w_action_record):
        summary = f"{_clean(action.get('verb'), '')} {_clean(action.get('direct_object'), '')}".strip()
        if not summary:
            continue
        entry = {
            "t_start": float(w_action_record.get("t_start", action.get("event_t_start", 0.0))),
            "t_end": float(w_action_record.get("t_end", action.get("event_t_end", 0.0))),
            "summary": summary,
            "source": "W_action",
            "confidence": _float(action.get("confidence"), 0.0),
        }
        if recent and recent[-1].get("summary") == summary:
            recent[-1] = entry
        else:
            recent.append(entry)
    return recent[-4:]


def _display_object(obj: dict) -> str:
    canonical = canonicalize_object_name(obj.get("canonical_name") or obj.get("raw_name", ""))
    part_of = canonicalize_object_name(obj.get("part_of", "none"))
    raw = canonicalize_object_name(obj.get("raw_name", canonical))
    if part_of not in {"none", "unknown", ""} and raw != canonical:
        return f"{canonical} of {part_of}"
    return raw if raw not in {"unknown", "none"} else canonical


def _w_primary_object_is_generic(w_action_record: dict) -> bool:
    actions = _valid_actions(w_action_record)
    if not actions:
        return False
    return is_generic_object_name(actions[0].get("direct_object", "unknown"))
