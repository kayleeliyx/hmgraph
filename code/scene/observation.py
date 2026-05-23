"""Parser and helpers for scene_obs_v1 records.

Import check:
  python -c "from code.scene.observation import parse_scene_observation_json_with_repair, derive_action_present_from_scene, summarize_scene_observation"
"""

from __future__ import annotations

import json
import re
from typing import Any

SCENE_TYPES = {
    "kitchen",
    "bathroom",
    "desk",
    "living_room",
    "bedroom",
    "hallway",
    "outdoor",
    "vehicle",
    "unknown",
}
OBJECT_CATEGORIES = {
    "appliance",
    "container",
    "tool",
    "utensil",
    "food",
    "furniture",
    "fixture",
    "surface",
    "door",
    "drawer",
    "other",
    "unknown",
}
ACTION_STATUSES = {"visible_action", "uncertain_action", "no_action"}


def _clean_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = " ".join(str(value).strip().lower().split())
    return text if text else default


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, 0.0), 1.0)


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() == "true"


def _strip_code_fence(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        return text.strip(), True
    return text, False


def _json_slice(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    if start >= 0:
        return text[start:]
    return text


def _balance_json_text(text: str) -> str:
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
    candidate = text.rstrip()
    if in_string:
        quote_idx = candidate.rfind('"', 0, max(0, len(candidate) - 1))
        candidate = candidate[:quote_idx] if quote_idx > 0 else candidate
    candidate = re.sub(r',\s*("[^"]*"\s*)?$', "", candidate.rstrip())
    candidate = re.sub(r',\s*$', "", candidate.rstrip())
    candidate += "]" * max(candidate.count("[") - candidate.count("]"), 0)
    candidate += "}" * max(candidate.count("{") - candidate.count("}"), 0)
    return candidate


def _normalize_visible_object(obj: dict) -> dict:
    name = _clean_text(obj.get("name"), "")
    category = _clean_text(obj.get("category"))
    if category not in OBJECT_CATEGORIES:
        category = "unknown"
    return {
        "name": name or "unknown",
        "category": category,
        "specific_type": _clean_text(obj.get("specific_type")),
        "location": _clean_text(obj.get("location")),
        "relative_position": _clean_text(obj.get("relative_position")),
        "state": _clean_text(obj.get("state")),
        "confidence": _clamp_float(obj.get("confidence"), 0.0),
    }


def _fill_defaults(data: dict) -> dict:
    scene_type = _clean_text(data.get("scene_type"))
    if scene_type not in SCENE_TYPES:
        scene_type = "unknown"
    visible_objects = data.get("visible_objects")
    if not isinstance(visible_objects, list):
        visible_objects = []
    hands = data.get("hands")
    if not isinstance(hands, dict):
        hands = {}
    action_presence = data.get("action_presence")
    missing_action_presence = not isinstance(action_presence, dict)
    if missing_action_presence:
        action_presence = {}
    evidence = action_presence.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    status = _clean_text(action_presence.get("status"), "uncertain_action")
    if status not in ACTION_STATUSES:
        status = "uncertain_action"
    return {
        "scene_type": scene_type,
        "view_context": _clean_text(data.get("view_context")),
        "spatial_layout": _clean_text(data.get("spatial_layout")),
        "visible_objects": [
            _normalize_visible_object(obj)
            for obj in visible_objects[:8]
            if isinstance(obj, dict)
        ],
        "hands": {
            "left_hand_visible": _bool(hands.get("left_hand_visible", False)),
            "right_hand_visible": _bool(hands.get("right_hand_visible", False)),
            "hand_locations": _clean_text(hands.get("hand_locations"), "none"),
        },
        "action_presence": {
            "status": status,
            "confidence": _clamp_float(action_presence.get("confidence"), 0.5),
            "evidence": {
                "hand_visible": _bool(evidence.get("hand_visible", False)),
                "hand_object_contact": _bool(evidence.get("hand_object_contact", False)),
                "object_motion": _bool(evidence.get("object_motion", False)),
                "object_state_change": _bool(evidence.get("object_state_change", False)),
                "tool_effect": _bool(evidence.get("tool_effect", False)),
                "material_transfer": _bool(evidence.get("material_transfer", False)),
            },
            "likely_action_region": _clean_text(action_presence.get("likely_action_region"), "unknown" if missing_action_presence else "none"),
            "no_action_reason": _clean_text(
                action_presence.get("no_action_reason"),
                "missing action_presence field" if missing_action_presence else "none",
            ),
        },
        "current_focus": _clean_text(data.get("current_focus")),
        "uncertainty_notes": _clean_text(data.get("uncertainty_notes"), "none"),
    }


def parse_scene_observation_json_with_repair(raw_output: str) -> tuple[dict, bool, dict]:
    repair_info = {"used_repair": False, "repair_method": "none", "parse_error": None}
    text, stripped = _strip_code_fence(raw_output if isinstance(raw_output, str) else "")
    if stripped:
        repair_info.update({"used_repair": True, "repair_method": "strip_code_fence"})
    candidates = [("none", text), ("balanced_braces", _balance_json_text(_json_slice(text)))]
    last_error: str | None = None
    for method, candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            parsed = _fill_defaults(data)
            if method != "none" or stripped:
                repair_info["used_repair"] = True
                repair_info["repair_method"] = method if method != "none" else "strip_code_fence"
            return parsed, True, repair_info
        except Exception as exc:
            last_error = str(exc)
    repair_info.update({"used_repair": True, "repair_method": "failed", "parse_error": last_error})
    return _fill_defaults({}), False, repair_info


def derive_action_present_from_scene(scene_obs: dict) -> bool:
    return (scene_obs.get("action_presence") or {}).get("status") == "visible_action"


def summarize_scene_observation(scene_obs: dict) -> dict:
    action_presence = scene_obs.get("action_presence") or {}
    return {
        "scene_type": scene_obs.get("scene_type", "unknown"),
        "view_context": scene_obs.get("view_context", "unknown"),
        "main_objects": [obj.get("name", "unknown") for obj in (scene_obs.get("visible_objects") or [])[:8]],
        "current_focus": scene_obs.get("current_focus", "unknown"),
        "action_status": action_presence.get("status", "uncertain_action"),
        "action_confidence": _clamp_float(action_presence.get("confidence"), 0.0),
    }
