"""Reparse cached W model responses into the shared, auditable W schema.

This module performs deterministic parsing, action gating, and target-window
timing checks only.  It never calls a model and never uses ground truth.
"""

from __future__ import annotations

from copy import deepcopy
import re

from code.w.prompt import (
    _parse_actions_from_data,
    filter_valid_target_actions,
    parse_w_json_with_repair,
)


REPROCESS_VERSION = "w_record_reprocess_v2"
_TIME_HINT = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$")


def _timing_status(action: dict, window_start: float, window_end: float) -> str:
    """Classify a normalized time hint relative to its 1-second W target."""
    hint = str(action.get("time_hint", "")).strip().lower()
    if hint in {"", "unknown", "none"}:
        return "full_window_fallback"
    match = _TIME_HINT.match(hint)
    if not match:
        return "invalid_hint_fallback"
    rel_start, rel_end = float(match.group(1)), float(match.group(2))
    if rel_end < rel_start:
        rel_start, rel_end = rel_end, rel_start
    span = max(0.0, window_end - window_start)
    if rel_end <= 0.0 or rel_start >= span:
        return "outside_target"
    if rel_start < 0.0 or rel_end > span:
        return "clipped_to_target"
    return "model_time_hint"


def _flatten(actions: list[dict]) -> dict:
    if not actions:
        return {
            "verb": "none",
            "direct_object": "none",
            "hand": "unknown",
            "with": "none",
            "purpose": "no accepted target-frame action",
            "manner": "none",
            "actions": [],
        }
    primary = next(
        (
            action
            for action in actions
            if action.get("interaction", {}).get("role") == "primary_manipulation"
        ),
        actions[0],
    )
    return {
        "verb": primary.get("verb", "unknown"),
        "direct_object": primary.get("direct_object", "unknown"),
        "hand": primary.get("interaction", {}).get("hand", "unknown"),
        "with": primary.get("interaction", {}).get("tool_used", "none"),
        "purpose": primary.get("interaction", {}).get("why_this_verb", "unknown"),
        "manner": primary.get("interaction", {}).get("role", "unknown"),
        "actions": actions,
    }


def reprocess_record(record: dict) -> dict:
    """Return a copy of one W record reparsed entirely from ``raw_output``."""
    repaired = deepcopy(record)
    window_start = float(repaired["t_start"])
    window_end = float(repaired["t_end"])
    raw_output = repaired.get("raw_output")
    if not isinstance(raw_output, str):
        raise ValueError(f"W window {repaired.get('window_idx')} has no raw_output string")

    data, json_ok, repair_info = parse_w_json_with_repair(raw_output)
    parsed_actions, schema_ok = _parse_actions_from_data(data, window_start, window_end)
    gated_actions, gate_rejections = filter_valid_target_actions(parsed_actions)

    accepted: list[dict] = []
    timing_rejections: list[dict] = []
    for action in gated_actions:
        action = deepcopy(action)
        action["timing_status"] = _timing_status(action, window_start, window_end)
        event_start = float(action.get("event_t_start", window_start))
        event_end = float(action.get("event_t_end", window_end))
        if action["timing_status"] == "outside_target" or event_end <= event_start:
            timing_rejections.append(
                {
                    "action": action,
                    "rejection_reason": "action time hint has no positive overlap with TARGET",
                    "rejection_stage": "target_timing_gate",
                }
            )
            continue
        accepted.append(action)

    rejected = []
    for rejection in gate_rejections:
        item = deepcopy(rejection)
        item.setdefault("rejection_stage", "target_evidence_gate")
        rejected.append(item)
    rejected.extend(timing_rejections)

    repaired["parsed"] = _flatten(accepted)
    repaired["parsed_actions"] = accepted
    repaired["parsed_raw_actions"] = parsed_actions
    repaired["valid_actions"] = accepted
    repaired["rejected_actions"] = rejected
    repaired["n_parsed_actions"] = len(parsed_actions)
    repaired["n_raw_actions"] = len(parsed_actions)
    repaired["n_valid_actions"] = len(accepted)
    repaired["n_rejected_actions"] = len(rejected)
    repaired["action_present"] = bool(accepted)
    repaired["parse_ok"] = bool(json_ok and schema_ok)
    repaired["parse_diagnostics"] = {
        "json_parse_ok": bool(json_ok),
        "schema_ok": bool(schema_ok),
        "used_repair": bool(repair_info.get("used_repair")),
        "repair_method": repair_info.get("repair_method", "none"),
        "parse_error": repair_info.get("parse_error"),
        "parsed_action_count": len(parsed_actions),
        "accepted_action_count": len(accepted),
        "rejected_action_count": len(rejected),
    }
    repaired["w_processing_version"] = REPROCESS_VERSION
    return repaired
