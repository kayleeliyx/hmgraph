"""Confidence calibration for L1 multimodal reasoner v3.2.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.confidence_reasoner import calibrate_l1_event_confidence"
"""

from __future__ import annotations


def calibrate_l1_event_confidence(event: dict, validation: dict) -> dict:
    updated = dict(event)
    raw = _clamp(event.get("confidence"))
    score = raw
    notes: list[str] = []
    rationale = event.get("confidence_rationale") or {}
    subtype = event.get("event_subtype")
    duration = _duration(event)

    if duration > 3.0 and subtype == "core_effect":
        score -= 0.20
        notes.append("long core_effect")
    elif duration > 3.0:
        score -= 0.05
        notes.append("long-duration event")
    if event.get("evidence_type") == "inferred":
        score -= 0.12
        notes.append("inferred event")
    if subtype == "ambiguous_part_manipulation":
        score -= 0.18
        notes.append("ambiguous part manipulation")
    elif subtype in {"support_contact", "state_or_continuation"}:
        score -= 0.12
        notes.append("support/state event")
    if rationale.get("visual_clarity") == "partially_occluded":
        score -= 0.08
        notes.append("partially occluded")
    elif rationale.get("visual_clarity") == "subtle":
        score -= 0.10
        notes.append("subtle visual evidence")
    elif rationale.get("visual_clarity") == "weak":
        score -= 0.18
        notes.append("weak visual evidence")
    if rationale.get("object_identity") == "supported_by_context":
        score -= 0.06
        notes.append("object identity context-supported")
    elif rationale.get("object_identity") == "ambiguous":
        score -= 0.16
        notes.append("ambiguous object identity")
    if rationale.get("temporal_localization") == "approximate":
        score -= 0.06
        notes.append("approximate temporal localization")
    elif rationale.get("temporal_localization") == "uncertain":
        score -= 0.14
        notes.append("uncertain temporal localization")
    if rationale.get("w_action_support") == "partial":
        score -= 0.07
        notes.append("partial W support")
    elif rationale.get("w_action_support") == "conflicting":
        score -= 0.18
        notes.append("conflicting W support")
    elif rationale.get("w_action_support") == "none" and rationale.get("visual_clarity") != "clear":
        score -= 0.12
        notes.append("no W support without clear visual evidence")
    if event.get("specificity_level") == "coarse_but_supported":
        score -= 0.14
        notes.append("coarse event may be L2-like")
    if event.get("hallucination_risk") == "medium":
        score -= 0.12
        notes.append("medium hallucination risk")
    elif event.get("hallucination_risk") == "high":
        score -= 0.30
        notes.append("high hallucination risk")
    if _schema_inconsistent(event):
        score -= 0.12
        notes.append("description/object schema inconsistency")

    updated["confidence_raw_model"] = raw
    updated["confidence"] = max(0.0, min(0.98, score))
    updated["confidence_calibration_notes"] = notes or ["no calibration penalty"]
    updated["qa_importance"] = _qa_importance(updated, duration)
    return updated


def _qa_importance(event: dict, duration: float) -> str:
    subtype = event.get("event_subtype")
    conf = float(event.get("confidence", 0.0))
    if subtype == "core_effect" and duration <= 3.0 and event.get("hallucination_risk") == "low" and conf >= 0.75:
        return "high"
    if subtype == "missing_from_W" and conf >= 0.75:
        return "high"
    if subtype in {"ambiguous_part_manipulation", "support_contact", "state_or_continuation"}:
        return "medium" if conf >= 0.65 else "low"
    return "medium"


def _duration(event: dict) -> float:
    try:
        return max(0.0, float(event.get("t_end_est", 0.0)) - float(event.get("t_start_est", 0.0)))
    except Exception:
        return 0.0


def _schema_inconsistent(event: dict) -> bool:
    desc = str(event.get("micro_action_description") or "").lower()
    name = str((event.get("direct_object") or {}).get("name") or "").lower()
    return any(x in desc for x in ("area", "region", "part", "top", "control")) and name in {"button", "knob", "switch", "handle", "lid", "cap", "door"} and name not in desc


def _clamp(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0
