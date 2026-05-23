"""Confidence recalibration for L1 multimodal postprocess v3.2.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.confidence import recalibrate_l1_confidence"
"""

from __future__ import annotations


def recalibrate_l1_confidence(event: dict, evidence_context: dict) -> tuple[dict, list[str]]:
    updated = dict(event)
    raw = _float(event.get("confidence"), 0.0)
    subtype = event.get("event_subtype", "")
    evidence_type = event.get("evidence_type", "")
    specificity = event.get("specificity_level", "")
    risk = event.get("hallucination_risk", "medium")
    visual = bool(event.get("visual_evidence")) or bool((event.get("structured_evidence") or {}).get("visual_discovery_candidate_ids"))
    uncertainty = _uncertainty_text(event)
    area_like = _area_like((event.get("direct_object") or {}).get("name", "")) or _area_like(event.get("micro_action_description", ""))
    duration = _duration(event)
    audit_support = set(evidence_context.get("w_audit_decisions") or [])
    ledger_disps = set(evidence_context.get("visual_dispositions") or [])
    notes: list[str] = []

    if risk == "high":
        score = 0.25
        notes.append("high hallucination risk")
    elif subtype == "core_effect" and evidence_type == "observed":
        if visual and not uncertainty and specificity == "specific_micro_action":
            score = 0.86
            notes.append("observed specific core_effect with visual support")
        elif visual or audit_support & {"confirmed", "corrected"}:
            score = 0.74
            notes.append("observed core_effect with some ambiguity")
        else:
            score = 0.66
            notes.append("observed core_effect retained with limited explicit evidence links")
    elif subtype == "ambiguous_part_manipulation" and evidence_type == "inferred":
        score = 0.56
        notes.append("inferred ambiguous part/area manipulation")
        if area_like:
            notes.append("area-like object naming is appropriate for uncertainty")
    elif subtype in {"support_contact", "state_or_continuation"}:
        score = 0.62
        notes.append("support/state event")
    elif evidence_type == "inferred":
        score = 0.58
        notes.append("inferred retained L1")
    else:
        score = 0.60
        notes.append("default retained L1 confidence band")

    if audit_support & {"confirmed", "corrected"}:
        score += 0.04
        notes.append("supported by normalized W audit")
    elif audit_support & {"partially_supported", "too_vague", "too_specific"}:
        score -= 0.03
        notes.append("partial or qualified W support")
    if ledger_disps & {"accepted_as_l1", "merged_into_l1"}:
        score += 0.03
        notes.append("visual candidate retained as final L1")
    if uncertainty:
        score -= 0.06
        notes.append("uncertainty notes indicate ambiguity")
    if duration > 3.5 and subtype == "core_effect":
        score -= 0.10
        notes.append("long core_effect")
    if subtype == "ambiguous_part_manipulation":
        score = min(score, 0.65)
    if subtype in {"support_contact", "state_or_continuation"}:
        score = min(score, 0.75)

    score = max(0.20, min(0.98, score))
    updated["confidence_before_postprocess"] = raw
    updated["confidence"] = round(score, 3)
    updated["confidence_calibration_notes"] = notes
    updated["qa_importance"] = _qa_importance(updated)
    return updated, notes


def _qa_importance(event: dict) -> str:
    subtype = event.get("event_subtype")
    conf = _float(event.get("confidence"), 0.0)
    if subtype == "core_effect" and event.get("evidence_type") == "observed" and conf >= 0.78:
        return "high"
    if subtype in {"ambiguous_part_manipulation", "support_contact", "state_or_continuation"}:
        return "medium" if conf >= 0.45 else "low"
    return "medium" if conf >= 0.55 else "low"


def _uncertainty_text(event: dict) -> bool:
    text = " ".join(str(event.get(k, "")) for k in ("uncertainty_notes", "why_not_more_specific", "micro_action_description")).lower()
    return any(x in text for x in ("unclear", "uncertain", "occlud", "subtle", "ambiguous", " or ", "less clear"))


def _area_like(value) -> bool:
    text = str(value or "").lower()
    return any(x in text for x in ("area", "region", "part", "top", "control"))


def _duration(event: dict) -> float:
    return max(0.0, _float(event.get("t_end"), 0.0) - _float(event.get("t_start"), 0.0))


def _float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
