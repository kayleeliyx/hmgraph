"""Schema-level L1 event repair for v3.2.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.event_repair import repair_l1_event"
"""

from __future__ import annotations


CAUTIOUS_AREA_TERMS = ("control area", "top area", "handle area", "object part", "part", "area", "region", "top")
SPECIFIC_PART_NAMES = {"button", "knob", "switch", "handle", "lid", "cap", "door"}


def repair_l1_event(event: dict) -> tuple[dict, list[str]]:
    repaired = dict(event)
    repaired["direct_object"] = dict(event.get("direct_object") or {})
    notes: list[str] = []
    duration = _duration(repaired)
    desc = str(repaired.get("micro_action_description") or "").lower()

    if repaired.get("event_subtype") == "core_effect" and duration > 3.0:
        if _area_like(desc) or repaired.get("specificity_level") in {"coarse_but_supported", "ambiguous_but_useful"}:
            repaired["event_subtype"] = "ambiguous_part_manipulation"
            if _clear_visual(repaired) is False:
                repaired["evidence_type"] = "inferred"
            repaired["specificity_level"] = "ambiguous_but_useful"
            notes.append("changed long area-like core_effect to ambiguous_part_manipulation")
        else:
            notes.append("long core_effect retained with warning")

    if repaired.get("event_subtype") == "core_effect" and _uncertain_area_action(repaired, desc):
        repaired["event_subtype"] = "ambiguous_part_manipulation"
        repaired["evidence_type"] = "inferred"
        repaired["specificity_level"] = "ambiguous_but_useful"
        area = _extract_area_name(desc)
        if area:
            repaired["direct_object"]["name"] = area
        notes.append("changed uncertain area-like core_effect to inferred ambiguous_part_manipulation")

    name = str(repaired["direct_object"].get("name") or "").lower().strip()
    if name in SPECIFIC_PART_NAMES and _area_like(desc) and name not in desc:
        area = _extract_area_name(desc)
        if area:
            repaired["direct_object"]["name"] = area
            notes.append(f"changed specific part name to cautious area name: {area}")
        if not repaired["direct_object"].get("part_of"):
            repaired["direct_object"]["part_of"] = "unknown"

    if repaired.get("event_subtype") == "support_contact" and duration > 3.0 and _holding_or_maintaining(desc):
        repaired["event_subtype"] = "state_or_continuation"
        notes.append("changed long support/contact hold to state_or_continuation")

    if notes:
        repaired.setdefault("repair_notes", [])
        repaired["repair_notes"] = list(repaired.get("repair_notes") or []) + notes
    return repaired, notes


def _duration(event: dict) -> float:
    try:
        return max(0.0, float(event.get("t_end_est", 0.0)) - float(event.get("t_start_est", 0.0)))
    except Exception:
        return 0.0


def _area_like(text: str) -> bool:
    return any(term in text for term in CAUTIOUS_AREA_TERMS)


def _extract_area_name(text: str) -> str:
    for term in ("top control area", "control area", "top area", "handle area", "object part"):
        if term in text:
            return term
    if "area" in text:
        return "object area"
    if "part" in text:
        return "object part"
    return ""


def _clear_visual(event: dict) -> bool:
    return (event.get("confidence_rationale") or {}).get("visual_clarity") == "clear"


def _holding_or_maintaining(text: str) -> bool:
    return any(word in text for word in ("hold", "holding", "maintain", "keeping", "steady"))


def _uncertain_area_action(event: dict, text: str) -> bool:
    name = str((event.get("direct_object") or {}).get("name") or "").lower()
    cautious_name = any(term in name for term in ("area", "region", "part"))
    uncertainty = any(term in text for term in (" or ", "unclear", "uncertain", "possibly", "appears to", "seems to"))
    return _area_like(text) and (cautious_name or uncertainty or event.get("specificity_level") == "ambiguous_but_useful")
