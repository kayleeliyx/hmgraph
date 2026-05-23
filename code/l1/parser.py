"""Parse and validate L1 multimodal reasoner JSON.

The canonical pipeline still exposes versioned function names because output
schemas and stage code reference them, but the implementation is flattened here
instead of importing legacy parser modules.
"""



import json
import re
from typing import Any

EVENT_SUBTYPES = {"core_effect", "support_contact", "state_or_continuation", "ambiguous_part_manipulation", "missing_from_W"}


def parse_l1_reasoner_output_v3(raw_output: str) -> tuple[dict, bool, dict]:
    info = {"used_repair": False, "repair_method": "none", "parse_error": None, "parser_repairs": []}
    text = raw_output if isinstance(raw_output, str) else ""
    stripped = _strip_fence(text)
    if stripped != text:
        info["used_repair"] = True
        info["repair_method"] = "strip_code_fence"
        info["parser_repairs"].append("stripped markdown fence")
    candidates = [stripped, _first_json_object(stripped), _slice_json(stripped), _balance(_first_json_object(stripped)), _balance(_slice_json(stripped))]
    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            return _fill_defaults(data, info), True, info
        except Exception as exc:
            last_error = str(exc)
    info.update({"used_repair": True, "repair_method": "failed", "parse_error": last_error})
    return _fill_defaults({}, info), False, info


def validate_l1_reasoner_output_v3(parsed: dict, segment: dict) -> dict:
    warnings: list[str] = []
    repairs = list(parsed.get("_parser_repairs") or [])
    ct0, ct1 = float(segment["context_t_start"]), float(segment["context_t_end"])
    seg0, seg1 = float(segment["t_start"]), float(segment["t_end"])
    events = parsed.get("l1_events") or []
    high = support_state = ambiguous = missing = observed = inferred = 0
    for idx, event in enumerate(events):
        if event.get("should_create_l1") is not True:
            warnings.append(f"l1_events[{idx}] should_create_l1 is not true")
        if event.get("evidence_type") not in {"observed", "inferred"}:
            warnings.append(f"l1_events[{idx}] invalid evidence_type")
        observed += int(event.get("evidence_type") == "observed")
        inferred += int(event.get("evidence_type") == "inferred")
        subtype = event.get("event_subtype")
        if subtype not in EVENT_SUBTYPES:
            warnings.append(f"l1_events[{idx}] invalid event_subtype")
        support_state += int(subtype in {"support_contact", "state_or_continuation"})
        ambiguous += int(subtype == "ambiguous_part_manipulation")
        missing += int(subtype == "missing_from_W")
        if _looks_relative(event, segment):
            event["t_start_est"] = seg0 + float(event.get("t_start_est", 0.0))
            event["t_end_est"] = seg0 + float(event.get("t_end_est", 0.0))
            repairs.append(f"converted relative timestamps for l1_events[{idx}]")
        t0, t1 = float(event.get("t_start_est", seg0)), float(event.get("t_end_est", seg1))
        if not (ct0 <= t0 <= ct1) or not (ct0 <= t1 <= ct1) or t1 <= t0:
            event["t_start_est"] = max(ct0, min(ct1, t0))
            event["t_end_est"] = max(event["t_start_est"] + 0.01, min(ct1, t1 if t1 > t0 else seg1))
            warnings.append(f"l1_events[{idx}] time clamped to segment context")
        rationale = event.get("confidence_rationale") or {}
        conf = float(event.get("confidence", 0.0))
        if rationale.get("visual_clarity") in {"partially_occluded", "subtle", "weak"} and conf >= 1.0:
            warnings.append(f"l1_events[{idx}] confidence 1.0 despite non-clear visual evidence")
        if rationale.get("object_identity") == "ambiguous" and event.get("evidence_type") == "observed" and conf > 0.85:
            warnings.append(f"l1_events[{idx}] observed confidence too high for ambiguous object identity")
        if rationale.get("w_action_support") in {"conflicting", "none"} and conf > 0.85 and rationale.get("visual_clarity") != "clear":
            warnings.append(f"l1_events[{idx}] high confidence despite weak/conflicting W support")
        if subtype in {"support_contact", "state_or_continuation"} and conf > 0.9:
            warnings.append(f"l1_events[{idx}] support/state event confidence >0.9")
        structured = event.get("structured_evidence") or {}
        if not event.get("visual_evidence") and not structured.get("visual_discovery_candidate_ids"):
            warnings.append(f"l1_events[{idx}] has no visual evidence and no visual discovery candidate")
        obj = event.get("direct_object") or {}
        if _part_whole_reversal_hint(obj):
            warnings.append(f"l1_events[{idx}] possible part_of reversal")
        high += int(event.get("hallucination_risk") == "high")
    hypo = parsed.get("rejected_or_hypothesized_actions") or []
    for idx, item in enumerate(hypo):
        if item.get("should_create_l1") is not False:
            item["should_create_l1"] = False
            repairs.append(f"forced hypothesized[{idx}] should_create_l1=false")
    if "gt" in json.dumps(parsed).lower():
        warnings.append("parsed output appears to mention GT")
    audit = parsed.get("w_action_audit") or []
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "parser_repairs": repairs,
        "n_l1_events": len([e for e in events if e.get("should_create_l1") is True]),
        "n_observed": observed,
        "n_inferred": inferred,
        "n_hypothesized": len(hypo),
        "n_high_hallucination_risk": high,
        "n_support_or_state_events": support_state,
        "n_ambiguous_part_manipulation_events": ambiguous,
        "n_missing_from_W_events": missing,
        "n_w_confirmed": _audit_count(audit, {"confirmed"}),
        "n_w_partially_supported": _audit_count(audit, {"partially_supported"}),
        "n_w_likely_wrong_or_too_specific": _audit_count(audit, {"likely_wrong", "too_specific", "too_vague"}),
    }


def _fill_defaults(data: dict, info: dict) -> dict:
    fixed_events, rejected = [], []
    for idx, event in enumerate(data.get("l1_events") if isinstance(data.get("l1_events"), list) else []):
        if not isinstance(event, dict):
            continue
        evidence = _choice(event.get("evidence_type"), {"observed", "inferred", "hypothesized"}, "inferred")
        if evidence == "hypothesized":
            rejected.append(_fix_rejected(event, info))
            info["parser_repairs"].append(f"moved hypothesized l1_events[{idx}] to rejected")
            continue
        obj = event.get("direct_object") if isinstance(event.get("direct_object"), dict) else {}
        rationale = event.get("confidence_rationale") if isinstance(event.get("confidence_rationale"), dict) else {}
        structured = event.get("structured_evidence") if isinstance(event.get("structured_evidence"), dict) else {}
        fixed_events.append({
            "event_id": str(event.get("event_id") or f"event_{idx:02d}"),
            "micro_action_description": str(event.get("micro_action_description") or ""),
            "verb": str(event.get("verb") or "unknown").lower(),
            "direct_object": {
                "name": str(obj.get("name") or "unknown").lower(),
                "part_of": str(obj.get("part_of") or "none").lower(),
                "state_before": str(obj.get("state_before") or "unknown").lower(),
                "state_after": str(obj.get("state_after") or "unknown").lower(),
                "location_before": str(obj.get("location_before") or "unknown").lower(),
                "location_after": str(obj.get("location_after") or "unknown").lower(),
            },
            "t_start_est": _float(event.get("t_start_est")),
            "t_end_est": _float(event.get("t_end_est")),
            "event_subtype": _choice(event.get("event_subtype"), EVENT_SUBTYPES, "core_effect"),
            "evidence_type": evidence if evidence in {"observed", "inferred"} else "inferred",
            "should_create_l1": event.get("should_create_l1", True) is not False,
            "specificity_level": _choice(event.get("specificity_level"), {"specific_micro_action", "coarse_but_supported", "ambiguous_but_useful"}, "specific_micro_action"),
            "why_this_is_l1": str(event.get("why_this_is_l1") or ""),
            "why_not_more_specific": str(event.get("why_not_more_specific") or ""),
            "confidence": _clamp(event.get("confidence")),
            "confidence_rationale": {
                "visual_clarity": _choice(rationale.get("visual_clarity"), {"clear", "partially_occluded", "subtle", "weak"}, "weak"),
                "object_identity": _choice(rationale.get("object_identity"), {"clear", "supported_by_context", "ambiguous"}, "ambiguous"),
                "temporal_localization": _choice(rationale.get("temporal_localization"), {"clear", "approximate", "uncertain"}, "uncertain"),
                "w_action_support": _choice(rationale.get("w_action_support"), {"strong", "partial", "conflicting", "none"}, "none"),
                "scene_context_support": _choice(rationale.get("scene_context_support"), {"strong", "partial", "none"}, "none"),
                "uses_later_context_to_correct_W": bool(rationale.get("uses_later_context_to_correct_W")),
                "is_missing_from_W": bool(rationale.get("is_missing_from_W")),
            },
            "visual_evidence": event.get("visual_evidence") if isinstance(event.get("visual_evidence"), list) else [],
            "structured_evidence": {
                "w_action_windows": structured.get("w_action_windows") if isinstance(structured.get("w_action_windows"), list) else [],
                "scene_windows": structured.get("scene_windows") if isinstance(structured.get("scene_windows"), list) else [],
                "rolling_context_objects": structured.get("rolling_context_objects") if isinstance(structured.get("rolling_context_objects"), list) else [],
                "visual_discovery_candidate_ids": structured.get("visual_discovery_candidate_ids") if isinstance(structured.get("visual_discovery_candidate_ids"), list) else [],
            },
            "supporting_contacts_or_context": event.get("supporting_contacts_or_context") if isinstance(event.get("supporting_contacts_or_context"), list) else [],
            "uncertainty_notes": str(event.get("uncertainty_notes") or "none"),
            "hallucination_risk": _choice(event.get("hallucination_risk"), {"low", "medium", "high"}, "medium"),
        })
    rejected.extend(_fix_rejected(x, info) for x in (data.get("rejected_or_hypothesized_actions") if isinstance(data.get("rejected_or_hypothesized_actions"), list) else []) if isinstance(x, dict))
    activity = data.get("activity_context_or_l2_hint") if isinstance(data.get("activity_context_or_l2_hint"), dict) else {}
    return {
        "segment_id": str(data.get("segment_id") or "unknown"),
        "segment_time": data.get("segment_time") if isinstance(data.get("segment_time"), dict) else {},
        "activity_context_or_l2_hint": {"description": str(activity.get("description") or "none"), "confidence": _clamp(activity.get("confidence")), "why_this_is_not_l1": str(activity.get("why_this_is_not_l1") or "none")},
        "visual_discovery_assessment": data.get("visual_discovery_assessment") if isinstance(data.get("visual_discovery_assessment"), dict) else {"used_visual_candidates": [], "rejected_visual_candidates": [], "visual_actions_missing_from_W": []},
        "w_action_audit": data.get("w_action_audit") if isinstance(data.get("w_action_audit"), list) else [],
        "l1_events": fixed_events,
        "rejected_or_hypothesized_actions": rejected,
        "reasoning_summary": str(data.get("reasoning_summary") or ""),
        "parse_confidence": _clamp(data.get("parse_confidence")),
        "_parser_repairs": info.get("parser_repairs", []),
    }


def _fix_rejected(item: dict, info: dict) -> dict:
    if item.get("should_create_l1") is not False:
        info["used_repair"] = True
        info["parser_repairs"].append("forced rejected/hypothesized action should_create_l1=false")
    return {"description": str(item.get("description") or ""), "verb": str(item.get("verb") or "unknown").lower(), "object": str(item.get("object") or "unknown").lower(), "evidence_type": _choice(item.get("evidence_type"), {"hypothesized", "rejected"}, "hypothesized"), "should_create_l1": False, "reason": str(item.get("reason") or ""), "what_evidence_would_be_needed": str(item.get("what_evidence_would_be_needed") or "")}


def parse_visual_discovery_output_v3(raw_output: str) -> tuple[dict, bool, dict]:
    parsed, ok, info = parse_l1_reasoner_output_v3(raw_output)
    # Reuse robust JSON extraction but preserve visual-discovery-native fields.
    try:
        data = json.loads(_balance(_slice_json(_strip_fence(raw_output if isinstance(raw_output, str) else ""))))
        if isinstance(data, dict):
            parsed = _fill_visual_defaults(data)
            return parsed, True, info
    except Exception as exc:
        info.update({"used_repair": True, "repair_method": "failed", "parse_error": str(exc)})
    return _fill_visual_defaults({}), False, info


def _fill_visual_defaults(data: dict) -> dict:
    candidates = []
    for idx, item in enumerate(data.get("visual_micro_action_candidates") if isinstance(data.get("visual_micro_action_candidates"), list) else []):
        if not isinstance(item, dict):
            continue
        obj = item.get("direct_object") if isinstance(item.get("direct_object"), dict) else {}
        evidence = _choice(item.get("evidence_type"), {"observed", "inferred", "hypothesized"}, "inferred")
        candidates.append({"candidate_id": str(item.get("candidate_id") or f"visual_{idx:02d}"), "micro_action_description": str(item.get("micro_action_description") or ""), "verb": str(item.get("verb") or "unknown").lower(), "direct_object": {"name": str(obj.get("name") or "unknown").lower(), "part_of": str(obj.get("part_of") or "none").lower(), "state_before": str(obj.get("state_before") or "unknown").lower(), "state_after": str(obj.get("state_after") or "unknown").lower(), "location_before": str(obj.get("location_before") or "unknown").lower(), "location_after": str(obj.get("location_after") or "unknown").lower()}, "t_start_est": _float(item.get("t_start_est")), "t_end_est": _float(item.get("t_end_est")), "evidence_type": evidence, "visual_evidence": item.get("visual_evidence") if isinstance(item.get("visual_evidence"), list) else [], "why_this_may_be_l1": str(item.get("why_this_may_be_l1") or ""), "why_not_more_specific": str(item.get("why_not_more_specific") or ""), "uncertainty_notes": str(item.get("uncertainty_notes") or ""), "hallucination_risk": _choice(item.get("hallucination_risk"), {"low", "medium", "high"}, "medium"), "should_consider_for_final_l1": bool(item.get("should_consider_for_final_l1")) and evidence != "hypothesized"})
    return {"segment_id": str(data.get("segment_id") or "unknown"), "visual_segment_summary": str(data.get("visual_segment_summary") or ""), "main_visible_objects": data.get("main_visible_objects") if isinstance(data.get("main_visible_objects"), list) else [], "visual_micro_action_candidates": candidates, "possible_missing_or_subtle_actions": data.get("possible_missing_or_subtle_actions") if isinstance(data.get("possible_missing_or_subtle_actions"), list) else [], "part_whole_notes": data.get("part_whole_notes") if isinstance(data.get("part_whole_notes"), list) else [], "visual_uncertainty_summary": str(data.get("visual_uncertainty_summary") or ""), "parse_confidence": _clamp(data.get("parse_confidence"))}


def _looks_relative(event: dict, segment: dict) -> bool:
    try:
        t0, t1 = float(event.get("t_start_est")), float(event.get("t_end_est"))
    except Exception:
        return False
    seg0, seg1 = float(segment["t_start"]), float(segment["t_end"])
    duration = max(0.0, seg1 - seg0)
    return 0.0 <= t0 <= duration + 1.5 and 0.0 < t1 <= duration + 1.5 and seg0 > 1.0


def _part_whole_reversal_hint(obj: dict) -> bool:
    name, part = str(obj.get("name", "")), str(obj.get("part_of", ""))
    # Conservative check only: a likely reversal is when the parent field is a
    # generic component word while the direct object is a larger object phrase.
    # Valid forms such as "cupboard door" part_of "cupboard" should not warn.
    component_like = {"door", "handle", "knob", "button", "lid", "cap", "switch", "lever", "drawer"}
    return bool(part and part != "none" and part in component_like and name not in component_like)


def _audit_count(items: list, decisions: set[str]) -> int:
    return sum(1 for x in items if isinstance(x, dict) and str(x.get("decision", "")).lower() in decisions)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _slice_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _balance(text: str) -> str:
    text = re.sub(r",\s*$", "", text.rstrip())
    return text + "]" * max(0, text.count("[") - text.count("]")) + "}" * max(0, text.count("{") - text.count("}"))


def _choice(value: Any, allowed: set[str], default: str) -> str:
    value = str(value or "").lower()
    return value if value in allowed else default


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, _float(value)))


# Evidence-ledger parsing and validation additions
LEDGER_DISPOSITIONS = {"accepted_as_l1", "merged_into_l1", "support_for_l1", "rejected_as_hypothesis", "rejected_as_duplicate", "rejected_due_to_weak_evidence"}
AUDIT_DECISIONS = {"confirmed", "corrected", "downgraded_to_support", "too_vague", "too_specific", "partially_supported", "likely_wrong", "missed_by_visual_discovery", "not_applicable"}
CANDIDATE_ROLES = {"final_candidate", "support_only", "uncertain_candidate", "reject_hypothesis"}


def _parse_l1_reasoner_output_base(raw_output: str) -> tuple[dict, bool, dict]:
    info = {"used_repair": False, "repair_method": "none", "parse_error": None, "parser_repairs": []}
    text = raw_output if isinstance(raw_output, str) else ""
    stripped = _strip_fence(text)
    candidates = [stripped, _first_json_object(stripped), _slice_json(stripped), _balance(_first_json_object(stripped)), _balance(_slice_json(stripped))]
    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            parsed = _fill_defaults(data, info)
            parsed["visual_candidate_ledger"] = [_fix_ledger(x) for x in data.get("visual_candidate_ledger", []) if isinstance(x, dict)]
            parsed["w_action_audit"] = [_fix_audit(x) for x in data.get("w_action_audit", []) if isinstance(x, dict)]
            return parsed, True, info
        except Exception as exc:
            last_error = str(exc)
    info.update({"used_repair": True, "repair_method": "failed", "parse_error": last_error})
    parsed = _fill_defaults({}, info)
    parsed["visual_candidate_ledger"] = []
    parsed["w_action_audit"] = []
    return parsed, False, info


def _parse_visual_discovery_output_base(raw_output: str) -> tuple[dict, bool, dict]:
    info = {"used_repair": False, "repair_method": "none", "parse_error": None, "parser_repairs": []}
    text = raw_output if isinstance(raw_output, str) else ""
    stripped = _strip_fence(text)
    candidates = [stripped, _first_json_object(stripped), _slice_json(stripped), _balance(_first_json_object(stripped)), _balance(_slice_json(stripped))]
    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            parsed = _fill_visual_defaults(data)
            for item in parsed.get("visual_micro_action_candidates", []):
                role = _choice(item.get("candidate_role"), CANDIDATE_ROLES, "")
                if not role:
                    role = "reject_hypothesis" if item.get("evidence_type") == "hypothesized" else "uncertain_candidate"
                item["candidate_role"] = role
                if role == "reject_hypothesis" or item.get("evidence_type") == "hypothesized":
                    item["should_consider_for_final_l1"] = False
            for item in parsed.get("possible_missing_or_subtle_actions", []):
                if isinstance(item, dict):
                    item["candidate_role"] = _choice(item.get("candidate_role"), {"uncertain_candidate", "reject_hypothesis"}, "uncertain_candidate")
                    if item.get("evidence_type") == "hypothesized":
                        item["should_consider_for_final_l1"] = False
            return parsed, True, info
        except Exception as exc:
            last_error = str(exc)
    info.update({"used_repair": True, "repair_method": "failed", "parse_error": last_error})
    return _fill_visual_defaults({}), False, info


def _validate_l1_reasoner_output_base(parsed: dict, segment: dict, visual_discovery: dict) -> dict:
    warnings: list[str] = []
    repairs = list(parsed.get("_parser_repairs") or [])
    ct0, ct1 = float(segment["context_t_start"]), float(segment["context_t_end"])
    events = parsed.get("l1_events") or []
    event_ids = {str(e.get("event_id")) for e in events}
    visual_ids = {str(c.get("candidate_id")) for c in visual_discovery.get("visual_micro_action_candidates", [])}
    ledger = parsed.get("visual_candidate_ledger") or []
    ledger_ids = {str(x.get("candidate_id")) for x in ledger}
    unaccounted = sorted(visual_ids - ledger_ids)
    for cid in unaccounted:
        warnings.append(f"visual candidate {cid} missing from ledger")
    for entry in ledger:
        if entry.get("disposition") in {"accepted_as_l1", "merged_into_l1"} and not (set(entry.get("linked_l1_event_ids") or []) & event_ids):
            warnings.append(f"ledger candidate {entry.get('candidate_id')} has no linked final L1")
    for idx, event in enumerate(events):
        if event.get("should_create_l1") is not True:
            warnings.append(f"l1_events[{idx}] should_create_l1 is not true")
        if event.get("evidence_type") not in {"observed", "inferred"}:
            warnings.append(f"l1_events[{idx}] invalid evidence_type")
        if event.get("event_subtype") not in EVENT_SUBTYPES:
            warnings.append(f"l1_events[{idx}] invalid event_subtype")
        t0, t1 = float(event.get("t_start_est", 0.0)), float(event.get("t_end_est", 0.0))
        if not (ct0 <= t0 <= ct1 and ct0 <= t1 <= ct1 and t1 > t0):
            warnings.append(f"l1_events[{idx}] time outside context")
        rationale = event.get("confidence_rationale") or {}
        conf = float(event.get("confidence", 0.0))
        if rationale.get("visual_clarity") in {"partially_occluded", "subtle", "weak"} and conf >= 1.0:
            warnings.append(f"l1_events[{idx}] confidence 1.0 despite non-clear visual evidence")
        if rationale.get("object_identity") == "ambiguous" and conf > 0.85:
            warnings.append(f"l1_events[{idx}] confidence too high for ambiguous object identity")
        if event.get("event_subtype") in {"support_contact", "state_or_continuation"} and conf > 0.9:
            warnings.append(f"l1_events[{idx}] support/state confidence >0.9")
        if event.get("specificity_level") == "coarse_but_supported" and event.get("event_subtype") == "core_effect":
            warnings.append(f"l1_events[{idx}] broad/coarse core effect")
        if _part_whole_reversal_hint(event.get("direct_object") or {}):
            warnings.append(f"l1_events[{idx}] possible part_of reversal")
    audit = parsed.get("w_action_audit") or []
    w_windows = {int(x.get("window_idx")) for x in segment.get("w_action_summaries", []) if (x.get("actions") or x.get("action_present"))}
    audit_windows = {int(x.get("window_idx")) for x in audit if x.get("window_idx") is not None}
    for idx in sorted(w_windows - audit_windows):
        warnings.append(f"W_action window {idx} missing from audit")
    return {
        "valid": not warnings,
        "warnings": warnings,
        "parser_repairs": repairs,
        "n_l1_events": len(events),
        "n_observed": sum(1 for e in events if e.get("evidence_type") == "observed"),
        "n_inferred": sum(1 for e in events if e.get("evidence_type") == "inferred"),
        "n_hypothesized": len(parsed.get("rejected_or_hypothesized_actions") or []),
        "n_high_hallucination_risk": sum(1 for e in events if e.get("hallucination_risk") == "high"),
        "n_core_effect": _subtype(events, "core_effect"),
        "n_support_contact": _subtype(events, "support_contact"),
        "n_state_or_continuation": _subtype(events, "state_or_continuation"),
        "n_ambiguous_part_manipulation": _subtype(events, "ambiguous_part_manipulation"),
        "n_missing_from_W": _subtype(events, "missing_from_W"),
        "n_visual_candidates": len(visual_ids),
        "n_visual_accepted": _disp(ledger, "accepted_as_l1"),
        "n_visual_merged": _disp(ledger, "merged_into_l1"),
        "n_visual_support": _disp(ledger, "support_for_l1"),
        "n_visual_rejected": sum(1 for x in ledger if str(x.get("disposition", "")).startswith("rejected")),
        "n_visual_unaccounted": len(unaccounted),
        "n_w_confirmed": _audit(audit, "confirmed"),
        "n_w_corrected": _audit(audit, "corrected"),
        "n_w_downgraded_to_support": _audit(audit, "downgraded_to_support"),
        "n_w_too_vague": _audit(audit, "too_vague"),
        "n_w_too_specific": _audit(audit, "too_specific"),
        "n_w_partially_supported": _audit(audit, "partially_supported"),
        "n_w_likely_wrong": _audit(audit, "likely_wrong"),
    }


def _fix_ledger(item: dict) -> dict:
    return {
        "candidate_id": str(item.get("candidate_id") or ""),
        "candidate_description": str(item.get("candidate_description") or ""),
        "disposition": _choice(item.get("disposition"), LEDGER_DISPOSITIONS, "rejected_due_to_weak_evidence"),
        "linked_l1_event_ids": item.get("linked_l1_event_ids") if isinstance(item.get("linked_l1_event_ids"), list) else [],
        "reason": str(item.get("reason") or ""),
    }


def _fix_audit(item: dict) -> dict:
    return {
        "window_idx": _int_or_none(item.get("window_idx")),
        "w_action": str(item.get("w_action") or ""),
        "decision": _choice(item.get("decision"), AUDIT_DECISIONS, "not_applicable"),
        "preferred_l1_interpretation": str(item.get("preferred_l1_interpretation") or ""),
        "linked_l1_event_ids": item.get("linked_l1_event_ids") if isinstance(item.get("linked_l1_event_ids"), list) else [],
        "reason": str(item.get("reason") or ""),
        "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _subtype(events: list[dict], subtype: str) -> int:
    return sum(1 for e in events if e.get("event_subtype") == subtype)


def _disp(ledger: list[dict], disposition: str) -> int:
    return sum(1 for x in ledger if x.get("disposition") == disposition)


def _audit(audit: list[dict], decision: str) -> int:
    return sum(1 for x in audit if x.get("decision") == decision)


# Additional validation rules
CAUTIOUS_AREA_TERMS = ("area", "region", "part", "top", "control")
SPECIFIC_PART_NAMES = {"button", "knob", "switch", "handle", "lid", "cap", "door"}


def parse_l1_reasoner_output(raw_output: str) -> tuple[dict, bool, dict]:
    return _parse_l1_reasoner_output_base(raw_output)


def parse_visual_discovery_output(raw_output: str) -> tuple[dict, bool, dict]:
    return _parse_visual_discovery_output_base(raw_output)


def validate_l1_reasoner_output(parsed: dict, segment: dict, visual_discovery: dict) -> dict:
    result = _validate_l1_reasoner_output_base(parsed, segment, visual_discovery)
    warnings = list(result.get("warnings") or [])
    long_core = 0
    schema = 0
    events = parsed.get("l1_events") or []

    for idx, event in enumerate(events):
        t0 = _float_default(event.get("t_start_est"), float(segment.get("t_start", 0.0)))
        t1 = _float_default(event.get("t_end_est"), float(segment.get("t_end", 0.0)))
        duration = max(0.0, t1 - t0)
        subtype = event.get("event_subtype")
        specificity = event.get("specificity_level")
        if duration > 3.0 and subtype == "core_effect":
            long_core += 1
            warnings.append(f"l1_events[{idx}] long core_effect duration {duration:.2f}s")
        if duration > 3.0 and specificity == "specific_micro_action":
            warnings.append(f"l1_events[{idx}] long specific_micro_action duration {duration:.2f}s")
        if _description_object_mismatch(event):
            schema += 1
            warnings.append(f"l1_events[{idx}] cautious description conflicts with specific direct_object")
        if subtype == "ambiguous_part_manipulation":
            notes = str(event.get("uncertainty_notes") or "").strip()
            if event.get("evidence_type") == "observed" and not notes:
                warnings.append(f"l1_events[{idx}] ambiguous_part_manipulation lacks uncertainty notes")
        rationale = event.get("confidence_rationale") or {}
        conf = _float_default(event.get("confidence"), 0.0)
        if rationale.get("w_action_support") in {"conflicting", "none", "partial"} and rationale.get("visual_clarity") != "clear" and conf > 0.85:
            warnings.append(f"l1_events[{idx}] confidence too high for weak W support without clear visual evidence")

    ledger = parsed.get("visual_candidate_ledger") or []
    event_ids = {str(e.get("event_id")) for e in events}
    for entry in ledger:
        if entry.get("disposition") == "support_for_l1" and event_ids and not (set(entry.get("linked_l1_event_ids") or []) & event_ids):
            warnings.append(f"ledger candidate {entry.get('candidate_id')} support_for_l1 has no linked final L1")

    result["warnings"] = warnings
    result["valid"] = not warnings
    result["n_long_core_effect_warnings"] = long_core
    result["n_schema_consistency_warnings"] = schema
    return result


def _description_object_mismatch(event: dict) -> bool:
    desc = str(event.get("micro_action_description") or "").lower()
    name = str((event.get("direct_object") or {}).get("name") or "").lower().strip()
    if not desc or not name:
        return False
    cautious = any(term in desc for term in CAUTIOUS_AREA_TERMS)
    return cautious and name in SPECIFIC_PART_NAMES and name not in desc


def _float_default(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
