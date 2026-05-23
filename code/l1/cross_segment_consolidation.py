"""Conservative cross-segment consolidation for L1 multimodal reasoner v3.2.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.cross_segment_consolidation import consolidate_l1_candidates_across_segments"
"""

from __future__ import annotations

import copy


def consolidate_l1_candidates_across_segments(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge only near-duplicates; otherwise preserve relations without overmerging."""
    ordered = sorted((copy.deepcopy(c) for c in candidates), key=lambda r: (float(r.get("t_start", 0.0)), float(r.get("t_end", 0.0)), r.get("l1_id", "")))
    final: list[dict] = []
    log: list[dict] = []
    for cand in ordered:
        _init(cand)
        duplicate_idx = _find_duplicate(final, cand)
        if duplicate_idx is not None:
            target = final[duplicate_idx]
            if _duration_union(target, cand) <= _max_merge_duration(target, cand):
                _merge_duplicate(target, cand)
                log.append({"decision": "merge", "source_l1_id": cand["l1_id"], "target_l1_id": target["l1_id"], "relation": "duplicate_of", "reason": "substantial temporal overlap with same object/action phase", "duration_constraint_checked": True})
                continue
        relation_idx, relation, reason = _find_relation(final, cand)
        if relation_idx is not None:
            target = final[relation_idx]
            target["related_l1_ids"].append({"l1_id": cand["l1_id"], "relation": relation, "reason": reason})
            if relation == "supports":
                target["supporting_l1_ids"].append(cand["l1_id"])
            cand["related_l1_ids"].append({"l1_id": target["l1_id"], "relation": _inverse_relation(relation), "reason": reason})
            log.append({"decision": "relation", "source_l1_id": cand["l1_id"], "target_l1_id": target["l1_id"], "relation": relation, "reason": reason, "duration_constraint_checked": True})
        final.append(cand)
    return final, log


def _init(cand: dict) -> None:
    cand.setdefault("merged_from_l1_ids", [cand.get("l1_id")])
    cand.setdefault("supporting_l1_ids", [])
    cand.setdefault("related_l1_ids", [])
    cand.setdefault("consolidation_notes", [])


def _find_duplicate(final: list[dict], cand: dict) -> int | None:
    for idx, item in enumerate(final):
        if not _same_object(item, cand):
            continue
        if not _same_action(item, cand):
            continue
        if not _substantial_overlap(item, cand):
            continue
        if _different_phase(item, cand):
            continue
        if _would_make_long_core(item, cand):
            continue
        return idx
    return None


def _find_relation(final: list[dict], cand: dict) -> tuple[int | None, str, str]:
    best: tuple[int | None, str, str] = (None, "", "")
    for idx, item in enumerate(final):
        if not (_same_object(item, cand) or _nearby(item, cand)):
            continue
        if _is_support(cand) and not _is_support(item):
            return idx, "supports", "support/contact event kept separate and linked to nearby core event"
        if _is_support(item) and not _is_support(cand):
            return idx, "overlaps_context", "nearby core event overlaps prior support/contact context"
        if _same_object(item, cand):
            return idx, "same_object_sequence", "same object but not merged because phases or timing differ"
        best = (idx, "follows", "temporally adjacent context")
    return best


def _merge_duplicate(target: dict, cand: dict) -> None:
    target["t_start"] = min(float(target["t_start"]), float(cand["t_start"]))
    target["t_end"] = max(float(target["t_end"]), float(cand["t_end"]))
    for mid in cand.get("merged_from_l1_ids", [cand["l1_id"]]):
        if mid not in target["merged_from_l1_ids"]:
            target["merged_from_l1_ids"].append(mid)
    target["confidence"] = max(float(target.get("confidence", 0.0)), float(cand.get("confidence", 0.0)))
    target["consolidation_notes"].append(f"merged near-duplicate {cand['l1_id']}")


def _same_object(a: dict, b: dict) -> bool:
    ao, bo = a.get("direct_object") or {}, b.get("direct_object") or {}
    an, bn = _norm(ao.get("name")), _norm(bo.get("name"))
    ap, bp = _norm(ao.get("part_of", "none")), _norm(bo.get("part_of", "none"))
    return an == bn and ap == bp and an not in {"", "unknown"}


def _same_action(a: dict, b: dict) -> bool:
    av, bv = _norm(a.get("verb")), _norm(b.get("verb"))
    if av == bv:
        return True
    ad, bd = _norm(a.get("micro_action_description")), _norm(b.get("micro_action_description"))
    return av and bv and av in bd and bv in ad


def _substantial_overlap(a: dict, b: dict) -> bool:
    overlap = min(float(a["t_end"]), float(b["t_end"])) - max(float(a["t_start"]), float(b["t_start"]))
    if overlap <= 0:
        return False
    shorter = min(_duration(a), _duration(b))
    return shorter > 0 and overlap / shorter >= 0.55


def _nearby(a: dict, b: dict) -> bool:
    return float(b.get("t_start", 0.0)) <= float(a.get("t_end", 0.0)) + 0.75


def _different_phase(a: dict, b: dict) -> bool:
    if a.get("event_subtype") != b.get("event_subtype"):
        return True
    if a.get("specificity_level") != b.get("specificity_level") and "ambiguous" in {a.get("specificity_level"), b.get("specificity_level")}:
        return True
    return False


def _would_make_long_core(a: dict, b: dict) -> bool:
    return _is_core(a) and _is_core(b) and _duration_union(a, b) > 3.0


def _duration_union(a: dict, b: dict) -> float:
    return max(float(a["t_end"]), float(b["t_end"])) - min(float(a["t_start"]), float(b["t_start"]))


def _max_merge_duration(a: dict, b: dict) -> float:
    return 3.0 if _is_core(a) or _is_core(b) else 5.0


def _duration(x: dict) -> float:
    return max(0.0, float(x.get("t_end", 0.0)) - float(x.get("t_start", 0.0)))


def _is_core(x: dict) -> bool:
    return x.get("event_subtype") == "core_effect"


def _is_support(x: dict) -> bool:
    return x.get("event_subtype") in {"support_contact", "state_or_continuation"}


def _inverse_relation(relation: str) -> str:
    return {"supports": "overlaps_context", "same_object_sequence": "same_object_sequence", "follows": "overlaps_context", "overlaps_context": "overlaps_context"}.get(relation, "overlaps_context")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().strip().split())
