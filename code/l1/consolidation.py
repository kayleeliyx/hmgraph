"""Conservative L1 consolidation for postprocess v3.2.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.consolidation import consolidate_l1_candidates"
"""

from __future__ import annotations

import copy


def consolidate_l1_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted((copy.deepcopy(c) for c in candidates), key=lambda x: (float(x.get("t_start", 0.0)), float(x.get("t_end", 0.0))))
    final: list[dict] = []
    log: list[dict] = []
    for cand in ordered:
        _init(cand)
        idx = _find_merge_target(final, cand)
        if idx is not None:
            target = final[idx]
            old = dict(target)
            _merge(target, cand)
            log.append({"decision": "merge", "source_l1_id": cand["l1_id"], "target_l1_id": target["l1_id"], "reason": "same adjacent core event with same object/action and similar description", "before": _brief(old), "after": _brief(target), "duration_constraint_checked": True})
            continue
        ridx, relation, reason = _find_relation(final, cand)
        if ridx is not None:
            final[ridx].setdefault("related_l1_ids", []).append({"l1_id": cand["l1_id"], "relation": relation, "reason": reason})
            cand.setdefault("related_l1_ids", []).append({"l1_id": final[ridx]["l1_id"], "relation": _inverse(relation), "reason": reason})
            log.append({"decision": "relation", "source_l1_id": cand["l1_id"], "target_l1_id": final[ridx]["l1_id"], "relation": relation, "reason": reason, "duration_constraint_checked": True})
        final.append(cand)
    return final, log


def _find_merge_target(final: list[dict], cand: dict) -> int | None:
    for idx, item in enumerate(final):
        if not (_same_object(item, cand) and _same_verb(item, cand)):
            continue
        if item.get("event_subtype") != cand.get("event_subtype"):
            continue
        if item.get("event_subtype") == "ambiguous_part_manipulation" or cand.get("event_subtype") == "ambiguous_part_manipulation":
            continue
        if not _adjacent_or_overlap(item, cand):
            continue
        if _description_similarity(item, cand) < 0.45:
            continue
        if _union_duration(item, cand) > _max_duration(item, cand):
            continue
        return idx
    return None


def _find_relation(final: list[dict], cand: dict) -> tuple[int | None, str, str]:
    for idx, item in enumerate(final):
        if not _near(item, cand):
            continue
        if _same_object(item, cand):
            return idx, "same_object_sequence", "same object but kept separate to avoid overmerge"
        if cand.get("event_subtype") in {"support_contact", "state_or_continuation"}:
            return idx, "supports", "support/state event linked without merging"
        return idx, "follows", "temporally adjacent event"
    return None, "", ""


def _merge(target: dict, cand: dict) -> None:
    target["t_start"] = min(float(target["t_start"]), float(cand["t_start"]))
    target["t_end"] = max(float(target["t_end"]), float(cand["t_end"]))
    for mid in cand.get("merged_from_l1_ids", [cand["l1_id"]]):
        if mid not in target["merged_from_l1_ids"]:
            target["merged_from_l1_ids"].append(mid)
    if cand["l1_id"] not in target["merged_from_l1_ids"]:
        target["merged_from_l1_ids"].append(cand["l1_id"])
    target.setdefault("consolidation_notes", []).append(f"merged duplicate adjacent event {cand['l1_id']}")
    if len(str(cand.get("micro_action_description", ""))) > len(str(target.get("micro_action_description", ""))):
        target["micro_action_description"] = cand["micro_action_description"]
    target["confidence"] = max(float(target.get("confidence", 0.0)), float(cand.get("confidence", 0.0)))
    target.setdefault("related_l1_ids", [])
    for rel in cand.get("related_l1_ids") or []:
        if rel not in target["related_l1_ids"]:
            target["related_l1_ids"].append(rel)


def _init(cand: dict) -> None:
    cand.setdefault("merged_from_l1_ids", [cand.get("l1_id")])
    cand.setdefault("supporting_l1_ids", [])
    cand.setdefault("related_l1_ids", [])
    cand.setdefault("consolidation_notes", [])


def _same_object(a: dict, b: dict) -> bool:
    ao, bo = a.get("direct_object") or {}, b.get("direct_object") or {}
    return _norm(ao.get("name")) == _norm(bo.get("name")) and _norm(ao.get("part_of", "none")) == _norm(bo.get("part_of", "none"))


def _same_verb(a: dict, b: dict) -> bool:
    return _norm(a.get("verb")) == _norm(b.get("verb"))


def _adjacent_or_overlap(a: dict, b: dict) -> bool:
    return float(b.get("t_start", 0.0)) <= float(a.get("t_end", 0.0)) + 0.15


def _near(a: dict, b: dict) -> bool:
    return float(b.get("t_start", 0.0)) <= float(a.get("t_end", 0.0)) + 0.75


def _description_similarity(a: dict, b: dict) -> float:
    aw, bw = set(_words(a.get("micro_action_description", ""))), set(_words(b.get("micro_action_description", "")))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _words(text) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "to", "from", "in", "on", "of", "with", "person"}
    return [w.strip(".,").lower() for w in str(text or "").split() if w.strip(".,").lower() not in stop]


def _union_duration(a: dict, b: dict) -> float:
    return max(float(a["t_end"]), float(b["t_end"])) - min(float(a["t_start"]), float(b["t_start"]))


def _max_duration(a: dict, b: dict) -> float:
    if a.get("event_subtype") == "core_effect" and b.get("event_subtype") == "core_effect":
        return 3.5
    return 5.0


def _brief(x: dict) -> dict:
    return {"l1_id": x.get("l1_id"), "t_start": x.get("t_start"), "t_end": x.get("t_end"), "verb": x.get("verb"), "object": (x.get("direct_object") or {}).get("name")}


def _inverse(relation: str) -> str:
    return {"supports": "overlaps_context", "same_object_sequence": "same_object_sequence", "follows": "overlaps_context"}.get(relation, "overlaps_context")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().strip().split())
