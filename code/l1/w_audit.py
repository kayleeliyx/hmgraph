"""Normalize W audit decisions for L1 postprocess v3.2.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.w_audit import normalize_w_audit"
"""

from __future__ import annotations


def normalize_w_audit(w_audit_entries: list[dict], final_candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    normalized, log = [], []
    for entry in w_audit_entries:
        item = dict(entry)
        old = item.get("decision", "not_applicable")
        linked = _resolve_links(item, final_candidates)
        item["linked_l1_event_ids"] = linked
        w_text = str(item.get("w_action") or "").lower().strip()
        reason = str(item.get("reason") or "").lower()
        pref = str(item.get("preferred_l1_interpretation") or "").lower()
        if not w_text or w_text == "none" or not linked:
            new = "not_applicable"
        else:
            cands = [c for c in final_candidates if c.get("l1_id") in linked]
            if any(_contact_text(w_text) and c.get("event_subtype") in {"core_effect", "ambiguous_part_manipulation"} for c in cands):
                new = "downgraded_to_support"
            elif any(_generic_text(w_text) for _ in cands) and pref:
                new = "too_vague"
            elif any(_agrees(w_text, c) for c in cands):
                new = "confirmed"
            elif "clearly" in reason or "matches" in reason or "accurately" in reason:
                new = "confirmed"
            elif pref:
                new = "corrected"
            elif "plausible" in reason or "partly" in reason or "partially" in reason:
                new = "partially_supported"
            else:
                new = "partially_supported"
        item["decision"] = new
        normalized.append(item)
        if new != old or linked != entry.get("linked_l1_event_ids", []):
            log.append({"type": "w_audit_normalization", "window_idx": item.get("window_idx"), "old_decision": old, "new_decision": new, "old_links": entry.get("linked_l1_event_ids", []), "new_links": linked, "reason": "normalized matched W action decision"})
    return normalized, log


def _resolve_links(entry: dict, candidates: list[dict]) -> list[str]:
    links = [str(x) for x in entry.get("linked_l1_event_ids") or []]
    out = []
    raw_to_final = {}
    for cand in candidates:
        cid = cand.get("l1_id")
        if cid:
            raw_to_final[str(cid)] = cid
        for mid in cand.get("merged_from_l1_ids") or []:
            raw_to_final[str(mid)] = cid
        raw_id = ((cand.get("raw_reasoner_event") or {}).get("event_id"))
        if raw_id:
            raw_to_final[str(raw_id)] = cid
    for link in links:
        if link in raw_to_final and raw_to_final[link] not in out:
            out.append(raw_to_final[link])
    if out:
        return out
    pref = str(entry.get("preferred_l1_interpretation") or entry.get("w_action") or "").lower()
    for cand in candidates:
        hay = " ".join(str(x) for x in [cand.get("verb"), (cand.get("direct_object") or {}).get("name"), cand.get("micro_action_description")]).lower()
        if pref and all(tok in hay for tok in _content_tokens(pref)[:2]):
            out.append(cand["l1_id"])
    return out


def _agrees(text: str, cand: dict) -> bool:
    obj = str((cand.get("direct_object") or {}).get("name") or "").lower()
    verb = str(cand.get("verb") or "").lower()
    return (verb and verb in text) or (obj and obj in text)


def _generic_text(text: str) -> bool:
    return any(x in text for x in ("object", "container", "appliance", "thing", "item"))


def _contact_text(text: str) -> bool:
    return any(x in text for x in ("reach", "grasp", "hold", "touch", "contact"))


def _content_tokens(text: str) -> list[str]:
    stop = {"the", "a", "an", "to", "for", "and", "or", "of", "in", "on"}
    return [x for x in text.replace(",", " ").split() if x not in stop]
