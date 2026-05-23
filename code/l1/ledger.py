"""Normalize visual candidate ledger dispositions for L1 postprocess v3.2.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.ledger import normalize_visual_candidate_ledger"
"""

from __future__ import annotations


def normalize_visual_candidate_ledger(ledger_entries: list[dict], final_candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    normalized, log = [], []
    raw_to_final = _raw_to_final(final_candidates)
    linked_counts: dict[str, int] = {}
    resolved_links = []
    for entry in ledger_entries:
        links = []
        for link in entry.get("linked_l1_event_ids") or []:
            resolved = raw_to_final.get(str(link), str(link))
            if resolved and resolved not in links:
                links.append(resolved)
        if not links:
            links = _match_by_text(entry, final_candidates)
        resolved_links.append(links)
        for link in links:
            linked_counts[link] = linked_counts.get(link, 0) + 1

    for entry, links in zip(ledger_entries, resolved_links):
        item = dict(entry)
        old = item.get("disposition", "")
        desc = str(item.get("candidate_description") or "").lower()
        if not links:
            new = "rejected_due_to_weak_evidence" if old != "rejected_as_hypothesis" else old
        elif _support_text(desc) or old == "support_for_l1":
            new = "support_for_l1"
        elif len(links) == 1 and linked_counts.get(links[0], 0) == 1:
            new = "accepted_as_l1"
        elif len(links) == 1 and linked_counts.get(links[0], 0) > 1:
            new = "merged_into_l1"
        else:
            new = "merged_into_l1"
        item["linked_l1_event_ids"] = links
        item["disposition"] = new
        normalized.append(item)
        if new != old or links != entry.get("linked_l1_event_ids", []):
            log.append({"type": "visual_ledger_normalization", "candidate_id": item.get("candidate_id"), "old_disposition": old, "new_disposition": new, "old_links": entry.get("linked_l1_event_ids", []), "new_links": links, "reason": "normalized visual candidate disposition"})
    return normalized, log


def _raw_to_final(candidates: list[dict]) -> dict[str, str]:
    out = {}
    for cand in candidates:
        cid = cand.get("l1_id")
        if not cid:
            continue
        out[str(cid)] = cid
        for mid in cand.get("merged_from_l1_ids") or []:
            out[str(mid)] = cid
        raw_id = ((cand.get("raw_reasoner_event") or {}).get("event_id"))
        if raw_id:
            out[str(raw_id)] = cid
    return out


def _match_by_text(entry: dict, candidates: list[dict]) -> list[str]:
    text = str(entry.get("candidate_description") or "").lower()
    segment_id = entry.get("segment_id")
    scoped = [c for c in candidates if not segment_id or c.get("segment_id") == segment_id]
    pools = [scoped, candidates] if scoped != candidates else [candidates]
    for pool in pools:
        out = []
        for cand in pool:
            obj = str((cand.get("direct_object") or {}).get("name") or "").lower()
            verb = str(cand.get("verb") or "").lower()
            if obj and obj in text and (not verb or verb in text or verb.split()[0] in text):
                out.append(cand["l1_id"])
        if out:
            return out[:1]
    return []


def _support_text(text: str) -> bool:
    return any(x in text for x in ("hold", "holding", "support", "context", "near", "maintain"))
