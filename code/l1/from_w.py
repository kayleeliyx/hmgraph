"""Deterministic per-hand L1 construction from reprocessed W records."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import statistics

from code.utils.alias_table import canonicalize


GROUPING_VERSION = "per_hand_exact_signature_v2"
PRIMARY_ROLES = {"primary_manipulation", "preparatory"}
SUPPORT_ROLES = {"support", "observation"}
VALID_HANDS = {"left", "right", "both", "unknown"}


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _object_name(action: dict) -> str:
    rich = action.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name"):
        return _norm(rich["name"])
    direct = action.get("direct_object")
    if isinstance(direct, dict):
        direct = direct.get("name")
    return _norm(direct)


def _hand(action: dict) -> str:
    hand = _norm(action.get("interaction", {}).get("hand") or action.get("hand"))
    return hand if hand in VALID_HANDS else "unknown"


def _role(action: dict) -> str:
    return _norm(action.get("interaction", {}).get("role"))


def _canonical_action(action: dict, aliases: dict) -> tuple[str, str]:
    verb = canonicalize(action.get("verb"), aliases.get("verb_aliases", {}))
    obj = canonicalize(_object_name(action), aliases.get("noun_aliases", {}))
    return verb, obj


def _source_ref(window_idx: int, action: dict) -> dict:
    return {"window_idx": int(window_idx), "action_idx": int(action["action_idx"])}


def _group_candidates(records: list[dict], aliases: dict) -> list[dict]:
    streams: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda row: (float(row["t_start"]), int(row["window_idx"]))):
        for action in record.get("valid_actions", []):
            if _role(action) not in PRIMARY_ROLES:
                continue
            verb, obj = _canonical_action(action, aliases)
            hand = _hand(action)
            streams[(hand, verb, obj)].append(
                {
                    "window_idx": int(record["window_idx"]),
                    "t_start": float(action.get("event_t_start", record["t_start"])),
                    "t_end": float(action.get("event_t_end", record["t_end"])),
                    "confidence": float(action.get("confidence", 0.0)),
                    "role": _role(action),
                    "timing_status": action.get("timing_status", "unknown"),
                    "ref": _source_ref(record["window_idx"], action),
                }
            )

    groups: list[dict] = []
    for (hand, verb, obj), items in streams.items():
        items.sort(key=lambda item: (item["window_idx"], item["t_start"], item["ref"]["action_idx"]))
        current: dict | None = None
        for item in items:
            if current is None or item["window_idx"] > current["last_window_idx"] + 1:
                if current is not None:
                    groups.append(current)
                current = {
                    "hand": hand,
                    "verb": verb,
                    "object": obj,
                    "t_start": item["t_start"],
                    "t_end": item["t_end"],
                    "last_window_idx": item["window_idx"],
                    "window_idxs": [item["window_idx"]],
                    "refs": [item["ref"]],
                    "confidences": [item["confidence"]],
                    "roles": [item["role"]],
                    "timing_statuses": [item["timing_status"]],
                }
                continue
            current["t_start"] = min(current["t_start"], item["t_start"])
            current["t_end"] = max(current["t_end"], item["t_end"])
            current["last_window_idx"] = max(current["last_window_idx"], item["window_idx"])
            if item["window_idx"] not in current["window_idxs"]:
                current["window_idxs"].append(item["window_idx"])
            current["refs"].append(item["ref"])
            current["confidences"].append(item["confidence"])
            current["roles"].append(item["role"])
            current["timing_statuses"].append(item["timing_status"])
        if current is not None:
            groups.append(current)
    return groups


def _group_to_node(group: dict, index: int) -> dict:
    duration = max(0.0, float(group["t_end"]) - float(group["t_start"]))
    confidence = round(statistics.mean(group["confidences"]), 3)
    preparatory_only = set(group["roles"]) == {"preparatory"}
    warnings = []
    if any(status in {"clipped_to_target", "invalid_hint_fallback"} for status in group["timing_statuses"]):
        warnings.append("contains_repaired_or_clipped_time_hint")
    return {
        "l1_id": f"wg2_{index:04d}",
        "segment_id": f"wg2_seg_{index:04d}",
        "t_start": round(float(group["t_start"]), 6),
        "t_end": round(float(group["t_end"]), 6),
        "hand": group["hand"],
        "verb": group["verb"],
        "direct_object": {"name": group["object"]},
        "micro_action_description": (
            f"From reprocessed W grouping ({group['hand']} hand): the person "
            f"{group['verb']} the {group['object']} over {duration:.2f}s "
            f"({len(group['window_idxs'])} W windows)."
        ),
        "event_subtype": "preparatory" if preparatory_only else "core_effect",
        "evidence_type": "observed",
        "specificity_level": "from_reprocessed_w",
        "confidence": confidence,
        "confidence_raw_model": confidence,
        "hallucination_risk": "n/a",
        "qa_importance": "unknown",
        "source": "w_grouping_v2",
        "grouping_version": GROUPING_VERSION,
        "merged_from_l1_ids": [],
        "supporting_l1_ids": [],
        "related_l1_ids": [],
        "warnings": warnings,
        "source_window_idxs": sorted(group["window_idxs"]),
        "source_action_refs": group["refs"],
        "source_roles": sorted(set(group["roles"])),
    }


def _support_observations(records: list[dict]) -> list[dict]:
    observations = []
    for record in records:
        for action in record.get("valid_actions", []):
            role = _role(action)
            if role not in SUPPORT_ROLES:
                continue
            observations.append(
                {
                    "support_id": f"ws_{len(observations) + 1:05d}",
                    "window_idx": int(record["window_idx"]),
                    "action_idx": int(action["action_idx"]),
                    "t_start": float(action.get("event_t_start", record["t_start"])),
                    "t_end": float(action.get("event_t_end", record["t_end"])),
                    "hand": _hand(action),
                    "verb": action.get("verb", "unknown"),
                    "direct_object": {"name": _object_name(action)},
                    "role": role,
                    "confidence": float(action.get("confidence", 0.0)),
                    "source_action": deepcopy(action),
                }
            )
    return observations


def build_l1_from_w(records: list[dict], aliases: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Return L1 nodes, support observations, and a complete W disposition ledger."""
    aliases = aliases or {}
    groups = _group_candidates(records, aliases)
    groups.sort(key=lambda group: (group["t_start"], group["hand"], group["verb"], group["object"]))
    nodes = [_group_to_node(group, index) for index, group in enumerate(groups, 1)]
    supports = _support_observations(records)

    owners = {
        (ref["window_idx"], ref["action_idx"]): node["l1_id"]
        for node in nodes
        for ref in node["source_action_refs"]
    }
    support_owners = {
        (row["window_idx"], row["action_idx"]): row["support_id"] for row in supports
    }
    ledger = []
    for record in sorted(records, key=lambda row: int(row["window_idx"])):
        window_idx = int(record["window_idx"])
        for action in record.get("valid_actions", []):
            key = (window_idx, int(action["action_idx"]))
            if key in owners:
                disposition, owner = "l1_owner", owners[key]
            elif key in support_owners:
                disposition, owner = "support_observation", support_owners[key]
            else:
                raise AssertionError(f"Accepted W action has no disposition: {key}")
            ledger.append(
                {
                    "window_idx": key[0],
                    "action_idx": key[1],
                    "accepted": True,
                    "role": _role(action),
                    "verb": action.get("verb"),
                    "direct_object": _object_name(action),
                    "disposition": disposition,
                    "owner_id": owner,
                }
            )
        for rejection_idx, rejection in enumerate(record.get("rejected_actions", [])):
            action = rejection.get("action") if isinstance(rejection, dict) else None
            ledger.append(
                {
                    "window_idx": window_idx,
                    "action_idx": action.get("action_idx") if isinstance(action, dict) else None,
                    "rejection_idx": rejection_idx,
                    "accepted": False,
                    "role": _role(action) if isinstance(action, dict) else "unknown",
                    "verb": action.get("verb") if isinstance(action, dict) else None,
                    "direct_object": _object_name(action) if isinstance(action, dict) else None,
                    "disposition": "rejected",
                    "owner_id": None,
                    "rejection_reason": rejection.get("rejection_reason") if isinstance(rejection, dict) else str(rejection),
                    "rejection_stage": rejection.get("rejection_stage") if isinstance(rejection, dict) else "unknown",
                }
            )
    return nodes, supports, ledger
