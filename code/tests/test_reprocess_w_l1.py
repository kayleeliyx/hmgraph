from __future__ import annotations

import json

from code.l1.from_w import build_l1_from_w
from code.w.records import reprocess_record


def _action(*, verb="grate", obj="cheese", hand="right", role="primary_manipulation", hint="0.1-0.9"):
    return {
        "verb": verb,
        "direct_object": {"name": obj, "category": "ingredient"},
        "interaction": {
            "hand": hand,
            "contact_type": "hold",
            "role": role,
            "object_change": "moved",
            "why_this_verb": "right hand holds tool and the object moves in TARGET frames",
        },
        "time_hint": hint,
        "confidence": 0.9,
    }


def _record(index, actions):
    return {
        "window_idx": index,
        "t_start": float(index),
        "t_end": float(index + 1),
        "raw_output": json.dumps({"actions": actions}),
        "valid_actions": [],
    }


def test_reprocess_rejects_time_hint_outside_target():
    row = reprocess_record(_record(10, [_action(hint="1.2-1.8")]))
    assert row["valid_actions"] == []
    assert row["rejected_actions"][0]["rejection_stage"] == "target_timing_gate"


def test_l1_groups_consecutive_per_hand_and_owns_every_primary_action():
    rows = [
        reprocess_record(_record(10, [_action()])),
        reprocess_record(_record(11, [_action()])),
        reprocess_record(_record(12, [_action(role="support", hand="left", obj="grater")])),
    ]
    nodes, supports, ledger = build_l1_from_w(rows, {})
    assert len(nodes) == 1
    assert nodes[0]["source_window_idxs"] == [10, 11]
    assert len(nodes[0]["source_action_refs"]) == 2
    assert len(supports) == 1
    assert sum(item["disposition"] == "l1_owner" for item in ledger) == 2
    assert sum(item["disposition"] == "support_observation" for item in ledger) == 1
