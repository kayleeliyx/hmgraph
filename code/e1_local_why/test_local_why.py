from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.e1_local_why.inference import build_plan
from code.e1_local_why.model import parse_decision
from code.e1_local_why.run import load_config, prepare


def test_strict_decision_contract() -> None:
    assert parse_decision('{"answer":"D","request":null}', True) == {
        "answer": "D",
        "request": None,
    }
    assert parse_decision(
        '{"answer":"A","request":{"start_sec":1,"end_sec":2.5}}', True
    ) == {"answer": "A", "request": {"start_sec": 1.0, "end_sec": 2.5}}
    with pytest.raises(ValueError):
        parse_decision('{"answer":"A","request":{"start_sec":1,"end_sec":2}}', False)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_decision('Answer: A', True)


def test_prepare_hides_e1_boundaries_and_gold() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "e1_local_why.yaml")
    queries, split, _ = prepare(config)
    assert len(queries) == 100
    assert len(split["scoring_labels"]) == 100
    forbidden = {"start_sec", "requested_start", "requested_end", "gold", "narration"}
    assert all(not (set(query) & forbidden) for query in queries)
    assert [query["question_id"] for query in queries] == [
        record["question_id"] for record in split["records"]
    ]
    assert all(query["option_order"] == record["option_order"] for query, record in zip(queries, split["records"]))


def test_plan_identity_uses_individual_ordered_frames() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "e1_local_why.yaml")
    queries, _, _ = prepare(config)
    frame = {"timestamp_sec": 1.0, "sha256": "a", "path": "/tmp/frame.jpg"}
    batch = {
        "start_sec": 0.0,
        "end_sec": 1.0,
        "ordered_frame_payload_hash": "payload",
        "frames": [frame],
    }
    plan = build_plan("local_e1", queries[0], [batch], [], 3, config)
    assert plan["frame_batches"][0]["ordered_frames"] == [
        {"timestamp_sec": 1.0, "sha256": "a"}
    ]
    assert "contact" not in json.dumps(plan).lower()

