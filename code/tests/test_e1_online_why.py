"""Protocol and integration checks for E1 online why-recognition."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from code.e1_online_why.frames import (
    build_or_reuse_contact_sheets,
    extract_view,
    validate_interval,
)
from code.e1_online_why.inference import parse_decision, run_query
from code.e1_online_why.run import load_config, prepare


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "e1_online_why.yaml"


class FakeClient:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.plans: list[Mapping[str, Any]] = []

    def decide(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        self.plans.append(plan)
        decision = self.decisions.pop(0)
        return {
            "raw_text": json.dumps(decision),
            "request_timestamp": "2026-09-30T00:00:00+00:00",
            "latency_seconds": 0.1,
            "request_id": f"fake-{len(self.plans)}",
            "response_model": "azure/gpt-5.6-sol",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }


def _query() -> dict[str, Any]:
    return {
        "question_id": "q1",
        "source_video_id": "video1",
        "source_video_path": "/unused/video.mp4",
        "query_time_sec": 10.0,
        "question": "Why perform <stir bowl>?",
        "displayed_choices": ["a", "b", "c", "d", "e"],
        "option_order": [0, 1, 2, 3, 4],
    }


def _fake_extract(
    query: Mapping[str, Any],
    start_sec: float,
    end_sec: float,
    view_index: int,
    cache_root: Path,
    frame_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    del cache_root, frame_settings
    return (
        {
            "cache_id": f"view-{view_index}-{start_sec}-{end_sec}",
            "source_video_id": query["source_video_id"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "requested_frame_count": 16,
            "actual_unique_frame_count": 2,
            "used_all_available_frames": True,
            "ordered_frame_payload_hash": f"hash-{view_index}-{start_sec}-{end_sec}",
            "frames": [
                {
                    "timestamp_sec": start_sec,
                    "path": f"/unused/{view_index}-0.jpg",
                    "sha256": f"sha-{view_index}-0",
                },
                {
                    "timestamp_sec": end_sec,
                    "path": f"/unused/{view_index}-1.jpg",
                    "sha256": f"sha-{view_index}-1",
                },
            ],
        },
        False,
    )


def _fake_contact_sheets(
    frame_record: Mapping[str, Any], cache_root: Path
) -> tuple[dict[str, Any], bool]:
    del cache_root
    return (
        {
            "cache_id": f"contact-{frame_record['cache_id']}",
            "contact_sheets": [
                {
                    "index": 0,
                    "path": "/unused/contact.jpg",
                    "sha256": f"contact-sha-{frame_record['cache_id']}",
                    "tiles": [
                        {
                            "tile": index + 1,
                            "frame_index": index,
                            "timestamp_sec": frame["timestamp_sec"],
                            "source_frame_sha256": frame["sha256"],
                        }
                        for index, frame in enumerate(frame_record["frames"])
                    ],
                }
            ],
        },
        False,
    )


def test_prepare_maps_all_videos_and_hides_boundaries_and_gold(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config["outputs"]["run_root"] = str(tmp_path / "run")
    queries, split = prepare(config)
    assert len(queries) == 100
    assert len(split["scoring_labels"]) == 100
    for query, source in zip(queries, split["records"]):
        assert query["query_time_sec"] == source["end_sec"]
        assert Path(query["source_video_path"]).is_file()
        assert query["displayed_choices"] == source["displayed_choices"]
        assert query["option_order"] == source["option_order"]
        assert "start_sec" not in query
        assert "requested_start" not in query
        assert "gold_letter" not in query


def test_parse_and_interval_validation() -> None:
    assert parse_decision('{"answer":"B","request":null}', True) == {
        "answer": "B",
        "request": None,
    }
    assert parse_decision(
        '{"answer":"C","request":{"start_sec":0,"end_sec":10}}', True
    )["request"] == {"start_sec": 0.0, "end_sec": 10.0}
    with pytest.raises(ValueError):
        parse_decision('{"answer":"C","request":{"start_sec":0,"end_sec":1}}', False)
    with pytest.raises(ValueError):
        validate_interval(-1.0, 2.0, 10.0)
    with pytest.raises(ValueError):
        validate_interval(1.0, 11.0, 10.0)
    validate_interval(0.0, 10.0, 10.0)


def test_loop_enforces_three_additional_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("code.e1_online_why.inference.extract_view", _fake_extract)
    monkeypatch.setattr(
        "code.e1_online_why.inference.build_or_reuse_contact_sheets", _fake_contact_sheets
    )
    client = FakeClient(
        [
            {"answer": "A", "request": {"start_sec": 0, "end_sec": 1}},
            {"answer": "B", "request": {"start_sec": 1, "end_sec": 2}},
            {"answer": "C", "request": {"start_sec": 2, "end_sec": 3}},
            {"answer": "D", "request": None},
        ]
    )
    trace = run_query(
        _query(),
        client,
        tmp_path / "frames",
        tmp_path / "responses",
        {"count": 16},
        {
            "provider": "openai_chat_completions",
            "base_url": "http://example.invalid/v1",
            "identifier": "azure/gpt-5.6-sol",
        },
        {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
        3,
        5.0,
    )
    assert trace["initial_answer"] == "A"
    assert trace["final_answer"] == "D"
    assert trace["additional_reads"] == 3
    assert len(trace["steps"]) == 4
    assert [plan["reads_remaining"] for plan in client.plans] == [3, 2, 1, 0]
    assert client.plans[-1]["allow_request"] is False


def test_invalid_request_exposes_no_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[float, float]] = []

    def tracking_extract(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], bool]:
        calls.append((float(args[1]), float(args[2])))
        return _fake_extract(*args, **kwargs)

    monkeypatch.setattr("code.e1_online_why.inference.extract_view", tracking_extract)
    monkeypatch.setattr(
        "code.e1_online_why.inference.build_or_reuse_contact_sheets", _fake_contact_sheets
    )
    client = FakeClient(
        [{"answer": "A", "request": {"start_sec": 9.0, "end_sec": 10.5}}]
    )
    trace = run_query(
        _query(),
        client,
        tmp_path / "frames",
        tmp_path / "responses",
        {"count": 16},
        {
            "provider": "openai_chat_completions",
            "base_url": "http://example.invalid/v1",
            "identifier": "azure/gpt-5.6-sol",
        },
        {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
        3,
        5.0,
    )
    assert calls == [(5.0, 10.0)]
    assert trace["terminated_by_invalid_request"] is True
    assert trace["additional_reads"] == 0


@pytest.mark.parametrize("record_index", [0, 1])
def test_two_real_initial_views_are_bounded_and_independent_of_action_start(
    record_index: int, tmp_path: Path
) -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config["outputs"]["run_root"] = str(tmp_path / "prepared")
    queries, split = prepare(config)
    query = queries[record_index]
    source = split["records"][record_index]
    query_time = float(query["query_time_sec"])
    initial_start = max(0.0, query_time - 5.0)
    frame_record, _ = extract_view(
        query,
        initial_start,
        query_time,
        0,
        tmp_path / "frames",
        {"count": 16, **config["frames"]["preprocessing"]},
    )
    assert frame_record["start_sec"] == initial_start
    assert frame_record["start_sec"] != source["start_sec"]
    assert 1 <= len(frame_record["frames"]) <= 16
    assert all(
        initial_start - 1e-6 <= float(frame["timestamp_sec"]) <= query_time + 1e-6
        for frame in frame_record["frames"]
    )
    contacts, _ = build_or_reuse_contact_sheets(frame_record, tmp_path / "contacts")
    assert len(contacts["contact_sheets"]) <= 4
    assert [
        tile["frame_index"]
        for sheet in contacts["contact_sheets"]
        for tile in sheet["tiles"]
    ] == list(range(len(frame_record["frames"])))
    assert all(Path(sheet["path"]).is_file() for sheet in contacts["contact_sheets"])
