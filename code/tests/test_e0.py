"""Focused integrity tests for the standalone E0 pilot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from code.e0.dataset import option_permutation, select_participant_balanced
from code.e0.c3_v2 import _reference_is_unchanged
from code.e0.frames import extract_or_reuse_frames
from code.e0.inference import (
    answer_or_reuse,
    api_key_from_model_config,
    build_answer_plan,
    retryable_api_error,
)
from code.e0.knowledge import build_knowledge_request, generate_self_or_reuse
from code.e0.knowledge import retrieve_external_v2_or_reuse
from code.e0.prompts import parse_action_semantics, retrieval_queries_v2


def _record(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "participant": "P01",
        "source_video_id": "P01-test",
        "source_clip_key": "P01-test:0.200000-0.700000",
        "requested_start": "00:00:00.200",
        "requested_end": "00:00:00.700",
        "start_sec": 0.2,
        "end_sec": 0.7,
        "action_phrase": "pick up tea towel",
        "question": "Why did the person <pick up tea towel>?",
        "displayed_choices": [
            "CHOICE_SECRET_ALPHA",
            "CHOICE_SECRET_BETA",
            "CHOICE_SECRET_GAMMA",
            "CHOICE_SECRET_DELTA",
            "CHOICE_SECRET_EPSILON",
        ],
        "option_order": [4, 0, 2, 1, 3],
        "SECRET_GOLD_MARKER": "must-never-cross-knowledge-boundary",
        "choices_marker": "must-never-cross-knowledge-boundary",
        "episode_marker": "must-never-cross-knowledge-boundary",
    }


def _frame_record(tmp_path: Path) -> dict:
    frames = []
    for index in range(2):
        path = tmp_path / f"f{index}.jpg"
        path.write_bytes(b"test")
        frames.append({"path": str(path), "timestamp_sec": 0.2 + index * 0.5, "sha256": str(index)})
    return {"ordered_frame_payload_hash": "shared-frame-hash", "frames": frames}


def test_deterministic_sampling_and_option_gold_remapping() -> None:
    items = {
        f"q-{participant}-{index}": {"participant": participant}
        for participant in [f"P{value:02d}" for value in range(1, 10)]
        for index in range(15)
    }
    selected_a, reserved_a, extras_a = select_participant_balanced(items, seed=42, sample_size=100)
    selected_b, reserved_b, extras_b = select_participant_balanced(items, seed=42, sample_size=100)
    assert selected_a == selected_b
    assert reserved_a == reserved_b
    assert extras_a == extras_b
    assert len(selected_a) == 100 and len(set(selected_a)) == 100
    distribution = {participant: sum(question_id.startswith(f"q-{participant}-") for question_id in selected_a) for participant in [f"P{value:02d}" for value in range(1, 10)]}
    assert sorted(distribution.values()) == [11, 11, 11, 11, 11, 11, 11, 11, 12]
    assert len(extras_a) == 1

    permutation = option_permutation("fixed-question", 42)
    original_gold = 3
    displayed_gold = permutation.index(original_gold)
    displayed_choices = [["A0", "A1", "A2", "A3", "A4"][index] for index in permutation]
    assert displayed_choices[displayed_gold] == "A3"
    assert permutation == option_permutation("fixed-question", 42)


def test_knowledge_request_whitelist_and_fake_client_sees_no_markers(tmp_path: Path) -> None:
    record = _record()
    request = build_knowledge_request(record)
    assert set(request.allowed_payload()) == {"action_phrase", "kitchen_context", "prompt_version"}
    serialized = json.dumps(request.allowed_payload())
    assert "must-never-cross" not in serialized
    captured: list[str] = []

    def fake_call(prompt: str, model: str, settings: dict):
        captured.append(prompt)
        return [
            "Tea towels can be moved so they are within reach for drying clean utensils and hands during kitchen work.",
            "A towel may be picked up to wipe a small spill or remove moisture from a nearby work surface.",
            "Dry fabric can provide grip or limited hand protection when handling an object, depending on heat and thickness.",
        ], {"request_id": "fake", "raw_text": "fake"}

    knowledge, reused = generate_self_or_reuse(
        request,
        tmp_path,
        "fake-provider",
        "fake-model",
        {"reasoning_effort": "low", "max_output_tokens": 800},
        fake_call,
    )
    assert not reused and len(knowledge["facts"]) == 3
    assert len(captured) == 1
    assert "must-never-cross" not in captured[0]
    assert "CHOICE_SECRET_ALPHA" not in captured[0]


def test_condition_payload_boundaries_and_shared_visual_inputs(tmp_path: Path) -> None:
    record = _record()
    frame_record = _frame_record(tmp_path)
    c2 = {"cache_id": "c2", "facts": ["fact a", "fact b", "fact c"]}
    c3 = {"cache_id": "c3", "facts": ["fact x", "fact y", "fact z"]}
    settings = {"reasoning_effort": "low", "max_output_tokens": 256, "image_detail": "high"}
    plans = {
        "C0": build_answer_plan(record, "C0", "p", "m", settings),
        "C1": build_answer_plan(record, "C1", "p", "m", settings, frame_record),
        "C2": build_answer_plan(record, "C2", "p", "m", settings, frame_record, c2),
        "C3": build_answer_plan(record, "C3", "p", "m", settings, frame_record, c3),
    }
    assert plans["C0"]["image_paths"] == []
    assert plans["C0"]["ordered_frame_payload_hash"] is None
    assert plans["C0"]["knowledge_cache_id"] is None
    assert plans["C1"]["knowledge_cache_id"] is None
    assert plans["C1"]["image_paths"] == plans["C2"]["image_paths"] == plans["C3"]["image_paths"]
    assert {plans[name]["ordered_frame_payload_hash"] for name in ("C1", "C2", "C3")} == {"shared-frame-hash"}
    assert all(plans[name]["displayed_choices"] == record["displayed_choices"] for name in plans)
    assert all(plans[name]["option_order"] == record["option_order"] for name in plans)


def test_separate_c2_and_c3_fact_count_policies(tmp_path: Path) -> None:
    record = _record()
    frame_record = _frame_record(tmp_path)
    weak_external = {"cache_id": "weak-c3", "facts": ["only one sourced fact"]}
    with pytest.raises(ValueError, match="requires 3–5 usable background facts"):
        build_answer_plan(
            record,
            "C3",
            "p",
            "m",
            {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
            frame_record,
            weak_external,
        )
    revised = build_answer_plan(
        record,
        "C3",
        "p",
        "m",
        {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
        frame_record,
        weak_external,
        minimum_c3_facts=1,
        provider_endpoint="https://example.invalid/v1",
        knowledge_policy_version="test-v2",
    )
    assert revised["knowledge_facts"] == ["only one sourced fact"]
    with pytest.raises(ValueError, match="C2 requires 3–5"):
        build_answer_plan(
            record,
            "C2",
            "p",
            "m",
            {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
            frame_record,
            weak_external,
        )


@pytest.mark.parametrize(
    ("phrase", "action", "object_phrase", "container", "concept", "uncertain"),
    [
        ("stir contents of pan", "stir", "contents", "pan", "frying pan", False),
        ("stir contents of frying pan", "stir", "contents", "frying pan", "frying pan", False),
        ("rotate knob of gas stove", "rotate", "control knob", "gas stove", "gas stove control knob", False),
        ("pinch onion", "pinch", "onion", None, "onion", False),
        ("move items", "move", "items", None, "items", True),
        ("turn dial", "turn", "dial", None, "dial", True),
        ("clean sponge", "clean", "sponge", None, "cleaning sponge", False),
        ("transfer skimmer", "transfer", "skimmer", None, "skimming utensil", False),
    ],
)
def test_c3_v2_action_parsing(
    phrase: str,
    action: str,
    object_phrase: str,
    container: str | None,
    concept: str,
    uncertain: bool,
) -> None:
    parsed = parse_action_semantics(phrase)
    assert parsed.action == action
    assert parsed.object_phrase == object_phrase
    assert parsed.container_or_appliance == container
    assert parsed.lookup_concept == concept
    assert bool(parsed.uncertainty) is uncertain
    queries = retrieval_queries_v2(phrase)
    assert queries[0]["kind"] == "action_kitchen_context"
    assert queries[-1]["kind"] == "broad_concept"
    assert len({row["query"] for row in queries}) == len(queries)


def test_c3_v2_context_search_disambiguation_and_exact_source_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_get_json(http, endpoint, params, timeout, max_retries):
        calls.append(dict(params))
        if params.get("generator") == "search":
            query = params["gsrsearch"]
            if query.startswith("tap spoon kitchen"):
                pages = [{"index": 1, "pageid": 10, "title": "Spoon", "pageprops": {}, "extract": "A spoon is a utensil with a shallow bowl at the end of a handle. Spoons are used in food preparation to measure, mix, stir, and serve ingredients.", "revisions": [{"revid": 100, "timestamp": "2026-01-01T00:00:00Z"}], "fullurl": "https://en.wikipedia.org/wiki/Spoon"}]
            elif query == "spoon":
                pages = [{"index": 1, "pageid": 11, "title": "Spoon (disambiguation)", "pageprops": {"disambiguation": ""}, "extract": "Spoon may refer to several unrelated meanings.", "revisions": [{"revid": 11, "timestamp": "2026-01-01T00:00:00Z"}], "fullurl": "https://en.wikipedia.org/wiki/Spoon_(disambiguation)"}]
            else:
                pages = []
            return {"query": {"pages": pages}, "continue": {"gsroffset": 5, "continue": "-||"}}
        raise AssertionError(f"unexpected request: {params}")

    monkeypatch.setattr("code.e0.knowledge._get_json", fake_get_json)

    class Headers:
        def update(self, value):
            pass

    class FakeSession:
        headers = Headers()

    settings = {
        "version": "e0_wikipedia_reference_v2",
        "provider": "wikipedia_mediawiki_api",
        "endpoint": "https://en.wikipedia.org/w/api.php",
        "max_search_results": 5,
        "max_pages": 5,
        "fact_count": 5,
        "extract_chars": 1200,
        "local_text_limit_chars": 1200,
        "fact_word_budget": 180,
        "timeout_seconds": 1,
        "user_agent": "test",
        "max_retries": 0,
    }
    request = build_knowledge_request({"action_phrase": "tap spoon", "choice": "SECRET"})
    result, reused = retrieve_external_v2_or_reuse(
        request, tmp_path, settings, session=FakeSession()
    )
    assert not reused and result["status"] == "usable"
    assert [row["query"] for row in result["search_audit"]] == [
        row["query"] for row in retrieval_queries_v2("tap spoon")
    ]
    assert any(row["status"] == "disambiguation" for row in result["page_acquisitions"])
    assert all("may refer" not in fact.casefold() for fact in result["facts"])
    assert len(result["page_acquisitions"]) == 2
    documents = {
        (row.get("page_id"), row.get("revision_id")): row
        for row in result["page_acquisitions"]
    }
    for fact, source in zip(result["facts"], result["fact_provenance"]):
        assert fact == source["supporting_span"]
        assert fact in documents[(source["page_id"], source["revision_id"])]["extract"]
    assert all(call.get("exchars", 1200) <= 1200 for call in calls)
    assert all(call.get("exintro") == 1 for call in calls if "prop" in call)
    assert all(call.get("exlimit", 1) <= 5 for call in calls if "prop" in call)


def test_revised_c3_fact_payload_changes_answer_identity(tmp_path: Path) -> None:
    record = _record()
    frame_record = _frame_record(tmp_path)
    settings = {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"}
    common = dict(
        record=record,
        condition="C3",
        provider="p",
        model="m",
        settings=settings,
        frame_record=frame_record,
        minimum_c3_facts=1,
        provider_endpoint="https://endpoint.invalid/v1",
        knowledge_policy_version="policy-v2",
    )
    first = build_answer_plan(knowledge_record={"cache_id": "same", "facts": ["fact one"]}, **common)
    same = build_answer_plan(knowledge_record={"cache_id": "same", "facts": ["fact one"]}, **common)
    changed = build_answer_plan(knowledge_record={"cache_id": "same", "facts": ["fact two"]}, **common)
    assert first["request_hash"] == same["request_hash"]
    assert first["request_hash"] != changed["request_hash"]


def test_revised_c3_zero_fact_is_an_explicit_non_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 1–5 usable background facts"):
        build_answer_plan(
            _record(),
            "C3",
            "p",
            "m",
            {"reasoning_effort": "medium", "max_output_tokens": 4096, "image_detail": "high"},
            _frame_record(tmp_path),
            {"cache_id": "abstained", "status": "abstained", "facts": []},
            minimum_c3_facts=1,
            provider_endpoint="https://endpoint.invalid/v1",
            knowledge_policy_version="policy-v2",
        )


def test_reference_hash_guard_detects_mutation(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text('{"immutable": true}\n', encoding="utf-8")
    from code.e0.dataset import sha256_file

    manifest = {
        "reference_artifacts": {
            "files": {
                "reference": {"path": str(reference), "sha256": sha256_file(reference)}
            }
        }
    }
    _reference_is_unchanged(manifest)
    reference.write_text('{"immutable": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable reference artifact changed"):
        _reference_is_unchanged(manifest)


def test_actual_extracted_frame_timestamps_stay_inside_annotation(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    record = _record()
    record["source_video_path"] = str(video)
    frame_record, reused = extract_or_reuse_frames(
        record,
        tmp_path / "frames",
        {
            "count": 16,
            "max_edge": 1280,
            "jpeg_quality": 90,
            "colorspace": "RGB",
            "resize": "preserve_aspect_ratio_lanczos_if_needed",
            "jpeg_subsampling": 0,
        },
    )
    assert not reused
    timestamps = [frame["timestamp_sec"] for frame in frame_record["frames"]]
    assert timestamps == sorted(set(timestamps))
    assert all(record["start_sec"] <= timestamp <= record["end_sec"] for timestamp in timestamps)
    assert frame_record["actual_unique_frame_count"] < 16
    assert frame_record["used_all_available_frames"]


class FakeAnswerClient:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, plan: dict) -> dict:
        self.calls += 1
        return {
            "raw_text": "B",
            "request_timestamp": "2026-01-01T00:00:00+00:00",
            "latency_seconds": 0.1,
            "request_id": "fake-response",
            "response_model": "fake-model",
            "status": "completed",
            "usage": None,
        }


def test_resume_avoids_duplicate_calls_and_rejects_incompatible_cache(tmp_path: Path) -> None:
    plan = build_answer_plan(
        _record(),
        "C0",
        "fake-provider",
        "fake-model",
        {"reasoning_effort": "low", "max_output_tokens": 256, "image_detail": "high"},
    )
    client = FakeAnswerClient()
    first, first_reused = answer_or_reuse(plan, tmp_path, client)
    second, second_reused = answer_or_reuse(plan, tmp_path, client)
    assert not first_reused and second_reused
    assert first == second and client.calls == 1

    cache_path = tmp_path / "C0" / f"{plan['request_hash']}.json"
    corrupted = json.loads(cache_path.read_text())
    corrupted["request_hash"] = "incompatible"
    cache_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="incompatible answer cache"):
        answer_or_reuse(plan, tmp_path, client)
    assert client.calls == 1


def test_hard_quota_error_is_not_retryable() -> None:
    RateLimitError = type("RateLimitError", (Exception,), {})
    error = RateLimitError("You have no credits remaining; credit_balance_exhausted")
    error.body = {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}
    assert not retryable_api_error(error)


def test_credential_loader_prefers_environment_and_reads_named_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "credential_source.py"
    source.write_text('SYNTHETIC_KEY = "marker-from-file"\n', encoding="utf-8")
    config = {
        "credential_environment_variable": "E0_SYNTHETIC_API_KEY",
        "credential_source_python_file": str(source),
        "credential_source_variable": "SYNTHETIC_KEY",
    }
    monkeypatch.delenv("E0_SYNTHETIC_API_KEY", raising=False)
    value, origin = api_key_from_model_config(config)
    assert value == "marker-from-file"
    assert origin == f"python_source:{source}#SYNTHETIC_KEY"

    monkeypatch.setenv("E0_SYNTHETIC_API_KEY", "marker-from-environment")
    value, origin = api_key_from_model_config(config)
    assert value == "marker-from-environment"
    assert origin == "environment:E0_SYNTHETIC_API_KEY"


def test_invalid_self_knowledge_is_terminal_and_not_regenerated(tmp_path: Path) -> None:
    calls = 0

    def invalid_call(prompt: str, model: str, settings: dict) -> tuple[list[str], dict]:
        nonlocal calls
        calls += 1
        return ["Only one fact."], {"raw_text": '{"facts":["Only one fact."]}'}

    request = build_knowledge_request(_record())
    arguments = (
        request,
        tmp_path,
        "fake-provider",
        "fake-model",
        {"max_output_tokens": 100},
        invalid_call,
    )
    with pytest.raises(ValueError, match="returned 1 facts"):
        generate_self_or_reuse(*arguments)
    with pytest.raises(ValueError, match="returned 1 facts"):
        generate_self_or_reuse(*arguments)
    assert calls == 1
