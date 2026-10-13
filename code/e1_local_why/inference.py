"""Prompt planning, resumable response caching, and two-condition query loops."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from code.e0.dataset import atomic_write_json, stable_hash

from . import PROMPT_VERSION, PROTOCOL_VERSION
from .frames import local_view, validate_interval
from .model import LocalQwenClient, parse_decision
from .prompts import SYSTEM_INSTRUCTIONS, prompt_text, response_schema


def build_plan(
    condition: str,
    query: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    previous_responses: Sequence[Mapping[str, Any]],
    reads_remaining: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = prompt_text(query, batches, previous_responses, reads_remaining)
    frame_identity = [
        {
            "start_sec": batch["start_sec"],
            "end_sec": batch["end_sec"],
            "ordered_frame_payload_hash": batch["ordered_frame_payload_hash"],
            "ordered_frames": [
                {"timestamp_sec": frame["timestamp_sec"], "sha256": frame["sha256"]}
                for frame in batch["frames"]
            ],
        }
        for batch in batches
    ]
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "condition": condition,
        "question_id": query["question_id"],
        "question": query["question"],
        "displayed_choices": query["displayed_choices"],
        "option_order": query["option_order"],
        "query_time_sec": query["query_time_sec"],
        "available_range": [0.0, query["query_time_sec"]],
        "frame_batches": frame_identity,
        "previous_responses": list(previous_responses),
        "reads_remaining": reads_remaining,
        "model": config["model"],
        "preprocessing": config["preprocessing"],
        "generation": config["generation"],
        "response_schema": response_schema(reads_remaining > 0),
        "prompt": prompt,
    }
    return {
        **identity,
        "request_hash": stable_hash(identity),
        "prompt_hash": stable_hash({"system": SYSTEM_INSTRUCTIONS, "prompt": prompt}),
        "batches": list(batches),
        "allow_request": reads_remaining > 0,
    }


def decide_or_reuse(
    plan: Mapping[str, Any], cache_root: Path, client: LocalQwenClient
) -> tuple[dict[str, Any], bool]:
    cache_path = cache_root / f"{plan['request_hash']}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_hash") != plan["request_hash"]:
            raise ValueError(f"incompatible local response cache at {cache_path}")
        return cached, True
    response = client.decide(plan)
    row = {
        "request_hash": plan["request_hash"],
        "question_id": plan["question_id"],
        "condition": plan["condition"],
        "prompt_version": PROMPT_VERSION,
        "prompt": plan["prompt"],
        "prompt_hash": plan["prompt_hash"],
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "reads_remaining": plan["reads_remaining"],
        "frame_batches": plan["frame_batches"],
        "raw_textual_response": response["raw_text"],
        **{key: value for key, value in response.items() if key != "raw_text"},
    }
    atomic_write_json(cache_path, row)
    return row, False


def _parse_response(response: Mapping[str, Any], allow_request: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return parse_decision(str(response["raw_textual_response"]), allow_request), None
    except Exception as exc:
        return None, {"category": exc.__class__.__name__, "message": str(exc)}


def run_action_clip(
    query: Mapping[str, Any],
    batch: Mapping[str, Any],
    client: LocalQwenClient,
    response_cache: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = build_plan("local_action_clip", query, [batch], [], 3, config)
    response, hit = decide_or_reuse(plan, response_cache, client)
    decision, parse_error = _parse_response(response, True)
    return {
        "condition": "local_action_clip",
        "question_id": query["question_id"],
        "query_time_sec": query["query_time_sec"],
        "annotation_selected_action_clip": True,
        "frame_batch": dict(batch),
        "answer": decision["answer"] if decision else None,
        "unexecuted_request": decision["request"] if decision else None,
        "parse_error": parse_error,
        "response": {**response, "cache_hit": hit},
        "model_calls_this_execution": int(not hit),
        "response_cache_hits_this_execution": int(hit),
        "execution_wall_latency_seconds": time.perf_counter() - started,
    }


def run_e1(
    query: Mapping[str, Any],
    client: LocalQwenClient,
    frame_cache: Path,
    response_cache: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    query_time = float(query["query_time_sec"])
    frame_settings = {
        "count": int(config["frames"]["count"]),
        **config["frames"]["extraction_preprocessing"],
    }
    initial, initial_hit = local_view(
        query,
        max(0.0, query_time - float(config["online"]["initial_lookback_seconds"])),
        query_time,
        0,
        frame_cache,
        frame_settings,
    )
    batches = [initial]
    valid_decisions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    terminal_error: dict[str, Any] | None = None
    response_hits = 0
    model_calls = 0
    frame_hits = int(initial_hit)
    max_reads = int(config["online"]["max_additional_reads"])
    while True:
        reads_used = len(batches) - 1
        reads_remaining = max_reads - reads_used
        plan = build_plan("local_e1", query, batches, valid_decisions, reads_remaining, config)
        response, hit = decide_or_reuse(plan, response_cache, client)
        response_hits += int(hit)
        model_calls += int(not hit)
        decision, parse_error = _parse_response(response, reads_remaining > 0)
        step = {**response, "cache_hit": hit, "parsed_decision": decision, "parse_error": parse_error}
        steps.append(step)
        if decision is None:
            terminal_error = {
                "type": "malformed_output",
                "step_index": len(steps) - 1,
                **(parse_error or {}),
            }
            break
        valid_decisions.append(decision)
        request = decision["request"]
        if request is None:
            break
        try:
            validate_interval(float(request["start_sec"]), float(request["end_sec"]), query_time)
        except Exception as exc:
            terminal_error = {
                "type": "invalid_request",
                "step_index": len(steps) - 1,
                "request": request,
                "category": exc.__class__.__name__,
                "message": str(exc),
            }
            break
        batch, batch_hit = local_view(
            query,
            float(request["start_sec"]),
            float(request["end_sec"]),
            len(batches),
            frame_cache,
            frame_settings,
        )
        batches.append(batch)
        frame_hits += int(batch_hit)
    initial_answer = valid_decisions[0]["answer"] if valid_decisions else None
    final_answer = valid_decisions[-1]["answer"] if valid_decisions else None
    return {
        "condition": "local_e1",
        "question_id": query["question_id"],
        "source_video_id": query["source_video_id"],
        "query_time_sec": query_time,
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "additional_reads": len(batches) - 1,
        "frame_batches": batches,
        "steps": steps,
        "terminal_error": terminal_error,
        "failed": not bool(valid_decisions),
        "model_calls_this_execution": model_calls,
        "response_cache_hits_this_execution": response_hits,
        "frame_cache_hits_this_execution": frame_hits,
        "execution_wall_latency_seconds": time.perf_counter() - started,
    }

