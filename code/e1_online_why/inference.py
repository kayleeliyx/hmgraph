"""Structured model-directed video reading and resumable response caching for E1."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from code.e0.dataset import atomic_write_json, stable_hash
from code.e0.inference import OpenAICompatibleChatClient, _usage_dict, utc_now

from . import LETTERS, PROMPT_VERSION
from .frames import build_or_reuse_contact_sheets, extract_view, validate_interval
from .prompts import SYSTEM_INSTRUCTIONS, prompt_text, response_schema


class OnlineWhyClient(Protocol):
    def decide(self, plan: Mapping[str, Any]) -> dict[str, Any]: ...


class OpenAIOnlineWhyClient(OpenAICompatibleChatClient):
    """Exact E0 OpenAI-compatible transport with an E1 structured response."""

    def decide(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        requested_at = utc_now()
        content: list[dict[str, Any]] = [{"type": "text", "text": plan["prompt"]}]
        image_count = sum(len(batch["contact_sheets"]) for batch in plan["batches"])
        if image_count > 16:
            raise ValueError(f"E1 contact-sheet transport unexpectedly contains {image_count} images")
        for batch_index, batch in enumerate(plan["batches"]):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"View {batch_index}: up to four 2-by-2 contact sheets, chronological. "
                        "Each tile is labeled with its source-recording timestamp."
                    ),
                }
            )
            for sheet in batch["contact_sheets"]:
                path = Path(str(sheet["path"]))
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                mapping = ", ".join(
                    f"tile {tile['tile']}=frame {int(tile['frame_index']) + 1} at "
                    f"{float(tile['timestamp_sec']):.3f}s"
                    for tile in sheet["tiles"]
                )
                content.append(
                    {
                        "type": "text",
                        "text": f"View {batch_index}, contact sheet {int(sheet['index']) + 1}: {mapping}",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                            "detail": plan["settings"]["image_detail"],
                        },
                    }
                )

        def operation() -> Any:
            return self.client.chat.completions.create(
                model=plan["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "e1_online_why_decision",
                        "strict": True,
                        "schema": response_schema(bool(plan["allow_request"])),
                    },
                },
                reasoning_effort=plan["settings"]["reasoning_effort"],
                max_completion_tokens=int(plan["settings"]["max_output_tokens"]),
            )

        response = self._with_retry(operation)
        raw_text = self._message_text(response)
        return {
            "raw_text": raw_text,
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": getattr(response, "_request_id", None) or response.id,
            "response_model": response.model,
            "status": "completed",
            "finish_reason": response.choices[0].finish_reason,
            "usage": _usage_dict(getattr(response, "usage", None)),
        }


def parse_decision(raw_text: str, allow_request: bool) -> dict[str, Any]:
    value = json.loads(raw_text)
    if not isinstance(value, dict) or set(value) != {"answer", "request"}:
        raise ValueError("E1 response must contain exactly answer and request")
    answer = value["answer"]
    if answer not in LETTERS:
        raise ValueError("E1 answer must be one displayed option label")
    request = value["request"]
    if request is None:
        return {"answer": answer, "request": None}
    if not allow_request:
        raise ValueError("E1 response requested video after the read budget was exhausted")
    if not isinstance(request, dict) or set(request) != {"start_sec", "end_sec"}:
        raise ValueError("E1 request must contain exactly start_sec and end_sec")
    start = request["start_sec"]
    end = request["end_sec"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
    ):
        raise ValueError("E1 requested interval must contain finite numeric endpoints")
    return {
        "answer": answer,
        "request": {"start_sec": float(start), "end_sec": float(end)},
    }


def build_decision_plan(
    query: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    previous_responses: Sequence[Mapping[str, Any]],
    reads_remaining: int,
    model_config: Mapping[str, Any],
    answer_settings: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = prompt_text(query, batches, previous_responses, reads_remaining)
    frame_identity = [
        {
            "start_sec": batch["start_sec"],
            "end_sec": batch["end_sec"],
            "ordered_frame_payload_hash": batch["ordered_frame_payload_hash"],
            "contact_sheet_hashes": [sheet["sha256"] for sheet in batch["contact_sheets"]],
        }
        for batch in batches
    ]
    identity = {
        "protocol_version": PROMPT_VERSION,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "question_id": query["question_id"],
        "question": query["question"],
        "displayed_choices": query["displayed_choices"],
        "option_order": query["option_order"],
        "query_time_sec": query["query_time_sec"],
        "available_range": [0.0, query["query_time_sec"]],
        "frame_batches": frame_identity,
        "previous_responses": list(previous_responses),
        "reads_remaining": reads_remaining,
        "provider": model_config["provider"],
        "provider_endpoint": model_config["base_url"],
        "model": model_config["identifier"],
        "settings": dict(answer_settings),
        "response_schema": response_schema(reads_remaining > 0),
    }
    return {
        **identity,
        "request_hash": stable_hash(identity),
        "prompt": prompt,
        "prompt_hash": stable_hash({"system": SYSTEM_INSTRUCTIONS, "prompt": prompt}),
        "batches": list(batches),
        "allow_request": reads_remaining > 0,
    }


def decide_or_reuse(
    plan: Mapping[str, Any], cache_root: Path, client: OnlineWhyClient
) -> tuple[dict[str, Any], bool]:
    cache_path = cache_root / f"{plan['request_hash']}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_hash") != plan["request_hash"]:
            raise ValueError(f"incompatible E1 response cache at {cache_path}")
        return cached, True
    response = client.decide(plan)
    decision = parse_decision(str(response["raw_text"]), bool(plan["allow_request"]))
    row = {
        "request_hash": plan["request_hash"],
        "question_id": plan["question_id"],
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": plan["prompt_hash"],
        "reads_remaining": plan["reads_remaining"],
        "frame_batches": plan["frame_batches"],
        "answer": decision["answer"],
        "request": decision["request"],
        "raw_textual_response": response["raw_text"],
        "request_timestamp": response["request_timestamp"],
        "latency_seconds": response["latency_seconds"],
        "request_id": response.get("request_id"),
        "response_model": response.get("response_model"),
        "finish_reason": response.get("finish_reason"),
        "token_usage": response.get("usage"),
    }
    atomic_write_json(cache_path, row)
    return row, False


def _trace_batch(
    frame_record: Mapping[str, Any],
    cache_hit: bool,
    contact_cache_root: Path,
) -> dict[str, Any]:
    contact_record, contact_hit = build_or_reuse_contact_sheets(frame_record, contact_cache_root)
    return {
        "cache_id": frame_record["cache_id"],
        "cache_hit": cache_hit,
        "source_video_id": frame_record["source_video_id"],
        "start_sec": frame_record["start_sec"],
        "end_sec": frame_record["end_sec"],
        "requested_frame_count": frame_record["requested_frame_count"],
        "actual_unique_frame_count": frame_record["actual_unique_frame_count"],
        "used_all_available_frames": frame_record["used_all_available_frames"],
        "ordered_frame_payload_hash": frame_record["ordered_frame_payload_hash"],
        "frames": [dict(frame) for frame in frame_record["frames"]],
        "contact_sheet_cache_id": contact_record["cache_id"],
        "contact_sheet_cache_hit": contact_hit,
        "contact_sheets": [dict(sheet) for sheet in contact_record["contact_sheets"]],
    }


def run_query(
    query: Mapping[str, Any],
    client: OnlineWhyClient,
    frame_cache_root: Path,
    response_cache_root: Path,
    frame_settings: Mapping[str, Any],
    model_config: Mapping[str, Any],
    answer_settings: Mapping[str, Any],
    max_additional_reads: int,
    initial_lookback_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    query_time = float(query["query_time_sec"])
    initial_start = max(0.0, query_time - initial_lookback_seconds)
    initial_record, frame_hit = extract_view(
        query, initial_start, query_time, 0, frame_cache_root, frame_settings
    )
    contact_cache_root = frame_cache_root.parent / "contact_sheets"
    batches = [_trace_batch(initial_record, frame_hit, contact_cache_root)]
    decisions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    response_cache_hits = 0
    provider_calls = 0
    invalid_request: dict[str, Any] | None = None

    while True:
        reads_used = len(batches) - 1
        reads_remaining = max_additional_reads - reads_used
        plan = build_decision_plan(
            query, batches, decisions, reads_remaining, model_config, answer_settings
        )
        response, response_hit = decide_or_reuse(plan, response_cache_root, client)
        response_cache_hits += int(response_hit)
        provider_calls += int(not response_hit)
        decision = {"answer": response["answer"], "request": response["request"]}
        decisions.append(decision)
        step = dict(response)
        step["cache_hit"] = response_hit
        steps.append(step)
        request = decision["request"]
        if request is None:
            break
        try:
            validate_interval(
                float(request["start_sec"]), float(request["end_sec"]), query_time
            )
        except ValueError as exc:
            invalid_request = {"request": request, "error": str(exc)}
            break
        if reads_remaining <= 0:
            invalid_request = {"request": request, "error": "read budget exhausted"}
            break
        view_index = len(batches)
        frame_record, hit = extract_view(
            query,
            float(request["start_sec"]),
            float(request["end_sec"]),
            view_index,
            frame_cache_root,
            frame_settings,
        )
        batches.append(_trace_batch(frame_record, hit, contact_cache_root))

    return {
        "protocol_version": PROMPT_VERSION,
        "question_id": query["question_id"],
        "source_video_id": query["source_video_id"],
        "query_time_sec": query_time,
        "available_range_sec": [0.0, query_time],
        "question": query["question"],
        "displayed_choices": list(query["displayed_choices"]),
        "option_order": list(query["option_order"]),
        "initial_answer": decisions[0]["answer"],
        "final_answer": decisions[-1]["answer"],
        "additional_reads": len(batches) - 1,
        "terminated_by_invalid_request": invalid_request is not None,
        "invalid_request": invalid_request,
        "frame_batches": batches,
        "steps": steps,
        "provider_calls_this_execution": provider_calls,
        "response_cache_hits_this_execution": response_cache_hits,
        "frame_cache_hits_this_execution": sum(int(batch["cache_hit"]) for batch in batches),
        "contact_sheet_cache_hits_this_execution": sum(
            int(batch["contact_sheet_cache_hit"]) for batch in batches
        ),
        "execution_wall_latency_seconds": time.perf_counter() - started,
    }
