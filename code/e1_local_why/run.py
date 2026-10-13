"""Run the fully local HD-EPIC action-clip and model-directed E1 evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, sha256_file, stable_hash
from code.e0.inference import utc_now
from code.e1_online_why.prompts import query_question

from . import PROTOCOL_VERSION
from .frames import action_clip_batch, load_e0_action_index, local_view
from .inference import build_plan, run_action_clip, run_e1
from .model import LocalQwenClient, parse_decision
from .report import generate_report

LOGGER = logging.getLogger("e1_local_why")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def load_config(path: Path) -> dict[str, Any]:
    root = project_root()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("local E1 configuration must be a YAML object")
    config = json.loads(json.dumps(raw))
    for section, key in (
        ("reference", "e0_split"),
        ("reference", "e0_reserved_ids"),
        ("reference", "e0_action_frame_index"),
        ("dataset", "video_root"),
        ("outputs", "run_root"),
        ("model", "local_path"),
    ):
        config[section][key] = _resolve(root, config[section][key])
    config["config_path"] = str(path.resolve())
    config["project_root"] = str(root)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config["experiment"]["development_sample_size"]) != 100:
        raise ValueError("local E1 must use the exact dev100")
    if int(config["frames"]["count"]) != 16:
        raise ValueError("local E1 requires 16 frames per view")
    if config["frames"]["transport"] != "individual_timestamp_associated_images":
        raise ValueError("local E1 forbids contact sheets")
    if float(config["online"]["initial_lookback_seconds"]) != 5.0:
        raise ValueError("local E1 requires the fixed five-second initial view")
    if int(config["online"]["max_additional_reads"]) != 3:
        raise ValueError("local E1 requires a three-read allowance")
    if not bool(config["model"]["local_files_only"]):
        raise ValueError("local model loading must be offline-only")
    if bool(config["generation"]["do_sample"]):
        raise ValueError("evaluation generation must be deterministic")
    if int(config["preprocessing"]["canvas_size"]) != 256:
        raise ValueError("the frozen image canvas is 256x256")


def prepare(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    split_path = Path(config["reference"]["e0_split"])
    reserved_path = Path(config["reference"]["e0_reserved_ids"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    reserved = json.loads(reserved_path.read_text(encoding="utf-8"))
    records = split.get("records", [])
    if split.get("sample_size") != 100 or len(records) != 100 or len(split.get("scoring_labels", {})) != 100:
        raise ValueError("reference split is not the saved E0 dev100")
    selected_ids = {str(record["question_id"]) for record in records}
    reserved_ids = {str(value) for value in reserved.get("question_ids", [])}
    if len(reserved_ids) != 400 or selected_ids & reserved_ids:
        raise ValueError("the saved selected/reserved 100/400 partition changed")
    queries = []
    for record in records:
        source = Path(str(record["source_video_path"]))
        if not source.is_file():
            participant = str(record["participant"])
            source = Path(config["dataset"]["video_root"]) / "HD-EPIC" / "Videos" / participant / f"{record['source_video_id']}.mp4"
        if not source.is_file():
            raise FileNotFoundError(f"source video is missing for {record['question_id']}: {source}")
        query = {
            "question_id": str(record["question_id"]),
            "participant": record["participant"],
            "source_video_id": record["source_video_id"],
            "source_video_path": str(source.resolve()),
            "query_time_sec": float(record["end_sec"]),
            "question": query_question(record),
            "displayed_choices": list(record["displayed_choices"]),
            "option_order": list(record["option_order"]),
        }
        forbidden = {"start_sec", "requested_start", "duration", "gold", "narration"} & set(query)
        if forbidden:
            raise AssertionError(f"hidden E1 fields leaked into query: {forbidden}")
        queries.append(query)
    fingerprint_material = {
        "protocol_version": PROTOCOL_VERSION,
        "split_sha256": sha256_file(split_path),
        "reserved_sha256": sha256_file(reserved_path),
        "action_frame_index_sha256": sha256_file(Path(config["reference"]["e0_action_frame_index"])),
        "question_ids": [query["question_id"] for query in queries],
        "model": config["model"],
        "preprocessing": config["preprocessing"],
        "frames": config["frames"],
        "online": config["online"],
        "generation": config["generation"],
    }
    fingerprint = stable_hash(fingerprint_material)
    run_root = Path(config["outputs"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_root / "inputs" / "queries.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "material_protocol_fingerprint": fingerprint,
            "records": queries,
        },
    )
    resolved = json.loads(json.dumps(config))
    resolved.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "material_protocol_fingerprint": fingerprint,
            "reference_split_sha256": fingerprint_material["split_sha256"],
            "reference_reserved_sha256": fingerprint_material["reserved_sha256"],
            "external_inference_api_used": False,
            "model_weights_downloaded": False,
            "prompts_preprocessing_generation_frozen_before_accuracy": True,
        }
    )
    atomic_write_json(run_root / "config.resolved.json", resolved)
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("material_protocol_fingerprint") != fingerprint:
            raise RuntimeError("existing local E1 run has a different material identity")
    else:
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment_name": config["experiment"]["name"],
            "material_protocol_fingerprint": fingerprint,
            "created_at": utc_now(),
            "selected_count": 100,
            "reserved_count": 400,
            "reserved_items_evaluated": False,
            "external_inference_api_used": False,
        }
        atomic_write_json(manifest_path, manifest)
    return queries, split, fingerprint


def _setup_logging(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(run_root / "run.log"), logging.StreamHandler(sys.stderr)],
        force=True,
    )


def _load_trace(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace.get("material_protocol_fingerprint") != fingerprint:
        raise RuntimeError(f"incompatible completed trace: {path}")
    return trace


def capacity_preflight(
    client: LocalQwenClient,
    query: Mapping[str, Any],
    config: Mapping[str, Any],
    run_root: Path,
    fingerprint: str,
) -> dict[str, Any]:
    preflight_root = run_root / "preflight"
    path = preflight_root / "capacity.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("material_protocol_fingerprint") != fingerprint:
            raise RuntimeError("incompatible capacity preflight exists")
        if existing.get("status") != "passed":
            raise RuntimeError("existing capacity preflight did not pass")
        return existing
    query_time = float(query["query_time_sec"])
    settings = {"count": 16, **config["frames"]["extraction_preprocessing"]}
    batches: list[dict[str, Any]] = []
    intervals = []
    for index in range(4):
        end = query_time - index * 5.0
        start = max(0.0, end - 5.0)
        if not start < end:
            raise RuntimeError("fixed preflight item lacks a 20-second available prefix")
        intervals.append((start, end))
    capacity = []
    for index, (start, end) in enumerate(intervals):
        batch, _ = local_view(
            query, start, end, index, run_root / "cache" / "frames", settings
        )
        batches.append(batch)
        plan = build_plan(
            "capacity_preflight",
            query,
            batches,
            [],
            3 - index,
            config,
        )
        response = client.decide(plan)
        parse_error = None
        try:
            parse_decision(response["raw_text"], 3 - index > 0)
        except Exception as exc:
            parse_error = {"category": exc.__class__.__name__, "message": str(exc)}
        if int(response["image_count"]) != 16 * (index + 1):
            raise RuntimeError("processor did not retain every accumulated preflight frame")
        capacity.append(
            {
                "frame_count": 16 * (index + 1),
                "intervals": [[a, b] for a, b in intervals[: index + 1]],
                **response,
                "parse_error": parse_error,
            }
        )
        LOGGER.info(
            "capacity preflight frames=%d tokens=%d peak=%.2fGiB latency=%.2fs",
            16 * (index + 1),
            response["input_token_count"],
            response["peak_gpu_memory_reserved_bytes"] / (1024**3),
            response["latency_seconds"],
        )
    passed = all(item["parse_error"] is None for item in capacity)
    payload = {
        "timestamp": utc_now(),
        "status": "passed" if passed else "failed_contract",
        "material_protocol_fingerprint": fingerprint,
        "question_id": query["question_id"],
        "image_policy": config["preprocessing"],
        "capacity": capacity,
    }
    atomic_write_json(path, payload)
    if not passed:
        raise RuntimeError("capacity preflight produced a malformed model response")
    return payload


def execute_items(
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    client: LocalQwenClient,
    fingerprint: str,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    run_root = Path(config["outputs"]["run_root"])
    response_cache = run_root / "cache" / "responses"
    frame_cache = run_root / "cache" / "frames"
    action_root = run_root / "traces" / "local_action_clip"
    e1_root = run_root / "traces" / "local_e1"
    for path in (response_cache, frame_cache, action_root, e1_root, run_root / "predictions"):
        path.mkdir(parents=True, exist_ok=True)
    action_index = load_e0_action_index(Path(config["reference"]["e0_action_frame_index"]))
    selected = list(queries[:limit] if limit is not None else queries)
    action_traces: dict[str, dict[str, Any]] = {}
    e1_traces: dict[str, dict[str, Any]] = {}
    for number, query in enumerate(selected, start=1):
        question_id = str(query["question_id"])
        action_path = action_root / f"{question_id}.json"
        action = _load_trace(action_path, fingerprint)
        if action is None:
            action = run_action_clip(
                query,
                action_clip_batch(question_id, action_index),
                client,
                response_cache,
                config,
            )
            action["material_protocol_fingerprint"] = fingerprint
            atomic_write_json(action_path, action)
        action_traces[question_id] = action
        e1_path = e1_root / f"{question_id}.json"
        e1 = _load_trace(e1_path, fingerprint)
        if e1 is None:
            e1 = run_e1(query, client, frame_cache, response_cache, config)
            e1["material_protocol_fingerprint"] = fingerprint
            atomic_write_json(e1_path, e1)
        e1_traces[question_id] = e1
        LOGGER.info(
            "completed %d/%d %s action=%s initial=%s final=%s reads=%d error=%s",
            number,
            len(selected),
            question_id,
            action.get("answer"),
            e1.get("initial_answer"),
            e1.get("final_answer"),
            e1.get("additional_reads"),
            bool(e1.get("terminal_error")),
        )
    order = [str(query["question_id"]) for query in queries]
    all_action = {
        question_id: trace
        for question_id in order
        if (trace := _load_trace(action_root / f"{question_id}.json", fingerprint)) is not None
    }
    all_e1 = {
        question_id: trace
        for question_id in order
        if (trace := _load_trace(e1_root / f"{question_id}.json", fingerprint)) is not None
    }
    atomic_write_jsonl(
        run_root / "predictions" / "local_action_clip.jsonl",
        [
            {
                "question_id": question_id,
                "answer": trace.get("answer"),
                "unexecuted_request": trace.get("unexecuted_request"),
                "parse_error": trace.get("parse_error"),
            }
            for question_id, trace in all_action.items()
        ],
    )
    atomic_write_jsonl(
        run_root / "predictions" / "local_e1.jsonl",
        [
            {
                "question_id": question_id,
                "initial_answer": trace.get("initial_answer"),
                "final_answer": trace.get("final_answer"),
                "additional_reads": trace.get("additional_reads"),
                "terminal_error": trace.get("terminal_error"),
            }
            for question_id, trace in all_e1.items()
        ],
    )
    return all_action, all_e1


def verify_integrity(
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    action_traces: Mapping[str, Mapping[str, Any]],
    e1_traces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(queries) != 100 or len(action_traces) != 100 or len(e1_traces) != 100:
        raise ValueError(
            f"integrity requires 100 queries and traces; got {len(queries)}, {len(action_traces)}, {len(e1_traces)}"
        )
    frame_records = 0
    for query in queries:
        question_id = str(query["question_id"])
        query_time = float(query["query_time_sec"])
        action = action_traces[question_id]
        if action["frame_batch"].get("source") != "exact_saved_e0_c1_action_clip_frames":
            raise ValueError(f"action condition provenance drift for {question_id}")
        trace = e1_traces[question_id]
        batches = trace["frame_batches"]
        if not 1 <= len(batches) <= 4 or int(trace["additional_reads"]) != len(batches) - 1:
            raise ValueError(f"invalid E1 read count for {question_id}")
        expected_start = max(0.0, query_time - 5.0)
        if abs(float(batches[0]["start_sec"]) - expected_start) > 1e-6 or abs(float(batches[0]["end_sec"]) - query_time) > 1e-6:
            raise ValueError(f"initial interval drift for {question_id}")
        for batch in [action["frame_batch"], *batches]:
            start, end = float(batch["start_sec"]), float(batch["end_sec"])
            if len(batch["frames"]) > 16 or not batch["frames"]:
                raise ValueError(f"invalid frame count for {question_id}")
            seen = set()
            for frame in batch["frames"]:
                timestamp = float(frame["timestamp_sec"])
                if not start - 1e-6 <= timestamp <= end + 1e-6:
                    raise ValueError(f"frame outside requested interval for {question_id}")
                if batch is not action["frame_batch"] and timestamp > query_time + 1e-6:
                    raise ValueError(f"E1 frame after query time for {question_id}")
                if timestamp in seen:
                    raise ValueError(f"duplicate frame timestamp inside a view for {question_id}")
                seen.add(timestamp)
                path = Path(frame["path"])
                if not path.is_file() or sha256_file(path) != frame["sha256"]:
                    raise ValueError(f"missing or corrupt frame for {question_id}: {path}")
                frame_records += 1
    audit = {
        "timestamp": utc_now(),
        "status": "passed",
        "question_count": 100,
        "action_trace_count": 100,
        "e1_trace_count": 100,
        "reserved_items_evaluated": False,
        "contact_sheets_created_or_used": False,
        "e1_annotated_action_start_exposed": False,
        "e1_gold_narration_or_prior_predictions_exposed": False,
        "frame_records_rehashed": frame_records,
    }
    atomic_write_json(Path(config["outputs"]["run_root"]) / "integrity_audit.json", audit)
    return audit


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preflight", "smoke", "run", "report", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def write_smoke_result(
    run_root: Path,
    queries: Sequence[Mapping[str, Any]],
    fingerprint: str,
    action: Mapping[str, Mapping[str, Any]],
    e1: Mapping[str, Mapping[str, Any]],
) -> None:
    fixed_ids = [str(queries[0]["question_id"]), str(queries[1]["question_id"])]
    failures = []
    for question_id in fixed_ids:
        action_trace = action.get(question_id)
        e1_trace = e1.get(question_id)
        if action_trace is None or action_trace.get("answer") is None:
            failures.append({"question_id": question_id, "condition": "local_action_clip"})
        if e1_trace is None or e1_trace.get("initial_answer") is None:
            failures.append({"question_id": question_id, "condition": "local_e1"})
    payload = {
        "timestamp": utc_now(),
        "status": "passed" if not failures else "failed_contract",
        "material_protocol_fingerprint": fingerprint,
        "fixed_question_ids": fixed_ids,
        "correctness_inspected_before_freeze": False,
        "action_trace_count": len(action),
        "e1_trace_count": len(e1),
        "failures": failures,
    }
    atomic_write_json(run_root / "preflight" / "smoke.json", payload)
    if failures:
        raise RuntimeError(f"smoke test produced missing valid answers: {failures}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    run_root = Path(config["outputs"]["run_root"])
    _setup_logging(run_root)
    queries, split, fingerprint = prepare(config)
    if args.stage == "prepare":
        return 0
    preflight_path = run_root / "preflight" / "capacity.json"
    client: LocalQwenClient | None = None
    if args.stage in {"preflight", "smoke", "run", "all"}:
        client = LocalQwenClient(config["model"], config["preprocessing"], config["generation"])
        environment = client.environment_details()
        environment.update(
            {
                "executable": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        atomic_write_json(run_root / "environment.json", environment)
        resolved = json.loads((run_root / "config.resolved.json").read_text(encoding="utf-8"))
        resolved["resolved_environment"] = environment
        atomic_write_json(run_root / "config.resolved.json", resolved)
    preflight: dict[str, Any] | None = None
    if args.stage in {"preflight", "smoke", "run", "all"}:
        assert client is not None
        preflight = capacity_preflight(client, queries[0], config, run_root, fingerprint)
    if args.stage == "preflight":
        return 0
    if args.stage == "smoke":
        assert client is not None
        action, e1 = execute_items(config, queries, client, fingerprint, limit=2)
        write_smoke_result(run_root, queries, fingerprint, action, e1)
        return 0
    if args.stage in {"run", "all"}:
        assert client is not None
        # Two fixed smoke examples exercise the same frozen pipeline and become resumable results.
        smoke_action, smoke_e1 = execute_items(config, queries, client, fingerprint, limit=2)
        write_smoke_result(
            run_root, queries, fingerprint, smoke_action, smoke_e1
        )
        execute_items(config, queries, client, fingerprint, limit=args.limit)
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        manifest["last_execution"] = {"timestamp": utc_now(), "limit": args.limit}
        atomic_write_json(run_root / "manifest.json", manifest)
    if args.stage in {"report", "all"}:
        action_root = run_root / "traces" / "local_action_clip"
        e1_root = run_root / "traces" / "local_e1"
        action = {
            query["question_id"]: trace
            for query in queries
            if (trace := _load_trace(action_root / f"{query['question_id']}.json", fingerprint)) is not None
        }
        e1 = {
            query["question_id"]: trace
            for query in queries
            if (trace := _load_trace(e1_root / f"{query['question_id']}.json", fingerprint)) is not None
        }
        if preflight is None and preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        generate_report(run_root, queries, split["scoring_labels"], action, e1, preflight)
        verify_integrity(config, queries, action, e1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
