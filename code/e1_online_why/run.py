"""CLI orchestration for E1 online why-recognition."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, sha256_file, stable_hash
from code.e0.inference import _redacted_error, api_key_from_model_config, utc_now

from . import PROTOCOL_VERSION
from .inference import OpenAIOnlineWhyClient, OnlineWhyClient, run_query
from .prompts import query_question
from .report import generate_report

LOGGER = logging.getLogger("e1_online_why")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    root = project_root()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E1 configuration must be a YAML object")
    config = json.loads(json.dumps(raw))
    for section, key in (
        ("reference", "e0_split"),
        ("reference", "e0_reserved_ids"),
        ("reference", "e0_c1_predictions"),
        ("dataset", "video_root"),
        ("outputs", "run_root"),
    ):
        config[section][key] = str(_resolve(root, config[section][key]))
    config["config_path"] = str(path.resolve())
    config["project_root"] = str(root)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config["experiment"]["name"] != "hd_epic_why_dev100_action_end_contact_sheet_v1":
        raise ValueError(
            "E1 run ID is frozen to hd_epic_why_dev100_action_end_contact_sheet_v1"
        )
    if int(config["experiment"]["development_sample_size"]) != 100:
        raise ValueError("E1 requires the saved dev100")
    if int(config["frames"]["count"]) != 16:
        raise ValueError("E1 requires 16 frames per view")
    if config["frames"].get("transport") != "four_2x2_timestamped_contact_sheets":
        raise ValueError("E1 requires the frozen four-contact-sheet transport")
    if float(config["online"]["initial_lookback_seconds"]) != 5.0:
        raise ValueError("E1 requires the fixed five-second initial view")
    if int(config["online"]["max_additional_reads"]) != 3:
        raise ValueError("E1 requires exactly three possible additional reads")
    if config["model"]["provider"] != "openai_chat_completions":
        raise ValueError("E1 must use the exact E0 OpenAI-compatible provider")
    if config["model"]["identifier"] != "azure/gpt-5.6-sol":
        raise ValueError("E1 must use azure/gpt-5.6-sol")


def resolve_video(video_root: Path, source_video_id: str) -> Path:
    participant = source_video_id.split("-", 1)[0]
    direct = video_root / "HD-EPIC" / "Videos" / participant / f"{source_video_id}.mp4"
    if direct.is_file():
        return direct.resolve()
    matches = list(video_root.glob(f"**/{source_video_id}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one source video for {source_video_id} under {video_root}; "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def prepare(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_path = Path(config["reference"]["e0_split"])
    reserved_path = Path(config["reference"]["e0_reserved_ids"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    reserved = json.loads(reserved_path.read_text(encoding="utf-8"))
    if split.get("sample_size") != 100 or len(split.get("records", [])) != 100:
        raise ValueError("E1 reference must be the exact saved E0 dev100")
    if len(split.get("scoring_labels", {})) != 100:
        raise ValueError("E0 dev100 scoring labels are missing")
    selected_ids = {str(record["question_id"]) for record in split["records"]}
    reserved_ids = {str(value) for value in reserved.get("question_ids", [])}
    if len(reserved_ids) != 400 or selected_ids & reserved_ids:
        raise ValueError("E0 selected and reserved ID partition is incompatible")
    video_root = Path(config["dataset"]["video_root"])
    queries: list[dict[str, Any]] = []
    for record in split["records"]:
        query = {
            "question_id": record["question_id"],
            "participant": record["participant"],
            "source_video_id": record["source_video_id"],
            "source_video_path": str(resolve_video(video_root, str(record["source_video_id"]))),
            "query_time_sec": float(record["end_sec"]),
            "question": query_question(record),
            "displayed_choices": list(record["displayed_choices"]),
            "option_order": list(record["option_order"]),
        }
        if "start_sec" in query or "requested_start" in query:
            raise AssertionError("annotated action start leaked into E1 query")
        queries.append(query)

    run_root = Path(config["outputs"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    inputs_root = run_root / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        inputs_root / "queries.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "reference_split_sha256": sha256_file(split_path),
            "reference_reserved_sha256": sha256_file(reserved_path),
            "records": queries,
        },
    )
    material = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_split_sha256": sha256_file(split_path),
        "reference_reserved_sha256": sha256_file(reserved_path),
        "query_ids": [query["question_id"] for query in queries],
        "frame_settings": config["frames"],
        "online": config["online"],
        "model": {
            "provider": config["model"]["provider"],
            "identifier": config["model"]["identifier"],
            "base_url": config["model"]["base_url"],
        },
        "answer": config["answer"],
    }
    fingerprint = stable_hash(material)
    resolved = json.loads(json.dumps(config))
    resolved.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "material_protocol_fingerprint": fingerprint,
            "reference_split_sha256": sha256_file(split_path),
            "reference_reserved_sha256": sha256_file(reserved_path),
            "credential_value_recorded": False,
        }
    )
    resolved["credential_configured"] = bool(api_key_from_model_config(config["model"])[0])
    atomic_write_json(run_root / "config.resolved.json", resolved)
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("material_protocol_fingerprint") != fingerprint:
            raise RuntimeError("existing E1 run has an incompatible material protocol")
        manifest = existing
    else:
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment_name": config["experiment"]["name"],
            "material_protocol_fingerprint": fingerprint,
            "created_at": utc_now(),
            "reference_split_path": str(split_path.resolve()),
            "reference_split_sha256": sha256_file(split_path),
            "reference_reserved_path": str(reserved_path.resolve()),
            "reference_reserved_sha256": sha256_file(reserved_path),
            "selected_count": 100,
            "reserved_items_evaluated": False,
            "annotated_action_start_exposed_to_model": False,
            "gold_labels_exposed_to_model": False,
        }
    atomic_write_json(manifest_path, manifest)
    return queries, split


def _read_traces(run_root: Path, order: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for question_id in order:
        path = run_root / "traces" / f"{question_id}.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def execute(
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    client: OnlineWhyClient,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = Path(config["outputs"]["run_root"])
    selected = list(queries[:limit] if limit is not None else queries)
    frame_cache = run_root / "cache" / "frames"
    response_cache = run_root / "cache" / "responses"
    traces_root = run_root / "traces"
    for path in (
        frame_cache,
        run_root / "cache" / "contact_sheets",
        response_cache,
        traces_root,
        run_root / "predictions",
    ):
        path.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def one(query: Mapping[str, Any]) -> dict[str, Any]:
        return run_query(
            query=query,
            client=client,
            frame_cache_root=frame_cache,
            response_cache_root=response_cache,
            frame_settings={
                "count": config["frames"]["count"],
                **config["frames"]["preprocessing"],
            },
            model_config=config["model"],
            answer_settings=config["answer"],
            max_additional_reads=int(config["online"]["max_additional_reads"]),
            initial_lookback_seconds=float(config["online"]["initial_lookback_seconds"]),
        )

    with ThreadPoolExecutor(max_workers=int(config["execution"]["concurrency"])) as executor:
        futures = {executor.submit(one, query): query for query in selected}
        for future in as_completed(futures):
            query = futures[future]
            try:
                trace = future.result()
                atomic_write_json(traces_root / f"{query['question_id']}.json", trace)
                results.append(trace)
                LOGGER.info(
                    "completed %s initial=%s final=%s reads=%d",
                    query["question_id"],
                    trace["initial_answer"],
                    trace["final_answer"],
                    trace["additional_reads"],
                )
            except Exception as exc:
                failure = {
                    "question_id": query["question_id"],
                    "error_category": exc.__class__.__name__,
                    "error": _redacted_error(exc),
                    "timestamp": utc_now(),
                }
                failures.append(failure)
                LOGGER.exception("failed %s", query["question_id"])
    order = [str(query["question_id"]) for query in queries]
    all_traces = _read_traces(run_root, order)
    predictions = [
        {
            "question_id": trace["question_id"],
            "query_time_sec": trace["query_time_sec"],
            "initial_answer": trace["initial_answer"],
            "final_answer": trace["final_answer"],
            "additional_reads": trace["additional_reads"],
            "trace_path": str((traces_root / f"{trace['question_id']}.json").resolve()),
        }
        for trace in all_traces
    ]
    atomic_write_jsonl(run_root / "predictions" / "predictions.jsonl", predictions)
    atomic_write_jsonl(run_root / "failures.jsonl", failures)
    return all_traces, failures


def e0_c1_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(config["reference"]["e0_c1_predictions"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    split = json.loads(Path(config["reference"]["e0_split"]).read_text(encoding="utf-8"))
    labels = split["scoring_labels"]
    correct = sum(
        int(row.get("parsed_answer") == labels[row["question_id"]]["gold_letter"])
        for row in rows
    )
    if len(rows) != 100 or correct != 83:
        raise ValueError("saved E0 C1 reference no longer matches 83/100")
    return {
        "correct": correct,
        "denominator": 100,
        "accuracy": 0.83,
        "prediction_count": len(rows),
        "input_context": "annotation-selected action clip frames",
    }


def verify_integrity(
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_root = Path(config["outputs"]["run_root"])
    by_id = {str(query["question_id"]): query for query in queries}
    if len(by_id) != 100:
        raise ValueError("E1 integrity: expected 100 unique query IDs")
    frame_count = 0
    contact_sheet_count = 0
    for trace in traces:
        question_id = str(trace["question_id"])
        query = by_id[question_id]
        query_time = float(query["query_time_sec"])
        batches = trace["frame_batches"]
        if not 1 <= len(batches) <= 4 or int(trace["additional_reads"]) != len(batches) - 1:
            raise ValueError(f"E1 integrity: invalid read count for {question_id}")
        initial = batches[0]
        expected_start = max(0.0, query_time - 5.0)
        if abs(float(initial["start_sec"]) - expected_start) > 1e-6:
            raise ValueError(f"E1 integrity: non-fixed initial view for {question_id}")
        if abs(float(initial["end_sec"]) - query_time) > 1e-6:
            raise ValueError(f"E1 integrity: initial view does not end at query time for {question_id}")
        if len(trace["steps"]) != len(batches):
            raise ValueError(f"E1 integrity: decisions and views disagree for {question_id}")
        if not trace.get("terminated_by_invalid_request") and trace["steps"][-1]["request"] is not None:
            raise ValueError(f"E1 integrity: completed trace has no final null request for {question_id}")
        for batch in batches:
            start = float(batch["start_sec"])
            end = float(batch["end_sec"])
            if not 0.0 <= start < end <= query_time + 1e-6:
                raise ValueError(f"E1 integrity: invalid frame batch for {question_id}")
            timestamps: set[float] = set()
            for frame in batch["frames"]:
                timestamp = float(frame["timestamp_sec"])
                if not start - 1e-6 <= timestamp <= end + 1e-6 or timestamp > query_time + 1e-6:
                    raise ValueError(f"E1 integrity: out-of-range frame for {question_id}")
                if timestamp in timestamps:
                    raise ValueError(f"E1 integrity: duplicate frame within view for {question_id}")
                timestamps.add(timestamp)
                if not Path(frame["path"]).is_file() or sha256_file(Path(frame["path"])) != frame["sha256"]:
                    raise ValueError(f"E1 integrity: missing or corrupt frame for {question_id}")
                frame_count += 1
            sheet_frame_indices = [
                int(tile["frame_index"])
                for sheet in batch.get("contact_sheets", [])
                for tile in sheet["tiles"]
            ]
            if sheet_frame_indices != list(range(len(batch["frames"]))):
                raise ValueError(f"E1 integrity: contact-sheet frame coverage drift for {question_id}")
            if not 1 <= len(batch.get("contact_sheets", [])) <= 4:
                raise ValueError(f"E1 integrity: contact-sheet count drift for {question_id}")
            for sheet in batch["contact_sheets"]:
                if not Path(sheet["path"]).is_file() or sha256_file(Path(sheet["path"])) != sheet["sha256"]:
                    raise ValueError(f"E1 integrity: corrupt contact sheet for {question_id}")
                contact_sheet_count += 1
    audit = {
        "timestamp": utc_now(),
        "status": "passed",
        "scope": "automated E1 artifact, temporal-boundary, and provenance checks",
        "selected_ids": 100,
        "reserved_ids": 400,
        "reserved_items_evaluated": False,
        "model_inputs_contain_annotated_start_fields": False,
        "model_inputs_contain_gold_fields": False,
        "completed_traces": len(traces),
        "frame_records_rehashed": frame_count,
        "contact_sheets_rehashed": contact_sheet_count,
        "reference_split_sha256": sha256_file(Path(config["reference"]["e0_split"])),
        "reference_reserved_sha256": sha256_file(Path(config["reference"]["e0_reserved_ids"])),
    }
    atomic_write_json(run_root / "integrity_audit.json", audit)
    return audit


def setup_logging(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(run_root / "run.log"), logging.StreamHandler(sys.stderr)],
        force=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "report", "all"), default="all")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    run_root = Path(config["outputs"]["run_root"])
    setup_logging(run_root)
    queries, split = prepare(config)
    if args.stage == "prepare":
        return 0
    failures: list[dict[str, Any]] = []
    if args.stage in {"run", "all"}:
        api_key, credential_source = api_key_from_model_config(config["model"])
        if not api_key:
            raise RuntimeError(
                f"required model credential is unavailable: {config['model']['credential_environment_variable']}"
            )
        client = OpenAIOnlineWhyClient(
            api_key=api_key,
            base_url=config["model"]["base_url"],
            timeout=float(config["execution"]["request_timeout_seconds"]),
            max_retries=int(config["execution"]["max_retries"]),
        )
        traces, failures = execute(config, queries, client, args.limit)
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        manifest["last_execution"] = {
            "timestamp": utc_now(),
            "limit": args.limit,
            "completed_trace_count": len(traces),
            "failure_count": len(failures),
            "credential_source": credential_source,
            "credential_value_recorded": False,
        }
        atomic_write_json(run_root / "manifest.json", manifest)
    if args.stage in {"report", "all"}:
        order = [str(query["question_id"]) for query in queries]
        traces = _read_traces(run_root, order)
        failure_path = run_root / "failures.jsonl"
        if failure_path.is_file():
            failures = [json.loads(line) for line in failure_path.read_text().splitlines() if line]
        generate_report(
            run_root,
            queries,
            split["scoring_labels"],
            traces,
            failures,
            e0_c1_reference(config),
        )
        verify_integrity(config, queries, traces)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
