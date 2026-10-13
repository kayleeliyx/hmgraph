"""Command-line orchestration for the standalone HD-EPIC E0 pilot."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import (
    PROTOCOL_VERSION,
    RETRIEVAL_V2_VERSION,
    RETRIEVAL_VERSION,
    SELF_KNOWLEDGE_PROMPT_VERSION,
)
from .dataset import (
    atomic_write_json,
    atomic_write_jsonl,
    build_split,
    load_official_annotation,
    sha256_file,
    stable_hash,
    validate_persisted_split,
)
from .frames import extract_many
from .inference import (
    AzureOpenAIChatClient,
    BedrockConverseClient,
    BedrockResponsesClient,
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
    ResponsesClient,
    api_key_from_model_config,
    build_answer_plan,
    merge_prediction_file,
    run_answer_plans,
    utc_now,
)
from .knowledge import (
    build_knowledge_request,
    generate_self_or_reuse,
    retrieve_external_or_reuse,
    unavailable_external_record,
)
from .report import CONDITIONS, generate_reports, read_predictions

LOGGER = logging.getLogger("e0")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_resolved_config(path: Path) -> dict[str, Any]:
    root = project_root()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E0 config must be a YAML object")
    config = json.loads(json.dumps(raw))
    config["dataset"]["json_path"] = str(_resolve_path(root, config["dataset"]["json_path"]))
    config["dataset"]["video_roots"] = [
        str(_resolve_path(root, value)) for value in config["dataset"]["video_roots"]
    ]
    config["dataset"]["video_overrides"] = {
        key: str(_resolve_path(root, value)) for key, value in config["dataset"].get("video_overrides", {}).items()
    }
    for key in ("run_root", "analysis_root", "log_path"):
        config["outputs"][key] = str(_resolve_path(root, config["outputs"][key]))
    config["retrieval"]["corpus_cache_location"] = str(
        _resolve_path(root, config["retrieval"]["corpus_cache_location"])
    )
    if config["experiment"].get("variant") == "c3_v2":
        config["reference"]["run_root"] = str(_resolve_path(root, config["reference"]["run_root"]))
    config["config_path"] = str(path.resolve())
    config["project_root"] = str(root)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    variant = str(config["experiment"].get("variant", "base"))
    expected_name = "hd_epic_why_dev100_c3_v2" if variant == "c3_v2" else "hd_epic_why_pilot"
    if config["experiment"]["name"] != expected_name:
        raise ValueError(f"canonical E0 output name for {variant} must be {expected_name}")
    if int(config["experiment"]["development_sample_size"]) != 100:
        raise ValueError("this E0 configuration requires exactly 100 development questions")
    if int(config["frames"]["count"]) != 16:
        raise ValueError("E0 v1 requires 16 target frames")
    if config["model"]["provider"] not in {
        "openai_responses",
        "openai_chat_completions",
        "azure_openai_chat_completions",
        "bedrock_converse",
        "bedrock_responses",
    }:
        raise ValueError("unsupported E0 model provider")
    if config["model"]["provider"] == "openai_chat_completions":
        base_url = str(config["model"].get("base_url", ""))
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OpenAI-compatible Chat Completions requires an HTTP(S) base URL")
    if config["model"]["provider"] == "azure_openai_chat_completions":
        if not str(config["model"].get("azure_endpoint", "")).startswith("https://"):
            raise ValueError("Azure OpenAI requires an HTTPS endpoint")
        if not config["model"].get("api_version"):
            raise ValueError("Azure OpenAI requires an API version")
    if config["model"]["provider"] in {"bedrock_responses", "bedrock_converse"} and not config["model"].get("region"):
        raise ValueError("Amazon Bedrock requires a region")
    if config["generation"]["prompt_version"] != SELF_KNOWLEDGE_PROMPT_VERSION:
        raise ValueError("configured self-knowledge prompt version disagrees with source")
    expected_retrieval = RETRIEVAL_V2_VERSION if variant == "c3_v2" else RETRIEVAL_VERSION
    if config["retrieval"]["version"] != expected_retrieval:
        raise ValueError(f"configured retrieval version must be {expected_retrieval}")
    if variant == "c3_v2":
        if int(config["retrieval"]["extract_chars"]) > 1200:
            raise ValueError("C3-v2 MediaWiki TextExtracts exchars must not exceed 1200")
        if int(config["retrieval"]["fact_count"]) != 3:
            raise ValueError("C3-v2 selector is fixed to at most three retained facts")
    if variant == "c3_v2" and not Path(config["reference"]["run_root"]).is_dir():
        raise FileNotFoundError("C3-v2 reference run root is unavailable")
    if int(config["execution"]["concurrency"]) < 1:
        raise ValueError("execution concurrency must be positive")
    annotation = Path(config["dataset"]["json_path"])
    if not annotation.is_file():
        raise FileNotFoundError(
            f"official annotation is absent at the configured path: {annotation}; "
            "automatic dataset switching is disabled"
        )


def _source_fingerprint(root: Path, config_path: Path) -> str:
    paths = sorted((root / "code" / "e0").glob("*.py"))
    paths += [root / "scripts" / "run_e0.sh", config_path]
    payload = []
    for path in paths:
        if path.is_file():
            payload.append({"path": str(path.relative_to(root)), "sha256": stable_hash(path.read_text(encoding="utf-8"))})
    return stable_hash(payload)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=root, check=True, capture_output=True, text=True
        ).stdout
        return {"revision": revision, "working_tree_dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"revision": None, "working_tree_dirty": None}


def _material_protocol(config: Mapping[str, Any], annotation_sha256: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_sha256": annotation_sha256,
        "experiment": config["experiment"],
        "video_roots": config["dataset"]["video_roots"],
        "video_overrides": config["dataset"]["video_overrides"],
        "frames": config["frames"],
        "model": {
            key: config["model"][key]
            for key in (
                "provider",
                "identifier",
                "base_url",
                "azure_endpoint",
                "api_version",
                "region",
            )
            if key in config["model"]
        },
        "generation": config["generation"],
        "answer": config["answer"],
        "retrieval": config["retrieval"],
    }


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _completed_model_artifact_counts(run_root: Path) -> dict[str, int]:
    predictions = 0
    for path in (run_root / "predictions").glob("C*.jsonl"):
        predictions += sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    c2_records = 0
    c2_index = run_root / "cache" / "knowledge" / "C2_index.json"
    if c2_index.is_file():
        c2_records = len(json.loads(c2_index.read_text(encoding="utf-8")).get("items", {}))
    return {"predictions": predictions, "C2_records": c2_records}


def prepare(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(config["project_root"])
    run_root = Path(config["outputs"]["run_root"])
    split_root = run_root / "splits"
    for path in (
        split_root,
        run_root / "cache" / "frames",
        run_root / "cache" / "knowledge",
        run_root / "cache" / "responses",
        run_root / "predictions",
        Path(config["outputs"]["analysis_root"]),
    ):
        path.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(config["dataset"]["json_path"])
    annotation_items, annotation = load_official_annotation(annotation_path)
    material = _material_protocol(config, annotation["sha256"])
    material_fingerprint = stable_hash(material)
    resolved_path = run_root / "config.resolved.json"
    protocol_replaced = False
    previous_protocol_fingerprint: str | None = None
    if resolved_path.is_file():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous.get("material_protocol_fingerprint") != material_fingerprint:
            counts = _completed_model_artifact_counts(run_root)
            if any(counts.values()):
                raise RuntimeError(
                    "canonical E0 run directory contains a materially incompatible protocol "
                    f"and completed model artifacts {counts}; refusing to mix results"
                )
            previous_protocol_fingerprint = previous.get("material_protocol_fingerprint")
            history_path = run_root / "protocol_history.json"
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
            history.append(
                {
                    "replaced_at": utc_now(),
                    "reason": "provider protocol replaced after zero successful predictions and zero C2 records",
                    "completed_model_artifacts": counts,
                    "previous_material_protocol_fingerprint": previous_protocol_fingerprint,
                    "replacement_material_protocol_fingerprint": material_fingerprint,
                    "previous_resolved_config": previous,
                }
            )
            atomic_write_json(history_path, history)
            protocol_replaced = True
            LOGGER.warning("replacing unsuccessful model protocol; previous metadata retained at %s", history_path)

    sample_size = int(config["experiment"]["development_sample_size"])
    dev_path = split_root / f"dev{sample_size}.json"
    reserved_path = split_root / "reserved_ids.json"
    if dev_path.is_file():
        split = json.loads(dev_path.read_text(encoding="utf-8"))
        validate_persisted_split(
            split,
            annotation["sha256"],
            int(config["experiment"]["seed"]),
            int(config["experiment"]["development_sample_size"]),
        )
        if not reserved_path.is_file():
            raise RuntimeError(f"persisted dev{sample_size} exists without reserved_ids.json")
        reserved = json.loads(reserved_path.read_text(encoding="utf-8"))
        if "source_clips" not in reserved:
            reserved["source_clips"] = {
                question_id: {
                    "participant": annotation_items[question_id]["participant"],
                    "source_video_id": annotation_items[question_id]["source_video_id"],
                    "source_clip_key": annotation_items[question_id]["source_clip_key"],
                    "start_sec": annotation_items[question_id]["start_sec"],
                    "end_sec": annotation_items[question_id]["end_sec"],
                }
                for question_id in reserved["question_ids"]
            }
            atomic_write_json(reserved_path, reserved)
        split_metadata = {
            "annotation": annotation,
            "participant_distribution": split["participant_distribution"],
            "extra_participants": split["extra_participants"],
            "duplicate_selected_clips": _duplicate_clips(split["records"]),
        }
        reused_split = True
    else:
        split, reserved, split_metadata = build_split(
            annotation_path=annotation_path,
            video_roots=[Path(value) for value in config["dataset"]["video_roots"]],
            overrides=config["dataset"]["video_overrides"],
            seed=int(config["experiment"]["seed"]),
            sample_size=int(config["experiment"]["development_sample_size"]),
        )
        # Persist the exact sample before any expensive frame, retrieval, or model operation.
        atomic_write_json(dev_path, split)
        atomic_write_json(reserved_path, reserved)
        reused_split = False

    for record in split["records"]:
        if not Path(record["source_video_path"]).is_file():
            raise FileNotFoundError(
                f"saved sample source video is no longer resolvable: {record['source_video_path']}"
            )
    if set(reserved["question_ids"]) & {record["question_id"] for record in split["records"]}:
        raise ValueError("reserved IDs overlap the saved development split")

    source_fingerprint = _source_fingerprint(root, Path(config["config_path"]))
    api_key, credential_source = api_key_from_model_config(config["model"])
    resolved = json.loads(json.dumps(config))
    resolved.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "annotation_sha256": annotation["sha256"],
            "material_protocol_fingerprint": material_fingerprint,
            "e0_source_fingerprint": source_fingerprint,
            "credential_configured": bool(api_key),
            "credential_source": credential_source,
            "credential_value_recorded": False,
        }
    )
    atomic_write_json(resolved_path, resolved)

    manifest_path = run_root / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if (
        previous_manifest
        and previous_manifest.get("material_protocol_fingerprint") != material_fingerprint
        and not protocol_replaced
    ):
        raise RuntimeError("existing E0 manifest is materially incompatible")
    retained_initial_source = previous_manifest.get("e0_source_fingerprint_initial", source_fingerprint)
    if protocol_replaced:
        previous_manifest = {}
    manifest = {
        **previous_manifest,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": config["experiment"]["name"],
        "material_protocol_fingerprint": material_fingerprint,
        "e0_source_fingerprint_initial": retained_initial_source,
        "e0_source_fingerprint_current": source_fingerprint,
        "annotation": annotation,
        "git": _git_state(root),
        "dev_split_path": str(dev_path.resolve()),
        "reserved_split_path": str(reserved_path.resolve()),
        "selected_count": len(split["records"]),
        "reserved_count": len(reserved["question_ids"]),
        "participant_distribution": split_metadata["participant_distribution"],
        "extra_participants": split_metadata["extra_participants"],
        "duplicate_selected_clips": split_metadata["duplicate_selected_clips"],
        "sample_reused": reused_split,
        "reserved_items_evaluated": False,
        "blockers": previous_manifest.get("blockers", []),
        "protocol_replaced_from": previous_protocol_fingerprint,
        "last_prepared_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    LOGGER.info(
        "prepared saved dev%d (reused=%s), participants=%s, reserved=%d",
        sample_size,
        reused_split,
        split["participant_distribution"],
        len(reserved["question_ids"]),
    )
    return split, manifest, resolved


def _duplicate_clips(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(str(record["source_clip_key"]), []).append(str(record["question_id"]))
    return {key: value for key, value in grouped.items() if len(value) > 1}


def _load_record_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for question_id, entry in payload.get("items", {}).items():
        if "record" in entry:
            result[question_id] = entry["record"]
            continue
        record_path = Path(entry["record_path"])
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["_record_path"] = str(record_path.resolve())
            result[question_id] = record
    return result


def _write_frame_index(path: Path, existing: Mapping[str, Any], new: Mapping[str, Mapping[str, Any]]) -> None:
    items: dict[str, Any] = {}
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"items": {}}
    items.update(previous.get("items", {}))
    for question_id, record in new.items():
        record_path = Path(record["frames"][0]["path"]).parent / "record.json"
        items[question_id] = {
            "cache_id": record["cache_id"],
            "record_path": str(record_path.resolve()),
        }
    atomic_write_json(path, {"items": items})


def _write_knowledge_index(path: Path, existing: Mapping[str, Any], new: Mapping[str, Mapping[str, Any]]) -> None:
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"items": {}}
    items = dict(previous.get("items", {}))
    for question_id, record in new.items():
        # Index records are small audit records; raw retrieved documents remain in the
        # content-addressed cache and are omitted here.
        indexed = {key: value for key, value in record.items() if key != "documents"}
        items[question_id] = {"record": indexed}
    atomic_write_json(path, {"items": items})


def _generate_c2_many(
    records: Sequence[Mapping[str, Any]],
    cache_root: Path,
    config: Mapping[str, Any],
    client: ResponsesClient,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str]]:
    by_action: dict[str, list[str]] = {}
    request_by_action: dict[str, Any] = {}
    for record in records:
        request = build_knowledge_request(record)
        by_action.setdefault(request.action_phrase, []).append(str(record["question_id"]))
        request_by_action[request.action_phrase] = request
    results: dict[str, dict[str, Any]] = {}
    stats = {"created": 0, "reused": 0, "failed": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=int(config["execution"]["concurrency"])) as executor:
        futures = {
            executor.submit(
                generate_self_or_reuse,
                request,
                cache_root,
                config["model"]["provider"],
                config["model"]["identifier"],
                config["generation"],
                client.generate_facts,
            ): action
            for action, request in request_by_action.items()
        }
        for future in as_completed(futures):
            action = futures[future]
            try:
                knowledge, was_reused = future.result()
                stats["reused" if was_reused else "created"] += 1
                for question_id in by_action[action]:
                    results[question_id] = knowledge
            except Exception as exc:
                stats["failed"] += 1
                errors.append(f"{exc.__class__.__name__}: {str(exc)[:300]}")
    return results, stats, errors


def _retrieve_c3_many(
    records: Sequence[Mapping[str, Any]], cache_root: Path, settings: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str]]:
    by_action: dict[str, list[str]] = {}
    request_by_action: dict[str, Any] = {}
    for record in records:
        request = build_knowledge_request(record)
        by_action.setdefault(request.action_phrase, []).append(str(record["question_id"]))
        request_by_action[request.action_phrase] = request
    results: dict[str, dict[str, Any]] = {}
    stats = {"created": 0, "reused": 0, "failed": 0, "weak_or_empty": 0}
    errors: list[str] = []
    for action, request in request_by_action.items():
        try:
            knowledge, was_reused = retrieve_external_or_reuse(request, cache_root, settings)
            stats["reused" if was_reused else "created"] += 1
            stats["weak_or_empty"] += int(knowledge.get("status") != "available")
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {str(exc)[:500]}"
            errors.append(error)
            knowledge = unavailable_external_record(request, settings, error)
            stats["failed"] += 1
        for question_id in by_action[action]:
            results[question_id] = knowledge
    return results, stats, errors


def _append_attempt_failures(path: Path, failures: Sequence[Mapping[str, Any]]) -> None:
    if not failures:
        return
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    atomic_write_jsonl(path, [*existing, *failures])


def execute_run(
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    manifest: dict[str, Any],
    conditions: Sequence[str],
    limit: int | None,
) -> tuple[dict[str, Any], int]:
    run_root = Path(config["outputs"]["run_root"])
    records = list(split["records"][:limit] if limit is not None else split["records"])
    reserved_ids = set(json.loads((run_root / "splits" / "reserved_ids.json").read_text(encoding="utf-8"))["question_ids"])
    if any(record["question_id"] in reserved_ids for record in records):
        raise AssertionError("a reserved question entered E0 execution")
    LOGGER.info("run stage: %d saved-manifest items, conditions=%s", len(records), list(conditions))
    blockers: list[str] = []
    operation_stats: dict[str, Any] = {}

    frame_index_path = run_root / "cache" / "frames" / "index.json"
    frames = _load_record_index(frame_index_path)
    if any(condition != "C0" for condition in conditions):
        new_frames, frame_stats = extract_many(
            records,
            run_root / "cache" / "frames",
            {"count": config["frames"]["count"], **config["frames"]["preprocessing"]},
        )
        frames.update(new_frames)
        _write_frame_index(frame_index_path, frames, new_frames)
        operation_stats["frames"] = frame_stats
        LOGGER.info("frames complete: %s", frame_stats)

    c2_index_path = run_root / "cache" / "knowledge" / "C2_index.json"
    c3_index_path = run_root / "cache" / "knowledge" / "C3_index.json"
    c2 = _load_record_index(c2_index_path)
    c3 = _load_record_index(c3_index_path)
    api_key, credential_source = api_key_from_model_config(config["model"])
    client: ResponsesClient | None = None
    if api_key and config["model"]["provider"] == "openai_responses":
        client = OpenAIResponsesClient(
            api_key,
            float(config["execution"]["request_timeout_seconds"]),
            int(config["execution"]["max_retries"]),
        )
    elif api_key and config["model"]["provider"] == "openai_chat_completions":
        client = OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=config["model"]["base_url"],
            timeout=float(config["execution"]["request_timeout_seconds"]),
            max_retries=int(config["execution"]["max_retries"]),
        )
    elif api_key and config["model"]["provider"] == "azure_openai_chat_completions":
        client = AzureOpenAIChatClient(
            api_key=api_key,
            azure_endpoint=config["model"]["azure_endpoint"],
            api_version=config["model"]["api_version"],
            timeout=float(config["execution"]["request_timeout_seconds"]),
            max_retries=int(config["execution"]["max_retries"]),
        )
    elif api_key and config["model"]["provider"] == "bedrock_responses":
        client = BedrockResponsesClient(
            api_key=api_key,
            region=config["model"]["region"],
            timeout=float(config["execution"]["request_timeout_seconds"]),
            max_retries=int(config["execution"]["max_retries"]),
        )
    elif api_key and config["model"]["provider"] == "bedrock_converse":
        client = BedrockConverseClient(
            api_key=api_key,
            region=config["model"]["region"],
            timeout=float(config["execution"]["request_timeout_seconds"]),
            max_retries=int(config["execution"]["max_retries"]),
        )
    credential_hint = str(config["model"].get("credential_environment_variable", "API credential"))

    if "C3" in conditions:
        new_c3, c3_stats, c3_errors = _retrieve_c3_many(
            records,
            run_root / "cache" / "knowledge",
            {**config["retrieval"], "max_retries": config["execution"]["max_retries"]},
        )
        c3.update(new_c3)
        _write_knowledge_index(c3_index_path, c3, new_c3)
        operation_stats["C3_knowledge"] = c3_stats
        if c3_errors:
            blockers.append(f"C3 external retrieval unavailable: {c3_errors[0]}")
        LOGGER.info("C3 retrieval complete: %s", c3_stats)

    if "C2" in conditions:
        if client is None:
            blockers.append(
                f"C2 self-knowledge generation is blocked because {credential_hint} and the configured credential source are unavailable."
            )
            operation_stats["C2_knowledge"] = {"created": 0, "reused": 0, "blocked": len(records)}
        else:
            new_c2, c2_stats, c2_errors = _generate_c2_many(
                records, run_root / "cache" / "knowledge", config, client
            )
            c2.update(new_c2)
            _write_knowledge_index(c2_index_path, c2, new_c2)
            operation_stats["C2_knowledge"] = c2_stats
            if c2_errors:
                blockers.append(f"C2 self-knowledge generation failed for {len(c2_errors)} unique actions: {c2_errors[0]}")
            LOGGER.info("C2 knowledge complete: %s", c2_stats)

    plans: list[dict[str, Any]] = []
    missing_dependencies = CounterLike()
    for record in records:
        question_id = str(record["question_id"])
        for condition in conditions:
            try:
                plans.append(
                    build_answer_plan(
                        record=record,
                        condition=condition,
                        provider=config["model"]["provider"],
                        model=config["model"]["identifier"],
                        settings={
                            **config["answer"],
                            "image_detail": config["frames"]["image_detail"],
                        },
                        frame_record=frames.get(question_id) if condition != "C0" else None,
                        knowledge_record=c2.get(question_id) if condition == "C2" else (c3.get(question_id) if condition == "C3" else None),
                    )
                )
            except ValueError as exc:
                missing_dependencies.add(f"{condition}: {exc}")
    if missing_dependencies.total:
        blockers.append(
            "Answer requests skipped for missing condition inputs: "
            + "; ".join(f"{key} ({count})" for key, count in missing_dependencies.items())
        )

    prediction_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if client is None:
        blockers.append(
            f"C0-C3 answer inference is blocked because {credential_hint} and the configured credential source are unavailable."
        )
        answer_stats = {"reused": 0, "created": 0, "blocked": len(plans), "failed": 0}
    else:
        prediction_rows, answer_stats, failures = run_answer_plans(
            plans,
            run_root / "cache" / "responses",
            client,
            int(config["execution"]["concurrency"]),
        )
        if failures:
            blockers.append(f"answer inference failed for {len(failures)} item-condition requests")
    operation_stats["answers"] = answer_stats
    _append_attempt_failures(run_root / "cache" / "responses" / "attempts.jsonl", failures)
    manifest_order = [str(record["question_id"]) for record in split["records"]]
    for condition in conditions:
        condition_rows = [row for row in prediction_rows if row["condition"] == condition]
        merge_prediction_file(run_root / "predictions" / f"{condition}.jsonl", manifest_order, condition_rows)

    manifest["blockers"] = list(dict.fromkeys(blockers))
    manifest["last_execution"] = {
        "timestamp": utc_now(),
        "limit": limit,
        "item_count": len(records),
        "conditions": list(conditions),
        "credential_configured": bool(api_key),
        "credential_source": credential_source,
        "operation_stats": operation_stats,
        "prediction_failures": len(failures),
        "reserved_items_evaluated": False,
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    LOGGER.info("answer stage complete: %s; blockers=%d", answer_stats, len(blockers))
    return manifest, 0 if not blockers else 3


class CounterLike:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}

    def add(self, key: str) -> None:
        self.data[key] = self.data.get(key, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.data.values())

    def items(self):
        return self.data.items()


def _load_all_indexes(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(config["outputs"]["run_root"])
    return (
        _load_record_index(run_root / "cache" / "frames" / "index.json"),
        _load_record_index(run_root / "cache" / "knowledge" / "C2_index.json"),
        _load_record_index(run_root / "cache" / "knowledge" / "C3_index.json"),
    )


def verify_integrity(
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    c2: Mapping[str, Mapping[str, Any]],
    c3: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    run_root = Path(config["outputs"]["run_root"])
    records = {str(record["question_id"]): record for record in split["records"]}
    expected_ids = set(records)
    expected_count = int(config["experiment"]["development_sample_size"])
    if len(expected_ids) != expected_count or len(split["scoring_labels"]) != expected_count:
        raise ValueError(
            f"integrity check: development record/label count is not exactly {expected_count}"
        )
    for question_id, record in records.items():
        label = split["scoring_labels"][question_id]
        displayed = record["option_order"].index(label["original_correct_idx"])
        if displayed != label["displayed_correct_idx"] or "ABCDE"[displayed] != label["gold_letter"]:
            raise ValueError(f"integrity check: displayed gold remapping drift for {question_id}")
    reserved = set(json.loads((run_root / "splits" / "reserved_ids.json").read_text(encoding="utf-8"))["question_ids"])
    expected_reserved = 500 - expected_count
    if expected_ids & reserved or len(reserved) != expected_reserved:
        raise ValueError("integrity check: selected/reserved split overlap or count mismatch")
    if set(frames) != expected_ids:
        raise ValueError(
            f"integrity check: expected {expected_count} frame records, found {len(frames)}"
        )
    frame_file_count = 0
    for question_id, frame_record in frames.items():
        record = records[question_id]
        selected = frame_record.get("frames", [])
        if not 1 <= len(selected) <= int(config["frames"]["count"]):
            raise ValueError(f"integrity check: invalid selected frame count for {question_id}")
        if len({frame["timestamp_sec"] for frame in selected}) != len(selected):
            raise ValueError(f"integrity check: duplicate selected frame timestamp for {question_id}")
        if [frame["timestamp_sec"] for frame in selected] != sorted(frame["timestamp_sec"] for frame in selected):
            raise ValueError(f"integrity check: nonchronological frames for {question_id}")
        for frame in selected:
            timestamp = float(frame["timestamp_sec"])
            if not float(record["start_sec"]) - 1e-6 <= timestamp <= float(record["end_sec"]) + 1e-6:
                raise ValueError(f"integrity check: out-of-interval frame for {question_id}")
            path = Path(frame["path"])
            if not path.is_file() or sha256_file(path) != frame["sha256"]:
                raise ValueError(f"integrity check: missing or corrupt frame file for {question_id}")
            frame_file_count += 1
    if set(c3) != expected_ids:
        raise ValueError(
            f"integrity check: expected {expected_count} C3 item records, found {len(c3)}"
        )
    weak_c3 = 0
    for question_id, knowledge in c3.items():
        if set(knowledge.get("request", {})) != {"action_phrase", "kitchen_context", "prompt_version"}:
            raise ValueError(f"integrity check: C3 request boundary violation for {question_id}")
        facts = knowledge.get("facts", [])
        provenance = knowledge.get("fact_provenance", [])
        if len(facts) > 5 or len(facts) != len(provenance):
            raise ValueError(f"integrity check: C3 fact/provenance count mismatch for {question_id}")
        if knowledge.get("status") == "available" and not 3 <= len(facts) <= 5:
            raise ValueError(f"integrity check: available C3 record has too few facts for {question_id}")
        for fact, source in zip(facts, provenance):
            if source.get("supporting_span") != fact or not str(source.get("url", "")).startswith("https://"):
                raise ValueError(f"integrity check: unsupported C3 fact for {question_id}")
            if source.get("revision_id") is None or not source.get("query"):
                raise ValueError(f"integrity check: incomplete C3 source identity for {question_id}")
        weak_c3 += int(knowledge.get("status") == "weak")
    for question_id, knowledge in c2.items():
        if set(knowledge.get("request", {})) != {"action_phrase", "kitchen_context", "prompt_version"}:
            raise ValueError(f"integrity check: C2 request boundary violation for {question_id}")

    prediction_counts: dict[str, int] = {}
    for condition in CONDITIONS:
        predictions = read_predictions(run_root / "predictions" / f"{condition}.jsonl")
        if not set(predictions) <= expected_ids or set(predictions) & reserved:
            raise ValueError(f"integrity check: {condition} contains a non-development or reserved ID")
        for question_id, prediction in predictions.items():
            record = records[question_id]
            if prediction.get("displayed_choices") != record["displayed_choices"] or prediction.get("option_order") != record["option_order"]:
                raise ValueError(f"integrity check: option drift in {condition}/{question_id}")
            if condition == "C0":
                if prediction.get("ordered_frame_payload_hash") is not None or prediction.get("knowledge_cache_id") is not None:
                    raise ValueError(f"integrity check: C0 received forbidden visual/knowledge input for {question_id}")
            else:
                if prediction.get("ordered_frame_payload_hash") != frames[question_id]["ordered_frame_payload_hash"]:
                    raise ValueError(f"integrity check: visual payload drift in {condition}/{question_id}")
                if condition == "C1" and prediction.get("knowledge_cache_id") is not None:
                    raise ValueError(f"integrity check: C1 received knowledge for {question_id}")
                expected_knowledge = c2.get(question_id) if condition == "C2" else c3.get(question_id)
                if condition in {"C2", "C3"} and (
                    expected_knowledge is None
                    or prediction.get("knowledge_cache_id") != expected_knowledge.get("cache_id")
                ):
                    raise ValueError(f"integrity check: knowledge identity drift in {condition}/{question_id}")
                if condition == "C3" and not 3 <= len(expected_knowledge.get("facts", [])) <= 5:
                    raise ValueError(
                        f"integrity check: C3 prediction used fewer than 3 facts for {question_id}"
                    )
        prediction_counts[condition] = len(predictions)
    return {
        "status": "passed",
        "scope": "automated artifact/provenance checks; not a human audit",
        "selected_ids": len(expected_ids),
        "reserved_ids": len(reserved),
        "frame_records": len(frames),
        "frame_files_rehashed": frame_file_count,
        "C3_item_records": len(c3),
        "C3_weak_item_records": weak_c3,
        "prediction_counts": prediction_counts,
    }


def report(config: Mapping[str, Any], split: Mapping[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    frames, c2, c3 = _load_all_indexes(config)
    manifest["automated_integrity_checks"] = verify_integrity(config, split, frames, c2, c3)
    atomic_write_json(Path(config["outputs"]["run_root"]) / "manifest.json", manifest)
    metrics = generate_reports(
        analysis_root=Path(config["outputs"]["analysis_root"]),
        run_root=Path(config["outputs"]["run_root"]),
        split=split,
        manifest=manifest,
        resolved_config=config,
        frames=frames,
        c2=c2,
        c3=c3,
    )
    LOGGER.info("reports generated at %s", config["outputs"]["analysis_root"])
    return metrics


def dry_run_plan(
    config: Mapping[str, Any], split: Mapping[str, Any], conditions: Sequence[str], limit: int | None
) -> None:
    records = list(split["records"][:limit] if limit is not None else split["records"])
    visual_count = len(records) if any(condition != "C0" for condition in conditions) else 0
    unique_actions = len({record["action_phrase"] for record in records})
    payload = {
        "dry_run": True,
        "resolved_paths": {
            "project_root": config["project_root"],
            "annotation": config["dataset"]["json_path"],
            "video_roots": config["dataset"]["video_roots"],
            "run_root": config["outputs"]["run_root"],
            "analysis_root": config["outputs"]["analysis_root"],
            "log_path": config["outputs"]["log_path"],
        },
        "saved_manifest_items": len(split["records"]),
        "planned_item_count": len(records),
        "planned_question_ids": [record["question_id"] for record in records],
        "conditions": list(conditions),
        "expected_operations": {
            "clip_frame_records": visual_count,
            "unique_C2_fact_requests": unique_actions if "C2" in conditions else 0,
            "unique_C3_retrievals": unique_actions if "C3" in conditions else 0,
            "answer_requests": len(records) * len(conditions),
            "paid_model_calls_made_by_dry_run": 0,
        },
        "provider": config["model"]["provider"],
        "model": config["model"]["identifier"],
        "credential_configured": bool(api_key_from_model_config(config["model"])[0]),
        "credential_source": api_key_from_model_config(config["model"])[1],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "configs" / "e0_pilot.yaml")
    parser.add_argument("--stage", choices=("prepare", "retrieve", "run", "report", "all"), default="all")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error("--limit must be from 1 through 100 and always slices the saved dev100 manifest")
    # Preserve user order while preventing accidental duplicate paid requests.
    if args.conditions is not None:
        args.conditions = list(dict.fromkeys(args.conditions))
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_resolved_config(args.config)
    _setup_logging(Path(config["outputs"]["log_path"]))
    if config["experiment"].get("variant") == "c3_v2":
        from .c3_v2 import execute_c3_v2

        return execute_c3_v2(config, args, LOGGER)
    if args.stage == "retrieve":
        raise SystemExit("--stage retrieve is only available for the C3-v2 development revision")
    conditions = args.conditions or list(CONDITIONS)
    LOGGER.info("E0 start stage=%s dry_run=%s", args.stage, args.dry_run)
    try:
        split, manifest, resolved = prepare(config)
        if args.dry_run:
            dry_run_plan(resolved, split, conditions, args.limit)
            return 0
        exit_code = 0
        if args.stage in {"run", "all"}:
            manifest, exit_code = execute_run(resolved, split, manifest, conditions, args.limit)
        if args.stage in {"report", "all"}:
            report(resolved, split, manifest)
        LOGGER.info("E0 finished with exit_code=%d", exit_code)
        return exit_code
    except Exception as exc:
        LOGGER.exception("E0 failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
