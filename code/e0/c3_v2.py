"""Isolated C3 retrieval revision over the immutable E0 dev100 reference run."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import PROTOCOL_VERSION, RETRIEVAL_V2_VERSION
from .dataset import atomic_write_json, atomic_write_jsonl, sha256_file, stable_hash
from .inference import (
    OpenAICompatibleChatClient,
    api_key_from_model_config,
    build_answer_plan,
    merge_prediction_file,
    run_answer_plans,
    utc_now,
)
from .knowledge import (
    V2_SELECTION_POLICY_FINGERPRINT,
    build_knowledge_request,
    retrieve_external_v2_or_reuse,
)
from .report import CONDITIONS, generate_c3_v2_reports, read_predictions
from .run import _git_state, _source_fingerprint


EXPECTED_ANNOTATION_SHA256 = "d22531be3f23a8c4dd182693edac2acb5b75cc5ea017228cd89716cb5b3048d4"
REFERENCE_CONDITIONS = ("C0", "C1", "C2", "C3")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame_records(reference_root: Path, split: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index_path = reference_root / "cache" / "frames" / "index.json"
    index = _read_json(index_path).get("items", {})
    records: dict[str, dict[str, Any]] = {}
    for item in split["records"]:
        question_id = str(item["question_id"])
        pointer = index.get(question_id)
        if not pointer:
            raise ValueError(f"reference frame index is missing {question_id}")
        record_path = Path(pointer["record_path"])
        if not record_path.is_file() or reference_root not in record_path.parents:
            raise ValueError(f"invalid reference frame record path for {question_id}: {record_path}")
        record = _read_json(record_path)
        if record.get("cache_id") != pointer.get("cache_id"):
            raise ValueError(f"reference frame identity mismatch for {question_id}")
        if record.get("preprocessing") is None:
            raise ValueError(f"reference frame preprocessing is absent for {question_id}")
        ordered_rows: list[dict[str, Any]] = []
        for frame in record.get("frames", []):
            timestamp = float(frame["timestamp_sec"])
            if not float(item["start_sec"]) - 1e-6 <= timestamp <= float(item["end_sec"]) + 1e-6:
                raise ValueError(f"reference frame outside interval for {question_id}: {timestamp}")
            frame_path = Path(frame["path"])
            if not frame_path.is_file() or sha256_file(frame_path) != frame["sha256"]:
                raise ValueError(f"reference JPEG hash mismatch for {question_id}: {frame_path}")
            ordered_rows.append({"timestamp_sec": timestamp, "sha256": frame["sha256"]})
        if stable_hash(ordered_rows) != record.get("ordered_frame_payload_hash"):
            raise ValueError(f"reference ordered frame hash mismatch for {question_id}")
        record["_record_path"] = str(record_path.resolve())
        records[question_id] = record
    return records


def _aggregate_frame_hashes(frames: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    record_rows: list[dict[str, str]] = []
    jpeg_rows: list[dict[str, str]] = []
    for question_id in sorted(frames):
        record = frames[question_id]
        record_rows.append(
            {"question_id": question_id, "record_sha256": sha256_file(Path(record["_record_path"]))}
        )
        for frame in record["frames"]:
            jpeg_rows.append({"path": str(frame["path"]), "sha256": str(frame["sha256"])})
    return {
        "frame_record_count": len(record_rows),
        "jpeg_count": len(jpeg_rows),
        "frame_record_set_hash": stable_hash(record_rows),
        "jpeg_set_hash": stable_hash(jpeg_rows),
    }


def _validate_reference_predictions(
    config: Mapping[str, Any],
    reference_root: Path,
    split: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    reference_config = _read_json(reference_root / "config.resolved.json")
    for key in ("provider", "identifier", "base_url"):
        if config["model"].get(key) != reference_config["model"].get(key):
            raise ValueError(f"reference/current answering model mismatch at model.{key}")
    if config["answer"] != reference_config["answer"]:
        raise ValueError("reference/current shared answering settings differ")
    if config["frames"] != reference_config["frames"]:
        raise ValueError("reference/current frame settings differ")

    ids = [str(record["question_id"]) for record in split["records"]]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in REFERENCE_CONDITIONS:
        path = reference_root / "predictions" / f"{condition}.jsonl"
        rows = read_predictions(path)
        expected_count = 100 if condition in {"C0", "C1", "C2"} else 91
        if len(rows) != expected_count or set(rows) - set(ids):
            raise ValueError(f"unexpected reference {condition} prediction population: {len(rows)}")
        for item in split["records"]:
            question_id = str(item["question_id"])
            row = rows.get(question_id)
            if row is None:
                if condition == "C3":
                    continue
                raise ValueError(f"reference {condition} is missing {question_id}")
            if row.get("displayed_choices") != item["displayed_choices"]:
                raise ValueError(f"reference option text mismatch: {condition}/{question_id}")
            if row.get("option_order") != item["option_order"]:
                raise ValueError(f"reference option order mismatch: {condition}/{question_id}")
            expected_frame_hash = None if condition == "C0" else frames[question_id]["ordered_frame_payload_hash"]
            if row.get("ordered_frame_payload_hash") != expected_frame_hash:
                raise ValueError(f"reference frame payload mismatch: {condition}/{question_id}")
            if row.get("provider") != config["model"]["provider"] or row.get("model") != config["model"]["identifier"]:
                raise ValueError(f"reference provider/model mismatch: {condition}/{question_id}")
            expected_settings = {
                **config["answer"],
                "image_detail": config["frames"]["image_detail"],
            }
            if row.get("inference_configuration") != expected_settings:
                raise ValueError(f"reference answer settings mismatch: {condition}/{question_id}")
        result[condition] = rows
    return result


def _reference_snapshot(reference_root: Path, frames: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    files = {
        "config": reference_root / "config.resolved.json",
        "manifest": reference_root / "manifest.json",
        "dev100": reference_root / "splits" / "dev100.json",
        "reserved_ids": reference_root / "splits" / "reserved_ids.json",
        "frame_index": reference_root / "cache" / "frames" / "index.json",
        **{
            f"predictions_{condition}": reference_root / "predictions" / f"{condition}.jsonl"
            for condition in REFERENCE_CONDITIONS
        },
    }
    return {
        "source_run": str(reference_root.resolve()),
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "frames": _aggregate_frame_hashes(frames),
    }


def _write_reference_frame_index(
    run_root: Path,
    reference_root: Path,
    frames: Mapping[str, Mapping[str, Any]],
) -> None:
    rows = {}
    for question_id, record in frames.items():
        record_path = Path(record["_record_path"])
        rows[question_id] = {
            "source_run": str(reference_root.resolve()),
            "cache_id": record["cache_id"],
            "record_path": str(record_path.resolve()),
            "record_sha256": sha256_file(record_path),
            "ordered_frame_payload_hash": record["ordered_frame_payload_hash"],
        }
    atomic_write_json(run_root / "cache" / "frames" / "reference_index.json", {"items": rows})


def prepare_revision(config: Mapping[str, Any], logger: logging.Logger) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(config["outputs"]["run_root"])
    analysis_root = Path(config["outputs"]["analysis_root"])
    reference_root = Path(config["reference"]["run_root"])
    for path in (
        run_root / "splits",
        run_root / "cache" / "frames",
        run_root / "cache" / "knowledge",
        run_root / "cache" / "responses",
        run_root / "predictions",
        analysis_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    split_path = reference_root / "splits" / "dev100.json"
    reserved_path = reference_root / "splits" / "reserved_ids.json"
    split = _read_json(split_path)
    reserved = _read_json(reserved_path)
    annotation_hash = sha256_file(Path(config["dataset"]["json_path"]))
    expected_hash = str(config["reference"]["expected_annotation_sha256"])
    if annotation_hash != expected_hash or annotation_hash != EXPECTED_ANNOTATION_SHA256:
        raise ValueError(f"annotation checksum mismatch: {annotation_hash}")
    if split.get("annotation_sha256") != annotation_hash or len(split.get("records", [])) != 100:
        raise ValueError("reference dev100 is not the expected annotation/sample")
    ids = [str(record["question_id"]) for record in split["records"]]
    if len(set(ids)) != 100 or len(reserved.get("question_ids", [])) != 400:
        raise ValueError("reference split cardinality or uniqueness mismatch")
    if set(ids) & set(reserved["question_ids"]):
        raise ValueError("development and reserved IDs overlap")

    frames = _load_frame_records(reference_root, split)
    baseline_predictions = _validate_reference_predictions(config, reference_root, split, frames)
    snapshot = _reference_snapshot(reference_root, frames)
    manifest_path = run_root / "manifest.json"
    old_manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    prior_snapshot = old_manifest.get("reference_artifacts")
    if prior_snapshot is not None and prior_snapshot != snapshot:
        raise RuntimeError("the immutable reference run changed since this C3-v2 revision was created")

    # These are independent copies of split metadata only. No symlink or writer in
    # the revision can mutate the reference run.
    for source, destination in (
        (split_path, run_root / "splits" / "dev100.json"),
        (reserved_path, run_root / "splits" / "reserved_ids.json"),
    ):
        if destination.is_file() and _read_json(destination) != _read_json(source):
            raise ValueError(f"revision split differs from immutable reference: {destination}")
        atomic_write_json(destination, _read_json(source))
    _write_reference_frame_index(run_root, reference_root, frames)

    material = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": "c3_v2",
        "annotation_sha256": annotation_hash,
        "reference_dev100_sha256": snapshot["files"]["dev100"]["sha256"],
        "reference_reserved_sha256": snapshot["files"]["reserved_ids"]["sha256"],
        "frames": config["frames"],
        "model": {
            key: config["model"][key]
            for key in ("provider", "identifier", "base_url")
        },
        "answer": config["answer"],
        "retrieval": config["retrieval"],
        "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
    }
    material_hash = stable_hash(material)
    resolved_path = run_root / "config.resolved.json"
    if resolved_path.is_file():
        old = _read_json(resolved_path)
        if old.get("material_protocol_fingerprint") != material_hash:
            has_results = any((run_root / "predictions").glob("*.jsonl"))
            validation_path = run_root / "cache" / "knowledge" / "retrieval_validation.json"
            validated_retrieval = (
                validation_path.is_file()
                and _read_json(validation_path).get("status") == "passed"
            )
            if has_results:
                raise RuntimeError("C3-v2 run contains incompatible material artifacts; use a new run ID")
            history_path = run_root / "protocol_history.json"
            history = _read_json(history_path) if history_path.is_file() else []
            history.append(
                {
                    "replaced_at": utc_now(),
                    "reason": (
                        "pre-inference source-only review rejected prior selector output"
                        if validated_retrieval
                        else "pre-inference retrieval validator rejected prior selector output"
                    ),
                    "previous_material_protocol_fingerprint": old.get("material_protocol_fingerprint"),
                    "replacement_material_protocol_fingerprint": material_hash,
                    "previous_retrieval_validation_status": (
                        _read_json(validation_path).get("status") if validation_path.is_file() else None
                    ),
                    "model_predictions_present": False,
                }
            )
            atomic_write_json(history_path, history)
    resolved = json.loads(json.dumps(config))
    resolved.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "annotation_sha256": annotation_hash,
            "material_protocol_fingerprint": material_hash,
            "e0_source_fingerprint": _source_fingerprint(
                Path(config["project_root"]), Path(config["config_path"])
            ),
            "credential_configured": bool(api_key_from_model_config(config["model"])[0]),
            "credential_source": api_key_from_model_config(config["model"])[1],
            "credential_value_recorded": False,
        }
    )
    atomic_write_json(resolved_path, resolved)
    manifest = {
        **old_manifest,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": config["experiment"]["name"],
        "development_revision": True,
        "material_protocol_fingerprint": material_hash,
        "annotation": {
            "path": config["dataset"]["json_path"],
            "source": config["dataset"]["official_source"],
            "sha256": annotation_hash,
            "item_count": 500,
        },
        "selected_count": 100,
        "reserved_count": 400,
        "sample_reused_from_reference": True,
        "reserved_items_evaluated": False,
        "reference_artifacts": snapshot,
        "baseline_prediction_counts": {
            condition: len(rows) for condition, rows in baseline_predictions.items()
        },
        "git": _git_state(Path(config["project_root"])),
        "last_prepared_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    logger.info(
        "prepared C3-v2 from immutable reference: 100 dev IDs, 400 reserved, "
        "%d JPEGs verified; no videos decoded and no baselines rerun",
        snapshot["frames"]["jpeg_count"],
    )
    return split, manifest, resolved


def _validate_retrieval_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(record.get("request", {})) != {"action_phrase", "kitchen_context", "prompt_version"}:
        errors.append("knowledge request boundary is not the three-field whitelist")
    queries = record.get("queries", [])
    audit = record.get("search_audit", [])
    if [row.get("query") for row in queries] != [row.get("query") for row in audit]:
        errors.append("not every planned query has an executed audit row")
    if queries and queries[0].get("kind") != "action_kitchen_context":
        errors.append("contextual search is not first")
    if any(row.get("status") not in {"hits", "no_hits", "api_error", "transport_error"} for row in audit):
        errors.append("search outcome is not explicit")
    acquisitions = record.get("page_acquisitions", [])
    allowed_acquisition = {
        "success", "disambiguation", "omitted_extract", "empty_content",
        "missing_page", "omitted_page", "api_error", "transport_error",
    }
    if any(row.get("status") not in allowed_acquisition for row in acquisitions):
        errors.append("page acquisition outcome is not explicit")
    facts = record.get("facts", [])
    provenance = record.get("fact_provenance", [])
    if len(facts) != len(provenance):
        errors.append("fact/provenance count mismatch")
    if record.get("status") == "usable" and not 1 <= len(facts) <= 5:
        errors.append("usable status requires 1-5 facts")
    if record.get("status") in {"abstained", "operational_error"} and facts:
        errors.append("non-usable status must not carry facts")
    by_source = {
        (row.get("page_id"), row.get("revision_id")): row
        for row in acquisitions
        if row.get("status") == "success"
    }
    for index, (fact, source) in enumerate(zip(facts, provenance)):
        document = by_source.get((source.get("page_id"), source.get("revision_id")))
        span = str(source.get("supporting_span") or "")
        if document is None:
            errors.append(f"fact {index} does not resolve to a successful page revision")
        elif fact != span or span not in str(document.get("extract") or ""):
            errors.append(f"fact {index} is not an exact captured supporting span")
        if source.get("supporting_span_sha256") != stable_hash(span):
            errors.append(f"fact {index} supporting-span hash mismatch")
        if not str(source.get("url") or "").startswith("https://en.wikipedia.org/"):
            errors.append(f"fact {index} has invalid source URL")
        source_status = (document or {}).get("status")
        if source_status == "disambiguation":
            errors.append(f"fact {index} came from disambiguation")
    return errors


def retrieve_revision(
    config: Mapping[str, Any], split: Mapping[str, Any], manifest: dict[str, Any], logger: logging.Logger
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache_root = Path(config["retrieval"]["corpus_cache_location"])
    action_records: dict[str, dict[str, Any]] = {}
    action_reused: dict[str, bool] = {}
    for item in split["records"]:
        action = str(item["action_phrase"])
        if action in action_records:
            continue
        request = build_knowledge_request(item)
        knowledge, reused = retrieve_external_v2_or_reuse(
            request, cache_root, config["retrieval"]
        )
        action_records[action] = knowledge
        action_reused[action] = reused

    item_records = {
        str(item["question_id"]): action_records[str(item["action_phrase"])]
        for item in split["records"]
    }
    validation_errors = {
        action: errors
        for action, record in action_records.items()
        if (errors := _validate_retrieval_record(record))
    }
    statuses = Counter(record["status"] for record in item_records.values())
    unique_statuses = Counter(record["status"] for record in action_records.values())
    validation = {
        "timestamp": utc_now(),
        "status": "passed" if not validation_errors else "failed",
        "scope": "automated source/provenance checks; not a human audit",
        "selected_items": len(item_records),
        "unique_action_requests": len(action_records),
        "item_status_counts": dict(sorted(statuses.items())),
        "unique_action_status_counts": dict(sorted(unique_statuses.items())),
        "created_unique_actions": sum(not value for value in action_reused.values()),
        "reused_unique_actions": sum(action_reused.values()),
        "validation_errors": validation_errors,
        "all_queries_recorded": all(record.get("search_audit") for record in action_records.values()),
        "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
    }
    atomic_write_json(cache_root / "retrieval_validation.json", validation)
    index = {
        "retrieval_version": RETRIEVAL_V2_VERSION,
        "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
        "items": {
            question_id: {
                "cache_id": record["cache_id"],
                "locator_id": record["locator_id"],
                "status": record["status"],
                "abstention_reason": record.get("abstention_reason"),
                "record_path": str(
                    (cache_root / "external_v2" / "records" / f"{record['cache_id']}.json").resolve()
                ),
                "record": record,
            }
            for question_id, record in item_records.items()
        },
    }
    atomic_write_json(cache_root / "C3_v2_index.json", index)
    abstentions = [
        {
            "question_id": str(item["question_id"]),
            "action_phrase": item["action_phrase"],
            "retrieval_status": item_records[str(item["question_id"])]["status"],
            "reason": item_records[str(item["question_id"])].get("abstention_reason"),
            "knowledge_cache_id": item_records[str(item["question_id"])]["cache_id"],
        }
        for item in split["records"]
        if item_records[str(item["question_id"])]["status"] != "usable"
    ]
    atomic_write_jsonl(cache_root / "C3_v2_abstentions.jsonl", abstentions)
    from .report import write_retrieval_review

    write_retrieval_review(Path(config["outputs"]["analysis_root"]), split, item_records)
    manifest["retrieval_validation"] = validation
    manifest["retrieval_abstention_count"] = len(abstentions)
    atomic_write_json(Path(config["outputs"]["run_root"]) / "manifest.json", manifest)
    if validation_errors:
        raise ValueError(f"revised retrieval validation failed for {len(validation_errors)} actions")
    logger.info(
        "C3-v2 retrieval validated before inference: item statuses=%s; unique created=%d reused=%d",
        dict(statuses), validation["created_unique_actions"], validation["reused_unique_actions"],
    )
    return item_records, validation


def load_retrieval(config: Mapping[str, Any], split: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache_root = Path(config["retrieval"]["corpus_cache_location"])
    index_path = cache_root / "C3_v2_index.json"
    validation_path = cache_root / "retrieval_validation.json"
    if not index_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("complete validated C3-v2 retrieval is required before answering")
    index = _read_json(index_path)
    validation = _read_json(validation_path)
    ids = [str(item["question_id"]) for item in split["records"]]
    if validation.get("status") != "passed" or set(index.get("items", {})) != set(ids):
        raise ValueError("C3-v2 retrieval index is incomplete or failed validation")
    if index.get("selection_policy_fingerprint") != V2_SELECTION_POLICY_FINGERPRINT:
        raise ValueError("C3-v2 retrieval index has an incompatible selector fingerprint")
    records = {question_id: row["record"] for question_id, row in index["items"].items()}
    errors = {question_id: _validate_retrieval_record(record) for question_id, record in records.items()}
    errors = {key: value for key, value in errors.items() if value}
    if errors:
        raise ValueError(f"cached C3-v2 retrieval failed revalidation for {len(errors)} items")
    return records, validation


def _answer_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**config["answer"], "image_detail": config["frames"]["image_detail"]}


def answer_revision(
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    manifest: dict[str, Any],
    retrieval: Mapping[str, Mapping[str, Any]],
    limit: int | None,
    logger: logging.Logger,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    reference_root = Path(config["reference"]["run_root"])
    frames = _load_frame_records(reference_root, split)
    chosen = list(split["records"][:limit] if limit is not None else split["records"])
    plans = []
    for item in chosen:
        question_id = str(item["question_id"])
        knowledge = retrieval[question_id]
        if knowledge["status"] != "usable":
            continue
        plans.append(
            build_answer_plan(
                item,
                "C3",
                config["model"]["provider"],
                config["model"]["identifier"],
                _answer_settings(config),
                frames[question_id],
                knowledge,
                minimum_c3_facts=1,
                provider_endpoint=config["model"]["base_url"],
                knowledge_policy_version=(
                    f"{RETRIEVAL_V2_VERSION}:{V2_SELECTION_POLICY_FINGERPRINT}"
                ),
            )
        )
    prediction_path = Path(config["outputs"]["run_root"]) / "predictions" / "C3.jsonl"
    existing = read_predictions(prediction_path)
    plan_by_id = {str(plan["question_id"]): plan for plan in plans}
    for question_id, row in existing.items():
        if question_id not in {str(item["question_id"]) for item in split["records"]}:
            raise ValueError(f"revision prediction is outside dev100: {question_id}")
        current = plan_by_id.get(question_id)
        if current is not None and row.get("request_hash") != current["request_hash"]:
            raise RuntimeError(f"stale C3-v2 answer cache/prediction for {question_id}")

    missing_calls = [
        plan
        for plan in plans
        if not (
            Path(config["outputs"]["run_root"])
            / "cache" / "responses" / "C3" / f"{plan['request_hash']}.json"
        ).is_file()
    ]
    api_key, credential_source = api_key_from_model_config(config["model"])
    if missing_calls and not api_key:
        raise RuntimeError(
            f"{config['model']['credential_environment_variable']} is required for "
            f"{len(missing_calls)} new C3-v2 answer requests"
        )
    # A cache-only resume never calls this object; a harmless sentinel is enough.
    client = OpenAICompatibleChatClient(
        api_key=api_key or "cache-only",
        base_url=config["model"]["base_url"],
        timeout=float(config["execution"]["request_timeout_seconds"]),
        max_retries=int(config["execution"]["max_retries"]),
    )
    rows, stats, failures = run_answer_plans(
        plans,
        Path(config["outputs"]["run_root"]) / "cache" / "responses",
        client,
        int(config["execution"]["concurrency"]),
    )
    merged = merge_prediction_file(
        prediction_path,
        [str(item["question_id"]) for item in split["records"]],
        rows,
    )
    execution_entry = {
        "timestamp": utc_now(),
        "limit": limit,
        "eligible_in_requested_slice": len(plans),
        "retrieval_abstentions_in_requested_slice": len(chosen) - len(plans),
        "answers": stats,
        "answer_failures": failures,
        "canonical_revised_C3_predictions": len(merged),
        "credential_source": credential_source,
        "reserved_items_evaluated": False,
    }
    history = list(manifest.get("execution_history", []))
    previous_execution = manifest.get("last_execution")
    if previous_execution and previous_execution not in history:
        history.append(previous_execution)
    history.append(execution_entry)
    manifest["execution_history"] = history
    manifest["last_execution"] = execution_entry
    atomic_write_json(Path(config["outputs"]["run_root"]) / "manifest.json", manifest)
    if failures:
        atomic_write_jsonl(
            Path(config["outputs"]["run_root"]) / "cache" / "responses" / "attempt_failures.jsonl",
            failures,
        )
    logger.info(
        "C3-v2 answer stage: eligible=%d abstained=%d reused=%d created=%d failed=%d canonical=%d",
        len(plans), len(chosen) - len(plans), stats["reused"], stats["created"],
        stats["failed"], len(merged),
    )
    return stats, failures


def _reference_is_unchanged(manifest: Mapping[str, Any]) -> None:
    snapshot = manifest["reference_artifacts"]
    for row in snapshot["files"].values():
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"immutable reference artifact changed: {row['path']}")


def _final_integrity_audit(
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    manifest: dict[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    retrieval: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _reference_is_unchanged(manifest)
    run_root = Path(config["outputs"]["run_root"])
    reference_root = Path(config["reference"]["run_root"])
    all_ids = [str(item["question_id"]) for item in split["records"]]
    reserved = set(_read_json(reference_root / "splits" / "reserved_ids.json")["question_ids"])
    if set(all_ids) & reserved:
        raise ValueError("saved development IDs overlap reserved IDs")
    revised = read_predictions(run_root / "predictions" / "C3.jsonl")
    eligible = {question_id for question_id in all_ids if retrieval[question_id]["status"] == "usable"}
    abstained = set(all_ids) - eligible
    if set(revised) != eligible:
        raise ValueError("revised C3 predictions do not exactly match retrieval-eligible IDs")
    if set(revised) & reserved:
        raise ValueError("reserved IDs were evaluated by C3-v2")

    settings = _answer_settings(config)
    request_hashes: set[str] = set()
    for item in split["records"]:
        question_id = str(item["question_id"])
        if question_id not in eligible:
            continue
        plan = build_answer_plan(
            item,
            "C3",
            config["model"]["provider"],
            config["model"]["identifier"],
            settings,
            frames[question_id],
            retrieval[question_id],
            minimum_c3_facts=1,
            provider_endpoint=config["model"]["base_url"],
            knowledge_policy_version=f"{RETRIEVAL_V2_VERSION}:{V2_SELECTION_POLICY_FINGERPRINT}",
        )
        row = revised[question_id]
        if row.get("request_hash") != plan["request_hash"]:
            raise ValueError(f"revised answer request hash mismatch for {question_id}")
        if row.get("prompt_hash") != plan["prompt_hash"] or row.get("assembled_prompt") != plan["prompt"]:
            raise ValueError(f"revised assembled prompt mismatch for {question_id}")
        if row.get("knowledge_facts") != retrieval[question_id]["facts"]:
            raise ValueError(f"revised answer fact payload mismatch for {question_id}")
        if row.get("displayed_choices") != item["displayed_choices"]:
            raise ValueError(f"revised answer choices mismatch for {question_id}")
        if row.get("ordered_frame_payload_hash") != frames[question_id]["ordered_frame_payload_hash"]:
            raise ValueError(f"revised answer frame hash mismatch for {question_id}")
        request_hashes.add(str(row["request_hash"]))
    if len(request_hashes) != len(revised):
        raise ValueError("duplicate effective C3-v2 answer requests")

    api_ids: set[str] = set()
    provenance_facts = 0
    operational_error_count = 0
    warnings_count = 0
    continuation_count = 0
    for action_record in {
        record["cache_id"]: record for record in retrieval.values()
    }.values():
        errors = _validate_retrieval_record(action_record)
        if errors:
            raise ValueError(f"final retrieval validation failed: {errors}")
        provenance_facts += len(action_record["facts"])
        operational_error_count += len(action_record.get("operational_errors", []))
        for search in action_record.get("search_audit", []):
            api_ids.add(str(search["api_request_id"]))
            warnings_count += int(bool(search.get("warnings")))
            continuation_count += int(bool(search.get("continuation")))
        for acquisition in action_record.get("page_acquisitions", []):
            api_ids.add(str(acquisition["api_request_id"]))
            warnings_count += int(bool(acquisition.get("warnings")))
    for api_id in api_ids:
        api_path = Path(config["retrieval"]["corpus_cache_location"]) / "external_v2" / "api" / f"{api_id}.json"
        api_record = _read_json(api_path)
        params = api_record["request_identity"]["params"]
        if "extracts" in str(params.get("prop", "")):
            if int(params.get("exchars", 0)) > 1200 or int(params.get("exintro", 0)) != 1:
                raise ValueError(f"unsupported TextExtracts parameters in {api_path}")
            if int(params.get("exlimit", 1)) > 1 and int(params.get("exintro", 0)) != 1:
                raise ValueError(f"multi-extract request lacks exintro in {api_path}")

    reference_predictions = _validate_reference_predictions(
        config, reference_root, split, frames
    )
    audit = {
        "timestamp": utc_now(),
        "status": "passed",
        "scope": "automated artifact/provenance checks; not a human audit",
        "selected_ids": len(all_ids),
        "reserved_ids": len(reserved),
        "reserved_items_evaluated": False,
        "reference_prediction_counts": {
            condition: len(rows) for condition, rows in reference_predictions.items()
        },
        "revised_C3_predictions": len(revised),
        "revised_C3_parse_failures": sum(
            row.get("parsed_answer") not in {"A", "B", "C", "D", "E"}
            for row in revised.values()
        ),
        "retrieval_eligible_items": len(eligible),
        "retrieval_abstention_items": len(abstained),
        "unique_answer_request_hashes": len(request_hashes),
        "answer_lineage": {
            "canonical_provider_response_records": len(revised),
            "imported_revised_answers": 0,
            "latest_resume_reused": int(
                manifest.get("last_execution", {}).get("answers", {}).get("reused", 0)
            ),
            "latest_resume_created": int(
                manifest.get("last_execution", {}).get("answers", {}).get("created", 0)
            ),
            "original_request_ids_preserved": all(bool(row.get("request_id")) for row in revised.values()),
        },
        "source_supported_facts_across_unique_packs": provenance_facts,
        "current_raw_api_records_referenced": len(api_ids),
        "retrieval_operational_errors": operational_error_count,
        "api_warning_records": warnings_count,
        "bounded_search_continuation_records": continuation_count,
        "frame_records": len(frames),
        "frame_files_rehashed": manifest["reference_artifacts"]["frames"]["jpeg_count"],
        "reference_artifact_hashes_unchanged": True,
    }
    manifest["automated_integrity_checks"] = audit
    atomic_write_json(run_root / "manifest.json", manifest)
    atomic_write_json(run_root / "integrity_audit.json", audit)
    return audit


def _dry_run(config: Mapping[str, Any], split: Mapping[str, Any], limit: int | None) -> None:
    records = list(split["records"][:limit] if limit is not None else split["records"])
    print(
        json.dumps(
            {
                "dry_run": True,
                "variant": "c3_v2",
                "resolved_paths": {
                    "annotation": config["dataset"]["json_path"],
                    "reference_run": config["reference"]["run_root"],
                    "run_root": config["outputs"]["run_root"],
                    "analysis_root": config["outputs"]["analysis_root"],
                },
                "saved_manifest_items": 100,
                "planned_items": len(records),
                "planned_question_ids": [row["question_id"] for row in records],
                "conditions": ["C3"],
                "operations": {
                    "resample": False,
                    "decode_videos": False,
                    "rerun_C0_C2": False,
                    "verify_reference_JPEGs": 1572,
                    "retrieve_unique_actions_full_manifest": len(
                        {row["action_phrase"] for row in split["records"]}
                    ),
                    "maximum_answer_requests_for_slice": len(records),
                    "paid_calls_in_dry_run": 0,
                },
                "provider": config["model"]["provider"],
                "endpoint": config["model"]["base_url"],
                "model": config["model"]["identifier"],
            },
            indent=2,
        )
    )


def execute_c3_v2(config: Mapping[str, Any], args: Any, logger: logging.Logger) -> int:
    conditions = args.conditions or ["C3"]
    if conditions != ["C3"]:
        raise SystemExit("C3-v2 accepts only --conditions C3; C0-C2 are immutable references")
    logger.info("C3-v2 start stage=%s dry_run=%s", args.stage, args.dry_run)
    try:
        split, manifest, resolved = prepare_revision(config, logger)
        if args.dry_run:
            _dry_run(resolved, split, args.limit)
            return 0
        retrieval: dict[str, dict[str, Any]] | None = None
        if args.stage in {"retrieve", "all"}:
            retrieval, _ = retrieve_revision(resolved, split, manifest, logger)
        if args.stage == "run":
            retrieval, _ = load_retrieval(resolved, split)
        failures: list[dict[str, Any]] = []
        if args.stage in {"run", "all"}:
            if retrieval is None:
                retrieval, _ = load_retrieval(resolved, split)
            _, failures = answer_revision(resolved, split, manifest, retrieval, args.limit, logger)
        if args.stage in {"report", "all"}:
            retrieval, _ = load_retrieval(resolved, split)
            frames = _load_frame_records(Path(resolved["reference"]["run_root"]), split)
            _final_integrity_audit(resolved, split, manifest, frames, retrieval)
            generate_c3_v2_reports(
                Path(resolved["outputs"]["analysis_root"]),
                Path(resolved["outputs"]["run_root"]),
                Path(resolved["reference"]["run_root"]),
                split,
                manifest,
                resolved,
                frames,
                retrieval,
            )
        strict = bool(resolved["retrieval"].get("strict_complete_coverage", False))
        retrieval_statuses = Counter(
            row["status"] for row in (retrieval or {}).values()
        )
        exit_code = int(bool(failures) or bool(retrieval_statuses.get("operational_error")))
        if strict and retrieval_statuses.get("abstained"):
            exit_code = 3
        logger.info(
            "C3-v2 finished exit_code=%d (honest retrieval abstentions are successful when strict coverage is false)",
            exit_code,
        )
        return exit_code
    except Exception as exc:
        logger.exception("C3-v2 failed: %s", exc)
        return 1
