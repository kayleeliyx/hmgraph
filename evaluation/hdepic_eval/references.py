"""Load immutable annotation references and their common semantic source masks.

Only explicitly supplied JSONL files are read. This module neither reads model
outputs nor discovers datasets, so test references are never opened by default.
Legacy WHY reviews supply source decisions only; their answer labels are ignored.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


JUDGEABLE_WHY = {"valid", "valid_with_nonpurpose_suffix"}
UNJUDGEABLE_WHY = {"ambiguous_attachment", "source_conflict"}


def _rows(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            result.append(value)
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _class(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Canonical class IDs must be nonnegative integers or null")
    return value


def canonical_pair(action: dict[str, Any]) -> list[int | None]:
    """Use one canonical main-action pair; never select an alternative pair."""
    pairs = action.get("main_action_class_ids", [])
    if not isinstance(pairs, list):
        raise ValueError("main_action_class_ids must be a list")
    if not pairs:
        return [None, None]
    if len(pairs) != 1 or not isinstance(pairs[0], (list, tuple)) or len(pairs[0]) not in (1, 2):
        raise ValueError("Expected exactly one canonical main-action class pair")
    pair = pairs[0]
    return [_class(pair[0]), _class(pair[1]) if len(pair) == 2 else None]


def _action(action: Any, video: str, label: str) -> tuple[float, float]:
    if not isinstance(action, dict):
        raise ValueError(f"{label} must be an action object")
    if action.get("video_id") != video:
        raise ValueError(f"{label} belongs to a different video")
    _text(action.get("narration_id"), f"{label}.narration_id")
    _text(action.get("action"), f"{label}.action")
    start = _time(action.get("start_sec"), f"{label}.start_sec")
    end = _time(action.get("end_sec"), f"{label}.end_sec")
    if end < start:
        raise ValueError(f"{label} has a reversed interval")
    return start, end


def _legacy_sources(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result = {}
    for row in _rows(path):
        identity = _text(row.get("id"), "Legacy WHY id")
        status = row.get("reference_status")
        if status not in JUDGEABLE_WHY | UNJUDGEABLE_WHY:
            raise ValueError(f"{identity}: unsupported legacy WHY source status {status!r}")
        source = {
            "reference_status": status,
            "purpose_core": row.get("purpose_core"),
            "reference_reason": _text(row.get("reference_reason"), f"{identity}: legacy reason"),
            "narration_id": row.get("narration_id"),
            "action_question": row.get("action_question"),
        }
        if identity in result and result[identity] != source:
            raise ValueError(f"{identity}: inconsistent legacy WHY common source mask or purpose core")
        result[identity] = source
    return result


def _why(record: dict[str, Any], legacy: dict[str, Any] | None,
         provenance: str) -> dict[str, Any]:
    empty = {"eligible": False, "reference": None, "original_reference": None,
             "source_status": "not_applicable", "source_provenance": "frozen_scoring_metadata",
             "original_narration": None}
    if record["task"] != "explain":
        if legacy is not None:
            raise ValueError(f"{record['id']}: WHY source review attached to a Track row")
        return empty
    reason = record.get("reason_annotation")
    if not isinstance(reason, dict):
        raise ValueError(f"{record['id']}: Explain row has no reason annotation")
    identity = record["id"]
    current = record["current_action"]
    if legacy is not None:
        if legacy.get("narration_id") is not None and legacy["narration_id"] != current["narration_id"]:
            raise ValueError(f"{identity}: legacy review is attached to another narration")
        question = (record.get("question_provenance") or {}).get("action_question_text", current["action"])
        if legacy.get("action_question") is not None and legacy["action_question"] != question:
            raise ValueError(f"{identity}: legacy review uses another action question")
    if reason.get("video_id") != record["video_id"] or reason.get("narration_id") != current["narration_id"]:
        raise ValueError(f"{identity}: reason is bound to another action/video")
    original = _text(reason.get("reason_text"), f"{identity}: reason text")
    narration = _text(reason.get("original_narration"), f"{identity}: original narration")
    start, end = reason.get("char_start"), reason.get("char_end")
    if (type(start) is not int or type(end) is not int or
            not 0 <= start < end <= len(narration) or narration[start:end] != original):
        raise ValueError(f"{identity}: reason offsets do not bind to the original narration")
    status = reason.get("status")
    if not isinstance(status, str) or not status.startswith("accepted"):
        raise ValueError(f"{identity}: Explain reference reason is not an accepted annotation")
    result = {**empty, "eligible": True, "reference": original,
              "original_reference": original, "original_narration": narration,
              "source_status": "annotation_derived_pending_review"}
    if legacy is None:
        return result
    if legacy["reference_reason"] != original:
        raise ValueError(f"{identity}: legacy WHY reference differs from the frozen annotation")
    status = legacy["reference_status"]
    result["source_status"] = status
    result["source_provenance"] = provenance
    result["eligible"] = status in JUDGEABLE_WHY
    if not result["eligible"]:
        result["reference"] = None
        return result
    core = _text(legacy.get("purpose_core"), f"{identity}: legacy purpose core")
    if core not in narration or core not in original:
        raise ValueError(f"{identity}: legacy purpose core is not an exact bound source substring")
    if status == "valid" and core != original:
        raise ValueError(f"{identity}: clean legacy purpose core changes the original reason")
    result["reference"] = core
    return result


def _goal(record: dict[str, Any]) -> dict[str, Any]:
    status = record.get("goal_status")
    if status not in {"available", "missing_activity", "ambiguous_activity"}:
        raise ValueError(f"{record['id']}: unsupported goal source status")
    result = {"eligible": False, "reference": None, "source_status": status,
              "profile": None, "annotation_id": None}
    if status != "available":
        # Missing/ambiguous activity annotations are unsupervised, not negative
        # labels for the existence of a goal.
        return result
    annotation = record.get("goal_annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"{record['id']}: available goal has no activity annotation")
    label = _text(annotation.get("label"), f"{record['id']}: activity label")
    annotation_id = _text(annotation.get("annotation_id"), f"{record['id']}: activity annotation id")
    video = annotation.get("resolved_video_id", annotation.get("video_id"))
    if video != record["video_id"]:
        raise ValueError(f"{record['id']}: goal annotation belongs to another resolved video")
    start = _time(annotation.get("start_sec"), f"{record['id']}: goal start")
    end = _time(annotation.get("end_sec"), f"{record['id']}: goal end")
    if not start <= record["query_time_sec"] < end:
        raise ValueError(f"{record['id']}: goal interval does not contain the query")
    from .semantics import make_goal_profile
    profile = make_goal_profile(label, source_annotation_ids=[annotation_id])
    return {**result, "eligible": True, "reference": label,
            "profile": profile, "annotation_id": annotation_id}


def load_references(scoring_path: str | Path,
                    legacy_why_judgments: str | Path | None = None) -> list[dict[str, Any]]:
    """Prepare references without rewriting annotations or opening any defaults.

    IDs and causal metadata are validated before reference construction. The
    legacy source mask applies identically to every model, including new ones.
    WHY references without a legacy source decision remain explicitly
    provisional. GOAL scope comes from the original activity label/profile;
    no session hierarchy is inferred or substituted.
    """
    records = _rows(scoring_path)
    legacy_sources = _legacy_sources(legacy_why_judgments)
    provenance = (f"legacy_frozen_why_review:{Path(legacy_why_judgments).resolve()}"
                  if legacy_why_judgments is not None else "frozen_scoring_metadata")
    seen = set()
    result = []
    for record in records:
        identity = _text(record.get("id"), "Reference id")
        if identity in seen:
            raise ValueError(f"Duplicate reference id: {identity}")
        seen.add(identity)
        task = record.get("task")
        if task not in {"track", "explain"}:
            raise ValueError(f"{identity}: unsupported task {task!r}")
        video = _text(record.get("video_id"), f"{identity}: video id")
        participant = _text(record.get("participant"), f"{identity}: participant")
        query = _time(record.get("query_time_sec"), f"{identity}: query time")
        current = record.get("current_action")
        start, end = _action(current, video, f"{identity}: current action")
        if task == "track" and not start <= query < end:
            raise ValueError(f"{identity}: Track query is outside the current action")
        if task == "explain" and not math.isclose(query, end, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{identity}: Explain query is not the current action end")
        future = record.get("future_actions")
        if not isinstance(future, list) or len(future) != 3:
            raise ValueError(f"{identity}: exactly three future actions are required")
        future_keys = []
        for index, action in enumerate(future):
            a_start, a_end = _action(action, video, f"{identity}: future action {index + 1}")
            if not a_start > query:
                raise ValueError(f"{identity}: future action does not start strictly after the query")
            future_keys.append((a_start, a_end, action["narration_id"]))
        if future_keys != sorted(future_keys):
            raise ValueError(f"{identity}: future actions are not in frozen chronological order")
        history = record.get("history_actions", [])
        if not isinstance(history, list):
            raise ValueError(f"{identity}: history must be a list")
        for index, action in enumerate(history):
            _, a_end = _action(action, video, f"{identity}: history action {index + 1}")
            if a_end > query:
                raise ValueError(f"{identity}: history includes an unfinished/future action")
            if task == "explain" and action["narration_id"] == current["narration_id"]:
                raise ValueError(f"{identity}: Explain target appears in completed history")
        frames = record.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError(f"{identity}: frames must be a list")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"{identity}: frame {index + 1} must be an object")
            timestamp = _time(frame.get("timestamp_sec"), f"{identity}: frame timestamp")
            if timestamp > query + 1e-6:
                raise ValueError(f"{identity}: input frame is from the future")
        question = record.get("question_provenance") or {}
        if not isinstance(question, dict):
            raise ValueError(f"{identity}: question provenance must be an object")
        question_text = question.get("action_question_text", current["action"])
        result.append({
            "id": identity, "task": task, "video_id": video,
            "participant": participant, "query_time_sec": query,
            "current_pair": canonical_pair(current),
            "next_pairs": [canonical_pair(action) for action in future],
            "question": _text(question_text, f"{identity}: action question"),
            "history": [{key: action[key] for key in
                         ("narration_id", "video_id", "action", "start_sec", "end_sec")}
                        for action in history],
            "why": _why(record, legacy_sources.get(identity), provenance),
            "goal": _goal(record),
        })
    return result


def load_reference_bundle(path: str | Path) -> list[dict[str, Any]]:
    """Read prepared frozen references without the dataset or legacy scripts."""
    records = _rows(path)
    seen = set()
    for record in records:
        identity = _text(record.get("id"), "Reference id")
        if identity in seen:
            raise ValueError(f"Duplicate reference id: {identity}")
        seen.add(identity)
        if record.get("task") not in {"explain", "track"}:
            raise ValueError("Prepared reference task must be explain or track")
        for key in ("video_id", "participant", "question"):
            _text(record.get(key), key)
        query = _time(record.get("query_time_sec"), "query_time_sec")
        pairs = [record.get("current_pair"), *record.get("next_pairs", [])]
        if len(pairs) != 4:
            raise ValueError("Prepared references require exactly three future pairs")
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Prepared class pairs need verb and noun components")
            for value in pair:
                _class(value)
        history = record.get("history")
        if not isinstance(history, list) or any(_time(h.get("end_sec"), "history end") > query for h in history):
            raise ValueError("Prepared history must be completed by the query")
        for field in ("why", "goal"):
            source = record.get(field)
            if not isinstance(source, dict) or type(source.get("eligible")) is not bool:
                raise ValueError("Prepared semantic sources need explicit eligibility")
            if not source["eligible"]:
                continue
            reference = _text(source.get("reference"), f"{field} reference")
            if field == "why":
                if record["task"] != "explain":
                    raise ValueError("Track cannot have an eligible WHY reference")
                original = _text(source.get("original_reference"), "original WHY reference")
                narration = _text(source.get("original_narration"), "original narration")
                if reference not in original or original not in narration:
                    raise ValueError("Prepared WHY purpose is not bound to the narration")
            else:
                profile = source.get("profile")
                if not isinstance(profile, dict) or profile.get("original_label") != reference:
                    raise ValueError("Prepared GOAL profile must preserve its original label")
                units = profile.get("required_units")
                if not isinstance(units, list) or not units or any(
                        not isinstance(unit, str) or not unit.strip() or unit not in reference for unit in units):
                    raise ValueError("Prepared GOAL units must be literal reference clauses")
    return records
