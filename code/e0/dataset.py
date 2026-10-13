"""Official HD-EPIC why-recognition loading and deterministic split creation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import LETTERS, PROTOCOL_VERSION

ACTION_RE = re.compile(r"<([^<>]+)>")
TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
    )


def parse_timestamp(value: str) -> float:
    match = TIME_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid HD-EPIC timestamp: {value!r}")
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"out-of-range HD-EPIC timestamp: {value!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds


def extract_action_phrase(question: str) -> str:
    matches = ACTION_RE.findall(question)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError(f"expected exactly one nonempty angle-bracket action in {question!r}")
    return matches[0].strip()


def participant_for_video(video_id: str) -> str:
    participant = video_id.split("-", 1)[0]
    if not re.fullmatch(r"P\d{2}", participant):
        raise ValueError(f"unexpected HD-EPIC video id: {video_id!r}")
    return participant


def source_clip_key(video_id: str, start_sec: float, end_sec: float) -> str:
    return f"{video_id}:{start_sec:.6f}-{end_sec:.6f}"


def _validate_item(question_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question IDs must be nonempty strings")
    inputs = item.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"video 1"}:
        raise ValueError(f"{question_id}: expected exactly inputs['video 1']")
    clip = inputs["video 1"]
    if not isinstance(clip, dict) or not all(key in clip for key in ("id", "start_time", "end_time")):
        raise ValueError(f"{question_id}: malformed video 1 input")
    video_id = clip["id"]
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"{question_id}: empty video id")
    start_sec = parse_timestamp(clip["start_time"])
    end_sec = parse_timestamp(clip["end_time"])
    if not end_sec > start_sec:
        raise ValueError(f"{question_id}: clip end must be after start")
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{question_id}: question must be nonempty")
    action_phrase = extract_action_phrase(question)
    choices = item.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 5
        or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
    ):
        raise ValueError(f"{question_id}: expected five nonempty choices")
    correct_idx = item.get("correct_idx")
    if isinstance(correct_idx, bool) or not isinstance(correct_idx, int) or not 0 <= correct_idx <= 4:
        raise ValueError(f"{question_id}: correct_idx must be an integer from 0 through 4")
    return {
        "question_id": question_id,
        "participant": participant_for_video(video_id),
        "source_video_id": video_id,
        "source_clip_key": source_clip_key(video_id, start_sec, end_sec),
        "requested_start": clip["start_time"],
        "requested_end": clip["end_time"],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "action_phrase": action_phrase,
        "question": question,
        "original_choices": list(choices),
        "original_correct_idx": correct_idx,
    }


def load_official_annotation(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("fine_grained_why_recognition.json must be an object keyed by question ID")
    if len(data) != len(set(data)):
        raise ValueError("duplicate question IDs")
    validated = {question_id: _validate_item(question_id, item) for question_id, item in data.items()}
    metadata = {
        "path": str(path.resolve()),
        "source": "https://github.com/hd-epic/hd-epic-annotations/blob/main/vqa-benchmark/fine_grained_why_recognition.json",
        "sha256": sha256_bytes(raw),
        "item_count": len(data),
    }
    if len(data) != 500:
        raise ValueError(
            f"unexpected official why-recognition item count {len(data)} (expected inspected release count 500)"
        )
    return validated, metadata


def option_permutation(question_id: str, seed: int, count: int = 5) -> list[int]:
    digest = hashlib.sha256(f"{PROTOCOL_VERSION}|options|{seed}|{question_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    permutation = list(range(count))
    rng.shuffle(permutation)
    return permutation


def select_participant_balanced(
    items: Mapping[str, Mapping[str, Any]], seed: int, sample_size: int
) -> tuple[list[str], list[str], list[str]]:
    participants: dict[str, list[str]] = defaultdict(list)
    for question_id, item in items.items():
        participants[str(item["participant"])].append(question_id)
    participant_names = sorted(participants)
    if sample_size <= 0 or len(participant_names) != 9:
        raise ValueError(
            f"E0 requires a positive sample size and nine participants; "
            f"got {sample_size} and {participant_names}"
        )
    base_count, remainder = divmod(sample_size, len(participant_names))
    if base_count < 1 or any(len(ids) < base_count for ids in participants.values()):
        raise ValueError(
            f"participant-balanced sampling requires at least {base_count} items per participant"
        )

    extra_rng = random.Random(seed)
    extra_participants = sorted(extra_rng.sample(participant_names, remainder))
    selected_by_participant: dict[str, list[str]] = {}
    for participant in participant_names:
        digest = hashlib.sha256(
            f"{PROTOCOL_VERSION}|sample|{seed}|{participant}".encode("utf-8")
        ).digest()
        item_rng = random.Random(int.from_bytes(digest[:16], "big"))
        candidates = sorted(participants[participant])
        item_rng.shuffle(candidates)
        take = base_count + int(participant in extra_participants)
        if len(candidates) < take:
            raise ValueError(
                f"participant {participant} has {len(candidates)} items but requires {take}"
            )
        selected_by_participant[participant] = candidates[:take]

    # Interleave participants so --limit is not dominated by one person.
    selected: list[str] = []
    for offset in range(base_count + int(bool(remainder))):
        for participant in participant_names:
            rows = selected_by_participant[participant]
            if offset < len(rows):
                selected.append(rows[offset])
    reserved = sorted(set(items) - set(selected))
    return selected, reserved, extra_participants


def resolve_video_path(
    video_id: str, video_roots: Sequence[Path], overrides: Mapping[str, str] | None = None
) -> Path:
    if overrides and video_id in overrides:
        candidate = Path(overrides[video_id]).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"configured video override does not exist: {candidate}")
    participant = participant_for_video(video_id)
    candidates: list[Path] = []
    for root in video_roots:
        candidates.extend((root / participant / f"{video_id}.mp4", root / f"{video_id}.mp4"))
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"expected one source video for {video_id}; found {existing}; tried {candidates}")
    return existing[0]


def build_split(
    annotation_path: Path,
    video_roots: Sequence[Path],
    overrides: Mapping[str, str],
    seed: int,
    sample_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    items, annotation = load_official_annotation(annotation_path)
    selected_ids, reserved_ids, extra_participants = select_participant_balanced(items, seed, sample_size)
    records: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    for question_id in selected_ids:
        item = items[question_id]
        permutation = option_permutation(question_id, seed)
        displayed_choices = [item["original_choices"][index] for index in permutation]
        displayed_correct_idx = permutation.index(item["original_correct_idx"])
        record = {
            key: item[key]
            for key in (
                "question_id",
                "participant",
                "source_video_id",
                "source_clip_key",
                "requested_start",
                "requested_end",
                "start_sec",
                "end_sec",
                "action_phrase",
                "question",
            )
        }
        record.update(
            {
                "source_video_path": str(
                    resolve_video_path(item["source_video_id"], video_roots, overrides)
                ),
                "displayed_choices": displayed_choices,
                "option_order": permutation,
            }
        )
        records.append(record)
        labels[question_id] = {
            "original_correct_idx": item["original_correct_idx"],
            "displayed_correct_idx": displayed_correct_idx,
            "gold_letter": LETTERS[displayed_correct_idx],
        }
    distribution = dict(sorted(Counter(record["participant"] for record in records).items()))
    duplicate_clips = {
        clip: ids
        for clip, ids in _group_clip_ids(records).items()
        if len(ids) > 1
    }
    split = {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_sha256": annotation["sha256"],
        "seed": seed,
        "sample_size": sample_size,
        "sampling": (
            f"{sample_size // len(distribution)} per participant plus one extra for "
            f"{sample_size % len(distribution)} seed-selected participants"
        ),
        "extra_participants": extra_participants,
        "participant_distribution": distribution,
        "records": records,
        "scoring_labels": labels,
    }
    reserved = {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_sha256": annotation["sha256"],
        "reserved_count": len(reserved_ids),
        "question_ids": reserved_ids,
        "source_clips": {
            question_id: {
                "participant": items[question_id]["participant"],
                "source_video_id": items[question_id]["source_video_id"],
                "source_clip_key": items[question_id]["source_clip_key"],
                "start_sec": items[question_id]["start_sec"],
                "end_sec": items[question_id]["end_sec"],
            }
            for question_id in reserved_ids
        },
        "policy": "Reserved IDs are persisted for a future split and are not evaluated by E0.",
    }
    split_metadata = {
        "annotation": annotation,
        "participant_distribution": distribution,
        "extra_participants": extra_participants,
        "duplicate_selected_clips": duplicate_clips,
    }
    return split, reserved, split_metadata


def _group_clip_ids(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_clip_key"])].append(str(record["question_id"]))
    return dict(grouped)


def validate_persisted_split(
    split: Mapping[str, Any], annotation_sha256: str, seed: int, sample_size: int
) -> None:
    if split.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("persisted development split uses an incompatible E0 protocol version")
    if split.get("annotation_sha256") != annotation_sha256:
        raise ValueError("persisted development split annotation checksum does not match the configured annotation")
    if split.get("seed") != seed or split.get("sample_size") != sample_size:
        raise ValueError("persisted development split seed/size is incompatible with the configured protocol")
    records = split.get("records")
    labels = split.get("scoring_labels")
    if not isinstance(records, list) or len(records) != sample_size or not isinstance(labels, dict):
        raise ValueError("persisted development split has an invalid record/label structure")
    ids = [record.get("question_id") for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError("persisted development split IDs and scoring-label IDs disagree")
    for record in records:
        if option_permutation(record["question_id"], seed) != record.get("option_order"):
            raise ValueError(f"persisted option permutation drift for {record['question_id']}")
        if len(record.get("displayed_choices", [])) != 5:
            raise ValueError(f"persisted choices malformed for {record['question_id']}")
