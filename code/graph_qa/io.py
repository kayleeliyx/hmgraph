"""I/O helpers for graph-QA experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_hdepic_questions(
    annotation_root: Path,
    *,
    video_id: str,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for path in sorted(annotation_root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for question_id, question in data.items():
            video_ids = [
                value.get("id")
                for value in question.get("inputs", {}).values()
                if isinstance(value, dict) and value.get("id")
            ]
            if video_ids and set(video_ids) == {video_id}:
                questions.append(
                    {
                        "category": path.stem,
                        "question_id": question_id,
                        **question,
                    }
                )
    return questions


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
