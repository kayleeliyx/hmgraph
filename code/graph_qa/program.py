"""Question-to-program decomposition for compositional graph retrieval.

RAVU decomposes questions into explicit retrieval/localization operators. This
module keeps that useful idea, but generalizes the program beyond localization
so the same backend can support recognition, state, counting, ordering,
tracking, causal, and future graph tasks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


TIME_RE = re.compile(r"<TIME\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s+video\s+\d+>", re.I)
BBOX_RE = re.compile(r"<BBOX\s+[^>]+>", re.I)
ANGLE_RE = re.compile(r"<([^>]+)>")


def hms_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _time_match_to_seconds(match: re.Match[str]) -> float:
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


@dataclass(frozen=True)
class Operation:
    """One extensible graph program operation.

    Names are strings so new operators can be registered without changing a
    closed enum or old experiment files.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionProgram:
    question: str
    retrieval_query: str
    intent: str
    answer_type: str
    operations: tuple[Operation, ...]
    temporal_scope: tuple[float, float] | None = None
    reference_times: tuple[float, ...] = ()
    references: tuple[str, ...] = ()
    requires_visual_identity: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QuestionProgram":
        """Accept a validated plan from a future LLM planner."""

        operations = tuple(
            operation if isinstance(operation, Operation) else Operation(**operation)
            for operation in value.get("operations", ())
        )
        temporal_scope = value.get("temporal_scope")
        return cls(
            question=str(value["question"]),
            retrieval_query=str(value.get("retrieval_query") or value["question"]),
            intent=str(value.get("intent") or "retrieve"),
            answer_type=str(value.get("answer_type") or "multiple_choice"),
            operations=operations,
            temporal_scope=tuple(temporal_scope) if temporal_scope is not None else None,
            reference_times=tuple(float(item) for item in value.get("reference_times", ())),
            references=tuple(str(item) for item in value.get("references", ())),
            requires_visual_identity=bool(value.get("requires_visual_identity", False)),
            metadata=dict(value.get("metadata") or {}),
        )


def question_temporal_scope(inputs: dict[str, Any]) -> tuple[float, float] | None:
    spans: list[tuple[float, float]] = []
    for value in inputs.values():
        if not isinstance(value, dict):
            continue
        if value.get("start_time") is not None and value.get("end_time") is not None:
            spans.append((hms_to_seconds(value["start_time"]), hms_to_seconds(value["end_time"])))
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def clean_retrieval_query(question: str) -> str:
    """Remove non-semantic markup while retaining reference-time language."""

    cleaned = BBOX_RE.sub("the visually indicated object", question)
    cleaned = TIME_RE.sub(lambda match: f"at {int(_time_match_to_seconds(match))} seconds", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class RuleBasedProgramPlanner:
    """Stable question-only planner for controlled graph comparisons.

    It never receives answer choices or ground-truth labels. The output is a
    compositional program that the generic retrieval engine executes.
    """

    def plan(
        self,
        *,
        question: str,
        inputs: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> QuestionProgram:
        inputs = inputs or {}
        normalized = question.lower()
        scope = question_temporal_scope(inputs)
        reference_times = tuple(_time_match_to_seconds(match) for match in TIME_RE.finditer(question))
        references = tuple(
            match.group(1).strip()
            for match in ANGLE_RE.finditer(question)
            if not match.group(1).upper().startswith(("TIME ", "BBOX "))
        )
        requires_visual_identity = bool(BBOX_RE.search(question))

        intent, answer_type = self._classify(normalized)
        operations: list[Operation] = []

        if scope is not None:
            # The input interval is benchmark-provided question context, not an
            # answer-choice timestamp. Localization uses it as a search scope;
            # recognition uses it as a tight evidence filter.
            margin = 0.5 if intent in {"recognize", "describe", "explain"} else 2.0
            operations.append(Operation("filter_time", {"start": scope[0], "end": scope[1], "margin": margin}))

        operations.append(Operation("retrieve_nodes", {"views": ["event", "entity", "relation", "attribute"]}))

        if requires_visual_identity:
            operations.append(Operation("resolve_reference", {"modality": "visual", "times": reference_times}))

        if intent == "localize":
            operations.extend(
                [
                    Operation("expand_entities", {"hops": 1}),
                    Operation("expand_hierarchy", {"hops": 1}),
                    Operation("select_temporal_spans", {}),
                ]
            )
        elif intent in {"recognize", "describe", "explain"}:
            operations.extend(
                [
                    Operation("expand_hierarchy", {"hops": 1}),
                    Operation("expand_temporal", {"before": 1, "after": 1}),
                ]
            )
        elif intent == "count":
            operations.extend(
                [
                    Operation("expand_entities", {"hops": 1}),
                    Operation("deduplicate_events", {}),
                    Operation("count_events", {}),
                ]
            )
        elif intent in {"track", "order"}:
            operations.extend(
                [
                    Operation("expand_entities", {"hops": 2}),
                    Operation("order_temporally", {}),
                ]
            )
        else:
            operations.append(Operation("expand_neighbors", {"hops": 1}))

        operations.append(Operation("rerank_evidence", {}))
        operations.append(Operation("answer", {"type": answer_type}))
        return QuestionProgram(
            question=question,
            retrieval_query=clean_retrieval_query(question),
            intent=intent,
            answer_type=answer_type,
            operations=tuple(operations),
            temporal_scope=scope,
            reference_times=reference_times,
            references=references,
            requires_visual_identity=requires_visual_identity,
            metadata={"planner": "rules_v1", "category_observed_for_trace_only": category},
        )

    @staticmethod
    def _classify(question: str) -> tuple[str, str]:
        # Classification is driven only by the question string. Benchmark
        # category is recorded in the trace but never changes the program.
        if question.startswith("when ") or " when did " in f" {question}":
            return "localize", "temporal_multiple_choice"
        if question.startswith("how many") or "number of times" in question:
            return "count", "count_multiple_choice"
        if "moved from/to" in question or "throughout the video" in question:
            return "track", "ordered_path_multiple_choice"
        if "what happened before" in question or "what happened after" in question or "following" in question:
            return "order", "multiple_choice"
        if question.startswith("why ") or "description for why" in question:
            return "explain", "multiple_choice"
        if question.startswith("how ") or "description for how" in question:
            return "describe", "multiple_choice"
        if question.startswith("where "):
            return "track", "multiple_choice"
        if question.startswith("which ") or question.startswith("what "):
            return "recognize", "multiple_choice"
        return "retrieve", "multiple_choice"


def operation_names(operations: Iterable[Operation]) -> set[str]:
    return {operation.name for operation in operations}
