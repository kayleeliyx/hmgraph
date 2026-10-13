"""Deterministic tests for graph-QA structure (no neural inference)."""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch

from code.graph_qa.adapters import build_temporal_graph
from code.graph_qa.answering import AnswerResult, format_answer_prompt
from code.graph_qa.pipeline import GraphQAPipeline
from code.graph_qa.program import RuleBasedProgramPlanner
from code.graph_qa.retrieval import CompositionalGraphRetriever, RetrievalConfig


class HashEmbedder:
    model_id = "test-hash"

    @staticmethod
    def _encode(texts: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            vector = torch.zeros(128, dtype=torch.float32)
            for token in text.lower().replace(";", " ").replace(":", " ").split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1
            rows.append(vector / vector.norm().clamp_min(1e-6))
        return torch.stack(rows) if rows else torch.empty((0, 128))

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(texts)


class TokenOverlapReranker:
    model_id = "test-overlap"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        return [len(query_tokens & set(document.lower().split())) / max(1, len(query_tokens)) for document in documents]


class FirstChoiceAnswerer:
    model_id = "test-first"

    def answer(self, *, program, choices, retrieval) -> AnswerResult:
        prompt = format_answer_prompt(program, choices, retrieval, 10000)
        return AnswerResult(0, "A", "A", tuple([1.0] + [0.0] * (len(choices) - 1)), prompt)


def _w_records() -> list[dict]:
    return [
        {
            "window_idx": 0,
            "t_start": 7.0,
            "t_end": 8.0,
            "parsed": {
                "actions": [
                    {
                        "action_idx": 0,
                        "verb": "open",
                        "direct_object": "cupboard door",
                        "hand": "left",
                        "confidence": 0.9,
                        "spatial_context": {"location": "counter", "near": ["mug"]},
                        "interaction": {"object_change": "opened", "motion_direction": "outward"},
                        "event_t_start": 7.2,
                        "event_t_end": 7.9,
                    }
                ]
            },
        },
        {
            "window_idx": 1,
            "t_start": 8.0,
            "t_end": 9.0,
            "parsed": {
                "actions": [
                    {
                        "action_idx": 0,
                        "verb": "pick up",
                        "direct_object": "mug",
                        "hand": "left",
                        "confidence": 0.95,
                        "spatial_context": {"location": "cupboard shelf"},
                        "interaction": {"object_change": "lifted"},
                        "event_t_start": 8.1,
                        "event_t_end": 8.8,
                    }
                ]
            },
        },
    ]


def _l1_records() -> list[dict]:
    return [
        {
            "l1_id": "l1_mug",
            "t_start": 8.0,
            "t_end": 9.0,
            "verb": "pick up",
            "direct_object": {"name": "mug"},
            "hand": "left",
            "micro_action_description": "Pick up the mug from the cupboard shelf.",
            "source_window_idxs": [1],
            "confidence": 0.95,
        }
    ]


def _retriever() -> CompositionalGraphRetriever:
    return CompositionalGraphRetriever(
        embedder=HashEmbedder(),
        reranker=TokenOverlapReranker(),
        config=RetrievalConfig(
            dense_top_k_per_view=8,
            rerank_top_k=8,
            seed_top_k=3,
            final_top_k=8,
            max_expanded_nodes=20,
            similarity_device="cpu",
        ),
    )


def test_adapters_keep_rich_structure_and_hierarchy() -> None:
    graph = build_temporal_graph(
        video_id="video",
        mode="w+l1",
        w_records=_w_records(),
        l1_records=_l1_records(),
    )
    relations = {edge.relation for edge in graph.edges.values()}
    assert {"involves", "occurs_at", "causes_state", "next", "contains"} <= relations
    assert graph.nodes["w:00001:action:0"].span.start == 8.1
    contains = [edge for edge in graph.edges.values() if edge.relation == "contains"]
    assert contains and contains[0].source_id == "l1:l1_mug"
    assert contains[0].target_id == "w:00001:action:0"


def test_program_is_question_only_and_general() -> None:
    planner = RuleBasedProgramPlanner()
    localization = planner.plan(
        question="When did the action <pick up mug> happen?",
        inputs={"video 1": {"start_time": "00:00:07.0", "end_time": "00:00:10.0"}},
        category="fine_grained_action_localization",
    )
    assert localization.intent == "localize"
    assert "pick up mug" in localization.retrieval_query
    assert "select_temporal_spans" in {operation.name for operation in localization.operations}
    counting = planner.plan(question="How many times did I open the cupboard?", inputs={})
    assert counting.intent == "count"
    assert "count_events" in {operation.name for operation in counting.operations}
    tracking = planner.plan(question="Where was the mug moved from/to throughout the video?", inputs={})
    assert tracking.intent == "track"
    assert "order_temporally" in {operation.name for operation in tracking.operations}


def test_compositional_retrieval_traverses_hierarchy_and_entities() -> None:
    graph = build_temporal_graph(
        video_id="video",
        mode="w+l1",
        w_records=_w_records(),
        l1_records=_l1_records(),
    )
    program = RuleBasedProgramPlanner().plan(
        question="When did the action <pick up mug> happen?",
        inputs={"video 1": {"start_time": "00:00:07.0", "end_time": "00:00:10.0"}},
    )
    trace = _retriever().retrieve(graph, program)
    evidence_ids = {item.node_id for item in trace.evidence}
    assert "l1:l1_mug" in evidence_ids or "w:00001:action:0" in evidence_ids
    assert any("expand_hierarchy" in path for item in trace.evidence for path in item.paths)


def test_bbox_question_records_graph_only_limitation() -> None:
    graph = build_temporal_graph(video_id="video", mode="w", w_records=_w_records())
    program = RuleBasedProgramPlanner().plan(
        question="Where did I put <BBOX 1 2 3 4> at <TIME 00:00:8.2 video 1>?",
        inputs={"video 1": {"id": "video"}},
    )
    trace = _retriever().retrieve(graph, program)
    assert program.requires_visual_identity
    assert any("graph-only" in warning for warning in trace.warnings)


def test_count_operator_produces_deduplicated_aggregate() -> None:
    graph = build_temporal_graph(video_id="video", mode="w", w_records=_w_records())
    program = RuleBasedProgramPlanner().plan(
        question="How many times did I pick up the mug?",
        inputs={},
    )
    trace = _retriever().retrieve(graph, program)
    assert "retrieved_event_cluster_count" in trace.aggregates
    assert trace.aggregates["retrieved_event_cluster_count"] >= 1


def test_choices_enter_only_after_retrieval() -> None:
    graph = build_temporal_graph(video_id="video", mode="w", w_records=_w_records())
    pipeline = GraphQAPipeline(retriever=_retriever(), answerer=FirstChoiceAnswerer())
    result = pipeline.answer_question(
        graph=graph,
        question_id="q",
        question="When did the action <pick up mug> happen?",
        choices=["8 seconds", "30 seconds"],
        inputs={},
    )
    serialized_trace = str(result.retrieval.to_dict())
    assert "30 seconds" not in serialized_trace
    assert "30 seconds" in result.answer.prompt


def main() -> int:
    tests = [
        test_adapters_keep_rich_structure_and_hierarchy,
        test_program_is_question_only_and_general,
        test_compositional_retrieval_traverses_hierarchy_and_entities,
        test_bbox_question_records_graph_only_limitation,
        test_count_operator_produces_deduplicated_aggregate,
        test_choices_enter_only_after_retrieval,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
