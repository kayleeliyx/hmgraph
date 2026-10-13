"""RAVU-style semantic retrieval, reranking, and compositional traversal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

import torch

from .models import TextEmbedder, TextReranker
from .program import Operation, QuestionProgram
from .schema import GraphNode, TemporalGraph, TemporalSpan, flatten_attributes


TEMPORAL_KINDS = {"action", "event", "observation", "state", "step", "activity"}
HIERARCHY_RELATIONS = {"contains", "part_of", "summarizes", "summarized_by"}
ENTITY_RELATIONS = {"involves", "uses", "near", "causes_state", "occurs_at", "inside", "positioned_at"}
TEMPORAL_RELATIONS = {"next", "previous", "precedes", "follows"}


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k_per_view: int = 24
    rerank_top_k: int = 32
    seed_top_k: int = 10
    final_top_k: int = 18
    reciprocal_rank_constant: int = 20
    traversal_decay: float = 0.78
    max_expanded_nodes: int = 80
    similarity_device: str = "cuda:0"


@dataclass(frozen=True)
class RetrievalDocument:
    id: str
    view: str
    text: str
    node_ids: tuple[str, ...]
    span: TemporalSpan | None = None


@dataclass
class EvidenceItem:
    node_id: str
    text: str
    kind: str
    span: TemporalSpan | None
    score: float
    dense_score: float = 0.0
    rerank_score: float = 0.0
    paths: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class RetrievalTrace:
    graph_id: str
    program: dict[str, Any]
    dense_candidates: list[dict[str, Any]]
    seed_nodes: list[str]
    operation_log: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    aggregates: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "program": self.program,
            "dense_candidates": self.dense_candidates,
            "seed_nodes": self.seed_nodes,
            "operation_log": self.operation_log,
            "aggregates": self.aggregates,
            "warnings": self.warnings,
            "evidence": [item.to_dict() for item in self.evidence],
        }


class VisualReferenceResolver(Protocol):
    """Optional AVA-style visual resolver kept outside graph-only evaluation."""

    def resolve(self, graph: TemporalGraph, program: QuestionProgram) -> dict[str, float]: ...


class GraphRetrievalIndex:
    """Dense multi-view index for one immutable graph."""

    def __init__(self, graph: TemporalGraph, embedder: TextEmbedder) -> None:
        self.graph = graph
        self.embedder = embedder
        self.documents = self._make_documents(graph)
        self.embeddings = embedder.encode_documents([document.text for document in self.documents])
        if len(self.documents) != self.embeddings.shape[0]:
            raise RuntimeError("Embedder returned a different number of rows than graph documents")

    @staticmethod
    def _make_documents(graph: TemporalGraph) -> list[RetrievalDocument]:
        documents: list[RetrievalDocument] = []
        for node in graph.nodes.values():
            primary_view = "event" if node.kind in TEMPORAL_KINDS else "entity"
            documents.append(
                RetrievalDocument(
                    id=f"node:{node.id}:summary",
                    view=primary_view,
                    text=_node_retrieval_text(node),
                    node_ids=(node.id,),
                    span=node.span,
                )
            )
            attribute_text = "; ".join(flatten_attributes(node.attributes))
            if attribute_text and attribute_text.lower() not in node.text.lower():
                documents.append(
                    RetrievalDocument(
                        id=f"node:{node.id}:attributes",
                        view="attribute",
                        text=f"{node.kind} {node.text}; {attribute_text}",
                        node_ids=(node.id,),
                        span=node.span,
                    )
                )
        for edge in graph.edges.values():
            source = graph.nodes[edge.source_id]
            target = graph.nodes[edge.target_id]
            relation_text = edge.text or f"{source.text} {edge.relation.replace('_', ' ')} {target.text}"
            documents.append(
                RetrievalDocument(
                    id=f"edge:{edge.id}",
                    view="relation",
                    text=f"relationship {edge.relation}: {relation_text}",
                    node_ids=(edge.source_id, edge.target_id),
                    span=edge.span or source.span or target.span,
                )
            )
        return documents


class CompositionalGraphRetriever:
    """Fixed retrieval backend reusable across graph-construction variants."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        reranker: TextReranker,
        config: RetrievalConfig | None = None,
        visual_resolver: VisualReferenceResolver | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.visual_resolver = visual_resolver
        self._indexes: dict[str, GraphRetrievalIndex] = {}

    def index(self, graph: TemporalGraph) -> GraphRetrievalIndex:
        index = self._indexes.get(graph.graph_id)
        if index is None:
            index = GraphRetrievalIndex(graph, self.embedder)
            self._indexes[graph.graph_id] = index
        return index

    def retrieve(self, graph: TemporalGraph, program: QuestionProgram) -> RetrievalTrace:
        index = self.index(graph)
        dense_candidates = self._dense_retrieve(index, program)
        seed_candidates = dense_candidates[: self.config.rerank_top_k]
        rerank_texts = [_contextual_node_text(graph, graph.nodes[item["node_id"]]) for item in seed_candidates]
        rerank_scores = self.reranker.score(program.retrieval_query, rerank_texts) if rerank_texts else []
        for candidate, rerank_score in zip(seed_candidates, rerank_scores):
            candidate["rerank_score"] = float(rerank_score)
            candidate["score"] = 0.35 * candidate["dense_score"] + 0.65 * float(rerank_score)
        seed_candidates.sort(key=lambda item: item["score"], reverse=True)
        seed_candidates = seed_candidates[: self.config.seed_top_k]

        scores = {item["node_id"]: float(item["score"]) for item in seed_candidates}
        dense_scores = {item["node_id"]: float(item["dense_score"]) for item in dense_candidates}
        initial_rerank = {item["node_id"]: float(item.get("rerank_score", 0.0)) for item in seed_candidates}
        paths = {item["node_id"]: ["dense_retrieval", "neural_rerank"] for item in seed_candidates}
        operation_log: list[dict[str, Any]] = []
        warnings: list[str] = []

        for operation in program.operations:
            before = len(scores)
            if operation.name in {"retrieve_nodes", "filter_time", "rerank_evidence", "answer"}:
                pass
            elif operation.name == "resolve_reference":
                if self.visual_resolver is None:
                    warnings.append(
                        "Question contains a visual/BBOX reference; graph-only mode cannot identify it. "
                        "The optional visual resolver was not enabled."
                    )
                else:
                    for node_id, score in self.visual_resolver.resolve(graph, program).items():
                        if node_id in graph.nodes and self._within_scope(graph.nodes[node_id], program):
                            self._merge_score(scores, paths, node_id, float(score), "visual_reference")
            elif operation.name == "expand_entities":
                self._expand_relations(graph, program, scores, paths, ENTITY_RELATIONS, operation)
            elif operation.name == "expand_hierarchy":
                self._expand_relations(graph, program, scores, paths, HIERARCHY_RELATIONS, operation)
            elif operation.name == "expand_temporal":
                self._expand_temporal(graph, program, scores, paths, operation)
            elif operation.name == "expand_neighbors":
                self._expand_relations(graph, program, scores, paths, None, operation)
            elif operation.name in {
                "select_temporal_spans",
                "deduplicate_events",
                "count_events",
                "order_temporally",
            }:
                # These operators affect evidence formatting/answer semantics;
                # canonical nodes are already uniquely keyed and timestamped.
                pass
            else:
                warnings.append(f"Unknown operation ignored: {operation.name}")
            operation_log.append(
                {"operation": operation.name, "arguments": operation.arguments, "before": before, "after": len(scores)}
            )
            if len(scores) > self.config.max_expanded_nodes:
                scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.config.max_expanded_nodes])

        node_ids = list(scores)
        final_texts = [_contextual_node_text(graph, graph.nodes[node_id]) for node_id in node_ids]
        final_rerank = self.reranker.score(program.retrieval_query, final_texts) if final_texts else []
        evidence: list[EvidenceItem] = []
        for node_id, rerank_score in zip(node_ids, final_rerank):
            node = graph.nodes[node_id]
            graph_score = scores[node_id]
            combined = 0.35 * graph_score + 0.65 * float(rerank_score)
            evidence.append(
                EvidenceItem(
                    node_id=node_id,
                    text=_contextual_node_text(graph, node),
                    kind=node.kind,
                    span=node.span,
                    score=combined,
                    dense_score=dense_scores.get(node_id, 0.0),
                    rerank_score=float(rerank_score or initial_rerank.get(node_id, 0.0)),
                    paths=paths.get(node_id, []),
                    attributes=node.attributes,
                    source=node.source,
                )
            )
        evidence.sort(key=lambda item: item.score, reverse=True)
        evidence = self._select_final_evidence(evidence, program)
        aggregates = self._derive_aggregates(evidence, program)
        if "order_temporally" in {operation.name for operation in program.operations}:
            evidence.sort(
                key=lambda item: (
                    item.span.start if item.span is not None else float("inf"),
                    -item.score,
                )
            )
        return RetrievalTrace(
            graph_id=graph.graph_id,
            program=program.to_dict(),
            dense_candidates=dense_candidates[: self.config.rerank_top_k],
            seed_nodes=[item["node_id"] for item in seed_candidates],
            operation_log=operation_log,
            aggregates=aggregates,
            warnings=warnings,
            evidence=evidence,
        )

    def _select_final_evidence(
        self,
        evidence: list[EvidenceItem],
        program: QuestionProgram,
    ) -> list[EvidenceItem]:
        """Guarantee timed evidence for temporal/count/order programs."""

        limit = self.config.final_top_k
        if program.intent not in {"localize", "count", "track", "order"}:
            return evidence[:limit]
        temporal = [item for item in evidence if item.span is not None and item.kind in TEMPORAL_KINDS]
        temporal_quota = min(len(temporal), max(1, (limit * 2) // 3))
        selected = temporal[:temporal_quota]
        selected_ids = {item.node_id for item in selected}
        selected.extend(item for item in evidence if item.node_id not in selected_ids)
        return selected[:limit]

    @staticmethod
    def _derive_aggregates(
        evidence: list[EvidenceItem],
        program: QuestionProgram,
    ) -> dict[str, Any]:
        operation_names = {operation.name for operation in program.operations}
        timed_events = [
            item
            for item in evidence
            if item.span is not None and item.kind in {"action", "event", "step", "activity"}
        ]
        ordered = sorted(timed_events, key=lambda item: (item.span.start, item.span.end, -item.score))
        aggregates: dict[str, Any] = {}
        if "select_temporal_spans" in operation_names:
            aggregates["selected_temporal_spans"] = [
                {
                    "node_id": item.node_id,
                    "start": item.span.start,
                    "end": item.span.end,
                    "text": item.text,
                    "score": item.score,
                }
                for item in sorted(timed_events, key=lambda item: item.score, reverse=True)
            ]
        if "count_events" in operation_names:
            clusters = _deduplicate_event_evidence(ordered)
            aggregates["retrieved_event_cluster_count"] = len(clusters)
            aggregates["retrieved_event_clusters"] = clusters
        if "order_temporally" in operation_names:
            aggregates["ordered_events"] = [
                {
                    "node_id": item.node_id,
                    "start": item.span.start,
                    "end": item.span.end,
                    "text": item.text,
                }
                for item in ordered
            ]
        return aggregates

    def _dense_retrieve(
        self,
        index: GraphRetrievalIndex,
        program: QuestionProgram,
    ) -> list[dict[str, Any]]:
        query_variants = [
            program.retrieval_query,
            f"event, action, and temporal evidence for: {program.retrieval_query}",
            f"entities, locations, states, and relationships for: {program.retrieval_query}",
        ]
        query_embeddings = self.embedder.encode_queries(query_variants)
        device = torch.device(self.config.similarity_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Dense similarity is configured for CUDA, but CUDA is unavailable")
        doc_embeddings = index.embeddings.to(device)
        query_embeddings = query_embeddings.to(device)
        similarities = query_embeddings @ doc_embeddings.T
        allowed_indices = [
            idx for idx, document in enumerate(index.documents) if self._document_allowed(document, program)
        ]
        by_node: dict[str, dict[str, Any]] = {}
        rrf_constant = self.config.reciprocal_rank_constant
        views = ("event", "entity", "attribute", "relation")
        for query_index in range(similarities.shape[0]):
            for view in views:
                subset = [idx for idx in allowed_indices if index.documents[idx].view == view]
                ranked = sorted(subset, key=lambda idx: float(similarities[query_index, idx]), reverse=True)
                for rank, document_index in enumerate(ranked[: self.config.dense_top_k_per_view], start=1):
                    document = index.documents[document_index]
                    raw_similarity = float(similarities[query_index, document_index])
                    for node_id in document.node_ids:
                        candidate = by_node.setdefault(
                            node_id,
                            {
                                "node_id": node_id,
                                "dense_score": 0.0,
                                "score": 0.0,
                                "matches": [],
                            },
                        )
                        candidate["dense_score"] += 1.0 / (rrf_constant + rank)
                        candidate["matches"].append(
                            {
                                "query_variant": query_index,
                                "view": view,
                                "document_id": document.id,
                                "similarity": raw_similarity,
                                "rank": rank,
                            }
                        )
        if not by_node:
            return []
        max_score = max(item["dense_score"] for item in by_node.values()) or 1.0
        for item in by_node.values():
            item["dense_score"] /= max_score
            item["score"] = item["dense_score"]
        return sorted(by_node.values(), key=lambda item: item["dense_score"], reverse=True)

    @staticmethod
    def _document_allowed(document: RetrievalDocument, program: QuestionProgram) -> bool:
        if program.temporal_scope is None or document.span is None:
            return True
        start, end = program.temporal_scope
        margin = 2.0 if program.intent == "localize" else 0.5
        return document.span.overlaps(start, end, margin)

    @staticmethod
    def _within_scope(node: GraphNode, program: QuestionProgram) -> bool:
        if program.temporal_scope is None or node.span is None:
            return True
        start, end = program.temporal_scope
        margin = 2.0 if program.intent == "localize" else 0.5
        return node.span.overlaps(start, end, margin)

    def _expand_relations(
        self,
        graph: TemporalGraph,
        program: QuestionProgram,
        scores: dict[str, float],
        paths: dict[str, list[str]],
        relations: set[str] | None,
        operation: Operation,
    ) -> None:
        frontier = list(scores)
        hops = max(1, int(operation.arguments.get("hops", 1)))
        for hop in range(hops):
            additions: list[tuple[str, float, str]] = []
            for node_id in frontier:
                parent_score = scores[node_id]
                for neighbor, edge in graph.neighbors(node_id, direction="both", relations=relations):
                    if not self._within_scope(neighbor, program):
                        continue
                    score = parent_score * (self.config.traversal_decay ** (hop + 1)) * edge.weight
                    additions.append((neighbor.id, score, f"{operation.name}:{edge.relation}:{node_id}"))
            frontier = []
            for node_id, score, path in additions:
                previous = scores.get(node_id, -1.0)
                self._merge_score(scores, paths, node_id, score, path)
                if score > previous:
                    frontier.append(node_id)

    def _expand_temporal(
        self,
        graph: TemporalGraph,
        program: QuestionProgram,
        scores: dict[str, float],
        paths: dict[str, list[str]],
        operation: Operation,
    ) -> None:
        seeds = list(scores)
        before_hops = int(operation.arguments.get("before", 1))
        after_hops = int(operation.arguments.get("after", 1))
        for seed_id in seeds:
            self._walk_direction(
                graph, program, scores, paths, seed_id, "in", before_hops, "temporal_before"
            )
            self._walk_direction(
                graph, program, scores, paths, seed_id, "out", after_hops, "temporal_after"
            )

    def _walk_direction(
        self,
        graph: TemporalGraph,
        program: QuestionProgram,
        scores: dict[str, float],
        paths: dict[str, list[str]],
        seed_id: str,
        direction: str,
        hops: int,
        label: str,
    ) -> None:
        frontier = [seed_id]
        seed_score = scores[seed_id]
        for hop in range(max(0, hops)):
            next_frontier: list[str] = []
            for node_id in frontier:
                for neighbor, edge in graph.neighbors(
                    node_id, direction=direction, relations=TEMPORAL_RELATIONS
                ):
                    if not self._within_scope(neighbor, program):
                        continue
                    score = seed_score * (self.config.traversal_decay ** (hop + 1))
                    self._merge_score(scores, paths, neighbor.id, score, f"{label}:{edge.relation}:{node_id}")
                    next_frontier.append(neighbor.id)
            frontier = next_frontier

    @staticmethod
    def _merge_score(
        scores: dict[str, float],
        paths: dict[str, list[str]],
        node_id: str,
        score: float,
        path: str,
    ) -> None:
        scores[node_id] = max(scores.get(node_id, 0.0), score)
        node_paths = paths.setdefault(node_id, [])
        if path not in node_paths:
            node_paths.append(path)


def _node_retrieval_text(node: GraphNode) -> str:
    time_text = ""
    if node.span is not None:
        time_text = f" from {node.span.start:.2f} to {node.span.end:.2f} seconds"
    aliases = f"; aliases: {', '.join(node.aliases)}" if node.aliases else ""
    return f"{node.kind}{time_text}: {node.text}{aliases}"


def _contextual_node_text(graph: TemporalGraph, node: GraphNode) -> str:
    parts = [_node_retrieval_text(node)]
    relations: list[str] = []
    for neighbor, edge in graph.neighbors(node.id, direction="both"):
        if edge.source_id == node.id:
            relations.append(f"{edge.relation.replace('_', ' ')} -> {neighbor.text}")
        else:
            relations.append(f"{edge.relation.replace('_', ' ')} <- {neighbor.text}")
        if len(relations) >= 8:
            break
    if relations:
        parts.append("relationships: " + "; ".join(relations))
    return "; ".join(parts)


def _deduplicate_event_evidence(events: list[EvidenceItem], max_gap: float = 1.5) -> list[dict[str, Any]]:
    """Collapse adjacent same-action evidence for the count operator trace."""

    clusters: list[dict[str, Any]] = []
    for item in events:
        verb = str(item.attributes.get("verb") or item.kind).strip().lower()
        obj = str(item.attributes.get("direct_object") or "").strip().lower()
        signature = f"{verb}|{obj}"
        if (
            clusters
            and clusters[-1]["signature"] == signature
            and item.span.start <= clusters[-1]["end"] + max_gap
        ):
            clusters[-1]["end"] = max(clusters[-1]["end"], item.span.end)
            clusters[-1]["node_ids"].append(item.node_id)
            clusters[-1]["score"] = max(clusters[-1]["score"], item.score)
        else:
            clusters.append(
                {
                    "signature": signature,
                    "start": item.span.start,
                    "end": item.span.end,
                    "node_ids": [item.node_id],
                    "text": item.text,
                    "score": item.score,
                }
            )
    return clusters
