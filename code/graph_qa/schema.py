"""Canonical, graph-builder-neutral temporal graph schema.

The retrieval backend depends only on these classes. W, L1, future L2/L3,
AVA, and other graph builders must enter through adapters rather than adding
builder-specific conditions to retrieval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Literal


Direction = Literal["out", "in", "both"]


@dataclass(frozen=True)
class TemporalSpan:
    """A half-open-ish video interval; overlap is inclusive at boundaries."""

    start: float
    end: float
    video_id: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Temporal span starts before zero: {self.start}")
        if self.end < self.start:
            raise ValueError(f"Temporal span ends before it starts: {self.start}, {self.end}")

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def overlaps(self, start: float, end: float, margin: float = 0.0) -> bool:
        return self.end >= start - margin and self.start <= end + margin


@dataclass
class GraphNode:
    """A typed graph node with optional temporal grounding.

    ``kind`` is intentionally a string, not a closed enum. New builders can
    introduce node types without changing the retrieval engine.
    """

    id: str
    kind: str
    text: str
    span: TemporalSpan | None = None
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """A directed, typed relationship between canonical graph nodes."""

    id: str
    source_id: str
    target_id: str
    relation: str
    text: str = ""
    span: TemporalSpan | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalGraph:
    """Small in-memory temporal property graph used by graph QA."""

    def __init__(self, video_id: str, graph_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.video_id = video_id
        self.graph_id = graph_id
        self.metadata = metadata or {}
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    def add_node(self, node: GraphNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate graph node id: {node.id}")
        if node.span is not None and node.span.video_id != self.video_id:
            raise ValueError(f"Node {node.id} belongs to video {node.span.video_id}, not {self.video_id}")
        self.nodes[node.id] = node
        self._outgoing[node.id] = []
        self._incoming[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"Duplicate graph edge id: {edge.id}")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(
                f"Edge {edge.id} references missing nodes: {edge.source_id} -> {edge.target_id}"
            )
        self.edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)

    def incident_edges(
        self,
        node_id: str,
        direction: Direction = "both",
        relations: set[str] | None = None,
    ) -> Iterator[GraphEdge]:
        edge_ids: list[str] = []
        if direction in {"out", "both"}:
            edge_ids.extend(self._outgoing.get(node_id, ()))
        if direction in {"in", "both"}:
            edge_ids.extend(self._incoming.get(node_id, ()))
        seen: set[str] = set()
        for edge_id in edge_ids:
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edge = self.edges[edge_id]
            if relations is None or edge.relation in relations:
                yield edge

    def neighbors(
        self,
        node_id: str,
        direction: Direction = "both",
        relations: set[str] | None = None,
    ) -> Iterator[tuple[GraphNode, GraphEdge]]:
        for edge in self.incident_edges(node_id, direction=direction, relations=relations):
            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
            yield self.nodes[other_id], edge

    def temporal_nodes(
        self,
        start: float,
        end: float,
        margin: float = 0.0,
        kinds: set[str] | None = None,
    ) -> list[GraphNode]:
        return [
            node
            for node in self.nodes.values()
            if node.span is not None
            and node.span.overlaps(start, end, margin)
            and (kinds is None or node.kind in kinds)
        ]

    def validate(self) -> None:
        for edge in self.edges.values():
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                raise ValueError(f"Dangling edge: {edge.id}")
            if edge.weight < 0:
                raise ValueError(f"Negative edge weight: {edge.id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "graph_id": self.graph_id,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    def __len__(self) -> int:
        return len(self.nodes)


def flatten_attributes(value: Any, prefix: str = "") -> list[str]:
    """Turn nested properties into stable retrieval text without losing keys."""

    parts: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            parts.extend(flatten_attributes(value[key], child_prefix))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            parts.extend(flatten_attributes(item, prefix))
    elif value not in (None, "", "unknown", "none", "n/a"):
        parts.append(f"{prefix.replace('_', ' ')}: {value}" if prefix else str(value))
    return parts


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
