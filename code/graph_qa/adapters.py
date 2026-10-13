"""Adapters from HMGraph W/L1 outputs into the canonical temporal graph."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Literal

from .schema import GraphEdge, GraphNode, TemporalGraph, TemporalSpan, flatten_attributes


GraphMode = Literal["w", "l1", "w+l1"]
EVENT_KINDS = {"action", "event", "observation", "step", "activity"}


def _name(value: Any, default: str = "unknown") -> str:
    if isinstance(value, dict):
        value = value.get("name", default)
    text = str(value or default).strip()
    return text if text else default


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _known(value: Any) -> bool:
    return value not in (None, "", "unknown", "none", "n/a", [], {})


def _span(record: dict[str, Any], video_id: str, default_end: float | None = None) -> TemporalSpan:
    start = float(record.get("event_t_start", record.get("t_start", 0.0)))
    end_value = record.get("event_t_end", record.get("t_end", default_end))
    end = float(end_value if end_value is not None else start)
    return TemporalSpan(start=max(0.0, start), end=max(start, end), video_id=video_id)


def _upsert_entity(graph: TemporalGraph, name: str, aliases: Iterable[str] = ()) -> str:
    canonical = name.strip() or "unknown"
    node_id = f"entity:{_slug(canonical)}"
    if node_id not in graph.nodes:
        graph.add_node(
            GraphNode(
                id=node_id,
                kind="entity",
                text=canonical,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias and alias != canonical)),
                attributes={"name": canonical},
                source={"adapter_created": True},
            )
        )
    return node_id


def _upsert_location(graph: TemporalGraph, name: str) -> str:
    node_id = f"location:{_slug(name)}"
    if node_id not in graph.nodes:
        graph.add_node(
            GraphNode(
                id=node_id,
                kind="location",
                text=name,
                attributes={"name": name},
                source={"adapter_created": True},
            )
        )
    return node_id


def _add_relation(
    graph: TemporalGraph,
    source_id: str,
    target_id: str,
    relation: str,
    *,
    span: TemporalSpan | None = None,
    text: str = "",
    attributes: dict[str, Any] | None = None,
) -> None:
    base = f"{source_id}:{relation}:{target_id}"
    suffix = sum(1 for edge_id in graph.edges if edge_id.startswith(base))
    graph.add_edge(
        GraphEdge(
            id=f"{base}:{suffix}",
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            text=text,
            span=span,
            attributes=attributes or {},
        )
    )


def _action_text(action: dict[str, Any]) -> str:
    verb = str(action.get("verb") or "unknown action")
    obj = _name(action.get("direct_object"))
    parts = [f"{verb} {obj}".strip()]
    for label, value in (
        ("hand", action.get("hand") or (action.get("interaction") or {}).get("hand")),
        ("with", action.get("with") or (action.get("interaction") or {}).get("tool_used")),
        ("purpose", action.get("purpose") or (action.get("interaction") or {}).get("why_this_verb")),
        ("manner", action.get("manner") or (action.get("interaction") or {}).get("motion_direction")),
    ):
        if _known(value):
            parts.append(f"{label}: {value}")
    spatial = action.get("spatial_context") or {}
    for key in ("location", "relative_position", "contained_in", "near"):
        if _known(spatial.get(key)):
            parts.extend(flatten_attributes({key: spatial[key]}))
    return "; ".join(parts)


def _add_action_relations(
    graph: TemporalGraph,
    node: GraphNode,
    action: dict[str, Any],
) -> None:
    obj_name = _name(action.get("direct_object"))
    if _known(obj_name):
        rich = action.get("direct_object_rich") or action.get("direct_object")
        aliases = []
        if isinstance(rich, dict):
            aliases = [str(rich.get(key)) for key in ("specific_type", "category") if _known(rich.get(key))]
        entity_id = _upsert_entity(graph, obj_name, aliases)
        _add_relation(graph, node.id, entity_id, "involves", span=node.span, text=f"acts on {obj_name}")

    interaction = action.get("interaction") or {}
    tool = action.get("with") or interaction.get("tool_used")
    if _known(tool):
        tool_id = _upsert_entity(graph, _name(tool))
        _add_relation(graph, node.id, tool_id, "uses", span=node.span, text=f"uses {_name(tool)}")

    spatial = action.get("spatial_context") or {}
    for key, relation in (
        ("location", "occurs_at"),
        ("contained_in", "inside"),
        ("relative_position", "positioned_at"),
    ):
        value = spatial.get(key)
        if _known(value):
            location_id = _upsert_location(graph, str(value))
            _add_relation(graph, node.id, location_id, relation, span=node.span, text=str(value))

    for nearby in spatial.get("near") or []:
        if _known(nearby):
            entity_id = _upsert_entity(graph, str(nearby))
            _add_relation(graph, node.id, entity_id, "near", span=node.span, text=f"near {nearby}")

    changes = interaction.get("object_change")
    if _known(changes) and changes != "no_visible_change" and obj_name != "unknown":
        state_id = f"state:{node.id}:{_slug(str(changes))}"
        graph.add_node(
            GraphNode(
                id=state_id,
                kind="state",
                text=f"{obj_name}: {changes}",
                span=node.span,
                attributes={"object": obj_name, "change": changes},
                source=node.source,
            )
        )
        _add_relation(graph, node.id, state_id, "causes_state", span=node.span, text=str(changes))


def _add_temporal_chain(graph: TemporalGraph, node_ids: list[str]) -> None:
    ordered = sorted(
        node_ids,
        key=lambda node_id: (
            graph.nodes[node_id].span.start if graph.nodes[node_id].span else float("inf"),
            graph.nodes[node_id].span.end if graph.nodes[node_id].span else float("inf"),
            node_id,
        ),
    )
    for before_id, after_id in zip(ordered, ordered[1:]):
        before = graph.nodes[before_id]
        after = graph.nodes[after_id]
        _add_relation(
            graph,
            before_id,
            after_id,
            "next",
            span=TemporalSpan(before.span.end, after.span.start, graph.video_id)
            if before.span and after.span and after.span.start >= before.span.end
            else None,
            text=f"{before.text} then {after.text}",
        )


def add_w_records(graph: TemporalGraph, records: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Add W actions and return source-window to canonical-node mapping."""

    window_nodes: dict[int, list[str]] = defaultdict(list)
    event_ids: list[str] = []
    for record in records:
        window_idx = int(record.get("window_idx", len(window_nodes)))
        parsed = record.get("parsed") or {}
        actions = parsed.get("actions") or record.get("valid_actions") or []
        for offset, action in enumerate(actions):
            action_idx = int(action.get("action_idx", offset))
            node_id = f"w:{window_idx:05d}:action:{action_idx}"
            span = _span(
                {
                    **action,
                    "t_start": record.get("t_start", window_idx),
                    "t_end": record.get("t_end", window_idx + 1),
                },
                graph.video_id,
            )
            attributes = {
                "level": "w",
                "verb": action.get("verb", "unknown"),
                "direct_object": _name(action.get("direct_object")),
                "hand": action.get("hand") or (action.get("interaction") or {}).get("hand"),
                "with": action.get("with") or (action.get("interaction") or {}).get("tool_used"),
                "purpose": action.get("purpose") or (action.get("interaction") or {}).get("why_this_verb"),
                "manner": action.get("manner") or (action.get("interaction") or {}).get("motion_direction"),
                "confidence": action.get("confidence"),
                "object_attributes": action.get("object_attributes") or {},
                "spatial_context": action.get("spatial_context") or {},
                "interaction": action.get("interaction") or {},
            }
            node = GraphNode(
                id=node_id,
                kind="action",
                text=_action_text(action),
                span=span,
                attributes=attributes,
                source={
                    "graph_level": "w",
                    "window_idx": window_idx,
                    "action_idx": action_idx,
                    "prompt_id": record.get("prompt_id"),
                    "model_id": record.get("model_id"),
                },
            )
            graph.add_node(node)
            _add_action_relations(graph, node, action)
            window_nodes[window_idx].append(node_id)
            event_ids.append(node_id)
    _add_temporal_chain(graph, event_ids)
    return dict(window_nodes)


def add_l1_records(graph: TemporalGraph, records: list[dict[str, Any]]) -> dict[str, str]:
    """Add L1 events and return source L1 id to canonical-node mapping."""

    mapping: dict[str, str] = {}
    event_ids: list[str] = []
    for offset, record in enumerate(records):
        raw_id = str(record.get("l1_id") or f"l1_{offset:05d}")
        node_id = f"l1:{raw_id}"
        action = {
            **record,
            "direct_object": record.get("direct_object", "unknown"),
            "interaction": record.get("interaction") or {},
            "spatial_context": record.get("spatial_context")
            or ({"location": record.get("location")} if _known(record.get("location")) else {}),
        }
        description = str(record.get("micro_action_description") or _action_text(action))
        signature = f"{record.get('verb', 'unknown')} {_name(record.get('direct_object'))}".strip()
        text = description if signature.lower() in description.lower() else f"{signature}; {description}"
        node = GraphNode(
            id=node_id,
            kind="action",
            text=text,
            span=_span(record, graph.video_id),
            attributes={
                "level": "l1",
                "verb": record.get("verb", "unknown"),
                "direct_object": _name(record.get("direct_object")),
                "hand": record.get("hand"),
                "with": record.get("with"),
                "purpose": record.get("purpose"),
                "manner": record.get("manner"),
                "event_subtype": record.get("event_subtype"),
                "evidence_type": record.get("evidence_type"),
                "specificity_level": record.get("specificity_level"),
                "confidence": record.get("confidence"),
                "source_window_idxs": record.get("source_window_idxs")
                or record.get("source_windows")
                or [],
            },
            source={
                "graph_level": "l1",
                "l1_id": raw_id,
                "segment_id": record.get("segment_id"),
                "source": record.get("source"),
            },
        )
        graph.add_node(node)
        _add_action_relations(graph, node, action)
        mapping[raw_id] = node_id
        event_ids.append(node_id)
    _add_temporal_chain(graph, event_ids)
    return mapping


def _add_hierarchy_edges(
    graph: TemporalGraph,
    l1_records: list[dict[str, Any]],
    l1_mapping: dict[str, str],
    window_nodes: dict[int, list[str]],
) -> None:
    for offset, record in enumerate(l1_records):
        raw_id = str(record.get("l1_id") or f"l1_{offset:05d}")
        parent_id = l1_mapping[raw_id]
        source_windows = record.get("source_window_idxs") or record.get("source_windows") or []
        for window_idx in source_windows:
            for child_id in window_nodes.get(int(window_idx), ()):
                _add_relation(
                    graph,
                    parent_id,
                    child_id,
                    "contains",
                    span=graph.nodes[child_id].span,
                    text=f"{graph.nodes[parent_id].text} contains {graph.nodes[child_id].text}",
                )


def build_temporal_graph(
    *,
    video_id: str,
    mode: GraphMode,
    w_records: list[dict[str, Any]] | None = None,
    l1_records: list[dict[str, Any]] | None = None,
) -> TemporalGraph:
    """Build W-only, L1-only, or hierarchical W+L1 graph with one API."""

    if mode not in {"w", "l1", "w+l1"}:
        raise ValueError(f"Unsupported graph mode: {mode}")
    if mode in {"w", "w+l1"} and w_records is None:
        raise ValueError(f"{mode} graph requires W records")
    if mode in {"l1", "w+l1"} and l1_records is None:
        raise ValueError(f"{mode} graph requires L1 records")

    graph = TemporalGraph(
        video_id=video_id,
        graph_id=f"{video_id}:{mode}",
        metadata={"mode": mode, "adapter_version": "hmgraph_w_l1_v1"},
    )
    window_nodes: dict[int, list[str]] = {}
    l1_mapping: dict[str, str] = {}
    if mode in {"w", "w+l1"}:
        window_nodes = add_w_records(graph, w_records or [])
    if mode in {"l1", "w+l1"}:
        l1_mapping = add_l1_records(graph, l1_records or [])
    if mode == "w+l1":
        _add_hierarchy_edges(graph, l1_records or [], l1_mapping, window_nodes)
    graph.validate()
    return graph
