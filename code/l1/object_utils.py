"""Object-key helpers for context-aware L1 diagnostics.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.object_utils import object_key, is_generic_object_name_safe"
"""

from __future__ import annotations

from typing import Any

try:
    from code.scene.object_utils import canonicalize_object_name, is_generic_object_name, is_part_object_name
except ImportError:
    from code.scene.object_utils import canonicalize_object_name, is_generic_object_name, is_part_object_name


def object_key(obj: dict | str) -> str:
    if isinstance(obj, dict):
        canonical = canonicalize_object_name(obj.get("canonical_name") or obj.get("raw_name") or obj.get("name") or "")
        part_of = canonicalize_object_name(obj.get("part_of") or "none")
    else:
        canonical = canonicalize_object_name(obj)
        part_of = "none"
    return f"{canonical}|part_of={part_of}"


def is_generic_object_name_safe(name: str) -> bool:
    return is_generic_object_name(name)


def extract_name_partof_from_w_action(action: dict) -> dict:
    direct_rich = action.get("direct_object_rich") or {}
    raw_name = direct_rich.get("name") or action.get("direct_object") or "unknown"
    specific_type = direct_rich.get("specific_type")
    if is_generic_object_name(raw_name) and specific_type and not is_generic_object_name(specific_type):
        raw_name = specific_type
    canonical = canonicalize_object_name(raw_name)
    part_of = "none"
    spatial = action.get("spatial_context") or {}
    attrs = action.get("object_attributes") or {}
    attr_part = canonicalize_object_name(attrs.get("part") or "none")

    if is_part_object_name(canonical):
        part_of = _infer_parent_for_part(canonical, spatial)
    elif attr_part not in {"none", "unknown", "whole_object", "whole object"} and attr_part != canonical:
        # Preserve the direct object as the parent and the manipulated part separately elsewhere.
        part_of = "none"
    return {
        "raw_name": canonicalize_object_name(raw_name),
        "canonical_name": canonical,
        "part_of": part_of,
        "source": "W_action",
    }


def _infer_parent_for_part(part_name: str, spatial: dict) -> str:
    contained = canonicalize_object_name(spatial.get("contained_in") or "none")
    if contained not in {"none", "unknown"} and not is_generic_object_name(contained):
        return contained
    near = [
        canonicalize_object_name(item)
        for item in (spatial.get("near") or [])
        if canonicalize_object_name(item) not in {"none", "unknown"} and not is_generic_object_name(item)
    ]
    if len(near) == 1:
        return near[0]
    location = canonicalize_object_name(spatial.get("location") or "")
    for marker in ("on top of ", "on the top of ", "on ", "of "):
        if location.startswith(marker):
            candidate = canonicalize_object_name(location[len(marker) :])
            if candidate.startswith("the "):
                candidate = candidate[4:]
            if candidate.startswith("top of "):
                candidate = candidate[7:]
            if candidate not in {"none", "unknown", ""} and not is_generic_object_name(candidate):
                return candidate
    return "none"


def maybe_part_object_from_w_action(action: dict) -> dict | None:
    attrs = action.get("object_attributes") or {}
    part = canonicalize_object_name(attrs.get("part") or "none")
    if part in {"none", "unknown", "whole_object", "whole object"}:
        return None
    direct = extract_name_partof_from_w_action(action)
    if part == direct["canonical_name"]:
        return None
    part_of = direct["canonical_name"] if direct["canonical_name"] not in {"unknown", "none"} else "none"
    return {
        "raw_name": part,
        "canonical_name": part,
        "part_of": part_of,
        "source": "W_action",
    }
