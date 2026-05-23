"""Warning cleanup for L1 postprocess v3.2.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.warning_cleanup import cleanup_l1_warnings"
"""

from __future__ import annotations


def cleanup_l1_warnings(event: dict) -> tuple[dict, list[str]]:
    updated = dict(event)
    original = list(event.get("warnings") or [])
    kept, notes = [], []
    conf = _float(event.get("confidence"), 0.0)
    duration = _float(event.get("t_end"), 0.0) - _float(event.get("t_start"), 0.0)
    for warning in original:
        low = warning.lower()
        if "confidence too high" in low and conf <= 0.9:
            notes.append(f"removed stale warning: {warning}")
            continue
        if "confidence 1.0" in low and conf < 1.0:
            notes.append(f"removed stale warning: {warning}")
            continue
        if "long core_effect" in low and not (event.get("event_subtype") == "core_effect" and duration > 3.0):
            notes.append(f"removed stale warning: {warning}")
            continue
        kept.append(warning)
    updated["original_warnings"] = original
    updated["warnings"] = kept
    if notes:
        updated["warning_cleanup_notes"] = notes
    return updated, notes


def _float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
