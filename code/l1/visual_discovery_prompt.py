"""Prompt builder for visual-first L1 discovery v3.1.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.visual_discovery_prompt import L1_VISUAL_DISCOVERY_PROMPT_ID, build_l1_visual_discovery_prompt"
"""

from __future__ import annotations

import json

L1_VISUAL_DISCOVERY_PROMPT_ID = "l1_visual_discovery"


def build_l1_visual_discovery_prompt(segment: dict, selected_frames: list[dict]) -> str:
    context = {
        "segment": {k: segment.get(k) for k in ("segment_id", "t_start", "t_end", "context_t_start", "context_t_end", "source_window_indices", "dominant_scene_type", "dominant_focus")},
        "selected_frames": [{"frame_id": f["frame_id"], "timestamp_sec": f["timestamp_sec"], "role": f["role"]} for f in selected_frames],
        "scene_summaries": segment.get("scene_summaries", []),
        "rolling_context_summaries": segment.get("rolling_context_summaries", []),
        "stable_objects_in_segment": segment.get("stable_objects_in_segment", []),
    }
    return (
        "You are inspecting an egocentric video segment to discover candidate L1 micro-actions.\n\n"
        "This is a visual-first pass. Do not use W_action; it is intentionally hidden. Use full-frame images, segment time, selected frame timestamps, scene type, stable objects, current focus, and rolling context only as background.\n"
        "L1 is a temporally localized human-object micro-action involving a person/hand/body and an object, object part, surface, tool, or material.\n\n"
        "Every candidate must be marked as exactly one candidate_role: final_candidate, support_only, uncertain_candidate, or reject_hypothesis.\n"
        "Also say what visual evidence would promote the candidate, what uncertainty remains, and keep timestamps absolute.\n"
        "If evidence is plausible but unclear, use evidence_type=\"inferred\" and candidate_role=\"uncertain_candidate\". Do not mark hypothesized candidates as should_consider_for_final_l1=true.\n"
        "Do not use GT. Do not invent hidden objects. If exact part identity is unclear, use cautious descriptions like top area, control area, handle area, or object part.\n\n"
        "Context JSON without W_action:\n"
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        "Return ONLY raw JSON:\n"
        "{\n"
        '  "segment_id": "string", "visual_segment_summary": "string", "main_visible_objects": [],\n'
        '  "visual_micro_action_candidates": [{"candidate_id": "string", "micro_action_description": "string", "verb": "string", "direct_object": {"name": "string", "part_of": "none", "state_before": "unknown", "state_after": "unknown", "location_before": "unknown", "location_after": "unknown"}, "t_start_est": 0.0, "t_end_est": 0.0, "evidence_type": "observed", "candidate_role": "final_candidate", "visual_evidence": [], "why_this_may_be_l1": "string", "why_not_more_specific": "string", "uncertainty_notes": "string", "hallucination_risk": "low", "should_consider_for_final_l1": true}],\n'
        '  "possible_missing_or_subtle_actions": [{"description": "string", "time_range": [0.0, 0.0], "evidence_type": "inferred", "visual_reason": "string", "what_is_occluded_or_unclear": "string", "candidate_role": "uncertain_candidate", "should_consider_for_final_l1": false}],\n'
        '  "part_whole_notes": [{"object_or_part": "string", "likely_parent_object": "string", "reason": "string", "confidence": 0.0}],\n'
        '  "visual_uncertainty_summary": "string", "parse_confidence": 0.0\n'
        "}\n"
    )
