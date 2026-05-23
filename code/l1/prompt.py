"""Prompt for L1 multimodal reasoner v3.2.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -c "from code.l1.prompt import L1_REASONER_PROMPT_ID, build_l1_reasoner_prompt"
"""

from __future__ import annotations

import json


L1_REASONER_PROMPT_ID = "l1_multimodal_reasoner"


def build_l1_reasoner_prompt(segment: dict, selected_frames: list[dict], visual_discovery: dict) -> str:
    payload = {
        "segment": {k: segment.get(k) for k in ("segment_id", "t_start", "t_end", "context_t_start", "context_t_end", "source_window_indices", "dominant_scene_type", "dominant_focus")},
        "selected_frames": [
            {"frame_id": f.get("frame_id"), "timestamp_sec": f.get("timestamp_sec"), "role": f.get("role")}
            for f in selected_frames
        ],
        "visual_discovery": visual_discovery,
        "w_action_summaries": segment.get("w_action_summaries", []),
        "scene_summaries": segment.get("scene_summaries", []),
        "rolling_context_summaries": segment.get("rolling_context_summaries", []),
        "stable_objects_in_segment": segment.get("stable_objects_in_segment", []),
    }
    return (
        "You are constructing final L1 micro-action events for an egocentric video segment.\n\n"
        "Core instruction: produce L1 micro-actions, not broad L2/activity summaries.\n"
        "An L1 event is a temporally localized human-object micro-action. It should usually correspond to one concrete object interaction, object state change, object location change, object relation change, part manipulation, or meaningful support/contact action.\n\n"
        "L1 is allowed to be uncertain. If exact object part or exact action is unclear, do not invent specificity. Use event_subtype=\"ambiguous_part_manipulation\" and evidence_type=\"inferred\" or observed with explicit uncertainty. Use cautious object names like \"top control area\", \"handle area\", \"object part\", or \"control area\" when the specific part is not clearly visible. Do not convert a cautious description into a too-specific structured object.\n\n"
        "W_action is noisy evidence. Confirm it only when selected frames and context support it. Correct it when object identity or action meaning is better understood from context. Downgrade it when it is only support/contact/continuation. Mark it too_vague, too_specific, partially_supported, or likely_wrong when appropriate.\n\n"
        "Visual discovery is evidence. Every visual candidate must be accepted, merged, used as support, or rejected. No visual candidate may disappear silently. If a candidate is support_only, it should link to the event it supports or be represented as support/state/continuation.\n\n"
        "Important anti-overcompression instruction: do not merge multiple distinct micro-actions into one long L1 event just because they involve the same object or same appliance. Same object does not mean same event. Adjacent interactions with the same object can be distinct L1 events.\n\n"
        "Long-duration constraint: a core_effect L1 event should usually be short and temporally localized. If an event spans more than about 3 seconds, it must be state_or_continuation, support_contact, ambiguous_part_manipulation, or split into smaller events. Do not output a 5+ second press/button-like single core_effect unless visual evidence clearly shows one continuous press-like action. If describing a long interaction with an object area, use ambiguous_part_manipulation or activity_context_or_l2_hint, not a single confident core_effect.\n\n"
        "Structured label consistency: micro_action_description, verb, direct_object.name, part_of, and event_subtype must agree. If the description says top control area or handle area, do not label direct_object.name as a specific part like button unless that exact part is visually clear. If exact part identity is unclear, use direct_object.name=\"top control area\", \"control area\", \"handle area\", or \"object part\" and explain uncertainty. Do not mark such cases as high-confidence specific_micro_action.\n\n"
        "Event subtypes: core_effect, support_contact, state_or_continuation, ambiguous_part_manipulation, missing_from_W.\n"
        "Evidence labels: observed means directly visible or strongly indicated by visible object motion/state change. inferred means partly occluded/subtle/context-dependent but supported by selected frames and structured context. hypothesized is plausible but insufficient and must not become final L1.\n"
        "Part-whole rule: direct_object.name is the manipulated object or part. direct_object.part_of is the parent/whole object. If direct object is a whole object, part_of=\"none\". If direct object is a component/part/area, part_of is the larger object. If uncertain, use \"unknown\" and explain.\n\n"
        "Return ONLY raw JSON with these top-level fields: segment_id, segment_time, activity_context_or_l2_hint, visual_candidate_ledger, w_action_audit, l1_events, rejected_or_hypothesized_actions, reasoning_summary, parse_confidence.\n"
        "visual_candidate_ledger entries must contain candidate_id, candidate_description, disposition, linked_l1_event_ids, reason.\n"
        "w_action_audit entries must contain window_idx, w_action, decision, preferred_l1_interpretation, linked_l1_event_ids, reason, evidence.\n"
        "l1_events entries must contain event_id, micro_action_description, verb, direct_object{name,part_of,state_before,state_after,location_before,location_after}, t_start_est, t_end_est, event_subtype, evidence_type, should_create_l1, specificity_level, why_this_is_l1, why_not_more_specific, confidence, confidence_rationale, visual_evidence, structured_evidence, supporting_contacts_or_context, uncertainty_notes, hallucination_risk.\n\n"
        "Critical requirements: every visual discovery candidate must appear in visual_candidate_ledger. Every W_action item must appear in w_action_audit. A visual candidate with disposition merged_into_l1 must have a non-empty linked_l1_event_ids list. Do not use merged_into_l1 when the candidate is only contextual support; use support_for_l1. Do not output long core_effect events that combine several micro-actions. Do not label every final event observed; use inferred when evidence is occluded, subtle, or context-dependent. Do not use GT.\n\n"
        "Evidence payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
