"""Prompt for scene_obs_v1 W-side scene observations.

Import check:
  python -c "from code.scene.observation_prompt import SCENE_OBS_PROMPT, SCENE_OBS_PROMPT_ID, SCENE_OBS_V2_PROMPT_ID, build_scene_obs_v2_prompt"
"""

from __future__ import annotations

SCENE_OBS_PROMPT_ID = "scene_obs_v1"
SCENE_OBS_V2_PROMPT_ID = "scene_obs_v2_object_context"

SCENE_OBS_PROMPT_BODY = """You are analyzing a 1-second egocentric video TARGET interval with BEFORE CONTEXT frames.
Output a compact scene/object/action-presence observation for the TARGET interval.

Output ONLY raw JSON. Do not wrap the answer in markdown or code fences.
Keep strings short.
visible_objects should include main scene objects even when no action is happening.
Include at most 8 visible_objects.
Prefer stable, task-relevant objects over tiny uncertain details.
Do not invent hidden objects.
Do not predict future actions.
Do not write intent predictions such as "the wearer is making coffee."
Prefer descriptive scene/object statements such as "mug near appliance on counter."
Use BEFORE CONTEXT only for scene continuity and object identity.
Use TARGET frames to decide action_presence for the current 1-second window.
Do not mark visible_action only because an action appeared in BEFORE CONTEXT.
No future frames. No crops.

action_presence.status rules:
- "visible_action": TARGET frames show clear hand-object contact, object motion, object state change, tool effect, or material/material transfer.
- "uncertain_action": hands are visible or near objects, but manipulation is unclear.
- "no_action": no visible hand-object action and no visible object change in TARGET frames.

Use "visible_action" only when TARGET frames show clear direct evidence:
- hand-object contact, or
- object motion, or
- object state change, or
- tool effect, or
- material transfer.

Use "uncertain_action" when hands are visible or near objects but the exact manipulation or object change is unclear.

Use "no_action" when no hand-object interaction or object change is visible.

Do not mark "visible_action" only because a hand is visible.
Do not give confidence 1.0 unless the action evidence is unambiguous.

Use this exact JSON schema:
{
  "scene_type": "<kitchen|bathroom|desk|living_room|bedroom|hallway|outdoor|vehicle|unknown>",
  "view_context": "<short phrase, e.g., counter area, cupboard area, sink area, tabletop>",
  "spatial_layout": "<short description of major layout>",
  "visible_objects": [
    {
      "name": "<short noun phrase>",
      "category": "<appliance|container|tool|utensil|food|furniture|fixture|surface|door|drawer|other|unknown>",
      "specific_type": "<more specific name or unknown>",
      "location": "<short phrase>",
      "relative_position": "<short phrase or unknown>",
      "state": "<short visible state, e.g., open, closed, upright, stationary, held, unknown>",
      "confidence": <0.0-1.0>
    }
  ],
  "hands": {
    "left_hand_visible": true,
    "right_hand_visible": false,
    "hand_locations": "<short phrase or none>"
  },
  "action_presence": {
    "status": "<visible_action|uncertain_action|no_action>",
    "confidence": <0.0-1.0>,
    "evidence": {
      "hand_visible": true,
      "hand_object_contact": false,
      "object_motion": false,
      "object_state_change": false,
      "tool_effect": false,
      "material_transfer": false
    },
    "likely_action_region": "<short phrase or none>",
    "no_action_reason": "<short phrase or none>"
  },
  "current_focus": "<short phrase>",
  "uncertainty_notes": "<short phrase or none>"
}
"""

SCENE_OBS_PROMPT = SCENE_OBS_PROMPT_BODY


def build_scene_obs_v2_prompt(object_context_text: str) -> str:
    """Build scene_obs_v2 prompt with object-only rolling context."""
    return (
        "Scene context so far:\n"
        f"{object_context_text}\n\n"
        "This context contains previously observed scene type, stable objects, "
        "object locations, and current focus.\n"
        "Use it only for object identity and scene grounding.\n"
        "It does not imply that any object is still visible in the TARGET frames.\n"
        "It does not imply any previous action continues or repeats.\n"
        "If the context and TARGET frames disagree, prefer TARGET frames.\n"
        "Use TARGET frames to decide action_presence.\n"
        "Do NOT include recent_action_evidence in the answer.\n"
        "Do NOT use previous verbs or previous W actions to infer the current window.\n"
        "Do not predict future actions.\n"
        "Do not write intent predictions such as \"the wearer is making coffee.\"\n"
        "Object context can help name visible objects, but cannot create an object "
        "that is not visually plausible in the TARGET frames.\n\n"
        + SCENE_OBS_PROMPT_BODY
    )
