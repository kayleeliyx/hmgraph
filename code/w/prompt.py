"""Prompt and parser for rich detailed W outputs.

Inputs:
  Raw Gemma text output for a short W video window.
Outputs:
  Parsed list of rich atomic hand-object actions with object attributes,
  spatial context, interaction metadata, time hints, and confidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

PROMPT_ID = "w_atomic_v4"
ADAPTIVE_CROPS_PROMPT_ID = PROMPT_ID + "_adaptive_crops"

PROMPT = """You are analyzing a short egocentric cooking video clip.

You will receive frames from three temporal regions:
1. BEFORE CONTEXT: frames before the target interval.
2. TARGET: frames from the target interval.
3. AFTER CONTEXT: frames after the target interval.

Your job is to output a detailed audit record of visible hand-object
interactions occurring in the TARGET interval only.

Use BEFORE CONTEXT and AFTER CONTEXT only to help infer:
- object identity
- manipulated whole object versus contact part
- action meaning
- visible object state change
- whether an action in the TARGET interval is preparatory, support, or primary

Very important:
- The TARGET frames are visually labeled TARGET FRAME.
- Only actions visible in frames labeled TARGET FRAME may appear in the JSON
  output.
- Frames labeled BEFORE CONTEXT or AFTER CONTEXT are not part of the answer.
- If an action is visible only in BEFORE CONTEXT or AFTER CONTEXT frames, do
  not output it.
- If the target frames show no hand-object action, output {"actions": []},
  even if context frames show an action.
- The output must describe only actions that are visibly happening in the
  TARGET interval.
- BEFORE CONTEXT and AFTER CONTEXT are only supporting context.
- Do not output an action that occurs only before the TARGET interval.
- Do not output an action that occurs only after the TARGET interval.
- Do not infer that an action happened in TARGET just because it appears in
  AFTER CONTEXT.
- Do not backfill future actions into earlier target windows.
- When uncertain whether the action is visible in TARGET, output no action
  rather than using context.
- Do not summarize the whole multi-second sequence.
- Do not jump ahead to a future action unless it has already started visibly
  in the TARGET interval.
- If the TARGET interval contains only a preparatory sub-action, output that
  preparatory action.
- If the TARGET interval contains the meaningful object-level state change,
  output that action.
- If no hand-object action is visible in the TARGET interval, output
  {"actions": []}.

Action-validity rule:
Do not output an action only because an object is visible.
A valid action requires visible action evidence in the TARGET frames:
hand-object contact, object motion, object state change, tool effect, or
material/liquid transfer.
If the TARGET frames only show an object, a surface, a kitchen scene, or
context from earlier frames, output {"actions": []}.
If the hand is not visible and the object does not visibly move or change
state in the TARGET frames, output {"actions": []}.
When uncertain whether an action is actually visible in the TARGET frames,
output {"actions": []} rather than guessing.

Output a detailed audit record of visible hand-object interactions.
List all visible atomic actions occurring in the TARGET interval, in temporal
order. Keep W detailed: include preparatory, support, and primary manipulation
actions that are visible in the TARGET interval. Be as specific as possible
about the object and its visible state.

Use this exact JSON schema and output ONLY valid JSON:
{
  "actions": [
    {
      "verb": "<specific action verb or short verb phrase, action only, no object noun>",
      "direct_object": {
        "name": "<short noun phrase>",
        "category": "<food|ingredient|liquid|utensil|tool|cookware|dishware|container|appliance|packaging|surface|fixture|control|cleaning_item|body_part|waste|unknown>",
        "specific_type": "<more specific type, or unknown>",
        "visible_text": "<exact visible label text, or none>"
      },
      "object_attributes": {
        "color": ["<visible color>", "..."],
        "material": ["<plastic|metal|glass|ceramic|wood|paper|cardboard|fabric|food|liquid|unknown>", "..."],
        "shape": "<round|cylindrical|rectangular|flat|irregular|elongated|bowl-shaped|bottle-shaped|unknown>",
        "size": "<tiny|small|medium|large|unknown>",
        "state": ["<open|closed|on|off|full|empty|partly_full|whole|cut|peeled|dirty|clean|wet|dry|mixed|unknown>", "..."],
        "amount": "<none|trace|small_amount|medium_amount|large_amount|full|unknown>",
        "condition": ["<clean|dirty|wet|dry|cooked|raw|chopped|peeled|broken|unknown>", "..."],
        "part": "<handle|lid|cap|button|knob|rim|blade|spout|opening|surface|whole_object|unknown>"
      },
      "spatial_context": {
        "location": "<short visible location phrase, or unknown>",
        "relative_position": "<relative position phrase, or unknown>",
        "contained_in": "<container/location, or none>",
        "near": ["<nearby object>", "..."]
      },
      "interaction": {
        "hand": "<left|right|both|unknown>",
        "tool_used": "<noun phrase or none>",
        "contact_type": "<none|touch|grasp|hold|press|pinch|pull|push|support|pour_from|pour_into|cut_with|wipe_with|unknown>",
        "role": "<primary_manipulation|preparatory|support|observation>",
        "object_change": "<short phrase such as opened, moved, lifted, pressed, tilted, cut, or no_visible_change>",
        "motion_direction": "<up|down|left|right|forward|backward|inward|outward|clockwise|counterclockwise|toward|away|unknown>",
        "why_this_verb": "<short visible evidence for choosing this verb instead of a generic hand-pose verb>"
      },
      "time_hint": "<start_sec>-<end_sec within the TARGET interval only, normalized to the 1-second target interval, e.g. 0.1-0.6>",
      "confidence": <number from 0.0 to 1.0>
    }
  ]
}

Guidelines:
- The "verb" field must contain only the action, never the object. Put the
  object only in direct_object.name. Good: verb="pick up",
  direct_object.name="mug". Bad: verb="pick up mug".
- Pick the verb by visible object effect first, hand pose second. If an
  object visibly changes state, position, orientation, containment, contact,
  or activation in the TARGET interval, name that effect as the verb.
- Choose direct_object.name as the object whose state, position, orientation,
  containment, or activation changes in the TARGET interval. If the hand
  touches a small part of a larger manipulated object, put the small contact
  part in object_attributes.part and put the manipulated whole object in
  direct_object.name.
- Use specific manipulation verbs when visible, such as pick up, put down,
  open, close, press, turn, twist, tilt, pour, scrape, cut, slice, peel,
  rinse, wipe, scrub, squeeze, stir, mix, insert, remove, lift, lower, place,
  release, plug in, unplug, switch on, or switch off.
- Choose the verb that best describes the visible contact and object effect
  in the TARGET interval,
  not merely the hand pose. Prefer a verb that names what happens to the
  object: opened, closed, picked up, placed, pressed, turned, tilted, poured,
  cut, wiped, rinsed, inserted, removed, or moved.
- If a hand grasps or pulls a cupboard handle/knob and the cupboard door opens
  in the TARGET interval, output verb="open", direct_object.name="cupboard door"
  or "cupboard", and object_attributes.part="handle" or "knob".
- If a hand grasps or pushes a cupboard handle/knob and the cupboard door
  closes in the TARGET interval, output verb="close",
  direct_object.name="cupboard door" or "cupboard", and
  object_attributes.part="handle" or "knob".
- If contact with a handle/knob causes a door or drawer to move in the TARGET
  interval, use open, close, pull, or push for the door/drawer, not squeeze for
  the knob.
- For small object-part interactions, focus on visible state changes in the
  TARGET frames. A valid action may involve a small part such as a button,
  knob, handle, lid, cap, slot, opening, rim, blade, or spout, but the action
  must still be visibly supported in the TARGET frames by hand contact, object
  motion, state change, tool effect, or material transfer.
- Contrastive examples:
  - If the finger depresses a button: verb="press", direct_object.name="button",
    not verb="hold".
  - If the mug leaves the surface: verb="pick up", direct_object.name="mug",
    not verb="hold".
  - If the mug is put onto a surface: verb="place", direct_object.name="mug",
    not verb="hold".
  - If the door opens: verb="open", direct_object.name="cupboard door",
    not verb="pull".
  - If the lid moves into closed position: verb="close",
    direct_object.name="lid", not verb="hold".
  - If the orange turns in the hand: verb="rotate",
    direct_object.name="orange", not verb="hold".
  - If water flows over the orange: verb="rinse",
    direct_object.name="orange", not verb="hold".
  - If water stops: verb="turn off", direct_object.name="tap",
    not verb="touch".
  - If liquid leaves the container: verb="pour",
    direct_object.name="liquid", not verb="hold".
- "reach for" is valid when the hand approaches an object in the TARGET
  interval before contact is
  visible. If contact is visible, use a contact verb such as touch, grasp,
  pick up, press, pull, push, open, or move.
- "hold" is valid when the main visible action in the TARGET interval is
  maintaining an object's position with no visible object change. If the
  object is being moved,
  opened, closed, tilted, poured from, pressed, turned, cut, wiped, rinsed,
  inserted, removed, picked up, or put down, use that more specific verb.
- Bimanual rule: when the two hands perform DIFFERENT actions on DIFFERENT
  objects in the TARGET interval, emit ONE entry per hand, each with
  interaction.hand set to "left" or "right" (never "both" in this case) and
  interaction.role set to "primary_manipulation" on BOTH. Do not collapse two
  different hand actions into a single entry. Example: if the left hand lifts
  the lid of a food recycling bin while the right hand drops an orange skin
  into the bin, output two entries: (verb="open" or "lift", object="bin lid",
  hand="left") and (verb="throw" or "drop", object="orange skin", hand="right").
- If one hand performs a primary manipulation and the other hand only steadies
  or supports an object with no independent action, keep the old behavior:
  emit the primary as primary_manipulation and the support as role="support".
- Discard / transfer-into-container rule: when an object visibly leaves the
  hand and enters a bin, container, sink, or other receptacle in the TARGET
  interval, use verb="throw", "drop", "discard", or "dispose" as appropriate
  (not "place" -- "place" implies setting down on a surface, not releasing into
  a container). why_this_verb must cite the visible release and the receiving
  container, for example: "orange skin leaves the right hand and falls into
  the open recycling bin" or "tea bag drops into the mug". A brief contact
  followed by visible release is valid evidence for these verbs.
- Avoid generic verbs such as hold, reach for, touch, manipulate, use, handle,
  or operate when a more specific physical action is visible.
- Fill "why_this_verb" with concrete visual evidence from the TARGET interval,
  optionally clarified by nearby context, not intent alone. For example:
  "finger depresses the button", "mug leaves the shelf", "water flows over
  the orange", "hand keeps the bowl steady with no object change". If you use
  "hold" or "reach for", the explanation must make clear why no more specific
  object-effect verb is visible in the TARGET interval.
- interaction.why_this_verb must cite concrete TARGET-frame evidence.
  Bad evidence: "orange is visible", "orange is on the countertop", "object
  appears to be used", "likely being placed".
  Good evidence: "left hand contacts the mug handle in the target frames",
  "mug leaves the lower shelf in the target frames", "cupboard door swings
  open while hand grips knob", "water flows over the orange", "knife blade
  contacts and cuts the orange".
  If why_this_verb cannot cite visible hand contact, object motion, state
  change, tool effect, or material transfer in the TARGET frames, output
  {"actions": []}.
- If an object leaves a surface, shelf, container, or hand in the TARGET
  interval, use pick up, remove, lift, take out, put down, place, or release as
  appropriate.
- If liquid flows out of a container in the TARGET interval, use "pour".
- If water flows over an object in the TARGET interval, use "rinse".
- If a tool changes an object's surface, shape, or cleanliness in the TARGET
  interval, use the tool action such as cut, slice, peel, wipe, scrub, scrape,
  or stir.
- Use visible nearby frames to understand object identity and action intent,
  but do not change the time scope of the answer.
- BEFORE CONTEXT and AFTER CONTEXT may help disambiguate what the target
  action means, but they must not create extra actions in the output.
- Do not hallucinate an action that has not started visibly in the TARGET
  interval.
- If the current visible action in the TARGET interval is opening a cupboard
  before retrieving a mug, output the visible target action as "open cupboard
  door"; do not jump directly to "pick up mug" unless the mug is visibly being
  lifted or removed in the TARGET interval.
- If before/after context suggests a future or earlier action, but that action
  is not visible in the TARGET interval, do not output it.
- Do not summarize as cooking, preparing, operating, juicing, washing, or
  cleaning if a more specific visible action can be named.
- For object attributes, report only visible evidence. Use "unknown", "none",
  or empty lists when details are not visible.
- If no hand-object action is visible in the TARGET interval, output {"actions": []}.

Example A (bimanual lift + discard):
TARGET interval shows the left hand lifting the brown lid of the food
recycling bin while the right hand simultaneously releases a used orange skin
half into the open bin.
Correct output: two actions in the same JSON object.
  - {verb: "lift", direct_object.name: "lid of food recycling bin",
     interaction.hand: "left", interaction.role: "primary_manipulation",
     interaction.contact_type: "grasp", interaction.object_change: "lid lifted",
     why_this_verb: "left hand grips the lid and the lid moves up off the bin"}
  - {verb: "throw", direct_object.name: "orange skin",
     interaction.hand: "right", interaction.role: "primary_manipulation",
     interaction.contact_type: "none", interaction.object_change: "released into bin",
     why_this_verb: "orange skin leaves the right hand and falls into the open recycling bin"}

Example B (bimanual hold + cap):
TARGET interval shows the left hand holding a milk bottle upright while the
right hand places and rotates a cap onto the bottle opening.
Correct output: two actions in the same JSON object.
  - {verb: "place", direct_object.name: "lid", interaction.hand: "right",
     interaction.role: "primary_manipulation", interaction.contact_type: "press",
     interaction.object_change: "lid placed on bottle",
     why_this_verb: "right hand lowers the cap onto the bottle opening"}
  - {verb: "hold", direct_object.name: "milk bottle", interaction.hand: "left",
     interaction.role: "support", interaction.contact_type: "hold",
     interaction.object_change: "no_visible_change",
     why_this_verb: "left hand keeps the milk bottle steady with no object change"}
"""

ADAPTIVE_CROPS_PROMPT = (
    "You are given full-frame TARGET images and optional TARGET DETAIL CROPS.\n\n"
    "Use full-frame TARGET images to understand the scene, objects, and temporal scope.\n"
    "Use TARGET DETAIL CROPS only as supplemental zoomed-in evidence for small "
    "hand-object interactions.\n"
    "Do not ignore the full-frame images.\n"
    "Do not describe actions visible only in BEFORE CONTEXT.\n"
    "Do not infer an action only from a crop unless the crop is consistent with the "
    "full-frame TARGET sequence.\n"
    "If a crop shows a small part such as a button, knob, handle, lid, cap, slot, "
    "opening, rim, spout, switch, lever, dial, top, edge, or hinge, describe the "
    "visible action at the object/state-change level when supported by TARGET evidence.\n"
    "If the full frames and crops disagree, prefer the full-frame TARGET temporal evidence.\n\n"
    + PROMPT
)

OBJECT_CATEGORIES = {
    "food",
    "ingredient",
    "liquid",
    "utensil",
    "tool",
    "cookware",
    "dishware",
    "container",
    "appliance",
    "packaging",
    "surface",
    "fixture",
    "control",
    "cleaning_item",
    "body_part",
    "waste",
    "unknown",
}
VALID_HANDS = {"left", "right", "both", "unknown"}
VALID_ROLES = {"primary_manipulation", "preparatory", "support", "observation"}
STRONG_CONTACT_TYPES = {
    "grasp",
    "hold",
    "press",
    "pinch",
    "pull",
    "push",
    "pour_from",
    "pour_into",
    "cut_with",
    "wipe_with",
}
WEAK_CONTACT_TYPES = {"none", "touch", "support", "unknown"}
STRONG_OBJECT_CHANGE_KEYWORDS = {
    "open",
    "opened",
    "close",
    "closed",
    "lift",
    "lifted",
    "lower",
    "lowered",
    "pick",
    "picked",
    "remove",
    "removed",
    "take",
    "taken",
    "place",
    "placed",
    "put",
    "moved",
    "tilt",
    "tilted",
    "rotate",
    "rotated",
    "turn",
    "turned",
    "cut",
    "sliced",
    "peeled",
    "rinsed",
    "wiped",
    "scrubbed",
    "poured",
    "transferred",
    "inserted",
    "released",
}
MATERIAL_TRANSFER_VERBS = {"pour", "rinse", "sprinkle", "add", "transfer"}
TOOL_EFFECT_VERBS = {"cut", "slice", "peel", "wipe", "scrub", "scrape", "stir", "mix"}
DISCARD_VERBS = {"throw", "throw away", "drop", "discard", "dispose", "toss", "bin", "release", "let go"}
DISCARD_EVIDENCE_TERMS = {
    "into the bin",
    "into bin",
    "in the bin",
    "into the recycling",
    "recycling bin",
    "food waste",
    "trash",
    "rubbish",
    "compost",
    "lets go",
    "releases",
    "drops",
    "thrown",
    "tossed",
    "dropped",
    "lid opens",
    "lid raises",
    "bin opens",
    "orange skin enters",
    "skin falls",
    "peel falls",
    "waste enters",
}
REACH_VERBS = {"reach", "reach for"}
EMPTY_OR_WEAK_CHANGES = {"", "unknown", "none", "no_visible_change", "no visible change"}
VISIBILITY_ONLY_PHRASES = {
    "is visible",
    "visible on",
    "is on the counter",
    "is on the countertop",
    "appears to be",
    "likely",
    "probably",
}
ACTION_EVIDENCE_TERMS = {
    "hand",
    "grip",
    "grasp",
    "hold",
    "touch",
    "contact",
    "lift",
    "leaves",
    "moves",
    "opens",
    "closes",
    "flows",
    "cuts",
    "wipes",
}
REACH_EVIDENCE_TERMS = {"hand", "approach", "reaches", "moves toward", "extends toward"}
TRANSFER_EVIDENCE_TERMS = {"flow", "flows", "pour", "pours", "water", "liquid", "enters", "leaves", "over"}
TOOL_EVIDENCE_TERMS = {
    "knife",
    "blade",
    "sponge",
    "cloth",
    "tool",
    "contacts",
    "scrubs",
    "cuts",
    "wipes",
}


def _clean_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = " ".join(str(value).strip().lower().split())
    return text if text else default


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        value = [] if value in (None, "", "unknown") else [value]
    return [_clean_text(item) for item in value if _clean_text(item, "")] or ["unknown"]


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _direct_object_name(action: Dict[str, Any]) -> str:
    direct_object = action.get("direct_object", "")
    if isinstance(direct_object, dict):
        return _clean_text(direct_object.get("name"), "")
    return _clean_text(direct_object, "")


def _validity_rejection_reason(action: Dict[str, Any]) -> str | None:
    verb = _clean_text(action.get("verb"), "")
    direct_object_name = _direct_object_name(action)
    interaction = action.get("interaction", {})
    if not isinstance(interaction, dict):
        interaction = {}
    hand = _clean_text(action.get("hand") or interaction.get("hand"))
    contact_type = _clean_text(interaction.get("contact_type"))
    role = _clean_text(interaction.get("role"))
    object_change = _clean_text(interaction.get("object_change"), "")
    why_this_verb = _clean_text(interaction.get("why_this_verb"), "")
    try:
        confidence = float(action.get("confidence"))
    except (TypeError, ValueError):
        confidence = 1.0

    if role == "observation":
        return "role is observation"
    if confidence < 0.35:
        return "confidence below 0.35"

    is_reach = verb in REACH_VERBS or verb.startswith("reach")
    if is_reach:
        if (
            hand in {"left", "right", "both"}
            and direct_object_name not in {"", "none", "unknown"}
            and _contains_any(why_this_verb, REACH_EVIDENCE_TERMS)
        ):
            return None
        return "reach lacks visible target-frame hand approach evidence"

    weak_visibility = _contains_any(why_this_verb, VISIBILITY_ONLY_PHRASES) and not _contains_any(
        why_this_verb, ACTION_EVIDENCE_TERMS
    )
    if weak_visibility:
        return "why_this_verb cites object visibility/presence without action evidence"

    if verb in MATERIAL_TRANSFER_VERBS and _contains_any(
        f"{why_this_verb} {object_change}", TRANSFER_EVIDENCE_TERMS
    ):
        return None

    # Discard / transfer-into-container verbs are valid when there is visible
    # release evidence in TARGET frames, even if contact is brief by nature.
    if verb in DISCARD_VERBS and (
        hand in {"left", "right", "both"}
        and (
            _contains_any(why_this_verb, DISCARD_EVIDENCE_TERMS)
            or _contains_any(object_change, DISCARD_EVIDENCE_TERMS)
            or _contains_any(object_change, {"released", "dropped", "thrown", "discarded"})
        )
    ):
        return None

    if verb in TOOL_EFFECT_VERBS and (
        contact_type in STRONG_CONTACT_TYPES or _contains_any(why_this_verb, TOOL_EVIDENCE_TERMS)
    ):
        return None

    if hand in {"left", "right", "both"} and contact_type in STRONG_CONTACT_TYPES:
        return None

    if hand in {"left", "right", "both"} and _contains_any(
        object_change, STRONG_OBJECT_CHANGE_KEYWORDS
    ):
        return None

    if hand == "unknown" and contact_type in WEAK_CONTACT_TYPES:
        return "unknown hand with weak/no contact evidence"
    if hand == "unknown" and (
        object_change in EMPTY_OR_WEAK_CHANGES or not _contains_any(object_change, STRONG_OBJECT_CHANGE_KEYWORDS)
    ):
        return "unknown hand with weak/no object-change evidence"
    if contact_type in WEAK_CONTACT_TYPES and object_change in EMPTY_OR_WEAK_CHANGES:
        return "weak contact and no visible object change"
    return "no conservative target-frame action evidence rule matched"


def filter_valid_target_actions(parsed_json: dict | list) -> Tuple[List[dict], List[dict]]:
    """Filter parsed W actions to those with direct TARGET-frame evidence."""
    if isinstance(parsed_json, dict):
        actions = parsed_json.get("actions", [])
    else:
        actions = parsed_json
    if not isinstance(actions, list):
        return [], []
    valid_actions: List[dict] = []
    rejected_actions: List[dict] = []
    for action in actions:
        if not isinstance(action, dict):
            rejected_actions.append(
                {"action": action, "rejection_reason": "action is not a dictionary"}
            )
            continue
        reason = _validity_rejection_reason(action)
        if reason is None:
            valid_actions.append(action)
        else:
            rejected_actions.append({"action": action, "rejection_reason": reason})
    return valid_actions, rejected_actions


def _extract_json(raw: str) -> Dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _strip_code_fence(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        return text.strip(), True
    return text, False


def _json_slice(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text[start:]


def _balance_json_text(text: str) -> str:
    """Best-effort close a truncated JSON object without adding actions."""
    in_string = False
    escape = False
    stack: list[str] = []
    last_safe = -1
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            if not in_string:
                last_safe = idx
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
            last_safe = idx
        elif char in "}]":
            if stack:
                stack.pop()
            last_safe = idx
        elif char in ",:":
            last_safe = idx
    candidate = text
    if in_string:
        quote_idx = text.rfind('"', 0, max(last_safe, 0))
        candidate = text[: quote_idx if quote_idx > 0 else max(last_safe, 0)]
    candidate = re.sub(r',\s*("[^"]*"\s*)?$', "", candidate.rstrip())
    candidate = re.sub(r',\s*$', "", candidate.rstrip())
    open_curly = candidate.count("{") - candidate.count("}")
    open_square = candidate.count("[") - candidate.count("]")
    candidate += "]" * max(open_square, 0)
    candidate += "}" * max(open_curly, 0)
    return candidate


def _extract_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text, flags=re.S)
    return match.group(1) if match else None


def _extract_number_field(text: str, field: str) -> float | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_object_block(text: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\{{', text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _extract_array_field(text: str, field: str) -> list:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[([^\]]*)', text, flags=re.S)
    if not match:
        return ["unknown"]
    values = re.findall(r'"([^"]+)"', match.group(1))
    return values or ["unknown"]


def _fill_action_defaults(action: dict) -> dict:
    direct_object = action.get("direct_object")
    if not isinstance(direct_object, dict):
        direct_object = {"name": direct_object if direct_object else "unknown"}
    direct_object.setdefault("name", "unknown")
    direct_object.setdefault("category", "unknown")
    direct_object.setdefault("specific_type", "unknown")
    direct_object.setdefault("visible_text", "none")
    action["direct_object"] = direct_object

    attrs = action.get("object_attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    attrs.setdefault("color", ["unknown"])
    attrs.setdefault("material", ["unknown"])
    attrs.setdefault("shape", "unknown")
    attrs.setdefault("size", "unknown")
    attrs.setdefault("state", ["unknown"])
    attrs.setdefault("amount", "unknown")
    attrs.setdefault("condition", ["unknown"])
    attrs.setdefault("part", "unknown")
    action["object_attributes"] = attrs

    spatial = action.get("spatial_context")
    if not isinstance(spatial, dict):
        spatial = {}
    spatial.setdefault("location", "unknown")
    spatial.setdefault("relative_position", "unknown")
    spatial.setdefault("contained_in", "none")
    spatial.setdefault("near", ["unknown"])
    action["spatial_context"] = spatial

    interaction = action.get("interaction")
    if not isinstance(interaction, dict):
        interaction = {}
    interaction.setdefault("hand", "unknown")
    interaction.setdefault("tool_used", "none")
    interaction.setdefault("contact_type", "unknown")
    interaction.setdefault("role", "primary_manipulation")
    interaction.setdefault("object_change", "unknown")
    interaction.setdefault("motion_direction", "unknown")
    interaction.setdefault("why_this_verb", "unknown")
    action["interaction"] = interaction

    action.setdefault("time_hint", "unknown")
    action.setdefault("confidence", 0.5)
    return action


def _fill_json_defaults(data: dict) -> dict:
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    data["actions"] = [
        _fill_action_defaults(action)
        for action in actions
        if isinstance(action, dict) and action.get("verb") is not None
    ]
    return data


def _salvage_actions_from_text(text: str) -> dict | None:
    """Extract already-present action fields from malformed JSON text."""
    actions: list[dict] = []
    starts = [match.start() for match in re.finditer(r'"verb"\s*:', text)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end]
        verb = _extract_string_field(chunk, "verb")
        if not verb:
            continue
        direct_block = _extract_object_block(chunk, "direct_object")
        attrs_block = _extract_object_block(chunk, "object_attributes")
        spatial_block = _extract_object_block(chunk, "spatial_context")
        interaction_block = _extract_object_block(chunk, "interaction")
        action = {
            "verb": verb,
            "direct_object": {
                "name": _extract_string_field(direct_block, "name") or "unknown",
                "category": _extract_string_field(direct_block, "category") or "unknown",
                "specific_type": _extract_string_field(direct_block, "specific_type") or "unknown",
                "visible_text": _extract_string_field(direct_block, "visible_text") or "none",
            },
            "object_attributes": {
                "color": _extract_array_field(attrs_block, "color"),
                "material": _extract_array_field(attrs_block, "material"),
                "shape": _extract_string_field(attrs_block, "shape") or "unknown",
                "size": _extract_string_field(attrs_block, "size") or "unknown",
                "state": _extract_array_field(attrs_block, "state"),
                "amount": _extract_string_field(attrs_block, "amount") or "unknown",
                "condition": _extract_array_field(attrs_block, "condition"),
                "part": _extract_string_field(attrs_block, "part") or "unknown",
            },
            "spatial_context": {
                "location": _extract_string_field(spatial_block, "location") or "unknown",
                "relative_position": _extract_string_field(spatial_block, "relative_position") or "unknown",
                "contained_in": _extract_string_field(spatial_block, "contained_in") or "none",
                "near": _extract_array_field(spatial_block, "near"),
            },
            "interaction": {
                "hand": _extract_string_field(interaction_block, "hand") or "unknown",
                "tool_used": _extract_string_field(interaction_block, "tool_used") or "none",
                "contact_type": _extract_string_field(interaction_block, "contact_type") or "unknown",
                "role": _extract_string_field(interaction_block, "role") or "primary_manipulation",
                "object_change": _extract_string_field(interaction_block, "object_change") or "unknown",
                "motion_direction": _extract_string_field(interaction_block, "motion_direction") or "unknown",
                "why_this_verb": _extract_string_field(interaction_block, "why_this_verb") or "unknown",
            },
            "time_hint": _extract_string_field(chunk, "time_hint") or "unknown",
            "confidence": _extract_number_field(chunk, "confidence") or 0.5,
        }
        actions.append(_fill_action_defaults(action))
    if not actions:
        return None
    return {"actions": actions}


def parse_w_json_with_repair(raw_output: str) -> tuple[dict, bool, dict]:
    """Parse W JSON with deterministic formatting repair.

    The returned JSON is only repaired for syntax and missing required fields;
    raw_output itself must still be preserved verbatim by callers.
    """
    repair_info = {
        "used_repair": False,
        "repair_method": "none",
        "parse_error": None,
    }
    text, stripped = _strip_code_fence(raw_output if isinstance(raw_output, str) else "")
    if stripped:
        repair_info.update({"used_repair": True, "repair_method": "strip_code_fence"})
    candidates: list[tuple[str, str]] = [("none", text), ("balanced_braces", _balance_json_text(_json_slice(text)))]
    last_error: str | None = None
    for method, candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            data = _fill_json_defaults(data)
            if method != "none" or stripped:
                repair_info["used_repair"] = True
                repair_info["repair_method"] = method if method != "none" else "strip_code_fence"
            return data, True, repair_info
        except Exception as exc:
            last_error = str(exc)

    salvaged = _salvage_actions_from_text(text)
    if salvaged is not None:
        repair_info.update(
            {
                "used_repair": True,
                "repair_method": "fill_missing_required_fields",
                "parse_error": last_error,
            }
        )
        return salvaged, True, repair_info

    repair_info.update(
        {
            "used_repair": True,
            "repair_method": "failed",
            "parse_error": last_error,
        }
    )
    return {}, False, repair_info


def parse_time_hint(time_hint: str, window_start: float, window_end: float) -> Tuple[float, float, bool]:
    """Convert a relative W time hint to absolute seconds."""
    text = _clean_text(time_hint, "")
    if text in {"", "unknown", "none"}:
        return window_start, window_end, True
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)$", text)
    if not match:
        return window_start, window_end, False
    rel_start = float(match.group(1))
    rel_end = float(match.group(2))
    if rel_end < rel_start:
        rel_start, rel_end = rel_end, rel_start
    span = max(window_end - window_start, 0.0)
    rel_start = min(max(rel_start, 0.0), span)
    rel_end = min(max(rel_end, 0.0), span)
    if rel_end <= rel_start:
        rel_end = min(span, rel_start + 0.1)
    return round(window_start + rel_start, 6), round(window_start + rel_end, 6), True


def _normalize_action(item: Dict[str, Any], idx: int, window_start: float, window_end: float) -> Tuple[Dict[str, Any] | None, bool]:
    parse_ok = True
    direct_object = item.get("direct_object", {})
    if not isinstance(direct_object, dict):
        direct_object = {"name": direct_object}
        parse_ok = False
    attrs = item.get("object_attributes", {})
    spatial = item.get("spatial_context", {})
    interaction = item.get("interaction", {})
    if not isinstance(attrs, dict):
        attrs = {}
        parse_ok = False
    if not isinstance(spatial, dict):
        spatial = {}
        parse_ok = False
    if not isinstance(interaction, dict):
        interaction = {}
        parse_ok = False

    category = _clean_text(direct_object.get("category"))
    if category not in OBJECT_CATEGORIES:
        category = "unknown"
        parse_ok = False
    hand = _clean_text(interaction.get("hand"))
    if hand not in VALID_HANDS:
        hand = "unknown"
        parse_ok = False
    role = _clean_text(interaction.get("role"))
    if role not in VALID_ROLES:
        role = "observation"
        parse_ok = False

    event_start, event_end, time_ok = parse_time_hint(
        str(item.get("time_hint", "")), window_start, window_end
    )
    parse_ok = parse_ok and time_ok
    try:
        confidence = float(item.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
        parse_ok = False
    confidence = min(max(confidence, 0.0), 1.0)

    verb = _clean_text(item.get("verb"), "")
    name = _clean_text(direct_object.get("name"), "")
    why_this_verb = _clean_text(interaction.get("why_this_verb"), "")
    if not verb or not name:
        return None, False
    if not why_this_verb:
        parse_ok = False

    action = {
        "action_idx": idx,
        "verb": verb,
        "direct_object": name,
        "direct_object_rich": {
            "name": name,
            "category": category,
            "specific_type": _clean_text(direct_object.get("specific_type")),
            "visible_text": _clean_text(direct_object.get("visible_text"), "none"),
        },
        "object_attributes": {
            "color": _clean_list(attrs.get("color")),
            "material": _clean_list(attrs.get("material")),
            "shape": _clean_text(attrs.get("shape")),
            "size": _clean_text(attrs.get("size")),
            "state": _clean_list(attrs.get("state")),
            "amount": _clean_text(attrs.get("amount")),
            "condition": _clean_list(attrs.get("condition")),
            "part": _clean_text(attrs.get("part")),
        },
        "spatial_context": {
            "location": _clean_text(spatial.get("location")),
            "relative_position": _clean_text(spatial.get("relative_position")),
            "contained_in": _clean_text(spatial.get("contained_in"), "none"),
            "near": _clean_list(spatial.get("near")),
        },
        "interaction": {
            "hand": hand,
            "tool_used": _clean_text(interaction.get("tool_used"), "none"),
            "contact_type": _clean_text(interaction.get("contact_type")),
            "role": role,
            "object_change": _clean_text(interaction.get("object_change"), "no_visible_change"),
            "motion_direction": _clean_text(interaction.get("motion_direction")),
            "why_this_verb": why_this_verb,
        },
        "hand": hand,
        "with": _clean_text(interaction.get("tool_used"), "none"),
        "purpose": "unknown",
        "manner": role,
        "time_hint": _clean_text(item.get("time_hint"), ""),
        "confidence": confidence,
        "event_t_start": event_start,
        "event_t_end": event_end,
    }
    return action, parse_ok


def parse(raw: str, window_start: float = 0.0, window_end: float = 1.0) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse rich atomic W JSON into normalized action dicts."""
    data = _extract_json(raw if isinstance(raw, str) else "")
    return _parse_actions_from_data(data, window_start, window_end)


def _parse_actions_from_data(data: Dict[str, Any] | None, window_start: float, window_end: float) -> Tuple[List[Dict[str, Any]], bool]:
    """Normalize an already-loaded W JSON object into action dicts."""
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        return [], False
    actions: List[Dict[str, Any]] = []
    parse_ok = True
    for idx, item in enumerate(data["actions"]):
        if not isinstance(item, dict):
            parse_ok = False
            continue
        action, item_ok = _normalize_action(item, idx, window_start, window_end)
        parse_ok = parse_ok and item_ok
        if action is not None:
            actions.append(action)
    return actions, parse_ok


def parse_repaired(raw: str, window_start: float = 0.0, window_end: float = 1.0) -> Tuple[List[Dict[str, Any]], bool, dict]:
    """Parse rich atomic W JSON using deterministic repair before normalization."""
    data, json_ok, repair_info = parse_w_json_with_repair(raw)
    actions, actions_ok = _parse_actions_from_data(data, window_start, window_end)
    return actions, bool(json_ok and actions_ok), repair_info
