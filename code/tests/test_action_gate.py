"""Unit-style checks for conservative W target-frame action gating.

Inputs:
  None.
Outputs:
  Prints pass/fail status for deterministic action-gate examples.
"""

from __future__ import annotations

from typing import Any

try:
    from code.w.prompt import filter_valid_target_actions
except ImportError:
    from code.w.prompt import filter_valid_target_actions


def _action(
    verb: str,
    direct_object_name: str,
    hand: str,
    contact_type: str,
    object_change: str,
    why_this_verb: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "verb": verb,
        "direct_object": {"name": direct_object_name},
        "interaction": {
            "hand": hand,
            "contact_type": contact_type,
            "role": "primary_manipulation",
            "object_change": object_change,
            "tool_used": "none",
            "motion_direction": "unknown",
            "why_this_verb": why_this_verb,
        },
        "confidence": confidence,
    }


def _check(name: str, action: dict[str, Any], expected_accept: bool) -> None:
    valid, rejected = filter_valid_target_actions({"actions": [action]})
    accepted = len(valid) == 1 and not rejected
    if accepted != expected_accept:
        raise AssertionError(
            f"{name}: expected_accept={expected_accept}, accepted={accepted}, "
            f"rejected={rejected}"
        )
    print(f"PASS {name}")


def main() -> int:
    _check(
        "bad orange hallucination",
        _action(
            "place",
            "orange",
            "unknown",
            "support",
            "moved",
            "orange is being set down on the countertop",
            0.7,
        ),
        False,
    )
    _check(
        "bad object-only visibility",
        _action(
            "pick up",
            "orange",
            "unknown",
            "unknown",
            "unknown",
            "orange is visible on the countertop",
            0.8,
        ),
        False,
    )
    _check(
        "valid cupboard open",
        _action(
            "open",
            "cupboard door",
            "left",
            "grasp",
            "opened",
            "cupboard door swings open while hand grips knob",
            0.8,
        ),
        True,
    )
    _check(
        "valid reach for knob",
        _action(
            "reach for",
            "cupboard knob",
            "left",
            "none",
            "no_visible_change",
            "left hand moves toward the cupboard knob in the target frames",
            0.7,
        ),
        True,
    )
    _check(
        "valid mug pickup",
        _action(
            "pick up",
            "mug",
            "left",
            "grasp",
            "lifted",
            "mug leaves the lower shelf while left hand grips the handle",
            0.9,
        ),
        True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
