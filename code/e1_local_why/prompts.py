"""Frozen prompts for the local E1 experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import LETTERS, PROMPT_VERSION

SYSTEM_INSTRUCTIONS = """You are evaluating the purpose of a current kitchen action at a specified query time. You may use only the question, choices, query time, available-range statement, and timestamped video frames supplied in this request. Each source frame is supplied as an individual image. Views are chronological, and frames within each view are chronological.

Return a JSON object with exactly two fields:
- "answer": your current best option label, one of A, B, C, D, or E.
- "request": either null to finish, or an object with numeric "start_sec" and "end_sec" for one additional interval you want to inspect.

Any requested interval must satisfy 0 <= start_sec < end_sec <= query time. It may overlap an earlier interval and may span the entire available prefix. Request more video only when inspecting a particular earlier interval could change or strengthen your answer. Do not explain your answer and do not include any fields besides answer and request."""


def response_schema(allow_request: bool) -> dict[str, Any]:
    request: dict[str, Any]
    if allow_request:
        request = {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                    },
                    "required": ["start_sec", "end_sec"],
                    "additionalProperties": False,
                },
            ]
        }
    else:
        request = {"type": "null"}
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list(LETTERS)},
            "request": request,
        },
        "required": ["answer", "request"],
        "additionalProperties": False,
    }


def prompt_text(
    query: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    previous_responses: Sequence[Mapping[str, Any]],
    reads_remaining: int,
) -> str:
    choices = list(query["displayed_choices"])
    if len(choices) != 5:
        raise ValueError("local E1 requires exactly five displayed choices")
    choice_text = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    batch_text = "\n".join(
        f"- View {index}: interval [{float(batch['start_sec']):.3f}, "
        f"{float(batch['end_sec']):.3f}] seconds; {len(batch['frames'])} individual "
        "timestamped frames follow."
        for index, batch in enumerate(batches)
    )
    history = ""
    if previous_responses:
        lines = []
        for index, response in enumerate(previous_responses, start=1):
            request = response.get("request")
            request_text = (
                "finish"
                if request is None
                else f"request [{float(request['start_sec']):.3f}, {float(request['end_sec']):.3f}]"
            )
            lines.append(f"- Earlier decision {index}: answer {response['answer']}; {request_text}.")
        history = "\n\nEarlier decisions in this same query:\n" + "\n".join(lines)
    request_rule = (
        f"You have {reads_remaining} additional video read(s) remaining. Return request=null "
        "if you can answer now; otherwise request one valid interval."
        if reads_remaining > 0
        else "No additional video reads remain. You must return request=null and your final answer."
    )
    return f"""Protocol: {PROMPT_VERSION}

Question:
{query['question']}

Choices:
{choice_text}

Query time: {float(query['query_time_sec']):.3f} seconds from the source recording start.
Available recording range: [0.000, {float(query['query_time_sec']):.3f}] seconds.
The annotated action start and duration are not provided.

Available views:
{batch_text}{history}

{request_rule}
Return only the required JSON object."""

