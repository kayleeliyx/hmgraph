"""Prompt construction for the frozen E1 online-video protocol."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from . import LETTERS, PROMPT_VERSION

SYSTEM_INSTRUCTIONS = """You are evaluating the purpose of a current kitchen action at a specified query time. You may use only the question, choices, query time, available-range statement, and timestamped video frames supplied in this request. To stay within the endpoint image limit, each view's up to 16 unique frames are displayed as up to four chronological 2-by-2 contact sheets. Every tile is labeled with its source-recording timestamp. Views and tiles are chronological.

Return a JSON object with exactly two fields:
- "answer": your current best option label, one of A, B, C, D, or E.
- "request": either null to finish, or an object with numeric "start_sec" and "end_sec" for one additional interval you want to inspect.

Any requested interval must satisfy 0 <= start_sec < end_sec <= query time. It may overlap an earlier interval and may span the entire available prefix. Request more video only when inspecting a particular earlier interval could change or strengthen your answer. Do not explain your answer and do not include any fields besides answer and request."""


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def query_question(record: Mapping[str, Any]) -> str:
    """Remove an explicit annotated interval if a source question contains one.

    Current HD-EPIC why questions normally say only ``in video 1``.  The
    replacements make the policy explicit and future-proof without changing the
    action phrase or choices.
    """

    question = str(record["question"])
    start = str(record.get("requested_start", ""))
    end = str(record.get("requested_end", ""))
    replacement = f"at query time {format_seconds(float(record['end_sec']))} seconds"
    if start and end:
        escaped_start = re.escape(start)
        escaped_end = re.escape(end)
        patterns = (
            rf"from\s+{escaped_start}\s+to\s+{escaped_end}",
            rf"between\s+{escaped_start}\s+and\s+{escaped_end}",
            rf"{escaped_start}\s*[-–—]\s*{escaped_end}",
        )
        for pattern in patterns:
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
    return question


def response_schema(allow_request: bool) -> dict[str, Any]:
    request_schema: dict[str, Any]
    if allow_request:
        request_schema = {
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
        request_schema = {"type": "null"}
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list(LETTERS)},
            "request": request_schema,
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
    choices = query["displayed_choices"]
    if len(choices) != 5:
        raise ValueError("E1 requires exactly five displayed choices")
    choice_text = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    batch_text = "\n".join(
        f"- View {index}: requested [{float(batch['start_sec']):.3f}, "
        f"{float(batch['end_sec']):.3f}] seconds; "
        f"{len(batch['frames'])} timestamped frames follow in "
        f"{len(batch['contact_sheets'])} contact sheets."
        for index, batch in enumerate(batches)
    )
    history = ""
    if previous_responses:
        lines = []
        for index, response in enumerate(previous_responses, start=1):
            request = response.get("request")
            request_text = "finish" if request is None else (
                f"request [{float(request['start_sec']):.3f}, "
                f"{float(request['end_sec']):.3f}]"
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
