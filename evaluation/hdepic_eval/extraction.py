"""Reference-blind field extraction; preserve explicit missing forecast slots."""
from __future__ import annotations

import ast
import html
import json
import re

FIELDS = ("action", "why", "goal", "next")
TAG = re.compile(r"</?\s*([A-Za-z][A-Za-z0-9_]*)\s*>")


class _AmbiguousJSON(dict):
    pass


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise KeyError("Duplicate JSON field")
        value[key] = item
    return value


def _structured(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?|\n?```$", "", text, flags=re.I)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except KeyError:
        return _AmbiguousJSON()
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def extract_field(raw, field):
    """Read one unambiguous field, regardless of the other fields' syntax."""
    if field not in FIELDS or field == "next":
        raise ValueError("Expected action, why or goal")
    raw = raw if isinstance(raw, str) else ""
    structured = _structured(raw)
    if isinstance(structured, _AmbiguousJSON):
        return {"text": None, "status": "ambiguous_json"}
    if structured is not None:
        value = structured.get(field)
        return {"text": value.strip() if isinstance(value, str) and value.strip() else None,
                "status": "json" if isinstance(value, str) and value.strip() else "missing_or_empty"}
    if field == "action":
        # Future child <action> tags must not become the current action.
        raw = re.split(r"<\s*next\s*>", raw, maxsplit=1, flags=re.I)[0]
    opens = list(re.finditer(r"<\s*" + field + r"\s*>", raw, re.I))
    closes = list(re.finditer(r"</\s*" + field + r"\s*>", raw, re.I))
    if not opens:
        # Explicit textual field labels are also accepted; no unlabeled guesses.
        pattern = r"^\s*(?:\*\*)?" + field + r"(?:\*\*)?\s*:\s*"
        labels = list(re.finditer(pattern, raw, re.I | re.M))
        if len(labels) != 1:
            return {"text": None, "status": "missing" if not labels else "ambiguous"}
        tail = raw[labels[0].end():]
        end = re.search(r"^\s*(?:\*\*)?(?:action|why|goal|next)(?:\*\*)?\s*:", tail, re.I | re.M)
        text = tail[:end.start()] if end else tail
        return {"text": text.strip() or None, "status": "labeled" if text.strip() else "empty"}
    if len(opens) != 1 or len(closes) > 1 or (closes and closes[0].start() < opens[0].end()):
        return {"text": None, "status": "ambiguous"}
    tail = raw[opens[0].end():]
    boundary = TAG.search(tail)
    if closes and (not boundary or boundary.group().replace(" ", "").lower() != f"</{field}>"):
        return {"text": None, "status": "crossed_or_ambiguous"}
    text = html.unescape(tail[:boundary.start()] if boundary else tail).strip()
    return {"text": text or None,
            "status": "empty" if not text else "closed" if closes else "recovered_unclosed"}


def _item(value):
    if isinstance(value, dict):
        value = value.get("action")
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = html.unescape(re.sub(r"^\s*[-*•]\s*", "", value)).strip()
    return value or None


def _parts(body, action_detector):
    children = list(re.finditer(r"<\s*action\s*>([^<>]*)</\s*action\s*>", body, re.I | re.S))
    opens = list(re.finditer(r"<\s*action\s*>", body, re.I))
    if opens:
        if len(children) == len(opens):
            return [_item(child.group(1)) for child in children], "nested_actions"
        # Do not shift later valid children across an explicitly broken child.
        return [None for _ in opens], "ambiguous_nested_actions"
    body = re.sub(r"<\s*(time|timestamp)\s*>[^<>]*</\s*\1\s*>", "", body, flags=re.I)
    try:
        value = json.loads(body.strip())
    except (ValueError, TypeError):
        try:
            value = ast.literal_eval(body.strip())
        except (ValueError, TypeError, SyntaxError):
            value = None
    if isinstance(value, list):
        return [_item(x) for x in value], "literal_list"
    times = list(re.finditer(r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]", body))
    if len(times) >= 2:
        return [_item(body[m.start():times[i+1].start() if i+1 < len(times) else len(body)].strip(" ,;\n"))
                for i, m in enumerate(times)], "timestamped_list"
    # Dot markers need whitespace/end: decimals are quantities or timestamps.
    # Do not consume line breaks after an empty marker; its next marker is a slot.
    numbered = list(re.finditer(r"(?<![\w.])([0-9]+)[.)](?=[ \t\r\n]|$)[ \t]*", body))
    if numbered and not body[:numbered[0].start()].strip():
        indices = [int(m.group(1)) for m in numbered]
        origin = 0 if indices[0] == 0 else 1
        if indices != sorted(set(indices)) or indices[0] < origin or indices[-1] > 100:
            return [None], "ambiguous_numbering"
        result = [None] * (indices[-1] - origin + 1)
        for i, match in enumerate(numbered):
            end = numbered[i + 1].start() if i + 1 < len(numbered) else len(body)
            result[indices[i] - origin] = _item(body[match.end():end])
        return result, "numbered_list"
    if ";" in body:
        # A trailing terminator adds no action, but an interior empty slot does.
        parts = body.strip().split(";")
        if len(parts) > 1 and not parts[-1].strip():
            parts.pop()
        return [_item(x) for x in parts], "semicolon_list"
    lines = [x for x in body.splitlines() if x.strip()]
    if len(lines) > 1:
        return [_item(x) for x in lines], "newline_list"
    arrows = re.split(r"\s*(?:→|->)\s*", body)
    if len(arrows) > 1:
        return [_item(x) for x in arrows], "arrow_list"
    commas = body.split(",")
    if len(commas) == 3 and action_detector and all(not x.strip() or action_detector(x) for x in commas):
        return [_item(x) for x in commas], "comma_list"
    return [_item(body)], "single_action"


def extract_next(raw, action_detector=None):
    """Return every explicit slot in generated order; scoring handles horizon."""
    raw = raw if isinstance(raw, str) else ""
    structured = _structured(raw)
    if isinstance(structured, _AmbiguousJSON):
        return {"actions": [], "total_actions": 0, "block_count": 0,
                "representations": ["ambiguous_json"]}
    if structured is not None:
        value = structured.get("next")
        if isinstance(value, list):
            actions = [_item(x) for x in value]
            return {"actions": actions, "total_actions": len(actions), "block_count": 1,
                    "representations": ["json_list"]}
        raw = f"<next>{value}</next>" if isinstance(value, str) else ""
    blocks = list(re.finditer(r"<\s*next\s*>", raw, re.I))
    if not blocks:
        labels = list(re.finditer(r"^\s*(?:\*\*)?next(?:\*\*)?\s*:\s*", raw, re.I | re.M))
        if len(labels) == 1:
            tail = raw[labels[0].end():]
            stop = re.search(r"^\s*(?:\*\*)?(?:action|why|goal)(?:\*\*)?\s*:", tail, re.I | re.M)
            raw = "<next>" + (tail[:stop.start()] if stop else tail) + "</next>"
            blocks = list(re.finditer(r"<next>", raw))
    actions, representations = [], []
    for block in blocks:
        tail = raw[block.end():]
        stop = re.search(r"</?\s*next\s*>|<\s*(?:why|goal)\s*>", tail, re.I)
        body = tail[:stop.start()] if stop else tail
        parts, representation = _parts(body, action_detector)
        actions.extend(parts)
        representations.append(representation)
    return {"actions": actions, "total_actions": len(actions), "block_count": len(blocks),
            "representations": representations}
