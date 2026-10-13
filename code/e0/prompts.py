"""Prompt builders for E0, kept explicit for audit and cache identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from . import LETTERS

KITCHEN_CONTEXT = "general kitchen setting"

ANSWER_INSTRUCTIONS = (
    "Answer the multiple-choice why-recognition question. Use only the supplied "
    "question, choices, images (when present), and background facts (when present). "
    "When images are present, they are chronological frames sampled only from the "
    "official action interval. "
    "Background facts are general possibilities and may not apply to the particular "
    "episode. Assess them against the available evidence. Return exactly one capital "
    "letter: A, B, C, D, or E. Do not provide an explanation."
)

FACT_HEADING = (
    "Additional background facts. These facts may or may not apply to this particular "
    "episode; assess them against the available evidence."
)


def answer_text(question: str, choices: Sequence[str], facts: Sequence[str] | None = None) -> str:
    if len(choices) != 5:
        raise ValueError(f"E0 requires exactly five choices, got {len(choices)}")
    choice_text = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    parts = [f"Question:\n{question}", f"Choices:\n{choice_text}"]
    if facts is not None:
        fact_text = "\n".join(f"- {fact}" for fact in facts) or "- No background facts were retrieved."
        parts.append(f"{FACT_HEADING}\n{fact_text}")
    parts.append("Answer with exactly one letter: A, B, C, D, or E.")
    return "\n\n".join(parts)


def self_knowledge_text(action_phrase: str) -> str:
    return f"""Consider the action: {action_phrase}.
In a kitchen setting, provide 3–5 concise general facts about the objects, functions, or circumstances that could explain why someone might perform this action. Describe general possibilities and any relevant conditions. You have no information about a particular person or recording. Do not claim to know the purpose in a specific episode. Use approximately 80–180 words total. Return only the facts."""


ACTION_PREFIXES = tuple(
    sorted(
        {
            "pick up", "put down", "take out", "place down", "switch on", "switch off",
            "turn on", "turn off", "wash", "rinse", "clean", "wipe", "dry", "peel",
            "chop", "slice", "cut", "dice", "grate", "stir", "mix", "whisk", "beat",
            "pour", "add", "remove", "take", "place", "put", "move", "hold", "open",
            "close", "tilt", "tap", "shake", "scrape", "scoop", "spread", "drain",
            "strain", "measure", "weigh", "press", "squeeze", "flip", "turn", "lift",
            "lower", "carry", "cover", "uncover", "wrap", "unwrap", "check", "touch",
            "lick", "tip", "transfer", "rotate", "grab", "hit", "rub", "push", "pull",
            "pinch", "invert", "slide",
        },
        key=lambda value: (-len(value.split()), value),
    )
)


def object_phrase_from_action(action_phrase: str) -> str:
    normalized = " ".join(action_phrase.lower().split())
    remainder = normalized
    for prefix in ACTION_PREFIXES:
        if normalized == prefix:
            return normalized
        if normalized.startswith(prefix + " "):
            remainder = normalized[len(prefix) :].strip()
            break
    words = remainder.split()
    while words and words[0] in {"a", "an", "the", "some", "with"}:
        words.pop(0)
    remainder = " ".join(words)
    normalized_objects = (
        ("potato chicken ball", "potato"),
        ("phone keyboard", "computer keyboard"),
        ("baking tray", "sheet pan"),
        ("chopping board", "cutting board"),
        ("food scraps", "food waste"),
        ("tea towel", "towel"),
        ("right hand", "hand"),
        ("left hand", "hand"),
        ("both hands", "hand"),
        ("index finger", "finger"),
        ("tablespoon", "tablespoon"),
        ("teaspoon", "teaspoon"),
        ("saucepan", "saucepan"),
        ("strainer", "sieve"),
        ("cupboard", "cupboard"),
        ("grater", "grater"),
        ("sponge", "sponge"),
        ("ladle", "ladle"),
        ("spoon", "spoon"),
        ("bowl", "bowl"),
        ("water", "water"),
        ("hand", "hand"),
    )
    for marker, canonical in normalized_objects:
        if marker in remainder:
            return canonical
    return " ".join(words[:5]) or normalized


def retrieval_queries(action_phrase: str) -> tuple[str, ...]:
    normalized = " ".join(action_phrase.split())
    object_phrase = object_phrase_from_action(normalized)
    return (
        f'intitle:"{object_phrase}"',
        f'"{object_phrase}" kitchen utensil',
        f'"{normalized}" cooking kitchen',
    )


@dataclass(frozen=True)
class ActionSemantics:
    """Answer-blind structure recovered only from the annotated action phrase."""

    action_phrase: str
    action: str
    raw_object_phrase: str
    object_phrase: str
    container_or_appliance: str | None
    lookup_concept: str
    canonical_titles: tuple[str, ...]
    uncertainty: str | None

    def audit_dict(self) -> dict[str, object]:
        return asdict(self)


# Generic language/word-sense mappings. These are deliberately phrase-level rather
# than question-ID rules and contain no episode purpose or answer information.
KITCHEN_SENSES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("temperature of water", "cooking temperature measurement", ("Cooking thermometer", "Thermometer")),
    ("knob of gas stove", "gas stove control knob", ("Control knob", "Gas stove", "Kitchen stove")),
    ("contents of frying pan", "frying pan", ("Frying pan",)),
    ("contents of pan", "frying pan", ("Frying pan",)),
    ("potato chicken ball", "potato food", ("Potato",)),
    ("phone keyboard", "computer keyboard", ("Computer keyboard",)),
    ("large mixing bowl", "mixing bowl", ("Bowl",)),
    ("chef's knife", "chef's knife", ("Chef's knife",)),
    ("chopping board", "cutting board", ("Cutting board",)),
    ("dough scraper", "bench scraper", ("Bench scraper",)),
    ("baking tray", "sheet pan", ("Sheet pan",)),
    ("food scraps", "food waste", ("Food waste",)),
    ("tea towel", "kitchen towel", ("Kitchen towel", "Dishcloth")),
    ("wooden spatula", "cooking spatula", ("Spatula",)),
    ("wooden spoon", "wooden spoon", ("Wooden spoon",)),
    ("kitchen scale", "weighing scale", ("Weighing scale",)),
    ("right hand", "hand", ()),
    ("left hand", "hand", ()),
    ("both hands", "hand", ()),
    ("index finger of right hand", "finger", ()),
    ("index finger", "finger", ()),
    ("frying pan", "frying pan", ("Frying pan",)),
    ("tablespoon", "tablespoon", ("Tablespoon",)),
    ("teaspoon", "teaspoon", ("Teaspoon",)),
    ("saucepan", "saucepan", ("Saucepan",)),
    ("strainer", "cooking sieve", ("Sieve",)),
    ("skimmer", "skimming utensil", ("Skimmer (utensil)",)),
    ("cupboard", "cupboard", ("Cupboard",)),
    ("colander", "colander", ("Colander",)),
    ("grater", "grater", ("Grater",)),
    ("sponge", "cleaning sponge", ("Sponge (tool)", "Dishwashing")),
    ("pan", "frying pan", ("Frying pan",)),
    ("ladle", "cooking ladle", ("Ladle (spoon)", "Ladle")),
    ("spatula", "cooking spatula", ("Spatula",)),
    ("spoon", "spoon", ("Spoon",)),
    ("bowl", "bowl", ("Bowl",)),
    ("plate", "dinner plate", ("Plate (dishware)",)),
    ("scale", "weighing scale", ("Weighing scale",)),
    ("tap", "faucet", ("Tap (valve)",)),
    ("kettle", "kettle", ("Kettle",)),
    ("knife", "kitchen knife", ("Kitchen knife",)),
    ("tray", "tray", ("Tray",)),
    ("recipe", "recipe", ("Recipe",)),
    ("fork", "fork", ("Fork",)),
    ("lid", "lid", ("Lid (container)", "Lid")),
    ("onion", "onion", ("Onion",)),
    ("potatoes", "potato", ("Potato",)),
    ("potato", "potato", ("Potato",)),
    ("cup", "cup", ("Cup",)),
    ("glass", "drinking glass", ("Drinkware",)),
    ("water", "water", ("Water",)),
    ("door", "door", ("Door",)),
    ("bag", "bag", ("Bag",)),
)


def parse_action_semantics(action_phrase: str) -> ActionSemantics:
    normalized = " ".join(action_phrase.lower().split())
    action = ""
    remainder = normalized
    for prefix in ACTION_PREFIXES:
        if normalized == prefix:
            action = prefix
            remainder = ""
            break
        if normalized.startswith(prefix + " "):
            action = prefix
            remainder = normalized[len(prefix) :].strip()
            break
    if not action:
        words = normalized.split(maxsplit=1)
        action = words[0] if words else ""
        remainder = words[1] if len(words) == 2 else ""

    raw_object = remainder
    object_phrase = remainder
    container: str | None = None
    if remainder.startswith("contents of "):
        object_phrase = "contents"
        container = remainder[len("contents of ") :].strip()
    elif remainder.startswith("knob of "):
        object_phrase = "control knob"
        container = remainder[len("knob of ") :].strip()
    elif remainder.startswith("temperature of "):
        object_phrase = "temperature"
        container = remainder[len("temperature of ") :].strip()

    lookup_source = raw_object
    lookup_concept = raw_object or normalized
    canonical_titles: tuple[str, ...] = ()
    for marker, concept, titles in KITCHEN_SENSES:
        if marker in lookup_source:
            lookup_concept = concept
            canonical_titles = titles
            break

    if action in {"rinse", "rub", "wash", "dry"} and lookup_concept == "hand":
        lookup_concept = "hand washing" if action != "dry" else "hand drying"
        canonical_titles = ("Hand washing",) if action != "dry" else ("Hand drying", "Hand washing")

    uncertainty = None
    if lookup_source in {"items", "dial", ""}:
        uncertainty = "underspecified_object"
        canonical_titles = ()
    elif lookup_concept in {"hand", "finger", "bag", "door"}:
        uncertainty = "generic_object_without_kitchen-specific_sense"

    return ActionSemantics(
        action_phrase=action_phrase,
        action=action,
        raw_object_phrase=raw_object,
        object_phrase=object_phrase,
        container_or_appliance=container,
        lookup_concept=lookup_concept,
        canonical_titles=canonical_titles,
        uncertainty=uncertainty,
    )


def retrieval_queries_v2(action_phrase: str) -> tuple[dict[str, str], ...]:
    semantics = parse_action_semantics(action_phrase)
    candidates = [
        {
            "kind": "action_kitchen_context",
            "query": f"{semantics.action} {semantics.lookup_concept} kitchen cooking use",
        },
    ]
    candidates.append({"kind": "broad_concept", "query": semantics.lookup_concept})
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate["query"].split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            result.append({**candidate, "query": normalized})
    return tuple(result)
