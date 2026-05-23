"""Build and apply HD-EPIC vocabulary aliases for W to L1 consolidation.

Inputs:
  HD-EPIC verb and noun class CSV files.
Outputs:
  JSON-serializable alias table with verb_aliases and noun_aliases.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping


def normalize(text: object) -> str:
    """Normalize labels for exact alias lookup."""
    value = "" if text is None else str(text)
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def _read_class_csv(path: Path) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = normalize(row.get("key", ""))
            if not key:
                continue
            aliases[key] = key
            instances_raw = row.get("instances", "[]")
            try:
                instances = ast.literal_eval(instances_raw)
            except (SyntaxError, ValueError):
                instances = []
            for instance in instances:
                alias = normalize(instance)
                if alias:
                    aliases[alias] = key
    return aliases


def build(hdepic_vocab_path: str | Path) -> Dict[str, Dict[str, str]]:
    """Build aliases from a directory containing HD-EPIC verb/noun CSVs."""
    root = Path(hdepic_vocab_path)
    if root.is_file():
        root = root.parent
    verb_path = root / "HD_EPIC_verb_classes.csv"
    noun_path = root / "HD_EPIC_noun_classes.csv"
    if not verb_path.exists() or not noun_path.exists():
        raise FileNotFoundError(
            f"Expected HD_EPIC_verb_classes.csv and HD_EPIC_noun_classes.csv under {root}"
        )
    return {
        "verb_aliases": _read_class_csv(verb_path),
        "noun_aliases": _read_class_csv(noun_path),
    }


def canonicalize(value: object, aliases: Mapping[str, str]) -> str:
    """Return canonical alias value, falling back to normalized input."""
    normalized = normalize(value)
    return aliases.get(normalized, normalized)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdepic-vocab-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    table = build(args.hdepic_vocab_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    print(f"W/L1 alias_table written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
