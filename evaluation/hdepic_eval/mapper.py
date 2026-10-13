"""Train-only version of Claude's heuristic free-text to HD-EPIC class mapper.

Mapping sees a generated phrase and a frozen lexicon, never its gold answer.
It remains a heuristic diagnostic until generated paraphrases are validated.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

MAPPER_VERSION = "hdepic_train_only_class_mapper_v1"
SKIP = set("the a an person is are was be being likely probably currently now then and will to using use with hand hands right left their his her i they he she it action next would may might continue continues continuing start starts".split())
DET = set("the a an some another other one two three its their his her my this that these those both all each more remaining second first third last same new few several any it them".split())
PREP = set("from into in on onto to with off out under inside at by for towards toward over using and while so then until before after near beside behind through around across against between back away down up".split())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text):
    text = text.lower().replace("-", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _class_id(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return int(value) if math.isfinite(value) and value.is_integer() and value >= 0 else None


def _lemmas(token):
    result = [token]
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        result += [stem, stem + "e"] + ([stem[:-1]] if len(stem) > 2 and stem[-1] == stem[-2] else [])
    if token.endswith("ies"):
        result.append(token[:-3] + "y")
    if token.endswith("es"):
        result.append(token[:-2])
    if token.endswith("s") and not token.endswith("ss"):
        result.append(token[:-1])
    if token.endswith("ed") and len(token) > 3:
        stem = token[:-2]
        result += [stem, token[:-1]] + ([stem[:-1]] if len(stem) > 2 and stem[-1] == stem[-2] else [])
    return result


class ActionMapper:
    def __init__(self, verb_map, noun_map, provenance=None):
        self.verb_map = dict(verb_map)
        self.noun_map = dict(noun_map)
        if any(type(v) is not int or v < 0 for v in (*self.verb_map.values(), *self.noun_map.values())):
            raise ValueError("Invalid lexicon class ID")
        self.provenance = provenance or {}

    def _noun(self, tokens):
        while tokens and tokens[0] in DET:
            tokens = tokens[1:]
        if not tokens:
            return None
        text = " ".join(tokens)
        if text in self.noun_map:
            return self.noun_map[text]
        chunk = []
        for token in tokens:
            if token in PREP and chunk:
                break
            chunk.append(token)
        candidates = [chunk]
        if "of" in chunk:
            candidates.insert(0, chunk[:chunk.index("of")])
        for candidate in candidates:
            for i in range(len(candidate)):
                text = " ".join(candidate[i:])
                for form in (text, text[:-1] if text.endswith("s") else None,
                             text[:-2] if text.endswith("es") else None):
                    if form and form in self.noun_map:
                        return self.noun_map[form]
        return None

    def map_text(self, phrase):
        if not isinstance(phrase, str) or not phrase.strip():
            return (None, None)
        tokens = normalize(phrase).split()
        for i, token in enumerate(tokens):
            for size in (3, 2, 1):
                if i + size > len(tokens):
                    continue
                for head in _lemmas(token):
                    candidate = " ".join([head] + tokens[i + 1:i + size])
                    if candidate in self.verb_map:
                        noun = self._noun(tokens[i + size:])
                        if token in SKIP:
                            # A skip word can itself be an official action:
                            # start dishwasher, hand mug, use knife. Retain it
                            # with an object; skip auxiliary start taking mug.
                            tail = tokens[i + size:]
                            next_is_verb = bool(tail) and any(
                                " ".join([form] + tail[1:width]) in self.verb_map
                                for width in (3, 2, 1) if len(tail) >= width
                                for form in _lemmas(tail[0]))
                            if noun is None or next_is_verb:
                                continue
                        # A negated action is not the asserted positive action.
                        if any(x in {"not", "never", "don't", "doesn't", "cannot", "can't"} for x in tokens[:i]):
                            return (None, None)
                        return self.verb_map[candidate], noun
        return (None, None)

    def to_dict(self):
        return {"version": MAPPER_VERSION, "verb_map": self.verb_map, "noun_map": self.noun_map,
                "provenance": self.provenance,
                "limitation": "Heuristic phrase/category agreement; generated paraphrase validation pending."}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != MAPPER_VERSION:
            raise ValueError("Unsupported mapper version")
        return cls(value["verb_map"], value["noun_map"], value["provenance"])


def build_mapper(annotations, manifest_path):
    """Fit only initial-SFT videos; add the released official class aliases."""
    import pandas as pd  # Needed only to read the released narration pickle.
    annotations, manifest_path = Path(annotations), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    videos = set(manifest["partitions"]["initial_sft"]["video_ids"])
    root = annotations / "narrations-and-action-segments"
    narration_path = root / "HD_EPIC_Narrations.pkl"
    frame = pd.read_pickle(narration_path)
    frame = frame[frame["video_id"].isin(videos) & frame["main_actions"].map(lambda x: len(x) >= 1)]
    frame = frame.sort_values(["video_id", "start_timestamp", "end_timestamp"])
    lexicons, conflicts = [], []
    hashes = {str(narration_path): sha256(narration_path), str(manifest_path): sha256(manifest_path)}
    for index, kind in enumerate(("verb", "noun")):
        lexicon = defaultdict(Counter)
        for row in frame.itertuples():
            action = row.main_actions[0]
            classes = row.main_action_classes[0] if row.main_action_classes else []
            class_id = _class_id(classes[index]) if len(classes) > index else None
            if len(action) > index and isinstance(action[index], str) and class_id is not None:
                phrase = normalize(action[index])
                if phrase:
                    lexicon[phrase][class_id] += 1
        table_path = root / f"HD_EPIC_{kind}_classes.csv"
        hashes[str(table_path)] = sha256(table_path)
        with table_path.open() as handle:
            for row in csv.DictReader(handle):
                for instance in ast.literal_eval(row["instances"]):
                    if ":" in instance:
                        continue
                    phrase = normalize(instance)
                    if phrase and phrase not in lexicon:
                        lexicon[phrase][int(row["id"])] += 1
        conflicts.append(sum(len(counts) > 1 for counts in lexicon.values()))
        lexicons.append({phrase: counts.most_common(1)[0][0] for phrase, counts in lexicon.items()})
    return ActionMapper(*lexicons, provenance={
        "fit_policy": "initial-SFT videos plus official class definitions; no held-out narration aliases",
        "fit_video_ids": sorted(videos), "fit_annotation_rows": len(frame), "source_sha256": hashes,
        "verb_alias_conflicts_resolved_by_training_frequency": conflicts[0],
        "noun_alias_conflicts_resolved_by_training_frequency": conflicts[1],
        "origin": "Claude scorer heuristics, extracted into train-only module; missing IDs, negation and valid-verb stopwords repaired"})
