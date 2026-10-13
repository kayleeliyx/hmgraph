#!/usr/bin/env python3
"""Class-level scoring of saved HD-EPIC SFT validation predictions, with trivial baselines.

Lexical exact match on free-text action phrases is close to meaningless ("pick up mug" vs
"pick up another mug"). This maps every predicted and reference phrase to HD-EPIC verb and
noun classes, checks the mapper against the gold classes of the references, and compares
base / epoch1 / epoch2 with baselines that never look at the video.

Usage: python -I score_hdepic_classes.py <hd-epic-annotations> <review_followup_v1 dir> <out_dir>

Inputs from the follow-up bundle: scoring/{base,epoch1,epoch2}.fields.jsonl and
forecast/{references,epoch1,epoch2}.all_triples.jsonl. Base NEXT triples are parsed from the raw
base response. The bigram baselines learn from TRAIN_PARTICIPANTS; change it if the split changes.
Outputs: class_level_summary.json, class_level_extra.json (fair subsets, paired bootstrap 95% CIs),
mapper_spotcheck.tsv (predicted phrase -> class, for eyeballing the mapper).
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ANN, FOLLOW, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_PARTICIPANTS = {"P01", "P02", "P03", "P04", "P05", "P06"}


def norm(s: str) -> str:
    s = s.lower().replace("-", " ")
    s = re.sub(r"\[[^\]]*\]", " ", s)          # timestamps like [12.0, 13.5]
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------- annotations and lexicons
df = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
df = df[df["main_actions"].map(len) >= 1].copy()
df["verb_s"] = df["main_actions"].map(lambda m: norm(m[0][0]))
df["noun_s"] = df["main_actions"].map(lambda m: norm(m[0][1]))


def cls(m, i):
    t = m[0] if len(m) else ()
    return int(t[i]) if len(t) > i and t[i] is not None and t[i] == t[i] else None


df["vc"] = df["main_action_classes"].map(lambda m: cls(m, 0))
df["nc"] = df["main_action_classes"].map(lambda m: cls(m, 1))
df = df.sort_values(["video_id", "start_timestamp", "end_timestamp"]).reset_index(drop=True)

verb_lex: dict = defaultdict(Counter)
noun_lex: dict = defaultdict(Counter)
for v, n, a, b in zip(df["verb_s"], df["noun_s"], df["vc"], df["nc"]):
    if a is not None:
        verb_lex[v][a] += 1
    if b is not None:
        noun_lex[n][b] += 1
for kind, lex in (("verb", verb_lex), ("noun", noun_lex)):
    table = pd.read_csv(ANN / "narrations-and-action-segments" / f"HD_EPIC_{kind}_classes.csv")
    for _, r in table.iterrows():
        for inst in ast.literal_eval(r["instances"]):
            if ":" in inst:
                continue
            s = norm(inst)
            if s and s not in lex:
                lex[s][int(r["id"])] += 1
verb_map = {s: c.most_common(1)[0][0] for s, c in verb_lex.items()}
noun_map = {s: c.most_common(1)[0][0] for s, c in noun_lex.items()}
verb_names = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_verb_classes.csv").set_index("id")["key"].to_dict()
noun_names = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv").set_index("id")["key"].to_dict()

SKIP = {"the", "a", "an", "person", "is", "are", "was", "be", "being", "likely", "probably",
        "currently", "now", "then", "and", "will", "to", "using", "use", "with", "hand", "hands",
        "right", "left", "their", "his", "her", "i", "they", "he", "she", "it", "action",
        "next", "would", "may", "might", "continue", "continues", "continuing", "start", "starts"}
DET = {"the", "a", "an", "some", "another", "other", "one", "two", "three", "its", "their", "his",
       "her", "my", "this", "that", "these", "those", "both", "all", "each", "more", "remaining",
       "second", "first", "third", "last", "same", "new", "few", "several", "any", "it", "them"}
PREP = {"from", "into", "in", "on", "onto", "to", "with", "off", "out", "under", "inside", "at",
        "by", "for", "towards", "toward", "over", "using", "and", "while", "so", "then", "until",
        "before", "after", "near", "beside", "behind", "through", "around", "across", "against",
        "between", "back", "away", "down", "up"}


def lemmas(tok: str):
    out = [tok]
    if tok.endswith("ing") and len(tok) > 4:
        b = tok[:-3]
        out += [b, b + "e"] + ([b[:-1]] if len(b) > 2 and b[-1] == b[-2] else [])
    if tok.endswith("ies"):
        out.append(tok[:-3] + "y")
    if tok.endswith("es"):
        out.append(tok[:-2])
    if tok.endswith("s") and not tok.endswith("ss"):
        out.append(tok[:-1])
    if tok.endswith("ed") and len(tok) > 3:
        b = tok[:-2]
        out += [b, tok[:-1]] + ([b[:-1]] if len(b) > 2 and b[-1] == b[-2] else [])
    return out


def parse_noun(toks):
    while toks and toks[0] in DET:
        toks = toks[1:]
    if not toks:
        return None
    s = " ".join(toks)
    if s in noun_map:
        return noun_map[s]
    chunk = []
    for t in toks:
        if t in PREP and chunk:
            break
        chunk.append(t)
    cands = [chunk]
    if "of" in chunk:
        cands.insert(0, chunk[: chunk.index("of")])
    for c in cands:
        for j in range(len(c)):
            s = " ".join(c[j:])
            for form in (s, s[:-1] if s.endswith("s") else None, s[:-2] if s.endswith("es") else None):
                if form and form in noun_map:
                    return noun_map[form]
    return None


def parse_action(phrase):
    """Map a free-text action phrase to (verb class, noun class); None where not found."""
    if not phrase:
        return (None, None)
    toks = norm(phrase).split()
    for i in range(len(toks)):
        if toks[i] in SKIP:
            continue
        for k in (3, 2, 1):
            if i + k > len(toks):
                continue
            for h in lemmas(toks[i]):
                cand = " ".join([h] + toks[i + 1:i + k])
                if cand in verb_map:
                    return (verb_map[cand], parse_noun(toks[i + k:]))
    return (None, None)


# ----------------------------------------------------------------- gold classes for each query
rows_by_video = {v: g.reset_index(drop=True) for v, g in df.groupby("video_id")}
by_nid = df.set_index("unique_narration_id")


def gold_for(query_id):
    video, task, key = query_id.split(":", 2)
    g = rows_by_video[video]
    if task == "track":
        t = float(key)
        cur = g[(g["start_timestamp"] <= t) & (g["end_timestamp"] >= t)]
        current = cur.iloc[0] if len(cur) == 1 else None
        explained = None
    else:
        r = by_nid.loc[key]
        t = float(r["end_timestamp"])
        current, explained = None, r
    nxt = g[g["start_timestamp"] > t].head(3)
    done = g[g["end_timestamp"] <= t]
    if explained is not None:
        done = done[done["unique_narration_id"] != key]
    last = done.sort_values(["end_timestamp", "start_timestamp"]).iloc[-1] if len(done) else None
    return t, current, explained, nxt, last


def pair(r):
    return None if r is None else (r["vc"], r["nc"])


def text(r):
    return None if r is None else f"{r['verb_s']} {r['noun_s']}"


# ----------------------------------------------------------------- bigram prior from training participants
trans = defaultdict(Counter)
action_freq = Counter()
for v, g in rows_by_video.items():
    if v[:3] not in TRAIN_PARTICIPANTS:
        continue
    seq = [(a, b) for a, b in zip(g["vc"], g["nc"])]
    action_freq.update(seq)
    for a, b in zip(seq, seq[1:]):
        trans[a][b] += 1
majority = action_freq.most_common(1)[0][0]


def bigram_next(x):
    return trans[x].most_common(1)[0][0] if x in trans and trans[x] else majority


def chain(x, n=3):
    out = []
    for _ in range(n):
        x = bigram_next(x)
        out.append(x)
    return out


# ----------------------------------------------------------------- load predictions
def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


fields = {s: {r["id"]: r for r in load_jsonl(FOLLOW / "scoring" / f"{s}.fields.jsonl")}
          for s in ("base", "epoch1", "epoch2")}
refs = {r["id"]: r for r in load_jsonl(FOLLOW / "forecast" / "references.all_triples.jsonl")}
triples = {s: {r["id"]: r for r in load_jsonl(FOLLOW / "forecast" / f"{s}.all_triples.jsonl")}
           for s in ("epoch1", "epoch2")}


def base_next_triple(raw):
    if not raw or "<next>" not in raw:
        return None
    body = raw.split("<next>", 1)[1].split("</next>", 1)[0]
    parts = [p for p in re.split(r"[;\n]|\d+\.\s", body) if norm(p)]
    return parts[:3] if parts else None


ids = list(refs)
records = []
mapper_check = Counter()
for qid in ids:
    t, current, explained, nxt, last = gold_for(qid)
    ref_triple = refs[qid]["reference_triple"]
    gold_next = [pair(r) for _, r in nxt.iterrows()]
    # alignment and mapper checks against the gold annotation rows
    gold_next_text = [text(r) for _, r in nxt.iterrows()]
    mapper_check["next_text_aligned"] += int([norm(x) for x in ref_triple] == gold_next_text)
    for rt, gp in zip(ref_triple, gold_next):
        mv, mn = parse_action(rt)
        mapper_check["ref_next_verb_ok"] += int(mv == gp[0]); mapper_check["ref_next_noun_ok"] += int(mn == gp[1])
        mapper_check["ref_next_n"] += 1
    task = qid.split(":")[1]
    rec = {"id": qid, "task": task, "t": t, "gold_next": gold_next,
           "gold_current": pair(current) if task == "track" else None,
           "explained": pair(explained) if task == "explain" else None,
           "last_done": pair(last)}
    if task == "track":
        ref_action = fields["epoch1"][qid]["reference_fields"]["action"]
        mapper_check["track_text_aligned"] += int(current is not None and norm(ref_action) == text(current))
        mv, mn = parse_action(ref_action)
        mapper_check["ref_cur_verb_ok"] += int(current is not None and mv == current["vc"])
        mapper_check["ref_cur_noun_ok"] += int(current is not None and mn == current["nc"])
        mapper_check["ref_cur_n"] += 1
    for s in ("base", "epoch1", "epoch2"):
        f = fields[s][qid]
        if task == "track":
            rec[f"{s}_current"] = parse_action(f["extracted_fields"].get("action"))
        if s == "base":
            tri = base_next_triple(f.get("raw_prediction"))
        else:
            tri = triples[s][qid]["predicted_triple"]
        rec[f"{s}_next"] = [parse_action(x) for x in tri] if tri else None
    records.append(rec)

n_next = mapper_check["ref_next_n"]
n_cur = mapper_check["ref_cur_n"]
summary = {
    "queries": len(records),
    "reference_next_triples_aligned_to_annotations": f"{mapper_check['next_text_aligned']}/{len(records)}",
    "track_reference_aligned_to_annotations": f"{mapper_check['track_text_aligned']}/{n_cur}",
    "mapper_on_reference_next_phrases": {"verb": round(mapper_check["ref_next_verb_ok"] / n_next, 3),
                                         "noun": round(mapper_check["ref_next_noun_ok"] / n_next, 3)},
    "mapper_on_reference_current_phrases": {"verb": round(mapper_check["ref_cur_verb_ok"] / n_cur, 3),
                                            "noun": round(mapper_check["ref_cur_noun_ok"] / n_cur, 3)},
}


# ----------------------------------------------------------------- metrics
def acc_pair(pred, gold):
    if gold is None:
        return None
    pv, pn = pred if pred else (None, None)
    return (int(pv == gold[0]), int(pn == gold[1]), int(pv == gold[0] and pn == gold[1]))


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def score_current(get):
    rows = [r for r in records if r["task"] == "track" and r["gold_current"] is not None]
    v = n = a = 0
    for r in rows:
        p = get(r)
        res = acc_pair(p, r["gold_current"])
        v += res[0]; n += res[1]; a += res[2]
    k = len(rows)
    return {"n": k, "verb": round(v / k, 3), "noun": round(n / k, 3), "action": round(a / k, 3)}


def score_next(get, task=None):
    rows = [r for r in records if len(r["gold_next"]) == 3 and (task is None or r["task"] == task)]
    out = Counter()
    for r in rows:
        p = get(r) or []
        p = (p + [(None, None)] * 3)[:3]
        g = r["gold_next"]
        res = acc_pair(p[0], g[0])
        out["next1_verb"] += res[0]; out["next1_noun"] += res[1]; out["next1_action"] += res[2]
        gm = Counter(g); pm = Counter(p)
        out["set_recall_action"] += sum((gm & pm).values()) / 3
        out["ed_verb"] += lev([x[0] for x in p], [x[0] for x in g]) / 3
        out["ed_noun"] += lev([x[1] for x in p], [x[1] for x in g]) / 3
        out["ed_action"] += lev(p, g) / 3
        out["all_three_same"] += int(p[0] == p[1] == p[2] and p[0] != (None, None))
    k = len(rows)
    return {"n": k, **{key: round(val / k, 3) for key, val in out.items()}}


def x_of(r):
    return r["explained"] if r["task"] == "explain" else r["last_done"]


current_table = {
    "base": score_current(lambda r: r["base_current"]),
    "epoch1": score_current(lambda r: r["epoch1_current"]),
    "epoch2": score_current(lambda r: r["epoch2_current"]),
    "baseline: copy last completed action (no video)": score_current(lambda r: r["last_done"]),
    "baseline: most frequent follower of last completed action (no video)":
        score_current(lambda r: bigram_next(r["last_done"]) if r["last_done"] else majority),
    "baseline: most frequent action in training": score_current(lambda r: majority),
}
next_table = {}
for task in (None, "track", "explain"):
    name = task or "all"
    next_table[name] = {
        "base": score_next(lambda r: r["base_next"], task),
        "epoch1": score_next(lambda r: r["epoch1_next"], task),
        "epoch2": score_next(lambda r: r["epoch2_next"], task),
        "baseline: repeat x three times (x = given action / last completed)":
            score_next(lambda r: [x_of(r)] * 3 if x_of(r) else None, task),
        "baseline: bigram chain from x (no video)":
            score_next(lambda r: chain(x_of(r)) if x_of(r) else None, task),
    }
    if task == "track":
        next_table[name]["oracle: bigram chain from TRUE current action"] = score_next(
            lambda r: chain(r["gold_current"]) if r["gold_current"] else None, task)

summary["current_action_track"] = current_table
summary["next_actions"] = next_table
summary["majority_action"] = f"{verb_names[majority[0]]} {noun_names[majority[1]]}"
(OUT / "class_level_summary.json").write_text(json.dumps(summary, indent=1))

# spot-check sheet of mapped predictions
lines = []
for r in records[:400:8]:
    if r["task"] != "track":
        continue
    f = fields["epoch1"][r["id"]]
    pv, pn = r["epoch1_current"]
    gv, gn = r["gold_current"] or (None, None)
    lines.append(f"{r['id']}\tpred='{f['extracted_fields'].get('action')}' -> {verb_names.get(pv)}/{noun_names.get(pn)}"
                 f"\tgold='{f['reference_fields']['action']}' -> {verb_names.get(gv)}/{noun_names.get(gn)}")
(OUT / "mapper_spotcheck.tsv").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=1))

# ----------------------------------------------------------------- extra: fair subsets and paired differences
import random
def cur_ok(r, key):
    p = r[key] or (None, None)
    return int(r["gold_current"] is not None and p[0] == r["gold_current"][0] and p[1] == r["gold_current"][1])
track = [r for r in records if r["task"] == "track" and r["gold_current"] is not None]
ext = [r for r in track if fields["base"][r["id"]]["field_extraction"]["action"]["extractable"]]
extra = {"current_action_on_base_extractable_subset": {
    "n": len(ext),
    "base": round(sum(cur_ok(r, "base_current") for r in ext) / len(ext), 3),
    "epoch1": round(sum(cur_ok(r, "epoch1_current") for r in ext) / len(ext), 3)}}
def next1_ok(r, key):
    p = (r[key] or [(None, None)])[0]
    g = r["gold_next"][0] if r["gold_next"] else None
    return int(g is not None and tuple(p) == tuple(g))
nx = [r for r in records if len(r["gold_next"]) == 3]
def boot(a, b, n=2000, seed=0):
    rnd = random.Random(seed); d = [x - y for x, y in zip(a, b)]; k = len(d)
    means = sorted(sum(d[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return round(sum(d) / k, 3), round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)
extra["paired_epoch1_minus_base"] = {
    "current_action": boot([cur_ok(r, "epoch1_current") for r in track], [cur_ok(r, "base_current") for r in track]),
    "next1_action": boot([next1_ok(r, "epoch1_next") for r in nx], [next1_ok(r, "base_next") for r in nx]),
    "next1_action_vs_repeat_baseline": boot([next1_ok(r, "epoch1_next") for r in nx],
                                           [int(x_of(r) is not None and tuple(x_of(r)) == tuple(r["gold_next"][0])) for r in nx]),
}
(OUT / "class_level_extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps(extra, indent=1))
