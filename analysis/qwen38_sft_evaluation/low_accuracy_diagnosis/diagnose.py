"""CPU-only diagnosis of low Qwen3.8-27B validation accuracy.

Reads frozen scored outputs and training sidecars; writes only into this folder.
No model, evaluator, judgment, or prediction file is modified.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

P = Path('/data/yaxuanli/ActGround')
sys.path.insert(0, str(P / 'evaluation'))
from hdepic_eval.mapper import ActionMapper
from hdepic_eval.actions import score_action_pairs

OUT = Path(__file__).resolve().parent
FULL = P / 'analysis/qwen38_sft_evaluation/full_validation/full_comparison'
DATA = P / 'datasets/hdepic_sft_v2_repaired/swift'
MODELS = ('base', 'epoch1', 'epoch2')

rows = lambda p: [json.loads(s) for s in open(p) if s.strip()]
norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', (s or '').lower())).strip()
pair = lambda a: tuple(a['main_action_class_ids'][0]) if a.get('main_action_class_ids') else (None, None)
words = lambda s: len((s or '').split())

refs = {r['id']: r for r in rows(FULL / 'references.jsonl')}
per = {m: {r['id']: r for r in rows(FULL / f'scores/per_example/{m}.jsonl')} for m in MODELS}
ids = [r['id'] for r in rows(FULL / 'references.jsonl')]
train = rows(DATA / 'initial_sft.scoring.jsonl')
mapper = ActionMapper.from_dict(json.load(open(P / 'evaluation/assets/mapper.json')))
label = lambda m, i, f: ((per[m][i]['semantic'][f].get('judgment') or {}).get('label'))
result = {}

# ---------------- GOAL ----------------
train_goals = Counter(); goal_participants = defaultdict(set)
for r in train:
    if r['goal_status'] == 'available':
        g = r['goal_annotation']['label']; train_goals[norm(g)] += 1; goal_participants[norm(g)].add(r['participant'])
val_goals = Counter(norm(refs[i]['goal']['reference']) for i in ids)
goal = {'train_unique_labels': len(train_goals), 'val_unique_labels': len(val_goals),
        'val_labels_seen_verbatim_in_training': sum(1 for g in val_goals if g in train_goals),
        'val_label_counts': val_goals.most_common(), 'models': {}}
for m in MODELS:
    preds = [norm(per[m][i]['fields']['goal']['text']) for i in ids]
    copied_q = sum(p == norm(refs[i]['question']) for p, i in zip(preds, ids))
    memorized = [p for p in preds if p in train_goals]
    labs = Counter(label(m, i, 'goal') for i in ids)
    contra = sum(bool(((per[m][i]['semantic']['goal'].get('judgment') or {}).get('flags') or {}).get('contradiction')) for i in ids)
    goal['models'][m] = {'labels': dict(labs), 'unique_predictions': len(set(preds)),
        'prediction_equals_queried_action': copied_q,
        'prediction_is_verbatim_training_goal_label': len(memorized),
        'memorized_by_source_participant': dict(Counter('/'.join(sorted(goal_participants[p])) for p in memorized)),
        'memorized_and_correct_or_partial': sum(1 for p, i in zip(preds, ids) if p in train_goals and label(m, i, 'goal') in ('correct', 'partial')),
        'contradiction_flags': contra,
        'top_predictions': Counter(preds).most_common(12)}
result['goal'] = goal

# ---------------- WHY ----------------
train_why = Counter(norm(r['reason_annotation']['reason_text']) for r in train if r['task'] == 'explain' and r.get('reason_annotation'))
def why_type(s):
    s = norm(s)
    for k in ('so that', 'in order to', 'because', 'to ', 'as '):
        if s.startswith(k.strip() + ' ') or s == k.strip():
            return k.strip()
    return 'other'
elig = [i for i in ids if refs[i]['why']['eligible']]
why = {'eligible': len(elig), 'train_explain_targets': sum(train_why.values()), 'train_unique_why': len(train_why),
       'reference_words_median': sorted(words(refs[i]['why']['reference']) for i in elig)[len(elig)//2],
       'reference_type_counts': dict(Counter(why_type(refs[i]['why']['reference']) for i in elig)), 'models': {}}
for m in MODELS:
    texts = {i: per[m][i]['fields']['why']['text'] or '' for i in elig}
    lens = sorted(words(t) for t in texts.values())
    by_type = defaultdict(Counter)
    for i in elig: by_type[why_type(refs[i]['why']['reference'])][label(m, i, 'why')] += 1
    # Correctness by answer-length tercile (base verbosity / hedging hypothesis).
    order = sorted(elig, key=lambda i: words(texts[i])); t = len(order) // 3
    terciles = {}
    for name, part in (('short', order[:t]), ('mid', order[t:2*t]), ('long', order[2*t:])):
        c = Counter(label(m, i, 'why') for i in part)
        terciles[name] = {'n': len(part), 'median_words': words(texts[part[len(part)//2]]), 'correct': c['correct'], 'partial': c['partial']}
    flags = Counter()
    for i in elig:
        for k, v in (((per[m][i]['semantic']['why'].get('judgment') or {}).get('flags')) or {}).items():
            flags[k] += bool(v)
    why['models'][m] = {'labels': dict(Counter(label(m, i, 'why') for i in elig)),
        'answer_words_median': lens[len(lens)//2], 'answer_words_max': lens[-1],
        'unique_answers': len(set(norm(x) for x in texts.values())),
        'verbatim_training_why': sum(norm(x) in train_why for x in texts.values()),
        'answer_type_counts': dict(Counter(why_type(x) for x in texts.values())),
        'correct_by_reference_type': {k: f"{v['correct']}/{sum(v.values())}" for k, v in sorted(by_type.items())},
        'correct_by_answer_length_tercile': terciles, 'flags': dict(flags)}
for later in ('epoch1', 'epoch2'):
    lost = [i for i in elig if label('base', i, 'why') == 'correct' and label(later, i, 'why') != 'correct']
    gained = [i for i in elig if label('base', i, 'why') != 'correct' and label(later, i, 'why') == 'correct']
    why[f'base_to_{later}'] = {'lost': len(lost), 'gained': len(gained),
        'lost_became': dict(Counter(label(later, i, 'why') for i in lost)),
        'base_words_on_lost_median': sorted(words(per['base'][i]['fields']['why']['text']) for i in lost)[len(lost)//2],
        'sft_words_on_lost_median': sorted(words(per[later][i]['fields']['why']['text']) for i in lost)[len(lost)//2]}
all3_wrong = [i for i in elig if all(label(m, i, 'why') != 'correct' for m in MODELS)]
any_right = [i for i in elig if any(label(m, i, 'why') == 'correct' for m in MODELS)]
why['oracle_any_model_correct'] = len(any_right); why['no_model_correct'] = len(all3_wrong)
why['no_model_correct_examples'] = [{'id': i, 'action': refs[i]['question'], 'reference': refs[i]['why']['reference'],
    'epoch1': per['epoch1'][i]['fields']['why']['text']} for i in all3_wrong[:40]]
comp = [i for i in elig if re.search(r'\b(all|excess|every|completely|remaining)\b', norm(refs[i]['why']['reference']))]
why['completeness_detail_references'] = {'n': len(comp), **{m: dict(Counter(label(m, i, 'why') for i in comp)) for m in MODELS}}
result['why'] = why

# ---------------- NEXT ----------------
def score(preds):
    out = {}
    for k in range(3):
        p = [tuple(preds[i][k]) if k < len(preds[i]) else (None, None) for i in ids]
        g = [tuple(refs[i]['next_pairs'][k]) for i in ids]
        s = score_action_pairs(p, g); out[f'pos{k+1}'] = {c: f"{s[c]['correct']}/{s[c]['denominator']}" for c in ('verb', 'noun', 'action')}
    tot = sum(int(out[f'pos{k+1}']['action'].split('/')[0]) for k in range(3))
    out['pooled_action'] = f"{tot}/{3*len(ids)} ({100*tot/(3*len(ids)):.1f}%)"
    return out
nxt = {'models': {}, 'baselines': {}}
for m in MODELS:
    preds = {i: per[m][i]['predicted_next'][:3] for i in ids}
    acts = [a for i in ids for a in per[m][i]['next_extraction']['actions'][:3]]
    unmapped = sum(1 for i in ids for p in per[m][i]['predicted_next'][:3] if None in p)
    hist_copy = sum(1 for i in ids for a in per[m][i]['next_extraction']['actions'][:3]
                    if norm(a) in {norm(h['action']) for h in refs[i]['history']} | {norm(refs[i]['question'])})
    nxt['models'][m] = {**score(preds), 'unmapped_slots': f'{unmapped}/{3*len(ids)}', 'slots_copying_history_or_query': f'{hist_copy}/{len(acts)}'}

# Reference-side structure: how often gold next class repeats the query or a history action.
rep = Counter()
for i in ids:
    seen = {tuple(refs[i]['current_pair'])} | {tuple(mapper.map_text(h['action'])) for h in refs[i]['history']}
    for k, g in enumerate(refs[i]['next_pairs']):
        rep[f'pos{k+1}_in_query_or_history'] += tuple(g) in seen
        rep[f'pos{k+1}_equals_query'] += tuple(g) == tuple(refs[i]['current_pair'])
nxt['reference_repetition'] = {k: f'{v}/{len(ids)}' for k, v in sorted(rep.items())}

# Causal, image-free baselines fitted on initial-SFT only (both tasks).
cond = defaultdict(lambda: [Counter(), Counter(), Counter()]); cond2 = defaultdict(lambda: [Counter(), Counter(), Counter()])
glob = [Counter(), Counter(), Counter()]
for r in train:
    cur = pair(r['current_action']); prev = pair(r['history_actions'][-1]) if r['history_actions'] else None
    for k, f in enumerate(r['future_actions'][:3]):
        fp = pair(f); cond[cur][k][fp] += 1; cond2[(prev, cur)][k][fp] += 1; glob[k][fp] += 1
def markov(i, bigram=False):
    cur = tuple(mapper.map_text(refs[i]['question']))
    prev = tuple(mapper.map_text(refs[i]['history'][-1]['action'])) if refs[i]['history'] else None
    out = []
    for k in range(3):
        for table in ([cond2.get((prev, cur))] if bigram else []) + [cond.get(cur), glob]:
            if table and sum(table[k].values()) >= (3 if table is not glob else 0):
                out.append(table[k].most_common(1)[0][0]); break
    return out
def history_continuation(i):
    """If the queried class occurred earlier in history, replay what followed it."""
    cur = tuple(mapper.map_text(refs[i]['question'])); h = [tuple(mapper.map_text(x['action'])) for x in refs[i]['history']]
    seq = h + [cur]; fallback = markov(i)
    for j in range(len(h) - 1, -1, -1):
        if h[j] == cur:
            cont = seq[j + 1:j + 4]
            return (cont + fallback[len(cont):])[:3]
    return fallback
nxt['baselines'] = {
    'repeat_queried_action_x3': score({i: [tuple(mapper.map_text(refs[i]['question']))] * 3 for i in ids}),
    'train_markov_given_query_class': score({i: markov(i) for i in ids}),
    'train_markov_given_last_history_and_query': score({i: markov(i, True) for i in ids}),
    'history_continuation_then_markov': score({i: history_continuation(i) for i in ids}),
}
# Order tolerance and class rarity: how much error is ordering vs unseen classes.
pf = Counter()
for r in train:
    for a in r['future_actions'][:3]:
        c = a.get('main_action_class_ids') or []
        if c and len(c[0]) == 2: pf[tuple(c[0])] += 1
bucket = lambda c: '0_unseen' if c == 0 else ('1_rare_lt30' if c < 30 else ('2_mid_lt200' if c < 200 else '3_freq_ge200'))
for m in MODELS:
    anyslot = pos1any = 0; hit = Counter(); n = Counter()
    for i in ids:
        p = [tuple(x) for x in per[m][i]['predicted_next'][:3]]; g = [tuple(x) for x in refs[i]['next_pairs']]
        anyslot += sum(x in g for x in p); pos1any += bool(p) and p[0] in g
        for k in range(3):
            b = bucket(pf[g[k]]); n[b] += 1; hit[b] += len(p) > k and p[k] == g[k]
    nxt['models'][m].update({'pred_slots_matching_any_gold_slot': f'{anyslot}/{3*len(ids)}',
        'pos1_matching_any_gold_slot': f'{pos1any}/{len(ids)}',
        'joint_by_gold_pair_train_frequency': {k: f'{hit[k]}/{n[k]}' for k in sorted(n)}})
# Predictability of the gold next action given the current class in training (top-1 rate).
tot = sum(sum(c[0].values()) for c in cond.values()); top = sum(c[0].most_common(1)[0][1] for c in cond.values() if c[0])
nxt['train_in_sample_top1_next_given_current_class'] = f'{top}/{tot} ({100*top/tot:.1f}%)'
result['next'] = nxt

(OUT / 'diagnosis.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
print(json.dumps({k: v for k, v in result.items()}, indent=1, default=str)[:20000])
