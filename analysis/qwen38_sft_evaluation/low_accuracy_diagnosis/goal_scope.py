"""Scope of the GOAL target vs HD-EPIC recipe steps, at each query time. CPU only."""
import bisect, json, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

A = Path('/data/kailaic/HDEPIC/data/hd-epic-annotations')
D = Path('/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/swift')
OUT = Path(__file__).resolve().parent / 'goal_scope.json'
nar = pd.read_pickle(A / 'narrations-and-action-segments/HD_EPIC_Narrations.pkl')
starts = defaultdict(list)
for v, s in zip(nar['video_id'], nar['start_timestamp']): starts[v].append(float(s))
for v in starts: starts[v].sort()
recipes = json.load(open(A / 'high-level/complete_recipes.json'))
steps = defaultdict(list)          # video -> (start, end, kind, step_id, text)
for rid, r in recipes.items():
    for cap in r['captures']:
        for kind in ('step_times', 'prep_times'):
            for sid, spans in cap.get(kind, {}).items():
                for sp in spans:
                    steps[sp['video']].append((sp['start'], sp['end'], kind, sid, r['steps'][sid], r['name']))

def summarize(part):
    rows = [json.loads(s) for s in open(D / f'{part}.scoring.jsonl')]
    out = {}
    for task in ('explain', 'track'):
        rs = [r for r in rows if r['task'] == task and r['goal_status'] == 'available']
        dur, n_act, frac, in_step, in_any, whole = [], [], [], 0, 0, 0
        step_dur = []
        for r in rs:
            g, t, v = r['goal_annotation'], r['query_time_sec'], r['video_id']
            d = g['end_sec'] - g['start_sec']; dur.append(d)
            s = starts[v]; n_act.append(bisect.bisect_left(s, g['end_sec']) - bisect.bisect_left(s, g['start_sec']))
            frac.append((t - g['start_sec']) / d if d else 0)
            whole += g['start_sec'] == 0 and abs(g['end_sec'] - max(s)) < 30
            hits = [x for x in steps[v] if x[0] <= t < x[1]]
            in_any += bool(hits); in_step += any(x[2] == 'step_times' for x in hits)
            step_dur += [x[1] - x[0] for x in hits]
        q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None
        out[task] = {'records_with_goal': len(rs),
            'goal_interval_seconds_p25_median_p75': [round(q(dur, .25)), round(q(dur, .5)), round(q(dur, .75))],
            'actions_inside_goal_interval_p25_median_p75': [q(n_act, .25), q(n_act, .5), q(n_act, .75)],
            'query_position_in_interval_median': round(st.median(frac), 2),
            'goal_interval_spans_roughly_whole_video': whole,
            'query_inside_a_recipe_step_or_prep_span': in_any,
            'query_inside_a_recipe_step_span': in_step,
            'covering_recipe_span_seconds_median': round(st.median(step_dur), 1) if step_dur else None}
    return out

result = {p: summarize(p) for p in ('initial_sft', 'validation')}
OUT.write_text(json.dumps(result, indent=1) + '\n'); print(json.dumps(result, indent=1))
