"""CPU scoring and report for the step-hint experiment.

NEXT and current action are scored by the unchanged standalone evaluator on the
650 selected queries. GOAL is not scored here. Paired comparisons use identical
query IDs; McNemar's exact test is reported for NEXT position 1.
"""
from __future__ import annotations

import json
from math import comb
import subprocess
import sys

from code.hdepic_sft_qwen38.common import PROJECT, rows, sha, write_json
from code.hdepic_sft_qwen38.evaluate import require
from .prepare import OUT, TRACK_RUN

MODELS = ('base', 'epoch1')
CONDITIONS = ('no_hint', 'true_step', 'shuffled_step', 'predicted_step')
LABELS = {'no_hint': 'No hint', 'true_step': 'True current step', 'shuffled_step': 'Shuffled (wrong) step',
          'predicted_step': "Model's own predicted step"}
EVAL = OUT / 'evaluation'


def no_hint_file(model, ids):
    """The full Track run's answers on the selected queries; prompts are byte-identical."""
    source = TRACK_RUN / 'predictions' / model / 'generation.jsonl'
    require((TRACK_RUN / 'predictions' / model / 'completed.json').exists(), f'Track run incomplete: {model}')
    expected = {r['id']: r['prompt_sha256'] for r in rows(OUT / 'inputs/prompts/true_step.jsonl')}
    from code.hdepic_qwen38_track_eval.generate import load_records
    track_prompts = {r['id']: r['prompt_sha256'] for r in load_records()[0] if r['id'] in ids}
    selected = [r for r in rows(source) if r['id'] in ids]
    require(len(selected) == len(ids) == len(expected), 'no_hint coverage mismatch')
    require(all(r['prompt_sha256'] == track_prompts[r['id']] for r in selected), 'no_hint prompt differs from Track prompt')
    path = EVAL / 'no_hint' / f'{model}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    write_json(EVAL / 'no_hint' / f'{model}.source.json', {'source': str(source), 'source_sha256': sha(source),
                                                          'selected': len(selected), 'sha256': sha(path)})
    return path


def mcnemar(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def analyze():
    ids = {r['id'] for r in rows(OUT / 'inputs/selection.jsonl')}
    models, comparisons = [], []
    for model in MODELS:
        for condition in CONDITIONS:
            if condition == 'no_hint':
                path = no_hint_file(model, ids)
            else:
                path = OUT / 'predictions' / model / condition / 'generation.jsonl'
                require((path.parent / 'completed.json').exists(), f'Incomplete: {model}/{condition}')
            models.append({'name': f'{model}_{condition}', 'tasks': ['track'], 'predictions': str(path)})
        comparisons += [[f'{model}_no_hint', f'{model}_{c}'] for c in CONDITIONS[1:]]
        comparisons.append([f'{model}_shuffled_step', f'{model}_true_step'])
    config = {'references': str(OUT / 'inputs/references.jsonl'), 'reference_format': 'prepared',
              'mapper': str(PROJECT / 'evaluation/assets/mapper.json'), 'output': 'results',
              'models': models, 'comparisons': comparisons}
    write_json(EVAL / 'config.json', config)
    subprocess.run([sys.executable, str(PROJECT / 'evaluation/evaluate.py'), 'score', '--config',
                    str(EVAL / 'config.json')], check=True)

    per = {m['name']: {r['id']: r for r in rows(EVAL / 'results/per_example' / f"{m['name']}.jsonl")} for m in models}
    hit = lambda rec, k: (len(rec['predicted_next']) > k and None not in rec['next_pairs'][k]
                          and rec['predicted_next'][k] == rec['next_pairs'][k])
    summary = {'queries': len(ids), 'models': {}}
    for model in MODELS:
        table = {}
        for condition in CONDITIONS:
            recs = per[f'{model}_{condition}']
            table[condition] = {f'next{k + 1}': sum(hit(r, k) for r in recs.values()) for k in range(3)}
            table[condition]['current_action'] = sum(None not in r['current_pair'] and r['predicted_current'] == r['current_pair']
                                                     for r in recs.values())
        paired = {}
        for before, after in [('no_hint', c) for c in CONDITIONS[1:]] + [('shuffled_step', 'true_step')]:
            a, b = per[f'{model}_{before}'], per[f'{model}_{after}']
            gained = sum(hit(b[i], 0) and not hit(a[i], 0) for i in ids)
            lost = sum(hit(a[i], 0) and not hit(b[i], 0) for i in ids)
            paired[f'{before}->{after}'] = {'next1_gained': gained, 'next1_lost': lost,
                                            'mcnemar_p': round(mcnemar(gained, lost), 4)}
        choices = rows(OUT / 'predictions' / model / 'predicted_step/step_choices.jsonl')
        summary['models'][model] = {'counts': table, 'paired_next1': paired,
            'step_prediction_correct': sum(c['correct'] for c in choices),
            'step_prediction_unparsed': sum(c['predicted_step_number'] is None for c in choices)}
    manifest = json.loads((OUT / 'inputs/manifest.json').read_text())
    summary['chance_step_accuracy'] = manifest['mean_chance_step_accuracy']
    write_json(EVAL / 'summary.json', summary)
    (OUT / 'report.md').write_text(render(summary))
    return summary


def render(summary):
    n = summary['queries']
    pct = lambda x: f'{x}/{n} ({100 * x / n:.1f}%)'
    lines = ['# Step-hint experiment: results', '',
             f'{n} validation Track queries inside an annotated recipe step. Each condition changes one prompt line. '
             'NEXT k = verb and noun both correct at position k.', '']
    for model, item in summary['models'].items():
        lines += [f'## {model}', '', '| Condition | NEXT 1 | NEXT 2 | NEXT 3 | Current action |', '|---|---|---|---|---|']
        for condition in CONDITIONS:
            c = item['counts'][condition]
            lines.append(f"| {LABELS[condition]} | {pct(c['next1'])} | {pct(c['next2'])} | {pct(c['next3'])} | {pct(c['current_action'])} |")
        lines += ['', '| Paired NEXT 1 | Gained | Lost | McNemar p |', '|---|---|---|---|']
        for key, value in item['paired_next1'].items():
            lines.append(f"| {key} | {value['next1_gained']} | {value['next1_lost']} | {value['mcnemar_p']} |")
        lines += ['', f"Step prediction: {pct(item['step_prediction_correct'])} correct "
                      f"(chance {100 * summary['chance_step_accuracy']:.1f}%), {item['step_prediction_unparsed']} unparsed.", '']
    lines += ['## Reading the result', '',
              '- True step beats both no hint and shuffled step: correct step information helps NEXT.',
              '- Predicted step also beats no hint: the model can produce useful step information itself.',
              '- True step helps but predicted step does not: step recognition is the weak link.',
              '- Neither helps: step information is not useful for this 3-action horizon with this model.',
              '', 'Single participant (P07), one greedy answer per query, heuristic action mapper; '
              'the recipe is assumed known for step prediction. Base answers cut off at 256 tokens lose NEXT.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
