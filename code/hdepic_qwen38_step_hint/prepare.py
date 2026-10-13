"""CPU-only inputs for the step-hint experiment (650 validation Track queries).

Only one prompt line changes between conditions; it is inserted after the action
history, before the answering instructions:

  no_hint         the full Track run's answers (identical prompt and settings; not regenerated)
  true_step       + "Current recipe step: <annotated step active at the query time>"
  shuffled_step   + the annotated step of another query (seeded; never the true step)
  predicted_step  + the step chosen by the same model in a separate step_prediction prompt

step_prediction shows the recipe name and its numbered steps, so it assumes the
recipe being prepared is known. Queries are those inside exactly one annotated
recipe-step interval (step_times; preparation intervals are not used).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re

from code.hdepic_sft_qwen38.common import PROJECT, RUN as SFT_RUN, read_json, rows, sha, write_json
from code.hdepic_sft_qwen38.evaluate import _validate_prompt, require
from code.hdepic_qwen38_baseline.data import prompt_digest
from code.hdepic_qwen38_track_eval.generate import OUT as TRACK_RUN, load_records

OUT = PROJECT / 'runs/hdepic_qwen38_27b_step_hint'
RECIPES = Path('/data/kailaic/HDEPIC/data/hd-epic-annotations/high-level/complete_recipes.json')
REFERENCES = PROJECT / 'evaluation/assets/references.jsonl'
SEED = 20261013
ANCHOR = '\nBased on the supplied observations'
HINT_CONDITIONS = ('true_step', 'shuffled_step')
STEP_SYSTEM = ('Use the timestamped frames and completed action history to answer the task. '
               'Return only the requested tagged field.')


def write_rows_once(path, values):
    """Deterministic inputs: an existing file must be byte-identical."""
    data = ''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text() == data, f'Prepared input changed: {path}')
    else:
        path.write_text(data)


def prompt(record, messages):
    result = {'id': record['id'], 'row_index': record['row_index'], 'messages': messages,
              'images': record['images'], 'prompt_sha256': prompt_digest(messages, record['images'])}
    _validate_prompt(result)
    return result


def with_hint(record, step_text):
    """Track prompt plus one line; with no step it is the unchanged Track prompt."""
    if step_text is None:
        return prompt(record, record['messages'])
    user = record['messages'][1]['content']
    require(user.count(ANCHOR) == 1, 'Unexpected Track prompt layout')
    user = user.replace(ANCHOR, f'\nCurrent recipe step: {step_text}{ANCHOR}')
    return prompt(record, [record['messages'][0], {'role': 'user', 'content': user}])


def step_question(record, recipe_name, steps):
    user = record['messages'][1]['content']
    require(user.startswith('Task: Track\n') and user.count(ANCHOR) == 1, 'Unexpected Track prompt layout')
    observed = user[:user.index(ANCHOR)].replace('Task: Track\n', 'Task: Step\n', 1)
    listing = '\n'.join(f'{number}. {text}' for number, (_, text) in enumerate(steps, 1))
    user = (f'{observed}\nRecipe being prepared: {recipe_name}\nRecipe steps:\n{listing}\n'
            'Which recipe step is the person currently doing? Answer with the step number in <step>.\n'
            'Return only this field: <step>.')
    return prompt(record, [{'role': 'system', 'content': STEP_SYSTEM}, {'role': 'user', 'content': user}])


def recipe_steps():
    recipes = json.loads(RECIPES.read_text())
    ordered, spans = {}, {}
    for rid, recipe in recipes.items():
        if not rid.startswith('P07_'):
            continue
        ordered[rid] = (recipe['name'], sorted(recipe['steps'].items()))
        for capture in recipe['captures']:
            for sid, intervals in capture.get('step_times', {}).items():
                for interval in intervals:
                    spans.setdefault(interval['video'], []).append(
                        (interval['start'], interval['end'], sid, rid))
    return ordered, spans


def shuffled(selection):
    """Seeded permutation with every assigned step different from the query's own step."""
    order = list(range(len(selection)))
    random.Random(SEED).shuffle(order)
    step = [s['step_id'] for s in selection]
    for i in range(len(order)):
        if step[order[i]] != step[i]:
            continue
        j = next(j for j in range(len(order))
                 if step[order[j]] != step[i] and step[order[i]] != step[j])
        order[i], order[j] = order[j], order[i]
    require(all(step[order[i]] != step[i] for i in range(len(order))), 'Shuffled control kept a true step')
    return order


def prepare():
    records, _, sources = load_records()
    spec = read_json(SFT_RUN / 'inputs/manifest.json')['partitions']['validation']
    require(sha(spec['scoring']) == spec['scoring_sha256'], 'Frozen validation scoring changed')
    scoring = {r['id']: r for r in rows(spec['scoring'])}
    ordered, spans = recipe_steps()
    selection, chosen = [], []
    for record in records:
        meta = scoring[record['id']]
        t = meta['query_time_sec']
        active = {(sid, rid) for start, end, sid, rid in spans.get(meta['video_id'], []) if start <= t < end}
        if not active:
            continue
        require(len(active) == 1, 'Query inside more than one annotated step: ' + record['id'])
        (sid, rid), = active
        name, steps = ordered[rid]
        number = [s for s, _ in steps].index(sid) + 1
        selection.append({'id': record['id'], 'video_id': meta['video_id'], 'query_time_sec': t,
                          'recipe_id': rid, 'recipe_name': name, 'recipe_step_count': len(steps),
                          'step_id': sid, 'step_number': number, 'step_text': steps[number - 1][1]})
        chosen.append(record)
    require(len(selection) == 650, f'Expected 650 step-covered Track queries, found {len(selection)}')
    for i, j in enumerate(shuffled(selection)):
        selection[i].update(shuffled_step_id=selection[j]['step_id'], shuffled_step_text=selection[j]['step_text'])
    prompts = {
        'step_prediction': [step_question(r, s['recipe_name'], ordered[s['recipe_id']][1]) for r, s in zip(chosen, selection)],
        'true_step': [with_hint(r, s['step_text']) for r, s in zip(chosen, selection)],
        'shuffled_step': [with_hint(r, s['shuffled_step_text']) for r, s in zip(chosen, selection)],
    }
    ids = {s['id'] for s in selection}
    references = [r for r in rows(REFERENCES) if r['id'] in ids]
    require(len(references) == 650 and all(r['task'] == 'track' for r in references), 'Reference subset mismatch')
    write_rows_once(OUT / 'inputs/selection.jsonl', selection)
    write_rows_once(OUT / 'inputs/references.jsonl', references)
    for name, values in prompts.items():
        write_rows_once(OUT / 'inputs/prompts' / f'{name}.jsonl', values)
    manifest = {'queries': 650, 'videos': len({s['video_id'] for s in selection}),
                'recipes': sorted({s['recipe_id'] for s in selection}),
                'mean_chance_step_accuracy': sum(1 / s['recipe_step_count'] for s in selection) / 650,
                'shuffle_seed': SEED, 'track_prompt_sources': sources,
                'no_hint_source': str(TRACK_RUN / 'predictions/<model>/generation.jsonl'),
                'files_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT / 'inputs').rglob('*.jsonl'))},
                'recipes_source': str(RECIPES), 'recipes_sha256': sha(RECIPES),
                'references_source': str(REFERENCES), 'references_sha256': sha(REFERENCES)}
    write_json(OUT / 'inputs/manifest.json', manifest, immutable=True)
    return manifest


def parse_step(text, count):
    match = re.search(r'<step>\s*(\d+)\s*</step>', text or '') or re.search(r'\b(\d+)\b', text or '')
    number = int(match.group(1)) if match else None
    return number if number is not None and 1 <= number <= count else None


def predicted_step_prompts(model, step_predictions):
    """After step_prediction completes: hint = the model's own chosen step.
    An unparseable choice gets no hint line (identical to no_hint) and counts as wrong."""
    selection = rows(OUT / 'inputs/selection.jsonl')
    by_id = {r['id']: r for r in load_records()[0]}
    ordered, _ = recipe_steps()
    choices, values = [], []
    for s in selection:
        row = step_predictions[s['id']]
        require(row['status'] == 'completed', 'Step prediction incomplete')
        number = parse_step(row['prediction'], s['recipe_step_count'])
        text = ordered[s['recipe_id']][1][number - 1][1] if number else None
        choices.append({'id': s['id'], 'predicted_step_number': number, 'correct': number == s['step_number']})
        values.append(with_hint(by_id[s['id']], text))
    folder = OUT / 'predictions' / model / 'predicted_step'
    write_rows_once(folder / 'step_choices.jsonl', choices)
    write_rows_once(folder / 'prompts.jsonl', values)
    return values


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
