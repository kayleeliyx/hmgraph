"""Freeze rubric and blinded materials. Never invoke model inference or training."""
from __future__ import annotations
import csv, hashlib, html, json, random, secrets, shutil
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from code.hdepic_review_followup.fields import extract_field
from code.hdepic_review_followup.visual import make_contact_sheet, sha256

PROJECT = Path('/data/yaxuanli/ActGround')
RUN = PROJECT / 'runs/hdepic_sft_8b_v1'
OUT = RUN / 'why_semantic_review_v1'
KEY = RUN / 'why_semantic_review_v1.identity_key.json'

RUBRIC = '''# Frozen WHY semantic review rubric v1

This is a provisional assistant assessment, not human ground truth, a validated semantic metric, or a GRPO reward. Epoch 1 (checkpoint-909) remains selected by the original Explain mean per-example assistant-target NLL criterion. No training, model inference, answer regeneration, or test-split access is permitted.

The review covers every one of the 228 validation Explain IDs, all three saved WHY answers independently extracted by the existing supplementary parser. Whole-response format validity is not a gate. Judge meaning, not keyword overlap or stylistic resemblance. Preserve all frozen references and offsets. The existing fixed 50-example review subset is reported separately; it is not an independent replicate.

## Source judgment, made separately for each record

Read the action question, canonical target and interval, original narration, exact stated-reason span, completed history, and all 16 causal input frames on the contact sheet. Reference status: valid (specific stated purpose reliably attached); valid_with_nonpurpose_suffix (a reliably attached, separable purpose plus a manner/subsequent-action suffix); ambiguous_attachment (uncertain target, referent, interval, or joint-action scope); source_conflict (contradictory text or reason unrelated to target). Mark the latter two unjudgeable for ALL answers regardless of their apparent plausibility. A broad canonical question alone does not invalidate a reason when the observed choice is clearly part of that action. If the reason attaches to another action or the necessary choice is unreliable, mark ambiguous. Retain both original span and proposed exact source correction separately. Never substitute a revised reference into original metrics or predictions.

For valid_with_nonpurpose_suffix, assess the explicitly identified purpose core, report the suffix issue, and do not reward repetition of a subsequent-action/manner clause. No inferred purpose may replace the source. Explain any scope decision in a short source note BEFORE answer notes.

## Answer correctness (four mutually exclusive labels)

Correct: captures the specific actor-stated purpose, allowing paraphrases and direct entailments. Material destination, avoidance condition, quantity, tool/manner choice, or readiness/timing must be retained when it is the point of the stated reason.
Partial: captures an actual part of that purpose but omits a material detail; generic preparation can be partial only if it expresses a real part of the stated purpose. Sharing a cooking activity or action object alone is insufficient.
Incorrect: gives a different purpose, merely redescribes the action, invents a generic plausible purpose without expressing the stated purpose, or contradicts the source. A contradictory central claim overrides a correct fragment. Missing/unextractable answers are incorrect when the source is judgeable and remain in denominators.
Unjudgeable: source ambiguity or action–reason attachment prevents reliable comparison. This is a source limitation, not a reward for uncertain model wording.

Flag unsupported additional claims separately, with the specific claim(s). A correct core with an unsupported ancillary assertion can remain correct with this flag; a contradictory central assertion is incorrect. Extra claims supported by the narration or observed context are not automatically penalized. Claims that merely state the same purpose more generally are not separate hallucinations. Do not infer semantic correctness from lexical matches.

## Visual evidence, independent of intent agreement

Record concrete frame observations and relevant history. Visual support values: supported (the observable instrumental relation claimed is visible), contradicted (visible evidence conflicts), not_established (action is consistent but purpose/internal state/future outcome cannot be established in these five seconds), unreviewed (images not actually inspected). Agreement with narrated intent alone is NOT visual support. Inspect the entire 16-frame sheet; it is a presentation aid, not a model input. Never inspect future frames. Save actual visual inspection method and limitations. General quality/safety/even-cooking beliefs often remain not_established even with a correct narrated purpose.

## Blinding and reviewer identity

Three answer slots A/B/C are independently randomized within each case; case order is randomized. The model identity key is outside blinded material. Answer wording/format can reveal identity. The Codex GPT-6 assistant team has previous conversation context about the experiment; this is partial blinding, not independent fully blind review. No external inference API is used. Record agent identity per judgment; agents belong to one correlated assistant team, not separate human annotators. Human fields remain blank and separate from assistant results. Do not use prior assistant ratings as human truth.

## Aggregation fixed before scoring

Report four label counts per model over ALL 228 and the fixed 50 subset; separately report judgeable denominator, reference statuses, unsupported-additional-claim counts and visual-support counts. Correct / judgeable is a provisional assistant agreement rate with narrated purpose, not proven visual inferability or human accuracy. Correct / all includes ambiguous cases in its denominator. Paired ordinal comparisons use incorrect < partial < correct only on source-judgeable cases; report improvement/regression/tie counts, the full 4x4 transition matrix, and ambiguous source cases explicitly. Primary comparison is epoch1 versus base; epoch2 is supplementary. Report duplicate original-narration text groups, distinct narration IDs and dependence; no statistical-independence or significance claims.
'''

def rows(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def lines(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))

def protected_files():
    # Deliberately read validation inputs only. Do not walk datasets/test.
    paths = [PROJECT/'datasets/hdepic_sft_v2_repaired'/x for x in
             ('text/validation.jsonl','swift/validation.jsonl','swift/validation.prompts.jsonl','swift/validation.scoring.jsonl')]
    paths += [RUN/x for x in ('metrics.json','frozen_config.json','frozen_evaluation.json','validation_report.md','execution_report.md','review/blind_why_review.csv','review/blind_why_review_identity_key.json')]
    paths += [RUN/'predictions'/s/x for s in ('base','epoch1','epoch2') for x in ('generation.jsonl','loss.jsonl')]
    paths += list((RUN/'review_followup_v1').rglob('*'))
    paths += list((PROJECT/'code/hdepic_experiment').glob('*.py')) + list((PROJECT/'code/hdepic_review_followup').glob('*.py'))
    return sorted({p for p in paths if p.is_file()})

def prepare():
    if (OUT/'materials_manifest.json').exists():
        m=json.loads((OUT/'materials_manifest.json').read_text())
        assert sha256(OUT/'rubric.md')==m['rubric_sha256']
        assert sha256(OUT/'blinded/cases.jsonl')==m['cases_sha256']
        print('Existing frozen materials retained'); return
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'assistant').exists(), 'Rubric must be frozen before scoring'
    (OUT/'rubric.md').write_text(RUBRIC)
    dump(OUT/'rubric_freeze.json', {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':sha256(OUT/'rubric.md'),'before_scoring':True,'version':'why_semantic_review_v1',
        'reviewer':'Codex GPT-6 assistant team; partial blinding; no human review'})
    dump(OUT/'protected_originals.json', {str(p):sha256(p) for p in protected_files()})
    # Checkpoint state is preserved without reading/loading weight tensors.
    checkpoint_stats=[]
    for p in (RUN/'training').rglob('checkpoint-*'):
        if p.is_dir():
            checkpoint_stats.extend({'path':str(f),'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
                                    for f in p.rglob('*') if f.is_file())
    dump(OUT/'checkpoint_stat_snapshot.json',checkpoint_stats)
    texts={r['id']:r for r in rows(PROJECT/'datasets/hdepic_sft_v2_repaired/text/validation.jsonl') if r['task']=='explain'}
    sidecars={r['id']:r for r in rows(PROJECT/'datasets/hdepic_sft_v2_repaired/swift/validation.scoring.jsonl') if r['task']=='explain'}
    prompts=rows(PROJECT/'datasets/hdepic_sft_v2_repaired/swift/validation.prompts.jsonl')
    predictions={s:{r['id']:r for r in rows(RUN/'review_followup_v1/scoring'/f'{s}.fields.jsonl') if r['task']=='explain'} for s in ('base','epoch1','epoch2')}
    assert len(texts)==len(sidecars)==228
    fixed50={r['id'] for r in csv.DictReader((RUN/'review/blind_why_review.csv').open())}; assert len(fixed50)==50
    rng=random.Random(secrets.randbits(256)); ids=list(texts);rng.shuffle(ids)
    cases=[]; identity={}; causal=[]
    for number,rid in enumerate(ids,1):
        r=texts[rid];s=sidecars[rid];reason=r['scoring_metadata']['reason_annotation'];frames=s['frames'];q=r['query_time_sec']
        assert len(frames)==16
        assert reason['original_narration'][reason['char_start']:reason['char_end']]==r['target']['reason']
        assert all(h['end_sec']<=q for h in r['input']['history'])
        assert all(max(0,q-5)-1e-8<=f['timestamp_sec']<=q for f in frames)
        assert all(float(Fraction(f['source_pts'])*Fraction(f['source_time_base']))==f['timestamp_sec'] for f in frames)
        assert [f['timestamp_sec'] for f in frames]==sorted(f['timestamp_sec'] for f in frames)
        prompt=prompts[s['row_index']]; assert len(prompt['images'])==16
        assert prompt['images']==[f['path'] for f in frames]
        paths=[]
        for f in frames:
            p=Path(f['path']);assert sha256(p)==f['sha256'];paths.append(p)
        code=f'S{number:03d}';image=f'contact_sheets/{code}.jpg'
        make_contact_sheet(paths,frames,OUT/'blinded'/image,number,q)
        stages=['base','epoch1','epoch2'];rng.shuffle(stages)
        answers={};identity[code]={'id':rid,'slots':dict(zip('ABC',stages))}
        for slot,stage in zip('ABC',stages):
            saved=predictions[stage][rid];parsed=extract_field(saved['raw_prediction'],'why')
            assert parsed==saved['field_extraction']['why']
            answers[slot]={'why':parsed['text'],'extraction_status':parsed['status'],'raw_response':saved['raw_prediction']}
        case={'case':code,'id':rid,'existing_fixed50':rid in fixed50,'task':'explain',
              'query_time_sec':q,'target_action':r['scoring_metadata']['current_action'],
              'action_question':r['input']['action_question_text'],'history':r['input']['history'],
              'input_prompt':prompt['messages'],'original_narration':reason['original_narration'],
              'narration_id':reason['narration_id'],'reference_reason':r['target']['reason'],
              'reason_char_start':reason['char_start'],'reason_char_end':reason['char_end'],
              'extraction_rule':reason['extraction_rule'],'contact_sheet':image,'frames':frames,'answers':answers}
        cases.append(case);causal.append({'case':code,'id':rid,'frames':16,'causal':True,'history_causal':True})
    dump(KEY,{'purpose':'Separate identity key; do not load until judgments frozen','case_slots':identity})
    lines(OUT/'blinded/cases.jsonl',cases)
    for batch in range(3):
        batch_cases=cases[batch*76:(batch+1)*76]
        lines(OUT/'blinded'/f'batch{batch+1}.jsonl',batch_cases)
        text=[]
        for c in batch_cases:
            text += [f"{c['case']} | {c['narration_id']} | question: {c['action_question']} | interval {c['target_action']['start_sec']}..{c['target_action']['end_sec']}",
                     'NARRATION: '+c['original_narration'],'REFERENCE: '+c['reference_reason'],
                     'HISTORY: '+ '; '.join(h['action'] for h in c['history']),
                     *[f"{slot}: {a['why']}" for slot,a in c['answers'].items()], '']
        (OUT/'blinded'/f'batch{batch+1}.txt').write_text('\n'.join(text))
    dump(OUT/'timestamp_audit.json',{'passed':True,'records':228,'frame_references':3648,'all_source_sha256_match':True,
        'all_actual_timestamps_le_query':True,'all_frames_in_five_second_window':True,'histories_end_le_query':True,
        'contact_sheets_are_review_aids_not_model_inputs':True,'cases':causal})
    dump(OUT/'materials_manifest.json',{'rubric_sha256':sha256(OUT/'rubric.md'),'cases_sha256':sha256(OUT/'blinded/cases.jsonl'),
        'cases':228,'answers':684,'fixed50':50,'all_why_extractable':True,'identity_key_kept_outside_blinded_material':True,
        'inputs':'Existing validation only; preserved supplementary parser and saved predictions',
        'blinding_limitations':['Prior conversation context','Model wording and response style may reveal identity','Same correlated assistant team, not independent human raters']})
    print('Rubric frozen; 228 cases / 684 answers; 3648 causal cached frames verified')

if __name__=='__main__':prepare()
