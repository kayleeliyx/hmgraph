"""Freeze two blinded new-model answers against the existing source protocol."""
from pathlib import Path
import csv,hashlib,io,json,secrets,shutil
from datetime import datetime,timezone
from code.hdepic_review_followup.fields import extract_field

PROJECT=Path('/data/yaxuanli/ActGround')
OLD=PROJECT/'runs/hdepic_sft_8b_v1/why_semantic_review_v1'
OUT=PROJECT/'runs/hdepic_why_semantic_comparison_v2'
KEY=PROJECT/'runs/hdepic_why_semantic_comparison_v2.identity_key.json'
RUNS={'qwen35':PROJECT/'runs/hdepic_qwen35_35b_baseline_v1', 'qwen38':PROJECT/'runs/hdepic_qwen38_27b_baseline_v1'}
HUMAN_FIELDS=('human_reference_validity','human_correct_target_attachment','human_answer_correctness','human_unsupported_claims','human_visual_support','human_reviewer_notes')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def jb(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def jl(values):return b''.join((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode() for v in values)
def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data, f'Immutable artifact changed: {path}'
    else:path.write_bytes(data)
def dump(path,value):put(path,jb(value))
def csv_bytes(records):
    h=io.StringIO(newline='');w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records);return h.getvalue().encode()

ADDENDUM='''# Frozen extension: same WHY evaluation for two additional models

The original rubric.md is byte-identical to the completed 8B WHY review. This
extension adds two saved-model answers per each of the same228 validation Explain
IDs, with fresh randomized A/B identities and inherited randomized source-case
order. Primary results reuse the earlier source-adjudication/purpose-core policy:
204 clean sources,22 reliably attached cores with separable nonpurpose suffixes,
and2 source-deficient cases. These are provisional assistant source decisions,
not human truth. Reviewers read narration, question, history and all16 causal
frames again. Any new source concern is written separately; original references,
source decisions and earlier answer ratings are not silently rewritten.

Correct/partial/incorrect/unjudgeable definitions and the unsupported-claim flag
remain unchanged. A plausible purpose is not automatically the stated purpose.
A correct core with an unsupported ancillary claim can remain correct with a flag;
a contradictory central claim is incorrect. Whole-response formatting is not a
gate. A missing/unextractable WHY remains incorrect on a judgeable source under
the original rubric, even when some raw text could be manually recovered. Record
that extraction failure explicitly rather than repairing the parser. The raw
answer remains visible. Source-deficient cases are unjudgeable for both answers.

This evaluates WHY only, as did the completed semantic evaluation. Goal/current-
action/next-action semantic scores are outside this frozen rubric. No training,
answer regeneration, test access, external inference API, or local model inference.

Reviewers may inspect only blinded cases, frozen rubric/source policy and cached
frames. Do not open the identity key, named predictions or prior answer judgments.
Write substantive short evidence notes for both answers; do not use keyword rules,
similarity scores or previous assistant ratings as truth. Existing source notes
are supplied for denominator/purpose-core comparability, not as new human labels.
Reviewer identity is Codex GPT-6 assistant team; exact deployed build unavailable.
Reviewers share prior project context and answer style may reveal identities.
Blinding is partial, and these are correlated assistant assessments, not
independent human annotations. Human judgment columns stay blank and separate.

All228 IDs remain in denominators;226 are source-judgeable. Report all four labels,
unsupported claims, visual support and clean204 subset, plus the fixed50 subset.
Pair each new model against original8B, selectedepoch1 and each other using the
same incorrect<partial<correct ordering and explicit unjudgeable pairs. Earlier
8B ratings are imported unchanged only after new judgments are frozen; epoch2 is
supplementary. Keep all previous artifacts and checkpoint-909 selection intact.
No statistical significance or validated reward claim is warranted from this
single P07 cohort and correlated assistant review. Identical WHY strings across
models should receive consistent meaning labels; contextual unsupported claims
and visual flags must be evidenced separately. Record QA edits before unblinding.
'''

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=rows(OLD/'blinded/cases.jsonl'); assert len(cases)==228
    sources={v['id']:v for v in rows(OLD/'source_issue_decisions.jsonl')}
    original_manifest=json.loads((OLD/'materials_manifest.json').read_text())
    assert sha(OLD/'rubric.md')==original_manifest['rubric_sha256']
    assert sha(OLD/'blinded/cases.jsonl')==original_manifest['cases_sha256']
    ids={c['id'] for c in cases}; assert ids==set(sources)
    protected=json.loads((RUNS['qwen38']/'audit/protected_originals.json').read_text())
    for run in RUNS.values():
        for p in run.rglob('*'):
            if p.is_file() and 'private' not in p.relative_to(run).parts and p.name!='pipeline.lock': protected[str(p)]=sha(p)
    assert all(Path(p).is_file() and sha(p)==h for p,h in protected.items())
    dump(OUT/'protected_originals.json',protected)
    predictions={};provenance={}
    for model,run in RUNS.items():
        gen=run/'predictions/generation.jsonl';mark=json.loads((run/'predictions/completed.json').read_text())
        assert mark['passed'] and mark['count']==228 and sha(gen)==mark['generation_sha256']
        pp={r['id']:r for r in rows(run/'inputs/validation_explain.prompts.jsonl')}
        preds={r['id']:r for r in rows(gen)};assert set(preds)==set(pp)==ids
        for c in cases:
            v=preds[c['id']]; assert v['status']=='completed' and v['prompt_sha256']==pp[c['id']]['prompt_sha256']
            assert pp[c['id']]['messages']==c['input_prompt'] and pp[c['id']]['images']==[f['path'] for f in c['frames']]
        predictions[model]=preds
        provenance[model]={'generation_sha256':sha(gen),'configuration_sha256':mark['configuration_sha256'],'count':228}
    if KEY.exists():seed=json.loads(KEY.read_text())['seed']
    else:seed=secrets.token_hex(32)
    outcases=[];mapping={}
    for c in cases:
        new={k:v for k,v in c.items() if k!='answers'}
        src=sources[c['id']]
        new['frozen_source_decision']={k:src[k] for k in ('reference_status','purpose_core','source_note')}
        order=sorted(predictions,key=lambda m:hashlib.sha256(f'{seed}:{c["id"]}:{m}'.encode()).hexdigest())
        mapping[c['case']]={'id':c['id'],'slots':dict(zip('AB',order))}
        new['answers']={}
        for slot,model in zip('AB',order):
            p=predictions[model][c['id']];f=extract_field(p['prediction'],'why')
            new['answers'][slot]={'why':f['text'],'extraction_status':f['status'],'field_extraction':f,'raw_response':p['prediction'],'finish_reason':p['finish_reason']}
        outcases.append(new)
    dump(KEY,{'seed':seed,'case_slots':mapping,'private':'Do not read until all new judgments are frozen; exclude from archives.'})
    put(OUT/'rubric.md',(OLD/'rubric.md').read_bytes());put(OUT/'extension_protocol.md',ADDENDUM.encode())
    put(OUT/'blinded/cases.jsonl',jl(outcases))
    for batch in range(1,4):
        rr=outcases[(batch-1)*76:batch*76];put(OUT/f'blinded/batch{batch}.jsonl',jl(rr))
        text=[]
        for c in rr:
            d=c['frozen_source_decision']
            text += [f"{c['case']} | {c['id']} | action: {c['action_question']} | interval {c['target_action']['start_sec']}–{c['target_action']['end_sec']}",
                    'NARRATION: '+c['original_narration'],'REFERENCE: '+c['reference_reason'],
                    'SOURCE POLICY: '+json.dumps(d,ensure_ascii=False),'HISTORY: '+'; '.join(h['action'] for h in c['history'])]
            for slot,a in c['answers'].items():text.append(f"{slot} [{a['extraction_status']}]: {a['why']}")
            text.append('')
        put(OUT/f'blinded/batch{batch}.txt','\n'.join(text).encode())
        source_pages=OLD/f'blinded/inspection_batch{batch}'
        for p in source_pages.glob('*.jpg'):put(OUT/f'blinded/inspection_batch{batch}'/p.name,p.read_bytes())
    for c in outcases:put(OUT/'blinded'/c['contact_sheet'],(OLD/'blinded'/c['contact_sheet']).read_bytes())
    human=[{'case':c['case'],'id':c['id'],'slot':slot,'action_question':c['action_question'],'original_narration':c['original_narration'],'reference_reason':c['reference_reason'],'answer_why':a['why'],'extraction_status':a['extraction_status'],'raw_response':a['raw_response'],'contact_sheet':c['contact_sheet'],**{k:'' for k in HUMAN_FIELDS}} for c in outcases for slot,a in c['answers'].items()]
    human_path=OUT/'blinded/human_review.csv'
    if not human_path.exists():put(human_path,csv_bytes(human))
    else:
        actual=list(csv.DictReader(human_path.open()));assert len(actual)==len(human)
        for a,b in zip(actual,human):assert all(a[k]==('' if b[k] is None else str(b[k])) for k in b if k not in HUMAN_FIELDS)
    put(OUT/'source_policy.jsonl',(OLD/'source_issue_decisions.jsonl').read_bytes())
    put(OUT/'shared_source_narrations.json',(OLD/'shared_source_narrations.json').read_bytes())
    manifest={'cases':228,'new_answer_judgments_required':456,'fixed50':sum(c['existing_fixed50'] for c in outcases),'rubric_sha256':sha(OUT/'rubric.md'),'extension_sha256':sha(OUT/'extension_protocol.md'),'cases_sha256':sha(OUT/'blinded/cases.jsonl'),'source_policy_sha256':sha(OUT/'source_policy.jsonl'),'prior_judgments_sha256':sha(OLD/'assistant/per_answer_judgments.jsonl'),'generation_provenance':provenance,'source_policy':'inherited provisional assistant source decisions; new concerns separate','new_model_inference':False,'test_access':False,'prior_answer_ratings_shown_to_reviewers':False}
    dump(OUT/'materials_manifest.json',manifest)
    freeze=OUT/'rubric_freeze.json'
    if not freeze.exists():dump(freeze,{'at_utc':datetime.now(timezone.utc).isoformat(),'rubric_sha256':manifest['rubric_sha256'],'extension_sha256':manifest['extension_sha256'],'cases_sha256':manifest['cases_sha256'],'before_new_scoring':True})
    print(json.dumps({'cases':228,'new_answers':456,'out':str(OUT),'source_policy':'204 clean +22 purpose-core +2 unjudgeable'},indent=2))
if __name__=='__main__':prepare()
