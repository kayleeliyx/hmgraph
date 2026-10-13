"""Check manual blinded judgments, freeze them, then aggregate saved model answers.

No inference, automatic semantic labeling, or reference mutation is implemented.
"""
from __future__ import annotations
import argparse
import csv
import html
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .prepare import PROJECT, OLD, OUT, KEY, RUNS, HUMAN_FIELDS, sha, rows, jb, jl, put, dump
from code.hdepic_review_followup.fields import extract_field

LABELS = ('correct', 'partial', 'incorrect', 'unjudgeable')
SUPPORTS = ('supported', 'contradicted', 'not_established', 'unreviewed')
MODELS = ('base', 'epoch1', 'epoch2', 'qwen35', 'qwen38')
NAMES = {'base':'Original Qwen3-VL-8B', 'epoch1':'Qwen3-VL-8B SFT epoch 1',
         'epoch2':'Qwen3-VL-8B SFT epoch 2 (supplementary)',
         'qwen35':'Untuned Qwen3.5-35B-A3B', 'qwen38':'Untuned Qwen3.8-27B'}
PAIRS = (('base','epoch1'),('base','qwen35'),('base','qwen38'),
         ('epoch1','qwen35'),('epoch1','qwen38'),('qwen35','qwen38'),
         ('base','epoch2'),('epoch1','epoch2'))


def csv_data(records, fields=None):
    fields = fields or list(dict.fromkeys(k for r in records for k in r))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for r in records:
        writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in r.items()})
    return stream.getvalue().encode()


def validate(cases, judgments, root=OUT, expected=228):
    assert len(cases)==len(judgments)==expected, 'Incomplete cases or judgments'
    assert len({c['case'] for c in cases})==expected
    assert {c['case'] for c in cases}==set(judgments)
    for c in cases:
        j = judgments[c['case']]
        assert j['id']==c['id'], (c['case'], 'wrong ID')
        frozen = c['frozen_source_decision']
        for k in ('reference_status','purpose_core'):
            assert j[k]==frozen[k], (c['case'], 'frozen source policy changed', k)
        for k in ('reviewer','source_note','visual_observation'):
            assert isinstance(j[k],str) and len(j[k])>=15, (c['case'],k)
        assert isinstance(j['visual_reviewed'],bool)
        assert j['inspection_pages'] or not j['visual_reviewed']
        for p in j['inspection_pages']:
            path=(root/p).resolve()
            assert path.is_relative_to(root.resolve()) and path.is_file(), (c['case'],p)
        ambiguous=j['reference_status'] in ('ambiguous_attachment','source_conflict')
        if not ambiguous:
            assert j['purpose_core'] and j['purpose_core'] in c['original_narration']
        assert set(j['answers'])==set('AB')
        for slot,a in j['answers'].items():
            assert a['label'] in LABELS and a['visual_support'] in SUPPORTS
            assert isinstance(a['evidence_note'],str) and len(a['evidence_note'])>=15
            assert isinstance(a['unsupported_additional_claims'],list)
            assert all(isinstance(s,str) and s.strip() for s in a['unsupported_additional_claims'])
            assert (a['label']=='unjudgeable')==ambiguous, (c['case'],slot,'source ambiguity')
            if not c['answers'][slot]['why'] and not ambiguous:
                assert a['label']=='incorrect', (c['case'],slot,'unextractable WHY')
            if not j['visual_reviewed']:
                assert a['visual_support']=='unreviewed'
        assert isinstance(j.get('new_source_concerns',[]),list)
        assert isinstance(j.get('proposed_corrections',[]),list)
        for proposal in j.get('proposed_corrections',[]):
            assert proposal.get('reason') or proposal.get('note'), (c['case'],'unexplained source proposal')
            if 'proposed_exact_span' in proposal:
                a,b=proposal['proposed_char_start'],proposal['proposed_char_end']
                assert c['original_narration'][a:b]==proposal['proposed_exact_span']


def metrics(records, models=MODELS):
    result={}
    for m in models:
        rr=[r for r in records if r['model']==m]
        counts={label:sum(r['label']==label for r in rr) for label in LABELS}
        n=len(rr); judgeable=n-counts['unjudgeable']
        result[m]={'denominator_all':n, 'judgeable':judgeable, 'label_counts':counts,
                   'correct_fraction_all':counts['correct']/n if n else None,
                   'correct_fraction_judgeable':counts['correct']/judgeable if judgeable else None,
                   'unsupported_additional_claims':sum(bool(r['unsupported_additional_claims']) for r in rr),
                   'visual_support_counts':{s:sum(r['visual_support']==s for r in rr) for s in SUPPORTS}}
    return result


def paired(records, before, after):
    by_id=defaultdict(dict)
    for r in records:
        if r['model'] in (before,after):
            assert r['model'] not in by_id[r['id']], 'Duplicate per-model ID'
            by_id[r['id']][r['model']]=r
    matrix={a:{b:0 for b in LABELS} for a in LABELS}
    counts=Counter(); examples=defaultdict(list); order={'incorrect':0,'partial':1,'correct':2}
    for rid,p in by_id.items():
        assert set(p)=={before,after}, ('Unpaired record',rid)
        a,b=p[before]['label'],p[after]['label'];matrix[a][b]+=1
        kind='unjudgeable_source' if 'unjudgeable' in (a,b) else 'improvement' if order[b]>order[a] else 'regression' if order[b]<order[a] else 'tie'
        counts[kind]+=1
        examples[kind].append({'case':p[before]['case'],'id':rid,'from':a,'to':b})
    return {'before':before,'after':after,'denominator_all':len(by_id),
            'judgeable_pairs':len(by_id)-counts['unjudgeable_source'],
            'counts':{k:counts[k] for k in ('improvement','regression','tie','unjudgeable_source')},
            'transition_matrix':matrix,'examples':dict(examples)}


def verify_mapping(cases, key, predictions):
    """Bind attribution to unchanged raw saved answers, not merely a mutable key."""
    assert set(key)=={c['case'] for c in cases}, 'Mapping case set changed'
    assert set(predictions)=={'qwen35','qwen38'}
    for c in cases:
        item=key[c['case']]
        assert item['id']==c['id'] and set(item['slots'])==set('AB')
        assert set(item['slots'].values())=={'qwen35','qwen38'}
        for slot,model in item['slots'].items():
            saved=predictions[model][c['id']]['prediction']
            answer=c['answers'][slot]
            assert answer['raw_response']==saved, (c['case'],slot,'Mapping does not match frozen prediction')
            parsed=extract_field(saved,'why')
            assert answer['why']==parsed['text'] and answer['extraction_status']==parsed['status']
            assert answer['field_extraction']==parsed


def load_checked():
    manifest=json.loads((OUT/'materials_manifest.json').read_text())
    for file,key in (('rubric.md','rubric_sha256'),('extension_protocol.md','extension_sha256'),
                     ('blinded/cases.jsonl','cases_sha256'),('source_policy.jsonl','source_policy_sha256')):
        assert sha(OUT/file)==manifest[key], f'Frozen material changed: {file}'
    files=[OUT/f'assistant/batch{i}.jsonl' for i in range(1,4)]
    missing=[str(p) for p in files if not p.is_file()]
    if missing:raise RuntimeError('Manual semantic judgments incomplete: '+str(missing))
    cases=rows(OUT/'blinded/cases.jsonl'); jj=[j for p in files for j in rows(p)]
    judgments={j['case']:j for j in jj}
    assert len(jj)==len(judgments), 'Duplicate judgment cases'
    validate(cases,judgments)
    # Identical extracted WHY text within a case must receive the same meaning label.
    conflicts=[]
    for c in cases:
        a,b=(c['answers'][s]['why'] for s in 'AB')
        if a and a.strip()==(b or '').strip() and judgments[c['case']]['answers']['A']['label']!=judgments[c['case']]['answers']['B']['label']:
            conflicts.append(c['case'])
    assert not conflicts, ('Blinded identical-answer consistency requires review',conflicts)
    return manifest, files, cases, judgments


def audit_preservation(cases):
    protected=json.loads((OUT/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
    stats=json.loads((OLD/'checkpoint_stat_snapshot.json').read_text())
    stat_changed=[r['path'] for r in stats if not Path(r['path']).is_file() or
                  Path(r['path']).stat().st_size!=r['size'] or Path(r['path']).stat().st_mtime_ns!=r['mtime_ns']]
    frames={};causal=[]
    for c in cases:
        assert len(c['frames'])==16
        times=[f['timestamp_sec'] for f in c['frames']]
        assert times==sorted(times) and max(times)<=c['query_time_sec']+1e-9
        assert all(h['end_sec']<=c['query_time_sec'] for h in c['history'])
        assert len(c['input_prompt'])==2 and [m['role'] for m in c['input_prompt']]==['system','user']
        for f in c['frames']:frames[f['path']]=f['sha256']
        causal.append(c['id'])
    frame_changed=[p for p,h in frames.items() if sha(p)!=h]
    assert not changed and not stat_changed and not frame_changed, (changed,stat_changed,frame_changed)
    return {'passed':True,'original_files_sha256_checked':len(protected),'checkpoint_stats_checked':len(stats),
            'unique_input_frames_sha256_checked':len(frames),'causal_records_checked':len(causal),
            'changed_originals':changed,'changed_checkpoints':stat_changed,'changed_frames':frame_changed,
            'training':False,'new_inference':False,'test_data_access':False}


def build_html(cases,judgments,records):
    e=lambda x:html.escape(str(x));blind=[];scored=[]
    by_id=defaultdict(list)
    for r in records:by_id[r['id']].append(r)
    for c in cases:
        source=(f"<h2>{c['case']} — {e(c['action_question'])}</h2><p>{e(c['id'])}; query {c['query_time_sec']:.3f}s</p>"
                f"<p><b>Narration:</b> {e(c['original_narration'])}</p><p><b>Frozen reference:</b> {e(c['reference_reason'])}</p>"
                f"<p><b>Assessed purpose core:</b> {e(c['frozen_source_decision']['purpose_core'])}</p>"
                f"<details><summary>Completed history and exact answering-model prompt</summary><p>{e('; '.join(h['action'] for h in c['history']))}</p>"
                f"<pre>{e(json.dumps(c['input_prompt'],ensure_ascii=False,indent=2))}</pre></details>")
        answers=''.join(f"<article><h3>Answer {s}</h3><p>{e(a['why'])}</p><p>Extraction: {e(a['extraction_status'])}</p><details><summary>Raw answer</summary><pre>{e(a['raw_response'])}</pre></details></article>" for s,a in c['answers'].items())
        blind.append(f"<section id='{c['case']}'>{source}<img loading='lazy' src='{c['contact_sheet']}' alt='16 causal frames'><div class='answers'>{answers}</div></section>")
        rated=''.join(f"<article><h3>{e(NAMES[r['model']])}: {e(r['label'])}</h3><p>{e(r['answer_why'])}</p><p>{e(r['evidence_note'])}</p><p>Unsupported additions: {e(r['unsupported_additional_claims'])}; visual: {e(r['visual_support'])}</p></article>" for r in by_id[c['id']])
        scored.append(f"<section id='{c['case']}'>{source}<img loading='lazy' src='blinded/{c['contact_sheet']}' alt='16 causal frames'><p>New visual inspection: {e(judgments[c['case']]['visual_observation'])}</p>{rated}</section>")
    header="<!doctype html><html><head><meta charset='utf-8'><style>body{font:16px system-ui;max-width:1600px;margin:auto;padding:20px}section{border-top:2px solid #ccc;padding:24px 0}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}.answers{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{padding:15px;background:#f4f6f8;margin:14px 0}</style></head><body>"
    put(OUT/'blinded/index.html',(header+"<h1>Blinded WHY review: 228 validation cases</h1><p>Human review pending. Answer names hidden; style and prior context limit blinding. Contact sheets present the 16 separate causal input images. Source narration/reference are reviewer-only evidence and were never supplied to the answering models. <a href='human_review.csv'>Blank human fields</a>; <a href='../rubric.md'>rubric</a>.</p>"+''.join(blind)+"</body></html>").encode())
    put(OUT/'reviewable_results.html',(header+"<h1>Provisional assistant WHY judgments — all five saved models</h1><p>Unblinded results. Earlier 8B judgments are imported unchanged; new answers were reviewed before their model mapping was opened. Narrated-intent agreement and visual support are separate.</p>"+''.join(scored)+"</body></html>").encode())


def metric_table(mm):
    lines=['| Model | Correct | Partial | Incorrect | Unjudgeable | Correct / judgeable | Unsupported additions |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m,x in mm.items():
        n=x['label_counts']; rate=x['correct_fraction_judgeable']
        lines.append(f"| {NAMES[m]} | {n['correct']} | {n['partial']} | {n['incorrect']} | {n['unjudgeable']} | {n['correct']}/{x['judgeable']} ({rate:.2%}) | {x['unsupported_additional_claims']} |")
    return lines


def report():
    manifest,files,cases,judgments=load_checked()
    # Validation and blinded QA occur above this line. Mapping is read only AFTER a frozen hash record.
    freeze={'at_utc':datetime.now(timezone.utc).isoformat(),'complete_cases':228,'complete_new_answers':456,
            'rubric_sha256':manifest['rubric_sha256'],'extension_sha256':manifest['extension_sha256'],
            'cases_sha256':manifest['cases_sha256'],'identity_key_sha256':sha(KEY),
            'batch_sha256':{str(p.relative_to(OUT)):sha(p) for p in files},'before_aggregation_unblinding':True,
            'reviewer_model':'Codex GPT-6 assistant team; exact deployed build unavailable',
            'blinding':'Partial: reviewers share prior context; style can reveal identities. Reviewers did not consult model mapping.'}
    fp=OUT/'assistant/judgments_freeze.json'
    if fp.exists():
        previous=json.loads(fp.read_text());assert {k:v for k,v in previous.items() if k!='at_utc'}=={k:v for k,v in freeze.items() if k!='at_utc'}
    else:dump(fp,freeze)
    key=json.loads(KEY.read_text())['case_slots']
    predictions={}
    for m,run in RUNS.items():
        generation=run/'predictions/generation.jsonl'
        assert sha(generation)==manifest['generation_provenance'][m]['generation_sha256']
        predictions[m]={r['id']:r for r in rows(generation)}
    verify_mapping(cases,key,predictions)
    new=[];sources=[];concerns=[];proposals=[]
    for c in cases:
        j=judgments[c['case']];assert key[c['case']]['id']==c['id']
        sources.append({k:j[k] for k in ('case','id','reviewer','reference_status','purpose_core','source_note','visual_reviewed','visual_observation','inspection_pages')} |
                       {'narration_id':c['narration_id'],'action_question':c['action_question'],'original_narration':c['original_narration'],
                        'original_reference':c['reference_reason'],'original_char_start':c['reason_char_start'],'original_char_end':c['reason_char_end'],
                        'existing_fixed50':c['existing_fixed50'],'primary_source_policy':'unchanged from original assistant review'})
        for x in j.get('new_source_concerns',[]):concerns.append({'case':c['case'],'id':c['id'],'concern':x,'primary_reference_changed':False})
        for x in j.get('proposed_corrections',[]):proposals.append({'case':c['case'],'id':c['id'],'proposal':x,'applied':False})
        for s,a in j['answers'].items():
            ans=c['answers'][s]
            new.append({'case':c['case'],'id':c['id'],'model':key[c['case']]['slots'][s],
                        'blinded_slot':s,'existing_fixed50':c['existing_fixed50'],'narration_id':c['narration_id'],
                        'action_question':c['action_question'],'reference_reason':c['reference_reason'],
                        'answer_why':ans['why'],'extraction_status':ans['extraction_status'],
                        'reference_status':j['reference_status'],'purpose_core':j['purpose_core'],'reviewer':j['reviewer'],
                        'assessment_status':'provisional_assistant_assessment',**a})
    prior_path=OLD/'assistant/per_answer_judgments.jsonl'
    assert sha(prior_path)==manifest['prior_judgments_sha256']
    prior=rows(prior_path);assert len(prior)==684
    records=prior+new
    for m in MODELS:
        rr=[r for r in records if r['model']==m]
        assert len(rr)==len({r['id'] for r in rr})==228
        assert {r['id'] for r in rr}=={c['id'] for c in cases}
    put(OUT/'assistant/new_per_answer_judgments.jsonl',jl(new))
    put(OUT/'assistant/new_per_answer_judgments.csv',csv_data(new))
    put(OUT/'assistant/all_five_models.jsonl',jl(records))
    put(OUT/'assistant/all_five_models.csv',csv_data(records))
    put(OUT/'source_issue_decisions.jsonl',jl(sources));put(OUT/'source_issue_decisions.csv',csv_data(sources))
    put(OUT/'new_source_concerns.jsonl',jl(concerns));put(OUT/'proposed_source_corrections.jsonl',jl(proposals))
    fixed=[r for r in records if r['existing_fixed50']]
    put(OUT/'assistant/existing_fixed50.jsonl',jl(fixed))
    model_provenance={}
    old_config=OLD.parent/'frozen_config.json'
    put(OUT/'provenance/qwen3_8b_frozen_config.json',old_config.read_bytes())
    for m in ('base','epoch1','epoch2'):
        model_provenance[m]={'base_model':json.loads(old_config.read_text())['model'],
                             'saved_stage':m,'prior_judgments_sha256':manifest['prior_judgments_sha256']}
    for m,run in RUNS.items():
        active=json.loads((run/'active_configuration.json').read_text())
        config=run/active['path'];assert sha(config)==active['file_sha256']
        conf=json.loads(config.read_text())
        assert conf['configuration_sha256']==manifest['generation_provenance'][m]['configuration_sha256']
        put(OUT/f'provenance/{m}_active_frozen_config.json',config.read_bytes())
        model_provenance[m]={k:conf[k] for k in ('repo_id','revision','dtype','generation','individual_images',
                            'min_pixels','max_pixels','enable_thinking','configuration_sha256')}
        model_provenance[m].update(manifest['generation_provenance'][m])
    summary={'assessment_status':'provisional_assistant_assessment_pending_human_review',
             'model_provenance':model_provenance,
             'selected_checkpoint':'epoch1/checkpoint-909 (unchanged original Explain NLL criterion)',
             'all_validation_explain':metrics(records),'existing_fixed50':metrics(fixed),
             'by_reference_status':{s:metrics([r for r in records if r['reference_status']==s]) for s in sorted({r['reference_status'] for r in records})},
             'paired':{f'{a}_vs_{b}':paired(records,a,b) for a,b in PAIRS},
             'fixed50_paired':{f'{a}_vs_{b}':paired(fixed,a,b) for a,b in PAIRS},
             'source_status_counts':dict(Counter(s['reference_status'] for s in sources)),
             'visual_reviewed_cases':sum(s['visual_reviewed'] for s in sources),
             'ambiguous_sources':[s for s in sources if s['reference_status'] in ('ambiguous_attachment','source_conflict')],
             'extraction_failures':{m:[{'case':r['case'],'id':r['id'],'extraction_status':r['extraction_status'],'label':r['label']} for r in new if r['model']==m and not r['answer_why']] for m in RUNS},
             'new_source_concerns':concerns,'proposed_corrections':proposals,
             'old_answer_judgments_reused_unchanged':684,'new_substantive_answer_judgments':456,
             'semantic_scope':'WHY only; goal and future-action semantic accuracy remain unmeasured',
             'new_inference':False,'training':False,'test_data_access':False}
    dump(OUT/'comparison.json',summary)
    write_report(summary,sources,manifest)
    build_html(cases,judgments,records)
    dump(OUT/'preservation_audit.json',audit_preservation(cases))
    package_source()
    complete=OUT/'completed.json'
    if not complete.exists():dump(complete,{'at_utc':datetime.now(timezone.utc).isoformat(),'cases':228,'new_judgments':456,'combined_judgments':1140,
        'judgments_freeze_sha256':sha(fp),'comparison_sha256':sha(OUT/'comparison.json'),'human_review_complete':False})
    else:
        cc=json.loads(complete.read_text());assert cc['judgments_freeze_sha256']==sha(fp) and cc['comparison_sha256']==sha(OUT/'comparison.json')
    print(json.dumps({'completed':True,'metrics':summary['all_validation_explain'],'output':str(OUT)},indent=2))


def write_report(summary,sources,manifest):
    mm=summary['all_validation_explain'];table=metric_table(mm)
    text=['# Same-rubric WHY comparison — provisional assistant review','',
          'Completed 456 new substantive judgments for the two untuned models, using the same 228 validation Explain IDs, frozen rubric, supplementary WHY parser, source-policy denominator and 16 cached causal frames as the completed 8B evaluation. The earlier 684 8B judgments are imported unchanged. This is an assistant assessment of agreement with narrated purpose, pending human review.','',*table,'',
          'The primary denominator is 226 source-judgeable examples; all 228 remain in four-label totals. There are 204 clean sources, 22 with a separable nonpurpose suffix assessed on the previously frozen exact purpose core, and two ambiguous sources. Correct/all rates and clean-source-only results are in comparison.json.','']
    for m in ('qwen35','qwen38'):
        x=mm[m]
        for ref in ('base','epoch1'):
            delta=100*(x['correct_fraction_judgeable']-mm[ref]['correct_fraction_judgeable'])
            text.append(f"{NAMES[m]} differs from {NAMES[ref]} by {delta:+.2f} percentage points in correct/judgeable. ")
    text+=['','| Earlier → later model | Improvements | Regressions | Ties | Unjudgeable pairs |','|---|---:|---:|---:|---:|']
    for a,b in PAIRS:
        p=summary['paired'][f'{a}_vs_{b}']['counts']
        text.append(f"| {NAMES[a]} → {NAMES[b]} | {p['improvement']} | {p['regression']} | {p['tie']} | {p['unjudgeable_source']} |")
    text+=['','Pairing uses incorrect < partial < correct on the same record. Full 4×4 matrices and every improved/regressed/tied ID are saved in comparison.json. No significance or independence claim is made.','',
           '## Existing fixed 50-example subset','',*metric_table(summary['existing_fixed50']),'',
           'This is the pre-existing subset, not a new sample or independent replicate. The full comparison is the primary descriptive result.','',
           '## Extraction and source limitations','',
           f"Independent WHY extraction fails for {len(summary['extraction_failures']['qwen35'])}/228 Qwen3.5 answers and {len(summary['extraction_failures']['qwen38'])}/228 Qwen3.8 answers. Under the unchanged rubric, missing/unextractable WHY is incorrect on a judgeable source; no manual parser recovery was used. Whole-response format validity is not a semantic gate. Raw answers and extraction statuses remain in blinded/cases.jsonl.",'']
    for s in summary['ambiguous_sources']:text.append(f"- {s['case']} / {s['narration_id']}: {s['source_note']}")
    text+=['',f"All {summary['visual_reviewed_cases']}/228 cases received renewed image inspection. Four-case sheets contain all 16 causal frames per case at reduced display size; full sheets may be consulted for detail. Display resolution limits fine detail. Visual support flags assess observable instrumental relations and are independent of agreement with narrated intent; a plausible visual explanation is not necessarily the stated purpose.",'',
           f"New source concerns: {len(summary['new_source_concerns'])}; new proposed source corrections: {len(summary['proposed_corrections'])}. These are saved separately and unapplied. Primary source decisions and original reason spans/offsets remain unchanged. The inherited source proposals are preserved at {OLD/'proposed_source_corrections.csv'}.",'',
           'The 228 records have 228 distinct narration IDs but 222 distinct narration texts, with three repeated-text groups documented in shared_source_narrations.json. All examples are from validation participant P07. Repeated contexts, one participant and one correlated assistant team limit generalization.','',
           '## Review method and boundaries','',
           'Codex GPT-6 assistants performed manual semantic reading with per-answer evidence notes, original narration, action question, completed history and causal input frames. No keyword-overlap metric, external inference API, local model judge, answer regeneration or test access was used. The exact deployed reviewer build is unavailable. Fresh within-case A/B order hid model identities, but shared prior conversation and recognizable answer styles limit blinding. New judgments were hashed before the identity key was opened for aggregation. Older ratings were not shown as answer labels to the new reviewers.','',
           'Unsupported additions are separately flagged and can accompany a correct central purpose. Different verbosity creates different opportunities for such flags; their counts are not validated hallucination rates. Human judgment fields stay separate in blinded/human_review.csv. Neither these assistant ratings nor the previous ones are human ground truth or a validated GRPO reward.','',
           '**Goal and future-action semantic correctness are not measured by this WHY-only review.** The large-model runs cover Explain only, so no Track comparison is implied. Output tags and lexical matches are not substitutes for semantic correctness.','',
           'Epoch 1 (checkpoint-909) stays selected under the original Explain NLL criterion. No model training, GRPO, new predictions, reference rewrite or checkpoint selection change occurred.','',
           'Review all answers and evidence in reviewable_results.html (unblinded), or use blinded/index.html and the separate blinded archive for human review. The private model-identity key remains outside both archives. The full result archive contains model identities in the comparison and judgments.']
    put(OUT/'comparison_report.md',('\n'.join(text)+'\n').encode())
    put(OUT/'README.md',b'''# Completed same-rubric WHY comparison

Read comparison_report.md and comparison.json for results. All scores are provisional assistant assessments, not human ground truth. The selected checkpoint remains checkpoint-909.

assistant/new_per_answer_judgments.csv contains 456 new ratings and evidence; assistant/all_five_models.csv includes the unchanged 684 prior ratings. Source issues and proposed edits are separate and unapplied. Human fields are blank in blinded/human_review.csv and preserved on rerun.

Open reviewable_results.html for all five saved models, or blinded/index.html for identity-hidden new-model answers. Cached contact sheets are review aids; models received 16 individual images. Neither references nor source narrations were supplied to the answering models.

Safe offline commands:
    bash scripts/run_hdepic_why_semantic_comparison_v2.sh prepare
    bash scripts/run_hdepic_why_semantic_comparison_v2.sh check
    bash scripts/run_hdepic_why_semantic_comparison_v2.sh report
    bash scripts/run_hdepic_why_semantic_comparison_v2.sh package

Preparation reuses frozen randomization and preserved frames. Check validates completed manual judgments without unblinding. Report refuses incomplete/changed frozen judgments, then aggregates without inference. Package reuses existing results. No stage trains, generates answers or accesses the test split. Source files and relevant tests are included under source/.
''')


def package_source():
    selected=list((PROJECT/'code/hdepic_why_semantic_comparison').glob('*.py'))
    selected += [PROJECT/'code/hdepic_review_followup/fields.py', PROJECT/'scripts/run_hdepic_why_semantic_comparison_v2.sh', PROJECT/'code/tests/test_hdepic_why_semantic_comparison.py']
    for p in selected:put(OUT/'source'/p.relative_to(PROJECT),p.read_bytes())
    dump(OUT/'source/code_hashes.json',{str(p.relative_to(PROJECT)):sha(p) for p in selected})


def package():
    assert (OUT/'completed.json').is_file(), 'Report is not completed'
    results={}
    for blinded in (False,True):
        suffix='.blinded.zip' if blinded else '.zip'
        archive=OUT.with_name(OUT.name+suffix)
        files=sorted(p for p in (OUT/'blinded' if blinded else OUT).rglob('*') if p.is_file())
        if blinded:files+= [OUT/'rubric.md',OUT/'extension_protocol.md']
        if not archive.exists():
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in files:z.write(p,Path(OUT.name)/p.relative_to(OUT))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert set(z.namelist())=={str(Path(OUT.name)/p.relative_to(OUT)) for p in files}
            import hashlib
            for p in files:assert hashlib.sha256(z.read(str(Path(OUT.name)/p.relative_to(OUT)))).hexdigest()==sha(p)
        results['blinded' if blinded else 'full']={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}
    results.update(identity_key_included=False,full_archive_contains_unblinded_results=True,blinded_archive_contains_mapping=False)
    dump(OUT.with_name(OUT.name+'.package.json'),results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('check','report','package'))
    stage=parser.parse_args().stage
    if stage=='check':
        _,_,cases,_=load_checked();print(json.dumps({'passed':True,'blinded_cases':len(cases),'identity_key_opened':False}))
    elif stage=='report':report()
    else:package()
