"""Validate completed manual assistant judgments, then unblind and aggregate."""
from __future__ import annotations
import csv, html, io, json, shutil, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .prepare import PROJECT,RUN,OUT,KEY,dump,lines,rows
from code.hdepic_review_followup.visual import sha256

LABELS=('correct','partial','incorrect','unjudgeable')
SOURCE_STATUSES=('valid','valid_with_nonpurpose_suffix','ambiguous_attachment','source_conflict')
SUPPORTS=('supported','contradicted','not_established','unreviewed')
HUMAN_FIELDS=('human_reference_validity','human_correct_target_attachment','human_answer_correctness','human_unsupported_claims','human_visual_support','human_reviewer_notes')

def csv_write(path,records,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in records for k in r));handle=io.StringIO();writer=csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    for row in records:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    path.write_text(handle.getvalue())

def validate(cases,judgments):
    assert len(cases)==len(judgments)==228
    assert {c['case'] for c in cases}==set(judgments)
    for c in cases:
        j=judgments[c['case']];assert j['id']==c['id']
        assert j['reference_status'] in SOURCE_STATUSES
        for k in ('reviewer','source_note','visual_observation'):
            assert isinstance(j[k],str) and len(j[k])>=15,(c['case'],k)
        assert isinstance(j['visual_reviewed'],bool)
        assert j['inspection_pages'] or not j['visual_reviewed']
        for page in j['inspection_pages']:
            path=(OUT/page).resolve()
            assert path.is_relative_to(OUT) and path.is_file(),(c['case'],'missing inspection page')
        assert set(j['answers'])==set('ABC')
        ambiguous=j['reference_status'] in ('ambiguous_attachment','source_conflict')
        if not ambiguous:
            core=j['purpose_core'];assert core and core in c['original_narration'],(c['case'],'not exact purpose core',core)
        for slot,a in j['answers'].items():
            assert a['label'] in LABELS and a['visual_support'] in SUPPORTS
            assert isinstance(a['evidence_note'],str) and len(a['evidence_note'])>=15
            assert isinstance(a['unsupported_additional_claims'],list)
            assert (a['label']=='unjudgeable')==ambiguous,(c['case'],slot,'ambiguous source consistency')
            if not j['visual_reviewed']:assert a['visual_support']=='unreviewed'
        for proposal in j.get('proposed_corrections',[]):
            if proposal['type'] in ('trim','span_completion'):
                assert proposal['original_span']==c['reference_reason']
            assert c['original_narration'][proposal['original_char_start']:proposal['original_char_end']]==proposal['original_span']
            start,end=proposal.get('proposed_char_start'),proposal.get('proposed_char_end')
            if start is not None or end is not None:
                assert isinstance(start,int) and isinstance(end,int)
                assert c['original_narration'][start:end]==proposal['proposed_exact_span']

def metrics(records):
    result={}
    for model in ('base','epoch1','epoch2'):
        rr=[r for r in records if r['model']==model];n=len(rr)
        counts={v:sum(r['label']==v for r in rr) for v in LABELS}
        judgeable=n-counts['unjudgeable']
        result[model]={'denominator_all':n,'judgeable':judgeable,'label_counts':counts,
            'correct_fraction_all':counts['correct']/n if n else None,
            'correct_fraction_judgeable':counts['correct']/judgeable if judgeable else None,
            'unsupported_additional_claims':sum(bool(r['unsupported_additional_claims']) for r in rr),
            'visual_support_counts':{s:sum(r['visual_support']==s for r in rr) for s in SUPPORTS}}
    return result

def paired(records,before,after):
    byid=defaultdict(dict)
    for r in records:byid[r['id']][r['model']]=r
    matrix={x:{y:0 for y in LABELS} for x in LABELS};counts=Counter();examples=defaultdict(list)
    order={'incorrect':0,'partial':1,'correct':2}
    for rid,pair in byid.items():
        a,b=pair[before]['label'],pair[after]['label'];matrix[a][b]+=1
        if 'unjudgeable' in (a,b):kind='unjudgeable_source'
        else:kind='improvement' if order[b]>order[a] else 'regression' if order[b]<order[a] else 'tie'
        counts[kind]+=1;examples[kind].append({'id':rid,'case':pair[before]['case'],'from':a,'to':b})
    return {'before':before,'after':after,'denominator_all':len(byid),
        'judgeable_pairs':len(byid)-counts['unjudgeable_source'],
        'counts':{k:counts[k] for k in ('improvement','regression','tie','unjudgeable_source')},
        'transition_matrix':matrix,'examples':dict(examples),
        'unsupported_claim_flag_transitions':dict(Counter(
            f"{bool(pair[before]['unsupported_additional_claims'])}->{bool(pair[after]['unsupported_additional_claims'])}"
            for pair in byid.values()))}

def build_html(cases,judgments):
    blocks=[]
    for c in cases:
        e=lambda value:html.escape(str(value))
        source=f"<details><summary>Source and completed history</summary><p><b>Narration {e(c['narration_id'])}</b>: {e(c['original_narration'])}</p><p>Frozen reason [{c['reason_char_start']}:{c['reason_char_end']}]: {e(c['reference_reason'])}</p><p>History: {e('; '.join(h['action'] for h in c['history']))}</p><pre>{e(json.dumps(c['input_prompt'],ensure_ascii=False,indent=2))}</pre></details>"
        answers=''.join(f"<article><h3>Answer {slot}</h3><p>{e(a['why'])}</p><details><summary>Saved raw response</summary><pre>{e(a['raw_response'])}</pre></details></article>" for slot,a in c['answers'].items())
        blocks.append(f"<section id='{c['case']}'><h2>{c['case']} | {e(c['action_question'])}</h2><p>{e(c['id'])} | query {c['query_time_sec']:.6f}s | {'existing fixed 50 subset' if c['existing_fixed50'] else 'remaining validation case'}</p><img loading='lazy' src='{c['contact_sheet']}' alt='All 16 ordered causal frames for {c['case']}'><p>Review aid; model input was 16 separate JPEGs. Each frame is labeled with its source timestamp.</p>{source}<div class='answers'>{answers}</div><p>Human fields are blank in <a href='human_review.csv'>human_review.csv</a>. Assistant judgments are stored separately.</p></section>")
    page="""<!doctype html><html><head><meta charset='utf-8'><title>Provisional WHY review — blinded answers</title><style>body{font:16px system-ui;max-width:1700px;margin:auto;padding:20px;color:#172437}section{border-top:2px solid #ccd;padding:24px 0}img{max-width:100%;height:auto}.answers{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{background:#f3f5f7;padding:16px}details{margin:16px 0}</style></head><body><h1>WHY semantic review v1 — blinded material</h1><p>228 validation Explain cases, three preserved saved answers per case. Partial blinding: randomized A/B/C order hides checkpoint names, but answer style and prior assistant context can reveal identities. No test data or new inference. Human review is pending; all assistant scores are provisional.</p><p><a href='../rubric.md'>Frozen rubric</a> · <a href='human_review.csv'>Blank human review fields</a> · <a href='cases.jsonl'>Source evidence and saved answers</a></p>"""+''.join(blocks)+"</body></html>"
    # Use only relative contact/human CSV links; parent rubric link separately checked below.
    (OUT/'blinded/index.html').write_text(page)

def preservation():
    frozen=json.loads((OUT/'protected_originals.json').read_text())
    changes=[p for p,h in frozen.items() if not Path(p).is_file() or sha256(p)!=h]
    stats=json.loads((OUT/'checkpoint_stat_snapshot.json').read_text())
    stat_changes=[r['path'] for r in stats if not Path(r['path']).is_file() or
                  Path(r['path']).stat().st_size!=r['size'] or Path(r['path']).stat().st_mtime_ns!=r['mtime_ns']]
    assert not changes and not stat_changes,(changes,stat_changes)
    frame_changes=[];seen=set()
    for c in rows(OUT/'blinded/cases.jsonl'):
        for f in c['frames']:
            if f['path'] in seen:continue
            seen.add(f['path'])
            if sha256(f['path'])!=f['sha256']:frame_changes.append(f['path'])
    assert not frame_changes,frame_changes
    return {'passed':True,'original_files_sha256_checked':len(frozen),'checkpoint_files_size_mtime_checked':len(stats),
            'unique_causal_cached_frames_sha256_rechecked':len(seen),'changed_cached_frames':frame_changes,
            'changed_originals':changes,'changed_checkpoint_files':stat_changes,'test_split_accessed':False,
            'no_original_predictions_references_metrics_or_checkpoints_modified':True}

def report():
    manifest=json.loads((OUT/'materials_manifest.json').read_text())
    assert sha256(OUT/'rubric.md')==manifest['rubric_sha256']
    assert sha256(OUT/'blinded/cases.jsonl')==manifest['cases_sha256']
    files=[OUT/'assistant'/f'batch{i}.jsonl' for i in range(1,4)]
    if not all(p.exists() for p in files):raise RuntimeError('Substantive assistant review not completed: '+str([str(p) for p in files if not p.exists()]))
    cases=rows(OUT/'blinded/cases.jsonl');jj=[j for p in files for j in rows(p)]
    judgments={j['case']:j for j in jj};assert len(jj)==len(judgments)
    validate(cases,judgments)
    # Freeze complete blinded judgments BEFORE opening separate identity key.
    freeze={'at_utc':datetime.now(timezone.utc).isoformat(),'complete_cases':228,'complete_answers':684,
            'rubric_sha256':manifest['rubric_sha256'],'blinded_cases_sha256':manifest['cases_sha256'],
            'identity_key_sha256':sha256(KEY),
            'batch_sha256':{str(p.relative_to(OUT)):sha256(p) for p in files},
            'before_aggregation_unblinding':True,'reviewer_model':'Codex GPT-6 assistant team (exact deployed build unavailable)',
            'blinding':'Partial; prior experiment context and identifiable answer style. No identity key read by batch reviewers.'}
    freeze_path=OUT/'assistant/judgments_freeze.json'
    if freeze_path.exists():
        prev=json.loads(freeze_path.read_text())
        assert prev['batch_sha256']==freeze['batch_sha256'],'Do not overwrite frozen judgments'
        assert prev['identity_key_sha256']==freeze['identity_key_sha256'],'Identity key changed after review'
    else:dump(freeze_path,freeze)
    key=json.loads(KEY.read_text())['case_slots']
    records=[];sources=[];corrections=[];human=[]
    for c in cases:
        j=judgments[c['case']];assert key[c['case']]['id']==c['id']
        sources.append({k:j[k] for k in ('case','id','reviewer','reference_status','purpose_core','source_note','visual_reviewed','visual_observation','inspection_pages')} | {'narration_id':c['narration_id'],'original_narration':c['original_narration'],'original_reference':c['reference_reason'],'original_char_start':c['reason_char_start'],'original_char_end':c['reason_char_end'],'action_question':c['action_question'],'query_time_sec':c['query_time_sec'],'existing_fixed50':c['existing_fixed50']})
        for p in j.get('proposed_corrections',[]):
            corrections.append({'case':c['case'],'id':c['id'],'narration_id':c['narration_id'],'status':'assistant_proposal_pending_human_review','original_reference_retained':True,**p})
        for slot,a in j['answers'].items():
            records.append({'case':c['case'],'id':c['id'],'model':key[c['case']]['slots'][slot],
                'blinded_slot':slot,'existing_fixed50':c['existing_fixed50'],'narration_id':c['narration_id'],
                'action_question':c['action_question'],'reference_reason':c['reference_reason'],
                'answer_why':c['answers'][slot]['why'],'reference_status':j['reference_status'],
                'purpose_core':j['purpose_core'],'reviewer':j['reviewer'],'assessment_status':'provisional_assistant_assessment',**a})
            human.append({'case':c['case'],'id':c['id'],'slot':slot,'action_question':c['action_question'],
                'original_narration':c['original_narration'],'frozen_reference_reason':c['reference_reason'],
                'answer_why':c['answers'][slot]['why'],'contact_sheet':c['contact_sheet'],
                **{k:'' for k in HUMAN_FIELDS}})
    lines(OUT/'assistant/per_answer_judgments.jsonl',records);csv_write(OUT/'assistant/per_answer_judgments.csv',records)
    lines(OUT/'assistant/existing_fixed50_judgments.jsonl',[r for r in records if r['existing_fixed50']])
    dump(OUT/'existing_fixed50_ids.json',[c['id'] for c in cases if c['existing_fixed50']])
    lines(OUT/'source_issue_decisions.jsonl',sources);csv_write(OUT/'source_issue_decisions.csv',sources)
    lines(OUT/'proposed_source_corrections.jsonl',corrections)
    csv_write(OUT/'proposed_source_corrections.csv',corrections) if corrections else (OUT/'proposed_source_corrections.csv').write_text('case,id,note\n')
    human_path=OUT/'blinded/human_review.csv'
    if human_path.exists():
        old=list(csv.DictReader(human_path.open()))
        assert len(old)==684
        # Preserve every human-entered field on rerun; initial rows have no assistant scores.
        bykey={(r['case'],r['slot']):r for r in old}
        for row in human:
            for k in HUMAN_FIELDS:row[k]=bykey[(row['case'],row['slot'])][k]
    csv_write(human_path,human)
    subset=[r for r in records if r['existing_fixed50']]
    narr=defaultdict(list)
    for c in cases:narr[c['original_narration']].append({'case':c['case'],'id':c['id'],'narration_id':c['narration_id'],'action_question':c['action_question']})
    duplicates=[{'original_narration':text,'instances':group} for text,group in narr.items() if len(group)>1]
    dump(OUT/'shared_source_narrations.json',{'record_ids':228,'unique_narration_ids':len({c['narration_id'] for c in cases}),
        'unique_original_narration_texts':len(narr),'shared_text_groups':duplicates,
        'limitation':'Separate narration IDs can repeat identical source text. Cases are not independent observations.'})
    summary={'assessment_status':'provisional_assistant_assessment_pending_human_review','checkpoint_selection':'epoch1/checkpoint-909 retained under original Explain NLL criterion',
        'all_validation_explain':metrics(records),'existing_fixed50':metrics(subset),
        'source_status_counts':dict(Counter(s['reference_status'] for s in sources)),
        'subset_source_status_counts':dict(Counter(s['reference_status'] for s in sources if s['existing_fixed50'])),
        'by_reference_status':{s:metrics([r for r in records if r['reference_status']==s]) for s in SOURCE_STATUSES},
        'paired':{f'{a}_vs_{b}':paired(records,a,b) for a,b in (('base','epoch1'),('base','epoch2'),('epoch1','epoch2'))},
        'fixed50_paired':{'base_vs_epoch1':paired(subset,'base','epoch1')},
        'ambiguous_source_cases':[s for s in sources if s['reference_status'] in ('ambiguous_attachment','source_conflict')],
        'visual_reviewed_cases':sum(s['visual_reviewed'] for s in sources),
        'shared_source_narration_text_groups':len(duplicates),'distinct_source_narration_texts':len(narr),
        'scoring_method':'Manual substantive source/answer semantic assessment under frozen rubric; no automatic lexical/similarity judge',
        'original_reference_changes':0,'original_answer_changes':0,'new_model_inference':0,'training_launched':False,'test_split_accessed':False,
        'limitations':['Human reason-label review pending','Single correlated Codex GPT-6 assistant team','Partial blinding; prior context and answer styles','Narrated intent agreement is not visual inferability','Single validation participant P07','No validated GRPO reward or human semantic ground truth']}
    dump(OUT/'comparison.json',summary)
    text=['# WHY semantic review v1','',
        'Completed 684 substantive assistant answer judgments for all 228 validation Explain IDs, using saved WHY fields independently of whole-response formatting. All semantic results below are **provisional assistant assessments**, pending human reason-label and answer review. Epoch 1 (`checkpoint-909`) remains selected under the original Explain mean per-example full assistant-target NLL criterion; this review does not change selection.','',
        'No training, answer regeneration, model inference, GRPO, or test-split access occurred. Original predictions, references, metrics and checkpoints are preserved.','',
        f"Source statuses: {dict(summary['source_status_counts'])}. All {summary['visual_reviewed_cases']}/228 cases have assistant visual inspection of all 16 causal cached frames. Compact pages contain every frame at 192 pixels per frame, with some tool displays resizing them to about 176 pixels per frame. Full contact sheets containing the 384-pixel cached frames were opened for fine detail where needed; tool display can also resize these. Fine detail remains limited, and observations assess visible instrumental relations, not proof of narrated intent.",'',
        '| Saved model | Correct | Partial | Incorrect | Unjudgeable | Judgeable denominator | Correct / judgeable | Correct / all 228 | Extra unsupported claims |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,x in summary['all_validation_explain'].items():
        z=x['label_counts'];text.append(f"| {m} | {z['correct']} | {z['partial']} | {z['incorrect']} | {z['unjudgeable']} | {x['judgeable']} | {x['correct_fraction_judgeable']:.2%} | {x['correct_fraction_all']:.2%} | {x['unsupported_additional_claims']} |")
    text += ['',
        'Base and epoch 1 each receive 72 correct judgments. Epoch 1 has fewer partial and more incorrect judgments; this assistant assessment does not demonstrate improved specific-purpose agreement despite improved original Explain NLL. Unsupported additional claims are flagged in 189 base answers versus 28 epoch-1 answers; answer length and verbosity create different opportunities for such claims, so this difference is not a validated hallucination-rate estimate. The original base whole-response format count of 0/2,209 did not imply zero semantic WHY content: all 228 saved WHY fields are independently extractable.','',
        'The 226 judgeable denominator includes 204 clean source references and 22 reliably attached purpose cores with separable nonpurpose suffixes. For those 22, judgments assess the explicitly identified purpose core under the pre-scoring rubric, while retaining the full frozen reference untouched. `comparison.json` also breaks results down by source status, so clean-reference-only results can be inspected separately. Two source-deficient cases are unjudgeable for every model and remain in the full denominator.']
    p=summary['paired']['base_vs_epoch1'];text += ['',f"Primary paired comparison, epoch 1 versus base: **{p['counts']['improvement']} improvements, {p['counts']['regression']} regressions, {p['counts']['tie']} ties** among {p['judgeable_pairs']} judgeable pairs; {p['counts']['unjudgeable_source']} ambiguous-source pairs remain in the full 228-case denominator. Ordering is incorrect < partial < correct. Full matrices and case IDs for every transition are in `comparison.json`.",'',
        '| Base → epoch 1 | Correct | Partial | Incorrect | Unjudgeable |','|---|---:|---:|---:|---:|']
    for a in LABELS:text.append('| '+a+' | '+' | '.join(str(p['transition_matrix'][a][b]) for b in LABELS)+' |')
    text += ['','The original fixed 50-example subset is reported separately below. It is included in the 228 cases and is not independent evidence.','',
        '| Model, fixed 50 subset | Correct | Partial | Incorrect | Unjudgeable | Judgeable denominator |','|---|---:|---:|---:|---:|---:|']
    for m,x in summary['existing_fixed50'].items():
        z=x['label_counts'];text.append(f"| {m} | {z['correct']} | {z['partial']} | {z['incorrect']} | {z['unjudgeable']} | {x['judgeable']} |")
    sp=summary['fixed50_paired']['base_vs_epoch1'];text += ['',f"Fixed-50 epoch1 versus base paired counts: {sp['counts']}.",'',
        f"There are 228 distinct source narration IDs and {len(narr)} distinct original narration texts, with {len(duplicates)} repeated-text groups. `shared_source_narrations.json` lists every group and associated record ID. Repeated text, related actions and one participant make independent-observation interpretations inappropriate.",'',
        'Source ambiguity and source corrections are kept separate from answer correctness. `source_issue_decisions.csv` records a decision, exact frozen reference, action question and visual observation for every case. `proposed_source_corrections.csv` contains assistant proposals only; no corrected reference was inserted into preserved exports or original scoring.','']
    for s in summary['ambiguous_source_cases']:text += [f"- {s['case']} `{s['narration_id']}` — {s['source_note']}"]
    text += ['','The reviewers were Codex GPT-6 assistants in one correlated team; the exact deployed model build is unavailable. The written rubric and randomized A/B/C materials were frozen before scoring. Reviewers did not open the separate identity key. Previous conversation context and answer styles limit blinding; these are not independent human annotations. Answer-identity mapping was opened by aggregation only after the complete judgment files were hashed.','',
        'Visual support is reported independently in `comparison.json`. A correct answer can still have visual support `not_established`: source narration supplies intent that five seconds of video do not establish. Generic plausible cooking purposes receive no automatic credit. Unsupported ancillary assertions are flagged separately; central contradictions are incorrect.','',
        'Open `blinded/index.html` for all 228 reviewable examples with all 16 source frames, original narration, frozen reason, action question, history and saved A/B/C answers. `blinded/human_review.csv` retains blank human fields; completed assistant judgments and evidence are separate under `assistant/`. The explicit identity key stays outside the archives. The full results archive contains unblinded per-model judgments; use the separate blinded-material archive for human review without per-answer model mappings. Semantic agreement with narration is not validated semantic accuracy, proof of observational inferability, or a validated GRPO reward.']
    (OUT/'comparison_report.md').write_text('\n'.join(text)+'\n')
    build_html(cases,judgments)
    # HTML paths are relative. Contacts, CSV and rubric are checked without accessing any test file.
    assert all((OUT/'blinded'/c['contact_sheet']).is_file() for c in cases)
    dump(OUT/'preservation_audit.json',preservation())
    src=OUT/'source/code/hdepic_why_semantic_review';src.mkdir(parents=True,exist_ok=True)
    for p in (PROJECT/'code/hdepic_why_semantic_review').glob('*.py'):shutil.copyfile(p,src/p.name)
    parser_dest=OUT/'source/code/hdepic_review_followup';parser_dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PROJECT/'code/hdepic_review_followup/fields.py',parser_dest/'fields.py')
    shutil.copyfile(PROJECT/'code/hdepic_review_followup/visual.py',parser_dest/'visual.py')
    metric_dest=OUT/'source/code/hdepic_experiment';metric_dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PROJECT/'code/hdepic_experiment/metrics.py',metric_dest/'metrics.py')
    for rel in ('scripts/run_hdepic_why_semantic_review_v1.sh','code/tests/test_hdepic_why_semantic_review.py'):
        dest=OUT/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/rel,dest)
    (OUT/'README.md').write_text('''# Completed provisional assistant WHY review

Read comparison_report.md for results, rubric.md for the pre-scoring criteria, assistant/per_answer_judgments.csv for all 684 decisions/evidence, and source_issue_decisions.csv for all 228 source judgments. Proposed source changes are unapplied in proposed_source_corrections.csv. The full results archive exposes model identities in analysis; the separate why_semantic_review_v1.blinded.zip includes only blinded review materials and rubric. The explicit identity key is outside both archives. Human fields in blinded/human_review.csv remain blank and separate.

Open blinded/index.html for every case. Each contact sheet contains the 16 ordered causal cached frames; contact sheets are presentation aids, not training inputs. Original individual JPEG paths, SHA256 and timestamps remain in blinded/cases.jsonl. No model weights or full frame cache are packaged.

Safe offline rerun from the existing project:
```
bash /data/yaxuanli/ActGround/scripts/run_hdepic_why_semantic_review_v1.sh prepare
bash /data/yaxuanli/ActGround/scripts/run_hdepic_why_semantic_review_v1.sh report
```
Preparation reuses the frozen rubric/order and cached sheets. Reporting reuses completed manual judgments, verifies frozen hashes, preserves human-entered fields and reaggregates without new inference or training. It fails rather than filling missing judgments or modifying frozen judgments. Epoch1/checkpoint-909 selection and original metrics are unchanged. Test data are never loaded by this workflow.
''')
    dump(OUT/'completed.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'cases':228,'answer_judgments':684,
        'assessment_status':'provisional_assistant_assessment','selected_checkpoint':'checkpoint-909','human_review_complete':False,
        'rubric_sha256':manifest['rubric_sha256'],'frozen_judgments_sha256':sha256(freeze_path)})
    dump(OUT/'source/code_hashes.json',{str(p.relative_to(OUT/'source')):sha256(p)
        for p in (OUT/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.sh')})
    archive=RUN/'why_semantic_review_v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipout:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():zipout.write(p,Path(OUT.name)/p.relative_to(OUT))
    blind_archive=RUN/'why_semantic_review_v1.blinded.zip'
    with zipfile.ZipFile(blind_archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipout:
        for p in sorted((OUT/'blinded').rglob('*')):
            if p.is_file():zipout.write(p,Path(OUT.name)/p.relative_to(OUT))
        zipout.write(OUT/'rubric.md',Path(OUT.name)/'rubric.md')
    dump(RUN/'why_semantic_review_v1.package.json',{'zip':str(archive),'sha256':sha256(archive),'bytes':archive.stat().st_size,
        'full_archive_contains_unblinded_results':True,'blinded_zip':str(blind_archive),'blinded_sha256':sha256(blind_archive),
        'identity_key_included':False,'blinded_archive_contains_model_mapping':False})
    print(json.dumps({'completed':True,'all_validation_explain':summary['all_validation_explain'],
        'paired_epoch1_vs_base':summary['paired']['base_vs_epoch1']['counts'],'zip':str(archive)},indent=2))

if __name__=='__main__':report()
