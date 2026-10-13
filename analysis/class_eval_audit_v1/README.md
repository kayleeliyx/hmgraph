**The original Claude headline scores reproduce, but recovering omitted base action text changes the improvement estimate.** This audit reads saved predictions only; it does not load a model, train, or replace original reports or checkpoint selection.

| Qwen3-VL-8B metric | Base | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Current-action verb–noun agreement, 1,981 Track records | 372/1,981 (18.78%) | 486/1,981 (24.53%) | 478/1,981 (24.13%) |
| First-next-action verb–noun agreement, all 2,209 validation records | 224/2,209 (10.14%) | 278/2,209 (12.58%) | 276/2,209 (12.49%) |
| First-next-action agreement, 228 Explain records only | 28/228 (12.28%) | 47/228 (20.61%) | 46/228 (20.18%) |

Base → epoch 1: current action +5.75 percentage points; first future action +2.44 points across all validation records. These are heuristic category-agreement diagnostics, not validated semantic accuracy or proof of improved visual understanding.

**Untuned larger models, the same 228 Explain records:** Qwen3.5-35B-A3B first-next-action agreement is 27/228 (11.84%); Qwen3.8-27B is 31/228 (13.60%). They contain no Track predictions. Goal meaning is not evaluated by this audit. The separate existing assistant WHY review remains 91/226 (40.27%) and 101/226 (44.69%), respectively.

**What changed in the audit:**

- Recover one unclosed current-action field up to the next field boundary. This recovers all 931 omitted base action texts without consulting reference answers.
- Read all repeated NEXT blocks, explicit nested action/time children, and clear newline, numbered, timestamped, and literal-list representations. Comma lists require three action-bearing parts under the fixed mapper. Missing positions remain misses; no references guide extraction.
- Use the frozen scoring sidecar's canonical main-action class IDs. Missing gold classes are excluded only from the affected component denominator; unknown predictions remain failures. One third-future-action noun is missing, so third-position noun/pair and complete-triple denominators are 2,208, while first-position pair remains 2,209.
- Fit phrase aliases on the 102 initial-SFT videos plus official class definitions. No held-out narration-derived aliases or reserved second-stage videos fit this mapper. Its parsing heuristics are retained from Claude. The corrected current/first-next headline scores match the all-participant-mapper variant.
- Preserve original source, reproduced scores, and row-level evidence. The full supplied JSON differs only in one auxiliary oracle-bigram set-recall rounding (.085 supplied, .084 locally); every supplied base/epoch1/epoch2 model score reproduces.

The free-text mapper is unvalidated on generated paraphrases. Related examples come from nine P07 videos; the original independent-row bootstrap intervals do not account for video clustering. This audit does not create new WHY judgments or measure goal meaning.

Files: [audit_summary.json](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/audit_summary.json), [audit source](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/audit_class_scores.py), [original scorer](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/score_hdepic_classes.original.py), [corrected per-example results](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/recovered_fields_with_initial_sft_and_official_mapper.per_example.jsonl).

Focused checks cover open tags, repeated blocks, nested fields, timestamped/numbered/literal/comma lists, ambiguous fields, and missing class labels. All five saved prediction sets are checked against their expected frozen validation IDs. Main evaluation code was not overwritten; this separate scorer makes the correction reproducible.

Rerun:

```bash
/data/yaxuanli/ActGround/.venvs/llavaction/bin/python -I \
  /data/yaxuanli/ActGround/analysis/class_eval_audit_v1/audit_class_scores.py \
  --project /data/yaxuanli/ActGround \
  --annotations /data/kailaic/HDEPIC/data/hd-epic-annotations \
  --claude-scorer /data/yaxuanli/ActGround/analysis/class_eval_audit_v1/score_hdepic_classes.original.py \
  --supplied-summary /data/yaxuanli/ActGround/analysis/class_eval_audit_v1/class_level_summary.supplied.json \
  --output /data/yaxuanli/ActGround/analysis/class_eval_audit_v1
```
