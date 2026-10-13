# Evaluation and next training: proposed protocol

This is a research proposal, not an implemented evaluation release or new experiment result. It is grounded in the frozen HD-EPIC exports, the repaired class-score audit, and INSIGHT v2. No training, model inference, or test predictions were run for this proposal.

## What the targets mean

- **WHY:** the explicit actor-stated reason attached to the queried action, extracted from narration. It is immediate purpose, not any plausible explanation. Source attachment and purpose-core validity still require human review.
- **GOAL:** the uniquely annotated activity interval containing the query time (`start <= t < end`), copied from `high-level/activities/*_recipe_timestamps.csv`. It is not generated from WHY or by an LLM. The field is better described as **active annotated activity**: labels vary from broad activities to phases, near-action steps, and compound descriptions. They do not form a consistent session-goal hierarchy. Missing or overlapping activity labels omit the target; missing annotation does not establish absence of a real goal.
- **NEXT:** the first three annotated fine-grained main verb/object actions whose starts are strictly after the query, in start/end/narration-ID order, within the same video. These are sequential positions, not three alternatives. An action already started before the query is excluded even if still ongoing. Three micro-actions are not a fixed duration or a long-term forecasting benchmark.
- **Current action:** the single ongoing action on Track queries; its gold name is withheld from the input. Explain instead supplies the action being explained.

One actual validation example at 34.83 seconds: action `move mug`; WHY `so that it fits in the cupboard`; GOAL `Unloading the dishwasher`; NEXT `pick up pair of tongs` → `open drawer` → `put tongs`. Other activity labels include `Preparing coffee`, `Line parchment paper on baking trays`, and compound mushroom/onion/rice preparation. Recipe IDs are separate metadata and are not the current GOAL target.

## Metric contract

Report the components separately. Whole-response syntax and target NLL are not capability scores.

| Component | Primary measurement | Secondary diagnostics |
| --- | --- | --- |
| WHY, Explain | Fully correct narrated-purpose answers / common source-valid records | Partial, incorrect, contradictions, unsupported additions; input observability assessed separately |
| GOAL, both tasks separately | Fully correct active-activity answers / common eligible records | Partial; wrong activity; overly broad or narrow scope; compound-label stratum |
| Current action, Track | Official verb–noun pair accuracy | Verb and noun accuracy; per-class recall; extraction and mapping coverage |
| NEXT, each task separately | Pair accuracy at positions 1, 2, 3; equal-weight mean of those three rates | Verb/noun accuracy by position; exact ordered triple; normalized class-sequence edit distance |

For WHY and GOAL, freeze a correct/partial/incorrect rubric. Partial earns no full-correct credit; report its rate explicitly. A supplementary score `(2 * correct + partial) / (2 * eligible)` is an ordinal summary with an explicit equal-spacing assumption, not a probability of correctness. Source validity is decided without model answers. Exclude invalid sources equally for every model; missing predictions remain failures. Record numerator, denominator and source exclusions for every metric.

For GOAL, create a source-grounded reference sheet for each distinct activity annotation: scope, essential activity/outcome, material phase details, and treatment of compound descriptions. Fix those requirements before judging outputs. For `Unloading the dishwasher`, `put clean dishes away from the dishwasher` is equivalent; `household chores` is too broad; `load dirty dishes` is wrong. For a detailed risotto-phase target, whether `prepare risotto` receives partial credit must be predetermined. Preserve the original annotation and version any interpreted reference. A uniform hierarchy or full-session intention requires new labels and a separate benchmark version.

Use frozen canonical main-action class IDs as gold. Validate the prediction-to-class mapper on generated answers across model families and styles, including unmapped and ambiguous phrases. Mapping may use the prediction, fixed taxonomy and permitted causal context, but must not select a class by consulting the gold answer. Official class definitions and training-derived aliases are legitimate; held-out narration-derived alias fitting is excluded. Unknown predictions remain misses, with mapping coverage reported to distinguish evaluation failures. Missing gold components exclude only that component. Repeated actions can be valid and receive no blanket penalty.

For the current three-action benchmark, positional accuracy is primary: it exposes first-step errors and later degradation. Edit distance is secondary because alignment can obscure position errors. If implemented, define the algorithm explicitly, map to class tokens, retain the first three predictions, pad missing positions with an unknown token and divide distance by three. Report missing and extra positions. Adding a 20-action horizon or best-of-five sampling creates a separate evaluation setting; re-score all compared models under that setting.

## Establishing reliability and comparing experiments

1. Human-check WHY attachment and GOAL reference scope independently of model outputs. Calibrate a fixed semantic judge against blinded human ratings spanning videos, model families, terse/verbose answers, partial answers and contradictions. Measure judge disagreements and model-specific bias; expand/adjudicate the audit when needed. Existing assistant ratings remain provisional until this calibration exists. A single similarity threshold is insufficient: loading and unloading can have similar vocabulary and opposite meanings.
2. Freeze IDs, source-validity masks, prompts, frame paths, history, decoding budget, parser, mapper and judge configuration. Save per-example extraction, classes, semantic decisions and concise justification. Version every change and re-score all archived model outputs affected by it.
3. Compare each fine-tuned checkpoint with its **own untuned backbone** on identical inputs and one fixed deployment prediction. The 8B-versus-27B comparison cannot isolate fine-tuning. Report Explain and Track separately because the supplied current-action information differs.
4. Report paired improvements/regressions, per-video results and paired uncertainty estimated by resampling videos rather than individual examples. P07 validation contains nine videos from one participant; its intervals cannot establish population-wide generalization. Preserve P08/P09 for final confirmation and report participants separately: test WHY examples are 11 P08 and 1,306 P09.
5. Use history-only and simple training-derived forecasting baselines to investigate whether gains depend on images. Existing histories contain annotated completed actions; current scores do not isolate visual reasoning. Inspect performance near activity transitions and by goal-label scope as diagnostics.
6. Continue using validation for development; use sealed test results for final confirmation after choices are frozen. Check historical evaluation exposure before calling any new test cohort fully untouched. Repeated tuning on a test set turns it into development data.

## What to take from INSIGHT

INSIGHT uses GRPO and explicit perception → intention → forecast generation, together with a separate hand–object-aware action-recognition stage. Its Ego4D setting observes eight action segments and predicts twenty verb–noun actions; evaluation takes the best of five sequences using normalized edit distance. Its intention is a high-level task description pseudo-labelled with GPT-4.1, scored through Sentence-BERT similarity. It provides no standalone validated immediate-WHY or activity-correctness metric. Forecast accuracy dominates the task reward (weight 0.85); intention similarity has weight 0.05. [Paper, pp. 4–5 and appendix](https://arxiv.org/html/2508.01742v2).

The released reward/parser handles two alphabetic words per action, and the released scorer compares action strings. Adapt them to HD-EPIC classes and multiword objects before use. The supplied documentation does not establish the causal inputs used to generate its pseudo-intentions. [Reward code](https://github.com/iLearn-Lab/AAAI26-INSIGHT/blob/main/CognitiveReasoning/my_rewards_intention.py), [evaluation code](https://github.com/iLearn-Lab/AAAI26-INSIGHT/blob/main/CognitiveReasoning/cal_ED.py).

## Recommended experiment order

1. Finish same-backbone untuned-versus-SFT evaluation of Qwen3.8-27B with the reliable field metrics above.
2. Test the intention hypothesis under matched causal inputs: direct NEXT versus predicted GOAL → NEXT. Add annotated/oracle GOAL → NEXT as a privileged-information diagnostic of whether the label could help; exclude it from fair deployment comparisons. Initially retain the three-action horizon.
3. Compare the same selected SFT checkpoint under **no additional training**, **matched extra SFT**, and **GRPO**, using the same reserved training pool and transparent data/compute budgets. The planned 146 Explain examples are a small pilot pool; they are not the whole 2,470-row second-stage split. Expanding to Track requires an explicitly matched design.
4. Begin with reliable NEXT class rewards; add GOAL/WHY rewards only after semantic and source calibration. Keep rewards task-aware and report every component: no missing GOAL annotation is a negative GOAL label. Do not let a formatting gate zero otherwise scoreable content. Measure reward spread within sampled groups; uninformative rewards provide little learning signal.
5. Evaluate on held-out examples with the frozen suite and independent human checks. Exact class metrics may legitimately be shared between training reward and evaluation, provided inputs are held out and mapping is validated. Optimizing a semantic judge and using its score alone as proof of improvement is insufficient; independent review is necessary to detect shortcuts.

Only after goal conditioning demonstrates forecasting benefit should the project expand the horizon or integrate predicted action histories and longer memory. Those changes should be separately ablated rather than bundled with GRPO.

Local evidence: [target construction](/data/yaxuanli/ActGround/code/hdepic_preparation/build.py:202), [repaired class-score audit](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/README.md), [existing WHY rubric](/data/yaxuanli/ActGround/runs/hdepic_why_semantic_comparison_v2/rubric.md), [source activities](/data/kailaic/HDEPIC/data/hd-epic-annotations/high-level/activities/P07_recipe_timestamps.csv).
