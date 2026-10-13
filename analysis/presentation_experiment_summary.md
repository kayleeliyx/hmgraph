**Goal:** Build a local model that uses past egocentric observations to identify the current action, explain its immediate purpose, identify the activity, and predict the next three actions. E0 tests background knowledge; E1 tests access to video history; SFT tests adaptation of Qwen3-VL-8B.

**E0 tested whether background knowledge improves WHY recognition.** The experiment used 100 participant-balanced HD-EPIC questions across nine participants, five choices per question, and `azure/gpt-5.6-sol`. Knowledge generation and retrieval used only the target action and general kitchen context, without answer choices or gold labels.

| Condition | Correct / evaluated | Accuracy |
| --- | ---: | ---: |
| Question + choices | 26/100 | 26.0% |
| + annotation-selected action-clip frames | 83/100 | 83.0% |
| + self-generated kitchen facts | 84/100 | 84.0% |
| + original Wikipedia retrieval | 68/91 | 74.7% on answered subset |
| + repaired Wikipedia retrieval | 72/87 | 82.8% on eligible subset |

Visual evidence produced the largest gain: +57 percentage points. Self-generated facts added one net correct answer: four corrections and three regressions. Original retrieval had nine missing answers and caused nine regressions versus one correction on its answered subset. The repair improved source selection and explicitly abstained on 13 items. On the same 87 eligible items, repaired retrieval and clip-only inference both scored 72/87; using clip-only answers for abstentions yielded 83/100. This pilot shows no net benefit from repaired external facts.

Sources: [E0](/data/yaxuanli/ActGround/analysis/e0/hd_epic_why_pilot/summary.md), [retrieval repair](/data/yaxuanli/ActGround/analysis/e0/hd_epic_why_dev100_c3_v2/summary.md).

**E1 tested WHY recognition using only video available by the annotated action end.** It reused the same 100 questions. The model initially received 16 frames from the preceding five seconds, then could request up to three additional past-video intervals. The annotated action start and future video were withheld. GPT views used timestamped contact sheets; the local Qwen model used individual frames.

| Model | Annotation-selected clip | Initial five-second view | After optional history reads |
| --- | ---: | ---: | ---: |
| GPT-5.6-sol | 83% | 64% | 70% |
| Qwen3-VL-30B-A3B-Instruct, local | 46% | 47% | 48% |

GPT history reads corrected nine answers and damaged three: +6 percentage points. Local Qwen corrected one answer: +1 point. History helped within these runs, but GPT remained below its annotation-selected-clip reference. Clip selection and image presentation differ between E0 and E1; this gap does not isolate the effect of the start boundary. These are action-end pilots, not continuous or pre-action evaluation, and they do not test the graph pipeline.

Sources: [GPT E1](/data/yaxuanli/ActGround/runs/e1_online_why/hd_epic_why_dev100_action_end_contact_sheet_v1/report/summary.md), [local E1](/data/yaxuanli/ActGround/runs/e1_local_why/qwen3_vl_30b_a3b_vllm023_tp2_256_dev100_v19/report/summary.md).

**The SFT dataset turns HD-EPIC annotations into two causal input–target tasks.** Source data contains 59,454 action annotations from 156 videos, nine participants, and about 41.3 source-video hours. The repaired dataset contains 25,288 examples.

| Split | Participants | Explain | Track | Total |
| --- | --- | ---: | ---: | ---: |
| Initial SFT | P01–P06 | 1,403 | 13,140 | 14,543 |
| Reserved second stage | P01–P06; separate videos | 146 | 2,324 | 2,470 |
| Validation | P07 | 228 | 1,981 | 2,209 |
| Test | P08–P09 | 1,317 | 4,749 | 6,066 |

- Input: 16 timestamped individual frames from the preceding five seconds, plus up to eight completed annotated action descriptions. No future frames or purpose labels enter the input.
- Explain: given the target action at its annotated end, generate `<why>`, optional `<goal>`, and `<next>`.
- Track: sampled every five seconds with exactly one ongoing action; generate `<action>`, optional `<goal>`, and `<next>`. The current-action label is withheld from the input.
- Targets: WHY comes from explicit narration reason spans; GOAL from activity annotations; NEXT contains three actions starting strictly after the query.
- Example: `move bowls` → WHY: `in order to close the drawer`; NEXT: close dishwasher drawer → open second drawer → pick up plate.

Explain is only 9.65% of initial-SFT examples. Six uncertain Explain labels were quarantined. Human reason-label review remains pending. The reserved 146 Explain records support a later matched extra-SFT/GRPO comparison and were excluded from initial SFT.

Sources: [dataset protocol and counts](/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/README.md), [final question repair](/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/README.md), [actual sample](/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/sample/training20.jsonl).

**Qwen3-VL-8B SFT improved current-action class agreement and modestly improved first-action forecasting, without demonstrating stronger WHY correctness.** Training completed two epochs over all 14,543 initial-SFT examples using text LoRA rank 16, frozen vision/aligner, assistant-only loss, and effective batch 16.

| Metric | Base | Epoch 1, selected | Epoch 2 |
| --- | ---: | ---: | ---: |
| Explain validation target NLL, lower is better | 3.315 | 2.156 | 2.308 |
| Track validation target NLL, lower is better | 3.415 | 2.211 | 2.396 |
| Current-action verb–noun agreement | 372/1,981 (18.78%) | 486/1,981 (24.53%) | 478/1,981 (24.13%) |
| First-next-action verb–noun agreement, all validation | 224/2,209 (10.14%) | 278/2,209 (12.58%) | 276/2,209 (12.49%) |
| Provisional semantic WHY correct | 72/226 (31.86%) | 72/226 (31.86%) | 74/226 (32.74%) |

Epoch 1 was selected by lower Explain validation NLL. That loss covers the full assistant target—WHY, GOAL when present, and NEXT—not WHY alone. The class scores above use the repaired offline audit of saved answers, rather than whole-response syntax or exact wording as a gate.

The subsequent assistant review assessed all 228 validation Explain answers independently of whole-response format; two ambiguous references were unjudgeable. Base → epoch 1 produced 47 improvements, 55 regressions, and 124 ties. Answers flagged for unsupported additions fell from 189 to 28, but verbosity affects this count; it is not a validated hallucination rate.

The class-agreement gains are +5.75 percentage points for current action and +2.44 points for the first future action. The free-text-to-class mapper is heuristic and has not been independently validated on generated paraphrases. These scores do not isolate visual understanding from action history. WHY ratings remain provisional assistant judgments; validation uses P07 only, and the test set has not been evaluated. Goal meaning remains unevaluated.

**Claude scorer audit:** The uploaded scorer reproduces its original model scores. Its 9.5% base current-action score discarded 931 answers lacking closing tags. Recovering their unambiguous action text raises base agreement to 18.78%; epoch 1 stays at 24.53%. The audit also reads all NEXT blocks and clear alternative list representations, uses frozen sidecar class IDs, and builds phrase mappings from initial-SFT videos plus the official vocabulary. Removing held-out narration-derived aliases did not change these corrected headline scores. Original predictions, reports, and checkpoint selection are preserved. Source and complete diagnostics: [class audit](/data/yaxuanli/ActGround/analysis/class_eval_audit_v1/README.md).

**Untuned larger-model comparison, shared Explain cohort:** Both models used the exact same 228 P07 question/history/image inputs as the 8B Explain evaluation. WHY uses the existing meaning-based assistant review (226 judgeable sources); first-next-action agreement uses the new common class audit (all 228 examples).

| Model | Provisional semantic WHY correct | First-next-action verb–noun agreement, Explain only |
| --- | ---: | ---: |
| Qwen3-VL-8B base | 72/226 (31.86%) | 28/228 (12.28%) |
| Qwen3-VL-8B SFT epoch 1 | 72/226 (31.86%) | 47/228 (20.61%) |
| Qwen3-VL-8B SFT epoch 2 | 74/226 (32.74%) | 46/228 (20.18%) |
| Untuned Qwen3.5-35B-A3B | 91/226 (40.27%) | 27/228 (11.84%) |
| Untuned Qwen3.8-27B | 101/226 (44.69%) | 31/228 (13.60%) |

The larger models perform better on this narrated-purpose review, while the trained 8B has higher first-step class agreement on this Explain cohort. The two columns use different evaluation methods and denominators. The larger runs contain no Track predictions; their current-action recognition cannot be compared here. None of these results measures goal meaning.

Sources: [validation](/data/yaxuanli/ActGround/runs/hdepic_sft_8b_v1/validation_report.md), [semantic WHY review](/data/yaxuanli/ActGround/runs/hdepic_sft_8b_v1/why_semantic_review_v1/comparison_report.md), [parser follow-up](/data/yaxuanli/ActGround/runs/hdepic_sft_8b_v1/review_followup_v1/supplementary_report.md).
