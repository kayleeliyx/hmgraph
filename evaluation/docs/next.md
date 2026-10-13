# Current action and NEXT evaluation v1

Current action is scored on Track. NEXT is the three fine-grained actions starting strictly after the query in the same video, ordered by annotated start/end/narration ID. Explain and Track are reported separately because Explain supplies the action being explained. A forecast is one ordered sequence, not three alternatives.

Use frozen **canonical main-action class IDs** as gold. Do not parse reference wording to reconstruct classes or substitute alternative `pair_class_ids`.

## Metrics

- Current action: verb accuracy, noun accuracy, joint verb–noun accuracy.
- NEXT: those three accuracies at positions 1, 2 and 3; equal-weight mean of the three position rates.
- Exact ordered sequence: all three positions correct, separately for verb/noun/pair; a stringent diagnostic.
- Normalized sequence edit distance: divide distance by three and average over gold-complete sequences. Both classic Levenshtein and restricted Damerau/optimal-string-alignment (OSA) are explicitly named. OSA counts an adjacent swap as one edit; Levenshtein counts it as two.

INSIGHT uses a transposition-aware sequence evaluator and the best of five forecasts of twenty actions on Ego4D. Its action anticipation metric motivates our OSA diagnostic; it does not measure GOAL correctness. Our single three-action class-based scores are a different benchmark setting. [Paper](https://arxiv.org/html/2508.01742v2), [official evaluator](https://github.com/iLearn-Lab/AAAI26-INSIGHT/blob/main/CognitiveReasoning/cal_ED.py).

## Extraction and denominator rules

1. Parse explicit fields independently. Recover clear unclosed fields and repeated NEXT blocks; accept XML, labeled fields, JSON/literal lists, numbered lists, timestamps and clear separators. No reference is consulted to choose an interpretation.
2. Preserve missing slots: `take bowl;;put bowl` means `[take bowl, unknown, put bowl]`. Unknown phrases stay in their position. No dropping, shifting, deduplication or reference-guided reordering.
3. Retain all intended query IDs. Missing or failed model outputs count as failures. An intended Explain-only run does not create current-action measurements.
4. Use the first three predictions; pad absent positions with unknown. Count excess/missing/empty slots diagnostically. Repeated actions can be valid.
5. Missing gold excludes only the affected component. For existing validation, the third future noun is unavailable on one record: third-position noun/pair and full noun/pair sequence denominators are 2,208; verb denominators remain 2,209.

## Prediction-to-class mapper

The reusable mapper retains Claude's morphology and noun-head heuristics, fits alias frequencies only on the 102 initial-SFT videos, and adds the official class definitions. It repairs missing-ID handling, negated actions and valid verbs suppressed as stopwords. It never sees a prediction's gold answer. Persist the mapper and its training/source hashes so future experiments use the same version.

This is still a heuristic category mapping. Unknowns count as misses and mapping coverage is reported. Validation on generated paraphrases remains for the later checking phase; these scores alone do not establish improved visual reasoning. Any future mapper change requires re-scoring every compared model.
