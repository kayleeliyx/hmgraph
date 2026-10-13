# What the evaluation measures

Evaluate each model on the **same frozen references**, with **Explain and Track reported separately**. Keep the four capabilities separate: there is no combined “overall accuracy.” The main scores are:

| Capability | Reference and decision | Main metric |
| --- | --- | --- |
| **WHY** — Explain | Specific purpose explicitly stated in the narration. Equivalent wording is accepted; a plausible invented reason is not. | Fully correct answers / eligible WHY references. |
| **GOAL** — Explain and Track | Activity annotation active at the query time, at its original scope. Equivalent wording is accepted. Essential tasks in a compound annotation must be covered. | Fully correct answers / eligible GOAL references. |
| **Current action** — Track | Generated action mapped to the canonical official verb and noun classes. | Correct verb–noun pairs / references with both classes. |
| **NEXT** — Explain and Track | The first three fine-grained actions starting strictly after the query, in chronological order. One sequence, not three alternative guesses. | Verb–noun accuracy at positions **1, 2, 3**, plus their arithmetic mean. |

WHY is the immediate purpose; GOAL can be a step, broader activity, or compound activity. It is not automatically the entire session's goal. NEXT evaluates the annotated verb and object, rather than a high-level recipe stage.

## Every additional score

**WHY and GOAL:**

- **Correct / partial / incorrect counts.** Partial means a genuine part is captured but a material detail or activity scope is missing. A central contradiction is incorrect.
- **Supplementary partial-credit score:** `(2 × correct + partial) / (2 × eligible)`. This is a defined ordinal summary, not a probability of correctness.
- **Flag counts:** unsupported addition, central contradiction, too broad, too narrow. Track how many judgments actually recorded each flag; an omitted flag is not a measured negative. An unsupported ancillary claim can accompany a correct core. These counts are diagnostics, not validated hallucination rates.
- **Judgment completion:** completed decisions / eligible examples, plus pending, missing-answer, and source-exclusion counts. A pending judgment is not an incorrect answer. Until all eligible answers are resolved, the final correctness and partial-credit scores remain **pending**, not zero.

**Current action:**

- **Verb accuracy:** correct verb IDs / references with a verb ID.
- **Noun accuracy:** correct noun IDs / references with a noun ID.
- **Verb–noun accuracy:** both IDs correct on the same example / references with both IDs.

**NEXT, separately for verbs, nouns, and verb–noun pairs:**

- **Position accuracy:** correct class at the exact future position / eligible references at that position.
- **Mean position accuracy:** arithmetic mean of the three position rates, rather than pooling their denominators.
- **Complete ordered sequence accuracy:** all three positions correct / references with all three required class labels.
- **Normalized Levenshtein distance ↓:** insertions, deletions, and substitutions each cost one. Report the mean distance divided by three.
- **Normalized restricted Damerau / OSA distance ↓:** the same edits, also allowing an adjacent swap for cost one. Again divided by three. Zero means an exact sequence match.

The edit scores evaluate **one three-action forecast**. They are not INSIGHT's best-of-five, twenty-action benchmark score.

## Denominators and diagnostic counts

Missing, failed, unextractable, or unmapped predictions retain their denominator and count as errors. Missing gold verb/noun labels remove only the affected component. Complete-sequence and edit metrics require all three gold labels for the scored component. WHY source exclusions are fixed across models; missing or ambiguous GOAL annotations are excluded rather than treated as “no goal.”

Also record mapped-prediction counts; short and extra NEXT forecasts; missing/extra positions; repeated NEXT blocks; explicit empty slots; and absent/failed generations. Only the first three predicted actions are scored; short forecasts preserve their positions and are padded with unknowns. Reports include task and video breakdowns, plus shared-cohort before/after comparisons.

**Current limitation:** semantic judgments and free-text-to-class mapping remain provisional. GOAL rules and judge packets are implemented; unresolved GOAL judgments have no final score. Human calibration and visualization are deferred.
