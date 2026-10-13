# HD-EPIC GOAL semantic evaluation v1

GOAL is the unique annotated activity active at the query timestamp. It is copied
from the activity interval; it is not extracted from WHY or generated as a new
session intention. Activity descriptions have heterogeneous scope: one step,
a larger activity, or a compound activity. Score against the annotation's scope.

## Source and profile contract

Create one source eligibility mask and reuse it for all compared models. A
missing/ambiguous activity or source conflict is excluded for every model. A
missing model answer on an eligible source is incorrect, not unjudgeable.

Retain the exact original label and activity annotation IDs. A profile records
literal required activity units, scope, and provenance. The automatic helper
splits clear semicolons/sentences and preserves conjunctions within units; it
does not create a hierarchy, infer session goals, or approve semantic coverage.
Compound, broad and conjoined descriptions carry `manual_scope_review_pending`.
Every automatically generated profile remains provisional. Before using a
compound profile as a future fixed benchmark, its essential scope must be agreed
without inspecting model answers. Human verification is deferred at this stage.

## Answer labels

- **Correct:** identifies all essential activity units at the agreed scope.
  Equivalent wording and direct entailment are accepted. Incidental objects and
  procedural details are not required unless they define an essential unit.
  Independently named essential tasks in a compound goal must all be covered.
- **Partial:** identifies a true subset of a compound activity, or expresses a
  genuine part of the activity but loses important specificity or scope.
- **Incorrect:** wrong activity, central contradiction, generic activity-free
  wording, or only a micro-action that does not express the annotated activity.
  A central contradiction overrides a correct fragment. Missing answers fail.

Unsupported ancillary additions are flagged separately. They can coexist with a
correct core; central contradictions cannot. Sentence similarity, keyword overlap
and output format are not semantic correctness metrics.
Do not reward a menu of mutually exclusive alternative goals whose union happens
to cover the reference. Judge the activity asserted by the answer.

## Examples fixed independently of model output

| Original activity | Prediction | Decision and reason |
|---|---|---|
| Unloading the dishwasher | Put clean dishes away from the dishwasher | Correct: same activity |
| Unloading the dishwasher | Put things away | Partial: true component, loses the specific activity |
| Unloading the dishwasher | Household chores | Incorrect: no actual activity content |
| Unloading the dishwasher | Load dirty dishes into the dishwasher | Incorrect: opposite activity |
| Finish preparing coffee | Finish making coffee | Correct: equivalent wording |
| Finish preparing coffee | Pick up a mug | Incorrect: micro-action alone does not identify the activity |
| Cook mushrooms with vegetable stock. In another saucepan, fry onions and garlic. Measure risotto rice. | Cook the mushrooms with stock | Partial: only one of three literal required activity units |
| Cook mushrooms with vegetable stock. In another saucepan, fry onions and garlic. Measure risotto rice. | Make risotto | Partial provisionally: broad genuine task, omits the annotated compound scope; profile requires scope review |

The examples are assistant-authored rubric illustrations, not human-approved
ground truth. Compound annotations that cannot support a stable essential-unit
decision should be marked source-unjudgeable in a shared future source revision,
with exclusions reported; do not choose different scopes for different models.

## Metric and completion rules

Primary: correct / source-eligible examples. Report correct, partial, incorrect,
source exclusions, missing predictions and pending judgments. Supplementary
ordinal score: `(2 * correct + partial) / (2 * eligible)`. This weighted score is
a convention, not a probability or an independently validated accuracy measure.

Pending judgments are not failures. If a present prediction has no judgment,
the complete score is `null` and the report status is `pending`; missing answers
are deterministic failures. Compare models only when source-mask hashes agree.
Report Explain and Track separately and do not combine GOAL with WHY or NEXT
into a single capability claim. Judge ratings remain provisional until calibrated.

The accompanying WHY rubric preserves the existing Codex semantic review of
actor-stated narrated purpose. Claude's uploaded `score_hdepic_classes.py` scores
action classes; it did not supply the WHY semantic rubric.
