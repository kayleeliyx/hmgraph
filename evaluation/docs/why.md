# WHY semantic evaluation v1

Origin: Codex's frozen narrated-purpose review in `runs/hdepic_why_semantic_comparison_v2/rubric.md`. Claude's uploaded `score_hdepic_classes.py` covers action classes, not WHY. This reusable version preserves the meaning criterion and separates source validity from model correctness. Existing ratings are assistant judgments; human validation is deferred.

The target is the actor's **specific stated purpose for the queried action**. An answer can be plausible yet fail to express that purpose. Equivalent wording is accepted.

| Label | Decision |
| --- | --- |
| Correct | Captures the specific purpose, including any material destination, avoidance condition, quantity, tool/manner choice or timing that explains the action. |
| Partial | Expresses a real part of that purpose but omits a material detail. |
| Incorrect | Gives a different reason, only restates the action, invents a generic explanation, contradicts the central purpose, or provides no scoreable answer. |

A central contradiction overrides a correct fragment. Unsupported ancillary claims are flagged separately; they do not automatically change a correct core to incorrect. Do not reward keyword overlap, verbosity or matching XML syntax.

Source ambiguity is a **shared eligibility decision**, not an answer label that a model can earn by abstaining. The existing validation review has 204 clean references, 22 references with a separately frozen purpose core, and two ambiguous sources excluded for every model. Preserve the original narration/span and the reviewed purpose core independently. A new parser may recover more text; reuse an old rating only when the extracted answer is identical.

Primary metric: `correct / eligible`. Report correct/partial/incorrect counts and missing predictions. Supplementary ordinal score: `(2 * correct + partial) / (2 * eligible)`, with an explicit equal-spacing convention. A nonempty answer awaiting a judgment keeps the final rate pending, rather than reporting accuracy on a judged subset.

The semantic judge receives the reference, queried action, frozen source context and the isolated answer. It does not know the model name. Narrated-purpose agreement does not establish visual inferability. This phase does not produce visual-support ratings or human-review artifacts.
