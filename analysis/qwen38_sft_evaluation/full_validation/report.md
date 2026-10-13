# Qwen3.8-27B: complete validation evaluation

**Generated 2026-10-11 20:07 EDT · 228/228 answers per model · Semantic review provisional**

For base → epoch 1, first-next-action joint accuracy improves 13.6% → 22.4%. Specific WHY agreement is 43.4% → 33.2%; complete GOAL agreement is 0.9% → 3.5%. Epoch 1 remains selected by the lowest validation loss. All three models are evaluated on the same 228 completed Explain queries. WHY and GOAL are provisional assistant judgments of reference agreement.

This full-cohort report supersedes the earlier 47-example preliminary comparison. The original report and its judgments are preserved; identical packet IDs and prompt hashes reuse the same decisions. The cohort has changed, so its percentages should not be compared as though they measured different models on identical samples.

## Evaluation coverage

Generated-answer evaluation: 228 Explain queries from nine validation videos, all participant P07. WHY has 226 eligible sources, with two source-quality exclusions fixed before model comparison. GOAL and NEXT use all 228. The input prompts, image preprocessing, greedy generation settings, and native model identity match across all models. No unfinished generation is counted as an error.

## Generated-answer accuracy

| Metric | Base | Epoch 1 | Epoch 2 | Epoch 1 − base | Epoch 2 − base |
| --- | --- | --- | --- | --- | --- |
| WHY: specific narrated purpose | 98/226 (43.4%) | 75/226 (33.2%) | 81/226 (35.8%) | -10.2 pp | -7.5 pp |
| GOAL: complete annotated activity | 2/228 (0.9%) | 8/228 (3.5%) | 11/228 (4.8%) | +2.6 pp | +3.9 pp |
| NEXT 1: verb + noun | 31/228 (13.6%) | 51/228 (22.4%) | 49/228 (21.5%) | +8.8 pp | +7.9 pp |
| NEXT 2: verb + noun | 13/228 (5.7%) | 17/228 (7.5%) | 12/228 (5.3%) | +1.8 pp | -0.4 pp |
| NEXT 3: verb + noun | 13/228 (5.7%) | 10/228 (4.4%) | 9/228 (3.9%) | -1.3 pp | -1.8 pp |
| NEXT: all positions pooled | 57/684 (8.3%) | 78/684 (11.4%) | 70/684 (10.2%) | +3.1 pp | +1.9 pp |
| NEXT: all three actions correct | 2/228 (0.9%) | 2/228 (0.9%) | 1/228 (0.4%) | +0.0 pp | -0.4 pp |

Changes are percentage points. All models use exactly the same examples and source-eligibility masks.

## Semantic decisions

| Field | Model | Eligible | Correct | Partial | Incorrect | Partial credit |
| --- | --- | --- | --- | --- | --- | --- |
| WHY | Base | 226 | 98 | 50 | 78 | 54.4% |
| WHY | Epoch 1 | 226 | 75 | 54 | 97 | 45.1% |
| WHY | Epoch 2 | 226 | 81 | 52 | 93 | 47.3% |
| GOAL | Base | 228 | 2 | 43 | 183 | 10.3% |
| GOAL | Epoch 1 | 228 | 8 | 92 | 128 | 23.7% |
| GOAL | Epoch 2 | 228 | 11 | 84 | 133 | 23.2% |

Correct means agreement with the specific narrated purpose (WHY) or all essential units of the annotated active activity (GOAL). Partial means a real component is present but a material detail or scope is missing. Incorrect includes another purpose/activity, central contradiction, or an unextractable answer from a completed generation. Partial-credit score = (2 × correct + partial) / (2 × eligible); strict correctness is the primary score. Assistant review uses identical frozen rubrics and judge packets that omit model names, with fixed source eligibility and hash-bound judgments. Orchestration and file paths can reveal model context, so this is not a fully blinded study. The judgments do not establish visual inferability or human-validated accuracy.

211/228 GOAL profiles remain marked for manual scope review. Activity labels vary in granularity, and some combine tasks. Strict GOAL correctness requires their essential activity units; naming only a relevant micro-action is not automatically correct. Keep complete and partial counts separate and calibrate compound scope with human review before treating this as a definitive intention metric.

### GOAL scope diagnostic

| Source profile | N | Base correct | Epoch 1 correct | Epoch 2 correct | Base partial credit | Epoch 1 partial credit | Epoch 2 partial credit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scope review flagged | 211 | 0/211 (0.0%) | 3/211 (1.4%) | 5/211 (2.4%) | 9.2% | 22.5% | 21.1% |
| Scope review not flagged | 17 | 2/17 (11.8%) | 5/17 (29.4%) | 6/17 (35.3%) | 23.5% | 38.2% | 50.0% |

Diagnostic only: these subsets use the original reference-side automatic scope flag, unchanged across all models. An unflagged profile is not human-validated and need not be a better reference. The main GOAL denominator remains all 228 examples. Differences between these subsets can reflect activity and video composition as well as annotation scope.

## NEXT components

| Position | Component | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| Position 1 | verb | 58/228 (25.4%) | 81/228 (35.5%) | 76/228 (33.3%) |
| Position 1 | noun | 83/228 (36.4%) | 96/228 (42.1%) | 92/228 (40.4%) |
| Position 1 | verb + noun | 31/228 (13.6%) | 51/228 (22.4%) | 49/228 (21.5%) |
| Position 2 | verb | 30/228 (13.2%) | 42/228 (18.4%) | 30/228 (13.2%) |
| Position 2 | noun | 54/228 (23.7%) | 54/228 (23.7%) | 55/228 (24.1%) |
| Position 2 | verb + noun | 13/228 (5.7%) | 17/228 (7.5%) | 12/228 (5.3%) |
| Position 3 | verb | 32/228 (14.0%) | 41/228 (18.0%) | 33/228 (14.5%) |
| Position 3 | noun | 45/228 (19.7%) | 51/228 (22.4%) | 45/228 (19.7%) |
| Position 3 | verb + noun | 13/228 (5.7%) | 10/228 (4.4%) | 9/228 (3.9%) |
| Pooled positions | verb | 120/684 (17.5%) | 164/684 (24.0%) | 139/684 (20.3%) |
| Pooled positions | noun | 182/684 (26.6%) | 201/684 (29.4%) | 192/684 (28.1%) |
| Pooled positions | verb + noun | 57/684 (8.3%) | 78/684 (11.4%) | 70/684 (10.2%) |

NEXT predicts the next three fine-grained annotated actions. Each position is mapped to official verb and noun classes by the frozen train-only heuristic mapper. Joint accuracy requires both classes at the same position. Missing slots are errors; repeated reference actions remain separate positions. The three-position pooled score has 684 decisions. The mapper still needs human checks on generated paraphrases, so class accuracy is an operational benchmark rather than a perfect measure of natural-language action correctness.

## Complete sequences and edit distances

| Metric | Component | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| Entire ordered sequence correct ↑ | verb | 4/228 (1.8%) | 7/228 (3.1%) | 3/228 (1.3%) |
| Entire ordered sequence correct ↑ | noun | 17/228 (7.5%) | 17/228 (7.5%) | 11/228 (4.8%) |
| Entire ordered sequence correct ↑ | action | 2/228 (0.9%) | 2/228 (0.9%) | 1/228 (0.4%) |
| Normalized Levenshtein ↓ | verb | 0.8173 | 0.7485 | 0.7763 |
| Normalized Levenshtein ↓ | noun | 0.7339 | 0.7047 | 0.7193 |
| Normalized Levenshtein ↓ | action | 0.9167 | 0.8801 | 0.8889 |
| Normalized Damerau-OSA ↓ | verb | 0.8129 | 0.7412 | 0.7690 |
| Normalized Damerau-OSA ↓ | noun | 0.7295 | 0.7003 | 0.7120 |
| Normalized Damerau-OSA ↓ | action | 0.9167 | 0.8787 | 0.8874 |

Exact ordered sequence requires all three predictions to match. Levenshtein allows insertions, deletions and substitutions; Damerau-OSA additionally allows an adjacent transposition. Distances are normalized by the three-action horizon; lower is better. There is one greedy sequence per example, not best-of-K sampling. These definitions do not make scores directly comparable to papers using other horizons, inputs, datasets or sampling protocols.

## Paired gains and losses

| Compared with base | Field | N | Both correct | Gained | Lost | Neither correct |
| --- | --- | --- | --- | --- | --- | --- |
| Epoch 1 | WHY | 226 | 48 | 27 | 50 | 101 |
| Epoch 1 | GOAL | 228 | 2 | 6 | 0 | 220 |
| Epoch 1 | NEXT 1 | 228 | 17 | 34 | 14 | 163 |
| Epoch 1 | NEXT 2 | 228 | 6 | 11 | 7 | 204 |
| Epoch 1 | NEXT 3 | 228 | 4 | 6 | 9 | 209 |
| Epoch 1 | All NEXT | 228 | 1 | 1 | 1 | 225 |
| Epoch 2 | WHY | 226 | 49 | 32 | 49 | 96 |
| Epoch 2 | GOAL | 228 | 2 | 9 | 0 | 217 |
| Epoch 2 | NEXT 1 | 228 | 18 | 31 | 13 | 166 |
| Epoch 2 | NEXT 2 | 228 | 4 | 8 | 9 | 207 |
| Epoch 2 | NEXT 3 | 228 | 3 | 6 | 10 | 209 |
| Epoch 2 | All NEXT | 228 | 1 | 0 | 1 | 226 |

## Results by video

| Video | Metric | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| P07-20240529-102652 | WHY | 7/11 (63.6%) | 5/11 (45.5%) | 5/11 (45.5%) |
| P07-20240529-102652 | GOAL | 2/11 (18.2%) | 4/11 (36.4%) | 4/11 (36.4%) |
| P07-20240529-102652 | NEXT 1 | 1/11 (9.1%) | 1/11 (9.1%) | 1/11 (9.1%) |
| P07-20240529-131737 | WHY | 8/15 (53.3%) | 4/15 (26.7%) | 5/15 (33.3%) |
| P07-20240529-131737 | GOAL | 0/15 (0.0%) | 0/15 (0.0%) | 0/15 (0.0%) |
| P07-20240529-131737 | NEXT 1 | 3/15 (20.0%) | 5/15 (33.3%) | 4/15 (26.7%) |
| P07-20240529-134410 | WHY | 5/18 (27.8%) | 4/18 (22.2%) | 2/18 (11.1%) |
| P07-20240529-134410 | GOAL | 0/19 (0.0%) | 0/19 (0.0%) | 0/19 (0.0%) |
| P07-20240529-134410 | NEXT 1 | 4/19 (21.1%) | 3/19 (15.8%) | 4/19 (21.1%) |
| P07-20240529-191007 | WHY | 15/31 (48.4%) | 10/31 (32.3%) | 10/31 (32.3%) |
| P07-20240529-191007 | GOAL | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) |
| P07-20240529-191007 | NEXT 1 | 7/32 (21.9%) | 12/32 (37.5%) | 13/32 (40.6%) |
| P07-20240529-194518 | WHY | 15/39 (38.5%) | 14/39 (35.9%) | 18/39 (46.2%) |
| P07-20240529-194518 | GOAL | 0/39 (0.0%) | 0/39 (0.0%) | 1/39 (2.6%) |
| P07-20240529-194518 | NEXT 1 | 3/39 (7.7%) | 11/39 (28.2%) | 9/39 (23.1%) |
| P07-20240530-102543 | WHY | 16/38 (42.1%) | 10/38 (26.3%) | 15/38 (39.5%) |
| P07-20240530-102543 | GOAL | 0/38 (0.0%) | 1/38 (2.6%) | 1/38 (2.6%) |
| P07-20240530-102543 | NEXT 1 | 4/38 (10.5%) | 6/38 (15.8%) | 7/38 (18.4%) |
| P07-20240530-111344 | WHY | 1/2 (50.0%) | 1/2 (50.0%) | 1/2 (50.0%) |
| P07-20240530-111344 | GOAL | 0/2 (0.0%) | 1/2 (50.0%) | 1/2 (50.0%) |
| P07-20240530-111344 | NEXT 1 | 0/2 (0.0%) | 0/2 (0.0%) | 0/2 (0.0%) |
| P07-20240531-101357 | WHY | 13/35 (37.1%) | 10/35 (28.6%) | 11/35 (31.4%) |
| P07-20240531-101357 | GOAL | 0/35 (0.0%) | 0/35 (0.0%) | 1/35 (2.9%) |
| P07-20240531-101357 | NEXT 1 | 5/35 (14.3%) | 6/35 (17.1%) | 5/35 (14.3%) |
| P07-20240531-201703 | WHY | 18/37 (48.6%) | 17/37 (45.9%) | 14/37 (37.8%) |
| P07-20240531-201703 | GOAL | 0/37 (0.0%) | 2/37 (5.4%) | 3/37 (8.1%) |
| P07-20240531-201703 | NEXT 1 | 4/37 (10.8%) | 7/37 (18.9%) | 6/37 (16.2%) |

## Full teacher-forced validation loss

| Task · N | Base | Epoch 1 | Epoch 2 | Epoch 1 relative change | Epoch 2 relative change |
| --- | --- | --- | --- | --- | --- |
| Explain · 228 | 2.1507 | 1.8163 | 1.9981 | -15.55% | -7.09% |
| Track · 1,981 | 1.8999 | 1.8546 | 2.0591 | -2.38% | +8.38% |

NLL averages each example's loss over its complete reference answer, with correct preceding tokens supplied. Explain includes WHY, GOAL, NEXT and termination, so this is not WHY accuracy. Epoch 1 lowers full Explain loss by 15.55% versus base. Epoch 2 has higher validation loss than epoch 1 for both tasks while training loss keeps falling, consistent with overfitting. Compared with base, epoch 2 Explain loss is still 7.09% lower, while Track loss is 8.38% higher. Checkpoint selection was fixed to Explain validation NLL and was not changed after viewing generated accuracy.

## Concrete gains and regressions

The examples below are the first chronological gains/losses for WHY, GOAL and NEXT-1 against epoch 1, up to two per category. The HTML report and examples.json contain all 228 examples.

### P07-20240529-131737 · 145.02 s · fold over kitchen paper

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that it is on top of the chickpeas | [incorrect] To contain the chickpeas and create a neat, manageable bundle for transport or storage.<br><br>Attributes the fold to a transport or storage bundle rather than placing paper over the chickpeas for drying. | [correct] so that the chickpeas are covered by the kitchen paper<br><br>Covering the chickpeas with kitchen paper directly matches placing the folded paper on top of them. | [incorrect] so that the kitchen paper is underneath the chickpeas<br><br>Putting the paper underneath the chickpeas directly contradicts the frozen purpose of placing it on top. |
| GOAL | Bake the chickpeas and wash lettuce leaves | [incorrect] fold over kitchen paper<br><br>Folding kitchen paper is a micro-action, not an identification of baking chickpeas and washing lettuce. | [incorrect] Drain chickpeas<br><br>Draining chickpeas states a preparation step rather than the annotated baking and lettuce washing activity. | [partial] Prepare chickpeas<br><br>Identifies chickpea preparation but loses baking specificity and omits the independent lettuce-washing task. |
| NEXT | press on chickpeas → press on chickpeas → tap chickpeas | fold over kitchen paper → fold over kitchen paper → fold over kitchen paper<br><br>0/3 class pairs correct · × / × / × | pick up baking tray → pick up kitchen paper → place kitchen paper<br><br>0/3 class pairs correct · × / × / × | pick up baking tray → pick up baking tray → place baking tray<br><br>0/3 class pairs correct · × / × / × |

### P07-20240529-191007 · 453.04 s · press tare button

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | because it didn't work the first time | [incorrect] Pressing the tare button zeros the scale's reading to account for the weight of the small bowl, allowing for the accurate measurement of the ingredients to be added next. This is a standard procedure in cooking to ensure precise quantities.<br><br>Explaining the general tare function does not explain pressing again because the first attempt failed. | [correct] because the first time I pressed it didn't work<br><br>Explicitly states that the first button press failed. | [partial] so that the scale is at zero<br><br>Captures the intended zeroing outcome of taring, but omits repeating it because the first press failed. |
| GOAL | Cook mushrooms with vegetable stock. In another saucepan, fry onions and garlic. Measure risotto rice | [incorrect] press tare button<br><br>Pressing tare does not identify cooking mushrooms, frying aromatics, or measuring risotto rice. | [incorrect] Weight biscuits, prepare the chocolate sauce<br><br>Weighing biscuits and preparing chocolate sauce identifies different ingredient and cooking activities. | [incorrect] Prepare the dough balls as well as the mixture of meat with other ingredients<br><br>Dough balls and a meat mixture identify a different preparation from cooking mushrooms, frying onions and garlic, and measuring rice. |
| NEXT | pick up jar of garlic → open cutlery drawer → reach into cutlery drawer | press tare button → press tare button → press tare button<br><br>0/3 class pairs correct · × / × / × | pick up jar of cocoa → open jar of cocoa → pick up spoon<br><br>1/3 class pairs correct · ✓ / × / × | pick up jar of black mustard → open lid of jar of black mustard → place lid<br><br>1/3 class pairs correct · ✓ / × / × |

### P07-20240529-102652 · 173.82 s · stack pile of three bowls

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that they fit better in the cupboard | [correct] To organize and consolidate the dishware by placing the bowls on top of each other, creating a stable and space-efficient stack on the shelf.<br><br>A space-efficient stack on the cupboard shelf captures making the bowls fit better; stability is additional. | [partial] so that they can be put away in the cupboard<br><br>Identifies cupboard storage, but omits the improved fit achieved by the particular stacking arrangement. | [partial] so that they can be placed in the cupboard<br><br>Captures placing the bowls in the cupboard but omits stacking them to improve their fit and use of space. |
| GOAL | Unloading the dishwasher | [incorrect] stack pile of three bowls<br><br>Stacking three bowls is a micro-action without identifying dishwasher unloading or putting the clean dishes away. | [incorrect] Putting away food<br><br>Putting away food identifies food storage rather than unloading clean dishes. | [incorrect] Putting away food<br><br>Putting away food identifies food storage rather than unloading clean dishes. |
| NEXT | open top drawer → take cutlery → transfer spoon | pick up small wire brush → drop small wire brush → pick up small wire brush<br><br>0/3 class pairs correct · × / × / × | close cupboard → pick up two chopping boards → pick up chopping board<br><br>0/3 class pairs correct · × / × / × | close cupboard → pick up two chopping boards → pick up chopping board<br><br>0/3 class pairs correct · × / × / × |

### P07-20240529-102652 · 841.36 s · shake spoon

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that all the coffee goes into the portafilter | [correct] Shaking the spoon is a deliberate action to dislodge any coffee grounds clinging to the inside of the spoon, ensuring that the full measured dose is transferred into the portafilter for a consistent brew.<br><br>Explicitly captures dislodging residual grounds so the full dose reaches the portafilter; consistent brewing adds an unstated further rationale. | [partial] so that the coffee falls into the portafilter<br><br>Captures coffee transfer into the portafilter, but not clearing all remaining coffee from the spoon. | [correct] so that the coffee that is stuck to it falls into the portafilter<br><br>Captures dislodging the coffee still on the spoon into the portafilter. |
| GOAL | Preparing coffee | [correct] prepare coffee<br><br>Preparing coffee identifies the same activity as the source annotation. | [correct] Prepare Coffee<br><br>Preparing coffee identifies the same activity as the source annotation. | [correct] Prepare Coffee<br><br>Preparing coffee identifies the same activity as the source annotation. |
| NEXT | tap coffee → remove excess coffee → put dosing spoon | [1] put down spoon → [2] pick up portafilter → [3] put portafilter on coffee machine<br><br>0/3 class pairs correct · × / × / × | put down dosing spoon → pick up coffee box → close coffee box<br><br>0/3 class pairs correct · × / × / × | put down dosing spoon → pick up tamper → press coffee<br><br>0/3 class pairs correct · × / × / × |

### P07-20240529-102652 · 106.59 s · pull bottom shelf of dishwasher

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that it can be emptied | [partial] Accessing the lower rack to load or unload items.<br><br>Includes unloading as a possible reason, but the load-or-unload answer does not commit to the stated emptying purpose. | [partial] in order to take out the pan<br><br>Taking out a pan is one possible part of emptying the rack, but narrows the stated unloading purpose to an unmentioned item. | [partial] in order to get the pan that is on it<br><br>Removing a pan expresses unloading part of the shelf, but omits emptying the shelf as a whole and assumes a pan is present. |
| GOAL | Unloading the dishwasher | [incorrect] Load the dishwasher.<br><br>Loading the dishwasher is the opposite of unloading it. | [correct] Empty dishwasher returning items to cupboards and drawers<br><br>Emptying the dishwasher and returning its contents to storage directly expresses unloading the dishwasher. | [correct] Empty dishwasher returning items to cupboards and drawers<br><br>Emptying the dishwasher and returning its contents to storage directly expresses unloading the dishwasher. |
| NEXT | pick up plate → place plate → pick up another plate | place frying pan → place frying pan → place frying pan<br><br>0/3 class pairs correct · × / × / × | pick up pan → pick up pan → pick up pan<br><br>0/3 class pairs correct · × / × / × | pick up pan → place pan → pick up another pan<br><br>0/3 class pairs correct · × / × / × |

### P07-20240529-102652 · 130.54 s · move saucepan

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that there is room for the frying pan in the drawer | [correct] Repositioning the saucepan within the open drawer to create space and organize the cookware alongside the frying pan.<br><br>Identifies creating space in the drawer alongside the frying pan, matching the stated reason for moving the saucepan. | [correct] so that the frying pan can be placed in the drawer<br><br>Making it possible to place the frying pan in the drawer captures creating room for it. | [partial] so that it is easier to place the frying pan<br><br>Captures facilitating placement of the frying pan, but omits making space specifically inside the drawer. |
| GOAL | Unloading the dishwasher | [partial] Organize cookware in the drawer<br><br>Organizing cookware in a drawer captures a put-away component but omits the dishwasher origin and the broader unloading activity. | [correct] Unload dishwasher and put clean crockery away<br><br>Unloading the dishwasher and storing clean crockery directly identifies the annotated activity. | [correct] Unload dishwasher and put clean crockery away<br><br>Unloading the dishwasher and storing clean crockery directly identifies the annotated activity. |
| NEXT | place frying pan → close drawer → pick up two glasses | Close the drawer → Turn around → Walk to the sink<br><br>0/3 class pairs correct · × / × / × | place frying pan → pick up saucepan → place saucepan<br><br>1/3 class pairs correct · ✓ / × / × | place frying pan → pick up saucepan → place saucepan<br><br>1/3 class pairs correct · ✓ / × / × |

### P07-20240529-131737 · 141.15 s · tip colander

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that the chickpeas tip out onto the kitchen paper | [correct] To empty the chickpeas from the colander onto the prepared kitchen paper for drying.<br><br>Captures transferring chickpeas out of the colander onto kitchen paper; drying is an additional inferred purpose. | [correct] so that the chickpeas fall onto the kitchen paper<br><br>Directly captures chickpeas falling from the colander onto kitchen paper. | [correct] so that the chickpeas fall on the kitchen paper<br><br>Captures transferring the chickpeas from the colander onto the kitchen paper. |
| GOAL | Bake the chickpeas and wash lettuce leaves | [incorrect] dry chickpeas<br><br>Drying chickpeas is a preparation operation rather than an identification of baking chickpeas and washing lettuce. | [incorrect] Drain chickpeas<br><br>Draining chickpeas states a preparation step rather than the annotated baking and lettuce washing activity. | [partial] Prepare chickpeas<br><br>Preparing chickpeas is relevant but leaves the baking unspecified and omits washing lettuce. |
| NEXT | replace colander → pick up chickpeas → fold over kitchen paper | shake colander → put down colander → pick up colander of chickpeas<br><br>0/3 class pairs correct · × / × / × | put down colander → pick up colander → put down colander<br><br>1/3 class pairs correct · ✓ / × / × | put down now empty colander → pick up baking tray → put down baking tray<br><br>1/3 class pairs correct · ✓ / × / × |

### P07-20240529-102652 · 586.36 s · push kettle

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that the cloth in the right hand can clean the work surface | [correct] The action of pushing the kettle is likely performed to clear space on the work surface, making it easier to continue wiping down the area effectively. This ensures that the cleaning process can proceed without obstruction.<br><br>Identifies removing the kettle as an obstruction so wiping the work surface can continue. | [correct] so that I can wipe the surface underneath it<br><br>Identifies clearing the kettle so the work surface underneath can be wiped. | [correct] so that I can clean the surface underneath it<br><br>Captures moving the kettle to expose the work surface beneath it for cleaning. |
| GOAL | Wiping down sink and work surfaces | [partial] The associated activity is cleaning the kitchen work surface.<br><br>Cleaning the work surface captures an essential component but omits cleaning the sink. | [partial] Clean and put stuff away<br><br>Cleaning captures the cleaning activity only broadly; sink and work surfaces are unspecified, and putting things away adds another task. | [partial] Clean and put stuff away<br><br>Cleaning captures the cleaning activity only broadly; sink and work surfaces are unspecified, and putting things away adds another task. |
| NEXT | clean work surface → clean work surface → clean work surface | [586.356, 587.356] wipe down wooden work surface → [587.356, 588.356] wipe down wooden work surface → [588.356, 589.356] wipe down wooden work surface]<br><br>3/3 class pairs correct · ✓ / ✓ / ✓ | push roll of kitchen paper → wipe down wooden work surface → pick up food bits<br><br>1/3 class pairs correct · × / ✓ / × | pull roll of kitchen paper → wipe down wooden work surface → pull cupboard<br><br>1/3 class pairs correct · × / ✓ / × |

### P07-20240529-131737 · 121.74 s · pick up circular tray

| Field | Reference | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- |
| WHY | so that the tray beneath it can be reached | [incorrect] Retrieving the circular tray from the drawer to prepare it for use, likely to place food on it for baking or to organize the oven space.<br><br>Preparing the circular tray for baking or oven organization is different from reaching the tray beneath it. | [incorrect] so that I can use it to cover the baking tray<br><br>Using the circular tray as a cover is different from moving it to reach the tray underneath. | [correct] so that I can access the baking tray underneath<br><br>Captures accessing the tray underneath; identifying it specifically as a baking tray is additional to the supplied reference. |
| GOAL | Bake the chickpeas and wash lettuce leaves | [incorrect] Retrieve baking equipment<br><br>Retrieving equipment is a preparatory micro-action, not an identification of baking chickpeas and washing lettuce. | [incorrect] Preheat oven and prepare baking tray<br><br>Preheating and preparing a tray are supporting steps; the answer does not identify chickpea baking or lettuce washing. | [incorrect] Preheat oven and prepare baking tray<br><br>Preheating and preparing a tray are supporting steps; the answer does not identify chickpea baking or lettuce washing. |
| NEXT | replace circular tray → carry over rectangular tray → place rectangular tray | Place the circular tray on the counter → Close the drawer → Turn around to face the kitchen island<br><br>1/3 class pairs correct · ✓ / × / × | close drawer → put circular tray → pick up baking tray<br><br>0/3 class pairs correct · × / × / × | pick up baking tray → put circular tray → close drawer<br><br>0/3 class pairs correct · × / × / × |

## Generation diagnostics

| Model | Completed | Finish reasons | Missing WHY | Missing GOAL | Missing NEXT slots | Tokens median (range) |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 228 | {"eos": 228} | 1 | 5 | 16 | 78 (35–169) |
| Epoch 1 | 228 | {"eos": 228} | 0 | 0 | 0 | 48 (38–67) |
| Epoch 2 | 228 | {"eos": 228} | 0 | 0 | 0 | 49 (38–68) |

Missing WHY counts only source-eligible examples. A missing extracted answer within a completed generation is distinct from an unfinished generation.

## Semantic flags

| Field | Model | Unsupported addition | Contradiction | Too broad | Too narrow |
| --- | --- | --- | --- | --- | --- |
| WHY | Base | 149/225 | 7/225 | 60/225 | 0/225 |
| WHY | Epoch 1 | 66/221 | 14/221 | 46/221 | 7/221 |
| WHY | Epoch 2 | 99/222 | 14/222 | 43/222 | 7/222 |
| GOAL | Base | 8/223 | 2/223 | 11/223 | 201/223 |
| GOAL | Epoch 1 | 80/228 | 56/228 | 40/228 | 90/228 |
| GOAL | Epoch 2 | 63/228 | 66/228 | 47/228 | 95/228 |

Flags describe explicitly reviewed semantic answers, not a separate overall accuracy. Each cell is flagged decisions / decisions with that flag recorded; missing answers and automatic identical-reference decisions may lack flag reviews. Unsupported ancillary text can coexist with a correct core. A central contradiction requires an incorrect label.

## Training and limitations

Training completed all 1,818 optimizer steps over two epochs: 14,543 initial-SFT examples (1,403 Explain and 13,140 Track), text LoRA rank 16, 116,727,808 trainable parameters, frozen vision/aligner, effective batch size 16, and 16 image frames per example. Epoch checkpoints 909 and 1818 were retained. The earlier audit verified both rank logs, finite losses/gradients, epoch checkpoint hashes and unchanged frozen code. An end-of-run audit failure was recovered without additional optimizer updates. The original run took about 95.16 hours, including validation/checkpointing; the short recovery marker does not represent training duration. The user-authorized cleanup removed 7 intermediate checkpoints and recovered 12.20 GiB. Both epoch checkpoint hashes remained unchanged.

The benchmark covers one validation participant and uses completed annotated action history as input. Test performance, current-action recognition and Track GOAL/NEXT generation accuracy remain unmeasured. Explain already names the queried action, so it cannot measure action recognition. Semantic judgments and generated-phrase mapping are provisional. Per-video breakdowns describe this cohort; no statistical significance or population generalization claim is made.

## Independent semantic review

Additional independent assistant review covered 80 shuffled, selected packets without the original labels or model context; 74/80 labels agreed. 6 disagreements remain visible in review_audit/ and concern rubric boundaries. This is a consistency check on selected samples, not human validation, an estimate of population accuracy, or a confidence interval. Independent QA ratings were not imported into the benchmark scores. The sensitivity table replaces only the disputed packet labels with the independent alternatives wherever those exact packets occur; all other ratings remain fixed. It is not an uncertainty interval for unreviewed rubric disagreements. Separately, initial ratings were corrected for consistent application of the frozen rubric: identical GOAL reference/answer text and material WHY details such as complete ingredient removal or improved fit. These corrections apply across reviewed outputs; full before/after records are retained in the consistency audits. Original matched-47 judgment files remain unchanged.

| Model | Field | Primary correct | Disputed labels replaced |
| --- | --- | --- | --- |
| Base | WHY | 98/226 (43.4%) | 98/226 (43.4%) |
| Base | GOAL | 2/228 (0.9%) | 2/228 (0.9%) |
| Epoch 1 | WHY | 75/226 (33.2%) | 75/226 (33.2%) |
| Epoch 1 | GOAL | 8/228 (3.5%) | 11/228 (4.8%) |
| Epoch 2 | WHY | 81/226 (35.8%) | 82/226 (36.3%) |
| Epoch 2 | GOAL | 11/228 (4.8%) | 11/228 (4.8%) |

## Validation loss by video

| Video | Task | N | Base | Epoch 1 | Epoch 2 |
| --- | --- | --- | --- | --- | --- |
| P07-20240529-102652 | explain | 11 | 1.7373 | 1.0985 | 1.1905 |
| P07-20240529-131737 | explain | 15 | 1.9955 | 1.6217 | 1.8065 |
| P07-20240529-134410 | explain | 19 | 2.6565 | 2.3827 | 2.5621 |
| P07-20240529-191007 | explain | 32 | 2.3732 | 2.0731 | 2.2681 |
| P07-20240529-194518 | explain | 39 | 1.9512 | 1.5666 | 1.6825 |
| P07-20240530-102543 | explain | 38 | 1.9873 | 1.7805 | 1.9947 |
| P07-20240530-111344 | explain | 2 | 1.9056 | 1.3262 | 1.4408 |
| P07-20240531-101357 | explain | 35 | 2.2215 | 1.7852 | 1.9818 |
| P07-20240531-201703 | explain | 37 | 2.2086 | 1.9515 | 2.1747 |
| P07-20240529-102652 | track | 133 | 1.5277 | 1.2807 | 1.3916 |
| P07-20240529-131737 | track | 91 | 1.6550 | 1.6575 | 1.8838 |
| P07-20240529-134410 | track | 156 | 2.2117 | 2.2124 | 2.4209 |
| P07-20240529-191007 | track | 299 | 2.1908 | 2.2013 | 2.4134 |
| P07-20240529-194518 | track | 281 | 1.7166 | 1.6265 | 1.7610 |
| P07-20240530-102543 | track | 326 | 1.8469 | 1.8846 | 2.1066 |
| P07-20240530-111344 | track | 24 | 1.5771 | 1.3674 | 1.5300 |
| P07-20240531-101357 | track | 399 | 1.8627 | 1.7444 | 1.9717 |
| P07-20240531-201703 | track | 272 | 2.0013 | 2.0194 | 2.2733 |

## Reproducibility

Snapshot: `full_comparison`. The canonical evaluator's `scores/manifest.json` records references, mapper, config, judgments and evaluator hashes. `scores/metrics.csv` and `scores/per_example/` preserve every metric and evaluated answer. `analysis.json` stores paired changes and diagnostics; `examples.json` and `report.html` expose every comparison. No evaluator or training implementation was changed for this report.
