# HD-EPIC evaluation

One standalone evaluator for WHY, GOAL, current action and the next three actions. Copy this folder to another location and run it with Python 3.10 or newer. Scoring with the bundled references and mapper uses only Python's standard library.

## Folder

```text
evaluation/
  evaluate.py       entry point
  config.json       current multi-model comparison
  hdepic_eval/      one implementation of the evaluator
  assets/          frozen validation references, mapper and cached WHY ratings
  docs/            metrics and the three scoring protocols
  tests/           CPU tests
  results/         current report and machine-readable scores
```

There is one active implementation. Protocol identifiers inside manifests record the scoring definition; they are not additional code versions. Historical experiment records remain in their original runs. A repeated evaluation updates its selected output folder and removes obsolete per-model output files owned by the preceding evaluation.

## Evaluate new results

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py score \
  --predictions /absolute/path/new_results.jsonl \
  --name qwen27_sft \
  --tasks explain \
  --output /absolute/path/new_evaluation
```

Use `--tasks explain track` for a run intended to generate both tasks. Task selection is explicit so omitted predictions cannot silently shrink the intended evaluation set. The references and mapper default to the frozen files in `assets/`. Use `--references` and `--mapper` to select other prepared scoring inputs deliberately.

Each prediction is one JSON object per line:

```json
{"id":"the exact benchmark query ID","prediction":"<why>...</why><goal>...</goal><next>take mug; place mug; close cupboard</next>"}
```

Optional fields: `status`, `task`, `prompt_sha256`. Missing status means completed; explicit non-completed status means failure. Missing query IDs remain failures. Duplicate IDs and IDs outside the declared reference cohort are rejected. Preserve exact IDs when generating new output versions.

## Compare several result versions

Edit the `models` and `comparisons` entries in `config.json`, then run:

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py score
```

The default config uses the existing 8B base/epoch1/epoch2 and 35B/27B untuned files. Paths resolve relative to the configuration file, not the terminal working directory. With `--config /path/comparison.json`, resolve its paths relative to that file. Compare checkpoints of the same backbone on identical intended query IDs and prompts. Explain and Track remain separate; 27B's existing generation workflow is Explain-only.

## Exactly what is measured

See [the complete metric definitions](docs/metrics.md). Main scores:

| Field | Correctness criterion | Main metric |
| --- | --- | --- |
| WHY | Specific immediate purpose stated in the narration | Correct / eligible references |
| GOAL | Annotated active activity, at its original scope | Correct / eligible references |
| Current action | Official verb and noun classes, Track only | Joint verb–noun accuracy |
| NEXT | Three sequential fine-grained actions starting after the query | Joint accuracy at positions 1, 2, 3 and their mean |

Additional metrics: semantic partial-credit score; verb/noun accuracy; full ordered-sequence accuracy; normalized Levenshtein and restricted Damerau/OSA distances. Diagnostic counts record semantic flags, missing outputs, pending judgments, source exclusions, mapping coverage and forecast completeness. No combined overall accuracy or whole-response syntax score is used.

Protocols: [WHY](docs/why.md), [GOAL](docs/goal.md), [NEXT and current action](docs/next.md). WHY comes from Codex's narrated-purpose rubric; Claude supplied action-class mapping heuristics. GOAL's scope is preserved rather than converted into a new session-goal hierarchy. INSIGHT's intention similarity is an auxiliary reward, and its forecasting metric informs NEXT; neither establishes GOAL correctness by itself.

## Semantic judging

Action/NEXT scores compute immediately from fixed class IDs. WHY/GOAL are meaning judgments. Cached WHY decisions are reused only for the identical answer and bound reference. Missing answers fail; identical-reference answers are sufficient matches. Nonidentical wording is never automatically wrong.

Unresolved answers produce `semantic_packets.jsonl` with frozen, model-blind prompts. Their model/query mapping is stored separately. Supply ratings using:

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py score \
  --config /absolute/path/comparison.json \
  --judgments /absolute/path/judgments.jsonl \
  --output /absolute/path/evaluation_results
```

A decision row contains `packet_id`, `prompt_sha256`, `judge`, `assessment_status`, and `judgment`:

```json
{"packet_id":"packet hash","prompt_sha256":"prompt hash","judge":"fixed judge identity","assessment_status":"provisional_automated_assessment","judgment":{"field":"goal","label":"correct","justification":"Specific reference agreement.","flags":{"unsupported_addition":false,"contradiction":false,"too_broad":false,"too_narrow":false}}}
```

An optional automated-judge client accepts a packet on stdin and returns its `judgment` object on stdout:

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py judge \
  --packets /absolute/path/semantic_packets.jsonl \
  --output /absolute/path/judgments.jsonl \
  --judge-name fixed-judge-identity -- /absolute/path/judge-client
```

No judge executes by default. Pending judgments produce `pending`/`null`, not zero or accuracy on a judged subset. `--require-complete` returns status 2 while judgments remain. Actual GOAL judging is still pending; semantic judgments and the action mapper remain provisional until the later checking phase. Human review and visualization are deferred.

## Outputs

- `report.md`: readable score table, including all three NEXT positions.
- `metrics.csv`: every named metric, component, position, numerator, denominator and status.
- `summary.json`: full task/video metrics and shared-cohort comparisons.
- `per_example/`: extraction, mapped classes and semantic decisions for each model.
- `manifest.json`: configuration, code, reference, mapper and rubric hashes.
- Reference/mapper snapshots and pending judge packets make each output folder reproducible.

An ordinal score uses `(2 × correct + partial) / (2 × eligible)`. It is supplementary, not a probability. Missing gold excludes only the affected class component. Source exclusions are shared across models. Repeated actions can be valid. Narrated/reference agreement does not establish visual inferability.

Run the tests from any working directory:

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py test
```
