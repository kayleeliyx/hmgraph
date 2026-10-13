# Complete Qwen 27B validation comparison

This folder is the final Explain evaluation. It reuses the 184 judgments from the earlier
47-example review (`../semantic_review/judgments.jsonl`); that preliminary report was removed.
All scoring uses the unchanged standalone evaluator, frozen reference eligibility,
GOAL profiles, WHY/GOAL rubrics, and train-only verb/noun mapper. WHY and GOAL
semantic decisions remain provisional assistant assessments until human calibration.

Every command below must run from `/data/yaxuanli/ActGround` after:

```bash
source scripts/hdepic_data_storage.sh
```

`prepare_and_score.py` snapshots the prediction JSONL files once, validates unique
IDs and completed status, and records hashes of exactly the bytes consumed. An
unfinished final JSONL line is excluded while generation is running. It checks
prompts, image preprocessing, generation settings, and native model identity
against the base for every scored example. It never starts GPU inference.

The default requires all 228 Explain predictions for every requested model.
Failed records, active records and not-yet-generated IDs are counted separately.
Any failure also prevents a complete-cohort claim. Incomplete generation is never
silently counted as incorrect. The explicit
`--completed-intersection` option restricts all models and references to the same
completed IDs; its cohort manifest labels the result as a partial cohort.

## Base review preparation

```bash
python analysis/qwen38_sft_evaluation/full_validation/prepare_and_score.py \
  --models base --name base_only
```

Reviewers should read `base_only/why_packets.jsonl` or `base_only/goal_packets.jsonl`.
Packets contain the fixed rubric and omit model names. This is assistant review,
not a fully blinded study: task orchestration and paths can reveal source context.
Do not use
`scores/semantic_packet_links.jsonl` or model prediction files to infer model
identity. Record `packet_id`, `prompt_sha256`, `judge`,
`assessment_status=provisional_assistant_assessment`, and `judgment` with the
field, label, reference-specific justification, and explicit semantic flags.
Write each reviewer's decisions to a separately owned JSONL file under this
folder; never overwrite the previous matched-47 judgments.

## Collecting additional completed answers during generation

Use the queue helper to collect fully finished ratings and assign a new immutable
batch from the requested model's completed generations:

```bash
python analysis/qwen38_sft_evaluation/full_validation/queue_review.py --model epoch1
# Switch to --model epoch2 when that checkpoint begins generating.
```

This checks exact packet-ID and prompt-hash coverage of every base and batch
field review. An unfinished judgment file is ignored in its entirety until it
is complete. It reads only `base_only/{why,goal}_judgments.jsonl` and
`review_batches/batch*/{why,goal}_judgments.jsonl`; independent QA is excluded.
The original 184 judgments are added automatically by `prepare_and_score.py`.
Invalid hashes, duplicates, unexpected IDs and conflicting decisions are errors.

New batches are numbered `batch002`, `batch003`, etc., with at most 40 packets
per field by default (`--max-per-field` can change this). Each contains immutable
`why_packets.jsonl`, `goal_packets.jsonl`, and a hashed `source_status.json`.
Packets already assigned in the base or any previous batch are never assigned
again, even if their review is unfinished. No empty batch is created. The helper
prints the new path and counts for dispatch to reviewers. It only replaces the
derived `in_progress/` snapshot and collection metadata; assigned packet files
and original predictions remain unchanged.

The lower-level snapshot command is also available:

```bash
python analysis/qwen38_sft_evaluation/full_validation/prepare_and_score.py \
  --models base epoch1 --completed-intersection --name epoch1_in_progress \
  --judgments path/to/additional_why_judgments.jsonl path/to/additional_goal_judgments.jsonl
```

The original 184 matched-47 judgments are always loaded and reused only when
packet ID and prompt hash match. Additional files are merged with duplicate and
conflict checks. Already judged packets disappear from the pending packet files.
This command can be rerun as generations complete. It replaces only its named
snapshot and derived scores; all original predictions and judgments remain intact.

## Final complete comparison

```bash
python analysis/qwen38_sft_evaluation/full_validation/queue_review.py --collect-only

python analysis/qwen38_sft_evaluation/full_validation/prepare_and_score.py \
  --models base epoch1 epoch2 --name full_comparison \
  --judgments analysis/qwen38_sft_evaluation/full_validation/completed_review_judgments.jsonl \
  --require-complete-semantics
```

The final command refuses generation cohorts below 228/model and returns nonzero
when semantic packets remain. `full_comparison/scores/summary.json`, `metrics.csv`,
`per_example/*.jsonl`, and `manifest.json` are the authoritative evaluator output.
228 Explain examples come from the same nine P07 validation videos. WHY uses the
frozen 226 eligible sources; GOAL uses 228. There is no generated Track/current
action accuracy or held-out test evaluation in this comparison.

Render the final report only after all three generations and semantic review are
complete:

```bash
python analysis/qwen38_sft_evaluation/full_validation/render_report.py
```

This checks full cohorts, zero pending semantic decisions and unchanged scored
input hashes, then creates `report.html`, `report.md`, `analysis.json` and
`examples.json`. The report separates full teacher-forced loss from generated
accuracy, preserves the original epoch-1 selection rule, includes all metric
components and per-video/paired counts, and makes all 228 answers searchable with
filters for correctness gains/losses. It refuses to emit a final report from a
partial or unreviewed snapshot.

The GOAL scope table is an auxiliary diagnostic using the unchanged automatic
source-side scope-review flag. Both flagged and unflagged profiles remain
provisional; an unflagged profile is not human-validated. Full GOAL metrics still
use all 228 examples. Any independent stratified semantic spot check is reported
as reviewer consistency only and never imported as benchmark labels.
Independent disagreements are preserved under `review_audit/`. The final report
also shows sensitivity to replacing only those disputed labels with the second
reviewer's labels. This is not a confidence interval or a human accuracy estimate.
`review_audit/*_consistency_corrections.json` separately record initial ratings
corrected for consistent rubric application: identical GOAL reference/answer text
and material WHY details such as complete ingredient removal or improved fit. The original
matched-47 judgment file remains unchanged.

After rendering, check the interactive report with the existing local browser:

```bash
node analysis/qwen38_sft_evaluation/full_validation/check_report.js
```

The browser checker verifies case coverage, filter counts, search, expansion,
desktop/mobile overflow and browser errors. It saves its results and screenshots
in this folder; runtime profiles and caches follow the sourced `/data` storage
environment.

## Metrics

- WHY and GOAL: counts correct/partial/incorrect; strict correct rate; ordinal
  score `(2 * correct + partial) / (2 * eligible)`; scope/contradiction/unsupported
  addition flags; source exclusions and judgment completion.
- NEXT: verb, noun, and joint verb+noun accuracy at positions 1, 2, and 3; mean
  position accuracy; complete ordered-sequence accuracy; normalized Levenshtein
  and Damerau-OSA distance (lower is better); missing or extra forecast positions.
- Paired comparisons and per-video results use exactly the same eligible IDs.
- Teacher-forced validation loss and training completion are documented in the
  parent `training_audit.json` and `analysis.json`; loss is not generated accuracy.

Do not infer improved semantic accuracy from loss reductions or formatting.
Do not claim significance or broad generalization from one validation participant.
