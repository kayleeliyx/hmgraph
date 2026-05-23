# HMGraph Week-1 Pipeline

## Purpose

This repository now uses a clean canonical Week-1 pipeline for generating W_action, scene observations, rolling context, combined evidence, and final L1 multimodal micro-action candidates. The accepted final L1 version is `v3_2_1`.

## Canonical 0-60 Command

```bash
cd /data2/yaxuanli/hmgraph_codex
python -m code.run_hmgraph_pipeline \
  --video-path /data2/yaxuanli/HD-EPIC/Videos/P01/P01-20240202-110250.mp4 \
  --output-dir /data2/yaxuanli/hmgraph_codex/week1_validation/P01-20240202-110250 \
  --gt-jsonl /data2/yaxuanli/hmgraph_codex/week1_validation/P01-20240202-110250/gt.jsonl \
  --start-sec 0 \
  --end-sec 60 \
  --pipeline-version v3_2_1 \
  --range-label 0_60 \
  --overwrite
```

Validate with:

```bash
python -m code.validate_hmgraph_pipeline_outputs \
  --output-dir /data2/yaxuanli/hmgraph_codex/week1_validation/P01-20240202-110250 \
  --start-sec 0 \
  --end-sec 60 \
  --pipeline-version v3_2_1 \
  --range-label 0_60
```

## Arbitrary Range Command

Use any `--start-sec`, `--end-sec`, and optional `--range-label`. The runner constructs output names from the range label.

```bash
python -m code.run_hmgraph_pipeline \
  --video-path /path/to/video.mp4 \
  --output-dir /path/to/output_dir \
  --start-sec 120 \
  --end-sec 180 \
  --pipeline-version v3_2_1 \
  --range-label 120_180 \
  --overwrite
```

## Full Video Command

Full-video support is available by omitting `--end-sec`; this resolves duration with `ffprobe` and uses `range_label=full` unless overridden. Do not run full video for Week-1 diagnostics unless explicitly requested.

```bash
python -m code.run_hmgraph_pipeline \
  --video-path /path/to/video.mp4 \
  --output-dir /path/to/output_dir \
  --start-sec 0 \
  --pipeline-version v3_2_1 \
  --range-label full \
  --overwrite
```


## Runtime Note

The generic runner sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for canonical subprocesses unless the user already set it. This is a CUDA memory allocator stability setting only. It is not a model prompt, model input, decoding, or accuracy change.

## GT Policy

GT may be loaded only as optional reference in quality reports. GT is not used for W construction, scene construction, rolling context, L1 prompting, L1 parsing, repair, postprocess, or consolidation.

At a high level, the pipeline turns raw egocentric video into a
  structured list of L1 micro-actions: short human-object actions
  like “open cupboard door,” “grab mug,” “press/control coffee
  machine area,” etc.

  The core idea is:

  video
    -> per-second W_action + scene observations
    -> rolling object/context memory
    -> multimodal L1 reasoning over short segments
    -> postprocess/cleanup/calibration
    -> final L1 event list

  1. code.stages.w_scene

  This is the first evidence extraction stage.

  It does two things per 1-second window:

  1. W_action
      - Ask the VLM: what human-object action is happening in this
        short window?
      - Output examples:
          - open / cupboard door
          - pick up / mug
          - reach for / knob
  2. scene_obs
      - Ask the VLM: what objects, scene type, current focus, and
        visible action evidence are present?
      - Output examples:
          - scene type: kitchen
          - visible objects: mug, cupboard, shelf, counter
          - current focus: taking mug from cupboard
          - action presence: visible_action / uncertain_action /
            no_action

  So w_scene gives rough per-second evidence.

  Important: W_action is noisy. It is not final truth.

  2. code.stages.scene_context

  This builds memory over time.

  It takes the W_action and scene observations and creates:

  rolling_context
  combined W + scene + rolling_context

  The point is to remember stable objects and part-whole relations
  across windows.

  Example:

  cupboard door part_of cupboard
  mug located on shelf
  mug later held in hand
  coffee machine on counter

  This helps later L1 reasoning avoid blindly trusting one bad W
  prediction.

  For example, if W says “container” but scene/context repeatedly
  shows “mug,” L1 can correct it to “mug.”

  3. code.stages.l1_reasoner

  This is the main L1 multimodal reasoning stage.

  It has several internal steps.

  3.1 code.l1.segment_proposer

  This groups per-second windows into short L1 reasoning segments.

  Rule:

  Group nearby seconds that likely belong to the same local
  action,
  but keep segments short, usually around 1-3 seconds.
  Add context before/after.

  Example:

  7-8s: cupboard opening
  8.5-12s: mug retrieval
  13-18s: coffee machine interaction

  It does not create final L1 events. It only decides where the
  model should look.

  3.2 code.l1.frame_selector

  This selects full video frames for each segment.

  Rules:

  Use full frames only.
  No crops.
  Sample core segment densely.
  Sample before/after context sparsely.
  Include first/middle/last frames.
  Include frames near source window boundaries.

  This gives the VLM enough visual evidence to reason over the
  segment.

  3.3 code.l1.visual_discovery_prompt

  This is the visual-first pass.

  The model sees selected frames and context, but not W_action.

  Goal:

  Discover possible L1 micro-actions from visual evidence first.

  This reduces W-copying.

  Example output:

  candidate: hand pulls cupboard door open
  candidate: hand grasps mug
  candidate: hand manipulates top/control area of coffee machine

  It labels candidates as:

  final_candidate
  support_only
  uncertain_candidate
  reject_hypothesis

  3.4 code.l1.prompt

  This is the final L1 reconciliation prompt.

  Now the model sees:

  visual discovery
  W_action
  scene_obs
  rolling_context
  selected frames

  Its job:

  Create final L1 events.
  Audit every W_action.
  Account for every visual candidate.
  Do not silently drop evidence.

  Rules:

  - W_action is noisy evidence, not ground truth.
  - Visual discovery is also evidence, not ground truth.
  - Every visual candidate must be:
      - accepted as L1
      - merged into L1
      - used as support
      - rejected with reason
  - Every W_action must be audited:
      - confirmed
      - corrected
      - downgraded to support
      - too vague
      - too specific
      - partially supported
      - likely wrong
      - not applicable

  This is the key reasoning step.

  3.5 code.l1.parser

  This parses the VLM JSON output.

  It does not invent new events.

  It does:

  strip markdown fences
  extract JSON
  fill conservative defaults
  normalize labels
  validate event schema
  move hypothesized events out of final L1
  warn about bad timestamps or schema issues

  It protects the pipeline from messy model output.

  3.6 code.l1.confidence_reasoner

  This calibrates confidence for raw L1 events before postprocess.

  Rule:

  Do not allow overconfident events.
  Observed clear core actions can be high.
  Inferred / ambiguous / weak visual evidence gets lower
  confidence.
  Confidence 1.0 should not happen.

  3.7 code.l1.event_repair

  This repairs schema-level issues.

  Example:

  If the model says:

  description: touch top control area
  object: button
  event_subtype: core_effect
  duration: long

  That is over-specific. Repair may change it to:

  object: control area
  event_subtype: ambiguous_part_manipulation
  evidence_type: inferred

  It does not create new L1 events. It only fixes bad structure.

  3.8 code.l1.cross_segment_consolidation

  This handles duplicates from overlapping segments.

  Rule:

  Do not merge just because object is the same.
  Only merge if same object, same action, overlapping/adjacent
  time, and same physical phase.
  Use relations instead of merging when unsure.

  Example:

  grab mug
  pick up mug

  may be linked or merged if they are the same continuous
  retrieval.

  But:

  coffee machine control interaction 13-15s
  coffee machine control interaction 16-18s

  should stay separate if merging would create one long vague
  action.

  4. code.stages.l1_postprocess

  This is the final v3_2_1 cleanup stage.

  It takes raw v3.2 L1 and produces the accepted final L1.

  4.1 code.l1.confidence

  This recalibrates confidence using evidence bands.

  Rule:

  Do not serially subtract penalties until confidence becomes 0.
  Use reasonable confidence bands.

  Examples:

  clear observed core action: 0.80-0.92
  ambiguous inferred manipulation: 0.45-0.65
  support/state: 0.50-0.75

  4.2 code.l1.consolidation

  This does final duplicate reduction.

  It is conservative.

  Rule:

  Merge obvious duplicates.
  Do not overmerge long same-object events.
  Keep support events as relations when needed.

  4.3 code.l1.ledger

  This normalizes the visual candidate ledger.

  It makes sure visual discovery candidates are accounted for
  correctly:

  accepted_as_l1
  merged_into_l1
  support_for_l1
  rejected_as_duplicate
  rejected_due_to_weak_evidence
  rejected_as_hypothesis

  4.4 code.l1.w_audit

  This normalizes W_action audit decisions.

  It fixes cases like:

  W says open cupboard door
  final L1 says open cupboard door

  That should be confirmed, not not_applicable.

  4.5 code.l1.warning_cleanup

  This removes stale warnings after confidence repair.

  Example:

  If a warning says “confidence too high,” but postprocess lowered
  confidence, the warning can be removed.

  Final Output

  The final output is:

  l1_candidates_multimodal_{range}_v3_2_1.jsonl

  Each L1 event has:

  time range
  verb
  object
  part_of
  event_subtype
  evidence_type
  confidence
  visual evidence
  structured evidence
  warnings
  QA importance
  + visual frames
  + scene observations
  + rolling object memory
  + visual-first discovery
  + postprocess calibration

  to produce short, useful human-object events.

  In One Sentence

  The pipeline first extracts noisy per-second action and scene
  evidence, builds object memory, asks a multimodal reasoner to
  discover and reconcile short human-object micro-actions, then
  postprocesses those events so every visual/W evidence item is
  accounted for, confidence is calibrated, and duplicates are
  reduced.