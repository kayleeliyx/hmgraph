# Human review

A standalone local browser app for inspecting saved ActGround predictions and recording human decisions. Python's standard library is the only runtime dependency. No model inference is performed.

## Start

```bash
python /data/yaxuanli/ActGround/human_review/review.py --background
```

Open **http://127.0.0.1:8765**. The default configuration loads the five existing model runs, all 2,209 validation examples, original videos, and the 16 cached input frames for each example.

The top navigation switches between **Human review** and **Video timeline**. Open the timeline directly at **http://127.0.0.1:8765/timeline**.

If the app runs on an SSH server, forward its port from your own computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Then open the same URL on your computer. Background mode stays running after the launching terminal closes. Check or stop it with:

```bash
python /data/yaxuanli/ActGround/human_review/review.py --status
python /data/yaxuanli/ActGround/human_review/review.py --stop
```

Logs are saved to `state/server.log`; saved labels remain after stopping. Run without `--background` for an interactive foreground server.

## Review workflow

1. Enter your reviewer name. Each reviewer has separate labels.
2. Choose a model and example. Search or filter by task, video, and review status.
3. Inspect the observed clip, input frames, and completed action history. **Show future** explicitly enables later footage for NEXT verification.
4. Select WHY, GOAL, Current, or NEXT. Compare the reference with the prediction and choose a label. Add a short reason or flag a reference/mapping issue when useful.
5. Move to the next example. Decisions save immediately; notes save after a short pause. Navigation waits for pending saves. The page remembers your position.

Model identities can be hidden for blind review. Automatic assessments are hidden initially and can be revealed separately. Human labels are never prefilled from automatic decisions. A comparison view shows other models' answers.

## What humans label

| Field | Decision | Meaning |
| --- | --- | --- |
| WHY | Correct / Partial / Incorrect / Uncertain | Agreement with the specific narrated purpose. A plausible different reason is insufficient. |
| GOAL | Correct / Partial / Incorrect / Uncertain | Agreement with the annotated activity at its original scope, including essential compound tasks. |
| Current | Joint action and separate verb/noun decisions | Track only. Check the fine-grained current action against its canonical classes. |
| NEXT 1–3 | Joint action and separate verb/noun decisions at each position | One ordered sequence of three fine-grained future actions. Preserve positions and repeated actions. |

**Uncertain** means the reviewer has not resolved the decision. It contributes no imported semantic judgment. **Reference issue** flags an annotation problem for later common-source adjudication; it does not change the evaluation denominator. **Mapper issue** separates a text-to-class mapping error from a wrong model answer.

Missing reference components cannot be labeled. Missing predictions can be marked incorrect. A wrong verb or noun makes a fully annotated joint action incorrect; incomplete component decisions can remain uncertain. The app records whether future footage had been exposed when a label was saved. Future footage verifies what happened; it does not establish that WHY or GOAL was inferable from the model's inputs.

## Saved labels and exports

Labels and their edit history live in `state/labels.sqlite3`, outside the evaluator. Back up this file after stopping the app. Evaluation results and dataset annotations are read-only.

Each label stores reviewer, example, model, field, decision, optional verb/noun decisions, notes, flags, evidence mode, revision, timestamp, and a fingerprint of the prediction/reference contract. Changed answers or references do not inherit old labels. Importing judgments alone does not invalidate existing labels. Concurrent edits from two tabs are detected instead of silently overwritten.

The export menu downloads the active reviewer's current labels:

- **JSONL:** complete annotation records, including uncertain decisions and issue flags.
- **CSV:** the same records for spreadsheets; formula-like text is escaped.
- **Evaluator judgments:** hash-bound WHY/GOAL decisions accepted by the standalone evaluator. Uncertain, source-issue, and nonsemantic records are excluded. Identical semantic packets are deduplicated; contradictory decisions must be resolved before this export.

Example after downloading `human_judgments.jsonl`:

```bash
python /data/yaxuanli/ActGround/evaluation/evaluate.py score \
  --judgments /absolute/path/human_judgments.jsonl \
  --output /absolute/path/evaluation_with_human_labels
```

Unreviewed semantic answers remain pending. A reviewed subset is not a completed evaluation of the whole cohort. Action labels are exported for auditing the mapper and class scores; the evaluator continues to compute action metrics from the fixed canonical mapping.

## Video timeline

The separate explorer covers all **156 dataset videos**, including training and test videos without evaluated predictions. Its source snapshot contains **59,454 action intervals**, **1,490 activity/GOAL intervals**, and **4,296 extracted WHY candidates**, of which **3,124** were accepted by the preparation pipeline and remain pending human validation.

- **GOAL and action bars:** original annotation start/end times across the full video. Overlaps and unannotated gaps remain visible.
- **WHY bars:** extracted purpose candidates aligned to their source action interval. Width shows the action's duration; it does not show how long an intention lasted. Accepted candidates are shown initially; rejected and quarantined candidates can be included.
- **Query markers:** the exact sampled Explain/Track evaluation times. Queries are distinct from the complete annotation timeline.
- **Human judgment markers:** saved decisions or unreviewed states at each eligible query/model/field. No automatic judgments are substituted. Videos without evaluated queries show no prediction results.

Select a video, zoom and pan the timeline, or click a bar to seek playback and inspect the label, narration and canonical classes. Related queries link directly to the corresponding example and field in Human review. The distribution chart switches between verbs, nouns and activities; selecting a category filters the source timeline. Frequency counts distinct source annotation intervals. Covered time merges overlapping intervals within each category; different categories can overlap in time. SVG export saves the current timeline view.

The explorer exposes full-video annotations and playback. Opening a video's full timeline is recorded in browser-local context, so subsequent human labels for that video retain the `future` evidence marker even when the review player itself is limited to the observed window. These decisions measure agreement with the reference; they do not establish visual inferability.

Source annotation problems are retained and reported. Four GOAL intervals have reversed endpoints, one GOAL label is empty, and some actions lack a unique canonical class pair. Raw intervals remain inspectable; invalid intervals are excluded from coverage calculations. Drawing uses video bounds without rewriting source values. Two source rows with the known P09 video-ID typo are linked using the preparation pipeline's existing correction.

The portable source asset is `assets/timeline_annotations.json.gz`. To rebuild it from the prepared dataset using only the standard library:

```bash
python /data/yaxuanli/ActGround/human_review/build_timeline.py --help
```

`timeline_annotations` and `timeline_video_inventory` in `config.json` select the source snapshot and media inventory. Both resolve beside that config. Without the source asset, the explorer explicitly displays a partial timeline assembled from available evaluation metadata.

## Use later results

Change `evaluation_dir` in `config.json` to another output from the standalone evaluator. That directory must contain `references.jsonl` and `per_example/*.jsonl`. Its `mapper.json` and `semantic_packets.jsonl` are used when present. Restart the app to load the new snapshot.

All configuration paths resolve relative to the configuration file:

```json
{
  "title": "My experiment review",
  "evaluation_dir": "/absolute/path/new_evaluation",
  "evaluation_config": "/absolute/path/comparison_config.json",
  "media_records": "/absolute/path/validation.jsonl",
  "video_inventory": "/absolute/path/video_inventory.json",
  "state_dir": "/absolute/path/review_labels"
}
```

`evaluation_config`, `media_records`, and `video_inventory` are optional. Without them the app still displays extracted predictions and frozen references. Raw answer text is displayed only when its archived hash matches. Media records must match the frozen example/task/time and action/purpose/activity references. Only registered media files are served. Relative video paths resolve beside the inventory; relative frame paths resolve beside the media records.

The whole `human_review/` folder can be copied elsewhere and run with its own configuration. It does not import the training or evaluation code. The semantic export contract is explicitly frozen to `hdepic_field_evaluation_v1` / `hdepic_semantics_v1`; incompatible packet hashes are rejected.

## Checks

```bash
python /data/yaxuanli/ActGround/human_review/review.py --test
```

The tests cover persistence, reviewer separation, revision conflicts, changed references, semantic export bindings, media identity, missing components, video byte-range seeking, HTTP write validation, timeline source preservation, interval union coverage and query-level human decisions. The optional browser checks in `tests/test_browser.cjs` and `tests/test_timeline_browser.cjs` use Playwright supplied separately; the app itself has no JavaScript dependencies. Current verification and a timeline preview are saved in `verification.json` and `timeline-preview.png`.

This is a local research tool. Reviewer names identify annotation records and are not authenticated accounts. Use a localhost connection or SSH tunnel for remote review.
