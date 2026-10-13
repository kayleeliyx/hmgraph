# Qwen3.8-27B SFT evaluation

The complete comparison is in **[full_validation/report.html](full_validation/report.html)**,
with a **[Markdown report](full_validation/report.md)** and
**[reproduction instructions](full_validation/README.md)**.

Base, epoch 1 and epoch 2 each have 228 completed Explain predictions on the same
nine P07 validation videos. All semantic packets are assessed. WHY has 226
eligible references; GOAL and NEXT use 228. WHY/GOAL ratings remain provisional
assistant assessments, with disagreements and consistency corrections recorded.
The canonical evaluator, source eligibility and class mapper are unchanged.

Training completed two epochs and 1,818 optimizer steps. All 2,209 validation
examples have base/epoch-1/epoch-2 teacher-forced losses. Epoch 1 remains selected
by Explain validation NLL. Track generation/current-action accuracy and test
performance remain unmeasured.

The authorized cleanup removed seven intermediate checkpoints and recovered
12.20 GiB. Both epoch checkpoints (909 and 1818), their hashes, and the original
47 prediction records were preserved. Audit records are in
`checkpoint_cleanup_20261011/` and `full_validation/protected_artifacts_audit.json`.

## Folder contents

- `full_validation/`: the final comparison (report, scores, review records, reproduction steps).
- `low_accuracy_diagnosis/`: analyses of why WHY/GOAL/NEXT accuracy is low.
- `semantic_review/`: the 184 WHY/GOAL judgments from the earlier 47-example review. The final
  comparison reuses them.
- `analysis.json`, `training_audit.json`: verified validation-loss and training evidence, read by
  `full_validation/render_report.py`.
- `checkpoint_cleanup_20261011/`, `cleanup_intermediate_checkpoints.py`: record of the authorized
  removal of intermediate checkpoints.

The preliminary 47-example report and its inputs were removed on 2026-10-13 because the full
comparison supersedes them.
