# Why validation accuracy is low (Qwen3.8-27B, Explain)

CPU-only analyses of the full 228-query Explain comparison in `../full_validation/`.

- `diagnose.py` → `diagnosis.json`
  - GOAL: how often answers copy training activity labels.
  - WHY: answer length, wording templates, gains and losses.
  - NEXT: image-free baselines, ordering tolerance, accuracy versus how often an action occurs in training.
- `goal_scope.py` → `goal_scope.json`: how long GOAL activity intervals are compared with the
  5 s frame window, and how often a query falls inside an annotated recipe step.

Run from the project root after `source scripts/hdepic_data_storage.sh`:

```bash
python3 -I analysis/qwen38_sft_evaluation/low_accuracy_diagnosis/diagnose.py
.venvs/llavaction/bin/python -I analysis/qwen38_sft_evaluation/low_accuracy_diagnosis/goal_scope.py
```
