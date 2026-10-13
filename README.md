# HMGraph

HMGraph builds hierarchical, action-centric memory graphs from HD-EPIC
egocentric cooking videos.

The implemented pipeline currently covers raw one-second visual observations
(`W`) and consolidated micro-actions (`L1`). The next planned stages are
goal-based recipe steps (`L2`) and session-level task graphs (`L3`).

## Repository layout

```text
code/       Pipeline implementation and tests
scripts/    Run and visualization entry points
runs/       Preserved experiment results (ignored by Git)
logs/       Runtime logs (ignored by Git)
```

Detailed design notes are in `project_notes.md`. The canonical pipeline and
validation commands are documented in `PIPELINE.md`.

The fixed RAVU-style graph retrieval and answering backend is documented in
`GRAPH_QA.md`. It compares W, L1, and hierarchical W→L1 graphs without changing
the retrieval or answer method between conditions.

## Preserved reference run

The current reference video is `P01-20240202-110250`:

```text
runs/P01-20240202-110250/gemma/  Complete Gemma W/L1 pipeline
runs/P01-20240202-110250/qwen/   Complete Qwen W + grouped-L1 baseline
```

Both directories retain their structured JSON results, configuration, logs,
and one-frame-per-second prediction overlay. Regenerable extracted and
diagnostic frame trees are intentionally not retained.

## Entry points

```bash
# Resume/run the Gemma L1 pipeline
scripts/run_l1.sh

# Start a fresh Qwen W + grouped-L1 run
scripts/run_l1_qwen.sh

# Render a prediction overlay (defaults to the preserved Gemma run)
python scripts/visualize_predictions.py
```

The Qwen migration is currently a working-tree checkpoint and should be
reviewed and committed before starting L2 development.
