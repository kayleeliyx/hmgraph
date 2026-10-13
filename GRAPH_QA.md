# Fixed RAVU-style graph QA backend

This repository now has one graph retrieval and answer backend that can be held
fixed while graph construction changes. It is **RAVU-style**, not a claim of an
exact RAVU reproduction: RAVU did not publish an official implementation, and
its Gemini 1.5 Flash dependency is no longer the practical open-model choice.

The main scientific comparison is:

```text
W graph ───┐
L1 graph ──┼─> same planner -> same retriever -> same reranker -> same answerer
W→L1 graph ┘
```

Only the graph changes. Retrieval hyperparameters, model checkpoints, prompts,
and answering are pinned by one JSON config.

## What happens for one question

1. The question-only planner emits explicit operations such as temporal
   filtering, node retrieval, entity expansion, hierarchy expansion, temporal
   expansion, ordering, or counting.
2. A dense model searches multiple graph views: events, entities, attributes,
   and relations.
3. Reciprocal-rank fusion combines the views.
4. Qwen reranks the initial candidates.
5. The program traverses typed graph edges. It can search location/state/entity
   relations, temporal neighbors, and W→L1 hierarchy links.
6. Temporal selection, ordering, and deduplicated count operators produce
   explicit structured results in the retrieval trace.
7. Qwen reranks the expanded evidence and answers from that evidence.

Answer choices are **not** retrieval queries. They are shown only to the final
answerer after evidence has been fixed. Ground truth is used only for scoring.

The program language is intentionally open-ended: operator names and node/edge
types are strings. Future L2/L3, AVA, scene graphs, knowledge graphs, object
tracks, audio nodes, or external knowledge can be added through adapters without
changing the retriever.

## Models in the primary fixed config

`configs/ravu_qwen3_fixed_v1.json` pins:

- `Qwen/Qwen3-Embedding-0.6B` for multi-view dense retrieval;
- `Qwen/Qwen3-Reranker-4B` for candidate and evidence reranking;
- the already-local `Qwen/Qwen3-VL-30B-A3B-Instruct` for answering.

These choices modernize RAVU's semantic-retrieval/reranking structure with open
Qwen models specialized for retrieval. `configs/ravu_mpnet_ablation_v1.json`
keeps RAVU's original `all-mpnet-base-v2` first stage as an ablation.
Exact Hugging Face commit revisions are pinned in both configs.

All neural inference is CUDA-only. The code raises an error rather than falling
back to CPU. The deterministic unit tests use fake models and are not inference
runs.

## Evaluation

```bash
cd /data2/yaxuanli/hmgraph_codex

/data/yaxuanli/miniconda3/envs/avagemma/bin/python -m code.evaluate_ravu_graph_qa \
  --config configs/ravu_qwen3_fixed_v1.json \
  --annotation-root /data2/yaxuanli/HD-EPIC/hd-epic-annotations/vqa-benchmark \
  --w-jsonl runs/P01-20240202-110250/qwen/w_outputs.jsonl \
  --l1-jsonl runs/P01-20240202-110250/qwen/l1_candidates_wgroup_qwen.jsonl \
  --output-dir runs/P01-20240202-110250/qa_ravu_qwen3_fixed_v1
```

The evaluator is resumable by `(question_id, graph mode)`. Use `--overwrite` to
start over. A targeted smoke run can add
`--question-id fine_grained_action_localization_945 --modes w+l1`.

On a shared server, set `CUDA_VISIBLE_DEVICES` to GPUs with enough free memory
before launching. Qwen3-VL uses automatic placement across the visible GPUs;
the embedder and reranker use visible devices 0 and 1 from the config.

The resumable background launcher defaults to physical GPUs 0 and 3:

```bash
scripts/run_ravu_graph_qa.sh
```

## Visual verification is a separate experiment

BBOX questions require pixels to identify the referenced object. In graph-only
mode the trace reports this limitation explicitly. The retriever exposes a
`VisualReferenceResolver` interface for a later AVA-style visual verification
condition, but it is intentionally disabled in the primary graph-construction
comparison so pixels cannot hide differences between W, L1, and W→L1 graphs.
