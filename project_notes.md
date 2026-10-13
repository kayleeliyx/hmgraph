# Project Notes: Hierarchical Action-Centric Memory Graph for Egocentric Cooking Video Understanding

---

## 1. Project Overview

This project builds a **hierarchical, action-centric graph memory system** for egocentric cooking video understanding, designed to run on wearable/mobile devices via a cloud-mobile co-design. The system is evaluated on the **HD-EPIC dataset** and compared against state-of-the-art systems including WorldMM, EgoGraph, and AVA.

**Target venues:** MobiSys, UbiComp/IMWUT, MobiCom

**Base codebase:** AVA (https://github.com/I-ESC/Project-Ava)

**Server setup:**
- 4x NVIDIA RTX 6000 GPUs
- HD-EPIC data at `/data2/yaxuanli/HD-EPIC/Videos/` (participants P01-P09)
- AVA repo at `/data/yaxuanli/AVA/` (treat as read-only; do not modify)
- Project working directory: `/data2/yaxuanli/hmgraph_codex/` (all new code, logs, and experiment outputs)
- Conda environments:
  - `ava` → Qwen2.5-VL-7B-Instruct-AWQ (4-bit quantized, ~4GB weights)
  - `avagemma` → Gemma multimodal model (currently default for L1 generation)

---

## 2. Key Related Papers

| Paper | Venue | What It Does | Key Limitation |
|---|---|---|---|
| **AVA** (base code) | NSDI 2026 | Flat event KG for long video analytics, agentic retrieval | Flat graph, entity-centric, no mobile |
| **EgoGraph** | arXiv 2026 | Temporal KG for egocentric video, entity-centric flat graph | Flat, entity-centric, no hierarchy |
| **WorldMM** | arXiv 2026 | Multi-memory agent (episodic+semantic+visual), fixed time windows | Fixed windows not semantic, no compression, no mobile |
| **EASG** | CVPR 2023 | Egocentric Action Scene Graphs with verb-centric structure | Flat, short clips only, no hierarchy |
| **Kaylee's Proposal** | Google Fellowship | Hierarchical memory graph for wearable systems (proposed, not built) | Not implemented |
| **HD-EPIC** | CVPR 2025 | Rich egocentric cooking dataset with 59K action annotations | Dataset paper |

---

## 3. Core Motivation: Human Memory is Hierarchical and Action-Centric

Supported by cognitive science literature:

- **Event Segmentation Theory** (Zacks et al., 2007) — brain automatically segments experience into nested event hierarchies at multiple timescales simultaneously
- **Episodic Memory** (Tulving, 1972, 2002) — memories organized around "what happened" (actions), not static objects
- **Cortical Hierarchy** (Baldassano et al., 2017) — fMRI shows different brain regions process second-level, minute-level, and hour-level events simultaneously
- **Goal Hierarchy** (Zacks & Tversky, 2001) — actions understood relative to their goal context ("why")
- **Chunking** (Miller, 1956; Cowan, 2001) — brain compresses information into chunks to manage limited memory
- **Memory Consolidation** (McClelland et al., 1995) — selective retention of novel/important information, forgetting redundant detail
- **Schema Theory** (Bartlett, 1932) — repeated experiences compressed into unified mental templates

**Key claim:** Existing systems (WorldMM, EgoGraph) use entity-centric flat graphs — contradicting cognitive science evidence. This project instantiates the known structure of human episodic memory computationally.

---

## 4. Pipeline and Graph Design

### 4.1 Pipeline Layers (W, L1, L2, L3)

The pipeline has **four data layers** but **only three conceptual hierarchy levels**. This distinction matters for how the work is framed in the paper.

```
W  (raw windows)         Sensor-rate VLM observations         NOT a cognitive level
                                  ↓ consolidate (denoise)
L1 (micro-actions)       Distinct verb+object events         First conceptual level
                                  ↓ lift by sub-goal
L2 (recipe steps)        Goal-grouped action chains          Second conceptual level
                                  ↓ aggregate
L3 (session/task goal)   One full task                        Third conceptual level
```

**W is plumbing, not a level.** It is the raw output of running the VLM on fixed 3-second windows. We preserve W verbatim for debugging, ablation, and audit, but it is not a cognitive level — humans do not perceive in 3-second slices, and HD-EPIC has no GT at this granularity. In the paper, W is described as "the system samples the video at fixed 3s windows and runs a VLM on each," not as a hierarchy level.

**L1 is the first real level.** Adjacent W records with the same canonicalized (verb, direct_object) are merged into a single L1 node spanning the full duration. HD-EPIC's action-level annotations are at L1 granularity, so L1 is what we evaluate against for action recognition / localization.

**L2 is the abstraction level.** Multiple distinct L1 nodes that share a sub-goal are lifted into one L2 step. L2 introduces information that does not exist at L1: a *goal description*. This is what makes the hierarchy load-bearing for `fine_grained_why_recognition` and `recipe_prep_localization`.

**L3 is the session-level goal.** One per video session in HD-EPIC; the unifying intent of all constituent L2 steps.

> **Note:** L4 routine nodes (multi-day patterns) are scoped as future work — HD-EPIC does not support strong claims about multi-day routines.

### 4.2 Record Schemas

**W record (one per 3s window):**
```
W = {
  window_idx, t_start, t_end, frames_path, frame_count,
  prompt_id, model_id,
  raw_output,                 # verbatim VLM string
  parsed: {verb, direct_object, hand, with, purpose, manner},
  parse_ok,
  inference_time_sec
}
```

**L1 node (consolidated micro-action):**
```
L1 = {
  l1_id, t_start, t_end, duration_sec,
  verb, direct_object, hand, with, purpose, manner,
  location,
  source_windows: [int],      # which W records merged into this L1
  consolidation_method,
  embedding: vector
}
```
*Why verb-centric matters:* WorldMM stores `[person, chops, onion]` — a triplet identical for "chop onion thinly" vs "chop onion roughly" vs "chop onion for salad". L1 stores purpose and manner, which is what `fine_grained_why_recognition` requires.

**L2 node (sub-goal / recipe step):**
```
L2 = {
  l2_id, description,
  start_time, end_time,
  goal,                       # the abstraction — does not exist at L1
  l1_children: [l1_id],
  embedding: vector
}
```
Maps directly to recipe step annotations in HD-EPIC (e.g., "Heat half the oil in a large frying pan, add bacon and cook").

**L3 node (session / task goal):**
```
L3 = {
  l3_id, recipe, video_id, date, duration,
  l2_children: [l2_id],
  ingredients: [...],
  embedding: vector
}
```

### 4.3 Graph Edges
- `CONTAINS(L2→L1)`: hierarchical grouping
- `CONTAINS(L3→L2)`: hierarchical grouping
- `NEXT(L1→L1)`: temporal ordering within step
- `PRECEDES(L2→L2)`: ordering of steps
- `INVOLVES(L1→entity)`: participating objects
- `UPDATES(L1→entity.state)`: object state changes

### 4.4 Construction Algorithms

**W → L1 (consolidation, *not* abstraction):**
1. Walk W records in temporal order.
2. Apply alias table to canonicalize (verb, direct_object) — e.g., cut→chop, grab→take. Alias table is built from HD-EPIC's verb/noun vocabulary BEFORE any predictions exist.
3. Merge consecutive W records that share the same canonical (verb, direct_object) AND `parse_ok=True`.
4. A run of N matching records becomes one L1 node with `t_start = first.t_start`, `t_end = last.t_end`.
5. Records with `parse_ok=False` break a run; they become singleton L1 nodes flagged `parse_failed=True`.

This step removes redundancy from sensor-rate sampling. It does not introduce new abstraction.

**L1 → L2 (lifting by shared sub-goal):**

Surface-level similarity is the wrong tool here. "Take onion" and "chop onion" are semantically dissimilar at the verb level despite belonging to the same recipe step. Earlier drafts proposed BERTScore for this — that is incorrect; BERTScore is a sentence-level evaluation metric for translation/summarization, not a goal-segmentation algorithm. Replaced with the following:

*Primary mechanism — LLM-based goal segmentation:*
Feed a temporally-ordered sequence of L1 descriptions to an LLM and ask it to insert step boundaries.
1. Format L1 sequence as `[idx] verb=X obj=Y purpose=Z` lines.
2. Process in overlapping chunks of ~50 L1 nodes to control context length; reconcile boundaries at chunk seams.
3. LLM outputs JSON: `[{step_id, start_idx, end_idx, step_description, goal}]`.
4. Steps must be contiguous; no reordering; typical step spans 5–30 L1 nodes.
5. Validate output: contiguity, full coverage of L1 indices, valid JSON.

*Secondary mechanism (ablation baseline) — change-point detection on L1 embeddings:*
1. Embed each L1 description.
2. Compute similarity between consecutive embeddings.
3. Detect local minima → step boundaries.
4. Used as the "no-LLM" comparison point in the ablation table to answer the reviewer question "is the LLM necessary for this step?"

**L2 → L3 (aggregation):**
1. One video session = one L3 node.
2. LLM summarizes all L2 descriptions → recipe name + episode description.
3. Extract ingredient list from L1 ADD-type actions across the session.

**Granularity calibration:** Before finalizing the L1→L2 algorithm, count actions per recipe step in HD-EPIC GT. Tune the L2 grouping to produce roughly that granularity. If HD-EPIC averages 8 L1 actions per step, the merger should target that — otherwise `recipe_step_localization` evaluation will be biased regardless of L1 quality.

**Why this beats WorldMM's "hierarchy":** WorldMM uses fixed time windows (30s, 3min, 10min, 1hr) — same content at different resolutions. Our L2 boundaries are determined by semantic content (same goal = same step), not clock time.

---

## 5. HD-EPIC Dataset and Evaluation

**Dataset stats:**
- 41 hours total, 9 kitchens, 9 participants (P01-P09)
- 69 recipes, 59K fine-grained actions, 263 annotations/minute
- Each video session: 15-30 minutes (one cooking session)
- Multiple sessions per participant across 2-9 days
- VQA benchmark: 26K questions across multiple categories
- Videos at: `/data2/yaxuanli/HD-EPIC/Videos/P01/` etc.

**Multi-session structure:** `recipe_recipe_recognition` questions use ~18 videos across ~3 days (~6 hours total) — this tests cross-session L3 aggregation.

### QA Category → Graph Level Mapping

| QA Category | Graph Level Tested | Can Answer? |
|---|---|---|
| `fine_grained_action_recognition` | L1 node quality | ✅ Core |
| `fine_grained_action_localization` | L1 timestamp | ✅ Core |
| `fine_grained_how_recognition` | L1 manner attribute | ✅ Core |
| `fine_grained_why_recognition` | **L1→L2 edge** (goal) | ✅ Hierarchy required |
| `recipe_prep_localization` | **L1→L2 edge** (prep) | ✅ Hierarchy required |
| `recipe_step_recognition` | L2 node quality | ✅ Core |
| `recipe_step_localization` | L2 temporal retrieval | ✅ Core |
| `recipe_rough_step_localization` | L2 retrieval (easier) | ✅ Core |
| `recipe_multi_step_localization` | Multi-L2 retrieval | ✅ Core |
| `recipe_following_activity_recognition` | **L2→L3 edge** | ✅ Hierarchy required |
| `recipe_multi_recipe_recognition` | L3 quality | ✅ Core |
| `recipe_recipe_recognition` | L3 cross-session | ✅ Core |
| `ingredient_ingredient_retrieval` | L1 temporal query | ✅ Core |
| `ingredient_ingredients_order` | L1 ordering | ✅ Core |
| `ingredient_ingredient_recognition` | L3 aggregation | ⚠️ Partial |
| `object_motion_object_movement_itinerary` | Object state | ⚠️ Partial |
| `nutrition_*` (all) | Nutrition DB needed | ❌ Out of scope |
| `object_motion_*_counting` | Frame-level tracking | ❌ Out of scope |
| `3d_perception_fixture_location` | 3D geometry | ❌ Out of scope |

**Key insight:** `fine_grained_why_recognition` and `recipe_following_activity_recognition` are the most important questions for proving hierarchy is necessary — flat graphs (WorldMM, EgoGraph) fundamentally cannot answer these.

---

## 6. Three Core Contributions

### Contribution 1: Action-Centric Hierarchical Graph (L1→L2→L3)

**What:** Replace entity-centric flat graphs with verb-primary hierarchical graph. Verb is the organizing primitive at all levels.

**Why it beats WorldMM:**
- WorldMM action triplet `[person, rubs, hands]` is identical for "rub hands to apply soap" vs "rub hands to remove oil" — loses purpose/manner
- L1 node stores purpose → directly answers `fine_grained_why_recognition`
- L2→L3 edges enable `recipe_following_activity_recognition` — WorldMM has no such structure
- L1→L2 edges enable `recipe_prep_localization` — flat graphs cannot distinguish prep from main step
- Content-driven semantic lifting vs WorldMM's dumb fixed time windows

**Cognitive science backing:** Zacks 2007, Tulving 1972, Baldassano 2017, Zacks & Tversky 2001

### Contribution 2: Budget-Aware Graph Compression

**What:** First graph memory system with storage budget constraint. Graph stays fixed size as video accumulates.

**Three mechanisms:**
1. **Salience-based node pruning:** Score each L1 node, prune lowest-salience when budget exceeded.
   - Salience = recency + query_frequency + uniqueness + connectivity
2. **Event folding:** Merge repetitive identical L1 actions (e.g., 5× `stir(pan)` → one node with duration).
3. **Hierarchical absorption:** Once L1 nodes are absorbed into L2, drop L1 nodes from device storage.

**Why no one else does this:** WorldMM, EgoGraph, AVA all grow unboundedly. Critical for mobile deployment.

**Cognitive science backing:** Miller 1956 (chunking), McClelland 1995 (selective consolidation), Bartlett 1932 (schema compression)

**Evaluation:** Graph size (MB) over time — flat line (ours) vs growing line (baselines). Accuracy vs graph size tradeoff curve.

### Contribution 3: Cloud-Mobile Delta Sync

**What:** Cloud-mobile co-design where heavy VLM graph construction runs in cloud every 15 minutes, only graph deltas transmitted to device, small model handles on-device QA.

**Architecture:**
```
Phone (Wearable Camera)
  → every 15 min: upload compressed keyframes + current graph hash + query log

Cloud (Unlimited GPU)
  → Big VLM (Qwen2.5-VL-72B): generate W → consolidate to L1
  → LLM-based lifting: L1→L2→L3
  → Budget-aware compression
  → Compute graph delta (only new/changed nodes)
  → Return: compressed delta (KBs, not MBs)

Phone (Updated)
  → Merge delta into local graph
  → Small model (Gemma small / Phi-3.5-mini): on-device QA
  → Query routing: simple→local, complex→wait for next sync
```

**Graph delta encoding:** Like Git for graphs.
```
delta = {
  nodes_added:   [new nodes with embeddings],
  nodes_deleted: [pruned node IDs],
  nodes_updated: [changed attribute nodes],
  edges_added:   [new edges],
  edges_deleted: [deleted edge IDs]
}
```

**Hardware target:** Jetson Orin Nano 8GB (~$250) as primary evaluation device. Also test on Raspberry Pi 5 8GB (~$80) for low-end comparison.

**Key result to show:** System answers queries in ~2-5 seconds on device (simple embedding lookup + small LLM). AVA's agentic search: 101-174 seconds — never feasible on mobile.

---

## 7. No Training Required (Training-Free System)

The system uses pre-trained models as components only:
- VLM for W generation: Gemma multimodal (current default) or Qwen2.5-VL-7B-AWQ
- LLM for L1→L2 lifting and summarization: Qwen2.5-32B (planned)
- Embeddings: JinaCLIP (planned)
- On-device QA: Gemma small / Phi-3.5-mini (planned)

**This is normal and acceptable for MobiSys/UbiComp.** WorldMM, EgoGraph, and AVA all train nothing. The contribution is the architecture and results.

**Optional training component (if needed):** Fine-tune a small VLM (Qwen2.5-VL-3B with LoRA) on EASG Ego4D dataset + HD-EPIC annotations to produce better structured W descriptions. Only pursue this if prompt-based generation quality is insufficient.

---

## 8. Positioning vs Competing Systems

| | WorldMM | EgoGraph | AVA | **This Project** |
|---|---|---|---|---|
| Primary node | Entity | Entity | Event | **Action (verb)** |
| Hierarchy | Fixed windows | ❌ Flat | ❌ Flat | **✅ L1→L2→L3** |
| Semantic lifting | ❌ | ❌ | ❌ | **✅ Content-driven** |
| Graph compression | ❌ | ❌ | ❌ | **✅ Budget-aware** |
| Mobile deployment | ❌ | ❌ | ❌ | **✅ Cloud-mobile** |
| On-device QA | ❌ | ❌ | ❌ | **✅ Jetson + small LLM** |
| Delta sync | ❌ | ❌ | ❌ | **✅ Graph delta** |
| Training needed | ❌ | ❌ | ❌ | **❌ (optional)** |
| HD-EPIC evaluation | ❌ | ❌ | ❌ | **✅** |

**Key differentiator from WorldMM specifically:**
- WorldMM's "hierarchy" = same content at 30s/3min/10min/1hr windows (not semantic)
- No graph compression — grows unboundedly
- No mobile deployment — cloud-only GPT-5
- Entity-centric triplets lose action attributes

---

## 9. Implementation Plan

**Base code:** AVA codebase at `/data/yaxuanli/AVA/` (read-only). All new code, logs, and outputs under `/data2/yaxuanli/hmgraph_codex/`.

**Changes from AVA:**

| AVA Component | Our Change |
|---|---|
| Generic description prompt | Action-focused structured prompt (verb/object/hand/purpose/manner) |
| Entity-centric graph nodes | Verb-centric L1 nodes |
| Flat graph structure | W → L1 → L2 → L3 pipeline (3 conceptual levels + 1 raw layer) |
| BERTScore semantic chunking | LLM-based goal segmentation for L1→L2; change-point detection as ablation |
| Vector DB storage | Keep + add hierarchy table |
| Tri-view retrieval (events/entities/frames) | Replace with level-aware hierarchical routing |
| Agentic tree search (forward/backward) | Replace with simple embedding search |
| No compression | Add salience pruning + event folding |
| No mobile | Add Jetson deployment + delta sync |

**Retrieval system (simple, not AVA's complex tree search):**
```python
def answer(question, graph):
    level = route_query(question)  # L1 / L2 / L1→L2 / L2→L3 / L3
    if level == "L1":       return lookup_by_timestamp(question, graph)
    if level == "L2":       return embed_search_l2(question, graph)
    if level == "L1→L2":    return traverse_up_l1_to_l2(question, graph)
    if level == "L2→L3":    return traverse_up_l2_to_l3(question, graph)
    if level == "L3":       return aggregate_l3_nodes(question, graph)
```

**Why simpler retrieval is a contribution:** 2-5 seconds vs AVA's 101-174 seconds. Enables mobile deployment.

### Code Layout (under /data2/yaxuanli/hmgraph_codex/)

```
/data2/yaxuanli/hmgraph_codex/
  project_notes.md
  code/
    __init__.py
    paths.py                      # central path constants
    hdepic_loader.py              # GT annotation loader for one video
    window_extractor.py           # 3s window slicing + frame sampling
    gemma_vlm.py                  # Gemma loading + inference
    l1_prompt.py                  # exact prompt + parser, with PROMPT_ID
    alias_table.py                # build verb/noun aliases from HD-EPIC vocab
    build_w.py                    # MAIN: produce W (raw window) outputs
    build_l1.py                   # MAIN: consolidate W → L1
    build_l2.py                   # MAIN (Week 2): lift L1 → L2
    build_l3.py                   # MAIN (Week 2): aggregate L2 → L3
    evaluate_l1.py                # MAIN: match L1 to GT, compute metrics
    evaluate_l2.py                # MAIN (Week 2): match L2 to recipe steps
    run_week1.sh                  # end-to-end driver
  week1_validation/
    P01-20240202-110250/
      frames/win_NNNNN/*.jpg
      w_outputs.jsonl             # raw window outputs (preserve verbatim)
      l1_graph.jsonl              # consolidated micro-actions
      l1_consolidation_log.jsonl  # which W's merged into which L1, with reason
      gt.jsonl
      matches_l1.csv
      per_l1_results.csv
      summary.json
      alias_table.json
      config.json
      run.log
```

**Rules:**
- `/data/yaxuanli/AVA/` is read-only. All new code under `/data2/yaxuanli/hmgraph_codex/code/`.
- W must be saved verbatim — never overwrite during consolidation.
- Every script: argparse CLI, no hardcoded paths inside functions, importable.
- Stamp `prompt_id` and `model_id` on every W record so prompt drift is traceable.

---

## 10. Ablation Study Design

Run all variants on same HD-EPIC QA subset:

| System | Description |
|---|---|
| WorldMM (baseline) | Entity-centric, fixed time windows |
| Flat action-centric | Verb-centric nodes, no hierarchy |
| L1+L2 only | No L3 episode nodes |
| Full L1+L2+L3 | Complete system (no compression) |
| Full + compression | Complete system with budget-aware compression |
| Full + compression + mobile | Full system on Jetson Orin Nano |
| L1→L2 with LLM segmentation | Primary lifting mechanism |
| L1→L2 with change-point only | "Is the LLM necessary?" ablation |

**Key ablation results to show:**
- `fine_grained_why_recognition` fails without L1→L2 edges
- `recipe_following_activity` fails without L2→L3 edges
- LLM-based lifting beats change-point-only lifting on `recipe_step_recognition`
- All three prove hierarchy + LLM lifting is necessary, not nice-to-have

---

## 11. Evaluation Metrics

**QA accuracy:** On HD-EPIC benchmark, by question category (L1, L2, cross-level, L3)

**L1 quality (intermediate):**
- verb_exact_match, verb_semantic_match (alias-aware)
- noun_exact_match, noun_semantic_match
- joint (verb AND noun) semantic match
- hand_match (where GT hand present)
- pure-window subset: same metrics restricted to L1 nodes whose max-overlap fraction with matched GT action ≥ 0.8

**L2 quality (intermediate):**
- step boundary IoU vs GT recipe step boundaries
- step description embedding similarity to GT step description
- L1 → L2 assignment accuracy (does each L1 belong to the right L2?)

**Graph quality:**
- Graph size (MB) over time: compression keeps it flat
- Node count over time
- Accuracy vs graph size tradeoff

**Mobile deployment:**
- Query latency (seconds) on Jetson Orin Nano 8GB
- Peak RAM usage (MB)
- Power consumption (Watts) — use tegrastats
- Tokens per second for small model
- Bandwidth per 15-min sync (KB)

**Comparison devices:**
- Raspberry Pi 5 8GB (~$80): low-end baseline
- Jetson Orin Nano 8GB (~$250): main evaluation
- Cloud (RTX 6000): upper bound

---

## 12. Prototype Strategy (Fast Path to Results)

**Do not build infrastructure first. Get one accuracy number first.**

### Week 1: Validate W → L1 quality on ONE full video

**Method:** Fixed-window segmentation (no GT boundaries used at inference time). End-to-end pipeline on a single full video.

1. Target video: `/data2/yaxuanli/HD-EPIC/Videos/P01/P01-20240202-110250.mp4`
2. Slice into fixed non-overlapping 3-second windows.
3. Run Gemma multimodal on each → save W records verbatim (`w_outputs.jsonl`).
4. Consolidate adjacent W records with same canonicalized (verb, object) → `l1_graph.jsonl`.
5. Match each L1 node to GT action via max-overlap; apply alias table; compute accuracy.

**Pass criterion:**
- max-overlap joint verb+noun semantic match (full set)
- pure-window joint verb+noun semantic match (L1 nodes with overlap ≥ 0.8)
- If pure-window ≥ 60% → VLM/prompt is doing its job; segmentation noise is the next thing to address. PROCEED.
- Below 60% → VLM/prompt is the bottleneck. Stop and revise prompt before scaling.

**Why fixed-window, not GT boundaries:** the real system has no GT at runtime. Using GT boundaries to extract clips for evaluation would measure VLM quality under oracle segmentation — useful as a separate diagnostic but not as the headline result.

### Week 2-3: L1 → L2 → L3 + first end-to-end QA

- Full pipeline on 5 HD-EPIC videos
- Implement L1 → L2 (LLM goal segmentation) and L2 → L3 (aggregation)
- Simple embedding retrieval
- Evaluate on 50 HD-EPIC questions from key categories
- Compare: our graph vs generic VLM vs flat-graph baseline

### Week 4-6: Full pipeline
- Proper vector DB
- All QA categories
- Full baseline comparisons (WorldMM-style, AVA, EgoGraph reproduction)

### Week 7-8: Compression + mobile
- Implement salience scoring + event folding
- Deploy on Jetson Orin Nano
- Measure latency, memory, power

### Week 9-10: Complete evaluation + ablations

---

## 13. Paper Contribution Statement

> *"We propose an action-centric hierarchical memory graph that computationally instantiates the known structure of human episodic memory. Unlike existing systems that use entity-centric flat graphs, our verb-primary L1→L2→L3 hierarchy enables cross-level reasoning (answering 'why' and 'high-level activity' questions) that flat graphs fundamentally cannot perform. We further introduce the first budget-aware graph compression mechanism for video memory systems, maintaining a fixed storage footprint as video accumulates. Finally, we design a cloud-mobile co-processing architecture with 15-minute delta sync that enables accurate on-device QA using a 2B parameter model on a Jetson Orin Nano, achieving 2-5 second query latency — 20-50x faster than cloud-only approaches."*

---

## 14. Current Status

- ✅ Research design complete
- ✅ Dataset identified (HD-EPIC)
- ✅ QA categories mapped to graph levels
- ✅ Related work analyzed (WorldMM, EgoGraph, AVA, EASG)
- ✅ Base codebase identified (AVA at `/data/yaxuanli/AVA/`, read-only)
- ✅ Pipeline structure formalized (W → L1 → L2 → L3)
- ✅ L1→L2 lifting algorithm designed (LLM segmentation primary, change-point ablation)
- 🔄 **In progress:** Week 1 — W generation + W→L1 consolidation + L1 evaluation on one full video
- ⬜ L1→L2 lifting (Week 2-3)
- ⬜ L3 episode aggregation
- ⬜ Retrieval system
- ⬜ End-to-end QA evaluation
- ⬜ Graph compression
- ⬜ Cloud-mobile sync
- ⬜ Jetson deployment

---

## 15. Open Design Questions (to revisit)

1. **L2 granularity calibration.** Need to count actions per recipe step in HD-EPIC GT before fixing the L1→L2 algorithm hyperparameters. If our merger produces 30 L1 per L2 and HD-EPIC has 8, `recipe_step_localization` will look bad regardless of L1 quality.
2. **W parse failure handling.** Currently breaks consolidation runs and creates singleton "parse_failed" L1 nodes. May want to retry with a fallback prompt instead.
3. **"None" / background windows.** Some 3s windows have no salient action (idle, walking). Currently the VLM still emits a verb. Worth deciding whether to add a "none" option to the W prompt or filter post-hoc.
4. **Model choice for W generation.** Currently Gemma multimodal in `avagemma`. If Week 1 fails the pure-window 60% threshold, the next experiment is whether Qwen2.5-VL-7B-AWQ in `ava` does better with the same prompt.
5. **Stride.** Currently stride = window = 3s (no overlap). Overlapping windows could help with action-boundary cases but multiply compute. Defer until we see a failure mode that overlap would fix.
