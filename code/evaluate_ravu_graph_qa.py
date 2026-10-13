"""Evaluate one fixed RAVU-style backend across W/L1 graph builds.

Retrieval receives only the question and its benchmark-provided input scope.
Answer choices are introduced after graph evidence has been retrieved. Ground
truth is used only after prediction for scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from code.graph_qa.adapters import GraphMode, build_temporal_graph
from code.graph_qa.answering import AnswererConfig, QwenChoiceAnswerer
from code.graph_qa.io import append_jsonl, load_hdepic_questions, read_jsonl
from code.graph_qa.models import (
    EmbedderConfig,
    Qwen3Reranker,
    RerankerConfig,
    TransformersTextEmbedder,
)
from code.graph_qa.pipeline import GraphQAPipeline
from code.graph_qa.program import RuleBasedProgramPlanner
from code.graph_qa.retrieval import CompositionalGraphRetriever, RetrievalConfig


def _dataclass_config(cls: type, value: dict[str, Any], **overrides: Any) -> Any:
    return cls(**{**value, **{key: item for key, item in overrides.items() if item is not None}})


def summarize(rows: list[dict[str, Any]], path: Path) -> None:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    lines = ["# RAVU-style HMGraph QA", "", f"Predictions: {len(rows)}", ""]
    lines.extend(["| Graph | Correct | Accuracy |", "|---|---:|---:|"])
    for mode in sorted(by_mode):
        group = by_mode[mode]
        correct = sum(bool(row["correct"]) for row in group)
        lines.append(f"| {mode} | {correct}/{len(group)} | {correct / len(group):.3f} |")
    lines.extend(["", "## By category", ""])
    categories = sorted({row["category"] for row in rows})
    modes = sorted(by_mode)
    lines.append("| Category | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---:|" * len(modes))
    for category in categories:
        cells: list[str] = []
        for mode in modes:
            group = [row for row in by_mode[mode] if row["category"] == category]
            correct = sum(bool(row["correct"]) for row in group)
            cells.append(f"{correct}/{len(group)}" if group else "-")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--w-jsonl", type=Path, required=True)
    parser.add_argument("--l1-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("w", "l1", "w+l1"), default=("w", "l1", "w+l1"))
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Evaluate only this question id; repeat to select more than one.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--embedder-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--answerer-device", default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    video_id = str(config["video_id"])
    questions = load_hdepic_questions(args.annotation_root, video_id=video_id)
    if args.question_id:
        requested_ids = set(args.question_id)
        questions = [question for question in questions if question["question_id"] in requested_ids]
        missing = requested_ids - {question["question_id"] for question in questions}
        if missing:
            raise RuntimeError(f"Requested question ids were not found for {video_id}: {sorted(missing)}")
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError(f"No questions found for {video_id}")
    w_records = read_jsonl(args.w_jsonl)
    l1_records = read_jsonl(args.l1_jsonl)
    graphs = {
        mode: build_temporal_graph(
            video_id=video_id,
            mode=mode,
            w_records=w_records,
            l1_records=l1_records,
        )
        for mode in args.modes
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    if prediction_path.exists() and args.overwrite:
        prediction_path.unlink()
    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    completed = {(row["question_id"], row["mode"]) for row in existing}

    embedder = TransformersTextEmbedder(
        _dataclass_config(EmbedderConfig, config["embedder"], device=args.embedder_device)
    )
    reranker = Qwen3Reranker(
        _dataclass_config(RerankerConfig, config["reranker"], device=args.reranker_device)
    )
    answerer = QwenChoiceAnswerer(
        _dataclass_config(AnswererConfig, config["answerer"], device=args.answerer_device)
    )
    retriever = CompositionalGraphRetriever(
        embedder=embedder,
        reranker=reranker,
        config=_dataclass_config(RetrievalConfig, config["retrieval"]),
    )
    pipeline = GraphQAPipeline(
        retriever=retriever,
        answerer=answerer,
        planner=RuleBasedProgramPlanner(),
    )

    total = len(questions) * len(graphs)
    finished = len(completed)
    for question in questions:
        for mode, graph in graphs.items():
            key = (question["question_id"], mode)
            if key in completed:
                continue
            # The pipeline sees neither correct_idx nor any correctness field.
            result = pipeline.answer_question(
                graph=graph,
                question_id=question["question_id"],
                question=question["question"],
                choices=question["choices"],
                inputs=question.get("inputs") or {},
                category=question["category"],
            )
            prediction = result.to_dict(include_prompt=args.save_prompts)
            prediction.update(
                {
                    "mode": mode,
                    "question": question["question"],
                    "choices": question["choices"],
                    "ground_truth_index": int(question["correct_idx"]),
                    "correct": result.answer.predicted_index == int(question["correct_idx"]),
                }
            )
            append_jsonl(prediction_path, prediction)
            existing.append(prediction)
            finished += 1
            print(
                f"[{finished}/{total}] {question['question_id']} {mode}: "
                f"{result.answer.predicted_label} correct={prediction['correct']}",
                flush=True,
            )

    summarize(existing, args.output_dir / "summary.md")
    manifest = {
        "protocol": "ravu_style_hmgraph_v1",
        "config": config,
        "config_path": str(args.config.resolve()),
        "annotation_root": str(args.annotation_root.resolve()),
        "w_jsonl": str(args.w_jsonl.resolve()),
        "l1_jsonl": str(args.l1_jsonl.resolve()),
        "modes": list(graphs),
        "question_count": len(questions),
        "retrieval_receives_choices": False,
        "ground_truth_used_before_prediction": False,
        "neural_cpu_fallback": False,
        "graph_stats": {
            mode: {"nodes": len(graph.nodes), "edges": len(graph.edges)} for mode, graph in graphs.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
