"""Evaluate HD-EPIC multiple-choice questions using W and/or grouped L1 evidence.

Ground-truth labels are loaded only after generation for scoring. They are never
included in retrieval queries or model prompts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


VIDEO_ID = "P01-20240202-110250"
LETTERS = "ABCDE"
TIME_RE = re.compile(r"<TIME\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s+video\s+\d+>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hms_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def time_token_to_seconds(match: re.Match[str]) -> float:
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def compact_action(action: dict[str, Any]) -> str:
    obj = action.get("direct_object", "unknown")
    if isinstance(obj, dict):
        obj = obj.get("name", "unknown")
    interaction = action.get("interaction") or {}
    spatial = action.get("spatial_context") or {}
    parts = [f"{action.get('verb', 'unknown')} {obj}"]
    hand = action.get("hand") or interaction.get("hand")
    if hand and hand != "unknown":
        parts.append(f"hand={hand}")
    location = spatial.get("location")
    relative = spatial.get("relative_position")
    if location and location != "unknown":
        parts.append(f"location={location}")
    if relative and relative != "unknown":
        parts.append(f"position={relative}")
    purpose = action.get("purpose")
    if purpose and purpose != "unknown":
        parts.append(f"purpose={purpose}")
    change = interaction.get("object_change")
    if change and change not in {"unknown", "no_visible_change"}:
        parts.append(f"change={change}")
    direction = interaction.get("motion_direction")
    if direction and direction not in {"unknown", "none"}:
        parts.append(f"motion={direction}")
    return ", ".join(parts)


def make_w_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for record in records:
        start = float(record.get("t_start", record.get("window_idx", 0)))
        end = float(record.get("t_end", start + 1))
        parsed = record.get("parsed") or {}
        actions = parsed.get("actions") or [] if isinstance(parsed, dict) else []
        if actions:
            text = "; ".join(compact_action(action) for action in actions)
        else:
            text = f"{parsed.get('verb', 'none')} {parsed.get('direct_object', 'none')}"
        documents.append({"start": start, "end": end, "text": text})
    return documents


def make_l1_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for record in records:
        obj = record.get("direct_object", "unknown")
        if isinstance(obj, dict):
            obj = obj.get("name", "unknown")
        text = f"{record.get('verb', 'unknown')} {obj}"
        hand = record.get("hand")
        if hand and hand != "unknown":
            text += f", hand={hand}"
        description = record.get("micro_action_description")
        if description:
            text += f", description={description}"
        documents.append({
            "start": float(record.get("t_start", 0)),
            "end": float(record.get("t_end", 0)),
            "text": text,
        })
    return documents


def load_questions(annotation_root: Path) -> list[dict[str, Any]]:
    questions = []
    for path in sorted(annotation_root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for question_id, question in data.items():
            ids = [
                value.get("id")
                for value in question.get("inputs", {}).values()
                if isinstance(value, dict)
            ]
            if ids and set(ids) == {VIDEO_ID}:
                questions.append({"category": path.stem, "question_id": question_id, **question})
    return questions


def retrieval_query(question: dict[str, Any]) -> str:
    # Choices are intentionally included: they often contain the discriminating
    # action/object words needed to retrieve evidence for generic question text.
    return question["question"] + " " + " ".join(question["choices"])


def question_range(question: dict[str, Any]) -> tuple[float, float] | None:
    inputs = list(question.get("inputs", {}).values())
    if inputs and inputs[0].get("start_time") and inputs[0].get("end_time"):
        return hms_to_seconds(inputs[0]["start_time"]), hms_to_seconds(inputs[0]["end_time"])
    return None


def retrieve(documents: list[dict[str, Any]], question: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    time_range = question_range(question)
    pool = documents
    if time_range:
        start, end = time_range
        pool = [doc for doc in documents if doc["end"] >= start - 2 and doc["start"] <= end + 2]
    if not pool:
        return []

    query = retrieval_query(question)
    corpus = [doc["text"] for doc in pool] + [query]
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(corpus)
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()

    # For temporal multiple-choice questions, guarantee evidence around every
    # proposed answer interval in addition to semantic top-k retrieval.
    selected: set[int] = set(scores.argsort()[::-1][:top_k].tolist())
    for choice in question.get("choices", []):
        times = [time_token_to_seconds(match) for match in TIME_RE.finditer(choice)]
        if times:
            low, high = min(times), max(times)
            overlaps = [
                (idx, score)
                for idx, (doc, score) in enumerate(zip(pool, scores))
                if doc["end"] >= low - 1 and doc["start"] <= high + 1
            ]
            selected.update(idx for idx, _ in sorted(overlaps, key=lambda item: item[1], reverse=True)[:3])

    ranked = sorted(selected, key=lambda idx: (pool[idx]["start"], pool[idx]["end"]))
    return [{**pool[idx], "score": float(scores[idx])} for idx in ranked]


def format_prompt(question: dict[str, Any], evidence: list[dict[str, Any]], source: str) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(question["choices"]))
    evidence_text = "\n".join(
        f"[{doc['start']:.1f}-{doc['end']:.1f}s] {doc['text']}" for doc in evidence
    ) or "No matching graph evidence was retrieved."
    return f"""You answer a multiple-choice question using only graph evidence from an egocentric video.
Evidence source: {source}.
Do not use outside knowledge. Timestamps in the choices are seconds/minutes in the same video.
If evidence is incomplete, choose the option best supported by it.

QUESTION:
{question['question']}

CHOICES:
{choices}

GRAPH EVIDENCE:
{evidence_text}

Return exactly one capital letter: A, B, C, D, or E."""


def parse_answer(text: str) -> int | None:
    match = re.search(r"\b([A-E])\b", text.upper())
    return LETTERS.index(match.group(1)) if match else None


def summarize(results: list[dict[str, Any]], output_path: Path) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_source[result["source"]].append(result)
    lines = ["# W/L1 QA Evaluation", "", f"Questions: {len(results) // 3}", ""]
    lines += ["| Source | Correct | Parsed | Accuracy |", "|---|---:|---:|---:|"]
    for source in ("w", "l1", "w+l1"):
        rows = by_source[source]
        parsed = sum(row["predicted_idx"] is not None for row in rows)
        correct = sum(row["correct"] for row in rows)
        lines.append(f"| {source} | {correct}/{len(rows)} | {parsed}/{len(rows)} | {correct/len(rows):.3f} |")
    lines += ["", "## Accuracy by category", "", "| Category | W | L1 | W+L1 |", "|---|---:|---:|---:|"]
    categories = sorted({row["category"] for row in results})
    for category in categories:
        cells = []
        for source in ("w", "l1", "w+l1"):
            rows = [row for row in by_source[source] if row["category"] == category]
            cells.append(f"{sum(row['correct'] for row in rows)}/{len(rows)}")
        lines.append(f"| {category} | {' | '.join(cells)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--w-jsonl", type=Path, required=True)
    parser.add_argument("--l1-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--answer-mode", choices=("generate", "choice_logits"), default="generate")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.annotation_root)
    if args.limit is not None:
        questions = questions[: args.limit]
    if args.limit is None and len(questions) != 78:
        raise RuntimeError(f"Expected 78 single-video questions, found {len(questions)}")

    w_docs = make_w_documents(read_jsonl(args.w_jsonl))
    l1_docs = make_l1_documents(read_jsonl(args.l1_jsonl))
    sources = {"w": w_docs, "l1": l1_docs, "w+l1": w_docs + l1_docs}

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU-only evaluation requested, but CUDA is unavailable")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    choice_token_ids = []
    if args.answer_mode == "choice_logits":
        for letter in LETTERS:
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Choice {letter} does not map to one token: {token_ids}")
            choice_token_ids.append(token_ids[0])
    jobs = []
    for question in questions:
        for source, documents in sources.items():
            evidence = retrieve(documents, question, args.top_k)
            prompt = format_prompt(question, evidence, source)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            jobs.append((question, source, evidence, text))

    results = []
    for batch_start in range(0, len(jobs), args.batch_size):
        batch = jobs[batch_start : batch_start + args.batch_size]
        inputs = tokenizer([job[3] for job in batch], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            if args.answer_mode == "choice_logits":
                logits = model(**inputs).logits[:, -1, choice_token_ids]
                predicted_letters = logits.argmax(dim=-1).tolist()
                responses = [LETTERS[index] for index in predicted_letters]
            else:
                outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
                responses = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for (question, source, evidence, _), response in zip(batch, responses):
            response = response.strip()
            predicted = parse_answer(response)
            ground_truth = int(question["correct_idx"])
            result = {
                "question_id": question["question_id"],
                "category": question["category"],
                "source": source,
                "question": question["question"],
                "choices": question["choices"],
                "predicted_idx": predicted,
                "predicted_letter": LETTERS[predicted] if predicted is not None else None,
                "ground_truth_idx": ground_truth,
                "ground_truth_letter": LETTERS[ground_truth],
                "correct": predicted == ground_truth,
                "raw_response": response,
                "evidence": evidence,
            }
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {question['question_id']} {source}: {response!r}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "predictions.jsonl"
    output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    summarize(results, args.output_dir / "summary.md")
    config = {
        "video_id": VIDEO_ID,
        "question_count": len(questions),
        "conditions": list(sources),
        "model_path": str(args.model_path),
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "device": args.device,
        "answer_mode": args.answer_mode,
        "ground_truth_prompt_leakage": False,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
