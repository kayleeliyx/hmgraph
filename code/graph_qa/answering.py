"""Evidence-grounded answering for graph QA."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch

from .models import require_cuda
from .program import QuestionProgram
from .retrieval import RetrievalTrace


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class AnswerResult:
    predicted_index: int | None
    predicted_label: str | None
    raw_response: str
    choice_scores: tuple[float, ...]
    prompt: str

    def to_dict(self, include_prompt: bool = False) -> dict[str, Any]:
        result = {
            "predicted_index": self.predicted_index,
            "predicted_label": self.predicted_label,
            "raw_response": self.raw_response,
            "choice_scores": list(self.choice_scores),
        }
        if include_prompt:
            result["prompt"] = self.prompt
        return result


class MultipleChoiceAnswerer(Protocol):
    model_id: str

    def answer(
        self,
        *,
        program: QuestionProgram,
        choices: Sequence[str],
        retrieval: RetrievalTrace,
    ) -> AnswerResult: ...


@dataclass(frozen=True)
class AnswererConfig:
    model_id: str
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    max_evidence_characters: int = 18000
    trust_remote_code: bool = True


class QwenChoiceAnswerer:
    """GPU-only Qwen answerer using constrained A..Z next-token scores."""

    def __init__(self, config: AnswererConfig) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.config = config
        self.model_id = config.model_id
        self.device = require_cuda(config.device)
        location: str | Path = Path(config.model_id).expanduser()
        if not Path(location).exists():
            location = config.model_id
        revision_kwargs = {"revision": config.revision} if config.revision else {}
        model_config = AutoConfig.from_pretrained(
            location,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            **revision_kwargs,
        )
        self.is_qwen3_vl = str(model_config.model_type).startswith("qwen3_vl")
        dtype = getattr(torch, config.dtype)
        if self.is_qwen3_vl:
            from transformers import Qwen3VLMoeForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                location,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                **revision_kwargs,
            )
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                location,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
                device_map="auto",
                **revision_kwargs,
            )
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                location,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                padding_side="left",
                **revision_kwargs,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                location,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
                **revision_kwargs,
            ).to(self.device)
        self.model.eval()

    def answer(
        self,
        *,
        program: QuestionProgram,
        choices: Sequence[str],
        retrieval: RetrievalTrace,
    ) -> AnswerResult:
        if not choices:
            raise ValueError("Multiple-choice answerer received no choices")
        if len(choices) > len(LETTERS):
            raise ValueError(f"Too many choices: {len(choices)}")
        prompt = format_answer_prompt(program, choices, retrieval, self.config.max_evidence_characters)
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied graph evidence. Do not invent visual facts that are not "
                    "represented in the graph. Return exactly one answer letter."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if self.is_qwen3_vl:
            content_messages = [
                {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
                for message in messages
            ]
            inputs = self.processor.apply_chat_template(
                content_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            inputs = encoded
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        labels = LETTERS[: len(choices)]
        token_ids: list[int] = []
        for label in labels:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Answer label {label!r} is not one token for {self.model_id}: {ids}")
            token_ids.append(ids[0])
        with torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1, token_ids].float()[0]
            probabilities = torch.softmax(logits, dim=0)
        predicted_index = int(probabilities.argmax().item())
        predicted_label = labels[predicted_index]
        return AnswerResult(
            predicted_index=predicted_index,
            predicted_label=predicted_label,
            raw_response=predicted_label,
            choice_scores=tuple(float(value) for value in probabilities.cpu()),
            prompt=prompt,
        )


def format_answer_prompt(
    program: QuestionProgram,
    choices: Sequence[str],
    retrieval: RetrievalTrace,
    max_evidence_characters: int,
) -> str:
    labels = LETTERS[: len(choices)]
    choice_text = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    evidence_lines: list[str] = []
    for rank, item in enumerate(retrieval.evidence, start=1):
        time = (
            f"[{item.span.start:.2f}-{item.span.end:.2f}s]"
            if item.span is not None
            else "[global]"
        )
        line = f"E{rank} score={item.score:.4f} {time} {item.kind}: {item.text}"
        if sum(len(existing) for existing in evidence_lines) + len(line) > max_evidence_characters:
            break
        evidence_lines.append(line)
    evidence_text = "\n".join(evidence_lines) or "No graph evidence was retrieved."
    warning_text = "\n".join(f"- {warning}" for warning in retrieval.warnings) or "- none"
    aggregate_text = (
        json.dumps(retrieval.aggregates, ensure_ascii=False, indent=2)
        if retrieval.aggregates
        else "No derived operator result."
    )
    return f"""Question type: {program.intent} / {program.answer_type}
Question: {program.question}

Choices:
{choice_text}

Retrieved graph evidence:
{evidence_text}

Derived graph-operator results:
{aggregate_text}

Known retrieval limitations:
{warning_text}

Reason internally about event identity, time, hierarchy, state, order, and count as relevant. Then return exactly one letter from {labels}."""


def parse_answer_label(text: str, number_of_choices: int) -> int | None:
    allowed = LETTERS[:number_of_choices]
    match = re.search(rf"\b([{allowed}])\b", text.upper())
    return allowed.index(match.group(1)) if match else None
