"""GPU-only neural model adapters for dense retrieval and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F


class TextEmbedder(Protocol):
    model_id: str

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor: ...

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor: ...


class TextReranker(Protocol):
    model_id: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def require_cuda(device: str = "cuda") -> torch.device:
    """Enforce the project's no-neural-CPU-inference policy."""

    if not device.startswith("cuda"):
        raise ValueError(f"Neural graph QA is GPU-only; received device={device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Neural graph QA requires CUDA, but torch.cuda.is_available() is false")
    resolved = torch.device(device)
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {resolved.index} is unavailable")
    return resolved


def _model_location(model_id_or_path: str, local_files_only: bool) -> str | Path:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    if local_files_only:
        # Transformers can still resolve a cached hub id when local-only.
        return model_id_or_path
    return model_id_or_path


@dataclass(frozen=True)
class EmbedderConfig:
    model_id: str
    revision: str | None = None
    pooling: str = "mean"
    query_instruction: str | None = None
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True


class TransformersTextEmbedder:
    """Direct Transformers embedding wrapper (no sentence-transformers dependency)."""

    def __init__(self, config: EmbedderConfig) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.model_id = config.model_id
        self.device = require_cuda(config.device)
        location = _model_location(config.model_id, config.local_files_only)
        revision_kwargs = {"revision": config.revision} if config.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            location,
            local_files_only=config.local_files_only,
            padding_side="left" if config.pooling == "last" else "right",
            **revision_kwargs,
        )
        dtype = getattr(torch, config.dtype)
        self.model = AutoModel.from_pretrained(
            location,
            local_files_only=config.local_files_only,
            dtype=dtype,
            **revision_kwargs,
        ).to(self.device)
        self.model.eval()

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        instruction = self.config.query_instruction
        prepared = [f"Instruct: {instruction}\nQuery: {text}" for text in texts] if instruction else list(texts)
        return self._encode(prepared)

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode(list(texts))

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.config.batch_size):
            inputs = self.tokenizer(
                texts[start : start + self.config.batch_size],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                if self.config.pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
                elif self.config.pooling == "last":
                    if bool((inputs["attention_mask"][:, -1] == 1).all()):
                        embeddings = hidden[:, -1]
                    else:
                        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
                        embeddings = hidden[
                            torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths
                        ]
                else:
                    raise ValueError(f"Unsupported embedding pooling: {self.config.pooling}")
                embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            batches.append(embeddings.cpu())
        return torch.cat(batches, dim=0)


@dataclass(frozen=True)
class RerankerConfig:
    model_id: str
    revision: str | None = None
    instruction: str = (
        "Given a video-memory question, decide whether the graph evidence is useful for answering it."
    )
    max_length: int = 2048
    batch_size: int = 8
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True


class Qwen3Reranker:
    """Official Qwen3 yes/no logit reranking recipe, implemented directly."""

    def __init__(self, config: RerankerConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.model_id = config.model_id
        self.device = require_cuda(config.device)
        location = _model_location(config.model_id, config.local_files_only)
        revision_kwargs = {"revision": config.revision} if config.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            location,
            local_files_only=config.local_files_only,
            padding_side="left",
            **revision_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            location,
            local_files_only=config.local_files_only,
            dtype=getattr(torch, config.dtype),
            **revision_kwargs,
        ).to(self.device)
        self.model.eval()
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
            'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), self.config.batch_size):
            pairs = [
                f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {document}"
                for document in documents[start : start + self.config.batch_size]
            ]
            tokenized = self.tokenizer(
                pairs,
                padding=False,
                truncation=True,
                max_length=self.config.max_length - len(self.prefix_ids) - len(self.suffix_ids),
            )
            for index, input_ids in enumerate(tokenized["input_ids"]):
                tokenized["input_ids"][index] = self.prefix_ids + input_ids + self.suffix_ids
            inputs = self.tokenizer.pad(tokenized, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, [self.false_id, self.true_id]].float()
                probability = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probability.cpu().tolist())
        return scores


class IdentityReranker:
    """Explicit no-neural-reranker option for tests and retrieval ablations."""

    model_id = "identity"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        del query
        return [0.0 for _ in documents]
