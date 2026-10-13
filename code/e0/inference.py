"""Vision-language API inference and resumable answer caching for E0."""

from __future__ import annotations

import ast
import base64
import json
import mimetypes
import os
import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import quote

import requests

from . import LETTERS, PROMPT_VERSION, SELF_KNOWLEDGE_PROMPT_VERSION
from .dataset import atomic_write_json, atomic_write_jsonl, stable_hash
from .prompts import ANSWER_INSTRUCTIONS, answer_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return None


def _redacted_error(exc: BaseException) -> str:
    text = f"{exc.__class__.__name__}: {exc}"
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", text)
    return text[:1200]


class ResponsesClient(Protocol):
    def generate_facts(
        self, prompt: str, model: str, settings: Mapping[str, Any]
    ) -> tuple[list[str], dict[str, Any]]: ...

    def answer(self, plan: Mapping[str, Any]) -> dict[str, Any]: ...


class OpenAIResponsesClient:
    def __init__(self, api_key: str, timeout: float, max_retries: int) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.max_retries = max_retries

    def _with_retry(self, operation: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if not retryable_api_error(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def generate_facts(
        self, prompt: str, model: str, settings: Mapping[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        requested_at = utc_now()

        def operation() -> Any:
            return self.client.responses.create(
                model=model,
                instructions=(
                    "Provide concise, general kitchen background facts. Do not infer anything "
                    "about a particular person, video, episode, or answer choice."
                ),
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "e0_background_facts",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "facts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 5,
                                }
                            },
                            "required": ["facts"],
                            "additionalProperties": False,
                        },
                    },
                    "verbosity": "low",
                },
                reasoning={"effort": settings["reasoning_effort"]},
                max_output_tokens=int(settings["max_output_tokens"]),
                store=False,
                metadata={"experiment": "e0", "operation": "self_knowledge"},
            )

        response = self._with_retry(operation)
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(
                f"self-knowledge response status={response.status}, "
                f"incomplete={getattr(response, 'incomplete_details', None)}"
            )
        result = json.loads(response.output_text)
        facts = result.get("facts")
        if not isinstance(facts, list) or any(not isinstance(fact, str) for fact in facts):
            raise ValueError("self-knowledge response has invalid facts")
        metadata = {
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": response.id,
            "response_model": response.model,
            "status": response.status,
            "usage": _usage_dict(getattr(response, "usage", None)),
            "raw_text": response.output_text,
        }
        return facts, metadata

    def answer(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        requested_at = utc_now()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": plan["prompt"]}]
        for image_path in plan["image_paths"]:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                raise ValueError(f"unsupported image MIME type: {path}")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{encoded}",
                    "detail": plan["settings"]["image_detail"],
                }
            )

        def operation() -> Any:
            return self.client.responses.create(
                model=plan["model"],
                instructions=ANSWER_INSTRUCTIONS,
                input=[{"role": "user", "content": content}],
                text={"verbosity": "low"},
                reasoning={"effort": plan["settings"]["reasoning_effort"]},
                max_output_tokens=int(plan["settings"]["max_output_tokens"]),
                store=False,
                metadata={
                    "experiment": "e0",
                    "question_id": str(plan["question_id"])[:512],
                    "condition": str(plan["condition"]),
                },
            )

        response = self._with_retry(operation)
        if response.status != "completed" or response.output_text is None:
            raise RuntimeError(
                f"answer response status={response.status}, "
                f"incomplete={getattr(response, 'incomplete_details', None)}"
            )
        return {
            "raw_text": response.output_text,
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": response.id,
            "response_model": response.model,
            "status": response.status,
            "usage": _usage_dict(getattr(response, "usage", None)),
        }


class BedrockResponsesClient(OpenAIResponsesClient):
    """OpenAI Responses interface backed by Amazon Bedrock."""

    def __init__(self, api_key: str, region: str, timeout: float, max_retries: int) -> None:
        from openai import OpenAI
        from openai.providers import bedrock

        self.client = OpenAI(
            provider=bedrock(region=region, api_key=api_key),
            timeout=timeout,
            max_retries=0,
        )
        self.max_retries = max_retries


class BedrockConverseError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Bedrock Converse HTTP {status_code}: {message[:900]}")
        self.status_code = status_code


class BedrockConverseClient:
    """Native Bedrock Converse client with bearer authentication and image support."""

    def __init__(self, api_key: str, region: str, timeout: float, max_retries: int) -> None:
        self.api_key = api_key
        self.region = region
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = requests.Session()

    def _post(self, model: str, payload: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, str]]:
        encoded_model = quote(model, safe=".:")
        url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{encoded_model}/converse"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            response = self.http.post(url, headers=headers, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                return response.json(), response.headers
            try:
                error_payload = response.json()
                message = str(error_payload.get("message") or error_payload.get("error") or error_payload)
            except ValueError:
                message = response.text
            if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                raise BedrockConverseError(response.status_code, message)
            time.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    @staticmethod
    def _response_text(response: Mapping[str, Any]) -> str:
        content = response.get("output", {}).get("message", {}).get("content", [])
        text_blocks = [str(block["text"]) for block in content if isinstance(block, dict) and "text" in block]
        text = "\n".join(text_blocks).strip()
        if not text:
            raise RuntimeError(
                "Bedrock response contains no textual content "
                f"(stopReason={response.get('stopReason')})"
            )
        return text

    @staticmethod
    def _thinking_fields(settings: Mapping[str, Any]) -> dict[str, Any] | None:
        budget = int(settings.get("thinking_budget_tokens", 0))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}} if budget > 0 else None

    def generate_facts(
        self, prompt: str, model: str, settings: Mapping[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        requested_at = utc_now()
        schema = {
            "type": "object",
            "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
            "required": ["facts"],
            "additionalProperties": False,
        }
        payload: dict[str, Any] = {
            "system": [
                {
                    "text": (
                        "Provide concise, general kitchen background facts. Do not infer anything "
                        "about a particular person, video, episode, or answer choice. Return only "
                        'a valid JSON object with the form {"facts":["fact 1","fact 2","fact 3"]}. '
                        "Do not use Markdown fences or include any other keys or commentary."
                    )
                }
            ],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": int(settings["max_output_tokens"])},
            "requestMetadata": {"experiment": "e0", "operation": "self_knowledge"},
        }
        if bool(settings.get("structured_output", True)):
            payload["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(schema, separators=(",", ":")),
                            "name": "e0_background_facts",
                            "description": "Three to five concise general kitchen background facts",
                        }
                    },
                }
            }
        thinking = self._thinking_fields(settings)
        if thinking is not None:
            payload["additionalModelRequestFields"] = thinking
        response, headers = self._post(model, payload)
        raw_text = self._response_text(response)
        result = json.loads(raw_text)
        facts = result.get("facts")
        if not isinstance(facts, list) or any(not isinstance(fact, str) for fact in facts):
            raise ValueError("self-knowledge response has invalid facts")
        metadata = {
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": headers.get("x-amzn-requestid") or headers.get("x-amzn-request-id"),
            "response_model": model,
            "status": "completed",
            "finish_reason": response.get("stopReason"),
            "usage": response.get("usage"),
            "provider_metrics": response.get("metrics"),
            "raw_text": raw_text,
        }
        return facts, metadata

    def answer(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        requested_at = utc_now()
        content: list[dict[str, Any]] = [{"text": plan["prompt"]}]
        for image_path in plan["image_paths"]:
            path = Path(image_path)
            image_format = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(
                path.suffix.lower().lstrip(".")
            )
            if image_format is None:
                raise ValueError(f"unsupported Bedrock image format: {path}")
            content.append(
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": base64.b64encode(path.read_bytes()).decode("ascii")},
                    }
                }
            )
        payload: dict[str, Any] = {
            "system": [{"text": ANSWER_INSTRUCTIONS}],
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": int(plan["settings"]["max_output_tokens"])},
            "requestMetadata": {
                "experiment": "e0",
                "question_id": str(plan["question_id"])[:256],
                "condition": str(plan["condition"]),
            },
        }
        thinking = self._thinking_fields(plan["settings"])
        if thinking is not None:
            payload["additionalModelRequestFields"] = thinking
        response, headers = self._post(str(plan["model"]), payload)
        raw_text = self._response_text(response)
        return {
            "raw_text": raw_text,
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": headers.get("x-amzn-requestid") or headers.get("x-amzn-request-id"),
            "response_model": plan["model"],
            "status": "completed",
            "finish_reason": response.get("stopReason"),
            "usage": response.get("usage"),
            "provider_metrics": response.get("metrics"),
        }


def _reference_http_proxy() -> Any | None:
    """Mirror the certificate-backed proxy hook used by the referenced Azure client."""

    try:
        import httpx
        from libfb.py.certpathpicker import get_client_credential_paths

        certificate, private_key = get_client_credential_paths()
        if not Path(certificate).is_file() or not Path(private_key).is_file():
            raise RuntimeError("referenced proxy client certificate files are unavailable")
        context = ssl.create_default_context()
        context.load_cert_chain(certificate, private_key)
        proxy = httpx.Proxy("https://fwdproxy:8082", ssl_context=context)
        return httpx.Client(proxy=proxy)
    except (ImportError, OSError, RuntimeError):
        return None


class AzureOpenAIChatClient:
    """Azure Chat Completions client matching the existing reference integration."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        from openai import AzureOpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "timeout": timeout,
            "max_retries": 0,
        }
        http_client = _reference_http_proxy()
        if http_client is not None:
            kwargs["http_client"] = http_client
        self.client = AzureOpenAI(**kwargs)
        self.max_retries = max_retries

    def _with_retry(self, operation: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if not retryable_api_error(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    @staticmethod
    def _message_text(response: Any) -> str:
        if not response.choices:
            raise RuntimeError("Azure response contains no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Azure response contains no textual content "
                f"(finish_reason={response.choices[0].finish_reason})"
            )
        return content

    def generate_facts(
        self, prompt: str, model: str, settings: Mapping[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        requested_at = utc_now()

        def operation() -> Any:
            return self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Provide concise, general kitchen background facts. Do not infer "
                            "anything about a particular person, video, episode, or answer choice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "e0_background_facts",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "facts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 5,
                                }
                            },
                            "required": ["facts"],
                            "additionalProperties": False,
                        },
                    },
                },
                reasoning_effort=settings["reasoning_effort"],
                max_completion_tokens=int(settings["max_output_tokens"]),
            )

        response = self._with_retry(operation)
        raw_text = self._message_text(response)
        result = json.loads(raw_text)
        facts = result.get("facts")
        if not isinstance(facts, list) or any(not isinstance(fact, str) for fact in facts):
            raise ValueError("self-knowledge response has invalid facts")
        metadata = {
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": getattr(response, "_request_id", None) or response.id,
            "response_model": response.model,
            "status": "completed",
            "finish_reason": response.choices[0].finish_reason,
            "usage": _usage_dict(getattr(response, "usage", None)),
            "raw_text": raw_text,
        }
        return facts, metadata

    def answer(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        requested_at = utc_now()
        content: list[dict[str, Any]] = [{"type": "text", "text": plan["prompt"]}]
        for image_path in plan["image_paths"]:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                raise ValueError(f"unsupported image MIME type: {path}")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                        "detail": plan["settings"]["image_detail"],
                    },
                }
            )

        def operation() -> Any:
            return self.client.chat.completions.create(
                model=plan["model"],
                messages=[
                    {"role": "system", "content": ANSWER_INSTRUCTIONS},
                    {"role": "user", "content": content},
                ],
                reasoning_effort=plan["settings"]["reasoning_effort"],
                max_completion_tokens=int(plan["settings"]["max_output_tokens"]),
            )

        response = self._with_retry(operation)
        raw_text = self._message_text(response)
        return {
            "raw_text": raw_text,
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "request_id": getattr(response, "_request_id", None) or response.id,
            "response_model": response.model,
            "status": "completed",
            "finish_reason": response.choices[0].finish_reason,
            "usage": _usage_dict(getattr(response, "usage", None)),
        }


class OpenAICompatibleChatClient(AzureOpenAIChatClient):
    """Chat Completions client for an explicitly configured OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.max_retries = max_retries


def parse_answer(text: str) -> str | None:
    stripped = text.strip()
    return stripped if stripped in LETTERS else None


def retryable_api_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    if name in {"APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    if name != "RateLimitError":
        return False
    # Rate-limit HTTP status is overloaded for hard billing/quota failures. Those
    # are not transient and must not be retried. The OpenAI SDK exposes the body on
    # current releases; keep a string fallback for compatible older releases.
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = str(error.get("code") or getattr(exc, "code", "")).lower()
    kind = str(error.get("type") or getattr(exc, "type", "")).lower()
    text = str(exc).lower()
    hard_markers = {
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "no credits remaining",
    }
    return not any(marker in code or marker in kind or marker in text for marker in hard_markers)


def build_answer_plan(
    record: Mapping[str, Any],
    condition: str,
    provider: str,
    model: str,
    settings: Mapping[str, Any],
    frame_record: Mapping[str, Any] | None = None,
    knowledge_record: Mapping[str, Any] | None = None,
    minimum_c3_facts: int = 3,
    provider_endpoint: str | None = None,
    knowledge_policy_version: str | None = None,
) -> dict[str, Any]:
    if condition not in {"C0", "C1", "C2", "C3"}:
        raise ValueError(f"unknown E0 condition: {condition}")
    if condition == "C0":
        if frame_record is not None or knowledge_record is not None:
            raise ValueError("C0 must not receive frames or knowledge")
        image_paths: list[str] = []
        frame_hash = None
        knowledge_cache_id = None
        facts = None
    else:
        if frame_record is None:
            raise ValueError(f"{condition} requires the shared action-clip frames")
        image_paths = [str(frame["path"]) for frame in frame_record["frames"]]
        frame_hash = str(frame_record["ordered_frame_payload_hash"])
        if condition == "C1":
            if knowledge_record is not None:
                raise ValueError("C1 must not receive knowledge")
            knowledge_cache_id = None
            facts = None
        elif condition == "C2":
            facts = [str(fact) for fact in (knowledge_record or {}).get("facts", [])]
            if not 3 <= len(facts) <= 5:
                raise ValueError("C2 requires 3–5 background facts")
            knowledge_cache_id = str(knowledge_record["cache_id"])
        else:
            facts = [str(fact) for fact in (knowledge_record or {}).get("facts", [])]
            if not minimum_c3_facts <= len(facts) <= 5:
                raise ValueError(
                    f"C3 requires {minimum_c3_facts}–5 usable background facts"
                )
            knowledge_cache_id = str(knowledge_record["cache_id"])
    prompt = answer_text(str(record["question"]), record["displayed_choices"], facts)
    identity = {
        "prompt_version": PROMPT_VERSION,
        "instructions": ANSWER_INSTRUCTIONS,
        "question_id": record["question_id"],
        "condition": condition,
        "question": record["question"],
        "displayed_choices": record["displayed_choices"],
        "option_order": record["option_order"],
        "ordered_frame_payload_hash": frame_hash,
        "knowledge_cache_id": knowledge_cache_id,
        "provider": provider,
        "model": model,
        "settings": dict(settings),
    }
    if provider_endpoint is not None or knowledge_policy_version is not None:
        identity.update(
            {
                "provider_endpoint": provider_endpoint,
                "knowledge_policy_version": knowledge_policy_version,
                "knowledge_facts": facts,
            }
        )
    return {
        **identity,
        "request_hash": stable_hash(identity),
        "prompt": prompt,
        "prompt_hash": stable_hash({"instructions": ANSWER_INSTRUCTIONS, "prompt": prompt}),
        "image_paths": image_paths,
    }


def _prediction_from_response(plan: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(response["raw_text"])
    parsed = parse_answer(raw)
    prediction = {
        "question_id": plan["question_id"],
        "condition": plan["condition"],
        "provider": plan["provider"],
        "model": plan["model"],
        "inference_configuration": dict(plan["settings"]),
        "prompt_version": plan["prompt_version"],
        "prompt_hash": plan["prompt_hash"],
        "request_hash": plan["request_hash"],
        "option_order": list(plan["option_order"]),
        "displayed_choices": list(plan["displayed_choices"]),
        "ordered_frame_payload_hash": plan["ordered_frame_payload_hash"],
        "knowledge_cache_id": plan["knowledge_cache_id"],
        "raw_textual_response": raw,
        "parsed_answer": parsed,
        "request_status": "completed",
        "error_category": None if parsed is not None else "parse_failure",
        "request_timestamp": response["request_timestamp"],
        "latency_seconds": response["latency_seconds"],
        "token_usage": response.get("usage"),
        "request_id": response.get("request_id"),
        "response_model": response.get("response_model"),
    }
    if "provider_endpoint" in plan:
        prediction.update(
            {
                "provider_endpoint": plan.get("provider_endpoint"),
                "knowledge_policy_version": plan.get("knowledge_policy_version"),
                "knowledge_facts": list(plan.get("knowledge_facts") or []),
                "answer_instructions": plan.get("instructions"),
                "assembled_prompt": plan.get("prompt"),
            }
        )
    return prediction


def answer_or_reuse(
    plan: Mapping[str, Any], cache_root: Path, client: ResponsesClient
) -> tuple[dict[str, Any], bool]:
    cache_path = cache_root / str(plan["condition"]) / f"{plan['request_hash']}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("request_hash") != plan["request_hash"]:
            raise ValueError(f"incompatible answer cache at {cache_path}")
        return cached, True
    response = client.answer(plan)
    prediction = _prediction_from_response(plan, response)
    atomic_write_json(cache_path, prediction)
    return prediction, False


def run_answer_plans(
    plans: Sequence[Mapping[str, Any]],
    cache_root: Path,
    client: ResponsesClient,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reused = 0
    created = 0
    if not plans:
        return rows, {"reused": 0, "created": 0, "failed": 0}, failures

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(answer_or_reuse, plan, cache_root, client): plan for plan in plans}
        for future in as_completed(futures):
            plan = futures[future]
            try:
                row, was_reused = future.result()
                rows.append(row)
                reused += int(was_reused)
                created += int(not was_reused)
            except Exception as exc:
                failures.append(
                    {
                        "question_id": plan["question_id"],
                        "condition": plan["condition"],
                        "provider": plan["provider"],
                        "model": plan["model"],
                        "request_hash": plan["request_hash"],
                        "request_status": "failed",
                        "error_category": exc.__class__.__name__,
                        "error": _redacted_error(exc),
                        "request_timestamp": utc_now(),
                    }
                )
    rows.sort(key=lambda row: (row["condition"], row["question_id"]))
    return rows, {"reused": reused, "created": created, "failed": len(failures)}, failures


def merge_prediction_file(
    path: Path, manifest_order: Sequence[str], new_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                question_id = str(row["question_id"])
                if question_id in existing:
                    raise ValueError(f"duplicate canonical prediction for {question_id} in {path}")
                existing[question_id] = row
    for row in new_rows:
        existing[str(row["question_id"])] = dict(row)
    ordered = [existing[question_id] for question_id in manifest_order if question_id in existing]
    atomic_write_jsonl(path, ordered)
    return ordered


def api_key_from_environment(variable_name: str = "OPENAI_API_KEY") -> str | None:
    return os.environ.get(variable_name)


def api_key_from_model_config(model_config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Load a key without copying it into E0 source, config, logs, or artifacts."""

    variable_name = str(model_config.get("credential_environment_variable", "OPENAI_API_KEY"))
    value = api_key_from_environment(variable_name)
    if value:
        return value, f"environment:{variable_name}"

    source = model_config.get("credential_source_python_file")
    symbol = model_config.get("credential_source_variable")
    if not source or not symbol:
        return None, None
    path = Path(str(source)).expanduser()
    if not path.is_file():
        return None, f"unavailable_file:{path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == symbol for target in targets):
            candidate = ast.literal_eval(node.value)
            if not isinstance(candidate, str) or not candidate:
                raise ValueError(f"credential source variable {symbol} is not a nonempty string")
            return candidate, f"python_source:{path}#{symbol}"
    raise ValueError(f"credential source variable {symbol} was not found in {path}")
