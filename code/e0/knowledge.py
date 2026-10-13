"""Answer-blind self-generated and external background knowledge for E0."""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote

import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from . import RETRIEVAL_VERSION, RETRIEVAL_V2_VERSION, SELF_KNOWLEDGE_PROMPT_VERSION
from .dataset import atomic_write_json, stable_hash
from .prompts import (
    KITCHEN_CONTEXT,
    object_phrase_from_action,
    parse_action_semantics,
    retrieval_queries,
    retrieval_queries_v2,
    self_knowledge_text,
)

_LAST_HTTP_CALL_MONOTONIC = 0.0


@dataclass(frozen=True)
class KnowledgeRequest:
    """The complete and deliberately minimal input boundary for knowledge stages."""

    action_phrase: str
    kitchen_context: str = KITCHEN_CONTEXT
    prompt_version: str = SELF_KNOWLEDGE_PROMPT_VERSION

    def allowed_payload(self) -> dict[str, str]:
        payload = asdict(self)
        if set(payload) != {"action_phrase", "kitchen_context", "prompt_version"}:
            raise AssertionError("knowledge request boundary changed")
        return payload


class TerminalKnowledgeGenerationError(ValueError):
    """A completed but protocol-invalid generation that must not be regenerated."""


def build_knowledge_request(record: Mapping[str, Any]) -> KnowledgeRequest:
    # Deliberately whitelist one record field. Choices, labels, identities, timestamps,
    # narration, and all other annotation metadata cannot cross this boundary.
    return KnowledgeRequest(action_phrase=str(record["action_phrase"]))


def _knowledge_cache_key(kind: str, request: KnowledgeRequest, settings: Mapping[str, Any]) -> str:
    return stable_hash({"kind": kind, "request": request.allowed_payload(), "settings": settings})


def generate_self_or_reuse(
    request: KnowledgeRequest,
    cache_root: Path,
    provider: str,
    model: str,
    generation: Mapping[str, Any],
    call: Callable[[str, str, Mapping[str, Any]], tuple[list[str], dict[str, Any]]],
) -> tuple[dict[str, Any], bool]:
    settings = {
        "provider": provider,
        "model": model,
        "prompt_version": request.prompt_version,
        "generation": dict(generation),
    }
    cache_id = _knowledge_cache_key("self_generated", request, settings)
    path = cache_root / "self" / f"{cache_id}.json"
    failure_path = cache_root / "self" / "failures" / f"{cache_id}.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("cache_id") != cache_id or failure.get("request") != request.allowed_payload():
            raise ValueError(f"incompatible terminal self-knowledge record at {failure_path}")
        raise TerminalKnowledgeGenerationError(str(failure.get("error") or "terminal invalid generation"))
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_id") != cache_id or cached.get("request") != request.allowed_payload():
            raise ValueError(f"incompatible self-knowledge cache at {path}")
        return cached, True

    prompt = self_knowledge_text(request.action_phrase)
    facts, response = call(prompt, model, generation)
    facts = [" ".join(str(fact).split()) for fact in facts if str(fact).strip()]
    if not 3 <= len(facts) <= 5:
        error = f"self-knowledge response returned {len(facts)} facts; expected 3–5"
        atomic_write_json(
            failure_path,
            {
                "cache_id": cache_id,
                "kind": "self_generated_terminal_failure",
                "request": request.allowed_payload(),
                "prompt": prompt,
                "settings": settings,
                "facts": facts,
                "status": "invalid_generation",
                "error": error,
                "response": response,
            },
        )
        raise TerminalKnowledgeGenerationError(error)
    payload = {
        "cache_id": cache_id,
        "kind": "self_generated",
        "request": request.allowed_payload(),
        "prompt": prompt,
        "settings": settings,
        "facts": facts,
        "word_count": sum(len(fact.split()) for fact in facts),
        "status": "available",
        "response": response,
    }
    atomic_write_json(path, payload)
    return payload, False


def _clean_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = " ".join(text.split())
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences: list[str] = []
    for candidate in candidates:
        sentence = candidate.strip()
        words = sentence.split()
        if 8 <= len(words) <= 55 and not sentence.lower().startswith(("this article", "the following")):
            sentences.append(sentence)
    return sentences


def _rank_sentences(action_phrase: str, candidates: Sequence[dict[str, Any]], max_facts: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    object_phrase = object_phrase_from_action(action_phrase)
    object_tokens = {token for token in re.findall(r"[a-z0-9]+", object_phrase.lower()) if len(token) > 2}
    relevant = [
        candidate
        for candidate in candidates
        if object_tokens
        & set(re.findall(r"[a-z0-9]+", f"{candidate['title']} {candidate['sentence']}".lower()))
    ]
    if not relevant:
        return []
    candidates = relevant
    query = f"{action_phrase} {object_phrase} kitchen purpose function use cooking cleaning measure transfer"
    texts = [f"{candidate['title']} {candidate['sentence']}" for candidate in candidates]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([query, *texts])
    scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), int(candidates[index]["source_rank"]), int(candidates[index]["sentence_rank"])),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    word_count = 0
    for index in ranked:
        candidate = candidates[index]
        normalized = str(candidate["sentence"]).casefold()
        if normalized in seen:
            continue
        words = len(str(candidate["sentence"]).split())
        if selected and word_count + words > 180:
            continue
        selected.append({**candidate, "score": float(scores[index])})
        seen.add(normalized)
        word_count += words
        if len(selected) >= max_facts:
            return selected
    return selected


def _get_json(
    http: requests.Session,
    endpoint: str,
    params: Mapping[str, Any],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    global _LAST_HTTP_CALL_MONOTONIC
    for attempt in range(max_retries + 1):
        elapsed = time.monotonic() - _LAST_HTTP_CALL_MONOTONIC
        if elapsed < 1.25:
            time.sleep(1.25 - elapsed)
        response = http.get(endpoint, params=params, timeout=timeout)
        _LAST_HTTP_CALL_MONOTONIC = time.monotonic()
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        if attempt >= max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else float(10 * (attempt + 1))
        time.sleep(min(30.0, max(5.0, wait)))
    raise AssertionError("unreachable")


def retrieve_external_or_reuse(
    request: KnowledgeRequest,
    cache_root: Path,
    settings: Mapping[str, Any],
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], bool]:
    queries = retrieval_queries(request.action_phrase)
    identity_settings = {
        "retrieval_version": RETRIEVAL_VERSION,
        "lookup_strategy": "direct_object_title_then_search_v1",
        "context": request.kitchen_context,
        "queries": queries,
        "settings": dict(settings),
    }
    locator_id = _knowledge_cache_key("external_locator", request, identity_settings)
    locator_path = cache_root / "external" / "locators" / f"{locator_id}.json"
    if locator_path.is_file():
        locator = json.loads(locator_path.read_text(encoding="utf-8"))
        record_path = Path(locator["record_path"])
        if not record_path.is_file():
            raise ValueError(f"external-knowledge locator points to missing cache: {record_path}")
        cached = json.loads(record_path.read_text(encoding="utf-8"))
        if cached.get("locator_id") != locator_id or cached.get("request") != request.allowed_payload():
            raise ValueError(f"incompatible external-knowledge cache at {record_path}")
        return cached, True

    http = session or requests.Session()
    http.headers.update({"User-Agent": str(settings.get("user_agent", "ActGround-E0/1.0"))})
    endpoint = str(settings["endpoint"])
    timeout = float(settings["timeout_seconds"])
    max_pages = int(settings["max_pages"])
    max_retries = int(settings.get("max_retries", 2))
    search_audit: list[dict[str, Any]] = []
    page_acquisitions: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    object_phrase = object_phrase_from_action(request.action_phrase)
    direct_payload = _get_json(
        http,
        endpoint,
        {
            "action": "query",
            "prop": "extracts|revisions|info",
            "titles": object_phrase,
            "redirects": 1,
            "explaintext": 1,
            "exchars": 5000,
            "rvprop": "ids|timestamp",
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        },
        timeout,
        max_retries,
    )
    direct_pages = [
        page
        for page in direct_payload.get("query", {}).get("pages", [])
        if "pageid" in page and not page.get("missing")
    ]
    search_audit.append(
        {
            "query": f"direct-title:{object_phrase}",
            "hits": [{"pageid": page.get("pageid"), "title": page.get("title")} for page in direct_pages],
        }
    )
    for source_rank, page in enumerate(direct_pages):
        title = str(page.get("title") or object_phrase)
        revision = (page.get("revisions") or [{}])[0]
        documents.append(
            {
                "page_id": int(page["pageid"]),
                "title": title,
                "url": str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "query": f"direct-title:{object_phrase}",
                "source_rank": source_rank,
                "extract": str(page.get("extract") or ""),
            }
        )

    object_tokens = {token for token in re.findall(r"[a-z0-9]+", object_phrase.lower()) if len(token) > 2}
    direct_relevant_count = sum(
        bool(object_tokens & set(re.findall(r"[a-z0-9]+", f"{document['title']} {sentence}".lower())))
        for document in documents
        for sentence in _clean_sentences(document["extract"])
    )
    page_hits: dict[int, dict[str, Any]] = {document["page_id"]: {} for document in documents}
    if direct_relevant_count < 3:
        for query in queries:
            response_payload = _get_json(
                http,
                endpoint,
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": max_pages,
                    "srnamespace": 0,
                    "format": "json",
                    "formatversion": 2,
                },
                timeout,
                max_retries,
            )
            hits = response_payload.get("query", {}).get("search", [])
            search_audit.append(
                {
                    "query": query,
                    "hits": [{"pageid": hit.get("pageid"), "title": hit.get("title")} for hit in hits],
                }
            )
            for hit in hits:
                page_id = hit.get("pageid")
                if isinstance(page_id, int) and page_id not in page_hits:
                    page_hits[page_id] = {"title": hit.get("title"), "query": query}
            if len(page_hits) >= max_pages:
                break

    search_hits = [(page_id, hit) for page_id, hit in page_hits.items() if hit][:max_pages]
    if search_hits:
        response_payload = _get_json(
            http,
            endpoint,
            {
                "action": "query",
                "prop": "extracts|revisions|info",
                "pageids": "|".join(str(page_id) for page_id, _ in search_hits),
                "explaintext": 1,
                "exchars": 5000,
                "rvprop": "ids|timestamp",
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
            },
            timeout,
            max_retries,
        )
        pages_by_id = {
            int(page["pageid"]): page
            for page in response_payload.get("query", {}).get("pages", [])
            if "pageid" in page
        }
    else:
        pages_by_id = {}
    for source_rank, (page_id, hit) in enumerate(search_hits, start=len(documents)):
        page = pages_by_id.get(page_id)
        if not page:
            continue
        title = str(page.get("title") or hit.get("title") or "")
        url = str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}")
        revision = (page.get("revisions") or [{}])[0]
        documents.append(
            {
                "page_id": page_id,
                "title": title,
                "url": url,
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "query": hit["query"],
                "source_rank": source_rank,
                "extract": str(page.get("extract") or ""),
            }
        )

    candidates: list[dict[str, Any]] = []
    for document in documents:
        for sentence_rank, sentence in enumerate(_clean_sentences(document["extract"])):
            candidates.append(
                {
                    "sentence": sentence,
                    "supporting_span": sentence,
                    "title": document["title"],
                    "url": document["url"],
                    "page_id": document["page_id"],
                    "revision_id": document["revision_id"],
                    "revision_timestamp": document["revision_timestamp"],
                    "query": document["query"],
                    "source_rank": document["source_rank"],
                    "sentence_rank": sentence_rank,
                }
            )
    selected = _rank_sentences(request.action_phrase, candidates, int(settings["fact_count"]))
    facts = [row["sentence"] for row in selected]
    retrieval_time = datetime.now(timezone.utc).isoformat()
    snapshot = [
        {
            "page_id": document["page_id"],
            "revision_id": document["revision_id"],
            "revision_timestamp": document["revision_timestamp"],
            "url": document["url"],
        }
        for document in documents
    ]
    cache_id = stable_hash(
        {
            "kind": "external_retrieval",
            "request": request.allowed_payload(),
            "identity_settings": identity_settings,
            "source_snapshot": snapshot,
            "selected": [
                {"url": row["url"], "revision_id": row["revision_id"], "sentence": row["sentence"]}
                for row in selected
            ],
        }
    )
    record_path = cache_root / "external" / "records" / f"{cache_id}.json"
    context_terms = {
        "kitchen", "cook", "cooking", "food", "utensil", "measure", "measurement",
        "bake", "baking", "clean", "cleaning", "wash", "washing", "dish", "container",
        "ingredient", "serve", "serving", "prepare", "preparing", "pan", "spoon", "bowl",
    }
    fact_tokens = set(re.findall(r"[a-z0-9]+", " ".join(facts).lower()))
    status = (
        "available"
        if len(facts) >= 3 and bool(context_terms & fact_tokens)
        else ("weak" if facts else "empty")
    )
    payload = {
        "cache_id": cache_id,
        "locator_id": locator_id,
        "kind": "external_retrieval",
        "request": request.allowed_payload(),
        "settings": identity_settings,
        "status": status,
        "retrieval_timestamp": retrieval_time,
        "queries": list(queries),
        "search_audit": search_audit,
        "source_snapshot": snapshot,
        "documents": documents,
        "facts": facts,
        "word_count": sum(len(fact.split()) for fact in facts),
        "fact_provenance": [
            {
                key: row[key]
                for key in (
                    "title",
                    "url",
                    "page_id",
                    "revision_id",
                    "revision_timestamp",
                    "query",
                    "supporting_span",
                    "score",
                    "source_rank",
                    "sentence_rank",
                )
            }
            for row in selected
        ],
        "extraction_ranking_method": "Wikipedia search API plus deterministic TF-IDF sentence ranking v1",
    }
    atomic_write_json(record_path, payload)
    atomic_write_json(
        locator_path,
        {
            "locator_id": locator_id,
            "cache_id": cache_id,
            "record_path": str(record_path.resolve()),
            "source_snapshot": snapshot,
        },
    )
    return payload, False


def unavailable_external_record(request: KnowledgeRequest, settings: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {
        "cache_id": None,
        "kind": "external_retrieval",
        "request": request.allowed_payload(),
        "settings": dict(settings),
        "status": "unavailable",
        "facts": [],
        "fact_provenance": [],
        "error": error,
    }


V2_SELECTOR_VERSION = "wikipedia_lead_function_selector_v2.6"
V2_SELECTION_POLICY = {
    "version": V2_SELECTOR_VERSION,
    "source_family": "English Wikipedia MediaWiki API",
    "disambiguation_facts_forbidden": True,
    "title_only_sentence_relevance_forbidden": True,
    "sentence_gate": "every_sentence_requires_function_or_kitchen_context; later_sentences_also_require_referent",
    "local_source_normalization": "collapse_whitespace_before_selection",
    "canonical_source_precedence": "exclusive_when_verified_canonical_page_available",
    "noncanonical_page_gate": "concept_in_title_and_kitchen_context",
    "source_acquisition": "two fixed generator-search batches with exintro plus verified canonical fallback",
    "sentence_word_bounds": [8, 65],
    "fact_count_bounds": [1, 3],
    "fact_word_budget": 180,
}
V2_SELECTION_POLICY_FINGERPRINT = stable_hash(V2_SELECTION_POLICY)

_TOKEN_STOP = {
    "a", "an", "and", "both", "contents", "for", "from", "in", "into", "item",
    "items", "large", "left", "of", "right", "the", "to", "up", "use",
    "with",
}
_CONTEXT_TERMS = {
    "appliance", "bake", "baking", "boil", "clean", "cleaning", "cook", "cooking",
    "dish", "dishware", "drain", "drink", "food", "fry", "frying", "ingredient",
    "kitchen", "measure", "mix", "oven", "pan", "prepare", "serve", "stove",
    "utensil", "wash", "weigh",
}
_FUNCTION_TERMS = {
    "allow", "allows", "contain", "container", "control", "controls", "controlling",
    "device", "dry", "function", "handle", "hold", "measure", "measures", "measuring",
    "regulate", "regulates", "release", "remove", "serve", "tool", "transfer", "wash",
}
_WRONG_DOMAIN_TERMS = {
    "airline", "album", "band", "collectible", "etymology", "genus", "geological",
    "magazine", "mythology", "publisher", "religion", "species", "tectonic",
}


def _tokens_v2(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in _TOKEN_STOP
    }


def _safe_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:800]}"


def _api_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "api_error": payload.get("error"),
        "warnings": payload.get("warnings"),
        "continuation": payload.get("continue"),
    }


def _cached_api_request_v2(
    http: requests.Session,
    cache_root: Path,
    endpoint: str,
    params: Mapping[str, Any],
    timeout: float,
    max_retries: int,
) -> tuple[dict[str, Any], bool]:
    request_identity = {
        "retrieval_version": RETRIEVAL_V2_VERSION,
        "endpoint": endpoint,
        "params": dict(params),
    }
    request_id = stable_hash(request_identity)
    path = cache_root / "external_v2" / "api" / f"{request_id}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("request_identity") != request_identity:
            raise ValueError(f"incompatible cached MediaWiki request at {path}")
        return cached, True
    acquired_at = datetime.now(timezone.utc).isoformat()
    try:
        payload = _get_json(http, endpoint, params, timeout, max_retries)
        diagnostics = _api_diagnostics(payload)
        status = "api_error" if diagnostics["api_error"] else "success"
        record = {
            "request_id": request_id,
            "request_identity": request_identity,
            "acquired_at": acquired_at,
            "status": status,
            **diagnostics,
            "payload": payload,
            "error": None,
        }
    except Exception as exc:
        record = {
            "request_id": request_id,
            "request_identity": request_identity,
            "acquired_at": acquired_at,
            "status": "transport_error",
            "api_error": None,
            "warnings": None,
            "continuation": None,
            "payload": None,
            "error": _safe_error(exc),
        }
    atomic_write_json(path, record)
    return record, False


def _search_v2(
    http: requests.Session,
    cache_root: Path,
    endpoint: str,
    query_spec: Mapping[str, str],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    api_record, reused = _cached_api_request_v2(
        http,
        cache_root,
        endpoint,
        {
            "action": "query",
            "list": "search",
            "srsearch": query_spec["query"],
            "srlimit": int(settings["max_search_results"]),
            "srnamespace": 0,
            "srprop": "snippet|titlesnippet",
            "format": "json",
            "formatversion": 2,
        },
        float(settings["timeout_seconds"]),
        int(settings.get("max_retries", 2)),
    )
    payload = api_record.get("payload") or {}
    hits = payload.get("query", {}).get("search", []) if api_record["status"] == "success" else []
    return {
        "kind": query_spec["kind"],
        "query": query_spec["query"],
        "api_request_id": api_record["request_id"],
        "api_cache_reused": reused,
        "status": (
            api_record["status"]
            if api_record["status"] != "success"
            else ("hits" if hits else "no_hits")
        ),
        "warnings": api_record.get("warnings"),
        "continuation": api_record.get("continuation"),
        "error": api_record.get("error") or api_record.get("api_error"),
        "hits": [
            {
                "pageid": hit.get("pageid"),
                "title": hit.get("title"),
                "snippet": " ".join(
                    re.sub(r"<[^>]+>", " ", html.unescape(str(hit.get("snippet") or ""))).split()
                ),
            }
            for hit in hits
        ],
    }


def _page_outcome_v2(page: Mapping[str, Any] | None) -> str:
    if not page:
        return "omitted_page"
    if page.get("missing") or "pageid" not in page:
        return "missing_page"
    if "disambiguation" in (page.get("pageprops") or {}):
        return "disambiguation"
    if "extract" not in page:
        return "omitted_extract"
    if not str(page.get("extract") or "").strip():
        return "empty_content"
    return "success"


def _fetch_page_v2(
    http: requests.Session,
    cache_root: Path,
    endpoint: str,
    settings: Mapping[str, Any],
    *,
    title: str | None = None,
    page_id: int | None = None,
    requested_by: Mapping[str, Any],
) -> dict[str, Any]:
    if (title is None) == (page_id is None):
        raise ValueError("exactly one page title or page ID is required")
    params: dict[str, Any] = {
        "action": "query",
        "prop": "extracts|revisions|info|pageprops",
        "redirects": 1,
        "explaintext": 1,
        "exintro": 1,
        "exchars": min(1200, int(settings["extract_chars"])),
        "exlimit": 1,
        "exsectionformat": "plain",
        "rvprop": "ids|timestamp",
        "inprop": "url",
        "ppprop": "disambiguation",
        "format": "json",
        "formatversion": 2,
    }
    params["titles" if title is not None else "pageids"] = title if title is not None else str(page_id)
    api_record, reused = _cached_api_request_v2(
        http,
        cache_root,
        endpoint,
        params,
        float(settings["timeout_seconds"]),
        int(settings.get("max_retries", 2)),
    )
    payload = api_record.get("payload") or {}
    pages = payload.get("query", {}).get("pages", []) if api_record["status"] == "success" else []
    page = pages[0] if len(pages) == 1 else None
    outcome = api_record["status"] if api_record["status"] != "success" else _page_outcome_v2(page)
    revision = ((page or {}).get("revisions") or [{}])[0]
    resolved_title = str((page or {}).get("title") or title or "")
    normalized = {
        "requested_by": dict(requested_by),
        "requested_title": title,
        "requested_page_id": page_id,
        "api_request_id": api_record["request_id"],
        "api_cache_reused": reused,
        "status": outcome,
        "warnings": api_record.get("warnings"),
        "continuation": api_record.get("continuation"),
        "error": api_record.get("error") or api_record.get("api_error"),
        "page_id": (page or {}).get("pageid"),
        "title": resolved_title,
        "url": str(
            (page or {}).get("fullurl")
            or (
                f"https://en.wikipedia.org/wiki/{quote(resolved_title.replace(' ', '_'))}"
                if resolved_title
                else ""
            )
        ),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "extract": " ".join(str((page or {}).get("extract") or "").split())[: int(settings["local_text_limit_chars"])],
        "redirects": payload.get("query", {}).get("redirects"),
    }
    if normalized["page_id"] is not None and normalized["revision_id"] is not None:
        page_identity = {
            "page_id": normalized["page_id"],
            "revision_id": normalized["revision_id"],
            "extraction": {
                "exintro": True,
                "explaintext": True,
                "exchars": min(1200, int(settings["extract_chars"])),
                "local_text_limit_chars": int(settings["local_text_limit_chars"]),
                "local_source_normalization": "collapse_whitespace",
            },
        }
        normalized["page_cache_id"] = stable_hash(page_identity)
        page_path = cache_root / "external_v2" / "pages" / f"{normalized['page_cache_id']}.json"
        if not page_path.is_file():
            atomic_write_json(page_path, {"page_identity": page_identity, **normalized})
    else:
        normalized["page_cache_id"] = None
    return normalized


def _generator_search_v2(
    http: requests.Session,
    cache_root: Path,
    endpoint: str,
    query_spec: Mapping[str, str],
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retrieve a bounded search result and its lead extracts in one valid request.

    TextExtracts permits multiple extracts when ``exintro`` is present. Every page
    returned by the generator is normalized to an explicit acquisition outcome;
    warnings and continuation are retained on the search audit and raw API record.
    """
    limit = min(int(settings["max_search_results"]), int(settings["max_pages"]))
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query_spec["query"],
        "gsrlimit": limit,
        "gsrnamespace": 0,
        "prop": "extracts|revisions|info|pageprops",
        "explaintext": 1,
        "exintro": 1,
        "exchars": min(1200, int(settings["extract_chars"])),
        "exlimit": limit,
        "exsectionformat": "plain",
        "rvprop": "ids|timestamp",
        "inprop": "url",
        "ppprop": "disambiguation",
        "format": "json",
        "formatversion": 2,
    }
    api_record, reused = _cached_api_request_v2(
        http,
        cache_root,
        endpoint,
        params,
        float(settings["timeout_seconds"]),
        int(settings.get("max_retries", 2)),
    )
    payload = api_record.get("payload") or {}
    raw_pages = payload.get("query", {}).get("pages", []) if api_record["status"] == "success" else []
    raw_pages = sorted(raw_pages, key=lambda row: (int(row.get("index", 10**9)), int(row.get("pageid", 10**9))))
    documents: list[dict[str, Any]] = []
    for raw in raw_pages:
        outcome = _page_outcome_v2(raw)
        revision = (raw.get("revisions") or [{}])[0]
        title = str(raw.get("title") or "")
        document = {
            "requested_by": {
                "kind": "context_search_hit",
                "query": query_spec["query"],
                "query_kind": query_spec["kind"],
                "title": title,
            },
            "requested_title": None,
            "requested_page_id": raw.get("pageid"),
            "api_request_id": api_record["request_id"],
            "api_cache_reused": reused,
            "status": outcome,
            "warnings": api_record.get("warnings"),
            "continuation": api_record.get("continuation"),
            "error": api_record.get("error") or api_record.get("api_error"),
            "page_id": raw.get("pageid"),
            "search_index": raw.get("index"),
            "title": title,
            "url": str(raw.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
            "extract": " ".join(str(raw.get("extract") or "").split())[: int(settings["local_text_limit_chars"])],
            "redirects": payload.get("query", {}).get("redirects"),
        }
        if document["page_id"] is not None and document["revision_id"] is not None:
            page_identity = {
                "page_id": document["page_id"],
                "revision_id": document["revision_id"],
                "extraction": {
                    "exintro": True,
                    "explaintext": True,
                    "exchars": min(1200, int(settings["extract_chars"])),
                    "local_text_limit_chars": int(settings["local_text_limit_chars"]),
                    "local_source_normalization": "collapse_whitespace",
                },
            }
            document["page_cache_id"] = stable_hash(page_identity)
            page_path = cache_root / "external_v2" / "pages" / f"{document['page_cache_id']}.json"
            if not page_path.is_file():
                atomic_write_json(page_path, {"page_identity": page_identity, **document})
        else:
            document["page_cache_id"] = None
        documents.append(document)
    audit = {
        "kind": query_spec["kind"],
        "query": query_spec["query"],
        "api_request_id": api_record["request_id"],
        "api_cache_reused": reused,
        "status": (
            api_record["status"]
            if api_record["status"] != "success"
            else ("hits" if raw_pages else "no_hits")
        ),
        "warnings": api_record.get("warnings"),
        "continuation": api_record.get("continuation"),
        "error": api_record.get("error") or api_record.get("api_error"),
        "hits": [
            {
                "pageid": row.get("page_id"),
                "title": row.get("title"),
                "extraction_selected": True,
                "extraction_outcome": row.get("status"),
            }
            for row in documents
        ],
    }
    return audit, documents


def _clean_sentences_v2(text: str) -> list[str]:
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)
    text = " ".join(text.split())
    result: list[str] = []
    for candidate in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text):
        sentence = candidate.strip()
        words = sentence.split()
        lowered = sentence.casefold()
        if not 8 <= len(words) <= 65:
            continue
        if not sentence.endswith((".", "!", "?")) or sentence.endswith("..."):
            continue
        if any(term in _tokens_v2(lowered) for term in _WRONG_DOMAIN_TERMS):
            continue
        if any(marker in lowered for marker in ("may refer to:", "can refer to:", "this article", "the following list")):
            continue
        if sentence.count("(") != sentence.count(")"):
            continue
        result.append(sentence)
    return result


def _page_relevance_v2(document: Mapping[str, Any], semantics: Any) -> tuple[bool, str]:
    if document.get("status") != "success":
        return False, str(document.get("status") or "acquisition_failed")
    title = str(document.get("title") or "")
    extract = str(document.get("extract") or "")
    title_key = re.sub(r"\s+", " ", title).strip().casefold()
    canonical_keys = {
        re.sub(r"\s+", " ", value).strip().casefold()
        for value in semantics.canonical_titles
    }
    if document.get("requested_by", {}).get("kind") == "canonical_title":
        return True, "resolved_generic_kitchen_sense_title"
    if title_key in canonical_keys:
        return True, "resolved_generic_kitchen_sense_title"
    title_tokens = _tokens_v2(title)
    combined_tokens = _tokens_v2(f"{title} {extract}")
    concept_tokens = _tokens_v2(semantics.lookup_concept)
    if not concept_tokens:
        return False, "underspecified_concept_without_resolvable_tokens"
    if not concept_tokens & title_tokens:
        return False, "resolved_page_does_not_mention_concept"
    if not (_CONTEXT_TERMS & combined_tokens):
        return False, "search_page_has_no_kitchen_context"
    if semantics.uncertainty:
        return True, "uncertain_object_resolved_by_title_and_kitchen_context"
    return True, "concept_in_title_and_kitchen_context_match"


def _select_supported_facts_v2(
    semantics: Any,
    documents: Sequence[Mapping[str, Any]],
    max_facts: int,
    word_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    page_reviews: list[dict[str, Any]] = []
    for source_rank, document in enumerate(documents):
        relevant, reason = _page_relevance_v2(document, semantics)
        sentences = _clean_sentences_v2(str(document.get("extract") or "")) if relevant else []
        page_reviews.append(
            {
                "page_id": document.get("page_id"),
                "title": document.get("title"),
                "status": document.get("status"),
                "accepted_as_relevant": relevant,
                "relevance_reason": reason,
                "usable_sentence_count": len(sentences),
            }
        )
        for sentence_rank, sentence in enumerate(sentences):
            sentence_tokens = _tokens_v2(sentence)
            function_hits = len(_FUNCTION_TERMS & sentence_tokens)
            context_hits = len(_CONTEXT_TERMS & sentence_tokens)
            referent_hits = len(_tokens_v2(semantics.lookup_concept) & sentence_tokens)
            if not (function_hits or context_hits):
                continue
            if sentence_rank > 1 and not (referent_hits and context_hits):
                continue
            if any(
                marker in sentence.casefold()
                for marker in (
                    "ceremonial", "decorative", "historically", "was founded", "was named",
                    "is derived from", "in ancient", "century", "was developed", "was established",
                    "folk art", "medical applications",
                )
            ) or re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})(?:s)?\b", sentence):
                continue
            candidates.append(
                {
                    "sentence": sentence,
                    "supporting_span": sentence,
                    "title": document["title"],
                    "url": document["url"],
                    "page_id": document["page_id"],
                    "revision_id": document["revision_id"],
                    "revision_timestamp": document["revision_timestamp"],
                    "query": document["requested_by"].get("query") or document["requested_by"].get("title"),
                    "source_rank": source_rank,
                    "sentence_rank": sentence_rank,
                    "page_relevance_reason": reason,
                    "function_hits": function_hits,
                    "context_hits": context_hits,
                    "referent_hits": referent_hits,
                }
            )
    if not candidates:
        return [], page_reviews
    ranking_query = (
        f"{semantics.action} {semantics.object_phrase} {semantics.container_or_appliance or ''} "
        f"{semantics.lookup_concept} kitchen function use condition physical consequence"
    )
    texts = [row["sentence"] for row in candidates]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
        [ranking_query, *texts]
    )
    similarities = (matrix[1:] @ matrix[0].T).toarray().ravel()
    for index, row in enumerate(candidates):
        row["tfidf_score"] = float(similarities[index])
        row["selector_score"] = (
            float(similarities[index])
            + 0.035 * min(3, int(row["function_hits"]))
            + 0.02 * min(3, int(row["context_hits"]))
        )
    candidates.sort(
        key=lambda row: (
            -float(row["selector_score"]),
            int(row["source_rank"]),
            int(row["sentence_rank"]),
            str(row["sentence"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    words = 0
    for row in candidates:
        normalized = re.sub(r"[^a-z0-9]+", " ", row["sentence"].casefold()).strip()
        if normalized in seen:
            continue
        count = len(row["sentence"].split())
        if selected and words + count > word_budget:
            continue
        selected.append(row)
        seen.add(normalized)
        words += count
        if len(selected) >= max_facts:
            break
    return selected, page_reviews


def retrieve_external_v2_or_reuse(
    request: KnowledgeRequest,
    cache_root: Path,
    settings: Mapping[str, Any],
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], bool]:
    if settings.get("version") != RETRIEVAL_V2_VERSION:
        raise ValueError("revised C3 retrieval version mismatch")
    semantics = parse_action_semantics(request.action_phrase)
    query_specs = retrieval_queries_v2(request.action_phrase)
    effective_settings = {
        key: settings[key]
        for key in (
            "version",
            "provider",
            "endpoint",
            "max_search_results",
            "max_pages",
            "fact_count",
            "extract_chars",
            "local_text_limit_chars",
            "fact_word_budget",
            "timeout_seconds",
            "user_agent",
        )
    }
    locator_identity = {
        "kind": "external_retrieval_v2_locator",
        "request": request.allowed_payload(),
        "action_semantics": semantics.audit_dict(),
        "queries": list(query_specs),
        "settings": effective_settings,
        "selector_version": V2_SELECTOR_VERSION,
        "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
    }
    locator_id = stable_hash(locator_identity)
    locator_path = cache_root / "external_v2" / "locators" / f"{locator_id}.json"
    if locator_path.is_file():
        locator = json.loads(locator_path.read_text(encoding="utf-8"))
        record_path = Path(locator["record_path"])
        cached = json.loads(record_path.read_text(encoding="utf-8"))
        if cached.get("locator_id") != locator_id or cached.get("request") != request.allowed_payload():
            raise ValueError(f"incompatible revised C3 cache at {record_path}")
        return cached, True

    http = session or requests.Session()
    http.headers.update({"User-Agent": str(settings["user_agent"])})
    endpoint = str(settings["endpoint"])
    search_audit: list[dict[str, Any]] = []
    page_acquisitions: list[dict[str, Any]] = []
    operational_errors: list[dict[str, Any]] = []
    for query_spec in query_specs:
        audit, acquired = _generator_search_v2(
            http, cache_root, endpoint, query_spec, settings
        )
        search_audit.append(audit)
        page_acquisitions.extend(acquired)
        if audit["status"] in {"api_error", "transport_error"}:
            operational_errors.append(
                {
                    "operation": "generator_search_extract",
                    "query": audit["query"],
                    "status": audit["status"],
                    "error": audit["error"],
                }
            )

    def title_key(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    canonical_keys = {title_key(value) for value in semantics.canonical_titles}
    resolved_keys = {
        title_key(str(row.get("title") or ""))
        for row in page_acquisitions
        if row.get("status") == "success"
    }
    if canonical_keys and not (canonical_keys & resolved_keys):
        for title in semantics.canonical_titles:
            document = _fetch_page_v2(
                http,
                cache_root,
                endpoint,
                settings,
                title=title,
                requested_by={"kind": "canonical_title", "title": title, "query": f"title:{title}"},
            )
            page_acquisitions.append(document)
            if document["status"] in {"api_error", "transport_error"}:
                operational_errors.append(
                    {
                        "operation": "canonical_title",
                        "title": title,
                        "status": document["status"],
                        "error": document["error"],
                    }
                )
            if document["status"] == "success":
                break

    # Select from unique page revisions. Verified canonical pages receive source
    # priority, but their existence never suppresses either contextual search.
    unique_documents: dict[tuple[Any, Any], dict[str, Any]] = {}
    for document in page_acquisitions:
        identity = (document.get("page_id"), document.get("revision_id"))
        existing = unique_documents.get(identity)
        if existing is None or document.get("requested_by", {}).get("kind") == "canonical_title":
            unique_documents[identity] = document
    documents = sorted(
        unique_documents.values(),
        key=lambda row: (
            0 if row.get("requested_by", {}).get("kind") == "canonical_title" else 1,
            0 if row.get("requested_by", {}).get("query_kind") == "action_kitchen_context" else 1,
            int(row.get("search_index") or 10**9),
            int(row.get("page_id") or 10**9),
        ),
    )

    verified_canonical = [
        row
        for row in documents
        if row.get("status") == "success"
        and (
            row.get("requested_by", {}).get("kind") == "canonical_title"
            or title_key(str(row.get("title") or "")) in canonical_keys
        )
    ]
    selection_documents = verified_canonical or documents

    selected, page_reviews = _select_supported_facts_v2(
        semantics,
        selection_documents,
        int(settings["fact_count"]),
        int(settings["fact_word_budget"]),
    )
    facts = [row["sentence"] for row in selected]
    if facts:
        status = "usable"
        abstention_reason = None
    elif operational_errors and not any(row.get("status") == "success" for row in documents):
        status = "operational_error"
        abstention_reason = "retrieval_operations_failed"
    else:
        status = "abstained"
        abstention_reason = (
            "underspecified_action_no_usable_source"
            if semantics.uncertainty == "underspecified_object"
            else "no_usable_supported_facts"
        )
    source_snapshot = [
        {
            "page_id": document.get("page_id"),
            "revision_id": document.get("revision_id"),
            "status": document.get("status"),
            "page_cache_id": document.get("page_cache_id"),
        }
        for document in documents
    ]
    fact_provenance = [
        {
            key: row[key]
            for key in (
                "title",
                "url",
                "page_id",
                "revision_id",
                "revision_timestamp",
                "query",
                "supporting_span",
                "source_rank",
                "sentence_rank",
                "page_relevance_reason",
                "tfidf_score",
                "selector_score",
            )
        }
        for row in selected
    ]
    for row in fact_provenance:
        row["supporting_span_sha256"] = stable_hash(row["supporting_span"])
    cache_id = stable_hash(
        {
            "kind": "external_retrieval_v2",
            "locator_id": locator_id,
            "source_snapshot": source_snapshot,
            "facts": facts,
            "fact_provenance": fact_provenance,
        }
    )
    payload = {
        "cache_id": cache_id,
        "locator_id": locator_id,
        "kind": "external_retrieval_v2",
        "request": request.allowed_payload(),
        "action_semantics": semantics.audit_dict(),
        "settings": effective_settings,
        "selector_version": V2_SELECTOR_VERSION,
        "selection_policy": V2_SELECTION_POLICY,
        "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
        "status": status,
        "abstention_reason": abstention_reason,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "queries": list(query_specs),
        "search_audit": search_audit,
        "page_acquisitions": page_acquisitions,
        "page_reviews": page_reviews,
        "operational_errors": operational_errors,
        "source_snapshot": source_snapshot,
        "facts": facts,
        "word_count": sum(len(fact.split()) for fact in facts),
        "fact_provenance": fact_provenance,
        "extraction_ranking_method": (
            "context-first and broad MediaWiki generator searches with bounded multi-page "
            "exintro plaintext extracts, plus verified canonical fallback; "
            f"deterministic local source/sentence filtering and TF-IDF ({V2_SELECTOR_VERSION})"
        ),
    }
    record_path = cache_root / "external_v2" / "records" / f"{cache_id}.json"
    atomic_write_json(record_path, payload)
    atomic_write_json(
        locator_path,
        {
            "locator_id": locator_id,
            "cache_id": cache_id,
            "record_path": str(record_path.resolve()),
            "selection_policy_fingerprint": V2_SELECTION_POLICY_FINGERPRINT,
            "source_snapshot": source_snapshot,
        },
    )
    return payload, False
