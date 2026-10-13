"""Frozen validation diagnostics for tagged HD-EPIC free-text responses.

These lexical and structural diagnostics do not measure semantic correctness.
No LLM judge or generated-text-to-action-class mapper is used.
"""
from __future__ import annotations

import hashlib
import html
import math
import re
import statistics
import unicodedata
from collections import Counter

PROTOCOL_VERSION = "hdepic_8b_validation_metrics_v1"
BLIND_REVIEW_SEED = 20261003
BLIND_REVIEW_SIZE = 50
STAGES = ("base", "epoch1", "epoch2")


def protocol():
    """Return a serializable protocol to include in the pre-baseline freeze."""
    return {
        "version": PROTOCOL_VERSION,
        "parser": "strict requested tags in order, no surrounding prose/unknown/duplicate/nested tags; nonempty fields; exactly three nonempty semicolon-delimited next actions",
        "normalization": "decode generated field HTML entities exactly once, then Unicode NFKC, casefold, collapse whitespace; retain punctuation and wording",
        "generation_denominator": "every frozen validation ID; no missing/error/empty/malformed/length/zero-token filtering",
        "format_failure_scoring": "zero for every applicable lexical field when the whole requested response is malformed or the generation row failed",
        "missing_field_diagnostics": "for each requested field, missing means no complete literal opening/closing tag pair, empty means at least one complete pair has whitespace-only content, duplicate means more than one complete pair; count even partial/malformed responses; denominators include all applicable records, and goal only goal-supervised records",
        "goal_denominator": "only records with a goal target annotation; unavailable goals have no lexical goal denominator",
        "next": "full ordered triple exact match and three separate position exact matches, each with all validation IDs in its denominator",
        "length_zero_diagnostics": "finish_reason=length and generated_tokens=0 are independently counted; they never change denominators",
        "checkpoint_selection": "require complete finite epoch1 and epoch2 validation loss runs, then choose lowest mean of per-example assistant-target Explain NLL; exact ties select epoch1; Track NLL reported separately; base not eligible",
        "nll_aggregation": "arithmetic mean of each example's answer-token mean NLL; token-weighted NLL additionally reported as a diagnostic; do not substitute token-weighted mean for checkpoint selection",
        "nll_target_scope": "complete assistant answer tokens under the frozen training-pipeline mask; Explain includes why, optional supervised goal, and three next actions; selection is not why-only token loss",
        "semantic_reason_quality": "unmeasured; normalized exact match is a lexical diagnostic and can penalize valid paraphrases or reward annotation wording",
        "action_classes": "no generated-text mapping to verb/noun IDs; free-text action exact match only",
        "blind_review": {"seed": BLIND_REVIEW_SEED, "size": BLIND_REVIEW_SIZE, "sampling": "rank validation Explain IDs by SHA256 of seed and ID, independent of predictions", "order": "independently hash-randomize base/selected A/B per ID; identity key saved separately", "human_status": "pending; reviewer fields blank"},
    }


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def expected_fields(record):
    if record["task"] not in ("explain", "track"):
        raise ValueError(f"Unexpected task: {record['task']}")
    return ["why" if record["task"] == "explain" else "action", *(["goal"] if "goal" in record["target"] else []), "next"]


def parse_prediction(text, task, has_goal):
    fields = ["why" if task == "explain" else "action", *(["goal"] if has_goal else []), "next"]
    if task not in ("explain", "track"):
        raise ValueError(f"Unexpected task: {task}")
    raw = text if isinstance(text, str) else ""
    field_diagnostics = {}
    for field in fields:
        complete = re.findall(r"<" + field + r">([^<>]*)</" + field + r">", raw, re.DOTALL)
        field_diagnostics[field] = {"missing": not complete, "empty": any(not x.strip() for x in complete), "duplicate": len(complete) > 1, "complete_tag_count": len(complete), "complete_nonempty": len(complete) == 1 and bool(complete[0].strip())}
    if not isinstance(text, str) or not text.strip():
        return {"valid": False, "fields": {}, "next_actions": [], "errors": ["empty_prediction"], "field_diagnostics": field_diagnostics}
    pattern = r"\A\s*" + r"\s*".join(r"<" + f + r">(?P<" + f + r">[^<>]*)</" + f + r">" for f in fields) + r"\s*\Z"
    match = re.fullmatch(pattern, text, re.DOTALL)
    if match is None:
        return {"valid": False, "fields": {}, "next_actions": [], "errors": ["unexpected_tags_order_or_surrounding_text"], "field_diagnostics": field_diagnostics}
    parsed = {f: html.unescape(match.group(f)).strip() for f in fields}
    errors = []
    if any(not parsed[f] for f in fields):
        errors.append("empty_requested_field")
    actions = [a.strip() for a in parsed["next"].split(";")]
    if len(actions) != 3 or any(not a for a in actions):
        errors.append("next_action_count_or_empty_action")
    return {"valid": not errors, "fields": parsed, "next_actions": actions, "errors": errors, "field_diagnostics": field_diagnostics}


def rows_by_id(rows):
    result = {}
    for row in rows:
        identity = row["id"]
        if identity in result:
            raise ValueError(f"Duplicate result ID: {identity}")
        result[identity] = row
    return result


def _expected_id_set(records, results):
    identities = [r["id"] for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate frozen record IDs")
    unexpected = set(results) - set(identities)
    if unexpected:
        raise ValueError(f"Result IDs outside frozen validation records: {sorted(unexpected)[:5]}")


def score_predictions(records, rows):
    """Count structural/lexical results over all expected IDs, without filtering."""
    predictions = rows_by_id(rows)
    _expected_id_set(records, predictions)
    counts = Counter(); errors = Counter(); field_counts = Counter(); per_example = []
    for record in records:
        identity, task = record["id"], record["task"]
        row = predictions.get(identity)
        raw = row.get("prediction", "") if row else ""
        successful = row is not None and row.get("status", "completed") == "completed"
        parsed = parse_prediction(raw, task, "goal" in record["target"])
        valid = successful and parsed["valid"]
        diagnostic = {
            "missing_prediction": row is None,
            "generation_error": row is not None and not successful,
            "empty_output": not isinstance(raw, str) or not raw.strip(),
            "malformed_output": not parsed["valid"],
            "finish_length": bool(row and row.get("finish_reason") == "length"),
            "zero_generated_tokens": bool(row is not None and row.get("generated_tokens") == 0),
        }
        for scope in ("all", task):
            counts[(scope, "expected")] += 1
            counts[(scope, "present")] += row is not None
            counts[(scope, "completed")] += successful
            counts[(scope, "strict_format_valid")] += valid
            for name, flag in diagnostic.items():
                counts[(scope, name)] += flag
        if not parsed["valid"]:
            errors.update(parsed["errors"])
        for field, field_status in parsed["field_diagnostics"].items():
            for scope in ("all", task):
                field_counts[(scope, field, "presence_denominator")] += 1
                for name in ("missing", "empty", "duplicate", "complete_nonempty"):
                    field_counts[(scope, field, name)] += field_status[name]
        primary = "why" if task == "explain" else "action"
        targets = {primary: record["target"]["reason" if task == "explain" else "current_action"]}
        if "goal" in record["target"]:
            targets["goal"] = record["target"]["goal"]
        exact = {f: bool(valid and normalized(parsed["fields"][f]) == normalized(target)) for f, target in targets.items()}
        target_next = record["target"]["next_actions"]
        if len(target_next) != 3:
            raise ValueError(f"Expected three target actions: {identity}")
        next_positions = [bool(valid and normalized(parsed["next_actions"][i]) == normalized(target_next[i])) for i in range(3)]
        exact["next_sequence"] = all(next_positions)
        exact.update({f"next_position_{i+1}": value for i, value in enumerate(next_positions)})
        exact["whole_record"] = all(exact.values())
        for field, match in exact.items():
            for scope in ("all", task):
                field_counts[(scope, field, "denominator")] += 1
                field_counts[(scope, field, "exact_matches")] += match
        per_example.append({"id": identity, "task": task, "goal_supervised": "goal" in record["target"], "strict_format_valid": valid, "parsed_fields": parsed["fields"], "parser_errors": parsed["errors"], "requested_field_diagnostics": parsed["field_diagnostics"], "exact": exact, **diagnostic})
    scopes = {}
    for scope in ("all", "explain", "track"):
        expected = counts[(scope, "expected")]
        values = {name: counts[(scope, name)] for name in ("expected", "present", "completed", "strict_format_valid", "missing_prediction", "generation_error", "empty_output", "malformed_output", "finish_length", "zero_generated_tokens")}
        values["strict_format_rate"] = values["strict_format_valid"] / expected if expected else None
        values["lexical_exact_match"] = {}
        values["requested_field_diagnostics"] = {}
        for field in ("why", "action", "goal", "next"):
            denominator = field_counts[(scope, field, "presence_denominator")]
            values["requested_field_diagnostics"][field] = {"denominator": denominator, **{name: field_counts[(scope, field, name)] for name in ("missing", "empty", "duplicate", "complete_nonempty")}}
        for field in ("why", "action", "goal", "next_sequence", "next_position_1", "next_position_2", "next_position_3", "whole_record"):
            denominator = field_counts[(scope, field, "denominator")]
            numerator = field_counts[(scope, field, "exact_matches")]
            values["lexical_exact_match"][field] = {"exact_matches": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}
        scopes[scope] = values
    return {"protocol_version": PROTOCOL_VERSION, "scopes": scopes, "parser_error_counts": dict(errors), "diagnostics_not_mutually_exclusive": True, "semantic_quality_measured": False, "per_example": per_example}


def aggregate_losses(records, rows):
    results = rows_by_id(rows)
    _expected_id_set(records, results)
    grouped = {"all": [], "explain": [], "track": []}; missing = Counter(); invalid = Counter(); per_example = []
    for record in records:
        row = results.get(record["id"])
        scopes = ("all", record["task"])
        error = None
        mean = row.get("mean_nll") if row else None
        count = row.get("answer_tokens") if row else None
        total = row.get("nll_sum") if row else None
        if row is None:
            error = "missing_loss"
            missing.update(scopes)
        elif row.get("status", "completed") != "completed":
            error = "loss_error"
        elif isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0:
            error = "nonfinite_or_invalid_mean_nll"
        elif isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            error = "zero_or_invalid_answer_tokens"
        elif isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0:
            error = "nonfinite_or_invalid_nll_sum"
        elif not math.isclose(total / count, mean, rel_tol=1e-5, abs_tol=1e-6):
            error = "nll_mean_sum_token_count_inconsistent"
        if error:
            invalid.update(scopes)
        else:
            for scope in scopes:
                grouped[scope].append((float(mean), float(total), count))
        per_example.append({"id": record["id"], "task": record["task"], "mean_nll": float(mean) if isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(mean) else None, "answer_tokens": count, "nll_sum": float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(total) else None, "valid": error is None, "error": error})
    summary = {}
    for scope in ("all", "explain", "track"):
        expected = len(records) if scope == "all" else sum(r["task"] == scope for r in records)
        valid = grouped[scope]
        complete = len(valid) == expected and expected > 0
        example_mean = statistics.fmean(x[0] for x in valid) if valid else None
        tokens = sum(x[2] for x in valid)
        token_mean = sum(x[1] for x in valid) / tokens if tokens else None
        summary[scope] = {"expected_examples": expected, "valid_examples": len(valid), "missing_examples": missing[scope], "invalid_examples": invalid[scope], "complete": complete, "mean_per_example_nll": example_mean if complete else None, "mean_among_valid_examples_diagnostic": example_mean, "token_weighted_nll": token_mean if complete else None, "answer_tokens": tokens}
    return {"protocol_version": PROTOCOL_VERSION, "aggregation": "mean_of_per_example_assistant_target_mean_nll", "scopes": summary, "per_example": per_example}


def select_checkpoint(loss_summaries):
    candidates = []
    for stage in ("epoch1", "epoch2"):
        summary = loss_summaries.get(stage, {})
        all_scope = summary.get("scopes", {}).get("all", {})
        scope = summary.get("scopes", {}).get("explain", {})
        value = scope.get("mean_per_example_nll")
        if all_scope.get("complete") and scope.get("complete") and isinstance(value, (int, float)) and math.isfinite(value):
            candidates.append((float(value), stage))
    if len(candidates) != 2:
        return {"selected_stage": None, "status": "pending_complete_finite_epoch_validation", "selection_metric": "mean_per_example_explain_nll", "candidates": [{"stage": stage, "mean_per_example_explain_nll": value} for value, stage in candidates]}
    best = min(candidates)
    return {"selected_stage": best[1], "status": "selected", "selection_metric": "mean_per_example_explain_nll", "selected_metric": best[0], "candidates": [{"stage": stage, "mean_per_example_explain_nll": value} for value, stage in sorted(candidates, key=lambda x: x[1])], "tie_break": "epoch1", "base_eligible": False}


def blind_review_ids(records, seed=BLIND_REVIEW_SEED, count=BLIND_REVIEW_SIZE):
    eligible = [r["id"] for r in records if r["task"] == "explain"]
    if len(eligible) != len(set(eligible)):
        raise ValueError("Duplicate validation Explain IDs")
    if len(eligible) < count:
        raise ValueError(f"Need {count} validation Explain IDs; only {len(eligible)} available")
    return sorted(eligible, key=lambda identity: hashlib.sha256(f"{PROTOCOL_VERSION}|blind-sample|{seed}|{identity}".encode()).hexdigest())[:count]


def blind_order(identity, seed=BLIND_REVIEW_SEED):
    digest = hashlib.sha256(f"{PROTOCOL_VERSION}|blind-order|{seed}|{identity}".encode()).digest()
    return ("base", "selected") if digest[0] & 1 == 0 else ("selected", "base")
