#!/usr/bin/env python3
"""Package saved validation artifacts for review; stdlib only, no inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs/hdepic_sft_8b_v1"
DEFAULT_OUTPUT = RUN / "hdepic_sft_8b_v1_review.zip"
REQUESTED = (
    "execution_report.md", "frozen_config.json", "frozen_evaluation.json",
    "validation_report.md", "predictions/base/generation.jsonl",
    "predictions/epoch1/generation.jsonl", "predictions/epoch2/generation.jsonl",
    "review/blind_why_review.csv",
)
SOURCE_FILES = (
    "code/__init__.py", "code/hdepic_experiment/__init__.py",
    "code/hdepic_experiment/common.py", "code/hdepic_experiment/eval.py",
    "code/hdepic_experiment/metrics.py", "code/hdepic_experiment/preflight.py",
    "code/hdepic_experiment/report.py", "code/hdepic_preparation/__init__.py",
    "code/hdepic_preparation/build.py", "code/hdepic_preparation/swift_smoke_audit.py",
    "code/e0/__init__.py", "code/e0/dataset.py", "code/e0/frames.py",
    "code/tests/test_hdepic_experiment_metrics.py",
    "scripts/package_hdepic_sft_8b_v1_review.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def jsonl_bytes(rows):
    return b"".join((json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode() for r in rows)


def rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(RUN) and output.suffix == ".zip"
    payloads, origins, original_hashes = {}, {}, {}

    def copy(path, name):
        path = Path(path).resolve()
        assert path.is_relative_to(PROJECT) and path.is_file()
        assert "identity_key" not in name and name not in payloads
        data = path.read_bytes()
        original_hashes[str(path)] = digest(data)
        payloads[name] = data
        origins[name] = {"source_path": str(path), "source_sha256": digest(data)}
        return data

    for name in REQUESTED:
        copy(RUN / name, name)
    config = json.loads(payloads["frozen_config.json"])
    dataset = Path(config["paths"]["dataset"])
    assert dataset == PROJECT / "datasets/hdepic_sft_v2_repaired"
    data_info = config["data"]["validation"]
    source_data = {}
    for key in ("text", "export", "scoring", "prompts"):
        raw = Path(data_info[key]).read_bytes()
        assert digest(raw) == data_info[key + "_sha256"]
        original_hashes[data_info[key]] = digest(raw)
        source_data[key] = rows(raw)
    text, exports, scoring, prompts = [source_data[k] for k in ("text", "export", "scoring", "prompts")]
    ids = [r["id"] for r in text]
    assert len(ids) == len(set(ids)) == 2209
    assert Counter(r["task"] for r in text) == {"explain": 228, "track": 1981}
    assert all(r["partition"] == "validation" and r["participant"] == "P07" for r in text)
    assert ids == json.loads(payloads["frozen_evaluation.json"])["ordered_validation_ids"]
    assert len(exports) == len(scoring) == len(prompts) == 2209
    references, inputs = [], []
    for i, (record, export, sidecar, prompt) in enumerate(zip(text, exports, scoring, prompts)):
        assert sidecar["id"] == ids[i] and sidecar["row_index"] == i
        assert prompt == {"messages": export["messages"][:2], "images": export["images"]}
        assert [m["role"] for m in export["messages"]] == ["system", "user", "assistant"]
        assert len(prompt["images"]) == 16
        references.append({"id": ids[i], "row_index": i, "partition": "validation",
                           "participant": record["participant"], "video_id": record["video_id"],
                           "task": record["task"], "query_time_sec": record["query_time_sec"],
                           "target": record["target"], "reference_assistant_text": export["messages"][2]["content"]})
        inputs.append({"id": ids[i], "row_index": i, "partition": "validation", "task": record["task"],
                       "messages": prompt["messages"], "images": prompt["images"],
                       "observed_input": record["input"]})
    payloads["references/validation.references.jsonl"] = jsonl_bytes(references)
    payloads["references/validation.inputs.jsonl"] = jsonl_bytes(inputs)
    copy(data_info["scoring"], "references/validation.scoring.jsonl")

    review = list(csv.DictReader(io.StringIO(payloads["review/blind_why_review.csv"].decode())))
    assert len(review) == len({r["id"] for r in review}) == 50
    by_id = {r["id"]: (i, r) for i, r in enumerate(text)}
    companion = []
    for blind in review:
        i, record = by_id[blind["id"]]
        reason = record["scoring_metadata"]["reason_annotation"]
        assert record["task"] == "explain"
        assert blind["action_question_text"] == record["input"]["action_question_text"]
        assert blind["original_narration"] == reason["original_narration"]
        assert blind["source_stated_reason"] == record["target"]["reason"]
        assert int(blind["reason_char_start"]) == reason["char_start"]
        assert int(blind["reason_char_end"]) == reason["char_end"]
        assert reason["original_narration"][reason["char_start"]:reason["char_end"]] == reason["reason_text"]
        companion.append({"blind_row": int(blind["blind_row"]), "id": record["id"],
                          "action_question_text": record["input"]["action_question_text"],
                          "original_narration": reason["original_narration"],
                          "source_narration_id": reason["narration_id"],
                          "reason_annotation": reason, "reference_target": record["target"],
                          "action_annotation": record["scoring_metadata"]["current_action"],
                          "query_time_sec": record["query_time_sec"], "observed_input": record["input"],
                          "prompt_messages": prompts[i]["messages"], "individual_images": prompts[i]["images"]})
    payloads["review/why_reference_sources.jsonl"] = jsonl_bytes(companion)

    stage_checks = {}
    for stage in ("base", "epoch1", "epoch2"):
        generated = rows(payloads[f"predictions/{stage}/generation.jsonl"])
        assert [r["id"] for r in generated] == ids
        assert all(r["partition"] == "validation" and r["stage"] == stage and r["status"] == "completed" for r in generated)
        for record in generated:
            audit = record["preprocessing_audit"]
            actual = audit["actual_generation_config"]
            assert audit["image_count"] == 16 and audit["complete_prompt_exact"]
            assert audit["assistant_target_in_generation_input"] is False
            assert actual["do_sample"] is False and actual["num_beams"] == 1 and actual["max_new_tokens"] == 256
            assert actual["eos_token_id"] == [151645, 151643] and actual["pad_token_id"] == 151643
        stage_checks[stage] = {"records": len(generated), "strict_format_valid": sum(r["strict_format_valid"] for r in generated),
                               "saved_parser_error_counts": dict(Counter(e for r in generated for e in r["parser_errors"])),
                               "finish_reason_counts": dict(Counter(r["finish_reason"] for r in generated)),
                               "generated_token_range": [min(r["generated_tokens"] for r in generated), max(r["generated_tokens"] for r in generated)],
                               "actual_template_stop_word_variants": sorted({tuple(r["preprocessing_audit"]["actual_generation_config"]["stop_words"]) for r in generated}),
                               "all_saved_hooks_prompt_only_16_images_greedy_max256": True}
        for filename in ("loss.jsonl", "metadata.json", "actual_pipeline.json", "completed.json", "metrics.per_example.jsonl", "loss.metrics.per_example.jsonl"):
            copy(RUN / f"predictions/{stage}/{filename}", f"predictions/{stage}/{filename}")
    payloads["review/saved_generation_audit_summary.json"] = json_bytes(stage_checks)
    for name in ("metrics.json", "experiment_completed.json", "failures_and_fixes.json", "preflight/preflight_audit.json",
                 "provenance/package_versions.json", "provenance/pip-freeze.txt", "provenance/code_snapshot.json", "provenance/torchrun_boundary_fix.json"):
        copy(RUN / name, name)
    for name in SOURCE_FILES:
        copy(PROJECT / name, "source/" + name)

    payloads["README_REVIEW.md"] = b"""# HD-EPIC SFT v1 review archive

This archive contains the eight requested saved artifacts unchanged, evaluation/parser/metric source and helper dependencies, all 2,209 validation reference targets and scoring rows keyed by record ID, and actual prompt-only inputs. No inference, training, new scores, parser changes, or dataset edits were performed to create it. Human reason-label review remains pending. Zero strictly valid base outputs is a custom-format statistic, not semantic accuracy.

Start with predictions/base/generation.jsonl: prediction and raw_generated_tokens_text retain raw answers; parser_errors and field_diagnostics show the unchanged parser decisions. review/saved_generation_audit_summary.json summarizes only saved decisions and effective generation hooks. The base has 1,946 tag/order/prose failures and 263 next-action delimiter/count failures. Review raw answers and prompts before making a quality claim.

The frozen metadata lists template_stop_words=[]; actual hooks record <|endoftext|> and <|im_end|>. Both actual strings correspond to the native Qwen EOS handling. Actual greedy/max-token/EOS settings are saved per prediction. This documentation discrepancy is retained, not corrected retroactively; these stop strings differ from the requested </why>, </action>, </goal>, </next> closing tags. The actual_pipeline canonical_generation_config is the original model default; the per-row actual_generation_config shows the effective request overrides.

references/validation.references.jsonl includes structured target and exact reference_assistant_text. references/validation.inputs.jsonl contains ID-aligned system/user messages, image paths, and observed inputs, with no assistant answer. references/validation.scoring.jsonl is the original byte-identical scoring sidecar, including verb/noun IDs, source annotations, reason spans/offsets, goals, and future actions. The three files share the frozen original validation order. No test or second-stage examples are included.

review/blind_why_review.csv is byte-identical to the original; answer A/B order and reviewer entries are preserved. It already includes action_question_text and original_narration for all 50 rows. review/why_reference_sources.jsonl adds the exact source reason provenance, annotated target action/timestamps, inputs and paths in the same blind-row order, without model assignments. The identity-key file is deliberately outside this archive. For blinded human review distribute only the CSV and its input/source companion: this complete technical archive also contains named raw model outputs and ordering source code, which a reviewer could use to infer identities.

source/code/hdepic_experiment/eval.py implements the installed local inference and teacher-forced loss pipeline; metrics.py contains the frozen parser and metric definitions; report.py aggregates and creates the review CSV. preflight.py and swift_smoke_audit.py implement complete-target/image/mask audits. Supporting imports and existing metric tests are included unchanged. Source is for inspection; the original evaluator requires the server's pinned environment, image paths and checkpoint. No automatic evaluation/training launcher is run by opening this archive.

No weights, adapters, optimizer states, image bytes, full frame cache, or model-identity key are included. Images are paths only, so this is a text/source review package, not independent visual validation. MANIFEST.json records each member's SHA256/bytes, original source hashes, source validation export hashes, and alignment checks. The adjacent ZIP.sha256 file checks the archive as a whole.
"""
    checks = {"validation_records": 2209, "task_counts": {"explain": 228, "track": 1981}, "review_examples": 50,
              "all_reference_scoring_prompt_and_prediction_ids_aligned": True, "review_question_narration_spans_match_source": True,
              "blind_csv_byte_identical": True, "blind_answer_order_preserved": True, "identity_key_included": False,
              "model_weights_included": False, "image_bytes_included": False, "test_records_included": False,
              "new_training_or_inference": False, "human_reason_label_review": "pending"}
    files = {name: {"sha256": digest(data), "bytes": len(data), **origins.get(name, {})} for name, data in sorted(payloads.items())}
    manifest = {"schema": "hdepic_sft_8b_v1_review_package_v1", "files": files, "checks": checks,
                "source_validation_exports": {key: {"path": data_info[key], "sha256": data_info[key + "_sha256"]} for key in ("text", "export", "scoring", "prompts")}}
    manifest["package_identity_sha256"] = digest(json_bytes(manifest))
    payloads["MANIFEST.json"] = json_bytes(manifest)
    assert all("identity_key" not in name and Path(name).suffix in {".json", ".jsonl", ".md", ".csv", ".py", ".txt"} for name in payloads)

    def verify(path):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == set(payloads)
            assert len(archive.namelist()) == len(payloads)
            assert all(digest(archive.read(name)) == digest(data) for name, data in payloads.items())

    if output.exists():
        verify(output)
    else:
        handle, temporary = tempfile.mkstemp(prefix=".review-zip-", suffix=".zip", dir=RUN)
        os.close(handle)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name, data in sorted(payloads.items()):
                    archive.writestr(name, data)
            verify(temporary)
            assert not output.exists(), "Refusing to replace an existing archive"
            os.replace(temporary, output)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    assert all(digest(Path(path).read_bytes()) == value for path, value in original_hashes.items()), "Source artifacts changed during packaging"
    archive_sha = digest(output.read_bytes())
    output.with_suffix(".zip.sha256").write_text(f"{archive_sha}  {output.name}\n")
    audit = {"passed": True, "zip": str(output), "zip_sha256": archive_sha, "zip_bytes": output.stat().st_size,
             "members": len(payloads), "package_identity_sha256": manifest["package_identity_sha256"],
             "original_files_verified_unchanged": len(original_hashes), "checks": checks}
    output.with_suffix(".audit.json").write_bytes(json_bytes(audit))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
