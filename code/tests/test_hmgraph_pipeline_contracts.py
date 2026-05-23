# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.tests.test_hmgraph_pipeline_contracts

from __future__ import annotations

import tempfile
from pathlib import Path

from code.run_hmgraph_pipeline import build_stages, count_jsonl_lines, make_range_label, output_paths
from code.stages.l1_postprocess import _ensure_final_l1_ledger_coverage
from code.validate_hmgraph_pipeline_outputs import _check_l1_schema, _check_times, read_jsonl


def test_range_label_generation() -> None:
    assert make_range_label(1, 2) == "1_2"
    assert make_range_label(0, 60) == "0_60"
    assert make_range_label(120, 180) == "120_180"
    assert make_range_label(0, None) == "full"
    assert make_range_label(0, 60, "custom range") == "custom_range"


def test_jsonl_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        assert count_jsonl_lines(path) == 2
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_l1_schema_and_confidence() -> None:
    record = {
        "l1_id": "l1",
        "t_start": 0.0,
        "t_end": 1.0,
        "micro_action_description": "open door",
        "verb": "open",
        "direct_object": {"name": "door", "part_of": "cabinet"},
        "event_subtype": "core_effect",
        "evidence_type": "observed",
        "confidence": 0.9,
        "structured_evidence": {"w_action_windows": [0]},
    }
    report = {"valid": True, "errors": [], "warnings": []}
    _check_l1_schema([record], report)
    assert report["valid"], report
    bad = dict(record, confidence=0.99)
    report = {"valid": True, "errors": [], "warnings": []}
    _check_l1_schema([bad], report)
    assert not report["valid"]


def test_time_validation() -> None:
    report = {"valid": True, "errors": [], "warnings": []}
    _check_times([{"t_start": 0.0, "t_end": 1.0}], 0.0, 5.0, "x", report)
    assert report["valid"]
    _check_times([{"t_start": -1.0, "t_end": 1.0}], 0.0, 5.0, "x", report)
    assert not report["valid"]


def test_output_path_construction() -> None:
    paths = output_paths("/tmp/out", 120, 180)

    assert paths["w"].name == "w_outputs.jsonl"
    assert paths["scene"].name == "scene_observations.jsonl"
    assert paths["rolling_v1_1"].name == "rolling_context.jsonl"
    assert paths["combined_v1_1"].name == "w_scene_combined.jsonl"
    assert paths["final_l1"].name == "l1_candidates.jsonl"
    assert paths["final_ledger"].name == "l1_evidence_ledger.jsonl"
    assert paths["final_quality"].name == "l1_quality_report.json"
    assert paths["run_manifest"].name == "pipeline_run_manifest.json"
    assert paths["validation_report"].name == "pipeline_validation_report.json"
    assert paths["l1_frames"].name == "l1_reasoning_frames"
    assert paths["scene_frames"].name == "scene_obs_frames"

    for path in paths.values():
        name = path.name
        assert "120_180" not in name

    full = output_paths("/tmp/out", 0, None)
    assert full["final_l1"].name == "l1_candidates.jsonl"


def test_dry_run_stage_construction() -> None:
    class Args:
        video_path = "/tmp/video.mp4"
        output_dir = "/tmp/out"
        gt_jsonl = "/tmp/gt.jsonl"
        start_sec = 0.0
        end_sec = 5.0
        range_label = None
        overwrite = False
    stages = build_stages(Args())
    assert [s.name for s in stages] == ["w_scene", "rolling_combined", "l1_reasoner", "l1_postprocess"]
    assert any("code.stages.w_scene" in " ".join(s.command) for s in stages)
    l1_stage = next(s for s in stages if s.name == "l1_reasoner")
    command = " ".join(l1_stage.command)
    assert "--frame-output-dir" in command
    assert "l1_reasoning_frames" in command


def test_evidence_ledger_id_matching_contract() -> None:
    final_ids = {"l1_a"}
    ledger = [{"visual_candidate_ledger": [{"linked_l1_event_ids": ["l1_a"]}], "w_action_audit": []}]
    linked = set()
    for rec in ledger:
        for entry in rec["visual_candidate_ledger"]:
            linked.update(entry["linked_l1_event_ids"])
    assert final_ids <= linked


def test_dry_run_does_not_create_model_outputs_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = output_paths(tmp, 0, 5)
        assert not paths["w"].exists()
        assert not paths["final_l1"].exists()


def test_postprocess_ledger_coverage_fallback() -> None:
    candidates = [
        {
            "l1_id": "l1_a",
            "segment_id": "seg_a",
            "micro_action_description": "pick up mug",
            "raw_reasoner_event": {"event_id": "raw_a"},
            "structured_evidence": {},
        }
    ]
    additions, log = _ensure_final_l1_ledger_coverage(candidates, [], [])
    assert additions[0]["linked_l1_event_ids"] == ["l1_a"]
    assert additions[0]["segment_id"] == "seg_a"
    assert additions[0]["source_reasoner_event_id"] == "raw_a"
    assert log and log[0]["l1_id"] == "l1_a"


def main() -> int:
    tests = [
        test_range_label_generation,
        test_jsonl_helpers,
        test_l1_schema_and_confidence,
        test_time_validation,
        test_output_path_construction,
        test_dry_run_stage_construction,
        test_evidence_ledger_id_matching_contract,
        test_postprocess_ledger_coverage_fallback,
        test_dry_run_does_not_create_model_outputs_contract,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
