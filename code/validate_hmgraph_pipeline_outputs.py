# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.validate_hmgraph_pipeline_outputs --output-dir /path/to/run --start-sec 0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from .run_hmgraph_pipeline import output_paths, resolve_end_sec
except ImportError:
    from run_hmgraph_pipeline import output_paths, resolve_end_sec


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def validate_outputs(
    output_dir: str | Path,
    start_sec: float,
    end_sec: float | None,
    range_label: str | None = None,
    video_path: str | None = None,
) -> dict:
    resolved_end = resolve_end_sec(video_path, start_sec, end_sec) if end_sec is None and video_path else end_sec
    paths = output_paths(output_dir, start_sec, end_sec, range_label)
    required = [
        paths["w"], paths["scene"], paths["rolling_v1_1"], paths["combined_v1_1"],
        paths["final_l1"], paths["final_ledger"], paths["final_quality"],
    ]
    report = {"valid": True, "errors": [], "warnings": [], "files": {}, "counts": {}}
    for path in required:
        _require_file(path, report)
    if not report["valid"]:
        return _finish(report, paths["validation_report"])

    w = _load(paths["w"], report)
    scene = _load(paths["scene"], report)
    combined = _load(paths["combined_v1_1"], report)
    l1 = _load(paths["final_l1"], report)
    ledger = _load(paths["final_ledger"], report)
    quality = _load_json(paths["final_quality"], report)
    if resolved_end is not None:
        expected = int(round(resolved_end - start_sec))
        _approx_count("W_action", len(w), expected, report)
        _approx_count("scene_obs", len(scene), expected, report)
        _approx_count("combined", len(combined), expected, report)
        _check_times(w, start_sec, resolved_end, "W_action", report)
        _check_times(scene, start_sec, resolved_end, "scene_obs", report)
        _check_times(combined, start_sec, resolved_end, "combined", report)
        _check_times(l1, start_sec, resolved_end, "L1", report)
    else:
        report["warnings"].append("end_sec unavailable; skipped approximate count and upper-bound time checks")
    _check_combined_schema(combined, report)
    _check_l1_schema(l1, report)
    _check_ledger(l1, ledger, report)
    _check_quality_count(l1, quality, report)
    return _finish(report, paths["validation_report"])


def _require_file(path: Path, report: dict) -> None:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    report["files"][str(path)] = {"exists": exists, "size": size}
    if not exists or size == 0:
        report["valid"] = False
        report["errors"].append(f"required file missing or empty: {path}")


def _load(path: Path, report: dict) -> list[dict]:
    try:
        records = read_jsonl(path)
        report["counts"][path.name] = len(records)
        return records
    except Exception as exc:
        report["valid"] = False
        report["errors"].append(str(exc))
        return []


def _load_json(path: Path, report: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"invalid JSON {path}: {exc}")
        return {}


def _approx_count(label: str, count: int, expected: int, report: dict) -> None:
    if abs(count - expected) > max(1, int(expected * 0.1)):
        report["valid"] = False
        report["errors"].append(f"{label} count {count} is not approximately expected {expected}")


def _check_times(records: list[dict], start: float, end: float, label: str, report: dict) -> None:
    for idx, record in enumerate(records):
        t0 = _float(record.get("t_start", record.get("t_start_est")), None)
        t1 = _float(record.get("t_end", record.get("t_end_est")), None)
        if t0 is None or t1 is None:
            continue
        if t0 < start - 1e-6 or t1 > end + 1e-6 or t1 < t0:
            report["valid"] = False
            report["errors"].append(f"{label}[{idx}] time outside range: {t0}-{t1}")


def _check_combined_schema(records: list[dict], report: dict) -> None:
    for idx, record in enumerate(records):
        if "W_action" not in record or "scene_observation" not in record:
            report["valid"] = False
            report["errors"].append(f"combined[{idx}] missing W_action or scene_observation")
        if "rolling_context_v1_1" not in record and "rolling_context" not in record:
            report["valid"] = False
            report["errors"].append(f"combined[{idx}] missing rolling context")


def _check_l1_schema(records: list[dict], report: dict) -> None:
    required = {"l1_id", "t_start", "t_end", "micro_action_description", "verb", "direct_object", "event_subtype", "evidence_type", "confidence"}
    for idx, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            report["valid"] = False
            report["errors"].append(f"L1[{idx}] missing fields {missing}")
        if record.get("evidence_type") == "hypothesized":
            report["valid"] = False
            report["errors"].append(f"L1[{idx}] has hypothesized evidence_type")
        if _float(record.get("confidence"), 0.0) > 0.98:
            report["valid"] = False
            report["errors"].append(f"L1[{idx}] confidence > 0.98")
        if not record.get("visual_evidence") and not record.get("structured_evidence"):
            report["valid"] = False
            report["errors"].append(f"L1[{idx}] missing visual_evidence and structured_evidence")


def _check_ledger(candidates: list[dict], ledger_records: list[dict], report: dict) -> None:
    final_ids = {str(c.get("l1_id")) for c in candidates}
    linked = set()
    for rec in ledger_records:
        for section in ("visual_candidate_ledger", "w_action_audit"):
            for entry in rec.get(section) or []:
                linked.update(str(x) for x in entry.get("linked_l1_event_ids") or [])
    missing = sorted(final_ids - linked)
    if missing:
        report["warnings"].append(f"ledger does not explicitly link final ids: {missing}")


def _check_quality_count(candidates: list[dict], quality: dict, report: dict) -> None:
    summary = quality.get("summary") or {}
    expected = summary.get("n_final_l1_candidates")
    if expected is not None and int(expected) != len(candidates):
        report["valid"] = False
        report["errors"].append(f"quality report candidate count {expected} != actual {len(candidates)}")


def _finish(report: dict, output_path: Path) -> dict:
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"validation_report={output_path}")
    print(f"valid={report['valid']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
    return report


def _float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--range-label")
    parser.add_argument("--video-path")
    args = parser.parse_args(argv)
    report = validate_outputs(args.output_dir, args.start_sec, args.end_sec, args.range_label, args.video_path)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
