# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.run_hmgraph_pipeline --video-path /path/to/video.mp4 --output-dir /path/to/run --start-sec 0 --overwrite

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("/data2/yaxuanli/hmgraph_codex")
DEFAULT_MODEL_PYTHON = Path("/data/yaxuanli/miniconda3/envs/avagemma/bin/python")


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]
    description: str


def make_range_label(start_sec: float, end_sec: float | None, range_label: str | None = None) -> str:
    if range_label:
        return _sanitize_label(range_label)
    if float(start_sec) == 0.0 and end_sec is None:
        return "full"
    if end_sec is None:
        return f"{_num(start_sec)}_full"
    return f"{_num(start_sec)}_{_num(end_sec)}"


def output_paths(
    output_dir: str | Path,
    start_sec: float,
    end_sec: float | None,
    range_label: str | None = None,
) -> dict[str, Path]:
    """Return canonical output paths for one run directory.

    Filenames are intentionally not versioned or time-range-tagged.
    Put each run in its own output directory, e.g.:
    runs/<video_id>/full
    runs/<video_id>/0_60
    """
    out = Path(output_dir)
    return {
        "w": out / "w_outputs.jsonl",
        "scene": out / "scene_observations.jsonl",
        "rolling_initial": out / "rolling_context_initial.jsonl",
        "combined_initial": out / "w_scene_combined_initial.jsonl",
        "rolling_v1_1": out / "rolling_context.jsonl",
        "combined_v1_1": out / "w_scene_combined.jsonl",
        "scene_context_report": out / "scene_context_quality_report.json",
        "scene_frames": out / "diagnostics" / "scene_obs_frames",
        "l1_frames": out / "l1_reasoning_frames",
        "segments": out / "l1_segments.jsonl",
        "visual": out / "l1_visual_discovery_outputs.jsonl",
        "reasoner": out / "l1_reasoner_outputs.jsonl",
        "pre_l1": out / "l1_candidates_pre_consolidation.jsonl",
        "raw_l1": out / "l1_candidates_raw.jsonl",
        "raw_ledger": out / "l1_evidence_ledger_raw.jsonl",
        "raw_quality": out / "l1_quality_report_raw.json",
        "final_l1": out / "l1_candidates.jsonl",
        "final_ledger": out / "l1_evidence_ledger.jsonl",
        "postprocess_log": out / "l1_postprocess_log.jsonl",
        "final_quality": out / "l1_quality_report.json",
        "run_manifest": out / "pipeline_run_manifest.json",
        "dry_run": out / "pipeline_dry_run.json",
        "validation_report": out / "pipeline_validation_report.json",
    }


def build_stages(args: argparse.Namespace) -> list[Stage]:
    end_sec = resolve_end_sec(args.video_path, args.start_sec, args.end_sec)
    p = output_paths(args.output_dir, args.start_sec, args.end_sec, args.range_label)
    py = resolve_python_executable(getattr(args, "python_executable", None))
    common_overwrite = ["--overwrite"] if args.overwrite else []
    scene_overwrite_w = ["--overwrite-w-action"] if args.overwrite else []
    gt = ["--gt-jsonl", args.gt_jsonl] if args.gt_jsonl else []
    time_args = ["--start-sec", str(args.start_sec), "--end-sec", str(end_sec)]
    return [
        Stage(
            name="w_scene",
            description="Generate/reuse W_action and generate scene_obs_v1.",
            inputs=[Path(args.video_path)],
            outputs=[p["w"], p["scene"], p["rolling_initial"], p["combined_initial"]],
            command=[
                py, "-m", "code.stages.w_scene",
                "--video-path", args.video_path,
                "--w-action-jsonl", str(p["w"]),
                "--scene-output-jsonl", str(p["scene"]),
                "--rolling-context-jsonl", str(p["rolling_initial"]),
                "--combined-output-jsonl", str(p["combined_initial"]),
                "--scene-frame-output-dir", str(p["scene_frames"]),
                *time_args,
                "--target-fps", "8",
                "--pre-context-sec", "3",
                "--post-context-sec", "0",
                "--context-fps", "2",
                "--context-mode", "past",
                *common_overwrite,
                *scene_overwrite_w,
            ],
        ),
        Stage(
            name="rolling_combined",
            description="Rebuild rolling_context_v1_1 and canonical combined evidence.",
            inputs=[p["w"], p["scene"]],
            outputs=[p["rolling_v1_1"], p["combined_v1_1"], p["scene_context_report"]],
            command=[
                py, "-m", "code.stages.scene_context",
                "--w-action-jsonl", str(p["w"]),
                "--scene-jsonl", str(p["scene"]),
                "--rolling-output-jsonl", str(p["rolling_v1_1"]),
                "--combined-output-jsonl", str(p["combined_v1_1"]),
                "--quality-report-json", str(p["scene_context_report"]),
                *time_args,
                *common_overwrite,
            ],
        ),
        Stage(
            name="l1_reasoner",
            description="Run L1 multimodal reasoner and write raw/intermediate outputs.",
            inputs=[Path(args.video_path), p["combined_v1_1"]],
            outputs=[p["segments"], p["visual"], p["reasoner"], p["pre_l1"], p["raw_l1"], p["raw_ledger"], p["raw_quality"]],
            command=[
                py, "-m", "code.stages.l1_reasoner",
                "--video-path", args.video_path,
                "--combined-jsonl", str(p["combined_v1_1"]),
                "--segments-output-jsonl", str(p["segments"]),
                "--visual-discovery-output-jsonl", str(p["visual"]),
                "--reasoner-output-jsonl", str(p["reasoner"]),
                "--pre-consolidation-output-jsonl", str(p["pre_l1"]),
                "--l1-output-jsonl", str(p["raw_l1"]),
                "--evidence-ledger-output-jsonl", str(p["raw_ledger"]),
                "--quality-report-json", str(p["raw_quality"]),
                *gt,
                *time_args,
                "--max-frames-per-segment", "24",
                "--frame-output-dir", str(p["l1_frames"]),
                *common_overwrite,
            ],
        ),
        Stage(
            name="l1_postprocess",
            description="Postprocess raw L1 candidates into final accepted L1 candidates.",
            inputs=[p["raw_l1"], p["pre_l1"], p["raw_ledger"], p["raw_quality"]],
            outputs=[p["final_l1"], p["final_ledger"], p["postprocess_log"], p["final_quality"]],
            command=[
                py, "-m", "code.stages.l1_postprocess",
                "--raw-candidates-jsonl", str(p["raw_l1"]),
                "--raw-preconsolidation-jsonl", str(p["pre_l1"]),
                "--raw-evidence-ledger-jsonl", str(p["raw_ledger"]),
                "--raw-quality-report-json", str(p["raw_quality"]),
                "--output-candidates-jsonl", str(p["final_l1"]),
                "--output-evidence-ledger-jsonl", str(p["final_ledger"]),
                "--output-postprocess-log-jsonl", str(p["postprocess_log"]),
                "--output-quality-report-json", str(p["final_quality"]),
                *gt,
                *common_overwrite,
            ],
        ),
    ]


def run(args: argparse.Namespace) -> dict:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stages = _select_stages(build_stages(args), args.stage)
    manifest = _base_manifest(args, stages)
    paths = output_paths(args.output_dir, args.start_sec, args.end_sec, args.range_label)
    if args.dry_run:
        _print_dry_run(stages, args)
        manifest["dry_run"] = True
        manifest["completed_at"] = _now()
        _write_json(manifest, paths["dry_run"], overwrite=True)
        return manifest

    for stage in stages:
        record = _run_stage(stage, args)
        manifest["stages"].append(record)
        _write_json(manifest, paths["run_manifest"], overwrite=True)
    manifest["completed_at"] = _now()
    _write_json(manifest, paths["run_manifest"], overwrite=True)
    return manifest


def resolve_end_sec(video_path: str | Path, start_sec: float, end_sec: float | None) -> float:
    if end_sec is not None:
        return float(end_sec)
    duration = _probe_video_duration(video_path)
    if duration <= start_sec:
        raise ValueError(f"Video duration {duration} is not greater than start_sec {start_sec}")
    return duration


def resolve_python_executable(requested: str | None = None) -> str:
    if requested:
        return requested
    if _python_can_import(sys.executable, "PIL"):
        return sys.executable
    if DEFAULT_MODEL_PYTHON.exists() and _python_can_import(str(DEFAULT_MODEL_PYTHON), "PIL"):
        return str(DEFAULT_MODEL_PYTHON)
    return sys.executable


def _run_stage(stage: Stage, args: argparse.Namespace) -> dict:
    print(f"\n=== Stage: {stage.name} ===")
    _check_inputs(stage.inputs)
    if args.skip_existing and _all_nonzero(stage.outputs):
        print(f"Skipping {stage.name}: outputs already exist and are nonzero")
        return _stage_record(stage, "skipped")
    _check_outputs_allowed(stage.outputs, args.overwrite)
    print("Command:", " ".join(stage.command))
    env = _stage_env()
    result = subprocess.run(stage.command, cwd=str(PROJECT_ROOT), text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed with return code {result.returncode}: {stage.name}")
    _check_outputs(stage.outputs)
    return _stage_record(stage, "completed")




def _stage_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _stage_record(stage: Stage, status: str) -> dict:
    return {
        "name": stage.name,
        "description": stage.description,
        "status": status,
        "command": stage.command,
        "inputs": [str(p) for p in stage.inputs],
        "outputs": {str(p): _file_info(p) for p in stage.outputs},
    }


def _base_manifest(args: argparse.Namespace, stages: list[Stage]) -> dict:
    return {
        "created_at": _now(),
        "completed_at": None,
        "args": vars(args),
        "project_git_status": _git_status(),
        "project_root": str(PROJECT_ROOT),
        "stages_planned": [_stage_record(stage, "planned") for stage in stages],
        "stages": [],
    }


def _print_dry_run(stages: list[Stage], args: argparse.Namespace) -> None:
    print("DRY RUN: no model or pipeline stage will be executed.")
    print(f"range_label={make_range_label(args.start_sec, args.end_sec, args.range_label)}")
    print(f"overwrite={args.overwrite} skip_existing={args.skip_existing}")
    for stage in stages:
        print(f"\n=== Stage: {stage.name} ===")
        print(stage.description)
        print("Command:", " ".join(stage.command))
        print("Inputs:")
        for path in stage.inputs:
            print(f"  {path} exists={path.exists()}")
        print("Outputs:")
        for path in stage.outputs:
            print(f"  {path} exists={path.exists()} would_overwrite={path.exists() and args.overwrite}")


def _select_stages(stages: list[Stage], selected: str) -> list[Stage]:
    if selected == "all":
        return stages
    for stage in stages:
        if stage.name == selected:
            return [stage]
    raise ValueError(f"Unknown stage {selected}")


def _check_inputs(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def _check_outputs_allowed(paths: list[Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs without --overwrite: {existing}")


def _check_outputs(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Stage did not produce required nonempty outputs: {missing}")


def _all_nonzero(paths: list[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _file_info(path: Path) -> dict:
    exists = path.exists()
    return {
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "jsonl_lines": count_jsonl_lines(path) if exists and path.suffix == ".jsonl" else None,
    }


def count_jsonl_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _write_json(data: dict, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_status() -> str:
    return "not_tracked"

def _probe_video_duration(video_path: str | Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
        )
        return float(out.strip())
    except Exception as exc:
        raise RuntimeError(
            "Full-range execution requires ffprobe to resolve --end-sec from the video duration. "
            "Provide --end-sec explicitly if ffprobe is unavailable."
        ) from exc


def _python_can_import(python_executable: str, module_name: str) -> bool:
    try:
        subprocess.check_call(
            [python_executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value.strip())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-jsonl")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--range-label")
    parser.add_argument("--python-executable")
    parser.add_argument("--stage", choices=("all", "w_scene", "rolling_combined", "l1_reasoner", "l1_postprocess"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
