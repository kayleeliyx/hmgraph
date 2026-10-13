#!/usr/bin/env python3
"""Overlay W and L1 predictions onto video frames sampled at 1 fps."""

import argparse
import json
import math
import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_obj_name(x):
    if isinstance(x, dict):
        return x.get("name", "?")
    if isinstance(x, str):
        return x
    return "?"


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_w_obj_name(action):
    rich = action.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name"):
        return rich["name"]
    return get_obj_name(action.get("direct_object", "?"))


def build_w_text(w_windows, t):
    matching = [w for w in w_windows if w.get("t_start", -1) <= t < w.get("t_end", -1)]
    if not matching:
        matching = [w for w in w_windows if w.get("window_idx") == t]
    if not matching:
        return ["(no W window)"]
    w = matching[0]
    if not w.get("action_present") or not w.get("valid_actions"):
        return ["(no action)"]
    lines = []
    for a in w["valid_actions"]:
        verb = a.get("verb", "?")
        obj = get_w_obj_name(a)
        conf = a.get("confidence", 0.0)
        lines.append(f"{verb} {obj} (conf {conf:.2f})")
    return lines


def build_l1_text(l1_events, t):
    matching = [e for e in l1_events if e.get("t_start", -1) <= t < e.get("t_end", -1)]
    if not matching:
        return ["(no L1 event)"]
    lines = []
    for e in matching:
        verb = e.get("verb", "?")
        obj = get_obj_name(e.get("direct_object", "?"))
        subtype = e.get("event_subtype", "?")
        conf = e.get("confidence", 0.0)
        halluc = e.get("hallucination_risk", "?")
        lines.append(f"{verb} {obj}  [{subtype}, conf {conf:.2f}, halluc={halluc}]")
        micro = e.get("micro_action_description", "")
        if micro:
            lines.append(f"  {micro}")
    return lines


def build_gt_text(gt_events, t):
    if gt_events is None:
        return None
    matching = [e for e in gt_events if e.get("t_start", -1) <= t < e.get("t_end", -1)]
    if not matching:
        return ["(no GT event)"]
    lines = []
    for e in matching:
        verb = e.get("verb", "?")
        obj = get_obj_name(e.get("direct_object", "?"))
        lines.append(f"{verb} {obj}")
        narration = e.get("narration", "")
        if narration:
            lines.append(f"  {narration.strip()}")
    return lines


PANEL_WIDTH = 900
FONT_SIZE = 28
LINE_SPACING = 6
SECTION_GAP = 18
WRAP_CHARS = 42

COLORS = {
    "header": (255, 255, 255),
    "w": (255, 255, 0),       # cyan in RGB
    "l1": (0, 255, 255),      # yellow in RGB
    "gt": (0, 255, 0),
}
# PIL uses RGB; we fix cyan/yellow:
COLORS = {
    "header": (255, 255, 255),
    "w": (0, 255, 255),
    "l1": (255, 255, 0),
    "gt": (0, 255, 0),
}


def try_load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_panel(frame_h, t, w_lines, l1_lines, gt_lines, font):
    panel = Image.new("RGB", (PANEL_WIDTH, frame_h), (0, 0, 0))
    draw = ImageDraw.Draw(panel)
    y = 10

    def draw_section(header, lines, header_color, line_color):
        nonlocal y
        draw.text((10, y), header, fill=header_color, font=font)
        y += FONT_SIZE + LINE_SPACING + 2
        for line in lines:
            wrapped = textwrap.wrap(line, width=WRAP_CHARS)
            if not wrapped:
                wrapped = [""]
            for wl in wrapped:
                if y + FONT_SIZE > frame_h - 5:
                    return
                draw.text((14, y), wl, fill=line_color, font=font)
                y += FONT_SIZE + LINE_SPACING
        y += SECTION_GAP

    draw_section(f"t = {t}s", [], COLORS["header"], COLORS["header"])
    draw_section("W (1s):", w_lines, COLORS["header"], COLORS["w"])
    draw_section("L1:", l1_lines, COLORS["header"], COLORS["l1"])
    if gt_lines is not None:
        draw_section("GT:", gt_lines, COLORS["header"], COLORS["gt"])

    return np.array(panel)[:, :, ::-1]  # RGB→BGR


def main():
    parser = argparse.ArgumentParser(description="Overlay W/L1 predictions on video")
    parser.add_argument("--video", default="/data2/yaxuanli/HD-EPIC/Videos/P01/P01-20240202-110250.mp4")
    parser.add_argument("--w-jsonl", default="/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/gemma/w_outputs.jsonl")
    parser.add_argument("--l1-jsonl", default="/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/gemma/l1_candidates.jsonl")
    parser.add_argument("--gt-jsonl", default=None)
    parser.add_argument("--out", default="/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/gemma/overlay_1fps.mp4")
    parser.add_argument("--out-fps", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    args = parser.parse_args()

    w_windows = load_jsonl(args.w_jsonl)
    l1_events = load_jsonl(args.l1_jsonl)
    gt_events = load_jsonl(args.gt_jsonl) if args.gt_jsonl else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / vid_fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_second = max(0, int(math.floor(args.start_sec)))
    last_exclusive = int(math.floor(duration)) + 1
    if args.end_sec is not None:
        last_exclusive = min(last_exclusive, int(math.ceil(args.end_sec)))
    seconds = list(range(first_second, last_exclusive))
    n_seconds = len(seconds)
    if not seconds:
        raise ValueError(
            f"No integer-second frames in requested range [{args.start_sec}, {args.end_sec})"
        )
    print(f"Video: {frame_w}x{frame_h} @ {vid_fps} fps, {frame_count:.0f} frames, {duration:.2f}s")
    print(f"Will render {n_seconds} seconds (t={seconds[0]}..{seconds[-1]})")
    print(f"W windows loaded: {len(w_windows)}, L1 events: {len(l1_events)}", end="")
    if gt_events is not None:
        print(f", GT events: {len(gt_events)}")
    else:
        print()

    out_w = frame_w + PANEL_WIDTH
    out_h = frame_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, args.out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {args.out}")

    font = try_load_font(FONT_SIZE)
    example_seconds = [
        seconds[0],
        seconds[len(seconds) // 3],
        seconds[2 * len(seconds) // 3],
    ]
    examples = {}

    for output_idx, t in enumerate(seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        w_lines = build_w_text(w_windows, t)
        l1_lines = build_l1_text(l1_events, t)
        gt_lines = build_gt_text(gt_events, t) if gt_events is not None else None

        panel_bgr = render_panel(frame_h, t, w_lines, l1_lines, gt_lines, font)

        if frame.shape[0] != out_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h))

        combined = np.hstack([frame, panel_bgr])
        writer.write(combined)

        if t in example_seconds:
            examples[t] = {"w": w_lines, "l1": l1_lines}

        if output_idx % 50 == 0:
            print(f"  rendered frame={output_idx}/{n_seconds-1}, source_t={t}s")

    writer.release()
    cap.release()

    print(f"\n=== ACCEPTANCE CHECK ===")
    print(
        f"Seconds rendered: {n_seconds} "
        f"(requested source times {seconds[0]}..{seconds[-1]})"
    )

    for t in sorted(examples):
        print(f"\n--- t={t}s ---")
        print("  W:", "; ".join(examples[t]["w"]))
        print("  L1:", "; ".join(examples[t]["l1"]))

    if os.path.exists(args.out):
        check = cv2.VideoCapture(args.out)
        out_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        print(f"\nOutput file: {args.out}")
        print(f"Output frame count: {out_frames} (expected {n_seconds}) — {'OK' if out_frames == n_seconds else 'MISMATCH'}")
    else:
        print(f"\nERROR: output file not found: {args.out}")


if __name__ == "__main__":
    main()
