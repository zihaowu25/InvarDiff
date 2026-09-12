#!/usr/bin/env python3
"""Aggregate candidate summaries, detect coarse/fine elbows, and plot curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _line_chart(
    frame: pd.DataFrame,
    module: str,
    elbow: float | None,
    path: Path,
    metric_label: str = "LPIPS-Alex macro mean",
):
    width, height = 1100, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 100, 55, 1040, 570
    xs = frame["threshold"].to_numpy(float)
    ys = frame["lpips"].to_numpy(float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = min(0.0, float(ys.min())), max(0.01, float(ys.max()))
    scale_x = lambda x: left + (right - left) * (x - xmin) / max(xmax - xmin, 1e-9)
    scale_y = lambda y: bottom - (bottom - top) * (y - ymin) / max(ymax - ymin, 1e-9)
    draw.text((left, 15), f"{module}: threshold vs {metric_label}", fill="black")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    points = [(scale_x(x), scale_y(y)) for x, y in zip(xs, ys)]
    if len(points) > 1:
        draw.line(points, fill=(45, 118, 182), width=4)
    for point, x, y in zip(points, xs, ys):
        draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=(45, 118, 182))
        draw.text((point[0]+7, point[1]-18), f"{x:.2f},{y:.3f}", fill="black")
    if elbow is not None:
        x = scale_x(elbow)
        draw.line((x, top, x, bottom), fill="crimson", width=3)
        draw.text((x+5, top), f"elbow={elbow:.2f}", fill="crimson")
    draw.text((right-120, bottom+30), "threshold", fill="black")
    draw.text((10, top), "LPIPS", fill="black")
    image.save(path)


def detect_elbow(frame: pd.DataFrame, step: float) -> float | None:
    frame = frame.sort_values("threshold")
    values = frame["lpips"].to_numpy(float)
    thresholds = frame["threshold"].to_numpy(float)
    increments = np.diff(values)
    minimum = 0.03 if step >= 0.099 else 0.003
    positives = []
    for index, increment in enumerate(increments):
        history = np.asarray(positives, dtype=float)
        significant = increment >= minimum
        if history.size >= 2:
            significant = significant and increment >= 2.0 * np.median(history)
        if significant:
            return float(thresholds[index + 1])
        if increment > 0:
            positives.append(float(increment))
    return None


def _module_hit_rate(summary_path: Path, item: dict, module: str) -> float:
    if module == "step":
        recorded = item.get("cache_hit_rates", {}).get("step")
        if recorded is not None:
            return float(recorded)
        book_path = (
            summary_path.parent / "cache_books" / f"{item['candidate_id']}.json"
        )
        step_book = json.loads(
            book_path.read_text(encoding="utf-8")
        )["step_cache_book"]
        return sum(step_book) / len(step_book) if step_book else float("nan")
    rate_key = (
        "double.joint_attn"
        if module in ("double.img_attn", "double.txt_attn")
        else module
    )
    recorded = item.get("module_hit_rates", {}).get(rate_key)
    if recorded is not None:
        return float(recorded)
    book_path = (
        summary_path.parent / "cache_books" / f"{item['candidate_id']}.json"
    )
    payload = json.loads(book_path.read_text(encoding="utf-8"))
    module_books = payload["module_cache_book"]
    if rate_key == "double.joint_attn":
        flat = [
            bool(img_value and txt_value)
            for img_step, txt_step in zip(
                module_books["double.img_attn"],
                module_books["double.txt_attn"],
            )
            for img_value, txt_value in zip(img_step, txt_step)
        ]
    else:
        flat = [
            bool(value) for step in module_books[module] for value in step
        ]
    return sum(flat) / len(flat) if flat else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        action="append",
        help="Candidate root; repeat to combine coarse endpoints with a fine grid.",
    )
    parser.add_argument("--module", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--threshold-min", type=float)
    parser.add_argument("--threshold-max", type=float)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="Optional single cumulative q=0 summary from the previous module.",
    )
    parser.add_argument(
        "--zero-lpips",
        type=float,
        help="Optional measured threshold-0 baseline to prepend to the curve.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        help="Use one prompt's LPIPS instead of the summary macro mean.",
    )
    args = parser.parse_args()

    def selected_lpips(item: dict) -> float:
        if args.prompt_index is None:
            return float(item["lpips_macro_mean"])
        values = item["lpips_prompt_means"]
        if not 0 <= args.prompt_index < len(values):
            raise ValueError(
                f"prompt index {args.prompt_index} is unavailable for "
                f"{item['candidate_id']}"
            )
        return float(values[args.prompt_index])

    rows = []
    for root in args.root:
        for path in sorted(root.glob("*/summary.json")):
            item = json.loads(path.read_text())
            threshold = float(
                item["step_threshold"]
                if args.module == "step"
                else item["thresholds"].get(args.module, 0.0)
            )
            rows.append({
                "candidate_id": item["candidate_id"],
                "module": args.module,
                "threshold": threshold,
                "lpips": selected_lpips(item),
                "wall_s": float(item["generation_wall_s_mean"]),
                "module_hit_rate": _module_hit_rate(path, item, args.module),
            })
    if args.baseline_summary is not None:
        path = args.baseline_summary
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "candidate_id": item["candidate_id"],
            "module": args.module,
            "threshold": 0.0,
            "lpips": selected_lpips(item),
            "wall_s": float(item["generation_wall_s_mean"]),
            "module_hit_rate": _module_hit_rate(path, item, args.module),
        })
    if not rows:
        raise SystemExit("No candidate summaries found")
    if args.zero_lpips is not None and not any(
        abs(row["threshold"]) < 1e-12 for row in rows
    ):
        rows.append({
            "candidate_id": "threshold_zero_baseline",
            "module": args.module,
            "threshold": 0.0,
            "lpips": args.zero_lpips,
            "wall_s": float("nan"),
            "module_hit_rate": 0.0,
        })
    frame = pd.DataFrame(rows).sort_values("threshold")
    if args.threshold_min is not None:
        frame = frame[frame["threshold"] >= args.threshold_min]
    if args.threshold_max is not None:
        frame = frame[frame["threshold"] <= args.threshold_max]
    if frame.empty:
        raise SystemExit("No candidates remain inside the threshold range")
    elbow = detect_elbow(frame, args.step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"{args.module}.csv", index=False)
    (args.output_dir / f"{args.module}.json").write_text(
        json.dumps({"module": args.module, "step": args.step, "elbow": elbow}, indent=2) + "\n"
    )
    metric_label = (
        "LPIPS-Alex macro mean"
        if args.prompt_index is None
        else f"LPIPS-Alex prompt {args.prompt_index}"
    )
    _line_chart(
        frame,
        args.module,
        elbow,
        args.output_dir / f"{args.module}.png",
        metric_label,
    )


if __name__ == "__main__":
    main()
