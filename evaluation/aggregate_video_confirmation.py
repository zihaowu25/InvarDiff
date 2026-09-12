#!/usr/bin/env python3
"""Aggregate 8-prompt x 2-seed video metrics and temporal diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Metric values must be finite and non-empty")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _plot(rows: list[dict], path: Path) -> None:
    width, height = 1200, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 55, 1140, 525
    values = [row["lpips_mean"] for row in rows]
    high = max(0.01, max(values))
    draw.text((left, 15), "Confirmation LPIPS by paired prompt and seed", fill="black")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    bar_width = max(10, int((right - left) / len(rows) * 0.65))
    for index, (row, value) in enumerate(zip(rows, values)):
        x = left + (right - left) * (index + 0.5) / len(rows)
        y = bottom - (bottom - top) * value / high
        draw.rectangle((x - bar_width / 2, y, x + bar_width / 2, bottom), fill=(45, 118, 182))
        draw.text((x - 15, bottom + 8), f"{row['seed']}:{row['prompt_index']}", fill="black")
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        action="append",
        required=True,
        help="Metrics directory for one seed; repeat exactly twice.",
    )
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=8)
    args = parser.parse_args()
    if len(args.metrics_root) != len(args.seed):
        parser.error("--metrics-root and --seed counts must match")

    rows = []
    frame_lpips, temporal_drift, temporal_l1 = [], [], []
    for root, seed in zip(args.metrics_root, args.seed):
        paths = sorted(root.glob("p*.json"))
        if len(paths) != args.expected_prompts:
            raise ValueError(
                f"{root}: expected {args.expected_prompts} metrics, got {len(paths)}"
            )
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            prompt_index = int(path.stem.removeprefix("p"))
            lpips = payload["frame_lpips"]
            temporal = payload["temporal"]
            row = {
                "seed": seed,
                "prompt_index": prompt_index,
                "lpips_mean": float(lpips["mean"]),
                "lpips_p95": float(lpips["p95"]),
                "lpips_max": float(lpips["max"]),
                "adjacent_lpips_drift_mean": float(
                    temporal["adjacent_lpips_abs_delta"]["mean"]
                ),
                "temporal_delta_l1_mean": float(
                    temporal["temporal_delta_l1"]["mean"]
                ),
            }
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite metric in {path}")
            rows.append(row)
            frame_lpips.extend(float(x) for x in payload["per_frame_lpips"])
            temporal_drift.extend(
                float(x) for x in temporal["per_transition_lpips_abs_delta"]
            )
            temporal_l1.extend(
                float(x) for x in temporal["per_transition_delta_l1"]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "pairs": len(rows),
        "seeds": args.seed,
        "prompts_per_seed": args.expected_prompts,
        "prompt_macro_lpips": _stat([row["lpips_mean"] for row in rows]),
        "all_frame_lpips": _stat(frame_lpips),
        "adjacent_lpips_abs_delta": _stat(temporal_drift),
        "temporal_delta_l1": _stat(temporal_l1),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _plot(rows, args.output_dir / "paired_lpips.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
