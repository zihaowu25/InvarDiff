#!/usr/bin/env python3
"""Aggregate paired reference-relative LPIPS deltas with a bootstrap bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def paired_bootstrap_upper(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 2027,
) -> tuple[float, float]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("baseline and candidate must be aligned 1-D arrays")
    if baseline.size < 2:
        raise ValueError("at least two paired observations are required")
    delta = candidate.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    bootstrap_means = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.percentile(bootstrap_means, 95))


def _load_means(root: Path) -> dict[str, float]:
    result = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_lpips = payload.get("frame_lpips")
        if isinstance(frame_lpips, dict) and "mean" in frame_lpips:
            result[str(path.relative_to(root))] = float(frame_lpips["mean"])
    return result


def _plot(deltas: np.ndarray, mean: float, upper: float, path: Path) -> None:
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 55, 950, 475
    low = min(0.0, float(deltas.min()))
    high = max(0.01, float(deltas.max()), upper)
    scale_y = lambda y: bottom - (bottom - top) * (y - low) / max(high - low, 1e-9)
    draw.text((left, 15), "Paired reference-relative LPIPS increase", fill="black")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    bar_width = max(8, int((right - left) / max(len(deltas), 1) * 0.65))
    for index, value in enumerate(deltas):
        x = left + (right - left) * (index + 0.5) / len(deltas)
        draw.rectangle(
            (x - bar_width / 2, scale_y(max(value, 0)), x + bar_width / 2, scale_y(min(value, 0))),
            fill=(70, 130, 180),
        )
    for value, color, label in ((mean, "darkgreen", "mean"), (upper, "crimson", "bootstrap p95")):
        y = scale_y(value)
        draw.line((left, y, right, y), fill=color, width=3)
        draw.text((right - 180, y - 18), f"{label}: {value:.4f}", fill=color)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--budget", type=float, default=0.2)
    args = parser.parse_args()
    baseline = _load_means(args.baseline_metrics)
    candidate = _load_means(args.candidate_metrics)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline).symmetric_difference(candidate))
        raise ValueError(f"Metric pairs do not align: {missing}")
    keys = sorted(baseline)
    left = np.asarray([baseline[key] for key in keys])
    right = np.asarray([candidate[key] for key in keys])
    mean, upper = paired_bootstrap_upper(
        left, right, samples=args.bootstrap_samples, seed=args.seed
    )
    deltas = right - left
    result = {
        "pairs": len(keys),
        "paired_keys": keys,
        "baseline_lpips": left.tolist(),
        "candidate_lpips": right.tolist(),
        "paired_deltas": deltas.tolist(),
        "delta_macro_mean": mean,
        "paired_bootstrap_95_upper": upper,
        "budget": args.budget,
        "passes_budget": mean <= args.budget and upper <= args.budget,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _plot(deltas, mean, upper, args.output.with_suffix(".png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
