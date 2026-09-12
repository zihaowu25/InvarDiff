#!/usr/bin/env python3
"""Aggregate Hunyuan pairwise CCMR shards into module-family rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def _bar_chart(summary: pd.DataFrame, path: Path):
    width, height = 1100, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 270, 45, 1040, 455
    values = summary.sort_values("median_db").reset_index(drop=True)
    maximum = max(1.0, float(values["median_db"].max()))
    row_h = (bottom - top) / max(1, len(values))
    draw.text((left, 12), "HunyuanVideo-1.5 median CCMR gain (dB)", fill="black")
    for index, row in values.iterrows():
        y0 = top + index * row_h + 5
        y1 = top + (index + 1) * row_h - 5
        x1 = left + (right - left) * float(row["median_db"]) / maximum
        draw.text((10, y0), str(row["module_family"]), fill="black")
        draw.rectangle((left, y0, x1, y1), fill=(45, 118, 182))
        draw.text((x1 + 6, y0), f'{float(row["median_db"]):.2f}', fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("pair*.csv"))
    if not paths:
        raise SystemExit("No pairwise CCMR CSV shards found")
    frames = []
    for pair_index, path in enumerate(paths):
        frame = pd.read_csv(path)
        frame["pair_index"] = pair_index
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary = (
        rows.groupby("module_family", sort=False)["g_ccmr_db"]
        .agg(count="size", median_db="median", mean_db="mean")
        .reset_index()
        .sort_values("median_db", ascending=False)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "hunyuan_ccmr_points.csv", index=False)
    summary.to_csv(args.output_dir / "hunyuan_ccmr_ranking.csv", index=False)
    payload = {
        "protocol": "hunyuan_pairwise_ccmr_v1",
        "num_pairs": len(paths),
        "ranking": summary.to_dict(orient="records"),
    }
    (args.output_dir / "hunyuan_ccmr_ranking.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _bar_chart(summary, args.output_dir / "hunyuan_ccmr_ranking.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
