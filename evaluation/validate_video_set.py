#!/usr/bin/env python3
"""Validate generated videos with ffprobe before metric aggregation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def _probe(path: Path) -> dict:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ], text=True)
    streams = json.loads(output).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream")
    return streams[0]


def _rate(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    result = float(numerator) / float(denominator)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite frame rate: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob(args.glob))
    if len(paths) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} videos, found {len(paths)} in {args.root}"
        )
    rows = []
    for path in paths:
        stream = _probe(path)
        row = {
            "path": str(path),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frames": int(stream["nb_read_frames"]),
            "fps": _rate(stream["avg_frame_rate"]),
            "duration": float(stream["duration"]),
        }
        expected = (args.width, args.height, args.frames)
        actual = (row["width"], row["height"], row["frames"])
        if actual != expected or abs(row["fps"] - args.fps) > 1e-6:
            raise ValueError(
                f"{path}: expected {expected} at {args.fps} fps, "
                f"got {actual} at {row['fps']} fps"
            )
        if not math.isfinite(row["duration"]) or row["duration"] <= 0:
            raise ValueError(f"{path}: invalid duration {row['duration']}")
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"valid": True, "videos": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"validated {len(rows)} videos")


if __name__ == "__main__":
    main()
