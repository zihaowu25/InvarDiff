#!/usr/bin/env python3
"""Regenerate existing Hunyuan candidates from Cache Books on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _candidate_command(candidate: Path, reference: Path, gpu: str) -> list[str]:
    summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
    policy = summary["policy"]
    if policy == "module-only":
        runner = REPO / "evaluation" / "run_hunyuan_module_candidate.py"
    elif policy == "step-layer":
        runner = REPO / "evaluation" / "run_hunyuan_step_layer_candidate.py"
    else:
        raise ValueError(f"Unsupported policy {policy!r} in {candidate}")

    command = [
        sys.executable,
        str(runner),
        "--candidate-id", candidate.name,
        "--thresholds", json.dumps(summary["thresholds"], separators=(",", ":")),
        "--output-root", str(candidate.parent),
        "--reference-root", str(reference),
        "--gpu", gpu,
        "--frames", str(summary["frames"]),
        "--steps", str(summary["steps"]),
        "--seed", str(summary["seed"]),
        "--prompt-set", summary.get("prompt_set", "screen"),
        "--reuse-cache-book",
    ]
    if policy == "module-only":
        command.extend([
            "--runtime-cache-device",
            summary.get("runtime_cache_device", "gpu"),
        ])
    else:
        command.extend(["--step-threshold", str(summary["step_threshold"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--skip-fresh",
        action="store_true",
        help="Skip summaries already marked fresh_process_reload=true.",
    )
    args = parser.parse_args()

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU index")
    candidates = sorted(
        candidate
        for root in args.root
        for summary in root.rglob("summary.json")
        if (candidate := summary.parent)
        if not args.skip_fresh
        or not json.loads(summary.read_text(encoding="utf-8")).get(
            "fresh_process_reload", False
        )
    )
    if not candidates:
        parser.error("No candidate summary.json files found")

    buckets = [candidates[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, assigned: list[Path]) -> None:
        for index, candidate in enumerate(assigned, start=1):
            print(f"[GPU {gpu}] {index}/{len(assigned)} reload {candidate}", flush=True)
            subprocess.run(
                _candidate_command(candidate, args.reference_root, gpu),
                cwd=REPO,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, assigned)
            for gpu, assigned in zip(gpus, buckets)
            if assigned
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
