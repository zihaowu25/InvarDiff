#!/usr/bin/env python3
"""Run one Hunyuan native step+layer candidate and aggregate paired LPIPS."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from run_hunyuan_module_candidate import (
    HUNYUAN_PYTHON,
    METRIC_PYTHON,
    MODEL,
    MODULE_FLAGS,
    OFFICIAL,
    REPO,
    _threshold_args,
)


def _run(command: list[str], log: Path, env: dict[str, str]) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - started


def _hit_rates(path: Path) -> dict[str, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    step_book = payload["step_cache_book"]
    module_books = payload["module_cache_book"]
    result: dict[str, float | None] = {
        "step": sum(step_book) / len(step_book) if step_book else None
    }
    for module, rows in module_books.items():
        flat = [bool(value) for row in rows for value in row]
        result[module] = sum(flat) / len(flat) if flat else None
    joint = [
        bool(img_value and txt_value)
        for img_row, txt_row in zip(
            module_books["double.img_attn"], module_books["double.txt_attn"]
        )
        for img_value, txt_value in zip(img_row, txt_row)
    ]
    result["double.joint_attn"] = sum(joint) / len(joint) if joint else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--step-threshold", type=float, required=True)
    parser.add_argument("--thresholds", required=True, help="JSON module map")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=3,
        help=(
            "Number of prompts used by this candidate. Low-cost screening "
            "defaults to three; confirmation callers must opt into more."
        ),
    )
    parser.add_argument(
        "--prompt-set",
        choices=("screen", "confirmation"),
        default="screen",
    )
    parser.add_argument("--reuse-cache-book", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Reuse existing videos, Cache Book, and generation log; only compute metrics.",
    )
    args = parser.parse_args()
    if not 0 <= args.step_threshold <= 1:
        parser.error("--step-threshold must be in [0, 1]")
    thresholds = {key: float(value) for key, value in json.loads(args.thresholds).items()}
    unknown = set(thresholds) - set(MODULE_FLAGS)
    if unknown:
        parser.error(f"Unknown module thresholds: {sorted(unknown)}")

    conditions = json.loads((REPO / "evaluation" / "tuning_conditions.json").read_text())
    prompts = conditions[f"{args.prompt_set}_prompts"]
    if args.prompt_limit is not None:
        if args.prompt_limit <= 0:
            parser.error("--prompt-limit must be positive")
        prompts = prompts[:args.prompt_limit]
    calibration_prompt = conditions["ccmr_prompts"][0]
    root = args.output_root / args.candidate_id
    books, videos, metrics = root / "cache_books", root / "videos", root / "metrics"
    paired_references = root / "paired_references"
    for path in (books, videos, metrics, paired_references):
        path.mkdir(parents=True, exist_ok=True)
    book_name = f"{args.candidate_id}.json"
    book_file = books / book_name
    if (args.reuse_cache_book or args.metrics_only) and not book_file.is_file():
        parser.error(f"Existing Cache Book not found: {book_file}")

    prompt_file = root / f"{args.prompt_set}_prompts.txt"
    prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(OFFICIAL)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        master_port = port_socket.getsockname()[1]
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    command = [
        str(HUNYUAN_PYTHON), "HunyuanVideo/sample_hunyuan_step_layer.py",
        "--resolution", "720p", "--model_path", str(MODEL),
        "--video_length", str(args.frames),
        "--num_inference_steps", str(args.steps),
        "--seed", str(args.seed), "--sr", "false", "--rewrite", "false",
        "--deterministic_attention", "true",
        "--offloading", "false", "--runtime_cache_device", "gpu",
        "--cache_book_path", str(books), "--cache_book_file", book_name,
        "--step_thres", f"{args.step_threshold:.2f}",
        "--prompt", calibration_prompt,
    ] + _threshold_args(thresholds)
    generation_command = command + ["--use_invardiff"] + [
        "--generation_prompt_file", str(prompt_file),
        "--output_dir", str(videos),
        "--paired_reference_dir", str(paired_references),
    ]
    generation_log_path = root / "generation_batch.log"
    calibration_log_path = root / "calibration.log"
    if args.metrics_only:
        if not generation_log_path.is_file():
            parser.error(f"Existing generation log not found: {generation_log_path}")
        missing = [
            str(videos / f"p{index}.mp4")
            for index in range(len(prompts))
            if not (videos / f"p{index}.mp4").is_file()
        ]
        missing.extend(
            str(paired_references / f"p{index}.mp4")
            for index in range(len(prompts))
            if not (paired_references / f"p{index}.mp4").is_file()
        )
        if missing:
            parser.error(f"Existing candidate videos not found: {missing}")
        calibration_wall = None
        generation_process_wall = None
    else:
        calibration_wall = None
        if not args.reuse_cache_book:
            # Keep the measured generation isolated from calibration hooks and
            # provisional decisions by loading the book in a new Python process.
            calibration_wall = _run(
                command + ["--invardiff_calibration"],
                calibration_log_path,
                env,
            )
        generation_process_wall = _run(
            generation_command,
            generation_log_path,
            env,
        )
    log = generation_log_path.read_text(encoding="utf-8")
    elapsed = [float(x) for x in re.findall(r"Generation elapsed: ([0-9.]+)s", log)]
    peaks = [float(x) for x in re.findall(r"Generation peak allocated: ([0-9.]+) GiB", log)]
    if args.metrics_only:
        # Permit p0-only reuse of an older multi-prompt screening run.
        elapsed = elapsed[:len(prompts)]
        peaks = peaks[:len(prompts)]
    if len(elapsed) != len(prompts) or len(peaks) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} timings/peaks, got {len(elapsed)}/{len(peaks)}"
        )

    means, p95s = [], []
    for index in range(len(prompts)):
        metric_path = metrics / f"p{index}.json"
        _run([
            str(METRIC_PYTHON), "evaluation/paired_lpips.py",
            "--reference", str(paired_references / f"p{index}.mp4"),
            "--candidate", str(videos / f"p{index}.mp4"),
            "--output", str(metric_path), "--device", "cuda", "--batch-size", "8",
        ], root / f"metric_p{index}.log", env)
        item = json.loads(metric_path.read_text())
        means.append(item["frame_lpips"]["mean"])
        p95s.append(item["frame_lpips"]["p95"])

    result = {
        "candidate_id": args.candidate_id,
        "model": "HunyuanVideo-1.5",
        "policy": "step-layer",
        "step_threshold": args.step_threshold,
        "thresholds": thresholds,
        "frames": args.frames,
        "steps": args.steps,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "prompt_set": args.prompt_set,
        "deterministic_attention": True,
        "paired_reference_root": str(paired_references),
        "reused_cache_book": args.reuse_cache_book,
        "metrics_only_resume": args.metrics_only,
        "lpips_macro_mean": sum(means) / len(means),
        "lpips_prompt_means": means,
        "lpips_prompt_p95": p95s,
        "cache_hit_rates": _hit_rates(book_file),
        "fresh_process_reload": True,
        "same_process_paired_reference": True,
        "calibration_wall_s": calibration_wall,
        "generation_process_wall_s": generation_process_wall,
        "combined_operation_wall_s": (
            None
            if generation_process_wall is None
            else generation_process_wall + (calibration_wall or 0.0)
        ),
        "generation_wall_s_mean": sum(elapsed) / len(elapsed),
        "generation_wall_s": elapsed,
        "generation_peak_allocated_gib_max": max(peaks),
        "generation_peak_allocated_gib": peaks,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
