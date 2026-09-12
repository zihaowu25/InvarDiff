#!/usr/bin/env python3
"""Run one Hunyuan layer-only threshold candidate and aggregate paired LPIPS."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
HUNYUAN_PYTHON = WORKSPACE / ".venvs" / "hunyuan15" / "bin" / "python"
METRIC_PYTHON = WORKSPACE / "miniforge3" / "bin" / "python"
MODEL = WORKSPACE / "models" / "HunyuanVideo-1.5"
OFFICIAL = WORKSPACE / "open-source" / "HunyuanVideo-1.5"
MODULE_FLAGS = {
    "double.img_attn": "--double_img_attn_thres",
    "double.txt_attn": "--double_txt_attn_thres",
    "double.img_mlp": "--double_img_mlp_thres",
    "double.txt_mlp": "--double_txt_mlp_thres",
    "single.attn": "--single_attn_thres",
    "single.mlp": "--single_mlp_thres",
}


def _run(command: list[str], log: Path, env: dict[str, str]) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - start


def _threshold_args(values: dict[str, float]) -> list[str]:
    result = []
    for name, flag in MODULE_FLAGS.items():
        result.extend([flag, f"{values.get(name, 0.0):.2f}"])
    return result


def _cache_hit_rates(path: Path) -> dict[str, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    module_books = payload["module_cache_book"]
    rates = {}
    for module, step_masks in module_books.items():
        flat = [bool(value) for step in step_masks for value in step]
        rates[module] = sum(flat) / len(flat) if flat else None
    joint = [
        bool(img_value and txt_value)
        for img_step, txt_step in zip(
            module_books["double.img_attn"], module_books["double.txt_attn"]
        )
        for img_value, txt_value in zip(img_step, txt_step)
    ]
    rates["double.joint_attn"] = sum(joint) / len(joint) if joint else None
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
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
        "--runtime-cache-device",
        choices=("auto", "gpu", "cpu"),
        default="gpu",
        help="Runtime tensor placement; use auto for native-frame confirmation.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("screen", "confirmation"),
        default="screen",
    )
    parser.add_argument(
        "--reuse-cache-book",
        action="store_true",
        help="Skip calibration and generate with an existing candidate Cache Book.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Reuse existing videos, Cache Book, and generation log; only compute metrics.",
    )
    parser.add_argument(
        "--resume-generation",
        action="store_true",
        help="Reuse a Cache Book and completed leading pN pairs, generating only missing pairs.",
    )
    args = parser.parse_args()
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
    books = root / "cache_books"
    videos = root / "videos"
    metrics = root / "metrics"
    paired_references = root / "paired_references"
    for path in (books, videos, metrics, paired_references):
        path.mkdir(parents=True, exist_ok=True)
    book_name = f"{args.candidate_id}.json"
    book_file = books / book_name
    if (args.reuse_cache_book or args.metrics_only) and not book_file.is_file():
        parser.error(f"Existing Cache Book not found: {book_file}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(OFFICIAL)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        master_port = port_socket.getsockname()[1]
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    common = [
        str(HUNYUAN_PYTHON), "HunyuanVideo/sample_hunyuan.py",
        "--resolution", "720p", "--model_path", str(MODEL),
        "--video_length", str(args.frames),
        "--num_inference_steps", str(args.steps),
        "--seed", str(args.seed), "--sr", "false", "--rewrite", "false",
        "--deterministic_attention", "true",
        "--offloading", "false",
        "--runtime_cache_device", args.runtime_cache_device,
        "--cache_book_path", str(books), "--cache_book_file", book_name,
    ] + _threshold_args(thresholds)
    prompt_file = root / f"{args.prompt_set}_prompts.txt"
    prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    generation_start = 0
    if args.resume_generation:
        if not book_file.is_file():
            parser.error(f"Existing Cache Book not found: {book_file}")
        while (
            generation_start < len(prompts)
            and (videos / f"p{generation_start}.mp4").is_file()
            and (paired_references / f"p{generation_start}.mp4").is_file()
        ):
            generation_start += 1
        pending_prompt_file = root / f"{args.prompt_set}_prompts_resume.txt"
        pending_prompt_file.write_text(
            "\n".join(prompts[generation_start:]) + "\n",
            encoding="utf-8",
        )
    else:
        pending_prompt_file = prompt_file
    generation_args = [
        "--prompt", calibration_prompt,
        "--use_invardiff",
        "--generation_prompt_file", str(pending_prompt_file),
        "--generation_index_offset", str(generation_start),
        "--output_dir", str(videos),
        "--paired_reference_dir", str(paired_references),
    ]
    generation_log_path = root / "generation_batch.log"
    active_generation_log_path = (
        root / "generation_resume.log"
        if args.resume_generation
        else generation_log_path
    )
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
            # Calibration and generation must not share a Python process. Hooks and
            # provisional runtime state are intentionally discarded before the
            # freshly saved Cache Book is loaded for the measured generation.
            calibration_wall = _run(
                common + ["--prompt", calibration_prompt, "--invardiff_calibration"],
                calibration_log_path,
                env,
            )
        generation_process_wall = None
        if generation_start < len(prompts):
            generation_process_wall = _run(
                common + generation_args,
                active_generation_log_path,
                env,
            )
    generation_log = generation_log_path.read_text(encoding="utf-8")
    if args.resume_generation and active_generation_log_path.is_file():
        old_elapsed = re.findall(r"Generation elapsed: ([0-9.]+)s", generation_log)
        old_peaks = re.findall(
            r"Generation peak allocated: ([0-9.]+) GiB", generation_log
        )
        resumed_log = active_generation_log_path.read_text(encoding="utf-8")
        generation_log = "\n".join(
            [
                *(f"Generation elapsed: {value}s" for value in old_elapsed[:generation_start]),
                *(f"Generation peak allocated: {value} GiB" for value in old_peaks[:generation_start]),
                resumed_log,
            ]
        )
    generation_elapsed = [
        float(value)
        for value in re.findall(r"Generation elapsed: ([0-9.]+)s", generation_log)
    ]
    generation_peak_gib = [
        float(value)
        for value in re.findall(
            r"Generation peak allocated: ([0-9.]+) GiB", generation_log
        )
    ]
    if args.metrics_only:
        # A low-cost audit may intentionally reuse only p0 from an older
        # multi-prompt run. Timings are ordered exactly like p0, p1, ... .
        generation_elapsed = generation_elapsed[:len(prompts)]
        generation_peak_gib = generation_peak_gib[:len(prompts)]
    if len(generation_elapsed) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} generation timings, got {len(generation_elapsed)}"
        )
    if len(generation_peak_gib) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} peak-memory records, "
            f"got {len(generation_peak_gib)}"
        )
    lpips_means = []
    lpips_p95 = []
    for index, prompt in enumerate(prompts):
        output = videos / f"p{index}.mp4"
        metric_path = metrics / f"p{index}.json"
        _run(
            [
                str(METRIC_PYTHON), "evaluation/paired_lpips.py",
                "--reference", str(paired_references / f"p{index}.mp4"),
                "--candidate", str(output), "--output", str(metric_path),
                "--device", "cuda", "--batch-size", "8",
            ],
            root / f"metric_p{index}.log", env,
        )
        item = json.loads(metric_path.read_text())
        lpips_means.append(item["frame_lpips"]["mean"])
        lpips_p95.append(item["frame_lpips"]["p95"])
    result = {
        "candidate_id": args.candidate_id,
        "model": "HunyuanVideo-1.5",
        "policy": "module-only",
        "thresholds": thresholds,
        "frames": args.frames,
        "steps": args.steps,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "prompt_set": args.prompt_set,
        "runtime_cache_device": args.runtime_cache_device,
        "deterministic_attention": True,
        "paired_reference_root": str(paired_references),
        "reused_cache_book": args.reuse_cache_book,
        "metrics_only_resume": args.metrics_only,
        "generation_resume": args.resume_generation,
        "lpips_macro_mean": sum(lpips_means) / len(lpips_means),
        "lpips_prompt_means": lpips_means,
        "lpips_prompt_p95": lpips_p95,
        "module_hit_rates": _cache_hit_rates(book_file),
        "fresh_process_reload": True,
        "same_process_paired_reference": True,
        "calibration_wall_s": calibration_wall,
        "generation_process_wall_s": generation_process_wall,
        "combined_operation_wall_s": (
            None
            if generation_process_wall is None
            else generation_process_wall + (calibration_wall or 0.0)
        ),
        "generation_wall_s_mean": sum(generation_elapsed) / len(generation_elapsed),
        "generation_wall_s": generation_elapsed,
        "generation_peak_allocated_gib_max": max(generation_peak_gib),
        "generation_peak_allocated_gib": generation_peak_gib,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
