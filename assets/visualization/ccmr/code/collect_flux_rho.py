#!/usr/bin/env python3
"""Collect condition-level FLUX rho without compact pair mirroring.

The regular FLUX collector is intentionally pairwise to control memory.  Its
compact pair tensor cannot support prompt-level rho dispersion, so this
collector runs one prompt at a time with a shared latent per seed and writes
only scalar rho rows.  It never enables Cache or decodes images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
FLUX_ROOT = REPO_ROOT / "FLUX"
sys.path.insert(0, str(FLUX_ROOT))

from common import (  # noqa: E402
    EPS,
    attempt_record,
    configure_determinism,
    environment_snapshot,
    json_hash,
    l1_distance,
    load_yaml,
    load_attempt_history,
    make_generator,
    read_rows,
    save_json_atomic,
    sha256_file,
    table_artifact,
    tensor_sha256,
    valid_rho_index,
    utc_now,
    write_rows,
)
from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive  # noqa: E402
from sample_flux import compute_l1_distance as online_l1_distance, compute_rate as online_compute_rate  # noqa: E402


MODULES = {
    "double": ("attn", "context_attn", "ff", "context_ff"),
    "single": ("attn", "mlp"),
}


def _load_prompts(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = HERE.parent / "conditions" / "flux_prompts.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    requested = set(config.get("prompt_ids", [item["id"] for item in values]))
    selected = [item for item in values if item["id"] in requested]
    if len(selected) != len(requested):
        missing = sorted(requested - {item["id"] for item in selected})
        raise ValueError(f"Unknown prompt IDs: {missing}")
    return selected


def _hook(storage: dict[str, dict[int, torch.Tensor]], family: str, layer: int, feature_device: str):
    def callback(_module, _inputs, output):
        block_outputs = output[2] if family == "double" else output[1]
        for name in MODULES[family]:
            value = block_outputs[name].detach()
            if value.ndim >= 3 and value.shape[0] == 1:
                value = value[0]
            value = value.contiguous()
            storage[f"{family}.{name}"][layer] = value if feature_device == "gpu" else value.cpu()

    return callback


def _run_prompt(pipe, dynamic_model, prompt: dict[str, Any], seed: int, config: dict[str, Any], latent: torch.Tensor, latent_ids: torch.Tensor, output_dir: Path) -> list[dict[str, Any]]:
    device = pipe._execution_device
    dtype = latent.dtype
    n_steps = int(config["num_inference_steps"])
    feature_device = str(config.get("feature_device", "cpu")).lower()
    if feature_device not in {"cpu", "gpu"}:
        raise ValueError("feature_device must be cpu or gpu")
    storage = {f"{family}.{name}": {} for family, names in MODULES.items() for name in names}
    hooks = []
    for layer, block in enumerate(dynamic_model.transformer_blocks):
        hooks.append(block.register_forward_hook(_hook(storage, "double", layer, feature_device)))
    for layer, block in enumerate(dynamic_model.single_transformer_blocks):
        hooks.append(block.register_forward_hook(_hook(storage, "single", layer, feature_device)))
    module_counts = {"double": len(dynamic_model.transformer_blocks), "single": len(dynamic_model.single_transformer_blocks)}
    previous = {
        key: [None] * module_counts[key.split(".", 1)[0]] for key in storage
    }
    previous_l1 = {
        key: [None] * module_counts[key.split(".", 1)[0]] for key in storage
    }
    previous_code_l1 = {
        key: [None] * module_counts[key.split(".", 1)[0]] for key in storage
    }
    previous_timestep: float | None = None
    rows: list[dict[str, Any]] = []
    original_prepare_latents = pipe.prepare_latents

    def prepared_latents(_batch_size, _channels, _height, _width, _dtype, _device, _generator, _latents=None):
        return latent, latent_ids

    pipe.prepare_latents = prepared_latents
    dynamic_model.reset()
    try:
        sampler = flux_sample_loop_progressive(
            pipe,
            prompt=prompt["prompt"],
            height=int(config["height"]),
            width=int(config["width"]),
            num_inference_steps=n_steps,
            guidance_scale=float(config.get("guidance_scale", 3.5)),
            true_cfg_scale=float(config.get("true_cfg_scale", 1.0)),
            max_sequence_length=int(config.get("max_sequence_length", 512)),
            generator=make_generator(seed, "cpu"),
            latents=None,
            output_type="latent",
        )
        with torch.inference_mode():
            for step_idx, state in enumerate(sampler):
                if "final_image" in state:
                    continue
                if any(len(values) == 0 for values in storage.values()):
                    raise RuntimeError(f"FLUX rho hooks did not capture all modules at step {step_idx}")
                for key, layer_values in storage.items():
                    for layer, current in layer_values.items():
                        current = current.detach().contiguous()
                        previous_feature = previous[key][layer]
                        if previous_feature is not None:
                            increment = l1_distance(previous_feature, current)
                            code_increment = online_l1_distance(previous_feature, current)
                            if previous_l1[key][layer] is not None and valid_rho_index(step_idx - 1, n_steps):
                                denominator = max(float(previous_l1[key][layer].item()), float(config.get("epsilon", EPS)))
                                rows.append({
                                    "run_id": output_dir.name,
                                    "model": "flux",
                                    "seed": seed,
                                    "condition_id": prompt["id"],
                                    "rho_scope": "condition",
                                    "module_family": key.split(".", 1)[0],
                                    "module_name": key.split(".", 1)[1],
                                    "layer_idx": layer,
                                    "score_step_idx": step_idx - 1,
                                    "scheduler_timestep": previous_timestep,
                                    "denoising_progress": (step_idx - 1) / max(n_steps - 1, 1),
                                    "l1_prev": float(previous_l1[key][layer].item()),
                                    "l1_next": float(increment.item()),
                                    "rho_clean": float(increment.item()) / denominator,
                                    "rho_code": float(online_compute_rate(code_increment, previous_code_l1[key][layer]).detach().float().cpu()),
                                    "valid": True,
                                })
                            previous_l1[key][layer] = increment.detach().float().cpu()
                            previous_code_l1[key][layer] = code_increment.detach().float()
                        previous[key][layer] = current
                for values in storage.values():
                    values.clear()
                previous_timestep = float(state["timestep"].detach().cpu())
    finally:
        pipe.prepare_latents = original_prepare_latents
        for hook in hooks:
            hook.remove()
        dynamic_model.reset()
    for key in storage:
        family, module_name = key.split(".", 1)
        for layer in range(module_counts[family]):
            for score_idx in (0, n_steps - 1):
                rows.append({
                    "run_id": output_dir.name, "model": "flux", "seed": seed,
                    "condition_id": prompt["id"], "rho_scope": "condition",
                    "module_family": family, "module_name": module_name, "layer_idx": layer,
                    "score_step_idx": score_idx, "scheduler_timestep": None,
                    "denoising_progress": score_idx / max(n_steps - 1, 1),
                    "l1_prev": None, "l1_next": None, "rho_clean": None, "rho_code": None,
                    "valid": False, "invalid_reason": "three_observation_boundary",
                })
    return rows


def _write_gzip_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.gz")
    write_rows(temporary, rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect condition-level FLUX rho")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.get("cache_enabled", False):
        raise ValueError("CCMR collector requires cache_enabled=false")

    from diffusers import FluxPipeline

    checkpoint = Path(config["model_path"])
    dtype = torch.bfloat16 if str(config.get("model_dtype", "bfloat16")).lower() in {"bfloat16", "bf16"} else torch.float16
    pipe = FluxPipeline.from_pretrained(str(checkpoint), torch_dtype=dtype, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    dynamic_model = DynamicFluxTransformer2DModel(pipe.transformer, int(config["num_inference_steps"]))
    dynamic_model.step_cache_bool = [False] * int(config["num_inference_steps"])
    dynamic_model.block_cache_enable = False
    dynamic_model.reset()
    pipe.transformer = dynamic_model
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prompts = _load_prompts(config)
    save_json_atomic(output_dir / "config.json", config)
    save_json_atomic(output_dir / "conditions.json", prompts)
    environment = environment_snapshot(REPO_ROOT, [checkpoint])
    if config.get("experiment_level") == "formal" and str(environment.get("git_status", "")).strip():
        raise RuntimeError("Formal collection requires a clean tracked Git worktree")
    save_json_atomic(output_dir / "environment.json", environment)
    resolved_hash = json_hash(config)
    manifest = {
        "run_id": output_dir.name, "model": "flux", "rho_scope": "condition",
        "cache_enabled": False, "decode_output": False, "resolved_config_hash": resolved_hash,
        "experiment_level": config.get("experiment_level", "smoke"),
        "paper_eligible": bool(config.get("paper_eligible", False)),
        "started_at": utc_now(), "shards": {},
    }
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    latent_hashes: dict[str, str] = {}
    for seed_value in config.get("seeds", [0]):
        seed = int(seed_value)
        configure_determinism(seed)
        latent_generator = make_generator(seed, "cpu")
        one, one_ids = pipe.prepare_latents(
            1,
            dynamic_model.config.in_channels // 4,
            int(config["height"]),
            int(config["width"]),
            dtype,
            device,
            latent_generator,
            None,
        )
        latent_hashes[str(seed)] = tensor_sha256(one)
        for prompt in prompts:
            shard_dir = output_dir / "shards" / f"seed_{seed}_prompt_{prompt['id']}"
            shard_path = shard_dir / "rho_per_condition.csv.gz"
            marker = shard_dir / "manifest.json"
            shard_key = f"{seed}:{prompt['id']}"
            retry_count, attempt_history = load_attempt_history(marker, resolved_hash)
            if args.resume and marker.exists() and shard_path.exists():
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                artifact = marker_value.get("table", {})
                if (marker_value.get("status") == "complete"
                        and marker_value.get("resolved_config_hash") == resolved_hash
                        and sha256_file(shard_path) == artifact.get("sha256")):
                    rows = read_rows(shard_path)
                    if len(rows) == int(artifact.get("row_count", -1)):
                        if (config.get("experiment_level") == "formal"
                                and marker_value.get("commit") != environment.get("git_commit")):
                            raise RuntimeError(
                                f"Formal FLUX rho shard {shard_key} was collected at a different commit"
                            )
                        all_rows.extend(rows)
                        manifest["shards"][shard_key] = marker_value
                        continue
            shard_started_at = utc_now()
            shard_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                rows = _run_prompt(pipe, dynamic_model, prompt, seed, config, one, one_ids, output_dir)
            except Exception as exc:
                failed_at = utc_now()
                wall_time_s = time.perf_counter() - shard_started
                peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
                peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0
                attempt_history.append(attempt_record(
                    status="failed", started_at=shard_started_at, finished_at=failed_at,
                    wall_time_s=wall_time_s, oom=isinstance(exc, torch.cuda.OutOfMemoryError),
                    peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
                    commit=environment.get("git_commit"), resolved_config_hash=resolved_hash,
                    error_type=type(exc).__name__, error=str(exc),
                ))
                save_json_atomic(marker, {
                    "status": "failed", "seed": seed, "condition_id": prompt["id"],
                    "resolved_config_hash": resolved_hash,
                    "experiment_level": config.get("experiment_level", "smoke"),
                    "started_at": shard_started_at, "failed_at": failed_at,
                    "wall_time_s": wall_time_s,
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
                    "branch": environment.get("git_branch"), "commit": environment.get("git_commit"),
                    "dirty_status": environment.get("git_status"),
                    "retry_count": retry_count, "attempt_history": attempt_history,
                })
                raise
            shard_dir.mkdir(parents=True, exist_ok=True)
            _write_gzip_rows_atomic(shard_path, rows)
            completed_at = utc_now()
            wall_time_s = time.perf_counter() - shard_started
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0
            attempt_history.append(attempt_record(
                status="complete", started_at=shard_started_at, finished_at=completed_at,
                wall_time_s=wall_time_s, oom=False,
                peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
                commit=environment.get("git_commit"), resolved_config_hash=resolved_hash,
            ))
            marker_value = {
                "status": "complete", "seed": seed, "condition_id": prompt["id"],
                "resolved_config_hash": resolved_hash, "latent_hash": latent_hashes[str(seed)],
                "branch": environment.get("git_branch"), "commit": environment.get("git_commit"),
                "dirty_status": environment.get("git_status"), "exact_command": [sys.executable, *sys.argv],
                "checkpoint": environment.get("checkpoints", [{}])[0], "environment_file": "../../environment.json",
                "condition_bank_hash": json_hash(prompts), "model_dtype": config.get("model_dtype"),
                "statistics_dtype": config.get("statistics_dtype"), "cache_enabled": False, "decode_output": False,
                "hook_locations": ["FluxTransformerBlock.forward:block_outputs", "FluxSingleTransformerBlock.forward:block_outputs"],
                "hook_count": len(dynamic_model.transformer_blocks) + len(dynamic_model.single_transformer_blocks),
                "experiment_level": config.get("experiment_level", "smoke"), "paper_eligible": bool(config.get("paper_eligible", False)),
                "offload": False, "resume_requested": bool(args.resume), "retry_count": retry_count,
                "attempt_history": attempt_history, "oom": False,
                "started_at": shard_started_at, "completed_at": completed_at, "wall_time_s": wall_time_s,
                "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
                "table": table_artifact(shard_path, len(rows)),
            }
            save_json_atomic(marker, marker_value)
            manifest["shards"][shard_key] = marker_value
            all_rows.extend(rows)
    _write_gzip_rows_atomic(output_dir / "tables" / "rho_per_condition.csv.gz", all_rows)
    save_json_atomic(output_dir / "latent_hashes.json", latent_hashes)
    summary = {
        "model": "flux", "run_id": output_dir.name, "rho_scope": "condition",
        "num_seeds": len(config.get("seeds", [0])), "num_conditions": len(prompts),
        "elapsed_s": time.perf_counter() - started, "row_counts": {"rho_per_condition": len(all_rows)},
    }
    save_json_atomic(output_dir / "summary.json", summary)
    manifest["completed_at"] = utc_now()
    save_json_atomic(output_dir / "run_manifest.json", manifest)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        save_json_atomic(output_dir / "memory.json", {
            "peak_allocated_gib": peak,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        })
        print(f"Peak allocated GPU memory: {peak:.3f} GiB")
    print(f"FLUX condition rho run complete: {output_dir} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
