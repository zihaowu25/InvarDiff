#!/usr/bin/env python3
"""Collect CCMR statistics from full-compute DiT trajectories.

Only read-only forward hooks are added to the existing DiT implementation.
No Cache module is imported or enabled by this collector.
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
DIT_ROOT = REPO_ROOT / "DiT"
sys.path.insert(0, str(DIT_ROOT))

from common import (  # noqa: E402
    EPS,
    DEGENERATE_THRESHOLD,
    centered_variance,
    clean_rho,
    ccmr_ratio,
    configure_determinism,
    environment_snapshot,
    gain_db,
    json_hash,
    deterministic_derangements,
    load_yaml,
    make_generator,
    read_rows,
    rms_feature,
    save_json_atomic,
    sha256_file,
    tensor_sha256,
    table_artifact,
    temporal_metrics,
    valid_rho_index,
    utc_now,
    write_rows,
    write_rows_atomic,
)
from models.dynamic_cache import SimilarityAnalyzer  # noqa: E402


MODULES = ("msa", "mlp")


def _feature_hook(storage: dict[str, dict[int, torch.Tensor]], layer: int):
    def hook(_module, _inputs, output):
        _x, block_output = output
        # DiT forward_with_cfg places conditional samples in the first half.
        # The collector trims the unconditional half in _run_trajectory.
        storage["msa"][layer] = block_output[0].detach()
        storage["mlp"][layer] = block_output[1].detach()

    return hook


def _make_model(config: dict[str, Any], device: torch.device):
    from diffusion import create_diffusion
    from download import find_model
    from models.dynamic_cache import DiT_models

    image_size = int(config["image_size"])
    input_size = image_size // 8
    model = DiT_models[config.get("model", "DiT-XL/2")](
        input_size=input_size,
        num_classes=int(config.get("num_classes", 1000)),
    ).to(device=device, dtype=torch.float32)
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    expected_hash = config.get("checkpoint_sha256")
    if expected_hash and sha256_file(checkpoint) != expected_hash:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
    state = find_model(str(checkpoint))
    model.load_state_dict(state)
    model.eval()
    diffusion = create_diffusion(str(int(config["num_inference_steps"])))
    return model, diffusion, checkpoint


def _condition_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = HERE.parent / "conditions" / "dit_classes.json"
    if not path.exists():
        return [{"id": int(x), "name": f"ImageNet class {x}"} for x in config["class_ids"]]
    values = json.loads(path.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item for item in values}
    return [by_id.get(int(x), {"id": int(x), "name": f"ImageNet class {x}"}) for x in config["class_ids"]]


def _per_sample_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a.detach().float() - b.detach().float()
    return diff.reshape(diff.shape[0], -1).abs().sum(dim=1)


def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.detach().float().reshape(a.shape[0], -1)
    bf = b.detach().float().reshape(b.shape[0], -1)
    return torch.nn.functional.cosine_similarity(af, bf, dim=1)


def _shuffled_difference_variances(
    current: torch.Tensor,
    previous: torch.Tensor,
    permutations: list[list[int]],
) -> list[float]:
    """Exact condition variance for many derangements without 100 tensor copies."""
    cur = current.detach().float().reshape(current.shape[0], -1)
    prev = previous.detach().float().reshape(previous.shape[0], -1)
    cur = cur - cur.mean(dim=0, keepdim=True)
    prev = prev - prev.mean(dim=0, keepdim=True)
    feature_count = float(cur.shape[1])
    constant = cur.square().sum() + prev.square().sum()
    cross = cur @ prev.transpose(0, 1)
    indices = torch.tensor(permutations, device=cross.device, dtype=torch.long)
    rows = torch.arange(cur.shape[0], device=cross.device).expand(indices.shape[0], -1)
    values = (constant - 2.0 * cross[rows, indices].sum(dim=1)) / (float(cur.shape[0]) * feature_count)
    return [float(value) for value in values.cpu()]


def _run_trajectory(model, diffusion, class_ids: list[int], seed: int, config: dict[str, Any], output_dir: Path):
    device = next(model.parameters()).device
    k = len(class_ids)
    latent_generator = make_generator(seed, "cpu")
    z_seed = torch.randn(1, 4, int(config["image_size"]) // 8, int(config["image_size"]) // 8, generator=latent_generator, device="cpu")
    z_cond = z_seed.expand(k, -1, -1, -1).clone().to(device)
    z = torch.cat([z_cond, z_cond], dim=0)
    labels = torch.tensor(class_ids + [int(config.get("num_classes", 1000))] * k, device=device, dtype=torch.long)
    # The same initial latent is intentionally copied to every condition.  A
    # hash must therefore be repeated, rather than hashing empty slices for
    # condition indices greater than zero.
    latent_hash = tensor_sha256(z_seed)
    latent_hashes = {str(seed): [latent_hash for _ in range(k)]}
    n_steps = int(config["num_inference_steps"])
    n_layers = len(model.blocks)
    storage = {name: {} for name in MODULES}
    hooks = [block.register_forward_hook(_feature_hook(storage, idx)) for idx, block in enumerate(model.blocks)]
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition", "alignment_control")}
    previous: dict[str, list[torch.Tensor | None]] = {name: [None] * n_layers for name in MODULES}
    previous_var: dict[str, list[float | None]] = {name: [None] * n_layers for name in MODULES}
    previous_l1: dict[str, list[torch.Tensor | None]] = {name: [None] * n_layers for name in MODULES}
    previous_code_l1: dict[str, list[torch.Tensor | None]] = {name: [None] * n_layers for name in MODULES}
    history: dict[str, list[list[torch.Tensor]]] = {name: [[] for _ in range(n_layers)] for name in MODULES}
    max_gap = max([int(x) for x in config.get("time_gaps", [1])], default=1)
    permutations = deterministic_derangements(
        k,
        int(config.get("alignment_shuffle_trials", 100)),
        int(config.get("shuffle_seed", 2027)) + seed,
    )
    permutation_hashes = [tensor_sha256(torch.tensor(value, dtype=torch.int64)) for value in permutations]

    observed_timestep: dict[str, float] = {"value": float("nan")}
    previous_scheduler_timestep: float | None = None
    original_forward = model.forward_with_cfg

    def observed_forward(x, t, y, cfg_scale):
        observed_timestep["value"] = float(t.detach().reshape(-1)[0].cpu())
        return original_forward(x, t, y, cfg_scale)

    sampler = diffusion.ddim_sample_loop_progressive(
        observed_forward,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs={"y": labels, "cfg_scale": float(config.get("cfg_scale", 4.0))},
        progress=False,
        device=device,
    )
    with torch.inference_mode():
        for step_idx, _sample in enumerate(sampler):
            if not storage["msa"]:
                raise RuntimeError(f"DiT hooks did not capture features at step {step_idx}")
            scheduler_timestep = observed_timestep["value"]
            denoising_progress = step_idx / max(n_steps - 1, 1)
            for module in MODULES:
                for layer in range(n_layers):
                    # Clone only the conditional half; this prevents the hook
                    # dictionary from retaining the unconditional branch.
                    current = storage[module][layer][:k].detach().contiguous()
                    raw = float(centered_variance(current).cpu())
                    prev = previous[module][layer]
                    prev_var = previous_var[module][layer]
                    valid = prev is not None
                    diff_var = float("nan")
                    v_base = float("nan")
                    gain = float("nan")
                    sim = float("nan")
                    if valid:
                        delta = current.float() - prev.float()
                        diff_var = float(centered_variance(delta).cpu())
                        v_base = 0.5 * (raw + float(prev_var))
                        gain = gain_db(v_base, diff_var, float(config.get("epsilon", EPS)))
                        centered_now = current.float() - current.float().mean(dim=0, keepdim=True)
                        centered_prev = prev.float() - prev.float().mean(dim=0, keepdim=True)
                        sim = float(_cosine_rows(centered_now, centered_prev).mean().cpu())
                        a_rms, d_rms, r_time = temporal_metrics(prev, current, float(config.get("epsilon", EPS)))
                        for condition_idx, (a_val, d_val) in enumerate(zip(
                            (float(torch.sqrt(0.5 * ((prev[condition_idx].float() ** 2).mean() + (current[condition_idx].float() ** 2).mean())).cpu()) for condition_idx in range(k)),
                            (float(torch.sqrt(((current[condition_idx].float() - prev[condition_idx].float()) ** 2).mean()).cpu()) for condition_idx in range(k)),
                        )):
                            rows["temporal_metrics"].append({
                                "run_id": output_dir.name, "model": "dit", "seed": seed,
                                "condition_id": class_ids[condition_idx], "module_family": "dit",
                                "module_name": module, "layer_idx": layer, "step_idx": step_idx,
                                "scheduler_timestep": scheduler_timestep, "denoising_progress": denoising_progress, "output_rms": a_val,
                                "diff_rms": d_val, "r_time": d_val / (a_val + float(config.get("epsilon", EPS))), "valid": True,
                            })
                        shuffled_variances = _shuffled_difference_variances(current, prev, permutations)
                        shuffled_gains = [gain_db(v_base, value, float(config.get("epsilon", EPS))) for value in shuffled_variances]
                        rows["condition_similarity"].append({
                            "run_id": output_dir.name, "model": "dit", "seed": seed,
                            "estimator": "exact_batch", "pair_id_or_condition_group": "all",
                            "module_family": "dit", "module_name": module, "layer_idx": layer,
                            "step_idx": step_idx, "scheduler_timestep": scheduler_timestep, "denoising_progress": denoising_progress,
                            "adjacent_condition_cosine": sim, "aligned_g_ccmr_db": gain,
                            "shuffled_g_ccmr_db": float(torch.tensor(shuffled_gains).mean()), "valid": True,
                        })
                        for shuffle_trial, (permutation, permutation_hash, shuffled_gain) in enumerate(
                            zip(permutations, permutation_hashes, shuffled_gains)
                        ):
                            rows["alignment_control"].append({
                                "run_id": output_dir.name, "model": "dit", "seed": seed,
                                "shuffle_trial": shuffle_trial,
                                "permutation_seed": int(config.get("shuffle_seed", 2027)) + seed,
                                "permutation": json.dumps(permutation, separators=(",", ":")),
                                "permutation_hash": permutation_hash,
                                "module_family": "dit", "module_name": module, "layer_idx": layer,
                                "step_idx": step_idx, "scheduler_timestep": scheduler_timestep,
                                "denoising_progress": denoising_progress,
                                "g_aligned_db": gain, "g_shuffled_db": shuffled_gain,
                                "delta_g_db": gain - shuffled_gain, "valid": True,
                            })
                        for i in range(k):
                            for j in range(i + 1, k):
                                raw_pair = float((((current[i].float() - current[j].float()) ** 2).mean() / 2.0).cpu())
                                prev_pair = float((((prev[i].float() - prev[j].float()) ** 2).mean() / 2.0).cpu())
                                diff_pair = float((((delta[i].float() - delta[j].float()) ** 2).mean() / 2.0).cpu())
                                rows["condition_distance"].append({
                                    "run_id": output_dir.name, "model": "dit", "seed": seed,
                                    "condition_i": class_ids[i], "condition_j": class_ids[j],
                                    "module_family": "dit", "module_name": module, "layer_idx": layer,
                                    "step_idx": step_idx, "scheduler_timestep": scheduler_timestep,
                                    "denoising_progress": denoising_progress, "raw_pair_distance": raw_pair,
                                    "diff_pair_distance": diff_pair, "valid": True,
                                })
                        for gap in config.get("time_gaps", [1]):
                            gap = int(gap)
                            if gap <= len(history[module][layer]):
                                gap_feature = history[module][layer][-gap]
                                gap_delta = current.float() - gap_feature.float()
                                gap_diff = float(centered_variance(gap_delta).cpu())
                                gap_prev = float(centered_variance(gap_feature).cpu())
                                rows["time_gap_metrics"].append({
                                    "run_id": output_dir.name, "model": "dit", "seed": seed,
                                    "module_family": "dit", "module_name": module, "layer_idx": layer,
                                    "step_idx": step_idx, "scheduler_timestep": scheduler_timestep,
                                    "denoising_progress": denoising_progress, "time_gap": gap, "v_raw_base": 0.5 * (raw + gap_prev),
                                    "v_diff_gap": gap_diff, "r_ccmr_gap": ccmr_ratio(0.5 * (raw + gap_prev), gap_diff, float(config.get("epsilon", EPS))),
                                    "g_ccmr_gap_db": gain_db(0.5 * (raw + gap_prev), gap_diff, float(config.get("epsilon", EPS))), "valid": True,
                                })
                    rows["ccmr_metrics"].append({
                        "run_id": output_dir.name, "model": "dit", "model_revision": str(config.get("checkpoint")),
                        "seed": seed, "estimator": "exact_batch", "num_conditions": k,
                        "module_family": "dit", "module_name": module, "layer_idx": layer,
                        "step_idx": step_idx, "scheduler_timestep": scheduler_timestep, "denoising_progress": denoising_progress,
                        "feature_numel": int(current[0].numel()), "v_raw": raw,
                        "v_raw_prev": prev_var if prev_var is not None else float("nan"),
                        "v_base": v_base, "v_diff": diff_var, "r_ccmr": ccmr_ratio(v_base, diff_var, float(config.get("epsilon", EPS))) if valid else float("nan"),
                        "g_ccmr_db": gain, "degenerate": bool(valid and v_base < float(config.get("degenerate_relative_threshold", DEGENERATE_THRESHOLD))), "valid": valid,
                    })
                    if prev is not None:
                        l1_now = _per_sample_l1(current, prev)
                        code_l1_now = torch.stack([
                            SimilarityAnalyzer.compute_l1_distance(prev[i], current[i]).detach().float().cpu()
                            for i in range(k)
                        ])
                        if previous_l1[module][layer] is not None and valid_rho_index(step_idx - 1, n_steps):
                            for condition_idx in range(k):
                                rho_clean = clean_rho(previous_l1[module][layer][condition_idx], l1_now[condition_idx], float(config.get("epsilon", EPS)))
                                rho_code = float(SimilarityAnalyzer.compute_rate(
                                    code_l1_now[condition_idx], previous_code_l1[module][layer][condition_idx]
                                ).cpu())
                                rows["rho_per_condition"].append({
                                    "run_id": output_dir.name, "model": "dit", "seed": seed,
                                    "condition_id": class_ids[condition_idx], "rho_scope": "condition", "module_family": "dit",
                                    "module_name": module, "layer_idx": layer, "score_step_idx": step_idx - 1,
                                    "scheduler_timestep": previous_scheduler_timestep, "denoising_progress": (step_idx - 1) / max(n_steps - 1, 1),
                                    "l1_prev": float(previous_l1[module][layer][condition_idx].cpu()), "l1_next": float(l1_now[condition_idx].cpu()),
                                    "rho_clean": rho_clean, "rho_code": rho_code, "valid": True,
                                })
                        previous_l1[module][layer] = l1_now
                        previous_code_l1[module][layer] = code_l1_now
                    previous[module][layer] = current
                    previous_var[module][layer] = raw
                    history[module][layer].append(current)
                    if len(history[module][layer]) > max_gap:
                        history[module][layer].pop(0)
            # Hooks close over the original dictionary; clear it in place so
            # the next diffusion step is captured as well.
            for name in MODULES:
                storage[name].clear()
            previous_scheduler_timestep = scheduler_timestep
    for hook in hooks:
        hook.remove()
    for module in MODULES:
        for layer in range(n_layers):
            for score_idx in (0, n_steps - 1):
                for condition_id in class_ids:
                    rows["rho_per_condition"].append({
                        "run_id": output_dir.name, "model": "dit", "seed": seed,
                        "condition_id": condition_id, "rho_scope": "condition", "module_family": "dit",
                        "module_name": module, "layer_idx": layer, "score_step_idx": score_idx,
                        "scheduler_timestep": None, "denoising_progress": score_idx / max(n_steps - 1, 1),
                        "l1_prev": None, "l1_next": None, "rho_clean": None, "rho_code": None,
                        "valid": False, "invalid_reason": "three_observation_boundary",
                    })
    return rows, latent_hashes, {
        "count": len(permutations),
        "seed": int(config.get("shuffle_seed", 2027)) + seed,
        "hashes": permutation_hashes,
    }


def _write_run(config: dict[str, Any], output_dir: Path, model, diffusion, checkpoint: Path, resume: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(output_dir / "config.json", config)
    conditions = _condition_rows(config)
    environment = environment_snapshot(REPO_ROOT, [checkpoint])
    save_json_atomic(output_dir / "conditions.json", conditions)
    save_json_atomic(output_dir / "environment.json", environment)
    resolved_hash = json_hash(config)
    run_manifest = {
        "run_id": output_dir.name, "model": "dit", "estimator": "exact_batch",
        "cache_enabled": False, "decode_output": False, "resolved_config_hash": resolved_hash,
        "experiment_level": config.get("experiment_level", "smoke"),
        "paper_eligible": bool(config.get("paper_eligible", False)),
        "started_at": utc_now(), "shards": {},
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_hashes: dict[str, Any] = {}
    started = time.perf_counter()
    for seed in config.get("seeds", [0]):
        seed = int(seed)
        configure_determinism(seed)
        seed_dir = output_dir / "shards" / f"seed_{seed}"
        seed_manifest_path = seed_dir / "manifest.json"
        if resume and seed_manifest_path.exists():
            candidate = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
            complete = candidate.get("status") == "complete" and candidate.get("resolved_config_hash") == resolved_hash
            shard_rows = {}
            if complete:
                for table, artifact in candidate.get("tables", {}).items():
                    path = seed_dir / Path(artifact["path"]).name
                    if not path.exists():
                        complete = False
                        break
                    if sha256_file(path) != artifact.get("sha256") or len(read_rows(path)) != int(artifact.get("row_count", -1)):
                        complete = False
                        break
                    shard_rows[table] = read_rows(path)
            if complete:
                for key, values in shard_rows.items():
                    all_rows.setdefault(key, []).extend(values)
                all_hashes[str(seed)] = candidate.get("latent_hashes", {}).get(str(seed), [])
                run_manifest["shards"][str(seed)] = candidate
                continue
        seed_started_at = utc_now()
        seed_started = time.perf_counter()
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            rows, hashes, permutation_manifest = _run_trajectory(model, diffusion, [int(x) for x in config["class_ids"]], seed, config, output_dir)
        except Exception as exc:
            save_json_atomic(seed_manifest_path, {
                "status": "failed", "seed": seed, "resolved_config_hash": resolved_hash,
                "experiment_level": config.get("experiment_level", "smoke"),
                "started_at": seed_started_at, "failed_at": utc_now(),
                "wall_time_s": time.perf_counter() - seed_started,
                "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                "error_type": type(exc).__name__, "error": str(exc),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
            })
            raise
        all_hashes.update(hashes)
        for key, values in rows.items():
            all_rows.setdefault(key, []).extend(values)
        artifacts = {}
        for key, values in rows.items():
            path = seed_dir / f"{key}.csv.gz"
            write_rows_atomic(path, values)
            artifacts[key] = table_artifact(path, len(values))
        shard_manifest = {
            "status": "complete", "seed": seed, "resolved_config_hash": resolved_hash,
            "latent_hashes": hashes, "derangements": permutation_manifest,
            "branch": environment.get("git_branch"), "commit": environment.get("git_commit"),
            "dirty_status": environment.get("git_status"), "exact_command": [sys.executable, *sys.argv],
            "checkpoint": environment.get("checkpoints", [{}])[0], "environment_file": "../../environment.json",
            "condition_bank_hash": json_hash(conditions), "model_dtype": config.get("model_dtype"),
            "statistics_dtype": config.get("statistics_dtype"), "cache_enabled": False, "decode_output": False,
            "hook_locations": ["DiTBlock.forward:block_output.msa", "DiTBlock.forward:block_output.mlp"],
            "hook_count": len(model.blocks), "experiment_level": config.get("experiment_level", "smoke"),
            "paper_eligible": bool(config.get("paper_eligible", False)), "offload": False,
            "resume_requested": bool(resume), "retry_count": 0, "oom": False,
            "started_at": seed_started_at, "completed_at": utc_now(), "wall_time_s": time.perf_counter() - seed_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
            "tables": artifacts,
        }
        save_json_atomic(seed_manifest_path, shard_manifest)
        run_manifest["shards"][str(seed)] = shard_manifest
    save_json_atomic(output_dir / "latent_hashes.json", all_hashes)
    data_dir = output_dir / "tables"
    for key, values in all_rows.items():
        write_rows(data_dir / f"{key}.csv.gz", values)
    summary = {
        "model": "dit", "run_id": output_dir.name, "num_seeds": len(config.get("seeds", [0])),
        "num_conditions": len(config["class_ids"]), "elapsed_s": time.perf_counter() - started,
        "row_counts": {key: len(value) for key, value in all_rows.items()}, "invalid": {},
    }
    save_json_atomic(output_dir / "summary.json", summary)
    run_manifest["completed_at"] = utc_now()
    save_json_atomic(output_dir / "run_manifest.json", run_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CCMR statistics for DiT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = load_yaml(config_path)
    if config.get("cache_enabled", False):
        raise ValueError("CCMR collector requires cache_enabled=false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, diffusion, checkpoint = _make_model(config, device)
    _write_run(config, output_dir, model, diffusion, checkpoint, resume=args.resume)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        save_json_atomic(output_dir / "memory.json", {"peak_allocated_gib": peak})
        print(f"Peak allocated GPU memory: {peak:.3f} GiB")
    print(f"DiT CCMR run complete: {output_dir}")


if __name__ == "__main__":
    main()
