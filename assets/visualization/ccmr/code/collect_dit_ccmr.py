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
    code_rho,
    configure_determinism,
    environment_snapshot,
    gain_db,
    l1_distance,
    load_yaml,
    make_generator,
    rms_feature,
    save_json_atomic,
    tensor_sha256,
    temporal_metrics,
    valid_rho_index,
    write_rows,
)


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
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition")}
    previous: dict[str, list[torch.Tensor | None]] = {name: [None] * n_layers for name in MODULES}
    previous_var: dict[str, list[float | None]] = {name: [None] * n_layers for name in MODULES}
    previous_l1: dict[str, list[torch.Tensor | None]] = {name: [None] * n_layers for name in MODULES}
    history: dict[str, list[list[torch.Tensor]]] = {name: [[] for _ in range(n_layers)] for name in MODULES}
    max_gap = max([int(x) for x in config.get("time_gaps", [1])], default=1)
    shuffle = torch.randperm(k, generator=make_generator(int(config.get("shuffle_seed", 2027)) + seed, "cpu")).to(device)
    if k > 1 and torch.all(shuffle == torch.arange(k, device=device)):
        shuffle = torch.roll(torch.arange(k, device=device), 1)

    sampler = diffusion.ddim_sample_loop_progressive(
        model.forward_with_cfg,
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
                                "scheduler_timestep": step_idx, "output_rms": a_val,
                                "diff_rms": d_val, "r_time": d_val / (a_val + float(config.get("epsilon", EPS))), "valid": True,
                            })
                        permuted = prev[shuffle].float()
                        shuffled_delta = current.float() - permuted
                        shuffled_diff = float(centered_variance(shuffled_delta).cpu())
                        rows["condition_similarity"].append({
                            "run_id": output_dir.name, "model": "dit", "seed": seed,
                            "estimator": "exact_batch", "pair_id_or_condition_group": "all",
                            "module_family": "dit", "module_name": module, "layer_idx": layer,
                            "step_idx": step_idx, "scheduler_timestep": step_idx,
                            "adjacent_condition_cosine": sim, "aligned_g_ccmr_db": gain,
                            "shuffled_g_ccmr_db": gain_db(v_base, shuffled_diff, float(config.get("epsilon", EPS))), "valid": True,
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
                                    "step_idx": step_idx, "raw_pair_distance": raw_pair,
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
                                    "step_idx": step_idx, "time_gap": gap, "v_raw_base": 0.5 * (raw + gap_prev),
                                    "v_diff_gap": gap_diff, "r_ccmr_gap": gap_diff / max(0.5 * (raw + gap_prev), float(config.get("epsilon", EPS))),
                                    "g_ccmr_gap_db": gain_db(0.5 * (raw + gap_prev), gap_diff, float(config.get("epsilon", EPS))), "valid": True,
                                })
                    rows["ccmr_metrics"].append({
                        "run_id": output_dir.name, "model": "dit", "model_revision": str(config.get("checkpoint")),
                        "seed": seed, "estimator": "exact_batch", "num_conditions": k,
                        "module_family": "dit", "module_name": module, "layer_idx": layer,
                        "step_idx": step_idx, "scheduler_timestep": step_idx,
                        "feature_numel": int(current[0].numel()), "v_raw": raw,
                        "v_raw_prev": prev_var if prev_var is not None else float("nan"),
                        "v_base": v_base, "v_diff": diff_var, "r_ccmr": diff_var / max(v_base, float(config.get("epsilon", EPS))) if valid else float("nan"),
                        "g_ccmr_db": gain, "degenerate": bool(valid and v_base < float(config.get("degenerate_relative_threshold", DEGENERATE_THRESHOLD))), "valid": valid,
                    })
                    if prev is not None:
                        l1_now = _per_sample_l1(current, prev)
                        if previous_l1[module][layer] is not None and valid_rho_index(step_idx - 1, n_steps):
                            for condition_idx in range(k):
                                rho_clean = clean_rho(previous_l1[module][layer][condition_idx], l1_now[condition_idx], float(config.get("epsilon", EPS)))
                                rho_code = code_rho(prev[condition_idx], current[condition_idx], current[condition_idx], 1.0e-8) if False else None
                                rows["rho_per_condition"].append({
                                    "run_id": output_dir.name, "model": "dit", "seed": seed,
                                    "condition_id": class_ids[condition_idx], "rho_scope": "condition", "module_family": "dit",
                                    "module_name": module, "layer_idx": layer, "score_step_idx": step_idx - 1,
                                    "l1_prev": float(previous_l1[module][layer][condition_idx].cpu()), "l1_next": float(l1_now[condition_idx].cpu()),
                                    "rho_clean": rho_clean, "rho_code": rho_code, "valid": True,
                                })
                        previous_l1[module][layer] = l1_now
                    previous[module][layer] = current
                    previous_var[module][layer] = raw
                    history[module][layer].append(current)
                    if len(history[module][layer]) > max_gap:
                        history[module][layer].pop(0)
            # Hooks close over the original dictionary; clear it in place so
            # the next diffusion step is captured as well.
            for name in MODULES:
                storage[name].clear()
    for hook in hooks:
        hook.remove()
    return rows, latent_hashes


def _write_run(config: dict[str, Any], output_dir: Path, model, diffusion, checkpoint: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(output_dir / "config.json", config)
    save_json_atomic(output_dir / "conditions.json", _condition_rows(config))
    save_json_atomic(output_dir / "environment.json", environment_snapshot(REPO_ROOT, [checkpoint]))
    run_manifest = {"run_id": output_dir.name, "model": "dit", "estimator": "exact_batch", "cache_enabled": False, "shards": {}}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_hashes: dict[str, Any] = {}
    started = time.perf_counter()
    for seed in config.get("seeds", [0]):
        seed = int(seed)
        configure_determinism(seed)
        rows, hashes = _run_trajectory(model, diffusion, [int(x) for x in config["class_ids"]], seed, config, output_dir)
        all_hashes.update(hashes)
        for key, values in rows.items():
            all_rows.setdefault(key, []).extend(values)
        seed_path = output_dir / "shards" / f"seed_{seed}.json"
        save_json_atomic(seed_path, {"seed": seed, "row_counts": {key: len(value) for key, value in rows.items()}})
        run_manifest["shards"][str(seed)] = {"status": "complete", "rows": {key: len(value) for key, value in rows.items()}}
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
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Already complete: {output_dir}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, diffusion, checkpoint = _make_model(config, device)
    _write_run(config, output_dir, model, diffusion, checkpoint)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        save_json_atomic(output_dir / "memory.json", {"peak_allocated_gib": peak})
        print(f"Peak allocated GPU memory: {peak:.3f} GiB")
    print(f"DiT CCMR run complete: {output_dir}")


if __name__ == "__main__":
    main()
