#!/usr/bin/env python3
"""Collect CCMR statistics from full-compute FLUX trajectories.

The existing dynamic FLUX wrapper is used only to expose the same block output
locations as Finegrained Cache.  Every step and every layer cache decision is
disabled during collection.
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
    DEGENERATE_THRESHOLD,
    centered_variance,
    ccmr_ratio,
    clean_rho,
    compact_gap_delta,
    compact_pair_difference,
    configure_determinism,
    environment_snapshot,
    gain_db,
    json_hash,
    l1_distance,
    load_yaml,
    make_generator,
    save_json_atomic,
    select_pairs,
    sha256_file,
    table_artifact,
    tensor_sha256,
    temporal_metrics,
    valid_rho_index,
    utc_now,
    write_rows,
)
from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive  # noqa: E402


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


def _hook(
    storage: dict[str, dict[int, torch.Tensor]],
    layer: int,
    family: str,
    compact: bool = False,
    feature_device: str = "cpu",
):
    def callback(_module, _inputs, output):
        block_outputs = output[2] if family == "double" else output[1]
        for name in MODULES[family]:
            # Statistics are computed after the full transformer call.  Move
            # each captured feature off the GPU immediately so a 1024px pair
            # does not retain every block activation alongside the model.
            value = block_outputs[name].detach()
            if compact:
                # With exactly two conditions, all cross-condition CCMR
                # quantities depend on this centered pair difference.  Keep
                # one FP32 pair tensor instead of both full activations.  The
                # pair is half the batch and preserves FP32 statistics.
                value = value[0].float() - value[1].float()
            storage[f"{family}.{name}"][layer] = value if feature_device == "gpu" else value.to("cpu")

    return callback


def _features_for_storage(storage, num_double: int, num_single: int, batch: int):
    result = {}
    for family, names, count in (("double", MODULES["double"], num_double), ("single", MODULES["single"], num_single)):
        for name in names:
            key = f"{family}.{name}"
            result[key] = [
                (storage[key][idx] if storage[key][idx].ndim == 2 else storage[key][idx][:batch])
                .detach().contiguous()
                for idx in range(count)
            ]
    return result


def _per_sample_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.detach().float() - b.detach().float()).reshape(a.shape[0], -1).abs().sum(dim=1)


def _pair_energy(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((((a.detach().float() - b.detach().float()) ** 2).mean() / 2.0).cpu())


def _pair_statistics(current: torch.Tensor, previous: torch.Tensor | None):
    """Vectorized statistics for the exact two-prompt FLUX batch.

    For K=2, centered population variance is ``mean((a-b)^2)/4``.  Computing
    the pair difference once avoids repeatedly materializing centered copies
    of a 1024px feature for cosine, shuffled and pair-distance diagnostics.
    """
    cur = current.detach().float()
    compact = cur.ndim == 2
    cur_pair = compact_pair_difference(cur)
    raw = float((cur_pair.square().mean() * 0.25).item())
    if previous is None:
        return cur, raw, None
    prev = previous.detach().float()
    prev_pair = compact_pair_difference(prev)
    delta = cur - prev
    delta_pair = delta if compact else cur_pair - prev_pair
    diff_var = float((delta_pair.square().mean() * 0.25).item())
    v_base = 0.5 * (raw + float((prev_pair.square().mean() * 0.25).item()))
    sim = float(torch.nn.functional.cosine_similarity(cur_pair.reshape(1, -1), prev_pair.reshape(1, -1), dim=1).item())
    shuffled_var = float(((cur_pair + prev_pair).square().mean() * 0.25).item())
    raw_pair_distance = float((cur_pair.square().mean() * 0.5).item())
    diff_pair_distance = float((delta_pair.square().mean() * 0.5).item())
    if compact:
        # The pair representation has no separate condition axis.  Mirror the
        # pair-level values for the two prompt rows; CCMR and rank statistics
        # remain exact, while temporal/rho diagnostics stay schema-compatible.
        prev_rms_scalar = prev_pair.square().mean().sqrt()
        cur_rms_scalar = cur_pair.square().mean().sqrt()
        delta_rms_scalar = delta_pair.square().mean().sqrt()
        temporal_a = torch.stack([torch.sqrt(0.5 * (prev_rms_scalar.square() + cur_rms_scalar.square()))] * 2)
        temporal_d = torch.stack([delta_rms_scalar] * 2)
        l1_scalar = delta_pair.abs().sum() * 0.5
        l1_now = torch.stack([l1_scalar, l1_scalar])
    else:
        prev_rms = prev.square().mean(dim=tuple(range(1, prev.ndim)))
        cur_rms = cur.square().mean(dim=tuple(range(1, cur.ndim)))
        delta_rms = delta.square().mean(dim=tuple(range(1, delta.ndim)))
        temporal_a = torch.sqrt(0.5 * (prev_rms + cur_rms))
        temporal_d = torch.sqrt(delta_rms)
        l1_now = delta.abs().reshape(delta.shape[0], -1).sum(dim=1)
    return cur, raw, {
        "prev": prev, "delta": delta, "diff_var": diff_var, "v_base": v_base,
        "sim": sim, "shuffled_var": shuffled_var,
        "raw_pair_distance": raw_pair_distance, "diff_pair_distance": diff_pair_distance,
        "temporal_a": temporal_a, "temporal_d": temporal_d, "l1_now": l1_now,
    }


def _run_pair(
    pipe,
    dynamic_model,
    pair: tuple[dict[str, Any], dict[str, Any]],
    seed: int,
    config: dict[str, Any],
    output_dir: Path,
    total_conditions: int,
):
    device = pipe._execution_device
    prompts = [pair[0]["prompt"], pair[1]["prompt"]]
    prompt_ids = [pair[0]["id"], pair[1]["id"]]
    dtype = torch.bfloat16
    generator = make_generator(seed, "cpu")
    num_channels = dynamic_model.config.in_channels // 4
    # Create one packed latent and duplicate it for the two prompts.  The
    # local progressive helper normally derives image IDs from an externally
    # supplied latent using an un-packed shape; provide a temporary
    # instance-level prepare_latents shim so it receives the correctly shaped
    # IDs while retaining the helper's official scheduler/denoising path.
    one, one_ids = pipe.prepare_latents(
        1,
        num_channels,
        int(config["height"]),
        int(config["width"]),
        dtype,
        device,
        generator,
        None,
    )
    latent = one.expand(2, -1, -1).clone()
    latent_ids = one_ids.expand(2, -1, -1).clone()
    latent_hashes = [tensor_sha256(one), tensor_sha256(one)]
    storage = {f"{family}.{name}": {} for family, names in MODULES.items() for name in names}
    compact_pair_stats = bool(config.get("compact_pair_stats", True))
    feature_device = str(config.get("feature_device", "cpu")).lower()
    if feature_device not in {"cpu", "gpu"}:
        raise ValueError("feature_device must be cpu or gpu")
    hooks = []
    for idx, block in enumerate(dynamic_model.transformer_blocks):
        hooks.append(block.register_forward_hook(_hook(storage, idx, "double", compact_pair_stats, feature_device)))
    for idx, block in enumerate(dynamic_model.single_transformer_blocks):
        hooks.append(block.register_forward_hook(_hook(storage, idx, "single", compact_pair_stats, feature_device)))
    n_steps = int(config["num_inference_steps"])
    module_counts = {"double": dynamic_model.num_layers, "single": dynamic_model.num_single_layers}
    previous: dict[str, list[torch.Tensor | None]] = {
        f"{family}.{name}": [None] * module_counts[family]
        for family, names in MODULES.items() for name in names
    }
    previous_var: dict[str, list[float | None]] = {
        f"{family}.{name}": [None] * module_counts[family]
        for family, names in MODULES.items() for name in names
    }
    previous_l1: dict[str, list[torch.Tensor | None]] = {
        f"{family}.{name}": [None] * module_counts[family]
        for family, names in MODULES.items() for name in names
    }
    history: dict[str, list[list[torch.Tensor]]] = {
        f"{family}.{name}": [[] for _ in range(module_counts[family])]
        for family, names in MODULES.items() for name in names
    }
    rows = {"ccmr_metrics": [], "condition_similarity": [], "condition_distance": [], "temporal_metrics": [], "time_gap_metrics": [], "rho_per_condition": [], "pair_metrics": [], "alignment_control": []}
    max_gap = max([int(x) for x in config.get("time_gaps", [1])], default=1)
    original_prepare_latents = pipe.prepare_latents

    def _prepared_latents(_batch_size, _channels, _height, _width, _dtype, _device, _generator, _latents=None):
        return latent, latent_ids

    pipe.prepare_latents = _prepared_latents
    try:
        sampler = flux_sample_loop_progressive(
            pipe,
            prompt=prompts,
            height=int(config["height"]), width=int(config["width"]),
            num_inference_steps=n_steps,
            guidance_scale=float(config.get("guidance_scale", 3.5)),
            true_cfg_scale=float(config.get("true_cfg_scale", 1.0)),
            max_sequence_length=int(config.get("max_sequence_length", 512)),
            generator=generator,
            latents=None,
            output_type="latent",
        )
        with torch.inference_mode():
            for step_idx, state in enumerate(sampler):
                if "final_image" in state:
                    continue
                denoising_progress = step_idx / max(n_steps - 1, 1)
                features = _features_for_storage(storage, module_counts["double"], module_counts["single"], 2)
                for key, feature_list in features.items():
                    family = key.split(".", 1)[0]
                    module_name = key.split(".", 1)[1]
                    for layer, current in enumerate(feature_list):
                        prev = previous[key][layer]
                        prev_var = previous_var[key][layer]
                        valid = prev is not None
                        cur_float, raw, stats = _pair_statistics(current, prev)
                        diff_var = v_base = gain = sim = float("nan")
                        if valid:
                            delta = stats["delta"]
                            diff_var = stats["diff_var"]
                            v_base = stats["v_base"]
                            gain = gain_db(v_base, diff_var, float(config.get("epsilon", EPS)))
                            sim = stats["sim"]
                            for condition_idx in range(2):
                                a = float(stats["temporal_a"][condition_idx].item())
                                d = float(stats["temporal_d"][condition_idx].item())
                                rows["temporal_metrics"].append({
                                    "run_id": output_dir.name, "model": "flux", "seed": seed,
                                    "condition_id": prompt_ids[condition_idx], "rho_scope": "pair_difference", "module_family": family,
                                    "module_name": module_name, "layer_idx": layer, "step_idx": step_idx,
                                    "scheduler_timestep": float(state["timestep"].detach().cpu()), "denoising_progress": denoising_progress, "output_rms": a,
                                    "diff_rms": d, "r_time": d / (a + float(config.get("epsilon", EPS))), "valid": True,
                                })
                            rows["condition_similarity"].append({
                                "run_id": output_dir.name, "model": "flux", "seed": seed,
                                "estimator": "pairwise" if config.get("condition_backend") == "pairwise" else "exact_batch",
                                "pair_id_or_condition_group": f"{prompt_ids[0]}__{prompt_ids[1]}",
                                "module_family": family, "module_name": module_name, "layer_idx": layer,
                                "step_idx": step_idx, "scheduler_timestep": float(state["timestep"].detach().cpu()),
                                "denoising_progress": denoising_progress,
                                "adjacent_condition_cosine": sim, "aligned_g_ccmr_db": gain,
                                "shuffled_g_ccmr_db": gain_db(v_base, stats["shuffled_var"], float(config.get("epsilon", EPS))), "valid": True,
                            })
                            shuffled_gain = gain_db(v_base, stats["shuffled_var"], float(config.get("epsilon", EPS)))
                            rows["alignment_control"].append({
                                "run_id": output_dir.name, "model": "flux", "seed": seed,
                                "pair_id": f"{prompt_ids[0]}__{prompt_ids[1]}", "shuffle_trial": 0,
                                "permutation": "[1,0]", "permutation_hash": tensor_sha256(torch.tensor([1, 0], dtype=torch.int64)),
                                "module_family": family, "module_name": module_name, "layer_idx": layer,
                                "step_idx": step_idx, "scheduler_timestep": float(state["timestep"].detach().cpu()),
                                "denoising_progress": denoising_progress, "g_aligned_db": gain,
                                "g_shuffled_db": shuffled_gain, "delta_g_db": gain - shuffled_gain, "valid": True,
                            })
                            rows["condition_distance"].append({
                                "run_id": output_dir.name, "model": "flux", "seed": seed,
                                "pair_id": f"{prompt_ids[0]}__{prompt_ids[1]}", "condition_i": prompt_ids[0], "condition_j": prompt_ids[1], "module_family": family,
                                "module_name": module_name, "layer_idx": layer, "step_idx": step_idx,
                                "scheduler_timestep": float(state["timestep"].detach().cpu()), "denoising_progress": denoising_progress,
                                "raw_pair_distance": stats["raw_pair_distance"],
                                "raw_prev_pair_distance": float((compact_pair_difference(prev).square().mean() * 0.5).item()),
                                "diff_pair_distance": stats["diff_pair_distance"], "valid": True,
                            })
                            for gap in config.get("time_gaps", [1]):
                                gap = int(gap)
                                if gap <= len(history[key][layer]):
                                    gap_feature = history[key][layer][-gap].float()
                                    gap_pair = compact_pair_difference(gap_feature)
                                    gap_raw = float((gap_pair.square().mean() * 0.25).item())
                                    gap_delta_pair = compact_gap_delta(cur_float, gap_feature)
                                    gap_diff = float((gap_delta_pair.square().mean() * 0.25).item())
                                    current_pair = compact_pair_difference(cur_float)
                                    rows["time_gap_metrics"].append({
                                        "run_id": output_dir.name, "model": "flux", "seed": seed,
                                        "estimator": "pairwise", "num_conditions": total_conditions,
                                        "pair_id": f"{prompt_ids[0]}__{prompt_ids[1]}",
                                        "condition_i": prompt_ids[0], "condition_j": prompt_ids[1],
                                        "module_family": family, "module_name": module_name, "layer_idx": layer,
                                        "step_idx": step_idx, "scheduler_timestep": float(state["timestep"].detach().cpu()),
                                        "denoising_progress": denoising_progress, "time_gap": gap, "v_raw_base": 0.5 * (raw + gap_raw),
                                        # Persist pair energies so the
                                        # aggregator can apply the same
                                        # finite-population correction as the
                                        # ordinary pairwise CCMR table when
                                        # multiple FLUX pairs are collected.
                                        "raw_current_pair_distance": float((current_pair.square().mean() * 0.5).item()),
                                        "raw_previous_pair_distance": float((gap_pair.square().mean() * 0.5).item()),
                                        "diff_pair_distance": float((gap_delta_pair.square().mean() * 0.5).item()),
                                        "v_diff_gap": gap_diff, "r_ccmr_gap": ccmr_ratio(0.5 * (raw + gap_raw), gap_diff, float(config.get("epsilon", EPS))),
                                        "g_ccmr_gap_db": gain_db(0.5 * (raw + gap_raw), gap_diff, float(config.get("epsilon", EPS))), "valid": True,
                                    })
                        rows["ccmr_metrics"].append({
                            "run_id": output_dir.name, "model": "flux", "model_revision": str(config.get("model_path")),
                            "seed": seed, "estimator": "pairwise" if config.get("condition_backend") == "pairwise" else "exact_batch",
                            # Pairwise rows represent a sample from the full
                            # prompt population.  Keep the population size in
                            # the row so the aggregator can apply the finite
                            # population correction (K-1)/(K-2).
                            "num_conditions": total_conditions, "module_family": family, "module_name": module_name,
                            "layer_idx": layer, "step_idx": step_idx, "scheduler_timestep": float(state["timestep"].detach().cpu()),
                            "denoising_progress": denoising_progress,
                            "feature_numel": int(current.numel()), "v_raw": raw,
                            "v_raw_prev": prev_var if prev_var is not None else float("nan"), "v_base": v_base,
                            "v_diff": diff_var, "r_ccmr": ccmr_ratio(v_base, diff_var, float(config.get("epsilon", EPS))) if valid else float("nan"),
                            "g_ccmr_db": gain, "degenerate": bool(valid and v_base < float(config.get("degenerate_relative_threshold", DEGENERATE_THRESHOLD))), "valid": valid,
                        })
                        if prev is not None:
                            l1_now = stats["l1_now"]
                            if previous_l1[key][layer] is not None and valid_rho_index(step_idx - 1, n_steps):
                                for condition_idx in range(2):
                                    rows["rho_per_condition"].append({
                                        "run_id": output_dir.name, "model": "flux", "seed": seed,
                                        "condition_id": prompt_ids[condition_idx], "rho_scope": "pair_difference", "module_family": family,
                                        "module_name": module_name, "layer_idx": layer, "score_step_idx": step_idx - 1,
                                        "l1_prev": float(previous_l1[key][layer][condition_idx].cpu()), "l1_next": float(l1_now[condition_idx].cpu()),
                                        "rho_clean": clean_rho(previous_l1[key][layer][condition_idx], l1_now[condition_idx], float(config.get("epsilon", EPS))),
                                        "rho_code": None, "valid": True,
                                    })
                            previous_l1[key][layer] = l1_now
                        # Persist only the compact pair trajectory.  It is
                        # already FP32 (the pair subtraction happens in the
                        # hook), but has half the batch volume of the original
                        # two-condition activation history.
                        previous[key][layer] = current.detach()
                        previous_var[key][layer] = raw
                        history[key][layer].append(current.detach())
                        if len(history[key][layer]) > max_gap:
                            history[key][layer].pop(0)
                # Preserve the dictionary object captured by the hooks.
                for key in storage:
                    storage[key].clear()
    finally:
        pipe.prepare_latents = original_prepare_latents
        for hook in hooks:
            hook.remove()
        dynamic_model.reset()
    # Pair metrics are intentionally kept independent of the condition-level
    # tables so the aggregator can apply the finite population correction.
    distance_by_key = {
        (r["module_family"], r["module_name"], r["layer_idx"], r["step_idx"]): r
        for r in rows["condition_distance"]
    }
    for row in rows["ccmr_metrics"]:
        if row["valid"]:
            pair = (prompt_ids[0], prompt_ids[1])
            drow = distance_by_key.get((row["module_family"], row["module_name"], row["layer_idx"], row["step_idx"]), {})
            rows["pair_metrics"].append({
                "run_id": output_dir.name, "model": "flux", "seed": seed,
                "pair_id": f"{pair[0]}__{pair[1]}", "condition_i": pair[0], "condition_j": pair[1],
                "pair_selection_seed": config.get("pair_selection_seed"), "module_family": row["module_family"],
                "module_name": row["module_name"], "layer_idx": row["layer_idx"], "step_idx": row["step_idx"],
                "scheduler_timestep": row["scheduler_timestep"],
                "raw_pair_energy": drow.get("raw_pair_distance"),
                "raw_prev_pair_energy": row.get("v_raw_prev"),
                "diff_pair_energy": drow.get("diff_pair_distance"), "shuffled_diff_pair_energy": None,
                "time_gap": None, "time_gap_diff_pair_energy": None, "valid": True,
            })
    return rows, latent_hashes


def _write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.gz")
    write_rows(temporary, rows)
    os.replace(temporary, path)


def _write_run(config: dict[str, Any], output_dir: Path, prompts: list[dict[str, Any]], pipe, dynamic_model, checkpoint: Path, resume: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(output_dir / "config.json", config)
    environment = environment_snapshot(REPO_ROOT, [checkpoint])
    save_json_atomic(output_dir / "conditions.json", prompts)
    save_json_atomic(output_dir / "environment.json", environment)
    pairs = select_pairs(len(prompts), int(config.get("num_condition_pairs", 1)), int(config.get("pair_selection_seed", 2027))) if config.get("condition_backend") == "pairwise" else [(0, 1)]
    save_json_atomic(output_dir / "pair_selection.json", {"seed": config.get("pair_selection_seed"), "pairs": [[prompts[i]["id"], prompts[j]["id"]] for i, j in pairs]})
    resolved_hash = json_hash(config)
    manifest = {
        "run_id": output_dir.name, "model": "flux", "estimator": config.get("condition_backend"),
        "cache_enabled": False, "decode_output": False, "resolved_config_hash": resolved_hash,
        "experiment_level": config.get("experiment_level", "smoke"),
        "paper_eligible": bool(config.get("paper_eligible", False)),
        "started_at": utc_now(), "shards": {},
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_hashes: dict[str, Any] = {}
    started = time.perf_counter()
    for seed in config.get("seeds", [0]):
        seed = int(seed); configure_determinism(seed)
        for pair_idx, (i, j) in enumerate(pairs):
            shard_dir = output_dir / "shards" / f"seed_{seed}_pair_{pair_idx}"
            marker = shard_dir / "manifest.json"
            if resume and marker.exists():
                try:
                    marker_value = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker_value = {}
                if marker_value.get("status") == "complete" and marker_value.get("resolved_config_hash") == resolved_hash:
                    shard_rows: dict[str, list[dict[str, Any]]] = {}
                    complete = True
                    for table in ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition", "pair_metrics", "alignment_control"):
                        table_path = shard_dir / f"{table}.csv.gz"
                        artifact = marker_value.get("tables", {}).get(table, {})
                        if (not table_path.exists() or sha256_file(table_path) != artifact.get("sha256")
                                or len(read_rows(table_path)) != int(artifact.get("row_count", -1))):
                            complete = False
                            break
                        shard_rows[table] = read_rows(table_path)
                    if complete:
                        for key, values in shard_rows.items():
                            all_rows.setdefault(key, []).extend(values)
                        # New shards persist the duplicated-latent hashes so a
                        # resumed run still has a complete reproducibility
                        # manifest.  Older shards may not contain this field;
                        # they remain readable, but their hash entry is left
                        # absent rather than fabricated.
                        if marker_value.get("latent_hashes") is not None:
                            all_hashes[f"{seed}:{pair_idx}"] = marker_value["latent_hashes"]
                        manifest["shards"][f"{seed}:{pair_idx}"] = marker_value
                        continue
            shard_started_at = utc_now()
            shard_started = time.perf_counter()
            device = pipe._execution_device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                rows, hashes = _run_pair(
                    pipe, dynamic_model, (prompts[i], prompts[j]), seed,
                    config, output_dir, len(prompts),
                )
            except Exception as exc:
                save_json_atomic(marker, {
                    "status": "failed", "seed": seed, "pair_idx": pair_idx,
                    "pair_id": f"{prompts[i]['id']}__{prompts[j]['id']}",
                    "resolved_config_hash": resolved_hash,
                    "experiment_level": config.get("experiment_level", "smoke"),
                    "started_at": shard_started_at, "failed_at": utc_now(),
                    "wall_time_s": time.perf_counter() - shard_started,
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                })
                raise
            all_hashes[f"{seed}:{pair_idx}"] = hashes
            for key, values in rows.items():
                all_rows.setdefault(key, []).extend(values)
            for table, values in rows.items():
                _write_rows_atomic(shard_dir / f"{table}.csv.gz", values)
            artifacts = {
                table: table_artifact(shard_dir / f"{table}.csv.gz", len(values))
                for table, values in rows.items()
            }
            marker_value = {
                "status": "complete",
                "seed": seed,
                "pair_idx": pair_idx,
                "pair_id": f"{prompts[i]['id']}__{prompts[j]['id']}",
                "resolved_config_hash": resolved_hash,
                "latent_hashes": hashes,
                "branch": environment.get("git_branch"), "commit": environment.get("git_commit"),
                "dirty_status": environment.get("git_status"), "exact_command": [sys.executable, *sys.argv],
                "checkpoint": environment.get("checkpoints", [{}])[0], "environment_file": "../../environment.json",
                "condition_bank_hash": json_hash(prompts), "pair_selection_hash": json_hash([[prompts[a]["id"], prompts[b]["id"]] for a, b in pairs]),
                "model_dtype": config.get("model_dtype"), "statistics_dtype": config.get("statistics_dtype"),
                "cache_enabled": False, "decode_output": False,
                "hook_locations": ["FluxTransformerBlock.forward:block_outputs", "FluxSingleTransformerBlock.forward:block_outputs"],
                "hook_count": len(dynamic_model.transformer_blocks) + len(dynamic_model.single_transformer_blocks),
                "experiment_level": config.get("experiment_level", "smoke"), "paper_eligible": bool(config.get("paper_eligible", False)),
                "offload": False, "resume_requested": bool(resume), "retry_count": 0, "oom": False,
                "started_at": shard_started_at, "completed_at": utc_now(), "wall_time_s": time.perf_counter() - shard_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                "row_counts": {key: len(value) for key, value in rows.items()},
                "tables": artifacts,
            }
            save_json_atomic(marker, marker_value)
            manifest["shards"][f"{seed}:{pair_idx}"] = marker_value
    for key, values in all_rows.items():
        _write_rows_atomic(output_dir / "tables" / f"{key}.csv.gz", values)
    save_json_atomic(output_dir / "latent_hashes.json", all_hashes)
    save_json_atomic(output_dir / "summary.json", {"model": "flux", "run_id": output_dir.name, "num_seeds": len(config.get("seeds", [0])), "num_pairs": len(pairs), "elapsed_s": time.perf_counter() - started, "row_counts": {key: len(value) for key, value in all_rows.items()}})
    manifest["completed_at"] = utc_now()
    save_json_atomic(output_dir / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CCMR statistics for FLUX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    if config.get("cache_enabled", False):
        raise ValueError("CCMR collector requires cache_enabled=false")
    from diffusers import FluxPipeline
    from dynamic_flux import DynamicFluxTransformer2DModel

    checkpoint = Path(config["model_path"])
    dtype = torch.bfloat16 if str(config.get("model_dtype", "bfloat16")).lower() in {"bfloat16", "bf16"} else torch.float16
    pipe = FluxPipeline.from_pretrained(str(checkpoint), torch_dtype=dtype, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the transformer resident on the GPU for deterministic feature
    # hooks and timing.  The local 4090 has sufficient memory for the BF16
    # smoke/formal configuration; CPU offload hooks attached before wrapping
    # the transformer can otherwise move captured tensors unexpectedly.
    pipe.to(device)
    dynamic_model = DynamicFluxTransformer2DModel(pipe.transformer, int(config["num_inference_steps"]))
    dynamic_model.step_cache_bool = [False] * int(config["num_inference_steps"])
    dynamic_model.block_cache_enable = False
    dynamic_model.reset()
    pipe.transformer = dynamic_model
    prompts = _load_prompts(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _write_run(config, output_dir, prompts, pipe, dynamic_model, checkpoint, resume=args.resume)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        save_json_atomic(output_dir / "memory.json", {"peak_allocated_gib": peak})
        print(f"Peak allocated GPU memory: {peak:.3f} GiB")
    print(f"FLUX CCMR run complete: {output_dir}")


if __name__ == "__main__":
    main()
