#!/usr/bin/env python3
"""Standalone Wan2.1 SeaCache + Finegrained Cache hybrid sampler.

Wan orchestration follows the official Wan2.1 generator.  SeaCache is adapted
from ali-vilab/SeaCache (commit 8dcf490).  The referenced checkout has no
visible license; confirm redistribution rights before publishing this file.
No executable module is imported from local baseline or open-source scripts.
"""

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from PIL import Image
import time

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

WAN_CACHE_MODULES = ("self_attn", "cross_attn", "ffn")
CACHE_SCOPE = "hybrid"
POLICY_VARIANT = "hybrid_seacache"
STEP_POLICY = "seacache"
RATE_METHOD = "three_point_l1"
RATE_CHUNK_SIZE = 1_048_576
SOURCE_COMMIT = "8dcf490"

def seacache_boundaries(args):
    if args.use_ret_steps:
        return 10, args.sample_steps * 2
    return 2, args.sample_steps * 2 - 2


def ab_from_scheduler(scheduler, idx: int):
    if scheduler is None:
        raise RuntimeError("SeaCache scheduler bridge is not active")
    sigma = float(scheduler.sigmas[idx]) if hasattr(scheduler, "sigmas") else (
        1.0 - (idx + 1) / float(getattr(scheduler, "num_inference_steps", idx + 1))
    )
    sigma = max(1e-6, min(1.0 - 1e-6, sigma))
    return 1.0 - sigma, sigma


def apply_sea_from_ab(x, a, b, power_exp=3.0, dims=(-2, -3, -4), eps=1e-16):
    """Official full FFT/IFFT separable SEA Wiener filter."""
    original_dtype = x.dtype
    x32 = x.contiguous().float()
    spectrum = torch.fft.fftn(x32, dim=dims)
    gain = None
    for axis in dims:
        frequency = torch.fft.fftfreq(
            x32.shape[axis], device=x32.device, dtype=torch.float32
        ).abs()
        signal_power = 1.0 / (frequency.pow(power_exp) + eps)
        axis_gain = (a * signal_power) / (
            a * a * signal_power + b * b + eps
        )
        shape = [1] * x32.ndim
        shape[axis] = axis_gain.numel()
        axis_gain = axis_gain.reshape(shape)
        gain = axis_gain if gain is None else gain * axis_gain
    mean_gain = gain.mean()
    if torch.isfinite(mean_gain) and mean_gain > 0:
        gain = gain / mean_gain
    return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)


def apply_sea_with_scheduler(x, scheduler, idx):
    a, b = ab_from_scheduler(scheduler, idx)
    return apply_sea_from_ab(x, a, b)


def distributed_relative_l1(current, previous):
    pair = torch.stack([
        (current - previous).abs().sum(dtype=torch.float32),
        previous.abs().sum(dtype=torch.float32),
    ])
    if dist.is_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float((pair[0] / pair[1].clamp_min(1e-16)).item())


@contextmanager
def sea_scheduler_bridge(model):
    """Expose the scheduler created inside the current official pipeline loop."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

    classes = (FlowUniPCMultistepScheduler, FlowDPMSolverMultistepScheduler)
    originals = {cls: cls.set_timesteps for cls in classes}
    try:
        for cls, original in originals.items():
            def wrapped(instance, *args, _original=original, **kwargs):
                result = _original(instance, *args, **kwargs)
                model.seacache_scheduler = instance
                return result
            cls.set_timesteps = wrapped
        yield
    finally:
        for cls, original in originals.items():
            cls.set_timesteps = original
        model.seacache_scheduler = None

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "A graceful portrait of a lady in simple elegant style.",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "image": "examples/i2v_input.JPG",
    },
    "flf2v-14B": {
        "prompt": "A small blue bird takes off from the ground and flaps its wings into a bright blue sky.",
        "first_frame": "examples/flf2v_input_first_frame.png",
        "last_frame": "examples/flf2v_input_last_frame.png",
    },
    "vace-1.3B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "A festive scene where a little girl in red spring clothing happily plays with a cute cartoon snake.",
    },
    "vace-14B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "A festive scene where a little girl in red spring clothing happily plays with a cute cartoon snake.",
    },
}


def compute_l1_distance(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    chunk_size: int = RATE_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ||x_end - x_start + eps||_1 with chunked FP32 sums."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    x_start = x_start.detach().reshape(-1)
    x_end = x_end.detach().reshape(-1)
    if x_start.numel() != x_end.numel():
        raise ValueError(
            "Rate feature sizes must match, got "
            f"{x_start.numel()} and {x_end.numel()}."
        )

    if x_end.is_cuda:
        compute_device = x_end.device
    elif x_start.is_cuda:
        compute_device = x_start.device
    else:
        compute_device = x_start.device
    total = torch.zeros((), device=compute_device, dtype=torch.float32)
    for start in range(0, x_start.numel(), chunk_size):
        end = min(start + chunk_size, x_start.numel())
        start_chunk = x_start[start:end].to(
            compute_device, dtype=torch.float32, non_blocking=True
        )
        end_chunk = x_end[start:end].to(
            compute_device, dtype=torch.float32, non_blocking=True
        )
        diff = end_chunk - start_chunk + 1e-8
        total += diff.abs().sum(dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def compute_rate(
    x_prev: torch.Tensor,
    x: torch.Tensor,
    x_post: torch.Tensor,
    diff_prev_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ||x_post-x_prev+eps||_1 / ||x-x_prev+eps||_1."""
    if diff_prev_norm is None:
        diff_prev_norm = compute_l1_distance(x_prev, x)
    diff_post_norm = compute_l1_distance(x_prev, x_post)
    return diff_post_norm / diff_prev_norm.clamp_min(1e-8)


class FeatureChangeAnalyzer:
    """Conditional-only three-point analyzer for Wan layer features."""

    def __init__(
        self,
        num_steps: int,
        num_layers: int,
        correction_mode: bool = False,
        module_cache_state: Optional[Dict[str, List[List[bool]]]] = None,
        active_modules=WAN_CACHE_MODULES,
        feature_device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.correction_mode = correction_mode
        self.module_cache_state = module_cache_state
        self.active_modules = tuple(active_modules)
        self.feature_device = feature_device

        unknown_modules = set(self.active_modules) - set(WAN_CACHE_MODULES)
        if unknown_modules:
            raise ValueError(f"Unknown Wan cache modules: {sorted(unknown_modules)}")
        if correction_mode and module_cache_state is None:
            raise ValueError(
                "Correction mode requires a provisional module cache book."
            )

        self.module_scores = {
            k: torch.ones((num_steps, num_layers), dtype=torch.float32)
            for k in WAN_CACHE_MODULES
        }
        self._module_state = {
            k: [False] * num_layers for k in self.active_modules
        }
        self._prev_module_feat = {
            k: [None] * num_layers for k in self.active_modules
        }
        self._current_module_feat = {
            k: [None] * num_layers for k in self.active_modules
        }
        self._prev_module_norm = {
            k: [None] * num_layers for k in self.active_modules
        }

    @staticmethod
    def _refresh_state(is_cached, active_state):
        if not is_cached:
            return True, False
        if not active_state:
            return True, True
        return False, True

    def _store_feature(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.detach()
        if self.feature_device == "gpu":
            return feature
        cpu = torch.empty_like(feature, device="cpu", pin_memory=True)
        cpu.copy_(feature, non_blocking=False)
        return cpu

    def _rotate_module(
        self,
        mod_name: str,
        layer_idx: int,
        feature: torch.Tensor,
    ):
        current = self._current_module_feat[mod_name][layer_idx]
        next_norm = compute_l1_distance(current, feature)
        self._prev_module_feat[mod_name][layer_idx] = current
        self._current_module_feat[mod_name][layer_idx] = self._store_feature(feature)
        self._prev_module_norm[mod_name][layer_idx] = next_norm

    def update_module(
        self,
        policy_step_idx: int,
        layer_idx: int,
        mod_name: str,
        feature: torch.Tensor,
    ):
        """Stream one conditional-branch module feature into its trajectory."""
        if mod_name not in self.active_modules or feature is None:
            return

        feature = feature.detach()
        current = self._current_module_feat[mod_name][layer_idx]
        if policy_step_idx == 0 or current is None:
            self._current_module_feat[mod_name][layer_idx] = self._store_feature(feature)
            return

        previous = self._prev_module_feat[mod_name][layer_idx]
        if policy_step_idx == 1 or previous is None:
            self._prev_module_feat[mod_name][layer_idx] = current
            self._current_module_feat[mod_name][layer_idx] = self._store_feature(feature)
            self._prev_module_norm[mod_name][layer_idx] = compute_l1_distance(
                current,
                feature,
            )
            return

        score_idx = policy_step_idx - 1
        score = compute_rate(
            previous,
            current,
            feature,
            self._prev_module_norm[mod_name][layer_idx],
        )
        self.module_scores[mod_name][score_idx, layer_idx] = float(score.item())

        if not self.correction_mode:
            self._rotate_module(mod_name, layer_idx, feature)
            return

        should_refresh, active_state = self._refresh_state(
            bool(self.module_cache_state[mod_name][score_idx][layer_idx]),
            self._module_state[mod_name][layer_idx],
        )
        self._module_state[mod_name][layer_idx] = active_state
        if should_refresh:
            self._rotate_module(mod_name, layer_idx, feature)

    def reset(self):
        self._prev_module_feat.clear()
        self._current_module_feat.clear()
        self._prev_module_norm.clear()


def _global_mean(value_sum: torch.Tensor, count: int) -> float:
    pair = torch.tensor(
        [float(value_sum.item()), float(count)],
        device=value_sum.device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float((pair[0] / pair[1].clamp_min(1.0)).item())


class SeaCachePolicy:
    """Official SeaCache Wan2.1 policy with independent CFG trajectories."""

    def __init__(self, args, enabled=True):
        self.ret_steps, self.cutoff_steps = seacache_boundaries(args)
        self.threshold = float(args.seacache_thresh)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        self.previous_input = [None, None]
        self.accumulated_distance = [0.0, 0.0]

    def decide(
        self,
        runtime_step_idx: int,
        x,
        e0,
        grid_sizes,
        first_block,
        scheduler,
        **_ignored,
    ) -> bool:
        if not self.enabled:
            return False
        slot = runtime_step_idx % 2
        with amp.autocast(dtype=torch.float32):
            modulation = (first_block.modulation + e0).chunk(6, dim=1)
            current = first_block.norm1(x).float() * (1 + modulation[1]) + modulation[0]
        try:
            from xfuser.core.distributed import get_sequence_parallel_world_size, get_sp_group
            if get_sequence_parallel_world_size() > 1:
                current = get_sp_group().all_gather(current, dim=1)
        except (ImportError, RuntimeError):
            pass
        previous = self.previous_input[slot]
        force_compute = (
            runtime_step_idx < self.ret_steps
            or runtime_step_idx >= self.cutoff_steps
            or previous is None
        )
        if force_compute:
            should_skip = False
            self.accumulated_distance[slot] = 0.0
        else:
            real_token_count = int(torch.prod(grid_sizes[0]).item())
            shape = (
                current.shape[0],
                int(grid_sizes[0, 0]),
                int(grid_sizes[0, 1]),
                int(grid_sizes[0, 2]),
                current.shape[-1],
            )
            filtered = apply_sea_with_scheduler(
                current[:, :real_token_count].reshape(shape),
                scheduler,
                runtime_step_idx // 2,
            ).reshape(shape[0], -1, shape[-1])
            if real_token_count < current.shape[1]:
                current = torch.cat(
                    [filtered, current[:, real_token_count:]], dim=1
                )
            else:
                current = filtered
            self.accumulated_distance[slot] += distributed_relative_l1(
                current, previous
            )
            should_skip = self.accumulated_distance[slot] < self.threshold
            if not should_skip:
                self.accumulated_distance[slot] = 0.0
        self.previous_input[slot] = current.detach().clone()
        return should_skip


def _to_bool_list(x: torch.Tensor):
    return x.to(dtype=torch.bool).cpu().tolist()

def _compute_quantile_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    valid = values[~torch.isnan(values)]
    if valid.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    threshold = torch.quantile(valid, q)
    return values < threshold

def _books_from_scores(
    module_scores: Dict[str, torch.Tensor],
    nonskip_rate: float,
    module_thres: Dict[str, float],
) -> Dict[str, List[List[bool]]]:
    if not 0.0 <= nonskip_rate <= 1.0:
        raise ValueError(f"nonskip_rate must be in [0, 1], got {nonskip_rate}.")
    for mod_name in WAN_CACHE_MODULES:
        threshold = module_thres[mod_name]
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {mod_name!r} must be in [0, 1], got {threshold}."
            )

    first_scores = next(iter(module_scores.values()))
    num_steps = int(first_scores.shape[0])
    num_layers = int(first_scores.shape[1])
    num_nonskip = max(1, int(nonskip_rate * num_steps))

    module_cache_book: Dict[str, torch.Tensor] = {}
    for mod_name in WAN_CACHE_MODULES:
        scores = module_scores[mod_name]
        if tuple(scores.shape) != (num_steps, num_layers):
            raise ValueError(
                f"Inconsistent score shape for {mod_name}: {tuple(scores.shape)}."
            )
        book = torch.zeros_like(scores, dtype=torch.bool)
        if module_thres[mod_name] > 0.0 and num_steps > 2:
            flat = scores[1:-1].reshape(-1)
            mod_mask = _compute_quantile_mask(flat, module_thres[mod_name])
            book[1:-1] = mod_mask.reshape(num_steps - 2, num_layers)
        book[:num_nonskip, :] = False
        book[-1, :] = False
        module_cache_book[mod_name] = book

    return {k: _to_bool_list(v) for k, v in module_cache_book.items()}


def _cache_book_config(args, model) -> Dict[str, object]:
    ret_steps, cutoff_steps = seacache_boundaries(args)
    return {
        "policy": POLICY_VARIANT,
        "step_policy": STEP_POLICY,
        "task": args.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "steps": args.sample_steps,
        "sample_solver": args.sample_solver,
        "sample_shift": args.sample_shift,
        "nonskip": args.nonskip_rate,
        "thresholds": {
            "self_attn": args.self_attn_thres,
            "cross_attn": args.cross_attn_thres,
            "ffn": args.ffn_thres,
        },
        "num_layers": len(model.blocks),
        "model_dim": model.dim,
        "seacache_thresh": args.seacache_thresh,
        "use_ret_steps": args.use_ret_steps,
        "ret_steps": ret_steps,
        "cutoff_steps": cutoff_steps,
        "scheduler_mode": "flow",
        "fft_dims": [-2, -3, -4],
        "power_exp": 3.0,
        "norm_mode": "mean",
        "source_commit": SOURCE_COMMIT,
    }


def _save_cache_books(
    file_path: str,
    module_cache_book: Dict[str, List[List[bool]]],
    config: Dict[str, object],
    observer_masks,
):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "cache_scope": CACHE_SCOPE,
        "policy": POLICY_VARIANT,
        "rate_method": RATE_METHOD,
        "config": config,
        "finegrained_cache": {"module_cache_book": module_cache_book},
        "step_policy_diagnostics": {
            "observer_hit_rate": {
                "conditional": sum(observer_masks[0]) / max(len(observer_masks[0]), 1),
                "unconditional": sum(observer_masks[1]) / max(len(observer_masks[1]), 1),
            }
        },
    }
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload


def _load_cache_books(
    file_path: str,
    args,
    model,
):
    num_layers = len(model.blocks)
    payload = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if dist.is_initialized():
        shared = [payload]
        dist.broadcast_object_list(shared, src=0)
        payload = shared[0]

    if payload.get("cache_scope") != CACHE_SCOPE:
        raise ValueError(
            "Legacy or incompatible cache book: expected "
            f"cache_scope={CACHE_SCOPE!r}. Re-run calibration with this script."
        )
    if payload.get("policy") != POLICY_VARIANT:
        raise ValueError(
            f"Expected policy {POLICY_VARIANT!r}, found {payload.get('policy')!r}."
        )
    if payload.get("rate_method") != RATE_METHOD:
        raise ValueError(
            f"Expected rate method {RATE_METHOD!r}, "
            f"found {payload.get('rate_method')!r}."
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Cache book is missing its calibration config.")
    expected_config = {
        "task": args.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "steps": args.sample_steps,
        "num_layers": num_layers,
        "model_dim": model.dim,
        "sample_solver": args.sample_solver,
        "sample_shift": args.sample_shift,
        "nonskip": args.nonskip_rate,
        "thresholds": {
            "self_attn": args.self_attn_thres,
            "cross_attn": args.cross_attn_thres,
            "ffn": args.ffn_thres,
        },
        "seacache_thresh": args.seacache_thresh,
        "use_ret_steps": args.use_ret_steps,
        "ret_steps": seacache_boundaries(args)[0],
        "cutoff_steps": seacache_boundaries(args)[1],
        "scheduler_mode": "flow",
        "fft_dims": [-2, -3, -4],
        "power_exp": 3.0,
        "norm_mode": "mean",
        "source_commit": SOURCE_COMMIT,
    }
    for key, expected_value in expected_config.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"Cache-book config mismatch for {key!r}: "
                f"expected {expected_value!r}, found {config.get(key)!r}."
            )

    module_cache_book = payload.get("finegrained_cache", {}).get(
        "module_cache_book", {}
    )
    if not isinstance(module_cache_book, dict) or not module_cache_book:
        raise ValueError(f"Invalid cache book format: {file_path}")

    for key in WAN_CACHE_MODULES:
        if key not in module_cache_book:
            raise ValueError(f"Missing module cache key '{key}' in {file_path}")
        book = module_cache_book[key]
        if not isinstance(book, list) or len(book) != args.sample_steps:
            raise ValueError(
                f"Invalid {key} cache-book step count: "
                f"expected {args.sample_steps}, got "
                f"{len(book) if isinstance(book, list) else 'non-list'}."
            )
        for step_idx, row in enumerate(book):
            if not isinstance(row, list) or len(row) != num_layers:
                raise ValueError(
                    f"Invalid {key} cache-book layer count at step {step_idx}: "
                    f"expected {num_layers}, got "
                    f"{len(row) if isinstance(row, list) else 'non-list'}."
                )
            if any(type(value) is not bool for value in row):
                raise ValueError(
                    f"{key} cache book at step {step_idx} must contain "
                    "JSON booleans only."
                )

    return module_cache_book


def _build_default_cache_book_filename(args) -> str:
    fmt = lambda value: format(float(value), "g")
    size = args.size.replace("*", "x")
    return (
        f"cache_book_hybrid_seacache_{args.task}_{size}_f{args.frame_num}"
        f"_steps{args.sample_steps}_ns{fmt(args.nonskip_rate)}"
        f"_seath{fmt(args.seacache_thresh)}_ret{int(args.use_ret_steps)}"
        f"_sattnth{fmt(args.self_attn_thres)}"
        f"_cattnth{fmt(args.cross_attn_thres)}"
        f"_ffnth{fmt(args.ffn_thres)}.json"
    )


def _prepare_common_inputs(self, x, t, context, seq_len, clip_fea=None, y=None):
    if self.model_type in ("i2v", "flf2v"):
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )
    try:
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )
        sp_size = get_sequence_parallel_world_size()
        if sp_size > 1:
            x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]
    except (ImportError, RuntimeError):
        pass
    return x, e, kwargs, grid_sizes


def _wan_block_forward_with_cache(
    block,
    x,
    kwargs,
    module_plan: Dict[str, bool],
    module_cache: Dict[str, Optional[torch.Tensor]],
):
    with amp.autocast(dtype=torch.float32):
        e = (block.modulation + kwargs["e"]).chunk(6, dim=1)

    if module_plan["self_attn"] and module_cache["self_attn"] is not None:
        y_self = module_cache["self_attn"]
    else:
        y_self = block.self_attn(
            block.norm1(x).float() * (1 + e[1]) + e[0],
            kwargs["seq_lens"],
            kwargs["grid_sizes"],
            kwargs["freqs"],
        )
    with amp.autocast(dtype=torch.float32):
        x = x + y_self * e[2]

    if module_plan["cross_attn"] and module_cache["cross_attn"] is not None:
        y_cross = module_cache["cross_attn"]
    else:
        y_cross = block.cross_attn(block.norm3(x), kwargs["context"], kwargs["context_lens"])
    x = x + y_cross

    if module_plan["ffn"] and module_cache["ffn"] is not None:
        y_ffn = module_cache["ffn"]
    else:
        y_ffn = block.ffn(block.norm2(x).float() * (1 + e[4]) + e[3])
    with amp.autocast(dtype=torch.float32):
        x = x + y_ffn * e[5]

    # VACE base block adds hint skip after parent forward.
    if hasattr(block, "block_id") and block.block_id is not None and "hints" in kwargs:
        x = x + kwargs["hints"][block.block_id] * kwargs.get("context_scale", 1.0)

    return x, {"self_attn": y_self, "cross_attn": y_cross, "ffn": y_ffn}


def _runtime_cache_target(self):
    mode = self.runtime_cache_device
    if mode != "auto":
        return mode
    free_bytes, _ = torch.cuda.mem_get_info(self.patch_embedding.weight.device)
    reserve = int(self.runtime_cache_gpu_reserve_gib * 1024**3)
    return "gpu" if free_bytes >= reserve else "cpu"


def _store_runtime_tensor(self, tensor):
    tensor = tensor.detach()
    if _runtime_cache_target(self) == "gpu":
        return tensor
    cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
    cpu.copy_(tensor, non_blocking=False)
    return cpu


def _load_runtime_tensor(tensor, device):
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


def _has_future_layer_hit(self, mod_name, policy_step_idx, layer_idx):
    if not self.finegrained_use:
        return False
    book = self.module_cache_book[mod_name]
    return any(
        bool(book[step][layer_idx])
        for step in range(policy_step_idx + 1, len(book))
    )


def _maybe_reset_runtime(self):
    if self.finegrained_timestep_idx >= self.finegrained_num_steps:
        self.finegrained_timestep_idx = 0
        self.step_residual_cache = [None, None]
        self.previous_step_was_cached = [False, False]
        if self.module_feature_cache:
            for mod_name in WAN_CACHE_MODULES:
                self.module_feature_cache[mod_name] = [
                    [None] * self.finegrained_num_layers for _ in range(2)
                ]
        if self.step_policy is not None:
            self.step_policy.reset()


def _finegrained_core_forward(self, x, e, kwargs, grid_sizes, hints_factory=None):
    runtime_step_idx = self.finegrained_timestep_idx
    cfg_slot = runtime_step_idx % 2
    policy_step_idx = runtime_step_idx // 2
    analyzer = self.feature_change_analyzer
    collect_conditional = analyzer is not None and cfg_slot == 0
    original_x = x

    if self.forced_step_masks is not None:
        would_skip = bool(self.forced_step_masks[cfg_slot][policy_step_idx])
    elif self.step_policy is not None:
        would_skip = self.step_policy.decide(
            runtime_step_idx,
            x=x,
            e0=kwargs["e"],
            grid_sizes=grid_sizes,
            first_block=self.blocks[0],
            scheduler=self.seacache_scheduler,
        )
    else:
        would_skip = False
    actual_step_hit = bool(
        self.execute_step_cache
        and would_skip
        and self.step_residual_cache[cfg_slot] is not None
    )

    if actual_step_hit:
        residual = _load_runtime_tensor(
            self.step_residual_cache[cfg_slot], x.device
        )
        x = x + residual
        self.step_hits[cfg_slot] += 1
    else:
        if hints_factory is not None:
            hints, context_scale = hints_factory(x, kwargs)
            kwargs["hints"] = hints
            kwargs["context_scale"] = context_scale

        for layer_idx, block in enumerate(self.blocks):
            allow_layer_cache = bool(
                self.finegrained_use
                and not self.finegrained_calib_running
                and not self.previous_step_was_cached[cfg_slot]
            )
            module_plan = {
                mod_name: bool(
                    allow_layer_cache
                    and policy_step_idx < len(self.module_cache_book[mod_name])
                    and self.module_cache_book[mod_name][policy_step_idx][layer_idx]
                )
                for mod_name in WAN_CACHE_MODULES
            }
            layer_cache = {}
            for mod_name in WAN_CACHE_MODULES:
                cached = None
                if allow_layer_cache and module_plan[mod_name]:
                    cached = self.module_feature_cache[mod_name][cfg_slot][layer_idx]
                    cached = _load_runtime_tensor(cached, x.device)
                layer_cache[mod_name] = cached
                if cached is None:
                    module_plan[mod_name] = False
                else:
                    self.layer_hits[cfg_slot] += 1

            x, outputs = _wan_block_forward_with_cache(
                block, x, kwargs, module_plan, layer_cache
            )

            for mod_name in WAN_CACHE_MODULES:
                if collect_conditional:
                    analyzer.update_module(
                        policy_step_idx, layer_idx, mod_name, outputs[mod_name]
                    )
                if not self.finegrained_calib_running:
                    if _has_future_layer_hit(
                        self, mod_name, policy_step_idx, layer_idx
                    ):
                        self.module_feature_cache[mod_name][cfg_slot][layer_idx] = (
                            _store_runtime_tensor(self, outputs[mod_name])
                        )
                    else:
                        self.module_feature_cache[mod_name][cfg_slot][layer_idx] = None

        residual = x - original_x
        if not self.finegrained_calib_running:
            self.step_residual_cache[cfg_slot] = _store_runtime_tensor(self, residual)
        self.computed_steps[cfg_slot] += 1

    if self.observer_masks is not None:
        self.observer_masks[cfg_slot][policy_step_idx] = would_skip
    self.previous_step_was_cached[cfg_slot] = actual_step_hit

    x = self.head(x, e)
    try:
        from xfuser.core.distributed import get_sequence_parallel_world_size, get_sp_group
        if get_sequence_parallel_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    except (ImportError, RuntimeError):
        pass
    x = self.unpatchify(x, grid_sizes)

    self.finegrained_timestep_idx += 1
    _maybe_reset_runtime(self)

    return [u.float() for u in x]


def finegrained_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    x, e, kwargs, grid_sizes = _prepare_common_inputs(self, x, t, context, seq_len, clip_fea, y)
    return _finegrained_core_forward(self, x, e, kwargs, grid_sizes)


def finegrained_vace_forward(
    self,
    x,
    t,
    vace_context,
    context,
    seq_len,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    x, e, kwargs, grid_sizes = _prepare_common_inputs(
        self,
        x,
        t,
        context,
        seq_len,
        None,
        None,
    )
    def hints_factory(current_x, current_kwargs):
        hints = self.forward_vace(
            current_x, vace_context, seq_len, current_kwargs
        )
        return hints, vace_context_scale

    return _finegrained_core_forward(
        self, x, e, kwargs, grid_sizes, hints_factory=hints_factory
    )


def _init_finegrained_runtime(
    model,
    num_steps: int,
    module_cache_book: Optional[Dict[str, List[List[bool]]]] = None,
    use_finegrained: bool = False,
    analyzer: Optional[FeatureChangeAnalyzer] = None,
    step_policy=None,
    forced_step_masks=None,
    observer_masks=None,
    runtime_cache_device="gpu",
    runtime_cache_gpu_reserve_gib=2.0,
):
    model.finegrained_num_steps = num_steps
    model.finegrained_num_layers = len(model.blocks)
    model.finegrained_use = use_finegrained
    model.finegrained_calib_running = analyzer is not None
    model.finegrained_timestep_idx = 0
    model.step_policy = step_policy
    model.forced_step_masks = forced_step_masks
    model.observer_masks = observer_masks
    model.seacache_scheduler = getattr(model, "seacache_scheduler", None)
    model.execute_step_cache = (
        step_policy is not None
        and forced_step_masks is None
        and analyzer is None
    )
    model.step_residual_cache = [None, None]
    model.previous_step_was_cached = [False, False]
    model.step_hits = [0, 0]
    model.computed_steps = [0, 0]
    model.layer_hits = [0, 0]
    model.runtime_cache_device = runtime_cache_device
    model.runtime_cache_gpu_reserve_gib = runtime_cache_gpu_reserve_gib

    # Runtime reuse needs independent conditional/unconditional tensor slots.
    # Calibration streams conditional features directly into the analyzer and
    # therefore does not allocate these redundant module-output caches.
    if analyzer is None:
        model.module_feature_cache = {
            mod_name: [[None] * model.finegrained_num_layers for _ in range(2)]
            for mod_name in WAN_CACHE_MODULES
        }
    else:
        model.module_feature_cache = {}

    policy_steps = num_steps // 2
    model.module_cache_book = module_cache_book if module_cache_book is not None else {
        mod_name: [[False] * model.finegrained_num_layers for _ in range(policy_steps)]
        for mod_name in WAN_CACHE_MODULES
    }
    model.feature_change_analyzer = analyzer


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"
    if "flf2v" in args.task or "vace" in args.task:
        raise ValueError(
            "The official Wan2.1 SeaCache implementation supports T2V/T2I and I2V only"
        )

    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, "
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )
    if args.sample_steps < 3:
        raise ValueError("sample_steps must be at least 3 for three-point rates.")
    if not 0.0 <= args.nonskip_rate <= 1.0:
        raise ValueError("nonskip_rate must be in [0, 1].")
    for name in (
        "self_attn_thres",
        "cross_attn_thres",
        "ffn_thres",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    if args.seacache_thresh < 0:
        raise ValueError("seacache_thresh must be non-negative")
    if args.runtime_cache_gpu_reserve_gib < 0:
        raise ValueError("runtime_cache_gpu_reserve_gib must be non-negative")



def _parse_args(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Standalone Wan2.1 SeaCache + Finegrained Cache hybrid sampler"
    )
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--save_file", type=str, default=None)

    parser.add_argument("--src_video", type=str, default=None)
    parser.add_argument("--src_mask", type=str, default=None)
    parser.add_argument("--src_ref_images", type=str, default=None)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="zh", choices=["zh", "en"])

    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--first_frame", type=str, default=None)
    parser.add_argument("--last_frame", type=str, default=None)

    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    # Finegrained Cache options
    parser.add_argument("--use_finegrained_cache", action="store_true", default=False)
    parser.add_argument("--finegrained_calibration", action="store_true", default=False)
    parser.add_argument("--cache_book_path", type=str, default="./cache_books")
    parser.add_argument("--cache_book_file", type=str, default=None)
    parser.add_argument("--nonskip_rate", type=float, default=0.1)
    # fast (default): self_attn=0.20, cross_attn=0.10, ffn=0.01;
    # balanced: self_attn=0.10, cross_attn=0.10, ffn=0.00;
    # slow: self_attn=0.10, cross_attn=0.00, ffn=0.00.
    parser.add_argument("--self_attn_thres", type=float, default=0.20)
    parser.add_argument("--cross_attn_thres", type=float, default=0.10)
    parser.add_argument("--ffn_thres", type=float, default=0.01)
    parser.add_argument("--disable_step_cache", action="store_true", default=False)
    parser.add_argument("--disable_progress_bar", action="store_true", default=False)
    parser.add_argument(
        "--calibration_feature_device",
        choices=["cpu", "gpu"],
        default="cpu",
    )
    parser.add_argument(
        "--runtime_cache_device",
        choices=["gpu", "cpu", "auto"],
        default="gpu",
    )
    parser.add_argument(
        "--runtime_cache_gpu_reserve_gib", type=float, default=2.0
    )
    parser.add_argument("--seacache_thresh", type=float, default=0.2)
    parser.add_argument("--use_ret_steps", action="store_true", default=False)

    args = parser.parse_args(cli_args)
    _validate_args(args)
    return args



def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)



def _setup_distributed(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_id = local_rank

    _init_logging(rank)

    if args.disable_progress_bar:
        os.environ["TQDM_DISABLE"] = "1"
        import importlib
        from functools import partial
        from tqdm import tqdm as tqdm_base
        for module_name in (
            "wan.text2video",
            "wan.image2video",
            "wan.first_last_frame2video",
            "wan.vace",
        ):
            importlib.import_module(module_name).tqdm = partial(
                tqdm_base, disable=True
            )

    if args.offload_model is None:
        args.offload_model = False

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp/dit_fsdp require distributed run"
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel requires distributed run"

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    return rank, world_size, device_id



def _make_prompt_expander(args, rank):
    if not args.use_prompt_extend:
        return None

    if args.prompt_extend_method == "dashscope":
        return DashScopePromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task or "flf2v" in args.task,
        )
    if args.prompt_extend_method == "local_qwen":
        return QwenPromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task,
            device=rank,
        )
    raise NotImplementedError(args.prompt_extend_method)



def _begin_memory_tracking():
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return device


def _log_peak_memory(description: str, device):
    if device is None:
        return 0.0
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    logging.info(f"Peak GPU memory for {description}: {peak_gib:.2f} GiB")
    return peak_gib


def _release_calibration_state(model, analyzer: FeatureChangeAnalyzer):
    analyzer.reset()
    if model.feature_change_analyzer is analyzer:
        model.feature_change_analyzer = None
    model.finegrained_calib_running = False
    model.module_feature_cache = {}
    model.forced_step_masks = None
    model.observer_masks = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect_calibration_stage(
    run_once,
    model,
    args,
    policy_steps: int,
    runtime_steps: int,
    num_layers: int,
    active_modules: Tuple[str, ...],
    description: str,
    module_cache_state: Optional[Dict[str, List[List[bool]]]] = None,
    step_policy=None,
    forced_step_masks=None,
):
    """Collect all active layer-module scores in one complete generation."""
    correction_mode = module_cache_state is not None
    analyzer = FeatureChangeAnalyzer(
        policy_steps,
        num_layers,
        correction_mode=correction_mode,
        module_cache_state=module_cache_state,
        active_modules=active_modules,
        feature_device=args.calibration_feature_device,
    )
    observer_masks = [[False] * policy_steps for _ in range(2)]
    _init_finegrained_runtime(
        model,
        num_steps=runtime_steps,
        use_finegrained=False,
        analyzer=analyzer,
        step_policy=step_policy,
        forced_step_masks=forced_step_masks,
        observer_masks=observer_masks,
        runtime_cache_device=args.runtime_cache_device,
        runtime_cache_gpu_reserve_gib=args.runtime_cache_gpu_reserve_gib,
    )
    memory_device = _begin_memory_tracking()

    try:
        with torch.inference_mode():
            calibration_output = run_once()
        del calibration_output
    except Exception:
        _release_calibration_state(model, analyzer)
        raise

    _log_peak_memory(description, memory_device)
    module_scores = {
        key: value.clone() for key, value in analyzer.module_scores.items()
    }
    _release_calibration_state(model, analyzer)
    del analyzer
    return module_scores, observer_masks


def _effective_correction_books(module_book, conditional_step_mask):
    effective = {
        key: [[bool(x) for x in row] for row in value]
        for key, value in module_book.items()
    }
    for step_idx, cached in enumerate(conditional_step_mask):
        if not cached:
            continue
        for key in WAN_CACHE_MODULES:
            effective[key][step_idx] = [True] * len(effective[key][step_idx])
            if step_idx + 1 < len(conditional_step_mask):
                effective[key][step_idx + 1] = [False] * len(
                    effective[key][step_idx + 1]
                )
    return effective


def _calibrate_finegrained(run_once, model, args):
    policy_steps = args.sample_steps
    runtime_steps = args.sample_steps * 2
    num_layers = len(model.blocks)

    module_thres = {
        "self_attn": args.self_attn_thres,
        "cross_attn": args.cross_attn_thres,
        "ffn": args.ffn_thres,
    }
    active_modules = tuple(
        mod_name
        for mod_name, threshold in module_thres.items()
        if threshold > 0.0
    )

    # Phase 1: one full run collects all active layer-module statistics.
    raw_policy = None if args.disable_step_cache else SeaCachePolicy(args)
    raw_module_scores, observer_masks = _collect_calibration_stage(
        run_once,
        model,
        args,
        policy_steps,
        runtime_steps,
        num_layers,
        active_modules,
        "raw SeaCache observer and layer calibration",
        step_policy=raw_policy,
    )
    provisional_module_cache_book = _books_from_scores(
        raw_module_scores,
        nonskip_rate=args.nonskip_rate,
        module_thres=module_thres,
    )
    del raw_module_scores
    correction_books = _effective_correction_books(
        provisional_module_cache_book, observer_masks[0]
    )

    # Phase 2: one full correction run follows the provisional layer paths.
    corrected_module_scores, _ = _collect_calibration_stage(
        run_once,
        model,
        args,
        policy_steps,
        runtime_steps,
        num_layers,
        active_modules,
        "layer cache correction",
        correction_books,
        forced_step_masks=observer_masks,
    )
    final_module_cache_book = _books_from_scores(
        corrected_module_scores,
        nonskip_rate=args.nonskip_rate,
        module_thres=module_thres,
    )
    del corrected_module_scores

    cache_file = args.cache_book_file or _build_default_cache_book_filename(args)
    cache_path = os.path.join(args.cache_book_path, cache_file)
    _save_cache_books(
        cache_path,
        final_module_cache_book,
        _cache_book_config(args, model),
        observer_masks,
    )
    logging.info(f"Finegrained Cache layer cache book saved to: {cache_path}")
    return final_module_cache_book


def _prepare_finegrained_books(model, args):
    module_cache_book = None

    if args.use_finegrained_cache and not args.finegrained_calibration:
        cache_file = args.cache_book_file or _build_default_cache_book_filename(args)
        cache_path = os.path.join(args.cache_book_path, cache_file)
        module_cache_book = _load_cache_books(
            cache_path,
            args,
            model,
        )
        logging.info(f"Loaded Finegrained Cache layer cache book: {cache_path}")

    _configure_hybrid_runtime(model, args, module_cache_book)


def _configure_hybrid_runtime(model, args, module_cache_book):
    step_policy = None
    if not args.disable_step_cache:
        step_policy = SeaCachePolicy(args)
    _init_finegrained_runtime(
        model,
        num_steps=args.sample_steps * 2,
        module_cache_book=module_cache_book,
        use_finegrained=args.use_finegrained_cache and module_cache_book is not None,
        analyzer=None,
        step_policy=step_policy,
        runtime_cache_device=args.runtime_cache_device,
        runtime_cache_gpu_reserve_gib=args.runtime_cache_gpu_reserve_gib,
    )


def _log_runtime_cache_stats(model, args):
    layers = len(model.blocks)
    computed_module_slots = max(
        sum(model.computed_steps) * layers * len(WAN_CACHE_MODULES), 1
    )
    total_layer_hits = sum(model.layer_hits)
    total_step_hits = sum(model.step_hits)
    combined_saved_modules = total_layer_hits + total_step_hits * layers * len(WAN_CACHE_MODULES)
    combined_slots = max(args.sample_steps * 2 * layers * len(WAN_CACHE_MODULES), 1)
    logging.info(
        "[Cache] step hits cond=%d/%d, uncond=%d/%d; "
        "layer hits on computed calls cond=%d, uncond=%d; "
        "computed-step layer skip=%.2f%%; "
        "combined effective module skip=%.2f%%",
        model.step_hits[0],
        args.sample_steps,
        model.step_hits[1],
        args.sample_steps,
        model.layer_hits[0],
        model.layer_hits[1],
        100.0 * total_layer_hits / computed_module_slots,
        100.0 * combined_saved_modules / combined_slots,
    )


def _timed_generate(task, callable_):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    start = time.perf_counter()
    output = callable_()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**3
        if torch.cuda.is_available() else 0.0
    )
    logging.info("[Sampling] %s: %.4fs, peak allocated %.2f GiB", task, elapsed, peak)
    return output


def generate(args):
    video = None
    rank, world_size, device = _setup_distributed(args)

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    prompt_expander = _make_prompt_expander(args, rank)

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        pipeline.model.forward = finegrained_forward.__get__(
            pipeline.model, pipeline.model.__class__
        )

        def run_once():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            t0 = time.perf_counter()

            with sea_scheduler_bridge(pipeline.model):
                out = pipeline.generate(
                    args.prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if rank == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                logging.info(f"[Sampling] {args.task}: {dt:.4f}s, peak allocated {peak:.2f} GiB")

            return out

        if args.finegrained_calibration:
            module_cache_book = _calibrate_finegrained(
                run_once, pipeline.model, args
            )

            if args.use_finegrained_cache:
                _configure_hybrid_runtime(pipeline.model, args, module_cache_book)
                video = run_once()
        else:
            _prepare_finegrained_books(pipeline.model, args)
            video = run_once()

    elif "i2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

        img = Image.open(args.image).convert("RGB")

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed,
                )
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        pipeline.model.forward = finegrained_forward.__get__(
            pipeline.model, pipeline.model.__class__
        )

        def run_once():
            with sea_scheduler_bridge(pipeline.model):
                return _timed_generate(args.task, lambda: pipeline.generate(
                    args.prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                ))

        if args.finegrained_calibration:
            module_cache_book = _calibrate_finegrained(
                run_once, pipeline.model, args
            )

            if args.use_finegrained_cache:
                _configure_hybrid_runtime(pipeline.model, args, module_cache_book)
                video = run_once()
        else:
            _prepare_finegrained_books(pipeline.model, args)
            video = run_once()

    elif "flf2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.first_frame is None or args.last_frame is None:
            args.first_frame = EXAMPLE_PROMPT[args.task]["first_frame"]
            args.last_frame = EXAMPLE_PROMPT[args.task]["last_frame"]

        first_frame = Image.open(args.first_frame).convert("RGB")
        last_frame = Image.open(args.last_frame).convert("RGB")

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=[first_frame, last_frame],
                    seed=args.base_seed,
                )
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanFLF2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # FLF2V shares WanModel forward signature, so Finegrained Cache is directly reusable.
        enable_finegrained = args.use_finegrained_cache or args.finegrained_calibration

        if enable_finegrained:
            pipeline.model.forward = finegrained_forward.__get__(pipeline.model, pipeline.model.__class__)

        def run_once():
            return pipeline.generate(
                args.prompt,
                first_frame,
                last_frame,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        if args.finegrained_calibration:
            module_cache_book = _calibrate_finegrained(run_once, pipeline.model, args)

            if args.use_finegrained_cache:
                _init_finegrained_runtime(
                    pipeline.model,
                    num_steps=args.sample_steps * 2,
                    module_cache_book=module_cache_book,
                    use_finegrained=args.use_finegrained_cache,
                    analyzer=None,
                )
                video = run_once()
        elif args.use_finegrained_cache:
            _prepare_finegrained_books(pipeline.model, args)
            video = run_once()
        else:
            video = run_once()

    elif "vace" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            args.src_video = EXAMPLE_PROMPT[args.task].get("src_video", None)
            args.src_mask = EXAMPLE_PROMPT[args.task].get("src_mask", None)
            args.src_ref_images = EXAMPLE_PROMPT[args.task].get("src_ref_images", None)

        pipeline = wan.WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        pipeline.model.forward = finegrained_vace_forward.__get__(
            pipeline.model, pipeline.model.__class__
        )

        src_video, src_mask, src_ref_images = pipeline.prepare_source(
            [args.src_video],
            [args.src_mask],
            [None if args.src_ref_images is None else args.src_ref_images.split(",")],
            args.frame_num,
            SIZE_CONFIGS[args.size],
            device,
        )

        def run_once():
            return pipeline.generate(
                args.prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        if args.finegrained_calibration:
            module_cache_book = _calibrate_finegrained(
                run_once, pipeline.model, args
            )

            if args.use_finegrained_cache:
                _configure_hybrid_runtime(pipeline.model, args, module_cache_book)
                video = run_once()
        else:
            _prepare_finegrained_books(pipeline.model, args)
            video = run_once()

    else:
        raise ValueError(f"Unknown task type: {args.task}")

    if video is not None:
        _log_runtime_cache_stats(pipeline.model, args)

    if rank == 0 and video is not None:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".png" if "t2i" in args.task else ".mp4"
            cache_params = "_hybrid" if args.use_finegrained_cache else "_full"
            args.save_file = (
                f"finegrained_{args.task}{cache_params}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_"
                f"{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"
            )
        save_path = os.path.abspath(args.save_file)
        if "t2i" in args.task:
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=save_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            cache_video(
                tensor=video[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        logging.info(f"Output will be saved to: {save_path}")

    logging.info("Finished.")


if __name__ == "__main__":
    generate(_parse_args())
