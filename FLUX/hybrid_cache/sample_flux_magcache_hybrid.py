#!/usr/bin/env python3
"""Standalone FLUX Finegrained Cache + MagCache sampler.

MagCache policy adapted from open-source/MagCache/MagCache4FLUX
(Apache-2.0, upstream commit df81cb1).  The implementation intentionally keeps
the official FLUX magnitude-ratio accumulator, K limit, retention prefix and
the mapped 28-step forced-refresh point.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from PIL import Image
from tqdm.auto import tqdm


logger = logging.get_logger(__name__)
METHOD = "magcache"
RATE_METHOD = "three_point_l1"
RATE_CHUNK_SIZE = 1_048_576
DOUBLE_RATE_KEYS = ("attn", "context_attn", "ff", "context_ff")
DOUBLE_CACHE_KEYS = (*DOUBLE_RATE_KEYS, "ip_attn")
SINGLE_KEYS = ("attn", "mlp")
MAG_RATIOS_28 = np.asarray(
    [
        1.0, 1.21094, 1.11719, 1.07812, 1.0625, 1.03906, 1.03125,
        1.03906, 1.02344, 1.03125, 1.02344, 0.98047, 1.01562,
        1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 0.99609,
        0.99609, 0.98047, 0.98828, 0.96484, 0.95703, 0.93359,
        0.89062,
    ],
    dtype=np.float64,
)
DEFAULT_PROMPTS = [
    "a photo of a broccoli",
    "A snowy mountain village at dusk, glowing windows and smoke rising.",
    "A golden retriever puppy jumping through autumn leaves.",
    "A surreal underwater city with glowing jellyfish and crystal towers.",
    "A group of astronauts planting a flag on Mars, red rocky landscape.",
    "A vintage sports car speeding down a coastal highway at sunset.",
    (
        "A stylish woman walks down a Tokyo street filled with warm glowing "
        "neon and animated city signage."
    ),
]
DEFAULT_CALIBRATION_PROMPT = (
    "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
)


def compute_l1_distance(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    chunk_size: int = RATE_CHUNK_SIZE,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    x_start = x_start.detach().reshape(-1)
    x_end = x_end.detach().reshape(-1)
    if x_start.numel() != x_end.numel():
        raise ValueError("Finegrained rate features must have equal sizes")
    total = torch.zeros((), device=x_start.device, dtype=torch.float32)
    for start in range(0, x_start.numel(), chunk_size):
        end = min(start + chunk_size, x_start.numel())
        diff = x_end[start:end] - x_start[start:end] + 1e-8
        total += diff.abs().sum(dtype=torch.float32)
    return total


def compute_rate(
    x_prev: torch.Tensor,
    x: torch.Tensor,
    x_post: torch.Tensor,
    diff_prev_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if diff_prev_norm is None:
        diff_prev_norm = compute_l1_distance(x_prev, x)
    return compute_l1_distance(x_prev, x_post) / diff_prev_norm.clamp_min(1e-8)


def nearest_interp(values: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if target_length == 1:
        return np.asarray([values[-1]], dtype=np.float64)
    scale = (len(values) - 1) / (target_length - 1)
    indices = np.round(np.arange(target_length) * scale).astype(int)
    return np.asarray(values[indices], dtype=np.float64)


class MagCachePolicy:
    def __init__(
        self,
        ratios: List[float],
        num_steps: int,
        threshold: float,
        max_consecutive: int,
        retention_ratio: float,
        enabled: bool = True,
    ):
        if len(ratios) != num_steps:
            raise ValueError("MagCache ratio count must equal num_steps")
        self.ratios = np.asarray(ratios, dtype=np.float64)
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.max_consecutive = int(max_consecutive)
        self.retention_ratio = float(retention_ratio)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        self.accumulated_ratio = 1.0
        self.accumulated_error = 0.0
        self.accumulated_steps = 0

    def decide(self, step_idx: int) -> bool:
        if not self.enabled:
            return False
        retention_steps = int(self.retention_ratio * self.num_steps + 0.5)
        if step_idx < retention_steps:
            return False
        current_scale = float(self.ratios[step_idx])
        self.accumulated_ratio *= current_scale
        self.accumulated_steps += 1
        self.accumulated_error += abs(1.0 - self.accumulated_ratio)
        mapped_idx = int(np.round(step_idx * ((28 - 1) / (self.num_steps - 1))))
        should_skip = (
            self.accumulated_error <= self.threshold
            and self.accumulated_steps <= self.max_consecutive
            and mapped_idx != 11
        )
        if not should_skip:
            self.accumulated_ratio = 1.0
            self.accumulated_error = 0.0
            self.accumulated_steps = 0
        return should_skip


def derive_magcache_book(
    ratios: List[float],
    num_steps: int,
    threshold: float,
    max_consecutive: int,
    retention_ratio: float,
) -> List[bool]:
    policy = MagCachePolicy(
        ratios,
        num_steps,
        threshold,
        max_consecutive,
        retention_ratio,
    )
    return [policy.decide(step_idx) for step_idx in range(num_steps)]


def finegrained_double_forward(
    block,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    attn_cache=None,
    context_attn_cache=None,
    ip_attn_cache=None,
    ff_cache=None,
    context_ff_cache=None,
):
    norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
        hidden_states, emb=temb
    )
    norm_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        block.norm1_context(encoder_hidden_states, emb=temb)
    )
    if attn_cache is not None and context_attn_cache is not None:
        attn_output = attn_cache
        context_attn_output = context_attn_cache
        ip_attn_output = ip_attn_cache
    else:
        attention_outputs = block.attn(
            hidden_states=norm_hidden,
            encoder_hidden_states=norm_context,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        attn_output, context_attn_output = attention_outputs[:2]
        ip_attn_output = (
            attention_outputs[2] if len(attention_outputs) == 3 else None
        )
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    if ff_cache is None:
        norm_hidden = block.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = block.ff(norm_hidden)
    else:
        ff_output = ff_cache
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output
    encoder_hidden_states = (
        encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    )
    if context_ff_cache is None:
        norm_context = block.norm2_context(encoder_hidden_states)
        norm_context = (
            norm_context * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        context_ff_output = block.ff_context(norm_context)
    else:
        context_ff_output = context_ff_cache
    encoder_hidden_states = (
        encoder_hidden_states
        + c_gate_mlp.unsqueeze(1) * context_ff_output
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    outputs = {
        "attn": attn_output,
        "context_attn": context_attn_output,
        "ip_attn": ip_attn_output,
        "ff": ff_output,
        "context_ff": context_ff_output,
    }
    return encoder_hidden_states, hidden_states, outputs


def finegrained_single_forward(
    block,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    attn_cache=None,
    mlp_cache=None,
):
    text_len = encoder_hidden_states.shape[1]
    joint_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = joint_states
    norm_states, gate = block.norm(joint_states, emb=temb)
    if mlp_cache is None:
        mlp_output = block.act_mlp(block.proj_mlp(norm_states))
    else:
        mlp_output = mlp_cache
    if attn_cache is None:
        attn_output = block.attn(
            hidden_states=norm_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
    else:
        attn_output = attn_cache
    projected = block.proj_out(torch.cat([attn_output, mlp_output], dim=2))
    joint_states = residual + gate.unsqueeze(1) * projected
    if joint_states.dtype == torch.float16:
        joint_states = joint_states.clip(-65504, 65504)
    encoder_hidden_states = joint_states[:, :text_len]
    hidden_states = joint_states[:, text_len:]
    return encoder_hidden_states, hidden_states, {
        "attn": attn_output,
        "mlp": mlp_output,
    }


class FinegrainedAnalyzer:
    def __init__(self, double_layers: int, single_layers: int):
        self.double_layers = int(double_layers)
        self.single_layers = int(single_layers)
        self.state: Dict[str, List[Dict[str, Any]]] = {}
        for key in DOUBLE_RATE_KEYS:
            self.state[f"double.{key}"] = [
                self._new_track() for _ in range(double_layers)
            ]
        for key in SINGLE_KEYS:
            self.state[f"single.{key}"] = [
                self._new_track() for _ in range(single_layers)
            ]

    @staticmethod
    def _new_track():
        return {
            "count": 0,
            "prev": None,
            "current": None,
            "norm": None,
            "active": False,
        }

    @staticmethod
    def _features(double_outputs, single_outputs):
        features = {}
        for key in DOUBLE_RATE_KEYS:
            features[f"double.{key}"] = [item[key] for item in double_outputs]
        for key in SINGLE_KEYS:
            features[f"single.{key}"] = [item[key] for item in single_outputs]
        return features

    def observe(
        self,
        double_outputs,
        single_outputs,
        step_idx: int,
        effective_books: Optional[Dict[str, List[List[bool]]]] = None,
    ) -> Dict[str, List[float]]:
        result: Dict[str, List[float]] = {}
        for trajectory, values in self._features(
            double_outputs, single_outputs
        ).items():
            result[trajectory] = []
            for block_idx, value in enumerate(values):
                feature = value.detach()
                track = self.state[trajectory][block_idx]
                if track["count"] == 0:
                    track["current"] = feature
                    track["count"] = 1
                    result[trajectory].append(float("nan"))
                    continue
                if track["count"] == 1:
                    track["prev"] = track["current"]
                    track["current"] = feature
                    track["norm"] = compute_l1_distance(
                        track["prev"], track["current"]
                    )
                    track["count"] = 2
                    result[trajectory].append(float("nan"))
                    continue
                score = compute_rate(
                    track["prev"],
                    track["current"],
                    feature,
                    track["norm"],
                )
                result[trajectory].append(float(score.item()))
                refresh = True
                if effective_books is not None:
                    cached_previous = bool(
                        effective_books[trajectory][step_idx - 1][block_idx]
                    )
                    if not cached_previous:
                        track["active"] = False
                        refresh = True
                    elif not track["active"]:
                        track["active"] = True
                        refresh = True
                    else:
                        refresh = False
                if refresh:
                    track["norm"] = compute_l1_distance(
                        track["current"], feature
                    )
                    track["prev"] = track["current"]
                    track["current"] = feature
        return result


class CalibrationRun:
    def __init__(
        self,
        num_steps: int,
        double_layers: int,
        single_layers: int,
        effective_books=None,
        collect_mag_ratios: bool = False,
    ):
        self.num_steps = num_steps
        self.double_layers = double_layers
        self.single_layers = single_layers
        self.analyzer = FinegrainedAnalyzer(double_layers, single_layers)
        self.effective_books = effective_books
        self.collect_mag_ratios = collect_mag_ratios
        self.previous_residual_norm = None
        self.mag_ratios = [1.0] * num_steps
        self.scores = {
            **{
                f"double.{key}": torch.ones(num_steps, double_layers)
                for key in DOUBLE_RATE_KEYS
            },
            **{
                f"single.{key}": torch.ones(num_steps, single_layers)
                for key in SINGLE_KEYS
            },
        }

    def callback(
        self,
        step_idx,
        double_outputs,
        single_outputs,
        residual_norm,
        would_skip,
    ):
        scores = self.analyzer.observe(
            double_outputs,
            single_outputs,
            step_idx,
            self.effective_books,
        )
        score_idx = step_idx - 1
        if step_idx >= 2:
            for key, values in scores.items():
                self.scores[key][score_idx] = torch.tensor(values)
        if self.collect_mag_ratios:
            current = residual_norm.detach()
            if self.previous_residual_norm is not None:
                ratio = (
                    current
                    / self.previous_residual_norm.clamp_min(1e-8)
                ).mean()
                self.mag_ratios[step_idx] = float(ratio.item())
            self.previous_residual_norm = current


class HybridFluxTransformer(nn.Module):
    def __init__(self, base_model: nn.Module, num_steps: int):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dtype = getattr(base_model, "dtype", torch.bfloat16)
        self.transformer_blocks = base_model.transformer_blocks
        self.single_transformer_blocks = base_model.single_transformer_blocks
        self.num_layers = len(self.transformer_blocks)
        self.num_single_layers = len(self.single_transformer_blocks)
        self.num_steps = int(num_steps)
        self.layer_books = None
        self.layer_cache_enabled = False
        self.execute_step_cache = True
        self.forced_observer_mask = None
        self.calibration_callback: Optional[Callable] = None
        self.step_policy: Optional[MagCachePolicy] = None
        self.reset_runtime()

    def cache_context(self, *args, **kwargs):
        return self.base_model.cache_context(*args, **kwargs)

    def reset_runtime(self):
        self.step_idx = 0
        self.previous_step_residual = None
        self.previous_step_was_cached = False
        self.double_cache = {
            key: [None] * self.num_layers for key in DOUBLE_CACHE_KEYS
        }
        self.single_cache = {
            key: [None] * self.num_single_layers for key in SINGLE_KEYS
        }
        self.step_hits = 0
        self.computed_steps = 0
        self.layer_hits = 0
        if self.step_policy is not None:
            self.step_policy.reset()

    def configure_layer_cache(self, books):
        self.layer_books = books
        self.layer_cache_enabled = books is not None

    def _layer_hit(self, trajectory: str, block_idx: int) -> bool:
        if not self.layer_cache_enabled or self.previous_step_was_cached:
            return False
        return bool(self.layer_books[trajectory][self.step_idx][block_idx])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        if self.step_idx >= self.num_steps:
            raise RuntimeError("Runtime state was not reset before a new sample")
        if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
            raise ValueError("ControlNet is not supported by this hybrid sampler")
        joint_attention_kwargs = dict(joint_attention_kwargs or {})
        if "ip_adapter_image_embeds" in joint_attention_kwargs:
            raise ValueError("IP-Adapter is not supported by this hybrid sampler")
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        if USE_PEFT_BACKEND:
            scale_lora_layers(self.base_model, lora_scale)

        hidden_states = self.base_model.x_embedder(hidden_states)
        original_hidden_states = hidden_states
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.base_model.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.base_model.time_text_embed(
                timestep, guidance, pooled_projections
            )
        )
        encoder_hidden_states = self.base_model.context_embedder(
            encoder_hidden_states
        )
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        image_rotary_emb = self.base_model.pos_embed(
            torch.cat((txt_ids, img_ids), dim=0)
        )

        if self.forced_observer_mask is not None:
            would_skip = bool(self.forced_observer_mask[self.step_idx])
        elif self.step_policy is not None:
            would_skip = self.step_policy.decide(self.step_idx)
        else:
            would_skip = False
        actual_skip = (
            self.execute_step_cache
            and would_skip
            and self.previous_step_residual is not None
        )
        double_outputs = []
        single_outputs = []
        if actual_skip:
            hidden_states = hidden_states + self.previous_step_residual
            self.step_hits += 1
        else:
            for block_idx, block in enumerate(self.transformer_blocks):
                joint_attn_hit = (
                    self._layer_hit("double.attn", block_idx)
                    and self._layer_hit("double.context_attn", block_idx)
                )
                joint_attn_hit = bool(
                    joint_attn_hit
                    and self.double_cache["attn"][block_idx] is not None
                    and self.double_cache["context_attn"][block_idx] is not None
                )
                ff_hit = bool(
                    self._layer_hit("double.ff", block_idx)
                    and self.double_cache["ff"][block_idx] is not None
                )
                context_ff_hit = bool(
                    self._layer_hit("double.context_ff", block_idx)
                    and self.double_cache["context_ff"][block_idx] is not None
                )
                self.layer_hits += 2 * int(joint_attn_hit)
                self.layer_hits += int(ff_hit) + int(context_ff_hit)
                encoder_hidden_states, hidden_states, outputs = (
                    finegrained_double_forward(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        self.double_cache["attn"][block_idx]
                        if joint_attn_hit else None,
                        self.double_cache["context_attn"][block_idx]
                        if joint_attn_hit else None,
                        self.double_cache["ip_attn"][block_idx]
                        if joint_attn_hit else None,
                        self.double_cache["ff"][block_idx]
                        if ff_hit else None,
                        self.double_cache["context_ff"][block_idx]
                        if context_ff_hit else None,
                    )
                )
                double_outputs.append(outputs)
                if self.layer_cache_enabled:
                    for key in DOUBLE_CACHE_KEYS:
                        self.double_cache[key][block_idx] = outputs[key]
            for block_idx, block in enumerate(self.single_transformer_blocks):
                attn_hit = bool(
                    self._layer_hit("single.attn", block_idx)
                    and self.single_cache["attn"][block_idx] is not None
                )
                mlp_hit = bool(
                    self._layer_hit("single.mlp", block_idx)
                    and self.single_cache["mlp"][block_idx] is not None
                )
                self.layer_hits += int(attn_hit) + int(mlp_hit)
                encoder_hidden_states, hidden_states, outputs = (
                    finegrained_single_forward(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        self.single_cache["attn"][block_idx]
                        if attn_hit else None,
                        self.single_cache["mlp"][block_idx]
                        if mlp_hit else None,
                    )
                )
                single_outputs.append(outputs)
                if self.layer_cache_enabled:
                    for key in SINGLE_KEYS:
                        self.single_cache[key][block_idx] = outputs[key]
            residual = hidden_states - original_hidden_states
            self.previous_step_residual = residual
            self.computed_steps += 1
            if self.calibration_callback is not None:
                residual_norm = residual.float().norm(p=2, dim=-1)
                self.calibration_callback(
                    self.step_idx,
                    double_outputs,
                    single_outputs,
                    residual_norm,
                    would_skip,
                )
        self.previous_step_was_cached = actual_skip
        hidden_states = self.base_model.norm_out(hidden_states, temb)
        output = self.base_model.proj_out(hidden_states)
        self.step_idx += 1
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self.base_model, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def threshold_config(args):
    return {
        "double.attn": args.attn_thres,
        "double.context_attn": args.context_attn_thres,
        "double.ff": args.ff_thres,
        "double.context_ff": args.context_ff_thres,
        "single.attn": args.single_attn_thres,
        "single.mlp": args.single_mlp_thres,
    }


def average_runs(runs: List[CalibrationRun]):
    return {
        key: torch.stack([run.scores[key] for run in runs]).mean(dim=0)
        for key in runs[0].scores
    }


def build_layer_books(
    average_scores,
    args,
    double_layers,
    single_layers,
    protected_steps,
):
    books = {}
    for trajectory, threshold_q in threshold_config(args).items():
        layers = double_layers if trajectory.startswith("double.") else single_layers
        if threshold_q == 0:
            book = torch.zeros(args.num_inference_steps, layers, dtype=torch.bool)
        else:
            threshold = torch.quantile(
                average_scores[trajectory][1:-1], threshold_q
            )
            book = average_scores[trajectory] < threshold
        book[:protected_steps] = False
        book[-1] = False
        books[trajectory] = book
    books["double.ip_attn"] = books["double.attn"].clone()
    return books


def apply_step_first(books, step_mask):
    effective = {key: value.clone() for key, value in books.items()}
    for step_idx, cached in enumerate(step_mask):
        if not cached:
            continue
        for book in effective.values():
            book[step_idx] = True
            if step_idx + 1 < len(step_mask):
                book[step_idx + 1] = False
    return effective


def books_to_lists(books):
    return {key: value.cpu().tolist() for key, value in books.items()}


def run_calibration_prompt(
    pipe,
    model,
    prompt,
    args,
    effective_books=None,
    collect_mag_ratios=False,
    observer_mask=None,
):
    run = CalibrationRun(
        args.num_inference_steps,
        model.num_layers,
        model.num_single_layers,
        effective_books,
        collect_mag_ratios,
    )
    model.reset_runtime()
    model.configure_layer_cache(None)
    model.execute_step_cache = False
    model.forced_observer_mask = observer_mask
    model.calibration_callback = run.callback
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        generator=generator,
        output_type="latent",
    )
    del result
    model.calibration_callback = None
    model.forced_observer_mask = None
    return run


def calibrate(pipe, model, prompts, args):
    print("Calibration pass 1/2: joint raw layer and MagCache statistics")
    raw_runs = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for prompt in tqdm(
        prompts,
        desc="Raw calibration",
        disable=args.disable_progress_bar,
    ):
        raw_runs.append(
            run_calibration_prompt(
                pipe, model, prompt, args, collect_mag_ratios=True
            )
        )
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Raw calibration time: {time.perf_counter() - start:.4f} s")
    print(
        "Raw calibration peak allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )
    ratios = np.mean(
        np.asarray([run.mag_ratios for run in raw_runs], dtype=np.float64),
        axis=0,
    ).tolist()
    provisional_step_book = derive_magcache_book(
        ratios,
        args.num_inference_steps,
        args.magcache_thresh,
        args.magcache_k,
        args.retention_ratio,
    )
    provisional_books = build_layer_books(
        average_runs(raw_runs),
        args,
        model.num_layers,
        model.num_single_layers,
        protected_steps=1,
    )
    effective_books = apply_step_first(
        provisional_books, provisional_step_book
    )
    del raw_runs
    gc.collect()
    torch.cuda.empty_cache()

    print("Calibration pass 2/2: hybrid-aware layer correction")
    corrected_runs = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for prompt in tqdm(
        prompts,
        desc="Correction calibration",
        disable=args.disable_progress_bar,
    ):
        corrected_runs.append(
            run_calibration_prompt(
                pipe,
                model,
                prompt,
                args,
                effective_books=effective_books,
                observer_mask=provisional_step_book,
            )
        )
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Correction calibration time: {time.perf_counter() - start:.4f} s")
    print(
        "Correction peak allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )
    final_books = build_layer_books(
        average_runs(corrected_runs),
        args,
        model.num_layers,
        model.num_single_layers,
        protected_steps=max(1, int(args.nonskip_rate * args.num_inference_steps)),
    )
    final_books = apply_step_first(final_books, provisional_step_book)
    return ratios, provisional_step_book, final_books


def fmt(value):
    return format(float(value), "g")


def default_cache_name(args):
    return (
        f"cache_book_hybrid_magcache_steps{args.num_inference_steps}"
        f"_ns{fmt(args.nonskip_rate)}_magth{fmt(args.magcache_thresh)}"
        f"_k{args.magcache_k}_r{fmt(args.retention_ratio)}"
        f"_attnth{fmt(args.attn_thres)}_ffth{fmt(args.ff_thres)}"
        f"_cattnth{fmt(args.context_attn_thres)}"
        f"_ctxffth{fmt(args.context_ff_thres)}"
        f"_sattnth{fmt(args.single_attn_thres)}"
        f"_smlpth{fmt(args.single_mlp_thres)}.json"
    )


def make_cache_book(args, model, ratios, step_book, layer_books):
    return {
        "cache_scope": "hybrid",
        "layer_policy": "finegrained_cache",
        "step_policy": METHOD,
        "layer_rate_method": RATE_METHOD,
        "config": {
            "model": "FLUX.1-dev",
            "height": args.height,
            "width": args.width,
            "steps": args.num_inference_steps,
            "nonskip_rate": args.nonskip_rate,
            "thresholds": threshold_config(args),
            "double_layers": model.num_layers,
            "single_layers": model.num_single_layers,
            "magcache_threshold": args.magcache_thresh,
            "magcache_k": args.magcache_k,
            "retention_ratio": args.retention_ratio,
            "source_commit": "df81cb1",
        },
        "step_policy_artifacts": {
            "mag_ratios": ratios,
            "step_cache_book": step_book,
        },
        "layer_cache_book": books_to_lists(layer_books),
    }


def validate_cache_book(book, args, model):
    if book.get("cache_scope") != "hybrid":
        raise ValueError("Cache Book scope mismatch")
    if book.get("layer_policy") != "finegrained_cache":
        raise ValueError("Layer policy mismatch")
    if book.get("step_policy") != METHOD:
        raise ValueError("Step policy mismatch")
    if book.get("layer_rate_method") != RATE_METHOD:
        raise ValueError("Layer rate method mismatch")
    config = book.get("config", {})
    expected = {
        "height": args.height,
        "width": args.width,
        "steps": args.num_inference_steps,
        "double_layers": model.num_layers,
        "single_layers": model.num_single_layers,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Cache Book {key} mismatch: {config.get(key)!r} != {value!r}"
            )
    if config.get("nonskip_rate") != args.nonskip_rate:
        raise ValueError("Cache Book nonskip_rate mismatch")
    if config.get("thresholds") != threshold_config(args):
        raise ValueError("Cache Book layer thresholds mismatch")
    if config.get("magcache_threshold") != args.magcache_thresh:
        raise ValueError("MagCache threshold mismatch")
    if config.get("magcache_k") != args.magcache_k:
        raise ValueError("MagCache K mismatch")
    if config.get("retention_ratio") != args.retention_ratio:
        raise ValueError("MagCache retention ratio mismatch")
    layer_books = book.get("layer_cache_book", {})
    expected_keys = {
        *(f"double.{key}" for key in DOUBLE_CACHE_KEYS),
        *(f"single.{key}" for key in SINGLE_KEYS),
    }
    if set(layer_books) != expected_keys:
        raise ValueError("Cache Book module keys mismatch")
    for trajectory, values in layer_books.items():
        expected_layers = (
            model.num_layers
            if trajectory.startswith("double.")
            else model.num_single_layers
        )
        if len(values) != args.num_inference_steps:
            raise ValueError(f"{trajectory} timestep count mismatch")
        if any(len(row) != expected_layers for row in values):
            raise ValueError(f"{trajectory} layer count mismatch")
    ratios = book.get("step_policy_artifacts", {}).get("mag_ratios")
    step_book = book.get("step_policy_artifacts", {}).get("step_cache_book")
    if len(ratios or []) != args.num_inference_steps:
        raise ValueError("MagCache ratio length mismatch")
    if len(step_book or []) != args.num_inference_steps:
        raise ValueError("MagCache step book length mismatch")


def resolve_model_path(path_value):
    path = Path(path_value).expanduser().resolve()
    if (path / "model_index.json").is_file():
        return path
    ref_file = path / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(f"No model_index.json or refs/main under {path}")
    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot = path / "snapshots" / revision
    if not (snapshot / "model_index.json").is_file():
        raise FileNotFoundError(f"No model_index.json under {snapshot}")
    return snapshot


def load_pipeline(args, cpu_offload):
    model_path = resolve_model_path(args.model_path)
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = HybridFluxTransformer(pipe.transformer, args.num_inference_steps)
    pipe.transformer = model
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe.to("cuda")
    return pipe, model


def load_prompts(args):
    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(
            line.strip()
            for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or list(DEFAULT_PROMPTS)


def save_images(images, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = images[0].size
    combined = Image.new("RGB", (width * len(images), height))
    for index, image in enumerate(images):
        combined.paste(image, (index * width, 0))
    timestamp = time.strftime("%m%d_%H%M%S")
    path = output_dir / (
        f"flux_{METHOD}_hybrid_steps{args.num_inference_steps}"
        f"_seed{args.seed}_{timestamp}.png"
    )
    combined.save(path)
    return path


def generate(pipe, model, prompts, args):
    images, times = [], []
    for prompt_idx, prompt in enumerate(prompts):
        model.reset_runtime()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        layer_slots_per_step = (
            model.num_layers * len(DOUBLE_RATE_KEYS)
            + model.num_single_layers * len(SINGLE_KEYS)
        )
        computed_layer_slots = model.computed_steps * layer_slots_per_step
        total_layer_slots = args.num_inference_steps * layer_slots_per_step
        layer_skip_ratio = (
            model.layer_hits / computed_layer_slots
            if computed_layer_slots else 0.0
        )
        combined_skip_ratio = (
            (model.step_hits * layer_slots_per_step + model.layer_hits)
            / total_layer_slots
            if total_layer_slots else 0.0
        )
        label = "warm-up" if prompt_idx == 0 and len(prompts) > 1 else "measured"
        print(
            f"Run {prompt_idx} ({label}): {elapsed:.4f} s, "
            f"step skip={model.step_hits / args.num_inference_steps:.2%}, "
            f"layer skip on computed steps={layer_skip_ratio:.2%}, "
            f"combined effective skip={combined_skip_ratio:.2%}, "
            f"peak allocated={peak_gib:.2f} GiB"
        )
        images.append(image)
        times.append(elapsed)
    measured = times[1:] if len(times) > 1 else times
    print(
        f"Sampling time: {np.mean(measured):.4f} ± "
        f"{np.std(measured):.4f} s"
    )
    path = save_images(images, Path(args.output_dir), args)
    print(f"Images saved to {path}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_model = script_dir.parent / "models--black-forest-labs--FLUX.1-dev"
    parser = argparse.ArgumentParser(
        description="Standalone FLUX MagCache + Finegrained Cache sampler"
    )
    parser.add_argument("--model-path", default=str(default_model))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--calibration-prompt",
        action="append",
        default=None,
    )
    parser.add_argument("--output-dir", default=str(script_dir.parent / "images"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finegrained-calibration", action="store_true")
    parser.add_argument("--use-finegrained-cache", action="store_true")
    parser.add_argument(
        "--cache-book-path",
        default=str(script_dir / "cache_books"),
    )
    parser.add_argument("--cache-book-file")
    parser.add_argument("--nonskip-rate", type=float, default=0.1)
    # fast (default): attn=0.30, context_attn=0.30, single_attn=0.12,
    # ff=0.22, context_ff=0.40, single_mlp=0.20;
    # balanced: 0.30, 0.30, 0.12, 0.22, 0.40, 0.11;
    # slow: 0.30, 0.30, 0.06, 0.21, 0.40, 0.06.
    parser.add_argument("--attn-thres", type=float, default=0.30)
    parser.add_argument("--context-attn-thres", type=float, default=0.30)
    parser.add_argument("--ff-thres", type=float, default=0.22)
    parser.add_argument("--context-ff-thres", type=float, default=0.40)
    parser.add_argument("--single-attn-thres", type=float, default=0.12)
    parser.add_argument("--single-mlp-thres", type=float, default=0.20)
    parser.add_argument("--disable-step-cache", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--magcache-k", type=int, default=4)
    parser.add_argument("--magcache-thresh", type=float, default=0.24)
    parser.add_argument("--retention-ratio", type=float, default=0.2)
    parser.add_argument(
        "--calibration-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--runtime-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    for name, value in threshold_config(args).items():
        if not 0 <= value <= 1:
            parser.error(f"{name} threshold must be in [0, 1]")
    if args.num_inference_steps < 2:
        parser.error("--num-inference-steps must be at least 2")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    cache_dir = Path(args.cache_book_path)
    cache_file = args.cache_book_file or default_cache_name(args)
    cache_path = cache_dir / cache_file
    book = None

    if args.finegrained_calibration:
        pipe, model = load_pipeline(args, args.calibration_cpu_offload)
        calibration_prompts = (
            args.calibration_prompt or [DEFAULT_CALIBRATION_PROMPT]
        )
        with torch.inference_mode():
            ratios, step_book, layer_books = calibrate(
                pipe, model, calibration_prompts, args
            )
        book = make_cache_book(
            args, model, ratios, step_book, layer_books
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(book, indent=2), encoding="utf-8"
        )
        print(f"Cache Book saved to {cache_path}")
        del pipe, model
        gc.collect()
        torch.cuda.empty_cache()
        if not args.use_finegrained_cache:
            return

    pipe, model = load_pipeline(args, args.runtime_cpu_offload)
    if args.use_finegrained_cache:
        if book is None:
            book = json.loads(cache_path.read_text(encoding="utf-8"))
        validate_cache_book(book, args, model)
        model.configure_layer_cache(book["layer_cache_book"])
        ratios = book["step_policy_artifacts"]["mag_ratios"]
    else:
        model.configure_layer_cache(None)
        ratios = nearest_interp(
            MAG_RATIOS_28, args.num_inference_steps
        ).tolist()
    model.step_policy = MagCachePolicy(
        ratios,
        args.num_inference_steps,
        args.magcache_thresh,
        args.magcache_k,
        args.retention_ratio,
        enabled=not args.disable_step_cache,
    )
    model.execute_step_cache = not args.disable_step_cache
    prompts = load_prompts(args)
    with torch.inference_mode():
        generate(pipe, model, prompts, args)


if __name__ == "__main__":
    main()
