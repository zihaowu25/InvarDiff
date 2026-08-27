"""Standalone FLUX sampler with layer-only three-point L1 caching.

Cross-step cache code is intentionally retained as comments for future
experiments. Runtime step cache decisions are always False.
"""

import json
import os
import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TRANSFORMER_MODULES = (
    "attn",
    "context_attn",
    "ip_attn",
    "ff",
    "context_ff",
)
TRANSFORMER_RATE_MODULES = (
    "attn",
    "context_attn",
    "ff",
    "context_ff",
)
SINGLE_TRANSFORMER_MODULES = ("attn", "mlp")
RATE_METHOD = "three_point_l1"
CACHE_BOOK_VERSION = 2
POLICY_VARIANT = "layer"
RATE_CHUNK_SIZE = 1_048_576


def compute_l1_distance(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    chunk_size: int = RATE_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ||x_end - x_start + 1e-8||_1 with chunked FP32 sums."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    x_start = x_start.detach().reshape(-1)
    x_end = x_end.detach().reshape(-1)
    if x_start.numel() != x_end.numel():
        raise ValueError(
            "Rate feature sizes must match, got "
            f"{x_start.numel()} and {x_end.numel()}."
        )

    total = torch.zeros((), device=x_start.device, dtype=torch.float32)
    for start in range(0, x_start.numel(), chunk_size):
        end = min(start + chunk_size, x_start.numel())
        diff = x_end[start:end] - x_start[start:end] + 1e-8
        total += diff.abs().sum(dtype=torch.float32)
    return total


def compute_rate(
    current_norm: torch.Tensor,
    previous_norm: torch.Tensor,
) -> torch.Tensor:
    """Return the ratio between the current and previous L1 displacements."""
    return current_norm / previous_norm.clamp_min(1e-8)


def register_hooks(model, transformer_keys, single_transformer_keys):
    """Register hooks only for modules with non-zero layer thresholds."""
    transformer_keys = tuple(transformer_keys)
    single_transformer_keys = tuple(single_transformer_keys)

    unknown_transformer = set(transformer_keys) - set(TRANSFORMER_RATE_MODULES)
    unknown_single = set(single_transformer_keys) - set(
        SINGLE_TRANSFORMER_MODULES
    )
    if unknown_transformer or unknown_single:
        raise ValueError(
            "Unknown hook keys: "
            f"transformer={sorted(unknown_transformer)}, "
            f"single={sorted(unknown_single)}"
        )

    transformer_blocks = {key: {} for key in transformer_keys}
    single_transformer_blocks = {key: {} for key in single_transformer_keys}

    def create_transformer_hook(index_block):
        def hook(module, inputs, output):
            _, _, block_outputs = output
            for key in transformer_keys:
                transformer_blocks[key][index_block] = block_outputs[key]
        return hook

    def create_single_transformer_hook(index_block):
        def hook(module, inputs, output):
            _, block_outputs = output
            for key in single_transformer_keys:
                single_transformer_blocks[key][index_block] = block_outputs[key]
        return hook

    hooks = []
    if transformer_keys:
        for index, block in enumerate(model.transformer_blocks):
            hooks.append(
                block.register_forward_hook(create_transformer_hook(index))
            )
    if single_transformer_keys:
        for index, block in enumerate(model.single_transformer_blocks):
            hooks.append(
                block.register_forward_hook(
                    create_single_transformer_hook(index)
                )
            )

    return transformer_blocks, single_transformer_blocks, hooks

class FeatureChangeAnalyzer:
    """Three-point analyzer for raw and cache-corrected layer calibration."""

    def __init__(
        self,
        num_transformer_layers: int,
        num_single_layers: int,
        transformer_keys=TRANSFORMER_RATE_MODULES,
        single_transformer_keys=SINGLE_TRANSFORMER_MODULES,
    ):
        self.num_transformer_layers = num_transformer_layers
        self.num_single_layers = num_single_layers
        self.transformer_keys = tuple(transformer_keys)
        self.single_transformer_keys = tuple(single_transformer_keys)
        self.Transformer_state = {
            key: [False] * self.num_transformer_layers
            for key in self.transformer_keys
        }
        self.SingleTransformer_state = {
            key: [False] * self.num_single_layers
            for key in self.single_transformer_keys
        }
        self._layer_count = 0

    # Cross-step cache is intentionally disabled in this layer-only variant.
    # The original three-point step scorer is retained below as comments.
    #
    # def step_forward(self, hidden_states):
    #     hidden_states = hidden_states.detach()
    #     if self._step_count == 0:
    #         self.current_hidden_states = hidden_states
    #         self._step_count = 1
    #         return None
    #
    #     if self._step_count == 1:
    #         self.prev_step_norm = compute_l1_distance(
    #             self.current_hidden_states,
    #             hidden_states,
    #         )
    #         self.current_hidden_states = hidden_states
    #         self._step_count = 2
    #         return None
    #
    #     current_norm = compute_l1_distance(
    #         self.current_hidden_states,
    #         hidden_states,
    #     )
    #     step_score = compute_rate(current_norm, self.prev_step_norm)
    #     self.prev_step_norm = current_norm
    #     self.current_hidden_states = hidden_states
    #     return step_score

    @staticmethod
    def _detach_feature(feature):
        return None if feature is None else feature.detach()

    def _collect_features(self, transformer_dict, single_dict):
        transformer_features = {
            key: [
                self._detach_feature(transformer_dict[key][index])
                for index in range(self.num_transformer_layers)
            ]
            for key in self.transformer_keys
        }
        single_features = {
            key: [
                self._detach_feature(single_dict[key][index])
                for index in range(self.num_single_layers)
            ]
            for key in self.single_transformer_keys
        }
        return transformer_features, single_features

    def _start_layer_state(self, transformer_features, single_features):
        self.current_Transformer = transformer_features
        self.current_SingleTransformer = single_features
        self._layer_count = 1

    def _initialize_layer_state(self, transformer_features, single_features):
        self.prev_Transformer_norm = {
            key: [None] * self.num_transformer_layers
            for key in self.transformer_keys
        }
        for key in self.transformer_keys:
            for block_idx, feature in enumerate(transformer_features[key]):
                current = self.current_Transformer[key][block_idx]
                if current is not None and feature is not None:
                    self.prev_Transformer_norm[key][block_idx] = (
                        compute_l1_distance(current, feature)
                    )

        self.prev_SingleTransformer_norm = {
            key: [None] * self.num_single_layers
            for key in self.single_transformer_keys
        }
        for key in self.single_transformer_keys:
            for block_idx, feature in enumerate(single_features[key]):
                current = self.current_SingleTransformer[key][block_idx]
                if current is not None and feature is not None:
                    self.prev_SingleTransformer_norm[key][block_idx] = (
                        compute_l1_distance(current, feature)
                    )

        self.current_Transformer = transformer_features
        self.current_SingleTransformer = single_features

        self._layer_count = 2

    @staticmethod
    def _should_refresh(
        cache_state,
        active_state,
        key,
        timestep_idx,
        block_idx,
    ):
        cached_on_previous_step = cache_state[key][timestep_idx - 1][block_idx]
        if not cached_on_previous_step:
            active_state[key][block_idx] = False
            return True
        if not active_state[key][block_idx]:
            active_state[key][block_idx] = True
            return True
        return False

    def step(self, transformer_dict, single_dict):
        """Collect uncorrected layer scores and rotate states in place."""
        transformer_features, single_features = self._collect_features(
            transformer_dict,
            single_dict,
        )
        if self._layer_count == 0:
            self._start_layer_state(transformer_features, single_features)
            return None, None
        if self._layer_count == 1:
            self._initialize_layer_state(transformer_features, single_features)
            return None, None

        transformer_scores = {}
        for key in self.transformer_keys:
            scores = []
            for block_idx, feature in enumerate(transformer_features[key]):
                current = self.current_Transformer[key][block_idx]
                if current is None or feature is None:
                    scores.append(float("nan"))
                    next_norm = None
                else:
                    next_norm = compute_l1_distance(current, feature)
                    score = compute_rate(
                        next_norm,
                        self.prev_Transformer_norm[key][block_idx],
                    )
                    scores.append(score.item())
                self.current_Transformer[key][block_idx] = feature
                self.prev_Transformer_norm[key][block_idx] = next_norm

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                current = self.current_SingleTransformer[key][block_idx]
                if current is None or feature is None:
                    scores.append(float("nan"))
                    next_norm = None
                else:
                    next_norm = compute_l1_distance(current, feature)
                    score = compute_rate(
                        next_norm,
                        self.prev_SingleTransformer_norm[key][block_idx],
                    )
                    scores.append(score.item())
                self.current_SingleTransformer[key][block_idx] = feature
                self.prev_SingleTransformer_norm[key][block_idx] = next_norm

            single_scores[key] = torch.tensor(scores)

        return transformer_scores, single_scores

    def step_correct(
        self,
        transformer_dict,
        single_dict,
        transformer_cache_state,
        single_cache_state,
        timestep_idx,
    ):
        """Collect layer scores while following provisional layer cache paths."""
        transformer_features, single_features = self._collect_features(
            transformer_dict,
            single_dict,
        )
        if timestep_idx == 0:
            self._start_layer_state(transformer_features, single_features)
            return None, None
        if timestep_idx == 1:
            self._initialize_layer_state(transformer_features, single_features)
            return None, None

        transformer_scores = {}
        for key in self.transformer_keys:
            scores = []
            for block_idx, feature in enumerate(transformer_features[key]):
                current = self.current_Transformer[key][block_idx]
                if current is None or feature is None:
                    scores.append(float("nan"))
                else:
                    next_norm = compute_l1_distance(current, feature)
                    score = compute_rate(
                        next_norm,
                        self.prev_Transformer_norm[key][block_idx],
                    )
                    scores.append(score.item())

                if self._should_refresh(
                    transformer_cache_state,
                    self.Transformer_state,
                    key,
                    timestep_idx,
                    block_idx,
                ):
                    self.current_Transformer[key][block_idx] = feature
                    self.prev_Transformer_norm[key][block_idx] = next_norm

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                current = self.current_SingleTransformer[key][block_idx]
                if current is None or feature is None:
                    scores.append(float("nan"))
                else:
                    next_norm = compute_l1_distance(current, feature)
                    score = compute_rate(
                        next_norm,
                        self.prev_SingleTransformer_norm[key][block_idx],
                    )
                    scores.append(score.item())

                if self._should_refresh(
                    single_cache_state,
                    self.SingleTransformer_state,
                    key,
                    timestep_idx,
                    block_idx,
                ):
                    self.current_SingleTransformer[key][block_idx] = feature
                    self.prev_SingleTransformer_norm[key][block_idx] = next_norm

            single_scores[key] = torch.tensor(scores)

        return transformer_scores, single_scores

def threshold_analyse(
    model,
    pipe,
    measure_prompts,
    nonskip_rate=0.1,
    # fast (default): attn=0.30, context_attn=0.30, single_attn=0.40,
    # ff=0.00, context_ff=0.00, single_mlp=0.00;
    # balanced: attn=0.30, context_attn=0.30, single_attn=0.12,
    # ff=0.22, context_ff=0.40, single_mlp=0.30;
    # slow: attn=0.30, context_attn=0.30, single_attn=0.10,
    # ff=0.00, context_ff=0.00, single_mlp=0.00.
    attn_thres=0.30,
    context_attn_thres=0.30,
    ff_thres=0.00,
    context_ff_thres=0.00,
    Single_attn_thres=0.40,
    Single_mlp_thres=0.00,
    seed=42,
):
    """Run raw and corrected calibration for layer cache only."""
    # Cross-step threshold parameter is intentionally disabled:
    # step_thres = 0.5

    device = pipe._execution_device
    num_timesteps = model.num_timesteps
    num_transformer_layers = model.num_layers
    num_single_layers = model.num_single_layers
    num_nonskip = max(1, int(nonskip_rate * num_timesteps))

    transformer_thresholds = {
        "attn": attn_thres,
        "context_attn": context_attn_thres,
        "ff": ff_thres,
        "context_ff": context_ff_thres,
    }
    single_thresholds = {
        "attn": Single_attn_thres,
        "mlp": Single_mlp_thres,
    }
    for module_name, threshold in {
        **transformer_thresholds,
        **{f"single_{key}": value for key, value in single_thresholds.items()},
    }.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {module_name!r} must be in [0, 1], "
                f"got {threshold}."
            )

    active_transformer_keys = tuple(
        key
        for key, threshold in transformer_thresholds.items()
        if threshold > 0.0
    )
    active_single_keys = tuple(
        key
        for key, threshold in single_thresholds.items()
        if threshold > 0.0
    )

    print(
        f"Cache scope: layer only\n"
        f"Rate method: {RATE_METHOD}\n"
        f"Active transformer scores: {active_transformer_keys}\n"
        f"Active single-transformer scores: {active_single_keys}\n"
        f"Transformer blocks numbers: {num_transformer_layers}\n"
        f"Single Transformer blocks numbers: {num_single_layers}"
    )

    def build_layer_cache_book(
        average_scores,
        thresholds,
        num_layers,
        protected_prefix,
    ):
        cache_book = {}
        for module_name, threshold_q in thresholds.items():
            if threshold_q == 0.0:
                cache_bool = torch.zeros(
                    num_timesteps,
                    num_layers,
                    dtype=torch.bool,
                )
            else:
                threshold_value = torch.quantile(
                    average_scores[module_name][1:-1],
                    threshold_q,
                )
                cache_bool = (
                    average_scores[module_name] < threshold_value
                )

            cache_bool[:protected_prefix, :] = False
            cache_bool[-1, :] = False
            cache_book[module_name] = cache_bool
        return cache_book

    def collect_layer_scores(
        description,
        transformer_cache_book=None,
        single_cache_book=None,
    ):
        correction_mode = transformer_cache_book is not None
        transformer_runs = {
            key: [] for key in active_transformer_keys
        }
        single_runs = {
            key: [] for key in active_single_keys
        }

        track_cuda_memory = torch.cuda.is_available()
        cuda_memory_device = (
            torch.cuda.current_device() if track_cuda_memory else None
        )
        if track_cuda_memory:
            torch.cuda.reset_peak_memory_stats(cuda_memory_device)

        with torch.inference_mode():
            for prompt in tqdm(measure_prompts, desc=description):
                transformer_features, single_features, hooks = register_hooks(
                    model,
                    active_transformer_keys,
                    active_single_keys,
                )
                analyzer = FeatureChangeAnalyzer(
                    num_transformer_layers,
                    num_single_layers,
                    active_transformer_keys,
                    active_single_keys,
                )
                run_transformer_scores = {
                    key: torch.ones(
                        num_timesteps,
                        num_transformer_layers,
                        dtype=torch.float32,
                    )
                    for key in active_transformer_keys
                }
                run_single_scores = {
                    key: torch.ones(
                        num_timesteps,
                        num_single_layers,
                        dtype=torch.float32,
                    )
                    for key in active_single_keys
                }

                # Cross-step score collection is intentionally disabled:
                # run_step_scores = torch.ones(
                #     num_timesteps,
                #     dtype=torch.float32,
                # )

                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

                try:
                    sampler = flux_sample_loop_progressive(
                        pipe,
                        prompt=prompt,
                        num_inference_steps=num_timesteps,
                        generator=generator,
                        output_type="latent",
                        max_sequence_length=512,
                    )

                    for step_data in sampler:
                        if "final_image" in step_data:
                            continue

                        timestep_idx = step_data["timestep_idx"]
                        score_idx = timestep_idx - 1
                        if correction_mode:
                            transformer_scores, single_scores = (
                                analyzer.step_correct(
                                    transformer_features,
                                    single_features,
                                    transformer_cache_book,
                                    single_cache_book,
                                    timestep_idx,
                                )
                            )
                        else:
                            transformer_scores, single_scores = analyzer.step(
                                transformer_features,
                                single_features,
                            )

                        # Cross-step cache scoring is intentionally disabled:
                        # step_score = analyzer.step_forward(
                        #     model.hidden_states_cache
                        # )
                        # if step_score is not None:
                        #     run_step_scores[score_idx] = step_score.cpu()

                        if transformer_scores is not None:
                            for key in active_transformer_keys:
                                run_transformer_scores[key][score_idx] = (
                                    transformer_scores[key].cpu()
                                )
                            for key in active_single_keys:
                                run_single_scores[key][score_idx] = (
                                    single_scores[key].cpu()
                                )
                finally:
                    for hook in hooks:
                        hook.remove()

                for key in active_transformer_keys:
                    transformer_runs[key].append(
                        run_transformer_scores[key]
                    )
                for key in active_single_keys:
                    single_runs[key].append(run_single_scores[key])

                del transformer_features, single_features, analyzer, hooks
                if track_cuda_memory:
                    torch.cuda.empty_cache()

        if track_cuda_memory:
            peak_gib = (
                torch.cuda.max_memory_allocated(cuda_memory_device)
                / (1024 ** 3)
            )
            print(f"Peak GPU memory for {description}: {peak_gib:.2f} GiB")

        average_transformer_scores = {
            key: torch.stack(values).mean(dim=0)
            for key, values in transformer_runs.items()
        }
        average_single_scores = {
            key: torch.stack(values).mean(dim=0)
            for key, values in single_runs.items()
        }
        return average_transformer_scores, average_single_scores

    model.eval()
    initial_transformer_scores, initial_single_scores = collect_layer_scores(
        description="Layer threshold analysis",
    )
    transformer_cache_book = build_layer_cache_book(
        initial_transformer_scores,
        transformer_thresholds,
        num_transformer_layers,
        protected_prefix=1,
    )
    transformer_cache_book["ip_attn"] = (
        transformer_cache_book["attn"].clone()
    )
    single_transformer_cache_book = build_layer_cache_book(
        initial_single_scores,
        single_thresholds,
        num_single_layers,
        protected_prefix=1,
    )

    avg_transformer_rates, avg_single_transformer_rates = (
        collect_layer_scores(
            description="Layer cache correction",
            transformer_cache_book=transformer_cache_book,
            single_cache_book=single_transformer_cache_book,
        )
    )
    transformer_cache_book = build_layer_cache_book(
        avg_transformer_rates,
        transformer_thresholds,
        num_transformer_layers,
        protected_prefix=num_nonskip,
    )
    transformer_cache_book["ip_attn"] = (
        transformer_cache_book["attn"].clone()
    )
    single_transformer_cache_book = build_layer_cache_book(
        avg_single_transformer_rates,
        single_thresholds,
        num_single_layers,
        protected_prefix=num_nonskip,
    )

    # Cross-step thresholding and step-first policy are intentionally disabled:
    #
    # step_threshold_value = torch.quantile(
    #     avg_step_rates[1:-1],
    #     step_thres,
    # )
    # step_cache_bool = avg_step_rates < step_threshold_value
    # step_cache_bool[:num_nonskip] = False
    # step_cache_bool[-1] = False
    #
    # for timestep_idx in range(num_timesteps):
    #     if step_cache_bool[timestep_idx]:
    #         for cache_book in (
    #             transformer_cache_book,
    #             single_transformer_cache_book,
    #         ):
    #             for module_cache in cache_book.values():
    #                 module_cache[timestep_idx, :] = True
    #                 if timestep_idx + 1 < num_timesteps:
    #                     module_cache[timestep_idx + 1, :] = False

    # DynamicFluxTransformer2DModel expects a step book. Keep the interface
    # compatible while guaranteeing that no whole-step cache is ever used.
    step_cache_bool = torch.zeros(num_timesteps, dtype=torch.bool)

    step_cache_bool = step_cache_bool.tolist()
    for key in transformer_cache_book:
        transformer_cache_book[key] = transformer_cache_book[key].tolist()
    for key in single_transformer_cache_book:
        single_transformer_cache_book[key] = (
            single_transformer_cache_book[key].tolist()
        )

    print_skip_ratio(
        transformer_cache_book,
        single_transformer_cache_book,
        num_transformer_layers,
        num_single_layers,
    )
    return (
        step_cache_bool,
        transformer_cache_book,
        single_transformer_cache_book,
        avg_transformer_rates,
        avg_single_transformer_rates,
    )

def print_skip_ratio(
    transformer_cache_book,
    single_transformer_cache_book,
    num_transformer_layers,
    num_single_layers
    ):
    num_timesteps = len(transformer_cache_book["attn"])

    total_transformer_modules = num_timesteps * num_transformer_layers * 5
    skipped_transformer = 0
    module_skip_counts = {key: 0 for key in transformer_cache_book.keys()}

    for step in range(num_timesteps):
        for block_idx in range(num_transformer_layers):
            if (
                transformer_cache_book["attn"][step][block_idx]
                and transformer_cache_book["context_attn"][step][block_idx]
            ):
                module_skip_counts["attn"] += 1
                module_skip_counts["context_attn"] += 1
                module_skip_counts["ip_attn"] += 1
                skipped_transformer += 3
            if transformer_cache_book["ff"][step][block_idx]:
                module_skip_counts["ff"] += 1
                skipped_transformer += 1
            if transformer_cache_book["context_ff"][step][block_idx]:
                module_skip_counts["context_ff"] += 1
                skipped_transformer += 1

    total_single_transformer_modules = num_timesteps * num_single_layers * 2
    skipped_single = 0
    single_module_skip_counts = {key: 0 for key in single_transformer_cache_book.keys()}

    for module_name in single_transformer_cache_book.keys():
        for step in range(num_timesteps):
            skipped_count = sum(single_transformer_cache_book[module_name][step])
            single_module_skip_counts[module_name] += skipped_count
            skipped_single += skipped_count

    total_modules = total_transformer_modules + total_single_transformer_modules
    total_skipped = skipped_transformer + skipped_single
    transformer_skip_ratio = (skipped_transformer / total_transformer_modules) * 100
    single_transformer_skip_ratio = (skipped_single / total_single_transformer_modules) * 100
    skip_ratio = (total_skipped / total_modules) * 100

    print(f"Total skip ratio: {skip_ratio:.2f}%")
    print(f"\nTotal transformer blocks skip ratio:{transformer_skip_ratio:.2f}%")
    print(f"Transformer blocks skip details:")
    for module_name, count in module_skip_counts.items():
        ratio = (count / (num_timesteps * num_transformer_layers)) * 100
        print(f"  {module_name:15s}: {ratio:5.2f}%")

    print(f"\nTotal single transformer blocks skip ratio:{single_transformer_skip_ratio:.2f}%")
    print(f"Single Transformer blocks skip details:")
    for module_name, count in single_module_skip_counts.items():
        ratio = (count / (num_timesteps * num_single_layers)) * 100
        print(f"  {module_name:15s}: {ratio:5.2f}%")

def cache_book_config(
    num_inference_steps,
    nonskip_rate,
    attn_thres,
    context_attn_thres,
    ff_thres,
    context_ff_thres,
    single_attn_thres,
    single_mlp_thres,
):
    return {
        "policy": POLICY_VARIANT,
        "steps": num_inference_steps,
        "nonskip": nonskip_rate,
        "attn": attn_thres,
        "context_attn": context_attn_thres,
        "ff": ff_thres,
        "context_ff": context_ff_thres,
        "single_attn": single_attn_thres,
        "single_mlp": single_mlp_thres,
    }


def cache_book_name(config):
    fmt = lambda value: format(float(value), "g")
    return (
        f"cache_book_layer_steps{config['steps']}_ns{fmt(config['nonskip'])}"
        f"_attnth{fmt(config['attn'])}_ffth{fmt(config['ff'])}"
        f"_cattnth{fmt(config['context_attn'])}"
        f"_ctxffth{fmt(config['context_ff'])}"
        f"_sattnth{fmt(config['single_attn'])}"
        f"_smlpth{fmt(config['single_mlp'])}.json"
    )

def load_cache_books(
    cache_book_path,
    cache_book_file,
    expected_rate_method=None,
    expected_cache_scope=None,
    expected_config=None,
):
    with open(os.path.join(cache_book_path, cache_book_file), "r") as file:
        cache_books = json.load(file)

    if cache_books.get("cache_version") != CACHE_BOOK_VERSION:
        raise ValueError("Incompatible cache book; re-run calibration.")
    saved_rate_method = cache_books.get("rate_method")
    if (
        expected_rate_method is not None
        and saved_rate_method != expected_rate_method
    ):
        raise ValueError(
            "Cache-book rate method mismatch: "
            f"expected {expected_rate_method!r}, "
            f"found {saved_rate_method!r}."
        )

    saved_cache_scope = cache_books.get("cache_scope")
    if (
        expected_cache_scope is not None
        and saved_cache_scope != expected_cache_scope
    ):
        raise ValueError(
            "Cache-book scope mismatch: "
            f"expected {expected_cache_scope!r}, "
            f"found {saved_cache_scope!r}."
        )

    if expected_config is not None and cache_books.get("config") != expected_config:
        raise ValueError(
            "Cache-book configuration mismatch: "
            f"expected {expected_config!r}, found {cache_books.get('config')!r}."
        )

    step_cache_book = cache_books["step_cache_book"]
    if any(step_cache_book):
        raise ValueError(
            "Layer-only cache book contains enabled cross-step decisions."
        )

    return (
        step_cache_book,
        cache_books["transformer_cache_book"],
        cache_books["single_transformer_cache_book"],
    )

def main(args):
    seed = args.seed
    torch.set_grad_enabled(False)

    print("Loading FLUX pipeline...")
    print("Cache scope: layer only")
    print(f"Rate method: {RATE_METHOD}")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    original_transformer = pipe.transformer

    num_inference_steps = args.num_inference_steps
    nonskip_rate = args.nonskip_rate
    # Cross-step cache threshold is intentionally disabled:
    # step_thres = 0

    attn_thres = args.attn_thres
    context_attn_thres = args.context_attn_thres
    ff_thres = args.ff_thres
    context_ff_thres = args.context_ff_thres

    Single_attn_thres = args.single_attn_thres
    Single_mlp_thres = args.single_mlp_thres

    cache_config = cache_book_config(
        num_inference_steps,
        nonskip_rate,
        attn_thres,
        context_attn_thres,
        ff_thres,
        context_ff_thres,
        Single_attn_thres,
        Single_mlp_thres,
    )
    cache_book_path = args.cache_book_path
    cache_book_file = args.cache_book_file or cache_book_name(cache_config)
    cache_book_full_path = os.path.join(cache_book_path, cache_book_file)

    dynamic_model = DynamicFluxTransformer2DModel(
        original_transformer,
        num_inference_steps,
    )
    dynamic_model.eval()
    pipe.transformer = dynamic_model

    run_calibration = args.generate_cache_books
    calibration_model_cpu_offload = True

    if run_calibration:
        if calibration_model_cpu_offload:
            print("Calibration model CPU offload: enabled")
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            pipe.to("cuda")
        ## There is no necessary correlation between the measure prompts and the test prompts.
        measure_prompts = [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            # "A futuristic cityscape with flying cars and neon lights.",
            # "An astronaut riding a horse on the moon.",
            # "A bouquet of wildflowers in a glass vase, watercolor style.",
            # "A majestic lion sitting on a rock, golden mane, sunset.",
            # "A serene landscape with mountains and a lake at sunset.",
            # "A cute cat playing with a ball of yarn.",
            # "A portrait of a woman in Renaissance style, oil painting.",
            # "A dragon flying over a burning village, epic fantasy.",
            # "A steaming cup of coffee on a wooden table, cozy morning.",
            # "A cyberpunk street with rain and neon signs, night.",
            # "A tropical beach at sunrise, palm trees, golden hour.",
            # "A magical library with floating books and glowing runes.",
            # "A vintage camera surrounded by old photographs, nostalgic mood.",
            # "A colorful parrot flying in a rainforest, vibrant colors.",
            # "A medieval castle on a hill, surrounded by fog.",
            # "A young woman in a traditional Japanese kimono, cherry blossoms.",
            # "A futuristic female cyborg with glowing blue eyes, silver armor.",
            # "A steaming bowl of ramen on a wooden counter, food photography.",
            # "A snowy forest with sunlight filtering through the trees.",
        ]
        # Single-prompt calibration is the default for low-cost threshold tuning.
        measure_prompts = [args.calibration_prompt]
        step_cache_book, transformer_cache_book, single_transformer_cache_book, \
        avg_transformer_rates, avg_single_transformer_rates = threshold_analyse(
            model=dynamic_model,
            pipe=pipe,
            measure_prompts=measure_prompts,
            nonskip_rate=nonskip_rate,
            # Cross-step cache threshold is intentionally disabled:
            # step_thres=step_thres,
            attn_thres=attn_thres,
            context_attn_thres=context_attn_thres,
            ff_thres=ff_thres,
            context_ff_thres=context_ff_thres,
            Single_attn_thres=Single_attn_thres,
            Single_mlp_thres=Single_mlp_thres,
            seed=seed
        )
        cache_books = {
            "cache_version": CACHE_BOOK_VERSION,
            "cache_scope": "layer_only",
            "config": cache_config,
            "rate_method": RATE_METHOD,
            "step_cache_book": step_cache_book,
            "transformer_cache_book": transformer_cache_book,
            "single_transformer_cache_book": single_transformer_cache_book
        }
        os.makedirs(cache_book_path, exist_ok=True)
        with open(cache_book_full_path, "w") as f:
            json.dump(cache_books, f, indent=2)

        print(f"\nCache books saved: {cache_book_full_path}")
        if args.calibration_only:
            return
        # Keep the standalone CLI convenient: a no-argument fast run performs
        # calibration once and then re-enters the normal generation path.
        args.generate_cache_books = False
        return main(args)

    else:
        step_cache_book, transformer_cache_book, single_transformer_cache_book = load_cache_books(
            cache_book_path=cache_book_path,
            cache_book_file=cache_book_file,
            expected_rate_method=RATE_METHOD,
            expected_cache_scope="layer_only",
            expected_config=cache_config,
        )
        pipe.to("cuda")
        dynamic_model.init_cache_book(transformer_cache_book, single_transformer_cache_book, step_cache_book)
        pipe.transformer = dynamic_model

        prompts = [
            "a photo of a broccoli",
            "A snowy mountain village at dusk, glowing windows and smoke rising.",
            "A golden retriever puppy jumping through autumn leaves.",
            "A surreal underwater city with glowing jellyfish and crystal towers.",
            "A group of astronauts planting a flag on Mars, red rocky landscape.",
            "A vintage sports car speeding down a coastal highway at sunset.",
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
            # "The grand, opulent lobby of an Art Deco skyscraper. Polished brass, geometric patterns on the marble floor, and a massive, intricate chandelier casting warm light. Cinematic, symmetrical, elegant, 1920s style.",
            # "A detailed portrait of a vibrant, colorful toucan perched on a mossy branch. The background is a lush, out-of-focus jungle with soft morning light filtering through the leaves. Photorealistic, natural style, high detail, shallow depth of field.",
            # "A steampunk robot reading a book in a Victorian library.",
            # "Macro portrait of a silver tabby cat with vivid green eyes, crisp whiskers, rich fur texture, shallow depth of field,\
            #     cyberpunk megacity at night, rain-slick streets reflecting vivid neon signs and holograms. Flying cars, towering glass and steel facades, cinematic wide angle, high contrast.",
            # "a photo of a white sandwich",
            # "a photo of a person"
        ]
        prompts = [args.prompt]
        images = []
        times = []

        for prompt in prompts:
            start_time = time.time()
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(seed)
                ).images[0]
            times.append(time.time() - start_time)
            images.append(image)

        if len(prompts)>1:
            times = np.array(times[1:])
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"Sampling time: {avg_time:.4f}±{std_time:.4f} s")
        else:
            print(f"Sampling time: {times[0]:.4f} s")

        width, height = images[0].size
        combined = Image.new("RGB", (width * len(images), height))
        for idx, img in enumerate(images):
            combined.paste(img, (idx * width, 0))

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%m%d_%H%M%S")
        save_name = (
            f"{args.output_dir}/imgs_{POLICY_VARIANT}_stp{num_inference_steps}"
            f"_n{nonskip_rate}"
            f"_attn{attn_thres}_ff{ff_thres}"
            f"_cattn{context_attn_thres}"
            f"_ctxff{context_ff_thres}"
            f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}"
            f"_{timestamp}.png"
        )
        combined.save(save_name)
        print(f"Images saved to {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX layer-only sampler")
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/InvarDiff/FLUX")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A cinematic photograph of a red fox sitting beside a moss-covered tree in a sunlit forest, natural colors, detailed fur, soft depth of field.")
    parser.add_argument("--calibration-prompt", default="A cinematic photograph of a red fox sitting beside a moss-covered tree in a sunlit forest, natural colors, detailed fur, soft depth of field.")
    parser.add_argument("--output-dir", default="images")
    parser.add_argument("--cache-book-path", default="./cache_books")
    parser.add_argument("--cache-book-file", default=None)
    parser.add_argument("--nonskip-rate", type=float, default=0.1)
    # fast (default): attn=0.30, context_attn=0.30, single_attn=0.40,
    # ff=0.00, context_ff=0.00, single_mlp=0.00;
    # balanced: attn=0.30, context_attn=0.30, single_attn=0.12,
    # ff=0.22, context_ff=0.40, single_mlp=0.30;
    # slow: attn=0.30, context_attn=0.30, single_attn=0.10,
    # ff=0.00, context_ff=0.00, single_mlp=0.00.
    parser.add_argument("--attn-thres", type=float, default=0.30)
    parser.add_argument("--context-attn-thres", type=float, default=0.30)
    parser.add_argument("--ff-thres", type=float, default=0.00)
    parser.add_argument("--context-ff-thres", type=float, default=0.00)
    parser.add_argument("--single-attn-thres", type=float, default=0.40)
    parser.add_argument("--single-mlp-thres", type=float, default=0.00)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--generate-cache-books", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    main(parser.parse_args())
