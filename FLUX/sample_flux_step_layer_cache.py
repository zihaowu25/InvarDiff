"""Standalone FLUX InvarDiff with full three-point cache correction.

Both cross-step and layer-level scores use a three-point L1 displacement ratio.
The second calibration pass corrects and rebuilds both policies. This file is
standalone apart from the shared dynamic FLUX runtime.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TRANSFORMER_MODULES = (
    "attn",
    "context_attn",
    "ip_attn",
    "ff",
    "context_ff",
)
SINGLE_TRANSFORMER_MODULES = ("attn", "mlp")
POLICY_VARIANT = "fullcorr_threepoint_l1"
RATE_METHOD = "three_point_l1"
RATE_CHUNK_SIZE = 1_048_576


TRANSFORMER_RATE_MODULES = tuple(
    key for key in TRANSFORMER_MODULES if key != "ip_attn"
)


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


def register_hooks(model, transformer_keys, single_transformer_keys):
    """Register hooks only for modules with non-zero cache thresholds."""
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
    """Three-point analyzer for raw and cache-corrected calibration passes."""

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
            key: [False] * num_transformer_layers
            for key in self.transformer_keys
        }
        self.SingleTransformer_state = {
            key: [False] * num_single_layers
            for key in self.single_transformer_keys
        }
        self._step_count = 0
        self._layer_count = 0
        self.step_state = False

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

    def step_forward(self, hidden_states):
        """Collect an uncorrected cross-step three-point score."""
        hidden_states = hidden_states.detach()
        if self._step_count == 0:
            self.current_hidden_states = hidden_states
            self._step_count = 1
            return None

        if self._step_count == 1:
            self.prev_hidden_states = self.current_hidden_states
            self.current_hidden_states = hidden_states
            self.prev_step_norm = compute_l1_distance(
                self.prev_hidden_states,
                self.current_hidden_states,
            )
            self._step_count = 2
            return None

        step_score = compute_rate(
            self.prev_hidden_states,
            self.current_hidden_states,
            hidden_states,
            self.prev_step_norm,
        )
        next_step_norm = compute_l1_distance(
            self.current_hidden_states,
            hidden_states,
        )
        self.prev_hidden_states = self.current_hidden_states
        self.current_hidden_states = hidden_states
        self.prev_step_norm = next_step_norm
        return step_score

    def step_forward_correct(
        self,
        hidden_states: torch.Tensor,
        step_cache_state: List[bool],
        timestep_idx: int,
    ) -> Optional[torch.Tensor]:
        """Collect a cache-path-corrected cross-step three-point score."""
        hidden_states = hidden_states.detach()
        if timestep_idx == 0:
            self.current_hidden_states = hidden_states
            return None

        if timestep_idx == 1:
            self.prev_hidden_states = self.current_hidden_states
            self.current_hidden_states = hidden_states
            self.prev_step_norm = compute_l1_distance(
                self.prev_hidden_states,
                self.current_hidden_states,
            )
            return None

        step_score = compute_rate(
            self.prev_hidden_states,
            self.current_hidden_states,
            hidden_states,
            self.prev_step_norm,
        )

        policy_idx = timestep_idx - 1
        should_refresh = False
        if not step_cache_state[policy_idx]:
            self.step_state = False
            should_refresh = True
        elif not self.step_state:
            self.step_state = True
            should_refresh = True

        if should_refresh:
            next_step_norm = compute_l1_distance(
                self.current_hidden_states,
                hidden_states,
            )
            self.prev_hidden_states = self.current_hidden_states
            self.current_hidden_states = hidden_states
            self.prev_step_norm = next_step_norm

        return step_score

    def _start_layer_state(self, transformer_features, single_features):
        self.current_Transformer = transformer_features
        self.current_SingleTransformer = single_features
        self._layer_count = 1

    def _initialize_layer_state(self, transformer_features, single_features):
        self.prev_Transformer = self.current_Transformer
        self.prev_SingleTransformer = self.current_SingleTransformer
        self.current_Transformer = transformer_features
        self.current_SingleTransformer = single_features

        self.prev_Transformer_norm = {
            key: [None] * self.num_transformer_layers
            for key in self.transformer_keys
        }
        for key in self.transformer_keys:
            for block_idx, feature in enumerate(transformer_features[key]):
                previous = self.prev_Transformer[key][block_idx]
                if previous is not None and feature is not None:
                    self.prev_Transformer_norm[key][block_idx] = (
                        compute_l1_distance(previous, feature)
                    )

        self.prev_SingleTransformer_norm = {
            key: [None] * self.num_single_layers
            for key in self.single_transformer_keys
        }
        for key in self.single_transformer_keys:
            for block_idx, feature in enumerate(single_features[key]):
                previous = self.prev_SingleTransformer[key][block_idx]
                if previous is not None and feature is not None:
                    self.prev_SingleTransformer_norm[key][block_idx] = (
                        compute_l1_distance(previous, feature)
                    )

        self._layer_count = 2

    @staticmethod
    def _should_refresh(cache_state, active_state, key, timestep_idx, block_idx):
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
                previous = self.prev_Transformer[key][block_idx]
                current = self.current_Transformer[key][block_idx]
                if previous is None or current is None or feature is None:
                    scores.append(float("nan"))
                    next_norm = None
                else:
                    score = compute_rate(
                        previous,
                        current,
                        feature,
                        self.prev_Transformer_norm[key][block_idx],
                    )
                    scores.append(score.item())
                    next_norm = compute_l1_distance(current, feature)

                self.prev_Transformer[key][block_idx] = current
                self.current_Transformer[key][block_idx] = feature
                self.prev_Transformer_norm[key][block_idx] = next_norm

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                previous = self.prev_SingleTransformer[key][block_idx]
                current = self.current_SingleTransformer[key][block_idx]
                if previous is None or current is None or feature is None:
                    scores.append(float("nan"))
                    next_norm = None
                else:
                    score = compute_rate(
                        previous,
                        current,
                        feature,
                        self.prev_SingleTransformer_norm[key][block_idx],
                    )
                    scores.append(score.item())
                    next_norm = compute_l1_distance(current, feature)

                self.prev_SingleTransformer[key][block_idx] = current
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
        """Collect layer scores while following provisional cache paths."""
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
                previous = self.prev_Transformer[key][block_idx]
                current = self.current_Transformer[key][block_idx]
                if previous is None or current is None or feature is None:
                    scores.append(float("nan"))
                else:
                    score = compute_rate(
                        previous,
                        current,
                        feature,
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
                    next_norm = (
                        None
                        if current is None or feature is None
                        else compute_l1_distance(current, feature)
                    )
                    self.prev_Transformer[key][block_idx] = current
                    self.current_Transformer[key][block_idx] = feature
                    self.prev_Transformer_norm[key][block_idx] = next_norm

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                previous = self.prev_SingleTransformer[key][block_idx]
                current = self.current_SingleTransformer[key][block_idx]
                if previous is None or current is None or feature is None:
                    scores.append(float("nan"))
                else:
                    score = compute_rate(
                        previous,
                        current,
                        feature,
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
                    next_norm = (
                        None
                        if current is None or feature is None
                        else compute_l1_distance(current, feature)
                    )
                    self.prev_SingleTransformer[key][block_idx] = current
                    self.current_SingleTransformer[key][block_idx] = feature
                    self.prev_SingleTransformer_norm[key][block_idx] = next_norm

            single_scores[key] = torch.tensor(scores)

        return transformer_scores, single_scores

def _empty_transformer_scores(
    num_timesteps: int,
    num_layers: int,
    keys,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.ones(num_timesteps, num_layers, dtype=torch.float32)
        for key in keys
    }


def _empty_single_transformer_scores(
    num_timesteps: int,
    num_layers: int,
    keys,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.ones(num_timesteps, num_layers, dtype=torch.float32)
        for key in keys
    }


def _average_score_dict(
    score_runs: Dict[str, List[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.stack(values).mean(dim=0)
        for key, values in score_runs.items()
    }


def _collect_scores(
    model,
    pipe,
    measure_prompts: List[str],
    seed: int,
    description: str,
    transformer_keys,
    single_transformer_keys,
    step_cache_state: Optional[List[bool]] = None,
    transformer_cache_state: Optional[Dict[str, List[List[bool]]]] = None,
    single_cache_state: Optional[Dict[str, List[List[bool]]]] = None,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """Collect averaged three-point scores for one calibration phase."""
    correction_mode = step_cache_state is not None
    if correction_mode and (
        transformer_cache_state is None or single_cache_state is None
    ):
        raise ValueError("Correction mode requires all provisional cache books.")

    transformer_keys = tuple(transformer_keys)
    single_transformer_keys = tuple(single_transformer_keys)
    device = pipe._execution_device
    num_timesteps = model.num_timesteps
    num_transformer_layers = model.num_layers
    num_single_layers = model.num_single_layers

    step_runs: List[torch.Tensor] = []
    transformer_runs = {key: [] for key in transformer_keys}
    single_runs = {key: [] for key in single_transformer_keys}

    track_cuda_memory = torch.cuda.is_available()
    cuda_memory_device = (
        torch.cuda.current_device() if track_cuda_memory else None
    )
    if track_cuda_memory:
        torch.cuda.reset_peak_memory_stats(cuda_memory_device)

    model.eval()
    with torch.inference_mode():
        for prompt in tqdm(measure_prompts, desc=description):
            transformer_features, single_features, hooks = register_hooks(
                model,
                transformer_keys,
                single_transformer_keys,
            )
            analyzer = FeatureChangeAnalyzer(
                num_transformer_layers,
                num_single_layers,
                transformer_keys,
                single_transformer_keys,
            )

            run_step_scores = torch.ones(num_timesteps, dtype=torch.float32)
            run_transformer_scores = _empty_transformer_scores(
                num_timesteps,
                num_transformer_layers,
                transformer_keys,
            )
            run_single_scores = _empty_single_transformer_scores(
                num_timesteps,
                num_single_layers,
                single_transformer_keys,
            )

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
                        transformer_scores, single_scores = analyzer.step_correct(
                            transformer_features,
                            single_features,
                            transformer_cache_state,
                            single_cache_state,
                            timestep_idx,
                        )
                        step_score = analyzer.step_forward_correct(
                            model.hidden_states_cache,
                            step_cache_state,
                            timestep_idx,
                        )
                    else:
                        transformer_scores, single_scores = analyzer.step(
                            transformer_features,
                            single_features,
                        )
                        step_score = analyzer.step_forward(
                            model.hidden_states_cache,
                        )

                    if step_score is not None:
                        run_step_scores[score_idx] = step_score.cpu()

                    if transformer_scores is not None:
                        for key in transformer_keys:
                            run_transformer_scores[key][score_idx] = (
                                transformer_scores[key].cpu()
                            )
                        for key in single_transformer_keys:
                            run_single_scores[key][score_idx] = (
                                single_scores[key].cpu()
                            )
            finally:
                for hook in hooks:
                    hook.remove()

            step_runs.append(run_step_scores)
            for key in transformer_keys:
                transformer_runs[key].append(run_transformer_scores[key])
            for key in single_transformer_keys:
                single_runs[key].append(run_single_scores[key])

            del transformer_features, single_features, analyzer, hooks
            if track_cuda_memory:
                torch.cuda.empty_cache()

    if track_cuda_memory:
        peak_gib = (
            torch.cuda.max_memory_allocated(cuda_memory_device) / (1024 ** 3)
        )
        print(f"Peak GPU memory for {description}: {peak_gib:.2f} GiB")

    return (
        torch.stack(step_runs).mean(dim=0),
        _average_score_dict(transformer_runs),
        _average_score_dict(single_runs),
    )

def _build_step_cache_book(
    step_scores: torch.Tensor,
    step_thres: float,
    num_nonskip: int,
) -> torch.Tensor:
    threshold = torch.quantile(step_scores[1:-1], step_thres)
    cache_book = step_scores < threshold
    cache_book[:num_nonskip] = False
    cache_book[-1] = False
    return cache_book


def _build_module_cache_books(
    transformer_scores: Dict[str, torch.Tensor],
    single_scores: Dict[str, torch.Tensor],
    attn_thres: float,
    ff_thres: float,
    context_ff_thres: float,
    single_attn_thres: float,
    single_mlp_thres: float,
    num_reserved_steps: int,
    num_timesteps: int,
    num_transformer_layers: int,
    num_single_layers: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    transformer_thresholds = {
        "attn": attn_thres,
        "context_attn": attn_thres,
        "ff": ff_thres,
        "context_ff": context_ff_thres,
    }
    single_thresholds = {
        "attn": single_attn_thres,
        "mlp": single_mlp_thres,
    }

    for module_name, threshold in {
        **transformer_thresholds,
        **{f"single_{key}": value for key, value in single_thresholds.items()},
    }.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {module_name!r} must be in [0, 1], got {threshold}."
            )

    transformer_books: Dict[str, torch.Tensor] = {}
    for module_name, threshold_q in transformer_thresholds.items():
        if threshold_q == 0.0:
            cache_book = torch.zeros(
                num_timesteps,
                num_transformer_layers,
                dtype=torch.bool,
            )
        else:
            threshold = torch.quantile(
                transformer_scores[module_name][1:-1],
                threshold_q,
            )
            cache_book = transformer_scores[module_name] < threshold

        cache_book[:num_reserved_steps, :] = False
        cache_book[-1, :] = False
        transformer_books[module_name] = cache_book

    transformer_books["ip_attn"] = transformer_books["attn"].clone()

    single_books: Dict[str, torch.Tensor] = {}
    for module_name, threshold_q in single_thresholds.items():
        if threshold_q == 0.0:
            cache_book = torch.zeros(
                num_timesteps,
                num_single_layers,
                dtype=torch.bool,
            )
        else:
            threshold = torch.quantile(
                single_scores[module_name][1:-1],
                threshold_q,
            )
            cache_book = single_scores[module_name] < threshold

        cache_book[:num_reserved_steps, :] = False
        cache_book[-1, :] = False
        single_books[module_name] = cache_book

    return transformer_books, single_books

def _apply_step_first_policy(
    step_cache_book: torch.Tensor,
    transformer_cache_book: Dict[str, torch.Tensor],
    single_transformer_cache_book: Dict[str, torch.Tensor],
) -> None:
    """Give whole-step caching priority and refresh every module next step."""

    num_timesteps = int(step_cache_book.shape[0])
    for timestep_idx in range(num_timesteps):
        if not step_cache_book[timestep_idx]:
            continue

        for cache_book in transformer_cache_book.values():
            cache_book[timestep_idx, :] = True
            if timestep_idx + 1 < num_timesteps:
                cache_book[timestep_idx + 1, :] = False

        for cache_book in single_transformer_cache_book.values():
            cache_book[timestep_idx, :] = True
            if timestep_idx + 1 < num_timesteps:
                cache_book[timestep_idx + 1, :] = False


def _books_to_lists(
    step_cache_book: torch.Tensor,
    transformer_cache_book: Dict[str, torch.Tensor],
    single_transformer_cache_book: Dict[str, torch.Tensor],
) -> Tuple[
    List[bool],
    Dict[str, List[List[bool]]],
    Dict[str, List[List[bool]]],
]:
    return (
        step_cache_book.to(dtype=torch.bool).cpu().tolist(),
        {
            key: value.to(dtype=torch.bool).cpu().tolist()
            for key, value in transformer_cache_book.items()
        },
        {
            key: value.to(dtype=torch.bool).cpu().tolist()
            for key, value in single_transformer_cache_book.items()
        },
    )


def _active_rate_keys(thresholds):
    for module_name, threshold in thresholds.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {module_name!r} must be in [0, 1], got {threshold}."
            )
    return tuple(
        key for key, threshold in thresholds.items() if threshold > 0.0
    )

def threshold_analyse(
    model,
    pipe,
    measure_prompts,
    nonskip_rate=0.1,
    step_thres=0.5,
    attn_thres=0.5,
    ff_thres=0.5,
    context_ff_thres=0.5,
    Single_attn_thres=0.5,
    Single_mlp_thres=0.5,
    seed=42,
):
    """Run full step-level and layer-level resampling calibration."""

    num_timesteps = model.num_timesteps
    num_transformer_layers = model.num_layers
    num_single_layers = model.num_single_layers
    num_nonskip = max(1, int(nonskip_rate * num_timesteps))

    transformer_thresholds = {
        "attn": attn_thres,
        "context_attn": attn_thres,
        "ff": ff_thres,
        "context_ff": context_ff_thres,
    }
    single_thresholds = {
        "attn": Single_attn_thres,
        "mlp": Single_mlp_thres,
    }
    active_transformer_keys = _active_rate_keys(transformer_thresholds)
    active_single_keys = _active_rate_keys(single_thresholds)

    print(
        f"Rate method: {RATE_METHOD}\n"
        f"Active transformer scores: {active_transformer_keys}\n"
        f"Active single-transformer scores: {active_single_keys}\n"
        f"Transformer blocks numbers: {num_transformer_layers}\n"
        f"Single Transformer blocks numbers: {num_single_layers}"
    )

    # Phase 1: complete-trajectory statistics and provisional policies.
    (
        initial_step_scores,
        initial_transformer_scores,
        initial_single_scores,
    ) = _collect_scores(
        model,
        pipe,
        measure_prompts,
        seed,
        description="Threshold analysis",
        transformer_keys=active_transformer_keys,
        single_transformer_keys=active_single_keys,
    )

    provisional_step_book = _build_step_cache_book(
        initial_step_scores,
        step_thres,
        num_nonskip,
    )
    provisional_transformer_books, provisional_single_books = (
        _build_module_cache_books(
            initial_transformer_scores,
            initial_single_scores,
            attn_thres,
            ff_thres,
            context_ff_thres,
            Single_attn_thres,
            Single_mlp_thres,
            # Preserve the baseline phase-1 behavior for a controlled
            # comparison: only the first module step is forced to recompute.
            num_reserved_steps=1,
            num_timesteps=num_timesteps,
            num_transformer_layers=num_transformer_layers,
            num_single_layers=num_single_layers,
        )
    )

    # Phase 2: recompute both step and module scores with cache-aware reference
    # updates.  The baseline FLUX implementation only corrects module scores.
    (
        corrected_step_scores,
        corrected_transformer_scores,
        corrected_single_scores,
    ) = _collect_scores(
        model,
        pipe,
        measure_prompts,
        seed,
        description="Step + layer cache correction",
        transformer_keys=active_transformer_keys,
        single_transformer_keys=active_single_keys,
        step_cache_state=provisional_step_book.tolist(),
        transformer_cache_state={
            key: value.tolist()
            for key, value in provisional_transformer_books.items()
        },
        single_cache_state={
            key: value.tolist()
            for key, value in provisional_single_books.items()
        },
    )

    final_step_book = _build_step_cache_book(
        corrected_step_scores,
        step_thres,
        num_nonskip,
    )
    final_transformer_books, final_single_books = _build_module_cache_books(
        corrected_transformer_scores,
        corrected_single_scores,
        attn_thres,
        ff_thres,
        context_ff_thres,
        Single_attn_thres,
        Single_mlp_thres,
        num_reserved_steps=num_nonskip,
        num_timesteps=num_timesteps,
        num_transformer_layers=num_transformer_layers,
        num_single_layers=num_single_layers,
    )

    _apply_step_first_policy(
        final_step_book,
        final_transformer_books,
        final_single_books,
    )

    (
        step_cache_book,
        transformer_cache_book,
        single_transformer_cache_book,
    ) = _books_to_lists(
        final_step_book,
        final_transformer_books,
        final_single_books,
    )

    print_skip_ratio(
        transformer_cache_book,
        single_transformer_cache_book,
        num_transformer_layers,
        num_single_layers,
    )

    # Return the same cache-book and score structure as the other FLUX samplers.
    return (
        step_cache_book,
        transformer_cache_book,
        single_transformer_cache_book,
        corrected_transformer_scores,
        corrected_single_scores,
    )


def print_skip_ratio(
    transformer_cache_book,
    single_transformer_cache_book,
    num_transformer_layers,
    num_single_layers,
):
    num_timesteps = len(transformer_cache_book["attn"])
    total_transformer = num_timesteps * num_transformer_layers * 5
    total_single = num_timesteps * num_single_layers * 2
    transformer_counts = {
        key: 0 for key in transformer_cache_book
    }
    single_counts = {
        key: 0 for key in single_transformer_cache_book
    }
    skipped_transformer = 0

    for timestep_idx in range(num_timesteps):
        for block_idx in range(num_transformer_layers):
            if (
                transformer_cache_book["attn"][timestep_idx][block_idx]
                and transformer_cache_book["context_attn"][timestep_idx][block_idx]
            ):
                transformer_counts["attn"] += 1
                transformer_counts["context_attn"] += 1
                transformer_counts["ip_attn"] += 1
                skipped_transformer += 3
            if transformer_cache_book["ff"][timestep_idx][block_idx]:
                transformer_counts["ff"] += 1
                skipped_transformer += 1
            if transformer_cache_book["context_ff"][timestep_idx][block_idx]:
                transformer_counts["context_ff"] += 1
                skipped_transformer += 1

    skipped_single = 0
    for module_name, cache_book in single_transformer_cache_book.items():
        count = sum(sum(step) for step in cache_book)
        single_counts[module_name] = count
        skipped_single += count

    total_skipped = skipped_transformer + skipped_single
    print(
        f"Total skip ratio: "
        f"{100.0 * total_skipped / (total_transformer + total_single):.2f}%"
    )
    print(
        "\nTotal transformer blocks skip ratio:"
        f"{100.0 * skipped_transformer / total_transformer:.2f}%"
    )
    print("Transformer blocks skip details:")
    for module_name, count in transformer_counts.items():
        ratio = 100.0 * count / (num_timesteps * num_transformer_layers)
        print(f"  {module_name:15s}: {ratio:5.2f}%")

    print(
        "\nTotal single transformer blocks skip ratio:"
        f"{100.0 * skipped_single / total_single:.2f}%"
    )
    print("Single Transformer blocks skip details:")
    for module_name, count in single_counts.items():
        ratio = 100.0 * count / (num_timesteps * num_single_layers)
        print(f"  {module_name:15s}: {ratio:5.2f}%")


def load_cache_books(
    cache_book_path,
    cache_book_file,
    expected_rate_method=None,
):
    with open(os.path.join(cache_book_path, cache_book_file), "r") as file:
        cache_books = json.load(file)

    saved_rate_method = cache_books.get("rate_method")
    if (
        expected_rate_method is not None
        and saved_rate_method != expected_rate_method
    ):
        raise ValueError(
            "Cache-book rate method mismatch: "
            f"expected {expected_rate_method!r}, found {saved_rate_method!r}."
        )

    return (
        cache_books["step_cache_book"],
        cache_books["transformer_cache_book"],
        cache_books["single_transformer_cache_book"],
    )

def _cache_book_name(
    num_inference_steps,
    nonskip_rate,
    step_thres,
    attn_thres,
    ff_thres,
    context_ff_thres,
    single_attn_thres,
    single_mlp_thres,
):
    return (
        f"cache_books_{POLICY_VARIANT}_stp{num_inference_steps}"
        f"_n{nonskip_rate}_th{step_thres}"
        f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
        f"_sattn{single_attn_thres}_smlp{single_mlp_thres}.json"
    )


def main():
    seed = 42
    torch.set_grad_enabled(False)

    print("Loading FLUX pipeline...")
    print("Calibration variant: three-point step + layer correction")
    print(f"Rate method: {RATE_METHOD}")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/root/autodl-tmp/InvarDiff/FLUX",
    )
    original_transformer = pipe.transformer

    # Experiment controls are local to this standalone comparison script.
    num_inference_steps = 28
    nonskip_rate = 0.1
    step_thres = 0.6

    attn_thres = 0.5
    ff_thres = 0.5
    context_ff_thres = 0.5
    single_attn_thres = 0.5
    single_mlp_thres = 0.5

    dynamic_model = DynamicFluxTransformer2DModel(
        original_transformer,
        num_inference_steps,
    )
    dynamic_model.eval()
    pipe.transformer = dynamic_model

    # Match the baseline debugging style: switch this flag to True to generate
    # the corrected cache book, then set it back to False for timed inference.
    run_calibration = 0
    calibration_model_cpu_offload = True

    cache_book_path = "./cache_books"
    cache_book_file = _cache_book_name(
        num_inference_steps,
        nonskip_rate,
        step_thres,
        attn_thres,
        ff_thres,
        context_ff_thres,
        single_attn_thres,
        single_mlp_thres,
    )
    cache_book_full_path = os.path.join(cache_book_path, cache_book_file)

    if run_calibration:
        if calibration_model_cpu_offload:
            print("Calibration model CPU offload: enabled")
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            pipe.to("cuda")
        measure_prompts = [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            # "A futuristic cityscape with flying cars and neon lights.",
            # "An astronaut riding a horse on the moon.",
            # "A bouquet of wildflowers in a glass vase, watercolor style.",
            # "A majestic lion sitting on a rock, golden mane, sunset.",
        ]

        (
            step_cache_book,
            transformer_cache_book,
            single_transformer_cache_book,
            _,
            _,
        ) = threshold_analyse(
            model=dynamic_model,
            pipe=pipe,
            measure_prompts=measure_prompts,
            nonskip_rate=nonskip_rate,
            step_thres=step_thres,
            attn_thres=attn_thres,
            ff_thres=ff_thres,
            context_ff_thres=context_ff_thres,
            Single_attn_thres=single_attn_thres,
            Single_mlp_thres=single_mlp_thres,
            seed=seed,
        )

        cache_books = {
            "rate_method": RATE_METHOD,
            "step_cache_book": step_cache_book,
            "transformer_cache_book": transformer_cache_book,
            "single_transformer_cache_book": single_transformer_cache_book,
        }
        os.makedirs(cache_book_path, exist_ok=True)
        with open(cache_book_full_path, "w") as file:
            json.dump(cache_books, file, indent=2)

        print(f"\nCache books saved: {cache_book_full_path}")
        return

    (
        step_cache_book,
        transformer_cache_book,
        single_transformer_cache_book,
    ) = load_cache_books(
        cache_book_path=cache_book_path,
        cache_book_file=cache_book_file,
        expected_rate_method=RATE_METHOD,
    )

    pipe.to("cuda")
    dynamic_model.init_cache_book(
        transformer_cache_book,
        single_transformer_cache_book,
        step_cache_book,
    )
    pipe.transformer = dynamic_model

    prompts = [
        "a photo of a broccoli",
        "A snowy mountain village at dusk, glowing windows and smoke rising.",
        "A golden retriever puppy jumping through autumn leaves.",
        "A surreal underwater city with glowing jellyfish and crystal towers.",
        "A group of astronauts planting a flag on Mars, red rocky landscape.",
        "A vintage sports car speeding down a coastal highway at sunset.",
        "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    ]

    images = []
    times = []
    for prompt in prompts:
        start_time = time.time()
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        times.append(time.time() - start_time)
        images.append(image)

    if len(prompts) > 1:
        measured_times = np.asarray(times[1:])
        print(
            "Sampling time: "
            f"{np.mean(measured_times):.4f}±{np.std(measured_times):.4f} s"
        )
    else:
        print(f"Sampling time: {times[0]:.4f} s")

    width, height = images[0].size
    combined = Image.new("RGB", (width * len(images), height))
    for image_idx, image in enumerate(images):
        combined.paste(image, (image_idx * width, 0))

    os.makedirs("images", exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M%S")
    save_name = (
        f"images/imgs_{POLICY_VARIANT}_stp{num_inference_steps}"
        f"_n{nonskip_rate}_th{step_thres}"
        f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
        f"_sattn{single_attn_thres}_smlp{single_mlp_thres}"
        f"_{timestamp}.png"
    )
    combined.save(save_name)
    print(f"Images saved to {save_name}")


if __name__ == "__main__":
    main()
