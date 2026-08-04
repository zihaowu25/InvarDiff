"""Standalone FLUX sampler with geometric layer-level cache scores.

Layer calibration supports ``negative_dot`` and L2 ``wedge`` scores without
importing implementation code from ``sample_flux.py``. Cross-step and
layer-level scores intentionally use separate interfaces and state.
"""

import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive

LAYER_RATE_METHODS = ("negative_dot", "wedge")
TRANSFORMER_RATE_KEYS = ("attn", "context_attn", "ff", "context_ff")
SINGLE_TRANSFORMER_RATE_KEYS = ("attn", "mlp")
LAYER_RATE_CHUNK_SIZE = 1_048_576


def compute_step_rate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the cross-step L1 norm ratio b / a from two scalar norms."""
    return b / a.clamp_min(1e-8)


def compute_layer_rate(
    a: torch.Tensor,
    b: torch.Tensor,
    method: str,
    chunk_size: int = LAYER_RATE_CHUNK_SIZE,
) -> torch.Tensor:
    """Return a geometric layer score using chunked FP32 reductions.

    Both scores follow the existing lower-score-means-cache rule. Chunking
    avoids materializing full-size FP32 copies of both feature deltas.
    """
    if method not in LAYER_RATE_METHODS:
        raise ValueError(
            f"Unsupported layer rate method: {method!r}. "
            f"Choose one of {LAYER_RATE_METHODS}."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(
            f"Layer delta sizes must match, got {a.numel()} and {b.numel()}."
        )

    dot = torch.zeros((), device=a.device, dtype=torch.float32)
    if method == "wedge":
        a_sq_norm = torch.zeros_like(dot)
        b_sq_norm = torch.zeros_like(dot)

    for start in range(0, a.numel(), chunk_size):
        end = min(start + chunk_size, a.numel())
        a_chunk = a[start:end].float()
        b_chunk = b[start:end].float()

        dot += torch.dot(a_chunk, b_chunk)
        if method == "wedge":
            a_sq_norm += torch.dot(a_chunk, a_chunk)
            b_sq_norm += torch.dot(b_chunk, b_chunk)

    if method == "negative_dot":
        return -dot

    wedge_sq = (
        a_sq_norm * b_sq_norm - dot.square()
    ).clamp_min(0.0)
    return torch.sqrt(wedge_sq)


def register_hooks(model, transformer_keys, single_transformer_keys):
    """Register hooks only for layer modules enabled by non-zero thresholds."""
    transformer_keys = tuple(transformer_keys)
    single_transformer_keys = tuple(single_transformer_keys)

    unknown_transformer = set(transformer_keys) - set(TRANSFORMER_RATE_KEYS)
    unknown_single = set(single_transformer_keys) - set(SINGLE_TRANSFORMER_RATE_KEYS)
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
        for i, block in enumerate(model.transformer_blocks):
            hooks.append(
                block.register_forward_hook(create_transformer_hook(i))
            )
    if single_transformer_keys:
        for i, block in enumerate(model.single_transformer_blocks):
            hooks.append(
                block.register_forward_hook(create_single_transformer_hook(i))
            )

    return transformer_blocks, single_transformer_blocks, hooks

class FeatureChangeAnalyzer:
    def __init__(
        self,
        num_transformer_layers: int,
        num_single_layers: int,
        layer_rate_method: str,
        transformer_keys=TRANSFORMER_RATE_KEYS,
        single_transformer_keys=SINGLE_TRANSFORMER_RATE_KEYS,
    ):
        if layer_rate_method not in LAYER_RATE_METHODS:
            raise ValueError(
                f"Unsupported layer rate method: {layer_rate_method!r}. "
                f"Choose one of {LAYER_RATE_METHODS}."
            )

        self.num_transformer_layers = num_transformer_layers
        self.num_single_layers = num_single_layers
        self.layer_rate_method = layer_rate_method
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
        self._step_count = 0
        self._layer_count = 0

    @staticmethod
    def _detach_feature(feature):
        return None if feature is None else feature.detach()

    def _collect_features(self, transformer_features_dict, single_features_dict):
        transformer_features = {
            key: [
                self._detach_feature(transformer_features_dict[key][i])
                for i in range(self.num_transformer_layers)
            ]
            for key in self.transformer_keys
        }
        single_features = {
            key: [
                self._detach_feature(single_features_dict[key][i])
                for i in range(self.num_single_layers)
            ]
            for key in self.single_transformer_keys
        }
        return transformer_features, single_features

    def step_forward(self, hidden_states):
        """Measure cross-step change with a scalar L1 norm ratio."""
        hidden_states = hidden_states.detach()
        if self._step_count == 0:
            self.current_hidden_states = hidden_states
            self._step_count = 1
            return None

        current_delta = hidden_states - self.current_hidden_states
        current_norm = current_delta.norm(p=1)
        if self._step_count == 1:
            self.prev_step_norm = current_norm
            self.current_hidden_states = hidden_states
            self._step_count = 2
            return None

        step_score = compute_step_rate(self.prev_step_norm, current_norm)
        self.prev_step_norm = current_norm
        self.current_hidden_states = hidden_states
        return step_score

    def _start_layer_state(self, transformer_features, single_features):
        self.current_Transformer = transformer_features
        self.current_SingleTransformer = single_features
        self._layer_count = 1

    def _initialize_layer_deltas(self, transformer_features, single_features):
        self.prev_Transformer_delta = {
            key: [None] * self.num_transformer_layers
            for key in self.transformer_keys
        }
        for key in self.transformer_keys:
            for block_idx, feature in enumerate(transformer_features[key]):
                old_feature = self.current_Transformer[key][block_idx]
                current_delta = (
                    None
                    if feature is None
                    else (feature - old_feature).detach()
                )
                self.prev_Transformer_delta[key][block_idx] = current_delta
                self.current_Transformer[key][block_idx] = feature

        self.prev_SingleTransformer_delta = {
            key: [None] * self.num_single_layers
            for key in self.single_transformer_keys
        }
        for key in self.single_transformer_keys:
            for block_idx, feature in enumerate(single_features[key]):
                old_feature = self.current_SingleTransformer[key][block_idx]
                current_delta = (feature - old_feature).detach()
                self.prev_SingleTransformer_delta[key][block_idx] = current_delta
                self.current_SingleTransformer[key][block_idx] = feature

        self._layer_count = 2

    def step(self, transformer_features_dict, single_features_dict):
        """Collect uncorrected scores and replace layer state in place."""
        transformer_features, single_features = self._collect_features(
            transformer_features_dict,
            single_features_dict,
        )
        if self._layer_count == 0:
            self._start_layer_state(transformer_features, single_features)
            return None, None
        if self._layer_count == 1:
            self._initialize_layer_deltas(transformer_features, single_features)
            return None, None

        transformer_scores = {}
        for key in self.transformer_keys:
            scores = []
            for block_idx, feature in enumerate(transformer_features[key]):
                if feature is None:
                    scores.append(float("nan"))
                    current_delta = None
                else:
                    current_delta = (
                        feature - self.current_Transformer[key][block_idx]
                    )
                    score = compute_layer_rate(
                        self.prev_Transformer_delta[key][block_idx],
                        current_delta,
                        self.layer_rate_method,
                    )
                    scores.append(score.item())

                self.prev_Transformer_delta[key][block_idx] = (
                    None if current_delta is None else current_delta.detach()
                )
                self.current_Transformer[key][block_idx] = feature

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                current_delta = (
                    feature - self.current_SingleTransformer[key][block_idx]
                )
                score = compute_layer_rate(
                    self.prev_SingleTransformer_delta[key][block_idx],
                    current_delta,
                    self.layer_rate_method,
                )
                scores.append(score.item())
                self.prev_SingleTransformer_delta[key][block_idx] = (
                    current_delta.detach()
                )
                self.current_SingleTransformer[key][block_idx] = feature

            single_scores[key] = torch.tensor(scores)

        return transformer_scores, single_scores

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

    def step_correct(
        self,
        transformer_features_dict,
        single_features_dict,
        transformer_cache_state,
        single_cache_state,
        timestep_idx,
    ):
        """Recompute layer scores while following the provisional cache path."""
        transformer_features, single_features = self._collect_features(
            transformer_features_dict,
            single_features_dict,
        )
        if timestep_idx == 0:
            self._start_layer_state(transformer_features, single_features)
            return None, None
        if timestep_idx == 1:
            self._initialize_layer_deltas(transformer_features, single_features)
            return None, None

        transformer_scores = {}
        for key in self.transformer_keys:
            scores = []
            for block_idx, feature in enumerate(transformer_features[key]):
                if feature is None:
                    scores.append(float("nan"))
                    current_delta = None
                else:
                    current_delta = (
                        feature - self.current_Transformer[key][block_idx]
                    )
                    score = compute_layer_rate(
                        self.prev_Transformer_delta[key][block_idx],
                        current_delta,
                        self.layer_rate_method,
                    )
                    scores.append(score.item())

                if self._should_refresh(
                    transformer_cache_state,
                    self.Transformer_state,
                    key,
                    timestep_idx,
                    block_idx,
                ):
                    self.prev_Transformer_delta[key][block_idx] = (
                        None if current_delta is None else current_delta.detach()
                    )
                    self.current_Transformer[key][block_idx] = feature

            transformer_scores[key] = torch.tensor(scores)

        single_scores = {}
        for key in self.single_transformer_keys:
            scores = []
            for block_idx, feature in enumerate(single_features[key]):
                current_delta = (
                    feature - self.current_SingleTransformer[key][block_idx]
                )
                score = compute_layer_rate(
                    self.prev_SingleTransformer_delta[key][block_idx],
                    current_delta,
                    self.layer_rate_method,
                )
                scores.append(score.item())

                if self._should_refresh(
                    single_cache_state,
                    self.SingleTransformer_state,
                    key,
                    timestep_idx,
                    block_idx,
                ):
                    self.prev_SingleTransformer_delta[key][block_idx] = (
                        current_delta.detach()
                    )
                    self.current_SingleTransformer[key][block_idx] = feature

            single_scores[key] = torch.tensor(scores)

        return transformer_scores, single_scores

    def reset(self):
        self.__init__(
            self.num_transformer_layers,
            self.num_single_layers,
            self.layer_rate_method,
            self.transformer_keys,
            self.single_transformer_keys,
        )


def _active_layer_keys(thresholds):
    for module_name, threshold in thresholds.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {module_name!r} must be in [0, 1], "
                f"got {threshold}."
            )
    return tuple(
        module_name
        for module_name, threshold in thresholds.items()
        if threshold > 0.0
    )


def _build_layer_cache_book(
    avg_scores,
    thresholds,
    num_timesteps,
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
                avg_scores[module_name][1:-1],
                threshold_q,
            )
            cache_bool = avg_scores[module_name] < threshold_value

        cache_bool[:protected_prefix, :] = False
        cache_bool[-1, :] = False
        cache_book[module_name] = cache_bool

    return cache_book


def threshold_analyse(
    model, pipe, measure_prompts,
    layer_rate_method = "negative_dot",
    nonskip_rate = 0.1,
    step_thres = 0.5,
    attn_thres = 0.5, # = context_attn_thres, ip_attn_thres
    ff_thres = 0.5,
    context_ff_thres = 0.5,
    Single_attn_thres = 0.5, # FluxSingleTransformerBlock
    Single_mlp_thres = 0.5,
    seed = 42,
):
    device = pipe._execution_device
    num_timesteps = model.num_timesteps
    num_transformer_layers = model.num_layers
    num_single_layers = model.num_single_layers
    num_nonskip = max(int(nonskip_rate * num_timesteps), 1)

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
    active_transformer_keys = _active_layer_keys(transformer_thresholds)
    active_single_keys = _active_layer_keys(single_thresholds)

    print(f"Layer rate method: {layer_rate_method}")
    print(f"Active transformer scores: {active_transformer_keys}")
    print(f"Active single-transformer scores: {active_single_keys}")
    print(
        f"Transformer blocks numbers: {num_transformer_layers}\n"
        f"Single Transformer blocks numbers: {num_single_layers}"
    )

    def collect_scores(
        description,
        transformer_cache_book=None,
        single_cache_book=None,
        collect_step_scores=False,
    ):
        all_transformer_scores = {
            key: [] for key in active_transformer_keys
        }
        all_single_scores = {
            key: [] for key in active_single_keys
        }
        all_step_scores = []
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
                    layer_rate_method,
                    active_transformer_keys,
                    active_single_keys,
                )
                run_transformer_scores = {
                    key: torch.ones(
                        num_timesteps,
                        num_transformer_layers,
                        device="cpu",
                    )
                    for key in active_transformer_keys
                }
                run_single_scores = {
                    key: torch.ones(
                        num_timesteps,
                        num_single_layers,
                        device="cpu",
                    )
                    for key in active_single_keys
                }
                if collect_step_scores:
                    run_step_scores = torch.ones(
                        num_timesteps,
                        device="cpu",
                    )

                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
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

                    if transformer_cache_book is None:
                        transformer_scores, single_scores = analyzer.step(
                            transformer_features,
                            single_features,
                        )
                    else:
                        transformer_scores, single_scores = analyzer.step_correct(
                            transformer_features,
                            single_features,
                            transformer_cache_book,
                            single_cache_book,
                            timestep_idx,
                        )

                    if collect_step_scores:
                        step_score = analyzer.step_forward(
                            model.hidden_states_cache
                        )
                        if step_score is not None:
                            run_step_scores[timestep_idx - 1] = step_score.cpu()

                    if transformer_scores is not None:
                        for key in active_transformer_keys:
                            run_transformer_scores[key][timestep_idx - 1] = (
                                transformer_scores[key].cpu()
                            )
                        for key in active_single_keys:
                            run_single_scores[key][timestep_idx - 1] = (
                                single_scores[key].cpu()
                            )

                if collect_step_scores:
                    all_step_scores.append(run_step_scores)
                for key in active_transformer_keys:
                    all_transformer_scores[key].append(
                        run_transformer_scores[key]
                    )
                for key in active_single_keys:
                    all_single_scores[key].append(run_single_scores[key])

                for hook in hooks:
                    hook.remove()
                del transformer_features, single_features
                del analyzer, hooks
                torch.cuda.empty_cache()

        avg_transformer_scores = {
            key: torch.stack(values).mean(dim=0)
            for key, values in all_transformer_scores.items()
        }
        avg_single_scores = {
            key: torch.stack(values).mean(dim=0)
            for key, values in all_single_scores.items()
        }
        avg_step_scores = (
            torch.stack(all_step_scores).mean(dim=0)
            if collect_step_scores
            else None
        )
        if track_cuda_memory:
            peak_gib = torch.cuda.max_memory_allocated(cuda_memory_device) / (1024 ** 3)
            print(f"Peak GPU memory for {description}: {peak_gib:.2f} GiB")

        return avg_transformer_scores, avg_single_scores, avg_step_scores

    model.eval()
    (
        avg_transformer_scores,
        avg_single_scores,
        avg_step_scores,
    ) = collect_scores(
        description="Threshold analysis",
        collect_step_scores=True,
    )

    step_threshold_value = torch.quantile(
        avg_step_scores[1:-1],
        step_thres,
    )
    step_cache_bool = avg_step_scores < step_threshold_value
    step_cache_bool[:num_nonskip] = False
    step_cache_bool[-1] = False

    transformer_cache_book = _build_layer_cache_book(
        avg_transformer_scores,
        transformer_thresholds,
        num_timesteps,
        num_transformer_layers,
        protected_prefix=1,
    )
    transformer_cache_book["ip_attn"] = transformer_cache_book["attn"].clone()
    single_transformer_cache_book = _build_layer_cache_book(
        avg_single_scores,
        single_thresholds,
        num_timesteps,
        num_single_layers,
        protected_prefix=1,
    )

    (
        corrected_transformer_scores,
        corrected_single_scores,
        _,
    ) = collect_scores(
        description="Cache correction",
        transformer_cache_book=transformer_cache_book,
        single_cache_book=single_transformer_cache_book,
        collect_step_scores=False,
    )

    transformer_cache_book = _build_layer_cache_book(
        corrected_transformer_scores,
        transformer_thresholds,
        num_timesteps,
        num_transformer_layers,
        protected_prefix=num_nonskip,
    )
    transformer_cache_book["ip_attn"] = transformer_cache_book["attn"].clone()
    single_transformer_cache_book = _build_layer_cache_book(
        corrected_single_scores,
        single_thresholds,
        num_timesteps,
        num_single_layers,
        protected_prefix=num_nonskip,
    )

    for timestep_idx in range(num_timesteps):
        if not step_cache_bool[timestep_idx]:
            continue
        for cache_book in (
            transformer_cache_book,
            single_transformer_cache_book,
        ):
            for module_cache in cache_book.values():
                module_cache[timestep_idx, :] = True
                if timestep_idx + 1 < num_timesteps:
                    module_cache[timestep_idx + 1, :] = False

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
        corrected_transformer_scores,
        corrected_single_scores,
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


def load_cache_books(
    cache_book_path,
    cache_book_file,
    expected_layer_rate_method=None,
):
    with open(os.path.join(cache_book_path, cache_book_file), "r") as f:
        cache_books = json.load(f)

    saved_method = cache_books.get("layer_rate_method")
    if (
        expected_layer_rate_method is not None
        and saved_method != expected_layer_rate_method
    ):
        raise ValueError(
            "Cache-book layer metric mismatch: "
            f"expected {expected_layer_rate_method!r}, found {saved_method!r}."
        )

    return (
        cache_books["step_cache_book"],
        cache_books["transformer_cache_book"],
        cache_books["single_transformer_cache_book"],
    )


def main():
    seed=42
    torch.set_grad_enabled(False)

    layer_rate_method = "negative_dot"  # "negative_dot" or "wedge"
    run_calibration = 1
    calibration_model_cpu_offload = True

    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/root/autodl-tmp/InvarDiff/FLUX"
    )
    original_transformer = pipe.transformer

    ## default measure 5 prompts, the number of measure prompts will have a slight impact on the speedup ratio.
    ## fast(3.3x): nonskip_rate=0.1, step_thres = 0.7, attn_thres=ff_thres=context_ff_thres=Single_attn_thres=Single_mlp_thres= 0.68 (measure 5 prompts)
    ## medium-1(2.9x): nonskip_rate=0.1, step_thres = 0.7, attn_thres=0.68, ff_thres=0, context_ff_thres=0, Single_attn_thres=0.68, Single_mlp_thres=0
    ## medium-2(2.6x): nonskip_rate=0.15, step_thres=0.7, attn_thres=0.68, ff_thres=0, context_ff_thres=0, Single_attn_thres=0.7, Single_mlp_thres=0
    ## slow(2.5x): nonskip_rate=0.22, step_thres=0.72, attn_thres=0.68, ff_thres=0.66, context_ff_thres=0, Single_attn_thres=0.68, Single_mlp_thres=0.62
    num_inference_steps = 28
    nonskip_rate = 0.1
    step_thres = 0 # Clearly affects the acceleration ratio and reduce the proportion of finegrained cache.

    attn_thres=0.5
    ff_thres= 0.5 # This threshold sometimes cause blemishes on the image.
    context_ff_thres=0.5

    Single_attn_thres=0.5
    Single_mlp_thres=0.5 # Lowering this threshold can reduce "moiré patterns".

    dynamic_model = DynamicFluxTransformer2DModel(
        original_transformer,
        num_inference_steps,
    )
    dynamic_model.eval()
    pipe.transformer = dynamic_model

    if run_calibration:
        if calibration_model_cpu_offload:
            print("Calibration model CPU offload: enabled")
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            pipe.to("cuda")
        ## There is no necessary correlation between the measure prompts and the test prompts.
        measure_prompts = [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            "A futuristic cityscape with flying cars and neon lights.",
            "An astronaut riding a horse on the moon.",
            "A bouquet of wildflowers in a glass vase, watercolor style.",
            "A majestic lion sitting on a rock, golden mane, sunset.",
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
        step_cache_book, transformer_cache_book, single_transformer_cache_book, \
        avg_transformer_rates, avg_single_transformer_rates = threshold_analyse(
            model=dynamic_model,
            pipe=pipe,
            measure_prompts=measure_prompts,
            layer_rate_method=layer_rate_method,
            nonskip_rate=nonskip_rate,
            step_thres=step_thres,
            attn_thres=attn_thres,
            ff_thres=ff_thres,
            context_ff_thres=context_ff_thres,
            Single_attn_thres=Single_attn_thres,
            Single_mlp_thres=Single_mlp_thres,
            seed=seed
        )
        cache_books = {
            "layer_rate_method": layer_rate_method,
            "step_cache_book": step_cache_book,
            "transformer_cache_book": transformer_cache_book,
            "single_transformer_cache_book": single_transformer_cache_book
        }
        cache_book_path = "./cache_books"
        os.makedirs(cache_book_path, exist_ok=True)

        thres_str = (
            f"layer{layer_rate_method}_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
            f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
            f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}"
        )
        cache_book_file = f"{cache_book_path}/cache_books_{thres_str}.json"
        with open(cache_book_file, "w") as f:
            json.dump(cache_books, f, indent=2)

        print(f"\nCache books saved: {cache_book_file}")

    else:
        cache_book_file = (
            f"cache_books_layer{layer_rate_method}_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
            f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
            f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}.json"
        )
        step_cache_book, transformer_cache_book, single_transformer_cache_book = load_cache_books(
            cache_book_path="./cache_books",
            cache_book_file=cache_book_file,
            expected_layer_rate_method=layer_rate_method,
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
        images = []
        times = []

        for prompt in prompts:
            start_time = time.time()
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
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

        os.makedirs("images", exist_ok=True)
        timestamp = time.strftime("%m%d_%H%M%S")
        save_name = (f"images/imgs_layer{layer_rate_method}_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
                     f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
                     f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}_{timestamp}.png")
        combined.save(save_name)
        print(f"Images saved to {save_name}")


if __name__ == "__main__":
    main()
