"""Standalone DiT sampler with layer-only relative-L1 caching.

This comparison variant calibrates and caches MSA/MLP modules only. Cross-step
cache is disabled, while the runtime step cache book is always False. It does
not import implementation code from sample_dit_step_layer.py.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cache_presets import add_preset_argument, apply_preset

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import create_diffusion
from download import find_model
from models.dynamic_cache import DiT_models, DynamicDiT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LAYER_MODULES = ("msa", "mlp")
RATE_METHOD = "relative_l1"
CACHE_BOOK_VERSION = 2
CACHE_SCOPE = "layer_only"
POLICY_VARIANT = "layer"
RATE_CHUNK_SIZE = 1_048_576


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values together with shape and dtype for pairing audits."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decode_vae_in_batches(vae, latents, batch_size: int):
    """Decode without changing sample order while bounding peak VAE memory."""
    scaled = latents / 0.18215
    if batch_size <= 0 or batch_size >= scaled.shape[0]:
        return vae.decode(scaled).sample
    return torch.cat(
        [vae.decode(chunk).sample for chunk in scaled.split(batch_size)],
        dim=0,
    )


def cache_execution_ledger(step_cache_book, msa_cache_book, mlp_cache_book):
    """Count the module decisions that the layer-only runtime executes."""
    ledger = {}
    for name, book in (("msa", msa_cache_book), ("mlp", mlp_cache_book)):
        reused = int(torch.as_tensor(book, dtype=torch.bool).sum().item())
        total = int(torch.as_tensor(book).numel())
        ledger[name] = {
            "total_positions": total,
            "effective_reuse": reused,
            "compute": total - reused,
        }
    step_book = torch.as_tensor(step_cache_book, dtype=torch.bool)
    reused = int(step_book.sum().item())
    ledger["whole_step"] = {
        "total_positions": int(step_book.numel()),
        "effective_reuse": reused,
        "compute": int(step_book.numel()) - reused,
    }
    return ledger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def register_hooks(model, active_modules=LAYER_MODULES):
    """Register hooks only for layer modules with non-zero thresholds."""
    active_modules = tuple(active_modules)
    unknown_modules = set(active_modules) - set(LAYER_MODULES)
    if unknown_modules:
        raise ValueError(f"Unknown layer modules: {sorted(unknown_modules)}")

    feature_dicts = {module_name: {} for module_name in active_modules}

    def create_hook(layer_idx):
        def hook(module, inputs, output):
            _, block_output = output
            if "msa" in feature_dicts:
                feature_dicts["msa"][layer_idx] = block_output[0].detach()
            if "mlp" in feature_dicts:
                feature_dicts["mlp"][layer_idx] = block_output[1].detach()

        return hook

    hooks = [
        block.register_forward_hook(create_hook(layer_idx))
        for layer_idx, block in enumerate(model.blocks)
    ] if active_modules else []

    return (
        feature_dicts.get("msa", {}),
        feature_dicts.get("mlp", {}),
        hooks,
    )


class FeatureChangeAnalyzer:
    """Analyzer for MSA/MLP trajectories with compressed relative-L1 state."""

    def __init__(self, num_layers, active_modules=LAYER_MODULES):
        self.num_layers = num_layers
        self.active_modules = tuple(active_modules)
        self.current_features = {
            key: [None] * num_layers for key in self.active_modules
        }
        self.previous_norms = {
            key: [None] * num_layers for key in self.active_modules
        }
        self.cache_states = {
            key: [False] * num_layers for key in self.active_modules
        }
        self._layer_count = 0

    # Cross-step cache is intentionally disabled in this layer-only variant.

    def _collect_features(self, msa_features_dict, mlp_features_dict):
        source_dicts = {
            "msa": msa_features_dict,
            "mlp": mlp_features_dict,
        }
        return {
            key: [
                source_dicts[key][layer_idx].detach()
                for layer_idx in range(self.num_layers)
            ]
            for key in self.active_modules
        }

    def _initialize_current_features(self, features):
        for key in self.active_modules:
            for layer_idx in range(self.num_layers):
                self.previous_norms[key][layer_idx] = compute_l1_distance(
                    self.current_features[key][layer_idx],
                    features[key][layer_idx],
                )
            self.current_features[key] = features[key]
        self._layer_count = 2

    @staticmethod
    def _should_refresh(
        cache_book,
        cache_states,
        timestep_idx,
        layer_idx,
    ):
        cached_on_previous_step = cache_book[timestep_idx - 1][layer_idx]
        if not cached_on_previous_step:
            cache_states[layer_idx] = False
            return True
        if not cache_states[layer_idx]:
            cache_states[layer_idx] = True
            return True
        return False

    def _format_scores(self, score_dict):
        return score_dict.get("msa"), score_dict.get("mlp")

    def step(self, msa_features_dict, mlp_features_dict):
        """Collect raw layer scores and rotate the compressed states in place."""
        features = self._collect_features(
            msa_features_dict,
            mlp_features_dict,
        )
        if self._layer_count == 0:
            for key in self.active_modules:
                self.current_features[key] = features[key]
            self._layer_count = 1
            return None, None

        if self._layer_count == 1:
            self._initialize_current_features(features)
            return None, None

        score_dict = {}
        for key in self.active_modules:
            scores = []
            for layer_idx, feature in enumerate(features[key]):
                current = self.current_features[key][layer_idx]
                current_norm = compute_l1_distance(current, feature)
                score = compute_rate(
                    current_norm,
                    self.previous_norms[key][layer_idx],
                )
                scores.append(score.item())

                self.previous_norms[key][layer_idx] = current_norm
                self.current_features[key][layer_idx] = feature

            score_dict[key] = torch.tensor(scores, dtype=torch.float32)

        return self._format_scores(score_dict)

    def step_correct(
        self,
        msa_features_dict,
        mlp_features_dict,
        msa_cache_book,
        mlp_cache_book,
        timestep_idx,
    ):
        """Collect scores while following provisional layer cache paths."""
        features = self._collect_features(
            msa_features_dict,
            mlp_features_dict,
        )
        if timestep_idx == 0:
            for key in self.active_modules:
                self.current_features[key] = features[key]
            self._layer_count = 1
            return None, None

        if timestep_idx == 1:
            self._initialize_current_features(features)
            return None, None

        cache_books = {
            "msa": msa_cache_book,
            "mlp": mlp_cache_book,
        }
        score_dict = {}
        for key in self.active_modules:
            scores = []
            for layer_idx, feature in enumerate(features[key]):
                current = self.current_features[key][layer_idx]
                current_norm = compute_l1_distance(current, feature)
                score = compute_rate(
                    current_norm,
                    self.previous_norms[key][layer_idx],
                )
                scores.append(score.item())

                if self._should_refresh(
                    cache_books[key],
                    self.cache_states[key],
                    timestep_idx,
                    layer_idx,
                ):
                    self.previous_norms[key][layer_idx] = current_norm
                    self.current_features[key][layer_idx] = feature

            score_dict[key] = torch.tensor(scores, dtype=torch.float32)

        return self._format_scores(score_dict)


def _build_layer_cache_book(
    average_scores,
    threshold_q,
    num_timesteps,
    num_layers,
    protected_prefix,
):
    if threshold_q == 0.0:
        cache_book = torch.zeros(
            num_timesteps,
            num_layers,
            dtype=torch.bool,
        )
    else:
        threshold = torch.quantile(average_scores[1:-1], threshold_q)
        cache_book = average_scores < threshold

    cache_book[:protected_prefix, :] = False
    cache_book[-1, :] = False
    return cache_book


def threshold_analyse(
    model,
    diffusion,
    class_labels,
    input_size,
    nonskip_rate=0,
    msa_thres=0.53,
    mlp_thres=0.39,
    num_analysis=1,
):
    """Run raw and cache-corrected calibration for MSA/MLP only."""
    # Cross-step threshold is intentionally disabled:
    # step_thres = 0.2
    for module_name, threshold in {
        "msa": msa_thres,
        "mlp": mlp_thres,
    }.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for {module_name!r} must be in [0, 1], "
                f"got {threshold}."
            )

    device = next(model.parameters()).device
    num_classes = len(class_labels)
    z = torch.randn(
        num_classes,
        4,
        input_size,
        input_size,
        device=device,
    )
    y = torch.tensor(class_labels, device=device)

    num_timesteps = diffusion.num_timesteps
    num_layers = len(model.blocks)
    num_nonskip = max(1, int(nonskip_rate * num_timesteps))
    thresholds = {"msa": msa_thres, "mlp": mlp_thres}
    active_modules = tuple(
        key for key, threshold in thresholds.items() if threshold > 0.0
    )

    print(
        f"Cache scope: layer only\n"
        f"Rate method: {RATE_METHOD}\n"
        f"Active layer scores: {active_modules}\n"
        f"Block numbers: {num_layers}"
    )

    def collect_scores(
        description,
        msa_cache_book=None,
        mlp_cache_book=None,
    ):
        correction_mode = msa_cache_book is not None
        score_runs = {key: [] for key in active_modules}
        msa_features, mlp_features, hooks = register_hooks(
            model,
            active_modules,
        )

        track_cuda_memory = torch.cuda.is_available()
        cuda_memory_device = (
            torch.cuda.current_device() if track_cuda_memory else None
        )
        if track_cuda_memory:
            torch.cuda.reset_peak_memory_stats(cuda_memory_device)

        try:
            with torch.inference_mode():
                for run_idx in tqdm(range(num_analysis), desc=description):
                    class_idx = run_idx % num_classes
                    current_z = z[class_idx:class_idx + 1]
                    current_y = y[class_idx:class_idx + 1]
                    analyzer = FeatureChangeAnalyzer(
                        num_layers,
                        active_modules,
                    )
                    run_scores = {
                        key: torch.ones(
                            num_timesteps,
                            num_layers,
                            dtype=torch.float32,
                        )
                        for key in active_modules
                    }

                    sampler = diffusion.ddim_sample_loop_progressive(
                        model,
                        current_z.shape,
                        current_z,
                        clip_denoised=False,
                        model_kwargs=dict(y=current_y),
                        progress=False,
                        device=device,
                    )
                    for timestep_idx, _ in enumerate(sampler):
                        if correction_mode:
                            msa_scores, mlp_scores = analyzer.step_correct(
                                msa_features,
                                mlp_features,
                                msa_cache_book,
                                mlp_cache_book,
                                timestep_idx,
                            )
                        else:
                            msa_scores, mlp_scores = analyzer.step(
                                msa_features,
                                mlp_features,
                            )

                        # Cross-step scoring is intentionally disabled:
                        # step_score = analyzer.step_forward(hidden_state[0])
                        # run_step_scores[timestep_idx - 1] = step_score.cpu()

                        module_scores = {
                            "msa": msa_scores,
                            "mlp": mlp_scores,
                        }
                        for key in active_modules:
                            if module_scores[key] is not None:
                                run_scores[key][timestep_idx - 1] = (
                                    module_scores[key].cpu()
                                )

                    for key in active_modules:
                        score_runs[key].append(run_scores[key])
                    del analyzer
                    if track_cuda_memory:
                        torch.cuda.empty_cache()
        finally:
            for hook in hooks:
                hook.remove()

        if track_cuda_memory:
            peak_gib = (
                torch.cuda.max_memory_allocated(cuda_memory_device)
                / (1024 ** 3)
            )
            print(f"Peak GPU memory for {description}: {peak_gib:.2f} GiB")

        return {
            key: torch.stack(values).mean(dim=0)
            for key, values in score_runs.items()
        }

    model.eval()
    initial_scores = collect_scores("Layer threshold analysis")
    provisional_msa_book = _build_layer_cache_book(
        initial_scores.get("msa"),
        msa_thres,
        num_timesteps,
        num_layers,
        num_nonskip,
    )
    provisional_mlp_book = _build_layer_cache_book(
        initial_scores.get("mlp"),
        mlp_thres,
        num_timesteps,
        num_layers,
        num_nonskip,
    )

    corrected_scores = collect_scores(
        "Layer cache correction",
        provisional_msa_book,
        provisional_mlp_book,
    )
    msa_cache_book = _build_layer_cache_book(
        corrected_scores.get("msa"),
        msa_thres,
        num_timesteps,
        num_layers,
        num_nonskip,
    )
    mlp_cache_book = _build_layer_cache_book(
        corrected_scores.get("mlp"),
        mlp_thres,
        num_timesteps,
        num_layers,
        num_nonskip,
    )

    # Cross-step thresholding and step-first policy are disabled:
    #
    # step_threshold = torch.quantile(avg_step_rates[1:-1], step_thres)
    # step_cache_book = avg_step_rates < step_threshold
    # for timestep_idx in range(num_timesteps):
    #     if step_cache_book[timestep_idx]:
    #         msa_cache_book[timestep_idx, :] = True
    #         mlp_cache_book[timestep_idx, :] = True

    step_cache_book = torch.zeros(num_timesteps, dtype=torch.bool)
    average_msa_scores = corrected_scores.get(
        "msa",
        torch.ones(num_timesteps, num_layers, dtype=torch.float32),
    )
    average_mlp_scores = corrected_scores.get(
        "mlp",
        torch.ones(num_timesteps, num_layers, dtype=torch.float32),
    )

    compute_skip_ratio(msa_cache_book, mlp_cache_book, num_layers)
    return (
        step_cache_book,
        msa_cache_book,
        mlp_cache_book,
        average_msa_scores,
        average_mlp_scores,
    )


def compute_skip_ratio(msa_cache_book, mlp_cache_book, num_layers):
    num_timesteps = len(msa_cache_book)
    total_modules = num_timesteps * num_layers * 2
    msa_modules = int(msa_cache_book.sum().item())
    mlp_modules = int(mlp_cache_book.sum().item())
    skipped_modules = msa_modules + mlp_modules

    print(
        f"Total skip ratio: {100.0 * skipped_modules / total_modules:.2f}%\n"
        f"MSA skip ratio: {200.0 * msa_modules / total_modules:.2f}%\n"
        f"MLP skip ratio: {200.0 * mlp_modules / total_modules:.2f}%"
    )


def cache_book_config(
    num_timesteps,
    nonskip_rate,
    msa_thres,
    mlp_thres,
):
    return {
        "policy": POLICY_VARIANT,
        "steps": num_timesteps,
        "nonskip": nonskip_rate,
        "msa": msa_thres,
        "mlp": mlp_thres,
    }


def cache_book_name(config):
    fmt = lambda value: format(float(value), "g")
    return (
        f"cache_book_layer_steps{config['steps']}_ns{fmt(config['nonskip'])}"
        f"_msath{fmt(config['msa'])}_mlpth{fmt(config['mlp'])}.json"
    )


def save_cache_books(
    step_cache_book,
    msa_cache_book,
    mlp_cache_book,
    num_timesteps,
    nonskip_rate,
    msa_thres,
    mlp_thres,
    cache_book_path="./cache_books",
    preset_name="default",
    resolved_thresholds=None,
):
    config = cache_book_config(
        num_timesteps,
        nonskip_rate,
        msa_thres,
        mlp_thres,
    )
    cache_books = {
        "cache_version": CACHE_BOOK_VERSION,
        "cache_scope": CACHE_SCOPE,
        "config": config,
        "rate_method": RATE_METHOD,
        "cache_preset": preset_name,
        "resolved_thresholds": resolved_thresholds or {
            "msa_thres": msa_thres,
            "mlp_thres": mlp_thres,
        },
        "step_cache_book": step_cache_book.tolist(),
        "msa_cache_book": msa_cache_book.tolist(),
        "mlp_cache_book": mlp_cache_book.tolist(),
    }
    os.makedirs(cache_book_path, exist_ok=True)
    cache_book_file = cache_book_name(config)
    cache_book_full_path = os.path.join(cache_book_path, cache_book_file)
    with open(cache_book_full_path, "w") as file:
        json.dump(cache_books, file, indent=2)

    print(f"\nCache books saved: {cache_book_full_path}")
    return cache_book_file


def load_cache_books(cache_book_path, cache_book_file):
    cache_book_full_path = os.path.join(cache_book_path, cache_book_file)
    with open(cache_book_full_path, "r") as file:
        cache_books = json.load(file)

    if cache_books.get("cache_version") != CACHE_BOOK_VERSION:
        raise ValueError("Incompatible cache book; re-run calibration.")
    if cache_books.get("cache_scope") != CACHE_SCOPE:
        raise ValueError(
            f"Expected cache scope {CACHE_SCOPE!r}, "
            f"found {cache_books.get('cache_scope')!r}."
        )
    if cache_books.get("rate_method") != RATE_METHOD:
        raise ValueError(
            f"Expected rate method {RATE_METHOD!r}, "
            f"found {cache_books.get('rate_method')!r}."
        )

    step_cache_book = torch.tensor(
        cache_books["step_cache_book"],
        dtype=torch.bool,
    )
    if bool(step_cache_book.any()):
        raise ValueError(
            "Layer-only cache book contains enabled cross-step decisions."
        )

    return (
        step_cache_book,
        torch.tensor(cache_books["msa_cache_book"], dtype=torch.bool),
        torch.tensor(cache_books["mlp_cache_book"], dtype=torch.bool),
    )


def main(args):
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = args.image_size // 8
    num_timesteps = args.num_timesteps
    diffusion = create_diffusion(str(num_timesteps))
    base_dit = DiT_models[args.model](
        input_size=input_size,
        num_classes=args.num_classes,
    ).to(device)

    print("Loading model...")
    print("Cache scope: layer only")
    print(f"Rate method: {RATE_METHOD}")
    dit_state_dict = find_model(args.dit_ckpt)
    base_dit.load_state_dict(dit_state_dict)
    base_dit.eval()

    all_classes = list(range(args.num_classes))
    default_labels = [207, 992, 387, 37, 142, 979, 417, 279]
    if args.class_label_file:
        with open(args.class_label_file, "r", encoding="utf-8") as file:
            class_labels = [int(line.strip()) for line in file if line.strip()]
        if not class_labels:
            raise ValueError("--class-label-file contains no labels")
        if any(label < 0 or label >= args.num_classes for label in class_labels):
            raise ValueError("class label is outside --num-classes")
    else:
        class_labels = default_labels[:args.num_sample_classes]
    if args.calibration_class_file:
        with open(args.calibration_class_file, "r", encoding="utf-8") as file:
            measure_labels = [int(line.strip()) for line in file if line.strip()]
        if not measure_labels:
            raise ValueError("--calibration-class-file contains no labels")
        if any(label < 0 or label >= args.num_classes for label in measure_labels):
            raise ValueError("calibration label is outside --num-classes")
    elif args.num_analysis <= 1:
        measure_labels = [class_labels[0]]
    else:
        measure_labels = random.sample(
            all_classes, min(args.num_analysis, len(all_classes))
        )

    cache_config = cache_book_config(
        num_timesteps,
        args.nonskip_rate,
        args.msa_thres,
        args.mlp_thres,
    )
    cache_file = cache_book_name(cache_config)
    if args.generate_cache_books:
        (
            step_cache_book,
            msa_cache_book,
            mlp_cache_book,
            _,
            _,
        ) = threshold_analyse(
            base_dit,
            diffusion,
            measure_labels,
            input_size,
            nonskip_rate=args.nonskip_rate,
            msa_thres=args.msa_thres,
            mlp_thres=args.mlp_thres,
            num_analysis=args.num_analysis,
        )
        save_cache_books(
            step_cache_book,
            msa_cache_book,
            mlp_cache_book,
            num_timesteps,
            args.nonskip_rate,
            args.msa_thres,
            args.mlp_thres,
            args.cache_book_path,
            args.cache_preset_resolved,
            args.cache_resolved_thresholds,
        )
        if args.calibration_only:
            return
    else:
        step_cache_book, msa_cache_book, mlp_cache_book = load_cache_books(
            args.cache_book_path,
            cache_file,
        )
        print(
            "Cache books loaded from: "
            f"{os.path.join(args.cache_book_path, cache_file)}"
        )

    dynamic_dit = DynamicDiT(
        base_dit,
        msa_cache_book,
        mlp_cache_book,
        step_cache_book,
    )
    dynamic_dit.eval()
    vae = AutoencoderKL.from_pretrained(
        args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    ).to(device)

    num_samples = len(class_labels)
    z = torch.randn(
        num_samples,
        4,
        input_size,
        input_size,
        device=device,
    )
    latent_sha256 = tensor_sha256(z)
    per_sample_latent_sha256 = [tensor_sha256(item) for item in z]
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor(
        [args.num_classes] * num_samples,
        device=device,
    )
    y = torch.cat([y, y_null], dim=0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.sample_times):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        samples = diffusion.ddim_sample_loop(
            dynamic_dit.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - start_time)
        dynamic_dit.reset_inference()

    if len(times) > 1:
        measured_times = np.asarray(times[1:])
        print(
            "Accelerated sampling time: "
            f"{np.mean(measured_times):.3f}±{np.std(measured_times):.3f} s"
        )
    else:
        print(f"Accelerated sampling time: {times[0]:.3f} s")

    samples, _ = samples.chunk(2, dim=0)
    samples = decode_vae_in_batches(vae, samples, args.vae_batch_size)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M%S")
    save_name = args.output_file or (
        f"{output_dir}/{POLICY_VARIANT}_NFE{num_timesteps}"
        f"_CFG{args.cfg_scale}_msa{args.msa_thres:.2f}"
        f"_mlp{args.mlp_thres:.2f}_seed{args.seed}_{timestamp}.png"
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_name)), exist_ok=True)
    save_image(
        samples,
        save_name,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Samples saved to {save_name}.")
    sample_items = []
    if args.save_individual_dir:
        os.makedirs(args.save_individual_dir, exist_ok=True)
        for index, (label, sample) in enumerate(zip(class_labels, samples)):
            individual_path = os.path.join(
                args.save_individual_dir,
                f"class_{label:04d}_seed_{args.seed}.png",
            )
            save_image(
                sample,
                individual_path,
                normalize=True,
                value_range=(-1, 1),
            )
            sample_items.append(
                {
                    "class_label": label,
                    "seed": args.seed,
                    "latent_sha256": per_sample_latent_sha256[index],
                    "candidate_path": os.path.abspath(individual_path),
                    "candidate_sha256": file_sha256(individual_path),
                }
            )
    if args.metrics_json:
        measured_times = times[1:] if len(times) > 1 else times
        metrics = {
            "schema_version": 2,
            "model": args.model,
            "image_size": args.image_size,
            "num_timesteps": num_timesteps,
            "cfg_scale": args.cfg_scale,
            "vae_batch_size": args.vae_batch_size,
            "seed": args.seed,
            "class_labels": class_labels,
            "latent_sha256": latent_sha256,
            "per_sample_latent_sha256": per_sample_latent_sha256,
            "items": sample_items,
            "sampling_seconds": times,
            "sampling_mean_seconds": float(np.mean(measured_times)),
            "sampling_std_seconds": float(np.std(measured_times)),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else None
            ),
            "execution_ledger": cache_execution_ledger(
                step_cache_book,
                msa_cache_book,
                mlp_cache_book,
            ),
            "output_file": os.path.abspath(save_name),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)), exist_ok=True)
        temporary = args.metrics_json + ".tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
            file.write("\n")
        os.replace(temporary, args.metrics_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample images using layer-only DynamicDiT"
    )
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-timesteps", type=int, default=250)
    parser.add_argument(
        "--dit-ckpt",
        type=str,
        default="./pretrained_models/DiT-XL-2-256x256.pt",
    )
    parser.add_argument("--num-sample-classes", type=int, default=10)
    parser.add_argument("--class-label-file", default=None)
    parser.add_argument("--calibration-class-file", default=None)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-times", type=int, default=6)
    parser.add_argument(
        "--vae",
        type=str,
        default="ema",
        choices=["mse", "ema"],
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=None,
        help="Optional local copy of the selected VAE for offline sampling.",
    )
    parser.add_argument("--generate-cache-books", action="store_true")
    parser.add_argument(
        "--cache-book-path",
        type=str,
        default="./cache_books",
    )
    parser.add_argument(
        "--nonskip-rate",
        type=float,
        default=0,
        help="Initial timestep ratio forced to recompute layer modules.",
    )
    # Selected module-only setting: MSA=0.53, MLP=0.39, one calibration class.
    # DiT-XL/2 512, DDIM-50: 1.754x sampling speedup over Full on
    # 100 classes x 2 seeds; this is not end-to-end latency.
    parser.add_argument("--msa-thres", type=float, default=0.53)
    parser.add_argument("--mlp-thres", type=float, default=0.39)
    add_preset_argument(parser, "dit_module", tiers=("default",))
    parser.add_argument("--num-analysis", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="images")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--save-individual-dir", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=0,
        help="Decode this many samples per VAE call; 0 keeps one-shot decoding.",
    )
    parser.add_argument("--calibration-only", action="store_true")

    debug_args = [
        "--model", "DiT-XL/2",
        "--image-size", "512",
        "--num-classes", "1000",
        "--num-timesteps", "50",
        "--dit-ckpt", "./pretrained_models/DiT-XL-2-512x512.pt",
        "--num-sample-classes", "1",
        "--cfg-scale", "4.0",
        "--seed", "0",
        "--sample-times", "1",
        "--nonskip-rate", "0",
        "--msa-thres", "0.53",
        "--mlp-thres", "0.39",
        "--num-analysis", "1",
        "--generate-cache-books",
    ]

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args(debug_args)
    args = apply_preset(
        args,
        "dit_module",
        {"--msa-thres": "msa_thres", "--mlp-thres": "mlp_thres"},
    )
    main(args)
