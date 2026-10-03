"""Shared cache-preset resolution without changing sampler algorithms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


PRESET_PATH = Path(__file__).with_name("cache_presets.json")
CACHE_BOOKS = Path(__file__).with_name("cache_books")
MODEL_BOOK_DIRECTORIES = {
    "dit": "DiT", "flux": "FLUX", "wan": "Wan2.1", "hunyuan": "HunyuanVideo",
}


def cache_book_directory(policy: str) -> Path:
    """Group all strategies for one model in its shared Cache Book folder."""
    model = policy.split("_", 1)[0]
    if model not in MODEL_BOOK_DIRECTORIES:
        raise ValueError(f"Unknown Cache Book model: {model}")
    return CACHE_BOOKS / MODEL_BOOK_DIRECTORIES[model]

# Sampling and calibration identity only: prompts, seeds, output paths and
# cache-enable flags must not change the reusable policy filename.
BOOK_CONFIG_FIELDS = (
    "model", "num_classes", "image_size", "height", "width", "resolution",
    "aspect_ratio", "task", "size", "frame_num", "video_length",
    "num_timesteps", "num_inference_steps", "sample_steps", "sample_solver",
    "sample_shift", "cfg_scale", "guidance_scale", "sample_guide_scale",
    "dtype", "deterministic_attention", "cfg_distilled", "enable_step_distill",
    "sparse_attn", "num_analysis", "nonskip_rate", "step_thres", "msa_thres",
    "mlp_thres", "attn_thres", "context_attn_thres", "ff_thres",
    "context_ff_thres", "single_attn_thres", "single_mlp_thres",
    "self_attn_thres", "cross_attn_thres", "ffn_thres",
    "double_img_attn_thres", "double_txt_attn_thres", "double_img_mlp_thres",
    "double_txt_mlp_thres", "magcache_thresh", "magcache_K", "magcache_k",
    "retention_ratio", "magcache_ratio_source", "seacache_thresh",
    "seacache_threshold", "use_ret_steps", "power_exp", "spatial_dims",
    "disable_step_cache",
)


def cache_book_filename(args: argparse.Namespace, policy: str) -> str:
    """Configuration-bearing name, following the original step/threshold style."""
    config = {key: getattr(args, key) for key in BOOK_CONFIG_FIELDS if hasattr(args, key)}
    # This entrypoint's geometry is fixed by its implementation, not CLI flags.
    if policy == "flux_step_layer":
        config.setdefault("height", 1024)
        config.setdefault("width", 1024)

    def canonical(value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return format(value, ".17g")
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    encoded = json.dumps({key: canonical(value) for key, value in config.items()}, sort_keys=True)
    digest = hashlib.sha256((policy + ":" + encoded).encode()).hexdigest()[:12]
    parts = ["cache_book", policy]
    if "image_size" in config:
        parts.append(f"{config['image_size']}x{config['image_size']}")
    elif "height" in config and "width" in config:
        parts.append(f"{config['width']}x{config['height']}")
    elif "size" in config:
        parts.append(str(config["size"]).replace("*", "x"))
    elif "resolution" in config:
        parts.append(str(config["resolution"]))
        if "aspect_ratio" in config:
            parts.append(str(config["aspect_ratio"]).replace(":", "x"))
    for label, keys in (
        ("f", ("frame_num", "video_length")),
        ("steps", ("num_timesteps", "num_inference_steps", "sample_steps")),
        ("ns", ("nonskip_rate",)), ("stepth", ("step_thres",)),
        ("msa", ("msa_thres",)), ("mlp", ("mlp_thres",)),
        ("attn", ("attn_thres",)), ("cattn", ("context_attn_thres",)),
        ("ff", ("ff_thres",)), ("cff", ("context_ff_thres",)),
        ("sattn", ("single_attn_thres", "self_attn_thres")),
        ("smlp", ("single_mlp_thres",)), ("cross", ("cross_attn_thres",)),
        ("ffn", ("ffn_thres",)), ("ia", ("double_img_attn_thres",)),
        ("ta", ("double_txt_attn_thres",)), ("im", ("double_img_mlp_thres",)),
        ("tm", ("double_txt_mlp_thres",)), ("mag", ("magcache_thresh",)),
        ("mk", ("magcache_K", "magcache_k")), ("r", ("retention_ratio",)),
        ("sea", ("seacache_thresh", "seacache_threshold")),
        ("kcal", ("num_analysis",)),
    ):
        for key in keys:
            if key in config and config[key] is not None:
                value = config[key]
                parts.append(label + (format(value, "g") if isinstance(value, (int, float)) else str(value)))
                break
    stem = re.sub(r"[^A-Za-z0-9_.-]", "-", "_".join(parts))
    return f"{stem}_{digest}.json"


def configure_cache_book(args: argparse.Namespace, policy: str) -> None:
    """Use the same policy-specific file for calibration and inference.

    Explicit folders keep their existing configuration-derived filenames.
    Boolean cache flags distinguish an omitted option from an explicit opt-out.
    """
    if not hasattr(args, "cache_book_path"):
        return
    # The historical Hunyuan MagCache preset name is retained for CLI
    # compatibility; its released Book is policy-specific like the others.
    policy = {"hunyuan_hybrid": "hunyuan_magcache_hybrid"}.get(policy, policy)
    calibrating = any(bool(getattr(args, key, False)) for key in (
        "generate_cache_books", "invardiff_calibration", "finegrained_calibration"
    ))
    for key in ("use_invardiff", "use_finegrained_cache"):
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, not calibrating)
    if args.cache_book_path is None:
        args.cache_book_path = str(cache_book_directory(policy))
    if getattr(args, "cache_book_file", None) is None:
        args.cache_book_file = cache_book_filename(args, policy)
    folder = Path(args.cache_book_path).resolve()
    file = getattr(args, "cache_book_file", None)
    enabled = not any(getattr(args, key, None) is False for key in (
        "use_invardiff", "use_finegrained_cache"
    ))
    if not calibrating and enabled and file and (folder / file).is_file():
        release = json.loads((folder / file).read_text(encoding="utf-8")).get("release")
        if release is not None:
            if release.get("policy") != policy:
                raise ValueError(f"Cache Book belongs to {release.get('policy')}, not {policy}.")
            for key, expected in release.get("execution_args", {}).items():
                if not hasattr(args, key):
                    raise ValueError(f"Sampler cannot validate released setting {key}.")
                actual = getattr(args, key)
                if actual != expected:
                    raise ValueError(f"Released Cache Book requires {key}={expected!r}, got {actual!r}; recalibrate for changed settings.")


def load_presets(path: Path = PRESET_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported cache preset schema")
    return payload["policies"]


def same_execution_config(saved: object, expected: dict) -> bool:
    """Compare Cache Book settings without treating a preset label as a setting."""
    if not isinstance(saved, dict):
        return False
    return {
        key: value for key, value in saved.items() if key != "cache_preset"
    } == {
        key: value for key, value in expected.items() if key != "cache_preset"
    }


def add_preset_argument(
    parser: argparse.ArgumentParser,
    policy: str,
    tiers: Iterable[str] | None = None,
) -> None:
    tiers = tuple(load_presets()[policy]["presets"] if tiers is None else tiers)
    parser.add_argument(
        "--cache-preset",
        choices=tiers,
        default="default" if "default" in tiers else tiers[0],
        help=f"Threshold preset from cache_presets.json for {policy}.",
    )


def apply_preset(
    args: argparse.Namespace,
    policy: str,
    flag_to_attr: dict[str, str],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Apply preset values except for threshold flags explicitly supplied.

    argparse does not retain whether a value came from a default. Inspecting
    the original option tokens preserves the documented precedence:
    explicit threshold > selected preset > parser legacy default.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    tier = args.cache_preset
    policies = load_presets()
    if policy not in policies or tier not in policies[policy]["presets"]:
        raise ValueError(f"Missing cache preset {policy}:{tier}")
    values = policies[policy]["presets"][tier]["thresholds"]
    explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for flag, attr in flag_to_attr.items():
        if flag not in explicit:
            if attr not in values:
                raise ValueError(f"Preset {policy}:{tier} lacks {attr}")
            setattr(args, attr, float(values[attr]))
    args.cache_preset_resolved = tier
    args.cache_resolved_thresholds = {
        attr: float(getattr(args, attr)) for attr in flag_to_attr.values()
    }
    configure_cache_book(args, policy)
    return args
