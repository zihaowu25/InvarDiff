#!/usr/bin/env python3
"""Low-cost pairwise CCMR collector for HunyuanVideo-1.5.

This is a read-only observer: cache decisions stay disabled, both prompts use
the same initial latent, and only scalar per-step/per-layer statistics are
written.  It reuses the exact six feature locations from sample_hunyuan.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OFFICIAL_ROOT = WORKSPACE_ROOT / "open-source" / "HunyuanVideo-1.5"
for path in (OFFICIAL_ROOT, REPO_ROOT / "HunyuanVideo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sample_hunyuan as sampler  # noqa: E402
from hyvideo.commons import PIPELINE_CONFIGS  # noqa: E402
from hyvideo.commons.infer_state import initialize_infer_state  # noqa: E402
from hyvideo.pipelines.hunyuan_video_pipeline import (  # noqa: E402
    HunyuanVideo_1_5_Pipeline,
)


EPS = 1e-12
CHUNK = 1_048_576


def _energy(value: torch.Tensor) -> float:
    flat = value.reshape(-1)
    total = torch.zeros((), device=value.device, dtype=torch.float32)
    for start in range(0, flat.numel(), CHUNK):
        part = flat[start : start + CHUNK].float()
        total += part.square().sum()
    return float((total / flat.numel()).cpu())


class PairwiseCCMRAnalyzer:
    def __init__(self, steps: int, depths: dict[str, int], do_cfg: bool):
        self.steps = steps
        self.depths = depths
        self.do_cfg = do_cfg
        self.previous = {
            name: [None] * depth for name, depth in depths.items()
        }
        self.previous_raw = {
            name: [None] * depth for name, depth in depths.items()
        }
        self.rows: list[dict] = []

    def update_module(self, step_idx, name, layer_idx, feature):
        feature = sampler._conditional_feature(feature.detach(), self.do_cfg)
        if feature.shape[0] != 2:
            raise ValueError(
                f"Pairwise CCMR requires two conditional features, got {feature.shape[0]}"
            )
        pair = feature[0] - feature[1]
        # For K=2, population variance around the condition mean is
        # mean((z0-z1)^2)/4.
        raw = _energy(pair) * 0.25
        previous = self.previous[name][layer_idx]
        previous_raw = self.previous_raw[name][layer_idx]
        if previous is not None:
            diff = _energy(pair - previous.to(pair.device)) * 0.25
            base = 0.5 * (raw + previous_raw)
            ratio = (diff + EPS) / (base + EPS)
            self.rows.append(
                {
                    "module_family": name,
                    "layer_idx": layer_idx,
                    "step_idx": step_idx,
                    "v_raw": raw,
                    "v_raw_prev": previous_raw,
                    "v_base": base,
                    "v_diff": diff,
                    "r_ccmr": ratio,
                    "g_ccmr_db": 10.0 * math.log10((base + EPS) / (diff + EPS)),
                }
            )
        # The low-cost collector runs on an 80 GiB H800.  Keeping the single
        # pair-difference history on-device avoids a host transfer for every
        # module at every step while remaining far below the available memory.
        self.previous[name][layer_idx] = pair.detach().to(dtype=torch.bfloat16)
        self.previous_raw[name][layer_idx] = raw

    def release(self):
        self.previous.clear()
        self.previous_raw.clear()


def _shared_latent_prepare(pipe):
    original = pipe.prepare_latents

    def prepare(
        _self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return original(
                batch_size, num_channels_latents, latent_height, latent_width,
                video_length, dtype, device, generator, latents,
            )
        one = original(
            1, num_channels_latents, latent_height, latent_width, video_length,
            dtype, device, generator, None,
        )
        return one.repeat(batch_size, 1, 1, 1, 1)

    pipe.prepare_latents = types.MethodType(prepare, pipe)


def collect(args):
    version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, "t2v", False, False, False
    )
    if version not in PIPELINE_CONFIGS:
        raise ValueError(f"Unsupported transformer configuration: {version}")
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        args.model_path,
        version,
        create_sr_pipeline=False,
        transformer_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        transformer_init_device=torch.device("cuda"),
    )
    infer_args = sampler.build_parser().parse_args(
        [
            "--prompt", args.prompt_a,
            "--resolution", args.resolution,
            "--model_path", args.model_path,
            "--offloading", "false",
            "--sr", "false",
        ]
    )
    pipe.apply_infer_optimization(
        initialize_infer_state(infer_args), False, False, False
    )
    sampler._patch_model(pipe.transformer)
    _shared_latent_prepare(pipe)
    depths = {
        name: (
            len(pipe.transformer.double_blocks)
            if name.startswith("double")
            else len(pipe.transformer.single_blocks)
        )
        for name in sampler.MODULES
    }
    do_cfg = float(pipe.config.guidance_scale) > 1.0
    analyzer = PairwiseCCMRAnalyzer(args.steps, depths, do_cfg)
    sampler._init_runtime(
        pipe.transformer,
        infer_args,
        args.steps,
        do_cfg,
        analyzer=analyzer,
        use=False,
    )
    pipe(
        prompt=[args.prompt_a, args.prompt_b],
        negative_prompt=["", ""],
        aspect_ratio="16:9",
        video_length=args.frames,
        prompt_rewrite=False,
        num_inference_steps=args.steps,
        seed=args.seed,
        enable_sr=False,
        output_type="latent",
    )
    rows = analyzer.rows
    if not rows:
        raise RuntimeError("No CCMR rows were collected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in sampler.MODULES:
        values = np.asarray(
            [row["g_ccmr_db"] for row in rows if row["module_family"] == name],
            dtype=np.float64,
        )
        if values.size == 0:
            summary[name] = {
                "count": 0,
                "applicable": False,
                "median_db": None,
                "mean_db": None,
                "p05_db": None,
            }
            continue
        summary[name] = {
            "count": int(values.size),
            "applicable": True,
            "median_db": float(np.median(values)),
            "mean_db": float(values.mean()),
            "p05_db": float(np.percentile(values, 5)),
        }
    payload = {
        "protocol": "hunyuan_pairwise_ccmr_v1",
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "seed": args.seed,
        "resolution": args.resolution,
        "frames": args.frames,
        "steps": args.steps,
        "shared_initial_latent": True,
        "cache_enabled": False,
        "summary": summary,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-a", required=True)
    parser.add_argument("--prompt-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", choices=("480p", "720p"), default="720p")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
