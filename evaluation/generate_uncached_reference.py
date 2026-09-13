#!/usr/bin/env python3
"""Generate small, unpatched references for zero-cache equivalence checks."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
LOWCOST_PROMPTS = REPO / "evaluation" / "lowcost_prompts.txt"
DIT_CLASSES = (207, 992, 387)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _flux(args: argparse.Namespace) -> None:
    from diffusers import FluxPipeline

    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.count]
    if len(prompts) != args.count:
        raise ValueError(f"Expected {args.count} prompts, found {len(prompts)}")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    images = []
    with torch.inference_mode():
        for prompt in prompts:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            images.append(
                pipe(
                    prompt=prompt,
                    height=args.image_size,
                    width=args.image_size,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg_scale,
                    generator=generator,
                ).images[0]
            )
    grid = Image.new("RGB", (args.image_size * len(images), args.image_size))
    for index, image in enumerate(images):
        grid.paste(image, (index * args.image_size, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)


def _dit(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO / "DiT"))
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image

    from diffusion import create_diffusion
    from download import find_model
    from models.dynamic_cache import DiT_models, DynamicDiT
    from sample_dit import load_cache_books

    _seed_everything(args.seed)
    input_size = args.image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=input_size,
        num_classes=1000,
    ).to("cuda")
    model.load_state_dict(find_model(str(args.dit_ckpt)))
    model.eval()
    diffusion = create_diffusion(str(args.steps))
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        local_files_only=True,
    ).to("cuda")
    labels = DIT_CLASSES[: args.count]
    z = torch.randn(len(labels), 4, input_size, input_size, device="cuda")
    y = torch.tensor(labels, device="cuda")
    z = torch.cat([z, z], dim=0)
    y = torch.cat([y, torch.tensor([1000] * len(labels), device="cuda")])
    def sample(active_model):
        result = diffusion.ddim_sample_loop(
            active_model.forward_with_cfg,
            z.shape,
            z.clone(),
            clip_denoised=False,
            model_kwargs={"y": y, "cfg_scale": args.cfg_scale},
            progress=False,
            device="cuda",
        )
        result, _ = result.chunk(2, dim=0)
        return vae.decode(result / 0.18215).sample

    with torch.inference_mode():
        samples = sample(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        samples,
        args.output,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )
    if args.q0_output:
        if not args.q0_cache_book:
            raise ValueError("--q0-cache-book is required with --q0-output")
        step, msa, mlp = load_cache_books(
            str(args.q0_cache_book.parent), args.q0_cache_book.name
        )
        wrapped = DynamicDiT(model, msa, mlp, step).eval()
        with torch.inference_mode():
            q0_samples = sample(wrapped)
        args.q0_output.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            q0_samples,
            args.q0_output,
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("dit", "flux"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--prompt-file", type=Path, default=LOWCOST_PROMPTS)
    parser.add_argument("--model-path", type=Path, default=WORKSPACE / "models" / "FLUX.1-dev")
    parser.add_argument(
        "--dit-ckpt",
        type=Path,
        default=REPO / "DiT" / "pretrained_models" / "DiT-XL-2-256x256.pt",
    )
    parser.add_argument("--q0-cache-book", type=Path)
    parser.add_argument("--q0-output", type=Path)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.model == "flux":
        args.steps = args.steps or 28
        args.image_size = args.image_size or 1024
        args.cfg_scale = 3.5 if args.cfg_scale == 4.0 else args.cfg_scale
        _flux(args)
    else:
        args.steps = args.steps or 250
        args.image_size = args.image_size or 256
        _dit(args)
    print(f"Saved unpatched {args.model} reference: {args.output}")


if __name__ == "__main__":
    main()
