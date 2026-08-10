import argparse
import statistics
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image


DEFAULT_PROMPTS = [
    "a photo of a broccoli",
    "A snowy mountain village at dusk, glowing windows and smoke rising.",
    "A golden retriever puppy jumping through autumn leaves.",
    "A surreal underwater city with glowing jellyfish and crystal towers.",
    "A group of astronauts planting a flag on Mars, red rocky landscape.",
    "A vintage sports car speeding down a coastal highway at sunset.",
    (
        "A stylish woman walks down a Tokyo street filled with warm glowing "
        "neon and animated city signage. She wears a black leather jacket, "
        "a long red dress, and black boots, and carries a black purse. She "
        "wears sunglasses and red lipstick. She walks confidently and "
        "casually. The street is damp and reflective, creating a mirror "
        "effect of the colorful lights. Many pedestrians walk about."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the unmodified FLUX.1-dev Diffusers pipeline."
    )
    parser.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument(
        "--model-path",
        default=(
            "/root/autodl-tmp/InvarDiff/FLUX/"
            "models--black-forest-labs--FLUX.1-dev"
        ),
        help=(
            "Local model snapshot or Hugging Face cache repository root. "
            "Pass an empty string to load --model-id from the Hub."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="/root/autodl-tmp/InvarDiff/FLUX",
    )
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require a complete local Hugging Face snapshot.",
    )
    return parser.parse_args()


def resolve_local_snapshot(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()

    if (path / "model_index.json").is_file():
        return path

    ref_file = path / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(
            f"No model_index.json or refs/main found under {path}"
        )

    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot_path = path / "snapshots" / revision
    if not (snapshot_path / "model_index.json").is_file():
        raise FileNotFoundError(
            f"No model_index.json found in resolved snapshot {snapshot_path}"
        )

    return snapshot_path


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FLUX baseline benchmark.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading the original FLUX.1-dev pipeline...")

    if args.model_path:
        model_source = resolve_local_snapshot(args.model_path)
        local_files_only = True
        print(f"Using local model snapshot: {model_source}")
    else:
        model_source = args.model_id
        local_files_only = args.local_files_only

    pipe = FluxPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        local_files_only=local_files_only,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print(
        "Benchmark configuration: "
        f"{args.width}x{args.height}, "
        f"{args.num_inference_steps} steps, "
        f"guidance={args.guidance_scale}, "
        f"seed={args.seed}, "
        "dtype=bfloat16"
    )
    print("Run 0 is a warm-up and is excluded from the final statistics.")

    images = []
    times = []

    with torch.inference_mode():
        for index, prompt in enumerate(DEFAULT_PROMPTS):
            generator = torch.Generator(device="cpu").manual_seed(args.seed)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            result = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            images.append(result.images[0])

            label = "warm-up" if index == 0 else "measured"
            print(f"Run {index} ({label}): {elapsed:.4f} s")

            del result

    measured_times = times[1:]
    mean_time = statistics.mean(measured_times)
    std_time = statistics.pstdev(measured_times)

    print(f"Measured runs: {len(measured_times)}")
    print(f"Original FLUX sampling time: {mean_time:.4f} ± {std_time:.4f} s")

    image_width, image_height = images[0].size
    combined = Image.new(
        "RGB",
        (image_width * len(images), image_height),
    )
    for index, image in enumerate(images):
        combined.paste(image, (index * image_width, 0))

    output_dir = Path(__file__).resolve().parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%m%d_%H%M%S")
    save_path = output_dir / (
        f"baseline_imgs_stp{args.num_inference_steps}"
        f"_cfg{args.guidance_scale}"
        f"_seed{args.seed}_{timestamp}.png"
    )
    combined.save(save_path)
    print(f"Images saved to {save_path}")


if __name__ == "__main__":
    main()
