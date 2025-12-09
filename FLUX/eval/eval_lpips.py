import torch
import lpips
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from diffusers import FluxPipeline
from load_imgs import load_prompts, generate_dataset, load_image_as_tensor
import sys
sys.path.append("..")
from dynamic_flux import DynamicFluxTransformer2DModel
from sample_InvarDiff_flux import load_cache_books

def calculate_lpips_dataset(
    original_dir="./original",
    accelerated_dir="./accelerated",
    device="cuda"
):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    original_images = sorted(Path(original_dir).glob("image_*.png"))
    accelerated_images = sorted(Path(accelerated_dir).glob("image_*.png"))
    
    print(f"Found {len(original_images)} original images")
    print(f"Found {len(accelerated_images)} accelerated images")
    
    if len(original_images) != len(accelerated_images):
        print("Warning: Number of images do not match!")
    
    num_images = min(len(original_images), len(accelerated_images))
    lpips_scores = []
    
    print(f"\nCalculating LPIPS for {num_images} image pairs...")
    
    for idx in tqdm(range(num_images), desc="LPIPS calculation"):
        orig_path = original_images[idx]
        accel_path = accelerated_images[idx]
        img_orig = load_image_as_tensor(orig_path, device)
        img_accel = load_image_as_tensor(accel_path, device)

        with torch.no_grad():
            lpips_score = loss_fn_alex(img_orig, img_accel)
        
        lpips_scores.append(lpips_score.item())
        
    lpips_scores = np.array(lpips_scores)
    
    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    min_lpips = np.min(lpips_scores)
    max_lpips = np.max(lpips_scores)
    
    return {
        "mean": mean_lpips,
        "std": std_lpips,
        "min": min_lpips,
        "max": max_lpips,
        "num_images": len(lpips_scores),
        "all_scores": lpips_scores.tolist()
    }

def main():
    # seed=42
    # torch.set_grad_enabled(False)
    # num_inference_steps = 28
    # nonskip_rate = 0.1
    # step_thres = 0.7

    # attn_thres=0.68
    # ff_thres=0.68
    # context_ff_thres=0.68

    # Single_attn_thres=0.68
    # Single_mlp_thres=0.68

    # print("Loading FLUX pipeline...")
    # pipe = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev",
    #     torch_dtype=torch.bfloat16, 
    #     cache_dir="/root/autodl-tmp/InvarDiff/FLUX"
    # )
    # original_transformer = pipe.transformer
    
    # dynamic_model = DynamicFluxTransformer2DModel(
    #     original_transformer,
    #     num_inference_steps,
    # )
    # dynamic_model.eval()
    # pipe.transformer = dynamic_model
    # cache_book_file = (
    #         f"cache_books_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
    #         f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
    #         f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}.json"
    #     )
    # step_cache_book, transformer_cache_book, single_transformer_cache_book = load_cache_books(
    #     cache_book_path="../cache_books",
    #     cache_book_file=cache_book_file
    # )
    # dynamic_model.init_cache_book(transformer_cache_book, single_transformer_cache_book, step_cache_book)
    # pipe.transformer = dynamic_model
    # pipe.to("cuda")

    original_dir = "./base_flux"
    accelerated_dir = "./InvarDiff_flux"
    # prompts = load_prompts("prompts.txt")
    # generate_dataset(
    #     pipe=pipe,
    #     prompts=prompts,
    #     num_inference_steps=28,
    #     seed=seed,
    #     output_dir=accelerated_dir
    # )
    
    print("LPIPS Evaluation...")
    results = calculate_lpips_dataset(
        original_dir=original_dir,
        accelerated_dir=accelerated_dir,
        device="cuda"
    )
    print("LPIPS Evaluation Results")
    print(f"Number of image pairs: {results['num_images']}")
    print("LPIPS: {:.4f}±{:.4f}".format(results['mean'], results['std']))
    print(f"Mean LPIPS:            {results['mean']:.6f}")
    print(f"Std LPIPS:             {results['std']:.6f}")
    print(f"Min LPIPS:             {results['min']:.6f}")
    print(f"Max LPIPS:             {results['max']:.6f}")

if __name__ == "__main__":
    main()