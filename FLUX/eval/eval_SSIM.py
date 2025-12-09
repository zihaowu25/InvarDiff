import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from load_imgs import load_image_as_tensor
from pytorch_msssim import ssim

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    
    Args:
        img1: tensor in range [-1, 1], shape (1, C, H, W)
        img2: tensor in range [-1, 1], shape (1, C, H, W)
    
    Returns:
        SSIM value (0-1, higher is better)
    """
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)
    
    return ssim_value.item()

def calculate_ssim_dataset(
    original_dir="./original",
    accelerated_dir="./accelerated",
    device="cuda"
):
    """
    Calculate SSIM for all paired images and compute mean and std
    """
    print("Calculating SSIM...")
    
    original_images = sorted(Path(original_dir).glob("image_*.png"))
    accelerated_images = sorted(Path(accelerated_dir).glob("image_*.png"))
    
    print(f"Found {len(original_images)} original images")
    print(f"Found {len(accelerated_images)} accelerated images")
    
    if len(original_images) != len(accelerated_images):
        print("Warning: Number of images do not match!")
    
    num_images = min(len(original_images), len(accelerated_images))
    ssim_scores = []
    
    print(f"\nCalculating SSIM for {num_images} image pairs...")
    
    for idx in tqdm(range(num_images), desc="SSIM calculation"):
        orig_path = original_images[idx]
        accel_path = accelerated_images[idx]
        
        img_orig = load_image_as_tensor(orig_path, device)
        img_accel = load_image_as_tensor(accel_path, device)
        
        ssim_score = calculate_ssim(img_orig, img_accel)
        
        ssim_scores.append(ssim_score)
        
    ssim_scores = np.array(ssim_scores)
    
    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    min_ssim = np.min(ssim_scores)
    max_ssim = np.max(ssim_scores)
    
    return {
        "mean": mean_ssim,
        "std": std_ssim,
        "min": min_ssim,
        "max": max_ssim,
        "num_images": len(ssim_scores),
        "all_scores": ssim_scores.tolist()
    }

def main():
    original_dir = "./base_flux"
    accelerated_dir = "./InvarDiff_flux"
    
    print("SSIM Evaluation...")
    results = calculate_ssim_dataset(
        original_dir=original_dir,
        accelerated_dir=accelerated_dir,
        device="cuda"
    )
    
    print("SSIM Evaluation Results")
    print(f"Number of image pairs: {results['num_images']}")
    print("SSIM: {:.4f}±{:.4f}".format(results['mean'], results['std']))
    print(f"Mean SSIM:            {results['mean']:.6f}")
    print(f"Std SSIM:             {results['std']:.6f}")
    print(f"Min SSIM:             {results['min']:.6f}")
    print(f"Max SSIM:             {results['max']:.6f}")

if __name__ == "__main__":
    main()