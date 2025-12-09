import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dit_load_imgs import load_image_as_tensor

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    
    Args:
        img1: tensor in range [-1, 1]
        img2: tensor in range [-1, 1]
    
    Returns:
        PSNR value in dB
    """
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse < 1e-10:
        return 100.0
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()

def calculate_psnr_dataset(
    original_dir="./original",
    accelerated_dir="./accelerated",
    device="cuda"
):
    print("Calculating PSNR...")
    original_images = sorted(Path(original_dir).glob("image_*.png"))
    accelerated_images = sorted(Path(accelerated_dir).glob("image_*.png"))
    
    print(f"Found {len(original_images)} original images")
    print(f"Found {len(accelerated_images)} accelerated images")
    
    if len(original_images) != len(accelerated_images):
        print("Warning: Number of images do not match!")
    
    num_images = min(len(original_images), len(accelerated_images))
    psnr_scores = []
    
    print(f"\nCalculating PSNR for {num_images} image pairs...")
    
    for idx in tqdm(range(num_images), desc="PSNR calculation"):
        orig_path = original_images[idx]
        accel_path = accelerated_images[idx]
        
        img_orig = load_image_as_tensor(orig_path, device)
        img_accel = load_image_as_tensor(accel_path, device)
        
        psnr_score = calculate_psnr(img_orig, img_accel)
        
        psnr_scores.append(psnr_score)
        
    psnr_scores = np.array(psnr_scores)
    
    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    min_psnr = np.min(psnr_scores)
    max_psnr = np.max(psnr_scores)
    
    return {
        "mean": mean_psnr,
        "std": std_psnr,
        "min": min_psnr,
        "max": max_psnr,
        "num_images": len(psnr_scores),
        "all_scores": psnr_scores.tolist()
    }

def main():

    print("PSNR Evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = calculate_psnr_dataset(
        original_dir="./base_dit",
        accelerated_dir="./InvarDiff_slow",
        device=device
    )
    print("PSNR Evaluation Results")
    print(f"Number of image pairs: {results['num_images']}")
    print("PSNR: {:.4f}±{:.4f} dB".format(results['mean'], results['std']))
    print(f"Mean PSNR:            {results['mean']:.6f} dB")
    print(f"Std PSNR:             {results['std']:.6f} dB")
    print(f"Min PSNR:             {results['min']:.6f} dB")
    print(f"Max PSNR:             {results['max']:.6f} dB")

if __name__ == "__main__":
    main()