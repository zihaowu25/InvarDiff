import torch
import lpips
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dit_load_imgs import load_class_labels, generate_dit_dataset, load_image_as_tensor
import sys
sys.path.append("..")
from diffusion import create_diffusion
from download import find_model
from models.dynamic_cache import DiT_models, DynamicDiT

def calculate_lpips_dataset(
    original_dir="./original",
    accelerated_dir="./accelerated",
    device="cuda"
):
    """
    Calculate LPIPS for all paired images and compute mean and std
    """
    print("Loading LPIPS model...")
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

    # seed = 42
    # torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model_name = "DiT-XL/2"
    # image_size = 256
    # num_classes = 1000
    # num_timesteps = 50
    # cfg_scale = 4.0
    # dit_ckpt = "../pretrained_models/DiT-XL-2-256x256.pt"
    # input_size = image_size // 8
    
    # print("Loading DiT model...")
    # diffusion = create_diffusion(str(num_timesteps))
    # base_DiT = DiT_models[model_name](
    #     input_size=input_size,
    #     num_classes=num_classes
    # ).to(device)
    
    # DiT_state_dict = find_model(dit_ckpt)
    # base_DiT.load_state_dict(DiT_state_dict)
    # base_DiT.eval()
    
    # class_labels = load_class_labels("class_labels.txt")
    # print(f"Loaded {len(class_labels)} class labels")
    
    # print("Generating original DiT images...")
    
    # generate_dit_dataset(
    #     model=base_DiT,
    #     diffusion=diffusion,
    #     class_labels=class_labels,
    #     input_size=input_size,
    #     num_inference_steps=num_timesteps,
    #     cfg_scale=cfg_scale,
    #     seed=seed,
    #     output_dir="./base_dit",
    #     device=device
    # )
    
    # num_inference_steps = 50
    # nonskip_rate = 0.0
    # step_thres = 0.61
    # msa_thres = 0.2
    # mlp_thres = 0.2
    
    # cache_book_file = (
    #     f"cache_books_stp{num_inference_steps}_n{nonskip_rate}_sth{step_thres}"
    #     f"_msa{msa_thres}_mlp{mlp_thres}.json"
    # )
  
    # from sample_InvarDiff_dit import load_cache_books
    # step_cache_book, msa_cache_book, mlp_cache_book = load_cache_books(
    #     cache_book_path="../cache_books",
    #     cache_book_file=cache_book_file
    # )
    # print(f"Cache books loaded from: ../cache_books/{cache_book_file}")

    # Dynamic_DiT = DynamicDiT(base_DiT, msa_cache_book, mlp_cache_book, step_cache_book)
    # Dynamic_DiT.eval()
    
    # generate_dit_dataset(
    #     model=Dynamic_DiT,
    #     diffusion=diffusion,
    #     class_labels=class_labels,
    #     input_size=input_size,
    #     num_inference_steps=num_timesteps,
    #     cfg_scale=cfg_scale,
    #     seed=seed,
    #     output_dir="./InvarDiff_slow",
    #     device=device
    # )
    
    print("LPIPS Evaluation...")
    results = calculate_lpips_dataset(
        original_dir="./base_dit",
        accelerated_dir="./InvarDiff_slow",
        device=device
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