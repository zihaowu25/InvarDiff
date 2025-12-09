import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import sys
sys.path.append("..")

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def generate_dataset(
    pipe,
    prompts,
    num_inference_steps=28,
    seed=42,
    output_dir="./images"
):
    """
    Generate images using specified model and save them with corresponding indices
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Total prompts: {len(prompts)}")
    times = []

    for idx, prompt in enumerate(tqdm(prompts, desc=f"Generating images")):
        start_time = time.time()
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images[0]
        times.append(time.time() - start_time)
        # Save with index: image_000.png, image_001.png, etc.
        image_path = os.path.join(output_dir, f"image_{idx:03d}.png")
        image.save(image_path)

    if len(times)>1:
        times = np.array(times[1:])
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Sampling time: {avg_time:.4f}±{std_time:.4f} s")
    else:
        print(f"Sampling time: {times[0]:.4f} s")
    
    print(f"Images saved to {output_dir}")

def load_image_as_tensor(image_path, device="cuda"):
    """
    Load image and convert to tensor normalized to [-1, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor.to(device)
