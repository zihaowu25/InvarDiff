import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import time

def load_class_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines() if line.strip()]
    return labels

def generate_dit_dataset(
    model,
    diffusion,
    class_labels,
    input_size,
    num_inference_steps=50,
    cfg_scale=4.0,
    seed=42,
    output_dir="./images",
    device="cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    model.eval()
    
    print(f"Total class labels: {len(class_labels)}")
    
    times = []
    for idx, class_label in enumerate(tqdm(class_labels, desc=f"Generating DiT images")):
        z = torch.randn(1, 4, input_size, input_size, device=device)
        y = torch.tensor([class_label], device=device)
        
        z_cfg = torch.cat([z, z], 0)
        y_null = torch.tensor([1000], device=device)
        y_cfg = torch.cat([y, y_null], 0)
        
        model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale)
        start_time = time.time()
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg,
            z_cfg.shape,
            z_cfg,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device
        )
        times.append(time.time() - start_time)
        if hasattr(model, "reset_inference"):
            model.reset_inference()
        samples = samples[:1]
        
        image = vae.decode(samples / 0.18215).sample
        image = torch.clamp(image, -1, 1)
        image = (image + 1) / 2
        image = (image * 255).clamp(0, 255).byte()
        
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_pil = Image.fromarray(image_np)
        
        image_path = os.path.join(output_dir, f"image_{idx:03d}.png")
        image_pil.save(image_path)

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