# InvarDiff for DiT

Fine-grained caching acceleration for DiT (Diffusion Transformer) models.

## Installation

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install timm
```

## Download Pre-trained Models

```bash
python download.py
```

This will download DiT-XL-2-256x256 and DiT-XL-2-512x512 checkpoints to `./pretrained_models/`.

## Quick Start

### Fast Mode (2.8× speedup)

```bash
python sample_InvarDiff_dit.py --model DiT-XL/2 --image-size 256 --num-classes 1000 --num-timesteps 50 --dit-ckpt ./pretrained_models/DiT-XL-2-256x256.pt --num-sample-classes 8 --cfg-scale 4.0 --seed 0 --sample-times 6 --nonskip-rate 0 --step-thres 0.63 --msa-thres 0.22 --mlp-thres 0.22
```

### Slow Mode (2.5× speedup, better quality)

```bash
python sample_InvarDiff_dit.py --model DiT-XL/2 --image-size 256 --num-classes 1000 --num-timesteps 50 --dit-ckpt ./pretrained_models/DiT-XL-2-256x256.pt --num-sample-classes 8 --cfg-scale 4.0 --seed 0 --sample-times 6 --nonskip-rate 0 --step-thres 0.61 --msa-thres 0.2 --mlp-thres 0.2
```

Generated images will be saved in `./images/` directory.

## Cache Calibration

To generate custom cache books for your own settings:

### Fast Mode (for example)

```bash
python sample_InvarDiff_dit.py --model DiT-XL/2 --image-size 256 --num-classes 1000 --num-timesteps 50 --dit-ckpt ./pretrained_models/DiT-XL-2-256x256.pt --num-sample-classes 8 --cfg-scale 4.0 --seed 42 --sample-times 6 --nonskip-rate 0 --step-thres 0.63 --msa-thres 0.22 --mlp-thres 0.22 --num-analysis 16 --generate-cache-books
```
