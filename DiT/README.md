# DiT module-level caching

[`sample_dit.py`](sample_dit.py) implements the standalone module-level
Cache Book for DiT-XL/2. Its selected configuration uses MSA threshold
`0.53`, MLP threshold `0.39`, and one calibration class at 512 × 512 with
50 DDIM steps. These are model- and sampler-specific settings, not universal
thresholds. The separate [`sample_dit_step_layer.py`](sample_dit_step_layer.py)
contains the cross-step/step-layer policy.

## Setup

Install PyTorch, torchvision, diffusers, transformers, accelerate, and timm.
Download the DiT checkpoint with [`download.py`](download.py), or supply your
own checkpoint using `--dit-ckpt`. The VAE can be loaded from its model ID or
provided with `--vae-path`.

## Calibrate and generate

Run from the repository root after setting `DIT_CKPT` to the checkpoint file:

```bash
python DiT/sample_dit.py \
  --dit-ckpt "$DIT_CKPT" --image-size 512 --num-timesteps 50 \
  --num-analysis 1 --generate-cache-books --calibration-only

python DiT/sample_dit.py \
  --dit-ckpt "$DIT_CKPT" --image-size 512 --num-timesteps 50 \
  --sample-times 1 --output-dir outputs/dit
```

The module-only preset is `default` and is selected automatically.
`--msa-thres` and `--mlp-thres` override individual values. Calibrate again
after changing the model, resolution, sampling schedule, or thresholds; a
512 × 512 Cache Book should not be silently reused for 256 × 256 inference.
The generated Cache Book and images are ignored by Git.
