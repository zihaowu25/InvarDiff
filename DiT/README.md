# DiT module-level caching

[`sample_dit.py`](sample_dit.py) implements the standalone module-level
Cache Book for DiT-XL/2. Its selected configuration uses MSA threshold
`0.55`, MLP threshold `0.55`, two protected initial steps (`--nonskip-rate 0.04`),
and one calibration class at 512 × 512 with 50 DDIM steps.
These are model- and sampler-specific settings, not universal
thresholds. The separate [`sample_dit_step_layer.py`](sample_dit_step_layer.py)
combines whole-step and module reuse. Its single `default` configuration at
512 × 512, DDIM-50 uses step/MSA/MLP quantiles `.55/.50/.15`, two protected
initial steps and one calibration class. These defaults were checked on
small visual and held-out sets. Use the same calibration/generation commands
below with the step-layer entrypoint.

## Setup

Install PyTorch, torchvision, diffusers, transformers, accelerate, and timm.
Download the DiT checkpoint with [`download.py`](download.py), or supply your
own checkpoint using `--dit-ckpt`. The VAE can be loaded from its model ID or
provided with `--vae-path`.

DiT-derived code and pretrained weights retain the upstream
[CC-BY-NC-4.0 license](LICENSE-DIT); see the repository's
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Load and generate

Run from the repository root after setting `DIT_CKPT` to the checkpoint file:

```bash
python DiT/sample_dit.py \
  --dit-ckpt "$DIT_CKPT" --image-size 512 --num-timesteps 50 \
  --class-label-file assets/demo/dit_module_class.txt --seed 3500 \
  --sample-times 1 --output-dir outputs/dit
```

The matching configuration-named JSON in `cache_books/DiT/` is loaded by
default, without calibration. The step-layer entrypoint uses the same folder
with a different strategy filename.
Both default to 512 × 512 and DDIM-50. `--vae-path` can point to a local VAE.
For a checked step-layer demo, use `sample_dit_step_layer.py` with
`--class-label-file assets/demo/dit_step_layer_class.txt --seed 3100`.

## Optional recalibration

```bash
python DiT/sample_dit.py --dit-ckpt "$DIT_CKPT" \
  --num-analysis 1 --seed 2027 --generate-cache-books --calibration-only

python DiT/sample_dit.py --dit-ckpt "$DIT_CKPT" \
  --sample-times 1 --output-dir outputs/dit_custom
```

The module-only preset is `default` and is selected automatically.
`--msa-thres` and `--mlp-thres` override individual values. Calibrate again
after changing the model, resolution, sampling schedule, or thresholds; a
512 × 512 Cache Book should not be silently reused for 256 × 256 inference.
Custom Cache Books and images are ignored by Git; the verified bundled books
are published. Calibration and inference both use `cache_books/DiT/`
and automatically derive the filename from the configuration.
Different configurations coexist; recalibrating the same configuration updates its file.
