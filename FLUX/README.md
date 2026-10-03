# FLUX.1-dev module-level caching

[`sample_flux.py`](sample_flux.py) implements the standalone module-level
Cache Book for FLUX.1-dev. The selected 1024 × 1024, 28-step configuration
uses double-stream attention/context attention/FF/context FF thresholds
`0.70/0.70/0.30/0.23` and single-stream attention/MLP thresholds
`0.50/0.02`, with two protected initial steps at 28 steps
(`--nonskip-rate 0.1`). It averages two calibration prompts by default; callers can
provide another set with repeated `--calibration-prompt` flags or a
`--calibration-prompt-file`.

Install the FLUX dependencies (including PyTorch, diffusers, transformers,
accelerate, safetensors, Pillow, NumPy, and tqdm) and obtain model access
under the provider's terms. From the repository root, after setting
`FLUX_MODEL` to a local model directory or supported model ID:

```bash
python FLUX/sample_flux.py --model-path "$FLUX_MODEL" \
  --prompt "A colorful toucan perched on a mossy branch in a sunlit tropical forest, photorealistic, crisp feathers and soft background." --seed 4102 \
  --output-dir outputs/flux
```

Inference loads the matching configuration-named JSON in `cache_books/FLUX/`
automatically. The step-layer entrypoint uses the same folder with a different
strategy filename.
To recalibrate and then use the resulting book without changing paths:

```bash
python FLUX/sample_flux.py --model-path "$FLUX_MODEL" \
  --generate-cache-books --calibration-only

python FLUX/sample_flux.py --model-path "$FLUX_MODEL" \
  --prompt "A blue kingfisher perched above a river in morning light"
```

Calibration and inference use the same configuration-derived filename in
`cache_books/FLUX/`. Different configurations coexist; recalibrating the same
configuration updates its existing file.

The module-only preset is `default` and is selected automatically. Explicit
threshold flags override its individual values. Cache Books are compatible
only with matching execution settings; recalibrate after changing model
weights, scheduler, step count, resolution, guidance, precision, or module
partition. Hybrid policies each use a single `hybrid` configuration; see
[`hybrid_cache/README.md`](hybrid_cache/README.md). The separate step-layer
entrypoint is [`sample_flux_step_layer.py`](sample_flux_step_layer.py). Its
single `default` uses step quantile `.52`, all six module quantiles `1`, two
protected steps (`--nonskip-rate .08`) and two calibration prompts at 1024 × 1024,
28 steps. These settings were checked on small visual and held-out sets.
Use the commands above with this entrypoint to calibrate and generate.
