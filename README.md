# Fine-Grained Caching for Diffusion Transformers with Few Calibration Conditions

Open-source implementation of **Fine-Grained Caching for Diffusion Transformers
with Few Calibration Conditions**. A few offline calibration trajectories
produce a fixed module-level Cache Book for training-free inference.
Step-layer caching and MagCache/SeaCache hybrid implementations are also included.

## Installation

Use Python 3.11 and a CUDA-compatible PyTorch installation.

```bash
git clone --branch dev-local https://github.com/zihaowu25/InvarDiff.git
cd InvarDiff
pip install -r requirements.txt
```

`requirements.txt` covers DiT, FLUX and Wan. For Wan, install the official
[Wan2.1](https://github.com/Wan-Video/Wan2.1) source and add it to `PYTHONPATH`.
For Hunyuan, use a separate environment with the official
[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
dependencies (PyTorch 2.6.0, diffusers 0.35.0, transformers 4.57.1).
Place its checkout beside InvarDiff or set `HUNYUAN_REPO` to its location.
Download each model's pretrained weights separately.

## Models and defaults

| Model | Module-cache entry point | Default sampling setting |
| --- | --- | --- |
| [DiT-XL/2](DiT/README.md) | `DiT/sample_dit.py` | 512 × 512, DDIM-50 |
| [FLUX.1-dev](FLUX/README.md) | `FLUX/sample_flux.py` | 1024 × 1024, 28 steps |
| [Wan2.1-T2V-1.3B](Wan2.1/README.md) | `Wan2.1/sample_wan.py` | 832 × 480, 81 frames, 50 steps |
| [HunyuanVideo-1.5](HunyuanVideo/README.md) | `HunyuanVideo/sample_hunyuan.py` | 720p, 121 frames, 50 steps |

Each sampler uses one default configuration from
[`cache_presets.json`](cache_presets.json). Use `sample_*_step_layer.py`
for combined step/module caching. Hybrid instructions:
[FLUX](FLUX/hybrid_cache/README.md),
[Wan](Wan2.1/hybrid_cache/README.md),
[Hunyuan](HunyuanVideo/hybrid_cache/README.md).

## Bundled Cache Books: inference and calibration

Samplers load the matching book from `cache_books/<model>/` by default.
See the model links above for checkpoint setup and generation examples.

For DiT, set `DIT_CKPT` to the 512-resolution checkpoint, then run:

```bash
python DiT/sample_dit.py --dit-ckpt "$DIT_CKPT" \
  --class-label-file assets/demo/dit_module_class.txt --seed 3500 \
  --sample-times 1 --output-dir outputs/dit
```

To calibrate a new book:

```bash
python DiT/sample_dit.py --dit-ckpt "$DIT_CKPT" \
  --num-analysis 1 --seed 2027 --generate-cache-books --calibration-only
```

| Entry type | Calibration flag |
| --- | --- |
| DiT / FLUX module or step-layer | `--generate-cache-books --calibration-only` |
| Wan / Hunyuan module or step-layer | `--invardiff_calibration` |
| FLUX hybrid | `--finegrained-calibration` |
| Wan / Hunyuan hybrid | `--finegrained_calibration` |

Books are named by configuration. Recalibrate after changing thresholds,
weights or sampling settings; the same configuration overwrites its existing
book. To select another book, use `--cache-book-path` / `--cache-book-file`
for DiT/FLUX, or `--cache_book_path` / `--cache_book_file` for video.
Details and released configurations: [Cache Books](cache_books/README.md).

## Evaluation and visualization

- [Evaluation and tests](evaluation/README.md)
- [CCMR visualization](assets/visualization/ccmr/README.md)
- [Visualization protocols](assets/ccmr_formal/README.md)

## License

Original contributions: [Apache-2.0](LICENSE.txt).
Third-party code and model weights retain their upstream terms; see
[NOTICE](NOTICE), [DiT](DiT/LICENSE-DIT) and
[HunyuanVideo](HunyuanVideo/LICENSE-HUNYUAN).
