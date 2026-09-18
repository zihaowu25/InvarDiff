# InvarDiff

Training-free, fine-grained offline caching for diffusion transformers. The
current standalone samplers construct a fixed **module-level Cache Book** from
a small calibration set and reuse selected module outputs during generation.
They do not require retraining the denoiser or an online routing network.

An [earlier InvarDiff preprint](https://arxiv.org/abs/2512.05134) describes
cross-step and layer-level caching. This repository also retains separate
step-layer and hybrid experiments; their schedules and reported speedups
should not be treated as direct module-only baselines.

## How it works

1. Run the pretrained denoiser on deterministic calibration trajectories and
   measure adjacent module-feature displacements.
2. Rank timestep–layer–module positions by the normalized adjacent L1
   displacement ratio. A second full-compute pass updates the scoring
   reference according to the provisional reuse history, then writes the
   final Cache Book.
3. During generation, compute or reuse each module according to that fixed
   book. Reuse reads the most recent computed output for the same branch; it
   does not mix conditional and unconditional CFG features.

The score prioritizes reuse; it is **not** a bound on image or video error.
Cache Books are tied to the model, scheduler, step count, resolution/frame
count, guidance, precision, and module partition. Recalibrate or validate
transfer when these settings change.

## Supported standalone samplers

| Model | Module-only entry point | Evaluated setting |
| --- | --- | --- |
| DiT-XL/2 | [`DiT/sample_dit.py`](DiT/sample_dit.py) | 512 × 512, DDIM-50 |
| FLUX.1-dev | [`FLUX/sample_flux.py`](FLUX/sample_flux.py) | 1024 × 1024, 28 steps |
| Wan2.1-T2V-1.3B | [`Wan2.1/sample_wan.py`](Wan2.1/sample_wan.py) | 832 × 480, 81 frames, 50 steps |
| HunyuanVideo-1.5 | [`HunyuanVideo/sample_hunyuan.py`](HunyuanVideo/sample_hunyuan.py) | 720p, 121 frames, 50 steps |

Each module-only sampler has **one selected configuration**, named `default`
in [`cache_presets.json`](cache_presets.json). The CLI selects it automatically;
individual threshold flags still override its values. The selected settings
were visually screened on limited conditions and are not universal quality
guarantees. Other policies in the JSON file belong to separate step-layer or
hybrid experiments and retain their own names.

## Getting started

Install PyTorch and the dependencies of the model you intend to run, then
obtain its pretrained weights under the model provider's terms. Model-specific
setup and arguments are documented in [`DiT/README.md`](DiT/README.md),
[`FLUX/README.md`](FLUX/README.md), [`Wan2.1/README.md`](Wan2.1/README.md),
and [`HunyuanVideo/README.md`](HunyuanVideo/README.md). From the repository
root, inspect the sampler options with, for example:

```bash
python DiT/sample_dit.py --help
python FLUX/sample_flux.py --help
python Wan2.1/sample_wan.py --help
python HunyuanVideo/sample_hunyuan.py --help
```

Calibrate before generation. For example, after setting `DIT_CKPT` to a DiT
checkpoint and installing its VAE:

```bash
python DiT/sample_dit.py \
  --dit-ckpt "$DIT_CKPT" --image-size 512 --num-timesteps 50 \
  --num-analysis 1 --generate-cache-books --calibration-only

python DiT/sample_dit.py \
  --dit-ckpt "$DIT_CKPT" --image-size 512 --num-timesteps 50 \
  --sample-times 1 --output-dir outputs/dit
```

The FLUX sampler defaults to two calibration prompts; use
`--calibration-prompt` more than once or `--calibration-prompt-file` to choose
another calibration set. Video samplers require the corresponding official
model repository and checkpoint; see their model-specific READMEs. Generated
media, Cache Books, model weights, environments, and experiment runs are
ignored by Git.

## Reproducibility and scope

- Compare methods under the same model, scheduler, resolution, frame count,
  guidance, precision, prompts/classes, and initial noise.
- Report the measurement boundary: denoising time, sampling time, and
  end-to-end time are not interchangeable. Cache reads/writes and model
  loading can materially affect wall-clock speed.
- For compatibility experiments, compare a fixed cross-step policy **with
  versus without** module caching; a standalone module policy and a
  cross-step policy do not form a like-for-like ranking.
- The repository does not include pretrained model weights. Outputs and
  experiment reports are local artifacts, not source files.

## Citation

The earlier cross-scale InvarDiff preprint is available as:

```bibtex
@misc{wu2025invardiffcrossscaleinvariancecaching,
  title={InvarDiff: Cross-Scale Invariance Caching for Accelerated Diffusion Models},
  author={Zihao Wu},
  year={2025},
  eprint={2512.05134},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.05134}
}
```

Please follow the licenses and use restrictions of each upstream model and
its weights. HunyuanVideo-specific license and attribution files are included
in [`HunyuanVideo/`](HunyuanVideo/).
