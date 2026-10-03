# Fine-Grained Caching for Diffusion Transformers with Few Calibration Conditions

This repository contains the open-source implementation of **Fine-Grained
Caching for Diffusion Transformers with Few Calibration Conditions**.

Training-free, fine-grained offline caching for diffusion transformers. The
current standalone samplers construct a fixed **module-level Cache Book** from
a small calibration set and reuse selected module outputs during generation.
They do not require retraining the denoiser or an online routing network.

An [earlier InvarDiff preprint](https://arxiv.org/abs/2512.05134) describes
cross-step and layer-level caching. This repository also retains separate
step-layer and hybrid experiments; their schedules and reported speedups
should not be treated as direct module-only baselines.

## Repository guide

- **Sampling:** model-specific setup and runnable examples in
  [DiT](DiT/README.md), [FLUX](FLUX/README.md), [Wan2.1](Wan2.1/README.md),
  and [HunyuanVideo](HunyuanVideo/README.md).
- **Compatibility:** separate MagCache and SeaCache integrations for
  [FLUX](FLUX/hybrid_cache/README.md), [Wan2.1](Wan2.1/hybrid_cache/README.md),
  and [HunyuanVideo](HunyuanVideo/hybrid_cache/README.md).
- **Visualization:** [CCMR collectors and plotting](assets/visualization/ccmr/README.md)
  and [reproducible protocols](assets/ccmr_formal/README.md).
- **Evaluation:** [paired image/video metrics and validation](evaluation/README.md).

Source, configuration files, prompt/class banks, tests, and license notices
are included. Model weights, generated experiment artifacts, local tools,
and personal notes are intentionally excluded from Git.

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

Module-only, step-layer and hybrid policies each use **one selected configuration**.
Module-only and step-layer presets are named `default`, and hybrid presets are named `hybrid` in
[`cache_presets.json`](cache_presets.json). Explicit threshold flags override
individual values.

| Module-only model | Module quantiles, in execution-group order | Protected steps | Calibration conditions |
| --- | --- | ---: | ---: |
| DiT | MSA / MLP: `.55 / .55` | 2 | 1 |
| FLUX | Double attention / context attention / FF / context FF / single attention / single MLP: `.70 / .70 / .30 / .23 / .50 / .02` | 2 | 2 |
| Wan | Self-attention / cross-attention / FFN: `.50 / .35 / .20` | 2 | 1 |
| Hunyuan | Double image attention / text attention / image MLP / text MLP / single attention / single MLP: `.90 / .45 / .04 / .12 / 0 / 0` | 2 | 1 |

These settings apply to the evaluated protocols above, not arbitrary samplers.
The separate step-layer samplers combine whole-step reuse with module reuse.
Their defaults were selected using small visual and held-out sets under the
same protocols; these checks are not formal quality benchmarks.

| Step-layer model | Step quantile | Module quantiles, in the order above | Protected steps | Calibration conditions |
| --- | ---: | --- | ---: | ---: |
| DiT | `.55` | `.50 / .15` | 2 | 1 |
| FLUX | `.52` | `1 / 1 / 1 / 1 / 1 / 1` | 2 | 2 |
| Wan | `.63` | `.82 / 1 / .82` | 2 | 1 |
| Hunyuan | `.70` | `.40 / .01 / .20 / .32 / 0 / 0` | 3 | 1 |

Use the corresponding `sample_*_step_layer.py` entrypoint and recalibrate its
Cache Book before generation. Quantiles of one still respect protected and
invalid positions and the final denoising step.

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

## Tests

Install `pytest`, NumPy, pandas, PyYAML, matplotlib, and the relevant model
dependencies, then run from the repository root:

```bash
python -m pytest -q
```

Some integration tests require the corresponding upstream model package.
Tests do not download weights or launch a full generation benchmark.

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
