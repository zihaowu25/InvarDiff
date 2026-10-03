# Wan2.1 Finegrained Hybrid Cache

This directory contains two standalone Wan2.1 samplers that combine a
whole-step cache with Finegrained Cache at the residual branches inside every
Wan Transformer block:

- `sample_wan_magcache_hybrid.py`
- `sample_wan_seacache_hybrid.py`

The scripts do not import one another, the existing Wan sampling scripts, or
executable files from the local open-source checkouts. Each file includes its
own Wan orchestration, patched forward, calibration, Cache Book validation,
timing, and output saving logic.

## Hybrid policy

The layer cache points are `self_attn`, `cross_attn`, and `ffn`. Their scores
use the same chunked FP32 relative L1 rate with compressed state:

```text
R = ||x_post - x + 1e-8||_1 / ||x - x_prev + 1e-8||_1
```

Wan calls the model twice per diffusion step. Even runtime calls are
conditional and odd calls are unconditional. Both branches share one layer
Cache Book, while their runtime tensors and all whole-step policy states remain
strictly separate. Calibration uses conditional layer features only.

Whole-step cache is evaluated first. A whole-step hit bypasses all Transformer
blocks and adds the cached residual to the current embedded tokens; the head
and unpatchify operations still run. Layer state is untouched on the hit. The
next computed call for that same CFG branch force-refreshes all layer caches.

Calibration is exactly two complete denoising passes:

1. A raw pass builds a provisional layer Cache Book and observes the step
   policy without skipping model computation.
2. A correction pass still computes the complete model while applying the
   provisional step/layer paths only to the feature analyzer's refresh/freeze
   state.

With both `--finegrained_calibration` and `--use_finegrained_cache`, a third
pass is the requested accelerated output generation, not an extra calibration
pass.

## Method support

| Script | Step policy | Supported official Wan tasks | Default threshold |
| --- | --- | --- | --- |
| MagCache | calibrated or published magnitude-ratio policy | T2V, T2I, I2V, VACE | `0.12` |
| SeaCache | dynamic scheduler-aware spectral relative L1 | T2V, T2I, I2V | `0.2` |

FLF2V is rejected by both scripts. SeaCache rejects VACE because its referenced
Wan2.1 implementation does not provide that path.

MagCache embeds the official Wan2.1 T2V-1.3B, T2V-14B, I2V-480P,
I2V-720P, VACE-1.3B, and VACE-14B ratio tables. Joint calibration stores the
conditional/unconditional ratios and static masks in the hybrid Cache Book.
SeaCache remains dynamic during output generation; its observer masks are
diagnostics only and are not saved as runtime policies.

## Usage

Run from the repository root with the official `wan` package importable,
as described in [model setup](../README.md).

Step-cache-only generation (no layer Cache Book):

```bash
python Wan2.1/hybrid_cache/sample_wan_seacache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --base_seed 42 --offload_model False --no-use_finegrained_cache
```

Two-pass Finegrained calibration only:

```bash
python Wan2.1/hybrid_cache/sample_wan_magcache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --cache_book_path ./cache_books/Wan2.1/custom_magcache \
  --finegrained_calibration
```

Load a Cache Book and generate with both cache scales:

```bash
python Wan2.1/hybrid_cache/sample_wan_magcache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --save_file outputs/wan_magcache_hybrid.mp4
```

Calibrate and then generate in one process:

```bash
python Wan2.1/hybrid_cache/sample_wan_seacache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --cache_book_path ./cache_books/Wan2.1/custom_seacache \
  --finegrained_calibration --use_finegrained_cache
```

Inference loads the matching configuration-named JSON in `cache_books/Wan2.1/`
by default at 832 × 480, 81 frames, 50 steps. Custom inference uses
`--cache_book_path` and `--cache_book_file` without the calibration flag.
Calibration without a folder uses that same directory and filename. Different
configurations coexist; recalibrating the same configuration updates its file.

For a layer-only ablation, add `--disable_step_cache` to both calibration and
inference and use the same custom folder. The bundled hybrid book is not a
layer-only book. For example:

```bash
python Wan2.1/hybrid_cache/sample_wan_magcache_hybrid.py \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --cache_book_path ./cache_books/Wan2.1/custom_layer_only \
  --disable_step_cache --finegrained_calibration --use_finegrained_cache \
  --save_file outputs/wan_layer_only.mp4
```

Use `--no-use_finegrained_cache` for step-only generation. Calibration features default to
pinned CPU memory (`--calibration_feature_device cpu`); runtime layer/residual
caches default to GPU. `--runtime_cache_device auto` moves newly stored cache
tensors to pinned CPU once free GPU memory falls below
`--runtime_cache_gpu_reserve_gib` (2 GiB by default).

Automatically derived Cache Book filenames include geometry, frame count, sampling steps,
non-skip rate, all layer thresholds, method-specific parameters and a configuration hash. They do
not contain the seed or trajectory-state labels. Loading strictly validates the
method, formula, task, geometry, solver, shift, model dimensions, thresholds,
and source commit.

## Sources and licensing

- MagCache source: [`Zehong-Ma/MagCache@df81cb1`](https://github.com/Zehong-Ma/MagCache/tree/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.1), Apache-2.0.
- SeaCache source: [`jiwoogit/SeaCache@8dcf490`](https://github.com/jiwoogit/SeaCache/tree/8dcf490/Wan2.1). No LICENSE or NOTICE file exists at that commit; attribution does not itself grant redistribution permission.

The Wan orchestration follows [`Wan-Video/Wan2.1@9737cba`](https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315), licensed under Apache-2.0.
See [third-party notices](../../NOTICE) for the release boundary.
