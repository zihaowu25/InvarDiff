# Wan2.1 Finegrained Hybrid Cache

This directory contains three standalone Wan2.1 samplers that combine a
whole-step cache with Finegrained Cache at the residual branches inside every
Wan Transformer block:

- `sample_wan_magcache_hybrid.py`
- `sample_wan_teacache_hybrid.py`
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
| TeaCache | dynamic polynomial accumulated relative L1 | T2V, T2I, I2V | `0.2` |
| SeaCache | dynamic scheduler-aware spectral relative L1 | T2V, T2I, I2V | `0.2` |

FLF2V is rejected by all three scripts. TeaCache and SeaCache reject VACE
because their referenced Wan2.1 implementations do not provide that path.

MagCache embeds the official Wan2.1 T2V-1.3B, T2V-14B, I2V-480P,
I2V-720P, VACE-1.3B, and VACE-14B ratio tables. Joint calibration stores the
conditional/unconditional ratios and static masks in the hybrid Cache Book.
TeaCache and SeaCache remain dynamic during output generation; their observer
masks are diagnostics only and are not saved as runtime policies.

## Usage

Step-cache-only generation (no layer Cache Book):

```bash
python sample_wan_teacache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --base_seed 42 --offload_model False
```

Two-pass Finegrained calibration only:

```bash
python sample_wan_magcache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --finegrained_calibration
```

Load a Cache Book and generate with both cache scales:

```bash
python sample_wan_magcache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --use_finegrained_cache
```

Calibrate and then generate in one process:

```bash
python sample_wan_seacache_hybrid.py \
  --task t2v-1.3B --size '832*480' \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'Two cats box under stage lights.' \
  --finegrained_calibration --use_finegrained_cache
```

Use `--disable_step_cache` for a layer-only ablation. Set all three layer
thresholds to zero for step-only parity. Calibration features default to
pinned CPU memory (`--calibration_feature_device cpu`); runtime layer/residual
caches default to GPU. `--runtime_cache_device auto` moves newly stored cache
tensors to pinned CPU once free GPU memory falls below
`--runtime_cache_gpu_reserve_gib` (2 GiB by default).

Cache Book filenames include task, size, frame count, sampling steps,
non-skip rate, all layer thresholds, and method-specific parameters. They do
not contain the seed or trajectory-state labels. Loading strictly validates the
method, formula, task, geometry, solver, shift, model dimensions, thresholds,
and source commit.

## Sources and licensing

- MagCache source: local commit `df81cb1`, Apache-2.0.
- TeaCache source: local commit `7c10efc`, Apache-2.0.
- SeaCache source: local commit `8dcf490`. No license file was visible in the
  referenced checkout. Confirm redistribution rights before publishing or
  redistributing the SeaCache-derived script.

The Wan orchestration follows the locally installed Wan2.1 implementation.
Later Wan/Hunyuan ports should use their corresponding model-specific source
constants and CFG layouts rather than reusing the values in these FLUX/Wan
scripts.
