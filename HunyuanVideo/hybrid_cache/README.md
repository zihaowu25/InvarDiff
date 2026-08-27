# HunyuanVideo-1.5 Finegrained Hybrid Cache

This directory contains three standalone HunyuanVideo-1.5 samplers that combine
a whole-step cache with a calibrated Finegrained Cache:

- `sample_hunyuan_magcache_hybrid.py`
- `sample_hunyuan_teacache_hybrid.py`
- `sample_hunyuan_seacache_hybrid.py`

The scripts intentionally do not import one another, the existing Hunyuan
sampling scripts, or executable files from the open-source checkouts. Each file
contains its own HunyuanVideo-1.5 forward adaptation, layer calibration,
whole-step policy, Cache Book I/O, timing, and video-saving path.

## Source and provenance

| Component | Source revision | Status in this port |
|---|---:|---|
| HunyuanVideo-1.5 | Tencent-Hunyuan `60783e7` | Native pipeline, model loading, T2V/I2V, SR, distributed execution and saving |
| MagCache | ComfyUI-MagCache `47bdd2a` | Official HunyuanVideo-1.5 policy and 20/40-step tables |
| TeaCache | TeaCache `7c10efc` | Faithful port of the official original-HunyuanVideo policy |
| SeaCache | SeaCache `8dcf490` | Faithful port of the official original-HunyuanVideo policy |

MagCache and TeaCache are distributed under Apache-2.0 in their source
repositories. No visible license file was found in the pinned SeaCache checkout;
confirm redistribution permission before publishing copied SeaCache code.

TeaCache and SeaCache do not currently publish model-specific
HunyuanVideo-1.5 coefficients or thresholds. Their scripts preserve the
published original-HunyuanVideo formulas and defaults, but must not be described
as official HunyuanVideo-1.5 parameterizations.

## Hybrid execution model

The execution hierarchy is step-first:

```text
prepare embeddings, timestep modulation and all conditioning
  -> evaluate the whole-step policy
     -> hit: embedded image hidden + cached whole-step residual
             do not read or update layer caches
     -> miss: if this CFG slot was cached at the previous step,
              force every layer branch to refresh once
              otherwise follow the calibrated layer Cache Book
              save the resulting whole-step residual
  -> always recompute final_layer and unpatchify
```

The whole-step residual is

```text
image hidden after all double/single blocks
- image hidden immediately before those blocks
```

`final_layer` remains timestep-conditioned and is therefore always recomputed.
The official pipeline performs unpatchification after `final_layer`, so it is
also outside the cache.

### Finegrained cache points

The port keeps six residual-branch trajectories:

| Key | Cached tensor |
|---|---|
| `double.img_attn` | Projected image attention output before its current gate |
| `double.txt_attn` | Projected text attention output before its current gate |
| `double.img_mlp` | Image MLP output before its current gate |
| `double.txt_mlp` | Text MLP output before its current gate |
| `single.attn` | Attention contribution after the attention slice of `linear2.fc` |
| `single.mlp` | MLP contribution after the MLP slice of `linear2.fc` |

Double-stream image/text attention uses a joint AND decision because both
outputs are produced by the same joint attention call. Single-stream attention
and MLP remain independent; their current gate, bias, summation and residual
addition are recomputed. This avoids storing the larger pre-projection MLP
activation.

The adaptation preserves ByT5 conditioning, vision conditioning, `timestep_r`,
sparse/SSTA attention arguments, local sequence-parallel token shards, official
offloading, FP8 configuration, LoRA loading and the separate SR Transformer.
The SR Transformer is not patched.

## Layer-rate formula and two-pass calibration

Every layer trajectory uses a relative L1 rate with compressed state:

```text
R = ||x_post - x + 1e-8||_1
    / clamp_min(||x - x_prev + 1e-8||_1, 1e-8)
```

L1 accumulation is chunked and performed in FP32 on the current accelerator.
The analyzer retains only the current detached feature and the precomputed
previous-displacement norm.
With `--calibration_feature_device cpu` (default), history features are stored in
pinned CPU memory and transferred chunk by chunk for scoring.

Calibration always consists of exactly two complete denoising passes:

1. **Raw pass**: execute every Transformer block and collect layer features.
   MagCache simultaneously collects token-wise residual L2 magnitude ratios;
   TeaCache and SeaCache record dynamic observer masks without skipping blocks.
2. **Correction pass**: still execute every block, but use the provisional
   whole-step and layer paths to control analyzer refresh/freeze behavior. A
   non-cached position refreshes the window; entering a cached segment refreshes
   once; consecutive cached positions freeze it; leaving the segment refreshes
   again.

Calibration-only therefore performs two denoising trajectories. Combining
`--finegrained_calibration` and `--use_finegrained_cache` performs those two
passes followed by one final hybrid generation (three trajectories total).
Raw and correction peak allocated CUDA memory values are saved in the Cache
Book.

## CFG adaptation

Normal HunyuanVideo-1.5 classifier-free guidance creates one Transformer batch
in `[unconditional, conditional]` order. Finegrained calibration scores only the
conditional half. Layer decisions are shared, while runtime tensors are stored
under separate CFG-slot keys.

MagCache's official HunyuanVideo-1.5 implementation calls conditional and
unconditional branches separately and keeps independent accumulators and
residuals. This port preserves that behavior inside the native batched pipeline:

- both slots hit: skip all double/single blocks;
- both slots miss: calculate the complete batch;
- mixed hit: slice all batch-shaped model inputs, calculate only the missing
  half, and concatenate the outputs back in native order.

TeaCache and SeaCache use one shared dynamic decision for the complete CFG batch,
matching their published HunyuanVideo implementations. Their cached layer
tensors remain slot-separated even though their decisions are shared.
Guidance-distilled models use one slot.

## Whole-step policies and defaults

| Method | Default | Predictor | Boundary behavior |
|---|---|---|---|
| MagCache | threshold `0.03`, `K=2`, retention `0.25` | accumulated magnitude-ratio error | strict `<`; two independent semantic CFG slots |
| TeaCache | threshold `0.15` | official fifth-order polynomial of relative L1 | first/last computed; history updated each step |
| SeaCache | threshold `0.20`, power `3.0`, dims `(-2,-3,-4)`, mean normalization | scheduler-aware spectral relative L1 | filtering/history update also occurs on first/last steps |

MagCache contains the official HunyuanVideo-1.5 20-step and 40-step ratio tables.
`--magcache_ratio_source auto` selects them only for standard, non-distilled,
non-sparse T2V at exactly 20 or 40 steps. I2V, other step counts, distilled
models and sparse models require joint calibration. The script never silently
uses a nearby T2V table for an unsupported configuration.

SeaCache reads the actual scheduler created by the HunyuanVideo-1.5 pipeline. It
uses flow-mode `(a,b) = (1-sigma, sigma)`, full complex FFT/IFFT over THW, the
published separable Wiener filter and mean spectrum normalization.

## Command-line interface

All scripts preserve the official `generate.py` arguments and add:

```text
--finegrained_calibration
--use_finegrained_cache
--cache_book_path
--cache_book_file
--nonskip_rate
--double_img_attn_thres
--double_txt_attn_thres
--double_img_mlp_thres
--double_txt_mlp_thres
--single_attn_thres
--single_mlp_thres
--disable_step_cache
--disable_progress_bar
--calibration_feature_device {cpu,gpu}
--runtime_cache_device {auto,gpu,cpu}
--runtime_cache_gpu_reserve_gib
```

Method-specific arguments are:

```text
MagCache: --magcache_thresh --magcache_K --retention_ratio
          --magcache_ratio_source {auto,official,calibrated}
TeaCache: --teacache_thresh
SeaCache: --seacache_thresh
```

Execution modes:

| Flags | Behavior |
|---|---|
| neither Finegrained flag | whole-step cache only |
| `--finegrained_calibration` | two calibration passes, save Cache Book, no video |
| `--use_finegrained_cache` | load Cache Book, perform one hybrid generation |
| both flags | two calibration passes, then one hybrid generation |
| use + `--disable_step_cache` | layer-only ablation |
| all six layer thresholds set to zero | step-only parity configuration |

`--runtime_cache_device auto` keeps a new tensor on GPU unless doing so would
reduce free memory below `--runtime_cache_gpu_reserve_gib` (default 2 GiB);
otherwise it stores the tensor in pinned CPU memory.

## Usage examples

Set paths once:

```bash
MODEL=/path/to/HunyuanVideo-1.5
OUT=./outputs
BOOKS=./cache_books
PROMPT='A cinematic tracking shot of a sailboat crossing a luminous bay.'
```

### MagCache step-only, official 20-step T2V

```bash
python sample_hunyuan_magcache_hybrid.py \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --num_inference_steps 20 --seed 123 --sr false \
  --output_path "$OUT/magcache_step_only.mp4"
```

### Two-pass calibration only

Replace `METHOD` with `magcache`, `teacache`, or `seacache`:

```bash
python "sample_hunyuan_${METHOD}_hybrid.py" \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --num_inference_steps 20 --seed 123 --sr false \
  --cache_book_path "$BOOKS" --finegrained_calibration
```

### Load and generate a hybrid result

```bash
python sample_hunyuan_teacache_hybrid.py \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --num_inference_steps 20 --seed 123 --sr false \
  --cache_book_path "$BOOKS" --use_finegrained_cache \
  --output_path "$OUT/teacache_hybrid.mp4"
```

### Calibrate and immediately generate

```bash
python sample_hunyuan_seacache_hybrid.py \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --num_inference_steps 20 --seed 123 --sr false \
  --cache_book_path "$BOOKS" \
  --finegrained_calibration --use_finegrained_cache \
  --output_path "$OUT/seacache_hybrid.mp4"
```

### Layer-only ablation

```bash
python sample_hunyuan_magcache_hybrid.py \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --num_inference_steps 20 --seed 123 --sr false \
  --cache_book_path "$BOOKS" --use_finegrained_cache \
  --disable_step_cache --output_path "$OUT/layer_only.mp4"
```

### 720p I2V MagCache joint calibration

I2V has no official HunyuanVideo-1.5 ratio table in the pinned MagCache source,
so joint calibration is mandatory:

```bash
python sample_hunyuan_magcache_hybrid.py \
  --model_path "$MODEL" --resolution 720p --prompt "$PROMPT" \
  --image_path /path/to/reference.png --num_inference_steps 20 \
  --magcache_ratio_source calibrated --seed 123 --sr false \
  --cache_book_path "$BOOKS" \
  --finegrained_calibration --use_finegrained_cache \
  --output_path "$OUT/magcache_i2v_hybrid.mp4"
```

Cache Books and output videos use their explicit paths. Without
`--cache_book_file`, the book name starts with
`cache_book_hybrid_<method>_` and includes model geometry, frames, steps and all
policy thresholds. Without `--output_path`, the official timestamped output
convention is used under `./outputs`.

## Cache Book format

All books contain:

```text
cache_scope: hybrid
policy: hybrid_<method>
step_policy: <method>
rate_method: relative_l1
config: model/task/geometry/steps/thresholds/source revisions
finegrained_cache.module_cache_book: six boolean [step][layer] books
calibration_peak_allocated_gib: raw and correction values
```

MagCache additionally stores semantic conditional/unconditional magnitude
ratios and native-order conditional/unconditional step masks. TeaCache and
SeaCache store only their observer mask and observer hit rate because their
runtime policies remain dynamic.

Loading rejects another method, old formats, a different model version, task,
resolution, frame count, step count, CFG/distillation/sparse mode, block count,
threshold, formula or source revision. Recalibrate after any rejected mismatch.

## Experimental protocol

For paper comparisons, keep the following identical across the baseline and all
hybrid methods:

- prompt and negative prompt;
- resolved prompt-rewrite result;
- seed, resolution, aspect ratio and frame count;
- number of denoising steps, scheduler and model version;
- guidance/CFG-distilled/step-distilled configuration;
- BF16/FP32, FP8, attention backend and sparse-attention settings;
- model/group offloading, SR and output-saving settings.

Report at least:

- synchronized Transformer/generation elapsed time;
- step skip ratio (per CFG slot for MagCache);
- layer skip ratio over computed steps/branches;
- combined effective skip ratio;
- peak allocated GPU memory for raw calibration, correction and generation;
- output quality metrics used by the paper.

State explicitly whether offline calibration time is excluded from amortized
inference latency. For fair dynamic-policy comparisons, do not include Cache
Book calibration in TeaCache/SeaCache per-sample runtime, but report it
separately as the Finegrained Cache preparation cost.

## Limitations and development notes

- MagCache official tables cover only standard HunyuanVideo-1.5 T2V at 20/40
  steps. Other configurations require a task-specific hybrid book.
- TeaCache and SeaCache use original-HunyuanVideo predictors and have not been
  officially fitted for HunyuanVideo-1.5, especially I2V and distilled models.
- Mixed CFG MagCache slices every batch-shaped conditioning tensor. Validate
  this path again whenever the official pipeline adds a new batch-shaped input.
- Finegrained single-stream projection requires `linear2.fc` to remain an
  unwrapped `torch.nn.Linear`. Single-block FP8 or an LoRA wrapper at that exact
  module is rejected rather than bypassed.
- Hybrid control flow is incompatible with the official cache helper and
  `torch.compile`; both combinations are rejected.
- Sequence parallel caches local token shards and all-reduces scalar scores.
  Multi-rank mixed-hit and sparse-attention paths require GPU validation before
  being marked experimentally verified.
- Prompt rewriting and `seed=-1` are resolved once and reused across raw,
  correction and final generation.

## Validation status

Completed in the current checkout:

- [x] standalone import and `--help` for all three scripts;
- [x] Python bytecode compilation;
- [x] official MagCache table length and accumulator-mask checks;
- [x] no cross-script or executable open-source imports;
- [x] Cache Book source/config validation implemented.

Requires model weights and the intended GPU/distributed environment:

- [ ] cache-disabled FP32/BF16 output comparison with official forward;
- [ ] official MagCache per-step mask/output comparison in ComfyUI;
- [ ] 720p T2V and I2V calibration/load/generation smoke tests;
- [ ] sequence-parallel mixed-hit collective test;
- [ ] sparse-attention, FP8, LoRA and SR integration tests.

Update this document whenever a formula, default, supported model variant,
Cache Book schema or validation result changes.
