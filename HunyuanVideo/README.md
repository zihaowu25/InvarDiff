# Finegrained Cache for HunyuanVideo-1.5

This directory provides two independent Finegrained Cache implementations for
HunyuanVideo-1.5:

| Script | Cache scope | Cache Book policy |
| --- | --- | --- |
| `sample_hunyuan.py` | Layer-only cache | `layer` |
| `sample_hunyuan_step_layer.py` | Step and layer cache | `stplayer` |

Both scripts contain complete model-loading, sampling, two-stage calibration,
Cache Book management, and video-saving workflows. They do not import
implementation code from each other. Model components, pipelines, schedulers,
and the super-resolution pipeline are loaded from the official
HunyuanVideo-1.5 repository.

## 1. Environment and directory layout

Install the dependencies and download a supported model by following the
official HunyuanVideo-1.5 instructions. The default directory layout is:

```text
/root/autodl-tmp/
├── HunyuanVideo-1.5/          # Official repository
└── InvarDiff/
    └── HunyuanVideo/
        ├── sample_hunyuan.py
        ├── sample_hunyuan_step_layer.py
        └── README.md
```

The scripts automatically add the sibling
`/root/autodl-tmp/HunyuanVideo-1.5` repository to the Python import path.
`--model_path` must point to a complete official model directory, which
normally contains the Transformer, VAE, scheduler, and text-encoder
components.

Display all available arguments with:

```bash
cd /root/autodl-tmp/InvarDiff/HunyuanVideo

python sample_hunyuan.py --help
python sample_hunyuan_step_layer.py --help
```

## 2. Algorithm

### 2.1 Relative L1 rate with compressed state

The same relative L1 rate is used for both step-level and layer-level
trajectories:

```text
diff_prev = x - x_prev + 1e-8
diff_post = x_post - x + 1e-8
rate = ||diff_post||₁ / clamp_min(||diff_prev||₁, 1e-8)
```

The L1 norms are accumulated in FP32 chunks. This avoids materializing a full
FP32 difference tensor. Each trajectory retains only the current feature and
the scalar norm of the previous displacement:

- `x`
- the precomputed scalar `||x - x_prev||₁`

A new score compares the displacement from `x` to `x_post` against the
previous displacement from `x_prev` to `x`. The first two observations
initialize the compressed state; the third observation produces the first
score. A lower rate indicates a more stable relative change and therefore gives
that position a higher cache priority.

### 2.2 Two-stage resampling calibration

Both implementations use two calibration passes:

1. **Raw calibration** performs a full denoising pass, collects the original
   relative-L1 scores, and builds a provisional Cache Book.
2. **Correction calibration** still performs all model computations, but
   updates or freezes each compressed reference according to the provisional
   cache path:
   - refresh at non-cached positions;
   - refresh once when entering a consecutive cached segment;
   - freeze the reference inside the consecutive cached segment;
   - resume normal refreshes after leaving the segment.

The correction scores are used to construct the final Cache Book. Normal
runtime feature caches are not allocated during either calibration pass.

### 2.3 Layer-level cache points

HunyuanVideo-1.5 contains double-stream and single-stream blocks. Finegrained Cache
maintains six independent module trajectories:

| Trajectory | Cached value |
| --- | --- |
| `double.img_attn` | H-dimensional image attention output after projection and before gating |
| `double.txt_attn` | H-dimensional text attention output after projection and before gating |
| `double.img_mlp` | Image MLP output before gating |
| `double.txt_mlp` | Text MLP output before gating |
| `single.attn` | H-dimensional attention contribution of the single-stream Linear2 projection |
| `single.mlp` | H-dimensional MLP contribution of the single-stream Linear2 projection |

Image and text attention in a double-stream block are produced by the same
joint-attention operation. The operation can therefore be skipped only when
both attention trajectories hit their caches. If either trajectory misses,
joint attention is recomputed and both attention caches are refreshed.

For a single-stream block, the concatenated output projection is split into two
mathematically equivalent contributions:

```text
attn_contrib = attn · W_attn
mlp_contrib  = mlp  · W_mlp
output = attn_contrib + mlp_contrib + bias
```

This design allows attention and MLP contributions to be scored and cached
independently. Both runtime caches remain H-dimensional, so the larger 4H MLP
activation does not need to be retained. The current step's bias, gates, and
residual additions are always recomputed.

### 2.4 Step-level cache

`sample_hunyuan_step_layer.py` additionally caches the image hidden state
after all double-stream and single-stream blocks and before `final_layer`.

On a step-level cache hit, the implementation:

- skips all double-stream and single-stream blocks;
- reuses the post-block image hidden state;
- still recomputes the current step's `final_layer` and unpatchify operation.

This implementation follows a step-first policy:

- a step-level hit makes every module cache decision at that step effectively
  cached;
- module-level caching is forcibly disabled at the next diffusion step so that
  module caches are refreshed;
- the same constraints are applied to both provisional and final Cache Books.

`sample_hunyuan.py` is strictly layer-only and does not create step scores,
step runtime caches, or a `step_cache_book`.

### 2.5 CFG and distributed execution

For a standard CFG model, one Transformer forward receives:

```text
[unconditional batch, conditional batch]
```

The implementation follows these rules:

- both CFG branches share the same Cache Book decisions;
- runtime caches store the complete batch, so features are never reused across
  the two branches;
- calibration scores use only the conditional half of the batch;
- CFG-distilled and step-distilled models have no doubled CFG batch, so their
  full batch is used for scoring.

With sequence parallelism, each rank caches its local token shard. L1 scalars
are reduced over the sequence-parallel group so that all ranks obtain the same
scores. Rank 0 saves the Cache Book, and the resulting policy is broadcast to
all ranks.

## 3. Quick start

The examples below assume:

```bash
export MODEL_PATH=/path/to/HunyuanVideo-1.5-model
export IMAGE_PATH=/path/to/reference.png
cd /root/autodl-tmp/InvarDiff/HunyuanVideo
```

### 3.1 Layer-only calibration and generation

```bash
python sample_hunyuan.py \
  --prompt "A cinematic shot of a cat walking through a rainy city." \
  --negative_prompt "" \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --image_path "$IMAGE_PATH" \
  --video_length 121 \
  --num_inference_steps 50 \
  --sr false \
  --invardiff_calibration \
  --use_invardiff \
  --cache_book_path ./cache_books \
  --output_path ./outputs/hunyuan_layer.mp4
```

This command performs three main-model denoising passes:

```text
raw calibration + correction calibration + final accelerated generation
```

Only the final pass is decoded and saved as a video.

### 3.2 Step-and-layer calibration and generation

```bash
python sample_hunyuan_step_layer.py \
  --prompt "A cinematic shot of a cat walking through a rainy city." \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --image_path "$IMAGE_PATH" \
  --video_length 121 \
  --num_inference_steps 50 \
  --sr false \
  --invardiff_calibration \
  --use_invardiff \
  --cache_book_path ./cache_books \
  --output_path ./outputs/hunyuan_step_layer.mp4
```

### 3.3 Calibration only

Omit `--use_invardiff`:

```bash
python sample_hunyuan_step_layer.py \
  --prompt "A cinematic shot of a cat walking through a rainy city." \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --image_path "$IMAGE_PATH" \
  --sr false \
  --invardiff_calibration \
  --cache_book_path ./cache_books
```

This mode performs only the raw and correction passes. It does not decode
latents, run super-resolution, or save a video.

### 3.4 Generate from an existing Cache Book

Use the automatically derived file name:

```bash
python sample_hunyuan_step_layer.py \
  --prompt "A different prompt for evaluation." \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --image_path "$IMAGE_PATH" \
  --num_inference_steps 50 \
  --sr false \
  --use_invardiff \
  --cache_book_path ./cache_books \
  --output_path ./outputs/hunyuan_cached.mp4
```

Or select an explicit file:

```bash
--cache_book_path ./cache_books \
--cache_book_file cache_book_stplayer_720p_i2v_1280x720_f121_steps50_....json
```

When `--cache_book_file` is supplied, the boolean policy stored in that file
is used directly. Threshold arguments in the current command do not rebuild or
alter that policy.

### 3.5 T2V and multi-GPU execution

For text-to-video generation, omit `--image_path`:

```bash
python sample_hunyuan.py \
  --prompt "Ocean waves under the moonlight." \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --sr false \
  --invardiff_calibration \
  --use_invardiff
```

Example with two sequence-parallel workers:

```bash
torchrun --nproc_per_node=2 sample_hunyuan_step_layer.py \
  --prompt "Ocean waves under the moonlight." \
  --resolution 720p \
  --model_path "$MODEL_PATH" \
  --sr false \
  --invardiff_calibration \
  --use_invardiff
```

## 4. Finegrained Cache arguments

### 4.1 Execution modes

The existing `--invardiff_*` spellings are retained as command-line
compatibility names; the implementation and documentation refer to this
feature as Finegrained Cache.

| Argument | Default | Description |
| --- | --- | --- |
| `--invardiff_calibration` | Disabled | Run raw and correction calibration and save a Cache Book |
| `--use_invardiff` | Disabled | Load or reuse the newly calibrated Cache Book for accelerated generation |
| `--cache_book_path` | `./cache_books` | Cache Book directory, relative to the current working directory |
| `--cache_book_file` | Automatically derived | Explicit Cache Book file name |

The number of main-model denoising passes depends on the selected mode:

| Mode | Denoising passes | Saves a video |
| --- | ---: | --- |
| Neither flag | 1 baseline pass | Yes |
| `--invardiff_calibration` only | 2 calibration passes | No |
| `--use_invardiff` only | 1 accelerated pass | Yes |
| Both flags | 2 calibration passes and 1 accelerated pass | Yes |

### 4.2 Thresholds

| Argument | Default | Script |
| --- | ---: | --- |
| `--nonskip_rate` | 0.1 | Both |
| `--step_thres` | 0.5 | Step-and-layer only |
| `--double_img_attn_thres` | 0.5 | Both |
| `--double_txt_attn_thres` | 0.5 | Both |
| `--double_img_mlp_thres` | 0.5 | Both |
| `--double_txt_mlp_thres` | 0.5 | Both |
| `--single_attn_thres` | 0.5 | Both |
| `--single_mlp_thres` | 0.5 | Both |

These thresholds are **quantile fractions**, not absolute rate cutoffs. For
example, `0.5` uses the median of the valid scores. Selection uses a strict
comparison:

```text
score < quantile(scores, threshold)
```

- A larger threshold generally creates more cache candidates and may improve
  speed, but it also increases the risk of quality loss.
- Setting a module threshold to `0` disables its analyzer state and cache.
- `nonskip_rate` protects the beginning of the denoising trajectory.
- The final diffusion step is always fully recomputed.
- Default thresholds provide a common experimental starting point; they are
  not claimed to be the best quality-speed configuration.

### 4.3 GPU and CPU memory

| Argument | Default | Description |
| --- | --- | --- |
| `--calibration_feature_device` | `cpu` | Store compressed feature history on pinned CPU memory or GPU memory |
| `--runtime_cache_device` | `auto` | Store runtime caches on `auto`, `gpu`, or `cpu` |
| `--runtime_cache_gpu_reserve_gib` | 2.0 | GPU free-memory reserve used by `auto` mode |
| `--offloading` | `true` | Official model CPU offloading |
| `--group_offloading` | Automatic | Official group offloading, selected according to GPU capacity |
| `--overlap_group_offloading` | `true` | Overlap transfers for speed at the cost of additional CPU memory |

`runtime_cache_device=auto` prefers GPU storage. If saving a tensor would
reduce estimated free GPU memory below the requested reserve, that tensor is
stored in pinned CPU memory instead. A CPU cache hit requires a transfer back
to the GPU, trading speed for lower GPU memory usage.

For an out-of-memory error, try the following settings:

```bash
--calibration_feature_device cpu
--runtime_cache_device auto
--runtime_cache_gpu_reserve_gib 4
--offloading true
--group_offloading true
--overlap_group_offloading false
--sr false
```

If host memory is also limited, disable overlapping group offload or use a
shorter video and lower resolution for initial experiments. GPU calibration
history should be used only when sufficient GPU memory is available.

## 5. Official model and optimization compatibility

The scripts support configurations available in the official configuration
table, including:

- 480p and 720p T2V;
- 480p and 720p I2V;
- CFG-distilled models;
- 720p sparse CFG-distilled models;
- 480p I2V step-distilled models.

The official super-resolution pipeline remains unchanged and is not cached by
Finegrained Cache. Calibration passes do not decode videos or run
super-resolution.
During final generation, `--sr` controls whether super-resolution runs.

Finegrained Cache mode retains support for:

- official model offloading and group offloading;
- SageAttention;
- the default FP8 GEMM configuration that targets only `double_blocks`;
- checkpoint loading;
- LoRA configurations that do not wrap the single-stream `linear2.fc`.

The following combinations are rejected:

- Finegrained Cache together with official `--enable_cache true`;
- Finegrained Cache together with `--enable_torch_compile true`;
- sparse attention together with SageAttention;
- FP8 `include_patterns` that include `single_blocks`;
- LoRA or quantization wrappers that replace single-stream `linear2.fc` with
  a type other than `torch.nn.Linear`.

The final two restrictions ensure that the split single-stream projection does
not bypass a quantization wrapper or LoRA adapter.

## 6. Cache Book format

Default file-name prefixes are:

```text
cache_book_layer_...
cache_book_stplayer_...
```

The file name encodes:

- Transformer version and task;
- actual width, height, and frame count;
- number of diffusion steps;
- `nonskip_rate`;
- step threshold for the step-and-layer version;
- all six module thresholds.

The file name contains neither the seed nor trajectory-state labels. The seed is
still recorded in JSON metadata.

The main JSON fields are:

```json
{
  "cache_scope": "layer or step_layer",
  "policy": "layer or stplayer",
  "rate_method": "relative_l1",
  "config": {},
  "step_cache_book": [],
  "module_cache_book": {}
}
```

`step_cache_book` exists only in step-and-layer files. Loading validates the
rate method, scope, policy, model version, task, geometry, frame count,
diffusion-step count, module keys, and double/single block counts. An old or
incompatible file is rejected with a request to recalibrate.

A Cache Book stores only boolean decisions; runtime feature tensors are never
serialized. Prompts and seeds may differ from the calibration run, but a
controlled algorithm comparison should keep the model, geometry, frame count,
diffusion steps, prompt, reference image, and seed identical.

## 7. Output, timing, and reproducible evaluation

- `--output_path` selects the final video path. Without it, output is written
  under `./outputs/`.
- `--save_pre_sr_video true` additionally saves the video before
  super-resolution.
- `--save_generation_config true` writes a matching `*_config.json` file.
- Raw calibration, correction calibration, and final generation report
  separate execution times.
- Each stage reports its own peak allocated GPU memory.
- `--seed -1` resolves to one random seed at startup. Raw calibration,
  correction calibration, and final generation then reuse that seed.
- With `--rewrite true`, the prompt is rewritten once and reused in every
  stage.

For a fair comparison between layer-only and step-and-layer caching:

1. Use the same model version, prompt, image, seed, resolution, frame count,
   and sampling-step count.
2. Disable super-resolution when measuring main-model denoising time.
3. Record raw/correction peak memory, final inference time, and output quality.
4. Do not enable another model-cache implementation simultaneously.
5. Report every threshold and the runtime-cache storage mode.

## 8. Troubleshooting

### Cache Book mismatch

The Cache Book does not match the current model version, task, geometry, or
sampling configuration. Do not manually alter the JSON file. Re-run the
Finegrained Cache calibration flag with the current configuration.

### Why does calibration not save a video?

When only the Finegrained Cache calibration flag is specified, the script produces a final
Cache Book without VAE decoding or super-resolution. Add `--use_invardiff` to
perform and save an accelerated generation after calibration.

### Why is CPU runtime caching slower?

Every CPU cache hit transfers a feature tensor back to the GPU. This is a
memory-speed tradeoff. If GPU memory permits, use:

```bash
--runtime_cache_device gpu
```

### Why does the single-stream cache report a `linear2.fc` type error?

The single-stream strategy accesses two weight slices of the original
`torch.nn.Linear`. Check whether FP8 was enabled for `single_blocks` or
whether a LoRA adapter wrapped this layer.

### Why is attention still computed when one double-stream attention cache hits?

Image and text attention share one joint-attention kernel. Actual computation
can be skipped only when both `double.img_attn` and `double.txt_attn` hit.
This joint decision is required by the Hunyuan double-stream architecture.
