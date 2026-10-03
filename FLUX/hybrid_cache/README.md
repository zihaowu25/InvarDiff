# FLUX Finegrained Hybrid Cache

This directory contains two standalone FLUX.1-dev sampling scripts:

| Script | Whole-step policy | Layer policy |
| --- | --- | --- |
| `sample_flux_magcache_hybrid.py` | MagCache | Finegrained Cache |
| `sample_flux_seacache_hybrid.py` | SeaCache | Finegrained Cache |

The files intentionally do not import each other or a shared runtime module.
Each file contains its own FLUX forward adaptation, two-stage layer
calibration, Cache Book handling, timing, and image-saving code.

## Hierarchical execution

Whole-step reuse always has priority. A whole-step hit skips every Transformer
block and leaves all layer-cache tensors unchanged. The first computed step
after a whole-step hit forces every layer module to refresh. Current timestep
conditioning, output normalization, and output projection are always
recomputed.

MagCache performs one joint raw pass that collects both magnitude ratios and
layer scores, followed by one hybrid-aware layer correction pass. SeaCache
remains dynamic: its raw calibration pass observes the official
would-skip path without actually skipping, and the correction pass uses that
per-prompt path only to update layer references. Dynamic observer masks are not
saved as runtime policies.

## Common modes

Run from the repository root after setting `FLUX_MODEL` to your downloaded
model directory or accessible model ID:

```bash
# Whole-step method only (disable the default module cache)
python FLUX/hybrid_cache/sample_flux_magcache_hybrid.py \
  --model-path "$FLUX_MODEL" --no-use-finegrained-cache --prompt "a photo of a cat"

# Two calibration passes; save a Cache Book without decoding an image
python FLUX/hybrid_cache/sample_flux_magcache_hybrid.py \
  --model-path "$FLUX_MODEL" \
  --finegrained-calibration \
  --cache-book-path ./cache_books/FLUX/custom_magcache \
  --calibration-prompt "a cinematic photo of a raccoon"

# Load a Cache Book and run hybrid generation
python FLUX/hybrid_cache/sample_flux_magcache_hybrid.py \
  --model-path "$FLUX_MODEL" \
  --prompt "a photo of a cat"

# Calibrate and immediately generate
python FLUX/hybrid_cache/sample_flux_magcache_hybrid.py \
  --model-path "$FLUX_MODEL" \
  --finegrained-calibration \
  --use-finegrained-cache \
  --cache-book-path ./cache_books/FLUX/custom_magcache \
  --prompt "a photo of a cat"

# Layer-only ablation: calibrate its own policy first, then generate.
python FLUX/hybrid_cache/sample_flux_magcache_hybrid.py \
  --model-path "$FLUX_MODEL" \
  --cache-book-path ./cache_books/FLUX/custom_layer_only \
  --finegrained-calibration --use-finegrained-cache \
  --disable-step-cache \
  --prompt "a photo of a cat"
```

Replace the script name with the SeaCache variant as needed. Run
`python <script> --help` for all options. The comparison defaults are BF16,
28 denoising steps, 1024×1024, guidance 3.5, and seed 42.

## Cache-specific defaults

- MagCache: threshold `0.24`, `K=4`, retention ratio `0.2`.
- SeaCache: threshold `0.3`, scheduler-aware flow filter,
  `power_exp=2.0`, spatial dimensions `(-2, -3)`, mean normalization.

Supply `--model-path "$FLUX_MODEL"` for your downloaded model. Inference
loads the configuration-named JSON in `cache_books/FLUX/` automatically. Calibration-only does not
generate an image; without an explicit output folder it uses the same directory
and configuration-derived filename as inference. Different configurations coexist;
recalibrating the same configuration updates its file. An optional custom folder
must be passed to both calibration and inference. Explicit module or cross-step parameter
changes require recalibration, not editing a loaded policy.

## Initial compatibility boundary

The first implementation targets standard FLUX.1-dev guidance-embedding
inference. ControlNet and IP-Adapter inputs are rejected explicitly. True CFG
is not exposed by these scripts because it requires independent positive and
negative runtime-cache slots.

## Source and licensing notes

The MagCache adaptation is based on its Apache-2.0 repository and includes a
source commit identifier in generated Cache Books.
The inspected SeaCache checkout did not contain a visible license file.
Confirm its redistribution terms before publishing or redistributing the
SeaCache-derived script.
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for the release boundary.
