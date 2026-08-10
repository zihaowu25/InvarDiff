# FLUX Finegrained Hybrid Cache

This directory contains three standalone FLUX.1-dev sampling scripts:

| Script | Whole-step policy | Layer policy |
| --- | --- | --- |
| `sample_flux_magcache_hybrid.py` | MagCache | Finegrained Cache |
| `sample_flux_teacache_hybrid.py` | TeaCache | Finegrained Cache |
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
layer scores, followed by one hybrid-aware layer correction pass. TeaCache and
SeaCache remain dynamic: their raw calibration pass observes the official
would-skip path without actually skipping, and the correction pass uses that
per-prompt path only to update layer references. Dynamic observer masks are not
saved as runtime policies.

## Common modes

```bash
# Official whole-step method only
python sample_flux_magcache_hybrid.py --prompt "a photo of a cat"

# Two calibration passes; save a Cache Book without decoding an image
python sample_flux_magcache_hybrid.py \
  --finegrained-calibration \
  --calibration-prompt "a cinematic photo of a raccoon"

# Load a Cache Book and run hybrid generation
python sample_flux_magcache_hybrid.py \
  --use-finegrained-cache \
  --prompt "a photo of a cat"

# Calibrate and immediately generate
python sample_flux_magcache_hybrid.py \
  --finegrained-calibration \
  --use-finegrained-cache \
  --prompt "a photo of a cat"

# Layer-only ablation
python sample_flux_magcache_hybrid.py \
  --use-finegrained-cache \
  --disable-step-cache \
  --prompt "a photo of a cat"
```

Replace the script name with the TeaCache or SeaCache variant as needed. Run
`python <script> --help` for all options. The comparison defaults are BF16,
28 denoising steps, 1024×1024, guidance 3.5, and seed 42.

## Cache-specific defaults

- MagCache: threshold `0.24`, `K=5`, retention ratio `0.1`.
- TeaCache: official FLUX polynomial and threshold `0.6`.
- SeaCache: threshold `0.3`, scheduler-aware flow filter,
  `power_exp=2.0`, spatial dimensions `(-2, -3)`, mean normalization.

Cache Books are saved below `./cache_books` by default and use
`cache_book_hybrid_<method>_...` names.

## Initial compatibility boundary

The first implementation targets standard FLUX.1-dev guidance-embedding
inference. ControlNet and IP-Adapter inputs are rejected explicitly. True CFG
is not exposed by these scripts because it requires independent positive and
negative runtime-cache slots.

## Source and licensing notes

The MagCache and TeaCache adaptations are based on their Apache-2.0
repositories and include source commit identifiers in generated Cache Books.
The inspected SeaCache checkout did not contain a visible license file.
Confirm its redistribution terms before publishing or redistributing the
SeaCache-derived script.
