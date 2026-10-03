# Reusable Cache Books

Books are grouped into four model folders: `DiT/`, `FLUX/`, `Wan2.1/` and
`HunyuanVideo/`. Each configuration-named JSON identifies its strategy; module,
step-layer and hybrid books share their model folder. Calibration and inference
use this same folder and filename.
These are fixed boolean schedules, not model weights or cached tensors from
another prompt. Runtime feature caches start empty for every generation.

| Model | Policies | Released protocol |
| --- | --- | --- |
| DiT-XL/2 | `dit_module`, `dit_step_layer` | 512 × 512, DDIM-50, CFG 4 |
| FLUX.1-dev | `flux_module`, `flux_step_layer`, `flux_magcache_hybrid`, `flux_seacache_hybrid` | 1024 × 1024, 28 steps, guidance 3.5 |
| Wan2.1-T2V-1.3B | `wan_module`, `wan_step_layer`, `wan_magcache_hybrid`, `wan_seacache_hybrid` | 832 × 480, 81 frames, 50 steps, UniPC, shift 5, CFG 5 |
| HunyuanVideo-1.5 | `hunyuan_module`, `hunyuan_step_layer`, `hunyuan_magcache_hybrid`, `hunyuan_seacache_hybrid` | 720p T2V, 16:9, 121 frames, 50 steps |

[`manifest.json`](manifest.json) contains each policy's provenance, checksum,
calibration inputs, small-set visual verdict, and measured sampling times.
Reload checks use prompts/classes and seeds different from calibration.
Acceptance checks that a loaded Book produces usable demos with real runtime
savings; it does not require pixel-identical output or unchanged fine detail.
Visual screening is not a formal benchmark, an error bound, or evidence that
every prompt benefits. Full remains available when fidelity is critical.

The model, scheduler, geometry, step count, guidance, precision, partition,
thresholds, and hybrid cross-step settings must match. A calibration seed is
provenance only: new generation seeds are allowed. A changed checkpoint or
LoRA requires its own calibration/validation even when its dimensions match.

Filenames follow the original step/threshold naming style and include geometry,
steps, the protected prefix and cache thresholds, with a short hash for other
execution settings. Different configurations have different
filenames; generation prompts and seeds do not change the filename. Recalibrating
the same configuration updates its file: copy it first to retain multiple books
for identical settings. The manifest describes the originally shipped files only.
Optional custom folders and filenames remain supported; see the
[main README](../README.md#bundled-cache-books-inference-and-calibration).
Do not edit decisions or assume that changing a threshold flag rebuilds a
loaded book. Timing depends on hardware, kernels, offloading, prompts, and
measurement boundary; the small-set speed checks are not guaranteed speedups.
