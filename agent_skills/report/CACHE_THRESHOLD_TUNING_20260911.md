# InvarDiff cache threshold tuning report

## Status

This report is generated during the threshold sweep. A configuration is marked
`validated` only after the 8-condition x 2-seed confirmation protocol passes.
Short smoke runs and incomplete candidates are never promoted to presets.

## Environment and protocol

- Repository branch: `dev-local`
- Starting commit: `dfee7edab5408d8bc8579ac306617a90a88c6306`
- Accelerator: 8 x NVIDIA H800 80GB
- Driver / framework: NVIDIA 535.129.03; PyTorch 2.6.0+cu124; CUDA 12.4
- LPIPS: AlexNet backbone, decoded RGB frames aligned by prompt, seed,
  scheduler, inference steps, geometry, frame count, FPS and preprocessing.
- Low-cost screening: 3 independent content conditions x seed 2027; every
  reported screening LPIPS is the arithmetic mean of the three paired samples.
  Video screening uses 17 frames. Candidates are advanced causally: the next
  threshold is launched only after the current three-sample mean is measured.
- Confirmation: 8 conditions x seeds 2027 and 2028; production frame counts.
- Coarse grid: threshold 0.00 through 0.90 in steps of 0.10.
- Coarse elbow: the LPIPS increment is at least 0.03 and at least twice the
  median of previous positive increments. With fewer than two previous
  intervals, the absolute 0.03 rule is used.
- Fine grid: the preceding 0.10 interval in steps of 0.01; corresponding
  minimum increment is 0.003.

Frame-aligned LPIPS is appropriate here as a low-cost cache-error screen
because every pair shares the model, prompt, seed, scheduler and decode
geometry. It is not treated as a complete video-quality metric: each metric
JSON also records frame p95/max, adjacent-frame LPIPS drift and temporal-delta
L1. Final confirmation reports all of these temporal diagnostics, rather than
interpreting a low framewise mean alone as temporal quality preservation.

The prompt set covers a low-motion locked shot, camera tracking, and
articulated human motion. Different independent model/script sweeps run on
different GPUs; thresholds inside one module sweep remain sequential because
elbow selection is causal.

The raw media, logs and temporary Cache Books are under
`runs/cache_tuning/`. Machine-readable CSV/JSON summaries and PNG figures are
kept beside this report or in the aggregate run directory.

## CCMR evidence and module order

### HunyuanVideo-1.5

The collector used four prompt pairs, one fixed seed, a shared initial latent,
720p, 17 frames and 50 inference steps. Cache decisions were disabled. Each
module family contains 10,584 valid `(pair, step, layer)` scalar observations.

| order | module family | median CCMR gain (dB) | mean gain (dB) |
|---:|---|---:|---:|
| 1 | `double.txt_attn` | 19.04 | 20.29 |
| 2 | `double.txt_mlp` | 18.50 | 21.11 |
| 3 | `double.img_attn` | 16.48 | 16.52 |
| 4 | `double.img_mlp` | 13.93 | 14.45 |

The loaded 720p T2V transformer has zero single-stream blocks, so
`single.attn` and `single.mlp` are not applicable and must remain at 0.00.
The complete data and ranking plot are in
`runs/cache_tuning/hunyuan_ccmr/aggregate/`.

### Existing formal evidence

The committed formal CCMR table gives the following module-family medians:

| model | order | module family | median gain (dB) |
|---|---:|---|---:|
| DiT | 1 | `msa` | 18.86 |
| DiT | 2 | `mlp` | 17.59 |
| FLUX | 1 | `context_ff` | 22.60 |
| FLUX | 2 | `context_attn` | 19.27 |
| FLUX | 3 | `attn` | 15.97 |
| FLUX | 4 | `ff` | 14.30 |
| FLUX | 5 | `single_attn` | 9.05 |
| FLUX | 6 | `single_mlp` | 6.79 |

- Wan2.1 has no formal CCMR asset in the current repository. Its historical
  pre-tuned settings are a search prior, not CCMR evidence.

## HunyuanVideo-1.5 module-only sweep

The first isolated sweep set only `double.txt_attn` above zero. Source and
Cache Book inspection showed that it cannot produce a cache hit: a double
block computes image and text attention jointly, and its runtime decision is
`double.img_attn AND double.txt_attn`. This is an algorithmic dependency, not
an experimental failure. The isolated sweep is retained as diagnostic data
under `coarse_double_txt_attn` but excluded from threshold selection.

To preserve the joint-attention algorithm while still following the CCMR
order, the formal sweep treats double attention as one coupled compute unit.
It first scans the higher-dB `double.txt_attn` threshold with the dependency
gate `double.img_attn=0.90`, then locks the selected text threshold and scans
the image threshold from 0.00. The final two values are selected independently
and must differ. MLP families remain 0.00 during this attention sweep.

The superseded exploratory `double.txt_mlp` jobs were stopped once the formal
four-pair CCMR order became available and are excluded from results.

The earlier four-pair diagnostic coarse curve uses `double.img_attn=0.90` as the
dependency gate and scans the higher-dB text threshold. Calibration occurs in
an isolated process; a fresh process then loads the Cache Book and generates
the original and cached videos consecutively in the same model instance. All
values below are macro means over four aligned 17-frame paired videos. They
are retained as higher-sample search evidence, while the current three-sample
causal sweep is used to continue threshold selection.

| text threshold | effective joint hit rate | LPIPS vs original | delta vs q=0 | generation seconds |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00% | 0.0000 | 0.0000 | 105.6 |
| 0.10 | 7.85% | 0.2172 | 0.2172 | 98.8 |
| 0.20 | 15.37% | 0.3228 | 0.3228 | 93.6 |
| 0.30 | 24.04% | 0.3355 | 0.3355 | 86.1 |
| 0.40 | 32.96% | 0.3733 | 0.3733 | 81.1 |
| 0.50 | 41.78% | 0.3948 | 0.3948 | 71.6 |
| 0.60 | 50.00% | 0.5729 | 0.5729 | 65.5 |
| 0.70 | 59.19% | 0.6191 | 0.6191 | 56.9 |

The q=0.00 result is measured rather than assumed. Its four paired candidate
videos are byte-identical to their original-forward references, so the
zero-hit acceptance control is exactly 0 LPIPS. The 0.00 to 0.10 jump is
0.2172, so the declared coarse rule identifies 0.10 as the first elbow and
schedules a 0.01 grid over `[0.00, 0.10]`. Thresholds above the first elbow
were already in flight on the other GPUs and are retained as target-selection
evidence, but the coarse search did not need to extend to 0.80 or 0.90.

Curve data and the PNG are in
`cache_threshold_tuning/hunyuan/module_only/coarse_double_txt_attn/` relative
to this report.

The earlier fresh-load paired 0.01 attention grid is:

| text threshold | effective joint hit rate | LPIPS vs original |
|---:|---:|---:|
| 0.00 | 0.00% | 0.0000 |
| 0.01 | 0.96% | 0.0868 |
| 0.02 | 1.81% | 0.1369 |
| 0.03 | 2.48% | 0.1761 |
| 0.04 | 3.15% | 0.2066 |
| 0.05 | 4.00% | 0.2113 |
| 0.06 | 4.59% | 0.2218 |
| 0.07 | 5.48% | 0.2148 |
| 0.08 | 6.19% | 0.2127 |
| 0.09 | 7.11% | 0.2177 |
| 0.10 | 7.85% | 0.2172 |

The first fine increment is 0.0868 and exceeds 0.003, so 0.01 is the measured
boundary for `double.txt_attn`. It also lands inside the slow target interval.
The CSV/JSON/PNG artifacts are in
`cache_threshold_tuning/hunyuan/module_only/fine_double_txt_attn/` relative to
this report.

The following text-MLP, image-attention, image-MLP and tier tables were also
derived downstream of that superseded grid. They are preserved to document
the exploratory search path, but will be replaced by fresh-load paired results
and cannot be used by `cache_presets.json` in their current form.

With `double.txt_attn=0.01` locked, the next CCMR family is
`double.txt_mlp`. Its coarse sweep is:

| text-MLP threshold | hit rate | LPIPS vs original | generation seconds |
|---:|---:|---:|---:|
| 0.00 | 0.00% | 0.1172 | 103.1 |
| 0.10 | 8.63% | 0.1087 | 103.0 |
| 0.20 | 17.33% | 0.1350 | 103.5 |
| 0.30 | 25.93% | 0.1330 | 102.6 |
| 0.40 | 34.44% | 0.1632 | 106.6 |
| 0.50 | 43.33% | 0.2244 | 102.7 |
| 0.60 | 52.11% | 0.2883 | 102.1 |
| 0.70 | 60.19% | 0.3905 | 101.9 |

The first qualifying positive coarse increment occurs from 0.30 to 0.40
(+0.0302), which schedules the `[0.30,0.40]` fine grid. The non-monotonic
quality values are retained: every threshold has an independently regenerated
cache-aware correction book, so monotonic LPIPS is not assumed.

The complete text-MLP fine grid has LPIPS values 0.1330, 0.1358, 0.1434,
0.1315, 0.1493, 0.2264, 0.1609, 0.1619, 0.1677, 0.1694 and 0.1632 at
thresholds 0.30 through 0.40. The 0.30 to 0.31 increment is 0.00274, below
the 0.003 fine criterion; the 0.31 to 0.32 increment is 0.00757, so 0.32 is
the retained boundary candidate. CSV/JSON/PNG artifacts are in
`assets/hunyuan_module/fine_double_txt_mlp/`.

With the text thresholds locked at `double.txt_attn=0.01` and
`double.txt_mlp=0.32`, the image-attention coarse sweep is:

| image-attention threshold | image hit rate | effective joint hit rate | LPIPS vs original | generation seconds |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00% | 0.00% | 0.1279 | 103.6 |
| 0.10 | 8.59% | 0.11% | 0.1238 | 104.2 |
| 0.20 | 17.74% | 0.33% | 0.1202 | 103.1 |
| 0.30 | 26.78% | 0.48% | 0.1213 | 107.5 |
| 0.40 | 34.93% | 0.59% | 0.1417 | 103.2 |
| 0.50 | 43.37% | 0.67% | 0.1429 | 102.9 |
| 0.60 | 52.15% | 0.74% | 0.1417 | 102.8 |
| 0.70 | 60.52% | 0.78% | 0.1416 | 102.2 |
| 0.80 | 69.89% | 0.89% | 0.1425 | 102.6 |
| 0.90 | 79.04% | 0.96% | 0.1434 | 102.3 |

No adjacent positive LPIPS increment reaches the 0.03 coarse criterion, so
there is no fine interval for this family inside the allowed range. The
measured safe upper boundary `double.img_attn=0.90` is retained. This value
is deliberately different from the text-attention threshold. The very small
effective joint-hit rate confirms that the locked 0.01 text gate, rather than
the image threshold, limits the coupled attention computation. Artifacts are
in `assets/hunyuan_module/coarse_double_img_attn/`.

The last applicable family is `double.img_mlp`. With all three earlier
families locked, its coarse measurements are:

| image-MLP threshold | hit rate | LPIPS vs original | generation seconds |
|---:|---:|---:|---:|
| 0.00 | 0.00% | 0.1434 | 102.3 |
| 0.10 | 6.00% | 0.3678 | 102.3 |
| 0.20 | 14.56% | 0.4402 | 101.1 |
| 0.30 | 23.15% | 0.4290 | 100.7 |
| 0.40 | 31.93% | 0.5584 | 98.9 |
| 0.50 | 40.81% | 0.6617 | 100.1 |
| 0.60 | 49.59% | 0.7287 | 96.0 |
| 0.70 | 59.19% | 0.7834 | 94.3 |
| 0.80 | 68.78% | 0.7681 | 92.3 |

The first 0.10 increment is +0.2245 and therefore identifies 0.10 as the
coarse elbow. The required fine grid is `[0.00,0.10]`. Coarse artifacts are
in `assets/hunyuan_module/coarse_double_img_mlp/`.

The fine measurements at 0.00 through 0.08 are 0.1434, 0.1428, 0.1428,
0.1856, 0.2097, 0.2844, 0.2803, 0.3764 and 0.3276 LPIPS. The first qualifying
positive fine increment is from 0.02 to 0.03 (+0.0428), so 0.03 is the
module-level boundary. Measurements beyond that first elbow are retained as
target-selection evidence, but are not described as being inside the
module-level safe region. Fine CSV/JSON/PNG artifacts are in
`assets/hunyuan_module/fine_double_img_mlp/`.

### Module-only screening candidates

The initial same-process screen identified these distinct candidate vectors:

| tier | img attn | txt attn | img MLP | txt MLP | screen LPIPS | screen seconds |
|---|---:|---:|---:|---:|---:|---:|
| fast | 0.90 | 0.01 | 0.20 | 0.32 | 0.4402 | 101.1 |
| balanced | 0.90 | 0.01 | 0.04 | 0.32 | 0.2097 | 102.8 |
| slow | 0.90 | 0.01 | 0.00 | 0.00 | 0.1172 | 103.1 |

The fast candidate uses the lowest-dB family beyond its 0.03 elbow. This is
explicitly a target-budget choice: among measured candidates in the required
0.35--0.45 interval, 0.20 has the lowest generation latency. The balanced
candidate is likewise selected by latency within 0.18--0.22. A zero threshold
means a family is disabled, not that multiple families share a uniform tuned
threshold. These values are now retained only as search hints. A later
reproducibility audit proved that calibration can leave runtime state which
changes generation even when the final Cache Book has zero hits. The candidate
runner previously calibrated and generated in that same process, so these
LPIPS values do not meet the fresh-process Cache Book requirement and must be
regenerated before any preset can be validated.

The 16 reference videos for seeds 2027 and 2028 have completed. `ffprobe`
validation confirmed both eight-video sets are single-stream 1280x720,
121-frame, 24 FPS videos.
Generation took approximately 31--34 minutes per video when run concurrently
on eight H800 GPUs; peak allocated CUDA memory was 52.69 GiB per process.
The validation manifest is
`assets/hunyuan_module/confirmation/reference_seed2027_ffprobe.json` and
`reference_seed2028_ffprobe.json` in the same directory.

The first six candidate-confirmation processes were stopped after this audit:
five had produced two of eight videos and the later `balanced` restart had not
yet produced a video. Their partial outputs remain under the ignored run tree
as failure evidence, but will not be aggregated or promoted. The uncached
references remain valid because they never ran calibration.

## Step-layer and external hybrid sweeps

### Reproducibility correction

Fresh-process reload removed calibration-state leakage, but repeated zero-hit
controls still exposed up to 0.15 LPIPS variation between independent
FlashAttention processes. Enabling FlashAttention 2's deterministic kernel
alone reduced but did not eliminate cross-GPU variation (one matched control
pair measured 0.0409 LPIPS). Therefore the formal protocol now generates an
original unpatched-forward reference immediately before each cache candidate
inside the same loaded process, on the same GPU, with the same prompt and seed.
Calibration remains isolated in an earlier process, and the resulting Cache
Book is freshly loaded for this paired generation process.

The paired zero-hit acceptance control passed on GPU 0 and GPU 1 for both
module-only and step-layer samplers: reference and candidate video hashes were
identical and LPIPS was exactly 0.0. The two GPUs also produced identical
reference hashes. All earlier independently generated LPIPS curves are retained
as diagnostic history but are not eligible for presets; formal curves below
this point use the same-process paired protocol.

Hunyuan step tuning begins with all six module thresholds fixed at 0.00, as
required by the step-first protocol. The table below is historical diagnostic
evidence from the superseded cross-process/reference protocol and is not used
for preset selection:

| step threshold | step hit rate | LPIPS vs original | generation seconds |
|---:|---:|---:|---:|
| 0.00 | 0.00% | 0.1306 | 103.2 |
| 0.10 | 6.00% | 0.3197 | 97.2 |

The first coarse increment is +0.1891, so the coarse elbow is 0.10 and the
fine grid is `[0.00,0.10]`. The q=0.20 job was stopped before model loading
once the elbow was known. Artifacts are in
`assets/hunyuan_step_layer/coarse_step_only/`.

The first fine-grid pass produced identical outputs for 0.01 and 0.02 (zero
step hits, LPIPS 0.0454), and identical outputs for 0.03 and 0.04 (one cached
step, LPIPS 0.2273). However, an independent same-process 0.00 repeat also had
zero hits but measured LPIPS 0.1375 and did not reproduce the earlier 0.00
video hashes. The audit isolated this to calibration runtime state: after
loading the 0.00 and 0.01 Cache Books in fresh processes on different GPUs,
all four corresponding video hashes were identical. A separately wrapped
module-only zero-threshold run was also byte-identical. All three canonical
zero-hit paths measure LPIPS 0.04539056. Candidate runners now always calibrate
and generate in separate Python processes, and the old same-process fine
points remain diagnostic only until their books are batch-reloaded.

After the fine step grid completes, module thresholds will be added in CCMR
order with the selected step-only output as the paired baseline. External
MagCache, TeaCache and SeaCache settings remain fixed at their sampler CLI
defaults.

## Three-sample low-cost screening checkpoint

The resumed screen uses exactly three independent content conditions, one
fixed seed, and the arithmetic mean of three paired LPIPS measurements. Image
models store the three outputs in one horizontal grid; video models use three
17-frame videos. These results are screening evidence, not the later native
frame/two-seed confirmation.

### DiT module-only

The causal boundary search found the MSA coarse elbow at 0.70 and the fine
elbow at 0.61, so its safe boundary is 0.60. With MSA fixed at 0.60, the MLP
coarse elbow was 0.70 and its fine elbow was also 0.61. Targeted adjacent
search then produced three non-uniform threshold pairs in the requested LPIPS
bands:

| tier | MSA | MLP | three-sample LPIPS |
|---|---:|---:|---:|
| fast | 0.60 | 0.65 | 0.4049 |
| balanced | 0.60 | 0.62 | 0.2137 |
| slow | 0.62 | 0.59 | 0.1073 |

The low-cost module-only values below are now captured in the versioned
presets. Native-frame/two-seed confirmation remains a separate gate; the
Hunyuan step-layer and hybrid budget evidence is summarized in the completion
addendum below.

### Current Hunyuan and FLUX boundaries

- Hunyuan step-only thresholds 0.01 and 0.02 have zero hits and exact paired
  LPIPS 0. Threshold 0.03 first enables one of 50 steps and measures 0.1814,
  locating the fine elbow at 0.03 and providing a balanced-band candidate.
- With Hunyuan `double.txt_attn=0.01` and `double.txt_mlp=0.12` locked,
  `double.img_attn=0.20` measures 0.0818 with a 0.33% effective joint-attention
  hit rate. The 0.30 coarse candidate is running.
- FLUX `context_ff` has its fine elbow at 0.06: 0.05 measures 0.0050 against
  the zero-cache reference, while 0.06 measures 0.0123, an adjacent increase
  above the 0.003 fine criterion. Its safe boundary is therefore 0.05.
- After locking `context_ff=0.05`, the first `context_attn=0.01` point adds
  0.0106 LPIPS and is retained as a target-budget candidate rather than a safe
  boundary.

### Current Wan and external-hybrid screen

- Wan `self_attn=0.01` measures 0.0188 over three prompts but enables only one
  of 1,500 module decisions. The earlier 0.10 point measures 0.0937 against the
  original. With 0.10 locked, adding `cross_attn=0.10` increases LPIPS by
  0.0203; the 0.20 coarse point is running.
- For Hunyuan hybrids, MagCache, TeaCache and SeaCache remain at their CLI
  defaults 0.03, 0.15 and 0.20. A hybrid candidate is not accepted merely
  because LPIPS is low: the logs must also show nonzero computed-step layer
  hits. Several 0.01--0.03 attention candidates have zero such hits because
  the external step policy already skips every step where their joint
  attention decision is true; these are controls, not valid hybrid presets.

## Failures and exclusions

- A 3-step Hunyuan CCMR smoke run verified feature collection but was excluded
  from the ranking.
- Four early `double.txt_mlp` candidates were interrupted before completion
  because the full CCMR ranking showed that `double.txt_attn` must be tuned
  first.
- The isolated `double.txt_attn` coarse sweep had zero effective attention
  cache hits because the paired image-attention gate was 0.00. It is not used
  as a quality curve or preset source.
- The first 121-frame confirmation launch assigned all eight prompts for one
  seed to a single GPU. Timing after five denoising steps projected roughly
  four hours per seed, so both jobs were stopped before any video was saved.
  Confirmation was relaunched as one prompt per GPU; these interrupted partial
  runs are excluded from every metric.
- The first step-only launcher attempt omitted argparse's required calibration
  prompt and exited before model loading. The harness was corrected and the
  zero-output attempts are excluded.
- The first native-frame candidate launch inherited the screening harness's
  forced GPU runtime-cache placement. Because 121-frame cache tensors are much
  larger than 17-frame tensors, the six jobs were stopped before calibration
  or output and relaunched with the sampler's native `auto` placement. This
  preserves a 2 GiB CUDA reserve and spills only cache tensors to pinned CPU
  when necessary; thresholds and cache decisions are unchanged.
- One `balanced` confirmation process initially failed before model loading
  with `EADDRINUSE`: concurrent one-GPU jobs inherited the same distributed
  rendezvous port. Candidate runners now ask the operating system for an
  ephemeral free `MASTER_PORT`, and only that zero-output job was relaunched.

## 2026-09-13 low-cost completion addendum

### Scope and reproducibility

The implementation is on branch `dev-local` (starting source commit
`dfee7edab5408d8bc8579ac306617a90a88c6306`). Screening used three independent
prompts, seed `2027`, 50 denoising steps and 17 video frames. Each prompt was
run in a separate GPU process where possible; LPIPS is the arithmetic mean of
the three paired reference-relative measurements. All video candidates were
calibrated in one process and then loaded by a fresh process before generation
(`fresh_process_reload=true` in the Hunyuan summaries). Raw videos, logs and
Cache Books remain in the ignored `runs/cache_tuning/` tree.

### Final module-only screening table

Values are reference-relative LPIPS macro means; `Pass` means the requested
screening interval was reached, while `Below`/`Unverified` is deliberately not
promoted as a quality claim.

| model | tier | independent module thresholds | LPIPS | target | status |
|---|---|---|---:|---:|---|
| DiT | fast | `msa=.60, mlp=.65` | 0.4049 | .35--.45 | Pass |
| DiT | balanced | `msa=.60, mlp=.62` | 0.2137 | .18--.22 | Pass |
| DiT | slow | `msa=.62, mlp=.59` | 0.1073 | .08--.12 | Pass |
| FLUX.1-dev | fast | `attn=.70, ctx_attn=.01, ff=.20, ctx_ff=.05, single_attn=.40, single_mlp=.02` | 0.3398 | .35--.45 | Below |
| FLUX.1-dev | balanced | `.30, .00, .04, .03, .10, .02` (same order) | 0.2014 | .18--.22 | Pass |
| FLUX.1-dev | slow | `.08, .00, .01, .02, .03, .00` (formal distinct preset) | 0.0753 | .08--.12 | Below |
| Wan2.1-1.3B | fast | `self=.25, cross=.30, ffn=.35` | 0.3549 | .35--.45 | Pass |
| Wan2.1-1.3B | balanced | `self=.10, cross=.15, ffn=.20` | 0.1942 | .18--.22 | Pass |
| Wan2.1-1.3B | slow | `self=.04, cross=.05, ffn=.07` | 0.1069 | .08--.12 | Pass |
| HunyuanVideo-1.5 | fast | `img_attn=.40, txt_attn=.01, img_mlp=.20, txt_mlp=.32` | 0.4259 | .35--.45 | Pass |
| HunyuanVideo-1.5 | balanced | `img_attn=.40, txt_attn=.01, img_mlp=.04, txt_mlp=.32` | 0.1841 | .18--.22 | Pass |
| HunyuanVideo-1.5 | slow | `img_attn=.40, txt_attn=.01, img_mlp=.02, txt_mlp=.00` | 0.0863 | .08--.12 | Pass |

The Hunyuan CCMR order is `double.txt_attn > double.txt_mlp >
double.img_attn > double.img_mlp` by median dB gain. Single-stream blocks are
absent in this transformer and stay disabled. DiT and FLUX orders follow the
formal CCMR assets; Wan has no formal dB asset and its order is explicitly a
pre-tuning prior. Every active module threshold is an independent field; the
vectors above are not a shared threshold.

The formal exact-to-0.01 values are stored in [`cache_presets.json`](../../cache_presets.json);
the direct CLI defaults are fast and list balanced/slow values beside them.
The FLUX slow value and its adjacent checks are detailed in the supplement
below; no below-band candidate was promoted. The Wan `.05/.07/.09` probe
measured 0.1335; the `.04/.05/.07`
intermediate probe closes the gap: its three prompt means are
`0.2007/0.1478/0.05197`, with frame-p95 maximum `0.2121` and temporal-delta
L1 mean `0.0180`.

### Step-layer and external hybrid budgets

Hunyuan step+layer (original-reference LPIPS) was measured as:

| tier | step | module vector (`img_attn,txt_attn,img_mlp,txt_mlp`) | LPIPS | layer delta vs step-only | paired-bootstrap 95% upper |
|---|---:|---|---:|---:|---:|
| fast | .20 | `.40,.01,.20,.32` | .3158 | +.0193 | .0273 |
| balanced | .03 | `.40,.01,.04,.12` | .1802 | −.0012 | .0246 |
| slow | .02 | `.40,.01,.00,.00` | .0950 | +.0950 | .1132 |

All three layer deltas satisfy the `.20` budget. Hybrid quality is measured
relative to its compatible step-only output, not mixed with original-relative
LPIPS:

| hybrid | module thresholds | delta LPIPS | bootstrap upper | budget result |
|---|---|---:|---:|---|
| MagCache | `img_attn=.90, txt_attn=.20` | .1170 | .1457 | Pass |
| TeaCache | `img_attn=.90, txt_attn=.20` | .1472 | .1621 | Pass |
| SeaCache | `img_attn=.90, txt_attn=.20` | .2233 | .3301 | Fail / not promoted |

MagCache, TeaCache and SeaCache external step parameters remain at their
existing CLI defaults. The SeaCache result exceeds the layer budget and is
therefore retained as a failure diagnostic rather than a valid preset.

### Curves, diagnostics and artifacts

`evaluation/build_tuning_report.py` generates the compact artifacts in this
directory: `threshold_summary.json/csv`, `module_curves.csv`,
`threshold_curves.png`, `module_lpips_curves.png`, `hybrid_delta.csv` and
`hybrid_lpips_delta.png`. The causal coarse/fine curves include the Hunyuan
CCMR-ranked families, DiT MSA, FLUX context-FF and Wan self-attention; raw
per-candidate JSON and Cache Books remain under `runs/cache_tuning/`.

Every `paired_lpips.py` result also stores frame p95/max, adjacent-frame LPIPS
drift and temporal-delta L1. For example, the accepted Hunyuan slow run has
frame p95 max `0.1433` and temporal-delta L1 mean `0.0203`; Wan fast-band2 has
frame p95 max `0.6136` and temporal-delta L1 mean `0.0283`. These diagnostics
are intentionally reported alongside the mean because framewise LPIPS alone
is not a temporal-quality guarantee. `ffprobe` checks on the 17-frame Hunyuan
and Wan outputs confirm single video streams at the expected geometry/FPS.

### Limitations and failures

No OOM occurred in the latest 17-frame screening; Hunyuan peak allocations
were approximately 37--56 GiB on H800 80 GB. The requested native 121-frame,
two-seed confirmation was not run in this low-cost pass, so no row above is a
production-quality validation. High-threshold Wan and FLUX probes showed
nonlinear LPIPS jumps and were excluded from presets. Missing target-band
points are explicitly marked `Below` or `Unverified`, never silently replaced
by a fabricated value. The additional FLUX `context_ff=.10` probe measured
0.3414 (still below the fast band), so the lower-risk `.05` context-FF fast
value remains the formal preset.

### FLUX external-hybrid paired screen

Using each method's external step policy as the paired baseline, the common
module vector was first screened at `.30/.20/.10/.05/.12/.02`. TeaCache was
then reduced once to the final distinct vector `.20/.10/.05/.03/.08/.01` after
its first result exceeded the 0.2 budget. Values below are three-panel grid
means, with a bootstrap upper bound over the three paired panels.

| method | module vector (`attn,ctx_attn,ff,ctx_ff,single_attn,single_mlp`) | delta LPIPS | bootstrap upper | budget |
|---|---|---:|---:|---|
| MagCache | `.20,.10,.05,.03,.08,.01` | .1639 | .1801 | Pass |
| TeaCache | `.20,.10,.05,.03,.08,.01` | .1397 | .1592 | Pass (rerun) |
| SeaCache | `.20,.10,.05,.03,.08,.01` | .1518 | .1645 | Pass |

The TeaCache first attempt (`.30/.20/.10/.05/.12/.02`) measured .2208/.2793
(mean/bootstrap upper) and is retained as a failure diagnostic. The paired
JSON and bootstrap records are under `runs/cache_tuning/flux_hybrid_lowcost/`.

### Wan2.1 external-hybrid paired screen

Wan was evaluated with unchanged external defaults. MagCache and SeaCache use
the shared module vector `self=.10, cross=.11, ffn=.12`; TeaCache uses its
separately confirmed vector `.40,.50,.60`. Each row uses three prompts,
17 frames and seed `2027`; the bootstrap upper bound resamples the three
prompt-level paired means. All methods remain within the requested `.20`
layer-quality budget in this screen.

| method | module vector (`self,cross,ffn`) | delta LPIPS | bootstrap upper | budget |
|---|---|---:|---:|---|
| MagCache | `.10,.11,.12` | .0167 | .0212 | Pass |
| TeaCache | `.40,.50,.60` | .0086 | .0140 | Pass |
| SeaCache | `.10,.11,.12` | .0490 | .0857 | Pass |

Prompt-level paired means were MagCache `.0109/.0244/.0148`, TeaCache
`.0094/.0163/.0000`, and SeaCache `.1199/.0100/.0172`. TeaCache produced
computed-step module skip ratios of `.27%/1.02%/.00%`; it therefore has real
module hits over the three-prompt set instead of the previous all-zero-hit
configuration. The SeaCache p0 value is the largest individual deviation but
remains below the aggregate budget. Original hybrid artifacts are under
`runs/cache_tuning/wan_hybrid_lowcost/`; the TeaCache confirmation is under
`runs/cache_tuning/quality_supplements/wan_teacache_probe_v3/`.

### Two-GPU quality confirmation supplement (2026-09-13)

This bounded supplement used only physical GPUs 6 and 7 and retained the
three-sample low-cost protocol. It did not measure or revise latency.

The true unpatched-model controls establish that a zero-threshold wrapper is
quality-equivalent when the inputs are identical:

| model | comparison | LPIPS | SHA-256 equality |
|---|---|---:|---|
| FLUX.1-dev | original pipeline vs q=0 wrapper | 0.0000 | yes |
| DiT-XL/2 | original forward vs q=0 wrapper, same latent | 0.0000 | yes |

For DiT, an independent run made after calibration is not a valid equivalence
control unless the latent RNG is restored: calibration consumes RNG state.
The paired same-latent check produced identical files (SHA-256 prefix
`69577537`); FLUX likewise produced identical files (prefix `976c10e0`). This
confirms the wrapper/reference quality measurement and isolates the previously
observed discrepancy to experiment seeding rather than cache arithmetic.

FLUX module-only confirmation used three prompts, seed 2027, 28 steps and
1024x1024 output. The original slow vector `.08,.00,.01,.02,.03,.00`
measured LPIPS `.07529`; `context_attn=.01` and `attn=.09` adjacent checks
both measured `.07534`. Raising only `context_attn` to `.10` or `.11` measured
the same `.078863`, still strictly below `.08`; neither point was promoted.
The fast vector
`.70,.01,.20,.05,.40,.02` measured `.33979`; `context_attn=.02`, `.05` and
`.10` measured `.33979`, `.33968` and `.34000`. Those changes are masked by
the combined cache decisions, so fast remains explicitly below the `.35-.45`
interval and its lower-risk preset is unchanged.

Wan TeaCache confirmation retained its external `teacache_thresh=.08` and
changed only the module thresholds. The accepted method-specific vector
`.40,.50,.60` has three-prompt mean LPIPS increase `.00857` relative to
TeaCache step-only and paired-bootstrap 95% upper `.01401`, both below `.20`.
The stronger `.60,.70,.80` probe was rejected after its first prompt measured
`.37993`; the lower `.20,.30,.40` probe had no runtime layer hits. The accepted
vector is therefore stored separately from the MagCache/SeaCache preset.
