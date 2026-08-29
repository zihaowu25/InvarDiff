# CCMR Visualization Experiment Report

## Scope and status

This report records the CCMR (condition-consistency mismatch ratio) experiment
for full-compute DiT and FLUX trajectories.  Cache decisions are disabled in
all collectors.  The hooks are read-only and are attached to the same module
outputs used by the layer-level cache implementation.  The current checkout
contains completed smoke tests and one formal-resolution pilot per model.  The
full multi-seed/pair FLUX matrix is intentionally not claimed as complete: the
1024px collector takes about 16 minutes for one pair on the RTX 4090, making 24
pairs × 3 seeds prohibitively expensive for this run.

The experiment is therefore suitable for validating the implementation,
geometry, statistics and plotting pipeline, and for reporting the pilot
resource cost.  It is not a replacement for the planned full population
estimate.

## Reproducibility metadata

| Item | Value |
| --- | --- |
| Repository | `/root/autodl-tmp/InvarDiff` |
| Git commit at collector start | `71dacede9a1a2336794008c3b826b15e56c9d2f1` |
| GPU | NVIDIA GeForce RTX 4090 (CUDA available) |
| Torch | 2.8.0+cu128 |
| Diffusers | 0.39.0 |
| Statistics dtype | FP32 accumulation |
| Cache | Disabled (`cache_enabled: false`) |
| Output root | `assets/visualization/ccmr/` |

The exact environment and checkpoint paths are stored in each run's
`environment.json`.  Latent hashes are written to `latent_hashes.json`.

## Configurations

### DiT

The formal pilot uses DiT-XL/2 with the 512 checkpoint, 512×512 geometry, 50
DDIM steps, CFG scale 4.0, one seed (`0`) and an exact conditional batch of
16 ImageNet classes.  The formal YAML retains the planned five seeds
`[0,1,2,3,4]`; only seed 0 was run in this pilot.  The smoke run uses the
matching 256 checkpoint, 256×256 geometry, two classes and six steps.

The collector calls `forward_with_cfg`, passing the conditional half followed
by the null-label half.  `DiTBlock` hooks record gate-before MSA and MLP
outputs; only the conditional half enters condition statistics.

### FLUX

The formal pilot uses the local FLUX.1-dev snapshot, 1024×1024, BF16, 28
steps, guidance 3.5, seed 0 and one deterministic unordered prompt pair.  The
formal YAML retains the planned 12-prompt bank and 24 pairs selected with
`pair_selection_seed=2027`; only one pair was executed.  The smoke run uses
512×512, six steps and one pair.

The collector wraps the local dynamic transformer with all six module families:
`double.attn`, `double.context_attn`, `double.ff`, `double.context_ff`,
`single.attn` and `single.mlp`.  For the two-prompt pilot it stores the FP32
pair difference `feature[0]-feature[1]`, which is mathematically sufficient
for K=2 population variance, aligned/shuffled gain and pair distance.  This
compact path avoids retaining both 1024px activations.  Per-condition temporal
and rho diagnostics are mirrored from the pair representation and are
explicitly marked `rho_scope=pair_difference`; they are not used for
condition-level dispersion or subset-stability conclusions.  A separate
`collect_flux_rho.py` collector is available when true per-prompt rho is
required.

## Statistics

For condition features `Z`, the collector computes population condition
variance for the raw activation and for the aligned temporal difference.  The
reported ratio and gain are

\[
R=\frac{V^{diff}}{(V^{raw}_t+V^{raw}_{t-1})/2+\epsilon},\qquad
G=10\log_{10}\frac{(V^{raw}_t+V^{raw}_{t-1})/2+\epsilon}
{V^{diff}+\epsilon}.
\]

The rho diagnostic uses only interior step indices (`1..T-2`) and the
two-point L1 ratio.  Boundary entries are invalid by construction.  All
statistics use FP32; input features keep their model dtype until the
accumulation operation.  Pairwise FLUX tables retain the full prompt
population size so the aggregator can apply the finite-population correction.

## Executed runs

| Run | Conditions | Steps | Collector elapsed | Peak allocated GPU | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `dit_smoke` | 2 | 6 | 0.874 s | 3.350 GiB | complete |
| `flux_smoke` | 2 | 6 | 8.537 s | 44.189 GiB | complete (pre-compact path) |
| `dit_final_pilot` | 16 | 50 | 132.804 s | 43.903 GiB | complete, seed 0 |
| `flux_timegap_fixed` | 1 pair | 28 | 996.068 s (16:36.1) | 32.785 GiB | complete, seed 0; corrected compact time-gap path |

The first FLUX smoke run is retained as a regression record for the original
full-feature hook path.  The earlier `flux_final_single_pilot` output is also
retained as a historical run, but its compact time-gap table is not used after
the token/condition indexing bug was found.  The corrected
`flux_timegap_fixed` run uses the same compact pair path and stays below the
96-GB process memory limit.  An experimental all-GPU compact path was rejected
after an immediate CUDA OOM (47.34 GiB allocated); it is not used in the
reported runs.

The formal pilot aggregate contains 7,056 CCMR rows, 6,848 condition-similarity
rows, 333,384 condition-distance rows, 52,112 temporal rows, 12,112 time-gap
rows and 50,912 rho rows.  The aggregate summary reports 2,688
condition-scope rho-stability cells (DiT only) and 1,000 fixed-subset stability
rows; compact FLUX pair-difference rho is correctly excluded from those
condition-level statistics.  It also reports 56 invalid DiT first-step cells
and 152 invalid FLUX first-step cells, exactly matching the expected boundary
initialization.

## Pilot quantitative summary

The following values are from
`data/aggregate_pilot/summary.json`.  Values are first averaged within each
seed and module; the interval is therefore a seed-level bootstrap summary.  As
this pilot contains one seed per model, the displayed interval collapses to the
observed value and must not be interpreted as population uncertainty.

| Model/module | Gain mean (dB) | Gain median (dB) | 2.5–97.5% bootstrap interval |
| --- | ---: | ---: | ---: |
| DiT MSA | 18.613 | 18.613 | [18.613, 18.613] |
| DiT MLP | 17.277 | 17.277 | [17.277, 17.277] |
| FLUX double attention | 16.129 | 16.129 | [16.129, 16.129] |
| FLUX double context attention | 17.786 | 17.786 | [17.786, 17.786] |
| FLUX double FF | 14.680 | 14.680 | [14.680, 14.680] |
| FLUX double context FF | 21.574 | 21.574 | [21.574, 21.574] |
| FLUX single attention | 10.357 | 10.357 | [10.357, 10.357] |
| FLUX single MLP | 7.713 | 7.713 | [7.713, 7.713] |

The lower single-stream gains indicate larger temporal mismatch than the
double-stream context FF in this pilot; this is a descriptive observation, not
a cache-threshold recommendation.  No LPIPS is reported because CCMR
collection intentionally performs latent/statistical observation only and
does not decode or compare generated images.

## Figures

`figures/pilot/` and `figures/smoke/` each contain 25 figure families, each as
PDF and 300-DPI PNG.  The families include module-specific
raw/difference/gain heatmaps, raw-vs-difference density, gain ECDF/violin, gain
over time, condition similarity and distance matrices, rho mean/median and
dispersion heatmaps, rho distributions, subset stability, temporal smoothness,
and time-gap ablation.  All plots are regenerated from aggregate scalar CSV
files; no activation tensor is required for plotting.  Colormaps are
configured in `configs/plot.yaml` and use the specified perceptually ordered
palettes and Okabe–Ito module colors.

### Plotting validation and scaling

The plotting code masks all non-finite values, including the intentionally
invalid first-step CCMR cells.  Variance heatmaps display `log10(V)` and use a
robust global 0.5--99.5 percentile color range; the percentile configuration
is stored in fractional form and converted to NumPy's percentage convention
before limits are computed.  Gain and rho heatmaps use the corresponding
finite robust limits, while each heatmap panel remains module-specific so
different block families are not silently pooled.  The rho panels use
`score_step_idx` (the rho table's schema) rather than the CCMR table's
`step_idx`.  Pairwise distance matrices aggregate repeated layer/time
observations by median, display raw/difference distances on a `log10` scale,
and mask the unobserved diagonal and pairs.  ECDFs are sorted step functions;
time-gap panels are split by model and show median/IQR.  When an optional table
is unavailable, the figure is rendered with an explicit no-data annotation
instead of an axes-only blank image.

## Validation performed

* Eighteen synthetic tests passed (`pytest assets/visualization/ccmr/code/tests -q`),
  including a check that pairwise time-gap gains are computed from aggregated
  energies rather than averaged per-pair dB values.
* All collector, aggregator and plotting modules pass `py_compile`.
* All four CLIs respond to `--help` without initializing a model.
* DiT smoke and 512px pilot completed with the expected hook counts and no
  NaN/Inf runtime failure.
* FLUX smoke and 1024px single-pair pilot completed with deterministic latent
  duplication, correct module hook counts and no GPU OOM in the compact CPU
  path.
* The corrected FLUX compact time-gap implementation was rerun.  For all
  4,104 gap-1 cells, `g_ccmr_gap_db` matches the adjacent CCMR gain exactly
  (maximum absolute difference 0.0 dB).
* Pairwise aggregation now emits one row per
  `(source_run, model, seed, module, layer, step)` cell, applies the same
  finite-population scaling to current and previous variance, and avoids pair
  duplication.
* Fixed-subset stability selects condition IDs once per trial and reuses that
  subset across every layer and step.  Compact FLUX rho mirrors are filtered
  from condition-level stability; use `collect_flux_rho.py` for true prompt
  rho.
* All regenerated pilot/smoke PNGs contain finite plotted data or an explicit
  no-data annotation; no figure relies on unfiltered `NaN` values.
* The aggregator duplication bug found during smoke processing was fixed and
  verified: two runs now produce 1,248 CCMR rows rather than repeated rows.

## Limitations and next run

1. The planned DiT five-seed and FLUX 24-pair × three-seed formal collections
   are not present in this run; only seed-0 pilots are complete.
2. The FLUX formal pilot uses one pair and `time_gaps=[1]` in its temporary
   execution config to control memory.  The checked-in formal YAML still
   specifies `[1,2,4]` and the full prompt bank.
3. K=2 compact FLUX temporal/rho rows are pair-difference diagnostics; they do
   not provide per-prompt rho dispersion.  Full condition consistency requires
   running the planned 24 unordered pairs and the lightweight per-prompt rho
   collector across the 12-prompt population.
4. LPIPS, CLIP/DINO, VBench and video metrics are outside this CCMR collector;
   they belong to the separate cache-tuning experiments.

To continue, run the commands in `commands.sh` with the formal YAMLs and
`--resume`, then aggregate only after every seed/pair shard is complete.  The
existing pilot data and figures are retained and can be compared with the
full-population aggregate.
