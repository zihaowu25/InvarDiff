# CCMR visualization experiment

This directory contains the deterministic full-compute CCMR experiment for
DiT and FLUX.  It is intentionally separate from model and Cache code.  The
collectors add read-only hooks at the exact module outputs reused by the
Finegrained Cache implementations; no cache decision is enabled during data
collection.

## Reproduction

Run the synthetic tests first:

```bash
python -m pytest assets/visualization/ccmr/code/tests -q
```

Then run smoke, pilot, and formal jobs in order.  Every collector supports
`--resume` and writes atomically completed seed/pair shards:

```bash
python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/visualization/ccmr/configs/dit_smoke.yaml \
  --output-dir assets/visualization/ccmr/data/runs/dit_smoke
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/visualization/ccmr/configs/flux_smoke.yaml \
  --output-dir assets/visualization/ccmr/data/runs/flux_smoke
```

The formal DiT job is 512×512 with the local `DiT-XL-2-512x512.pt` checkpoint,
16 exact conditions, and five seeds.  The formal FLUX job is 1024×1024 and
uses 24 deterministic prompt pairs from the frozen 12-prompt bank.

Aggregate only after all shards are complete:

```bash
python assets/visualization/ccmr/code/aggregate_ccmr.py \
  --input assets/visualization/ccmr/data/runs/dit_final \
          assets/visualization/ccmr/data/runs/flux_final \
  --config assets/visualization/ccmr/configs/aggregate.yaml \
  --output-dir assets/visualization/ccmr/data/combined
python assets/visualization/ccmr/code/plot_ccmr.py \
  --config assets/visualization/ccmr/configs/plot.yaml \
  --input-dir assets/visualization/ccmr/data/combined \
  --output-dir assets/visualization/ccmr/figures/exploration
```

The aggregate uses population variance, preserves invalid/degenerate cells,
and applies the finite-population correction for pairwise data.  All figures
are rendered as PDF and 300-DPI PNG from scalar tables only.  See
`reports/REPORT.md` for the executed configuration, memory, failures and
quantitative conclusions.

Plotting is robust to the `NaN` values used for invalid boundary steps.  Raw
and difference variance heatmaps use `log10` values with a global robust
0.5--99.5 percentile color range, and each module is shown in its own panel.
Rho heatmaps use the table's `score_step_idx`; repeated condition-distance
observations are reduced by their median across layers and time.  ECDFs are
sorted step functions, time-gap plots are split by model with IQR bands, and
condition-distance plots expose raw, difference, and contraction panels.  The
FLUX pairwise time-gap path persists current/previous/difference pair energies
so aggregation applies the finite-population correction before computing dB
gain; it does not average per-pair gains.
For the compact two-prompt FLUX pilot, the rho mean/median heatmaps use a
clearly labelled pair-difference diagnostic fallback, while condition-level
rho dispersion and subset stability remain unavailable until the dedicated
per-prompt rho collector is run.  Optional plots with no valid rows are emitted
with an explicit no-data annotation.

The pairwise FLUX collector stores compact prompt differences.  Those rows are
valid for pair-level CCMR and time-gap diagnostics, but must not be interpreted
as per-prompt rho.  Run the lightweight condition-scope collector when rho
dispersion or calibration-subset stability is needed:

```bash
python assets/visualization/ccmr/code/collect_flux_rho.py \
  --config assets/visualization/ccmr/configs/flux_final.yaml \
  --output-dir assets/visualization/ccmr/data/runs/flux_rho_final \
  --resume
```

It runs one prompt at a time, reuses one latent per seed, and persists a
complete `seed_<seed>_prompt_<id>` shard after every prompt.  The regular FLUX
collector uses the same durable per-pair shard layout, so interrupted long
jobs can resume without losing completed pair statistics.
