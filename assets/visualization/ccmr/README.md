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
