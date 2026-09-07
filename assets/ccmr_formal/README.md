# Formal CCMR experiment bundle

This directory contains the formal CCMR collection requested for the ICLR 2027
analysis.  It is deliberately separate from the earlier smoke and pilot
outputs in `assets/visualization/ccmr/`.

The collectors are read-only observers: all DiT and FLUX cache decisions are
disabled, the initial latent is fixed while conditions change, and statistics
are accumulated in FP32.  The implementation lives in
`assets/visualization/ccmr/code/`; this directory contains the immutable run
configuration copies, logs, compressed scalar tables, figures, and manifests.

## Formal configurations

* DiT: `configs/dit_final.yaml`, 256x256, 50 DDIM steps, CFG 4, 16 exact
  ImageNet conditions, seeds 0--4, float32.
* FLUX: `configs/flux_final.yaml`, 1024x1024, 28 steps, BF16, guidance 3.5,
  12 prompts and 24 uniformly selected unordered pairs per seed.

The DiT 256 checkpoint is available on the RTX 4090 host and was used for the
formal run.  A one-pair FLUX pilot is stored separately before attempting the
full pair matrix; its runtime is recorded in the report.  A full FLUX matrix
must not be inferred from that pilot.

## Reproduction

```bash
cd /root/autodl-tmp/InvarDiff
python -m pytest assets/visualization/ccmr/code/tests -q
python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs/dit_final.yaml \
  --output-dir assets/ccmr_formal/data/runs/dit_final --resume
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs/flux_final.yaml \
  --output-dir assets/ccmr_formal/data/runs/flux_final --resume
python assets/visualization/ccmr/code/aggregate_ccmr.py \
  --input assets/ccmr_formal/data/runs/dit_final assets/ccmr_formal/data/runs/flux_final \
  --config assets/ccmr_formal/configs/aggregate.yaml \
  --output-dir assets/ccmr_formal/data/combined/final
python assets/visualization/ccmr/code/plot_ccmr.py \
  --config assets/ccmr_formal/configs/plot.yaml \
  --input-dir assets/ccmr_formal/data/combined/final \
  --output-dir assets/ccmr_formal/figures/exploration/final
```

Each run stores `config.json`, `conditions.json`, `environment.json`,
`latent_hashes.json`, `memory.json`, per-seed shards, and an atomic
`summary.json`.  Aggregated tables are compressed CSV and can be plotted again
without loading model weights.  The report is kept under
`agent_skills/report/` as requested.
