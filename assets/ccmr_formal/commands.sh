#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/InvarDiff

python -m pytest assets/visualization/ccmr/code/tests -q

# Formal DiT: completed on the RTX 4090 host.
python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs/dit_final.yaml \
  --output-dir assets/ccmr_formal/data/runs/dit_final --resume

# One-pair FLUX 1024px pilot used to measure the required wall-clock cost.
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs/flux_final_pilot.yaml \
  --output-dir assets/ccmr_formal/data/runs/flux_final_pilot --resume

# The full FLUX configuration is retained verbatim in configs/flux_final.yaml.
# It requires 24 pairs x 3 seeds and is not silently substituted by the pilot.

python assets/visualization/ccmr/code/aggregate_ccmr.py \
  --input assets/ccmr_formal/data/runs/dit_final assets/ccmr_formal/data/runs/flux_final_pilot \
  --config assets/ccmr_formal/configs/aggregate.yaml \
  --output-dir assets/ccmr_formal/data/combined/final_pilot

python assets/visualization/ccmr/code/plot_ccmr.py \
  --config assets/ccmr_formal/configs/plot.yaml \
  --input-dir assets/ccmr_formal/data/combined/final_pilot \
  --output-dir assets/ccmr_formal/figures/exploration/final_pilot_v2
