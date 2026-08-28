#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m pytest assets/visualization/ccmr/code/tests -q
python assets/visualization/ccmr/code/collect_dit_ccmr.py --config assets/visualization/ccmr/configs/dit_smoke.yaml --output-dir assets/visualization/ccmr/data/runs/dit_smoke
python assets/visualization/ccmr/code/collect_flux_ccmr.py --config assets/visualization/ccmr/configs/flux_smoke.yaml --output-dir assets/visualization/ccmr/data/runs/flux_smoke
python assets/visualization/ccmr/code/collect_dit_ccmr.py --config assets/visualization/ccmr/configs/dit_final.yaml --output-dir assets/visualization/ccmr/data/runs/dit_final --resume
python assets/visualization/ccmr/code/collect_flux_ccmr.py --config assets/visualization/ccmr/configs/flux_final.yaml --output-dir assets/visualization/ccmr/data/runs/flux_final --resume
python assets/visualization/ccmr/code/aggregate_ccmr.py --input assets/visualization/ccmr/data/runs/dit_final assets/visualization/ccmr/data/runs/flux_final --output-dir assets/visualization/ccmr/data/combined
python assets/visualization/ccmr/code/plot_ccmr.py --config assets/visualization/ccmr/configs/plot.yaml --input-dir assets/visualization/ccmr/data/combined --output-dir assets/visualization/ccmr/figures/exploration
