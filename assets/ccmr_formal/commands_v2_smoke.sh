#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/InvarDiff"
cd "$ROOT"
mkdir -p assets/ccmr_formal/logs_v2

python -m pytest assets/visualization/ccmr/code/tests -q \
  |& tee assets/ccmr_formal/logs_v2/tests.log

python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs_v2/dit_512_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/dit_512_smoke_v2 --resume \
  |& tee assets/ccmr_formal/logs_v2/collector_dit_512_smoke.log

python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs_v2/flux_pairwise_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_smoke_v2 --resume \
  |& tee assets/ccmr_formal/logs_v2/collector_flux_pair_smoke.log

python assets/visualization/ccmr/code/collect_flux_rho.py \
  --config assets/ccmr_formal/configs_v2/flux_rho_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_smoke_v2 --resume \
  |& tee assets/ccmr_formal/logs_v2/collector_flux_rho_smoke.log

python assets/visualization/ccmr/code/validate_smoke_artifacts.py \
  --output assets/ccmr_formal/smoke_manifest_v2.json \
  |& tee assets/ccmr_formal/logs_v2/validate_smoke.log
