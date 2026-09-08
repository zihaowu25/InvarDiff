#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/InvarDiff"
cd "$ROOT"

# Phase 0: model-free protocol tests.
python -m pytest assets/visualization/ccmr/code/tests -q \
  | tee assets/ccmr_formal/logs_v2/tests.log

# Phase 1: inspect each smoke manifest before formal collection.
python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs_v2/dit_512_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/dit_512_smoke_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_dit_512_smoke.log
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs_v2/flux_pairwise_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_smoke_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_flux_pair_smoke.log
python assets/visualization/ccmr/code/collect_flux_rho.py \
  --config assets/ccmr_formal/configs_v2/flux_rho_smoke.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_smoke_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_flux_rho_smoke.log

# Phases 2-3: execute only after Phase-0 audit and smoke approval.
python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs_v2/dit_512_formal.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_dit_512_formal.log
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs_v2/flux_pairwise_formal.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_flux_pair_formal.log
python assets/visualization/ccmr/code/collect_flux_rho.py \
  --config assets/ccmr_formal/configs_v2/flux_rho_formal.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 --resume \
  | tee assets/ccmr_formal/logs_v2/collector_flux_rho_formal.log

python assets/visualization/ccmr/code/aggregate_ccmr.py \
  --input assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
          assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2 \
          assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --config assets/ccmr_formal/configs_v2/aggregate_formal.yaml \
  --output-dir assets/ccmr_formal/data/combined/formal_v2 \
  | tee assets/ccmr_formal/logs_v2/aggregate.log

python assets/visualization/ccmr/code/validate_artifacts.py \
  --output assets/ccmr_formal/formal_manifest_v2.json

python assets/visualization/ccmr/code/export_paper_tables.py \
  --combined assets/ccmr_formal/data/combined/formal_v2 \
  --output-dir assets/ccmr_formal/tables/paper \
  --dit-run assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
  --flux-pair-run assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2 \
  --flux-rho-run assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --tests-log assets/ccmr_formal/logs_v2/tests.log

python assets/visualization/ccmr/code/compose_main_figure.py \
  --combined assets/ccmr_formal/data/combined/formal_v2 \
  --tables-dir assets/ccmr_formal/tables/paper \
  --output-dir assets/ccmr_formal/figures/paper \
  --dit-run assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
  --flux-pair-run assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2 \
  --flux-rho-run assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --tests-log assets/ccmr_formal/logs_v2/tests.log \
  | tee assets/ccmr_formal/logs_v2/plot.log
