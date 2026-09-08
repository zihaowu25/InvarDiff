#!/usr/bin/env bash
set -euo pipefail

if [[ "${CCMR_FORMAL_APPROVED:-}" != "YES" ]]; then
  echo "Formal collection is locked. Set CCMR_FORMAL_APPROVED=YES only after smoke audit approval." >&2
  exit 2
fi

ROOT="/root/autodl-tmp/InvarDiff"
cd "$ROOT"
mkdir -p assets/ccmr_formal/logs_v2

python assets/visualization/ccmr/code/collect_dit_ccmr.py \
  --config assets/ccmr_formal/configs_v2/dit_512_formal.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 --resume \
  2>&1 | tee assets/ccmr_formal/logs_v2/collector_dit_512_formal.log
python assets/visualization/ccmr/code/collect_flux_ccmr.py \
  --config assets/ccmr_formal/configs_v2/flux_pairwise_formal12.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 --resume \
  2>&1 | tee assets/ccmr_formal/logs_v2/collector_flux_pair_formal12.log
python assets/visualization/ccmr/code/collect_flux_rho.py \
  --config assets/ccmr_formal/configs_v2/flux_rho_formal.yaml \
  --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 --resume \
  2>&1 | tee assets/ccmr_formal/logs_v2/collector_flux_rho_formal.log

python assets/visualization/ccmr/code/pair_count_stability.py \
  --run assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 \
  --output-dir assets/ccmr_formal/tables/pair_count_stability \
  2>&1 | tee assets/ccmr_formal/logs_v2/pair_count_stability.log

python assets/visualization/ccmr/code/aggregate_ccmr.py \
  --input assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
          assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 \
          assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --config assets/ccmr_formal/configs_v2/aggregate_formal.yaml \
  --output-dir assets/ccmr_formal/data/combined/formal_v2 \
  2>&1 | tee assets/ccmr_formal/logs_v2/aggregate.log

python assets/visualization/ccmr/code/validate_artifacts.py \
  --output assets/ccmr_formal/formal_manifest_v2.json

python assets/visualization/ccmr/code/export_paper_tables.py \
  --combined assets/ccmr_formal/data/combined/formal_v2 \
  --output-dir assets/ccmr_formal/tables/paper \
  --dit-run assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
  --flux-pair-run assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 \
  --flux-rho-run assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --tests-log assets/ccmr_formal/logs_v2/tests.log

python assets/visualization/ccmr/code/compose_main_figure.py \
  --combined assets/ccmr_formal/data/combined/formal_v2 \
  --tables-dir assets/ccmr_formal/tables/paper \
  --output-dir assets/ccmr_formal/figures/paper \
  --dit-run assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 \
  --flux-pair-run assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 \
  --flux-rho-run assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 \
  --tests-log assets/ccmr_formal/logs_v2/tests.log \
  2>&1 | tee assets/ccmr_formal/logs_v2/plot.log
