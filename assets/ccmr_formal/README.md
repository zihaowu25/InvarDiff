# Formal CCMR experiment bundle

> Current legacy status: **DiT-256 complete mechanism run plus FLUX-1024
> one-pair pilot**. These artifacts are exploratory and not paper eligible.
> Formal v2 results are admitted only by `validate_artifacts.py`; paper tools
> never fall back to pilot data.

## Audited v2 workflow

The v2 protocol is under `configs_v2/`. Smoke commands are isolated in
`commands_v2_smoke.sh`. Formal commands are isolated in
`commands_v2_formal.sh` and additionally require the explicit environment
gate `CCMR_FORMAL_APPROVED=YES`. Phase 0 changes and tests code only; formal
DiT-512 and FLUX-1024 collection cannot follow smoke automatically.

Formal scalar shards live under `data/runs_v2/`. Each completed shard records
a resolved-config hash, row counts, SHA-256 checksums, latent hashes, and an
explicit status. `--resume` reuses a shard only after these checks pass. Full
activation trajectories are never persisted.

The active FLUX formal protocol is the cost-reduced T02 revision. It uses a
preregistered balanced cycle of 12 unordered prompt pairs for each of three
seeds (36 pairwise shards total), at 1024x1024 and 28 steps. The deterministic
pair list is generated with seed 2027; every prompt has degree two and the
graph is one connected cycle. The interrupted 24-pair run is archived under
`data/runs_v2/flux_pairwise_formal24_aborted_20260908/`, is explicitly marked
ineligible, and is rejected by the formal aggregator.

The first T02 shard is collected with `--max-new-shards 1` and audited as an
integration gate. Only after its pair ID, commit, checksums, exact row counts,
memory, and attempt history pass is the same run resumed for the remaining
35 shards. Long collectors are launched under `screen`; the full formal
command file must not be used as an unattended integration-smoke command.

The mechanism experiment shares an initial latent within each seed and varies
only the class or prompt. This isolates the CCMR mechanism; it is distinct
from deployment evidence based on independently generated calibration items.

`rho_clean` is the pure-L1 mathematical diagnostic. `rho_code` calls the live
cache distance/rate implementation, including elementwise epsilon, and the
formal gate checks configured numerical tolerances.

The natural-language reports are stored locally in the Git-ignored
`agent_skills/report/` directory. Their sortable naming convention is
`CCMR_REPORT_R<round>_<stage>_<YYYYMMDD>.md`; they must never be staged.

This directory contains the CCMR formal-evaluation workflow and retained
legacy exploratory artifacts. It is separate from earlier development plots
in `assets/visualization/ccmr/`.

The collectors are read-only observers: all DiT and FLUX cache decisions are
disabled, the initial latent is fixed while conditions change, and statistics
are accumulated in FP32.  The implementation lives in
`assets/visualization/ccmr/code/`; this directory contains the immutable run
configuration copies, logs, compressed scalar tables, figures, and manifests.

## Legacy configurations

* DiT: `configs/dit_final.yaml`, 256x256, 50 DDIM steps, CFG 4, 16 exact
  ImageNet conditions, seeds 0--4, float32. This is an explicit exploratory
  protocol deviation.
* FLUX: the completed legacy artifact is a 1024x1024, one-seed, one-pair
  pilot. The old `configs/flux_final.yaml` describes intended coverage, not
  completed coverage.

The DiT-256 and one-pair FLUX artifacts remain useful regression references.
Neither may be mixed into the v2 aggregate or presented as paper-complete.

## Legacy reproduction only

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

These commands reproduce exploratory v1 outputs only. For formal collection,
use the split v2 command files. V2 runs store auditable scalar shard manifests and can
be aggregated and plotted without model weights. Natural-language reports are
not stored under `agent_skills/` or any other tracked repository path.
