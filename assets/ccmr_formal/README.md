# CCMR collection and visualization protocols

This directory contains reproducible configurations and command files for
the DiT/FLUX CCMR mechanism experiments. Collectors are read-only observers:
all cache decisions are disabled, initial noise is fixed within each seed,
and the class or text condition varies. Only scalar statistics are saved;
full activation trajectories are not persisted.

## Setup and smoke checks

Install the corresponding model dependencies and download the weights first.
Set `checkpoint` (DiT) and `model_path` (FLUX) in the YAML files to your local
weights. The provided FLUX paths describe a Hugging Face snapshot layout;
replace them with your downloaded model directory. Preserve the model
revision and DiT checkpoint hash when reproducing the specified protocol.

From the repository root:

```bash
bash assets/ccmr_formal/commands_v2_smoke.sh
```

This runs synthetic tests, small collection jobs, and artifact validation.
Review the resulting manifest before starting full collection.

## Formal collection

The `configs_v2/` protocols use DiT-XL/2 at 512² with 50 DDIM steps and
FLUX.1-dev at 1024² with 28 steps. The active FLUX pairwise protocol uses
12 deterministic prompt pairs per seed; a separate per-prompt collector
provides rho dispersion and calibration-subset statistics. Pair-level
differences must not be interpreted as per-prompt rho.

Full collection is expensive and requires explicit opt-in:

```bash
CCMR_FORMAL_APPROVED=YES bash assets/ccmr_formal/commands_v2_formal.sh
```

The workflow collects scalar shards, aggregates statistics, validates
completeness/checksums, and exports tables and figures. `--resume` reuses
completed shards only after validation. Formal tools reject incomplete
or incompatible runs rather than substituting pilot results.

`rho_clean` is a pure-L1 diagnostic; `rho_code` follows the implemented
distance/rate calculation, including epsilon handling. These measurements
describe the cache-priority mechanism, not a bound on generation error.

## Source and outputs

Collectors, plotting, aggregation, and tests are in
[`../visualization/ccmr/`](../visualization/ccmr/README.md). Generated `data/`,
`figures/`, `logs_v2/`, `tables/`, and run manifests are local, Git-ignored
artifacts. They are not distributed with the source repository.

The separate `configs/` and `commands.sh` retain earlier exploratory
DiT-256/FLUX pilot protocols. Do not mix these outputs with formal v2 runs.
