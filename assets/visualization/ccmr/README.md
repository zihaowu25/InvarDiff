# CCMR visualization tools

Read-only collectors observe the module outputs used by the DiT and FLUX
Cache Books, with caching disabled. Included source covers scalar collection,
aggregation, rho/variance heatmaps, condition-distance and time-gap plots,
calibration-subset stability, artifact validation, and paper-figure export.

## Run

Install the model dependencies plus NumPy, pandas, PyYAML, matplotlib, and
pytest. Run the synthetic tests from the repository root:

```bash
python -m pytest assets/visualization/ccmr/code/tests -q
```

For the current smoke/formal protocol and commands, see
[`../../ccmr_formal/README.md`](../../ccmr_formal/README.md). The local
`configs/` and `commands.sh` provide separate exploratory examples; update
their checkpoint paths before use.

Collectors support `--resume` and save completed seed/pair shards atomically.
Aggregation and plotting operate on scalar tables, without loading model
weights. Generated `data/`, `figures/`, and `reports/` are ignored by Git;
conditions, configurations, code, and tests remain tracked.

## Interpretation

- The aggregate uses population variance and finite-population correction
  for pairwise data; invalid or degenerate cells remain explicit.
- Pairwise FLUX rows describe prompt differences, not individual-prompt rho.
  Use `collect_flux_rho.py` for per-prompt rho and subset stability.
- Heatmaps respect the stored score-step indices; invalid boundary values
  are not treated as valid reuse positions.
- Figures are exported as PDF and PNG. A completed plot or collector does
  not itself establish perceptual quality or a measured speedup.
