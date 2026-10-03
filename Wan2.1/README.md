# Wan2.1 module-level caching

Install the [official Wan2.1](https://github.com/Wan-Video/Wan2.1) dependencies
and obtain the model weights under the provider's terms. Make its `wan`
package importable (for example, add the upstream checkout to `PYTHONPATH`).
Set `WAN_MODEL` to your downloaded checkpoint directory.

## Calibrate and generate

Run from the repository root:

```bash
python Wan2.1/sample_wan.py \
  --task t2v-1.3B --size '832*480' --ckpt_dir "$WAN_MODEL" \
  --frame_num 81 --sample_steps 50 --base_seed 42 \
  --prompt "Two cats boxing under bright stage lights" \
  --invardiff_calibration --use_invardiff \
  --save_file outputs/wan.mp4
```

This performs two full-compute calibration passes followed by accelerated
generation. Omit `--use_invardiff` for calibration only; omit
`--invardiff_calibration` to generate from an existing matching Cache Book.
`--cache_book_path` and `--cache_book_file` select its location.

## Defaults and overrides

[`sample_wan.py`](sample_wan.py) uses one `default` module-only configuration
at 832 × 480, 81 frames, and 50 steps: self-attention/cross-attention/FFN
quantiles `.50/.35/.20`, two protected steps (`--nonskip_rate .04`), and one
calibration condition. Explicit `--self_attn_thres`, `--cross_attn_thres`,
and `--ffn_thres` flags override individual values.

[`sample_wan_step_layer.py`](sample_wan_step_layer.py) combines whole-step
and module reuse. Its single `default` uses step/self-attention/cross-attention/
FFN quantiles `.63/.82/1/.82`, two protected steps, and one calibration
condition. Use the same commands with the step-layer entrypoint and a newly
calibrated book. These settings were screened on small visual/held-out sets,
not a formal quality benchmark.

Recalibrate when changing weights, sampler, steps, resolution, frame count,
guidance, precision, or thresholds. For MagCache/SeaCache compatibility,
see [`hybrid_cache/README.md`](hybrid_cache/README.md). Inspect all options:

```bash
python Wan2.1/sample_wan.py --help
python Wan2.1/sample_wan_step_layer.py --help
```
