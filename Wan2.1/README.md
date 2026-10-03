# Wan2.1 module-level caching

Install the [official Wan2.1](https://github.com/Wan-Video/Wan2.1) dependencies
and obtain the model weights under the provider's terms. Make its `wan`
package importable (for example, add the upstream checkout to `PYTHONPATH`).
Set `WAN_MODEL` to your downloaded checkpoint directory.

## Load and generate

Run from the repository root:

```bash
python Wan2.1/sample_wan.py \
  --task t2v-1.3B --size '832*480' --ckpt_dir "$WAN_MODEL" \
  --frame_num 81 --sample_steps 50 --base_seed 3300 \
  --prompt "A golden retriever runs along a beach while the camera pans smoothly to follow it." \
  --save_file outputs/wan.mp4
```

This directly loads the matching configuration-named JSON in `cache_books/Wan2.1/`;
the step-layer entrypoint uses the same folder with a different strategy filename. Use
`--no-use_invardiff` for Full without caching.

## Optional recalibration

```bash
python Wan2.1/sample_wan.py --ckpt_dir "$WAN_MODEL" \
  --prompt "A red fox walks through a snowy forest while the camera tracks from the side." --base_seed 2027 \
  --invardiff_calibration

python Wan2.1/sample_wan.py --ckpt_dir "$WAN_MODEL" \
  --prompt "Two cats boxing under bright stage lights" --base_seed 43 \
  --save_file outputs/wan_custom.mp4
```

Calibration performs two full-compute passes. Explicitly add
`--use_invardiff` to also generate immediately afterwards. Without an output
folder it writes to `cache_books/Wan2.1/` using the
same configuration-derived filename as inference. Different configurations
coexist; recalibrating the same configuration updates its file.

## Defaults and overrides

[`sample_wan.py`](sample_wan.py) uses one `default` module-only configuration
at 832 × 480, 81 frames, and 50 steps: self-attention/cross-attention/FFN
quantiles `.50/.35/.20`, two protected steps (`--nonskip_rate .04`), and one
calibration condition. Explicit `--self_attn_thres`, `--cross_attn_thres`,
and `--ffn_thres` flags override individual values.

[`sample_wan_step_layer.py`](sample_wan_step_layer.py) combines whole-step
and module reuse. Its single `default` uses step/self-attention/cross-attention/
FFN quantiles `.63/.82/1/.82`, two protected steps, and one calibration
condition. Use the same commands with the step-layer entrypoint. These settings
were screened on small visual/held-out sets,
not a formal quality benchmark.
For a checked step-layer demo, use `sample_wan_step_layer.py`, seed `3100`,
and prompt `A bicyclist rides slowly past parked cars as the camera tracks smoothly from the side.`

Recalibrate when changing weights, sampler, steps, resolution, frame count,
guidance, precision, or thresholds. For MagCache/SeaCache compatibility,
see [`hybrid_cache/README.md`](hybrid_cache/README.md). Inspect all options:

```bash
python Wan2.1/sample_wan.py --help
python Wan2.1/sample_wan_step_layer.py --help
```
