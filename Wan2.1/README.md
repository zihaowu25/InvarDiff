# Wan2.1 module-level caching

Use the official Wan2.1 instructions to install dependencies and download model checkpoints first.

[`sample_wan.py`](sample_wan.py) is the standalone module-only sampler.
Its selected 832 × 480, 81-frame, 50-step configuration uses self-attention,
cross-attention, and FFN thresholds `0.50/0.35/0.20`, two protected initial
steps (`--nonskip_rate 0.04`), and one calibration condition. The `default` module
preset is selected automatically; explicitly supplied threshold flags take
precedence. Calibrate a new Cache Book when the model or execution settings
change. Hybrid policies each use one `hybrid` configuration. The separate
[`sample_wan_step_layer.py`](sample_wan_step_layer.py) combines whole-step and
module reuse. Its single `default` at 832 × 480, 81 frames and 50 steps uses
step/self-attention/cross-attention/FFN quantiles `.63/.82/1/.82`, two protected
steps and one calibration condition. These settings were checked on small
visual and held-out sets; the same cache-generation flags apply.

Inspect the available options from the repository root:

```bash
python Wan2.1/sample_wan.py --help
```

You can change parameters in the script (for example in `debug_args`) or pass them from command line.

Common parameters you may want to modify:

- `--task`: model/task type, such as `t2v-1.3B`, `t2v-14B`, `i2v-14B`.
- `--ckpt_dir`: checkpoint folder.
- `--size`: output resolution, such as `832*480` or `1280*720`.
- `--frame_num`: number of output frames.
- `--sample_steps`: number of sampling steps.
- `--sample_shift`: sampling shift value.
- `--sample_guide_scale`: CFG guidance scale.
- `--base_seed`: random seed for reproducibility.
- `--save_file`: output path and filename.

Module-level cache controls (the `--use_invardiff` and
`--invardiff_calibration` spellings are retained as command-line compatibility
aliases):

- `--use_invardiff`: enable Finegrained Cache acceleration.
- `--invardiff_calibration`: run Finegrained Cache calibration before acceleration.
- `--cache_book_path`: directory for cache books.
- `--cache_book_file`: specific cache book filename.
- `--nonskip_rate`, `--self_attn_thres`, `--cross_attn_thres`,
  `--ffn_thres`: module cache policy thresholds. Whole-step reuse is not
  enabled by this sampler; step-layer and hybrid implementations are separate.
