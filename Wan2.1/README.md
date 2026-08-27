# Finegrained Cache for Wan2.1

Use the official Wan2.1 instructions to install dependencies and download model checkpoints first.

Then run:

```bash
python sample_wan.py
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

Finegrained Cache controls (the `--use_invardiff` and
`--invardiff_calibration` spellings are retained as command-line compatibility
aliases):

- `--use_invardiff`: enable Finegrained Cache acceleration.
- `--invardiff_calibration`: run Finegrained Cache calibration before acceleration.
- `--cache_book_path`: directory for cache books.
- `--cache_book_file`: specific cache book filename.
- `--nonskip_rate`, `--step_thres`, `--self_attn_thres`, `--cross_attn_thres`, `--ffn_thres`: cache policy thresholds.
