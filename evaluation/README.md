# Evaluation and validation

Run these tools from the repository root. Generated media and metric reports
are local artifacts and should be written under `outputs/` or `runs/`.

## Paired fidelity

`paired_lpips.py` compares one image pair or matched video pair. It records
LPIPS, PSNR, SSIM, hashes, and video temporal diagnostics. Install PyTorch,
NumPy, Pillow, OpenCV, and `lpips` (metric weights may need downloading).

```bash
python evaluation/paired_lpips.py \
  --reference outputs/full.mp4 --candidate outputs/cached.mp4 \
  --device cuda --output outputs/paired_metrics.json
```

Use the same prompt/class, seed, initial noise, resolution, frame count,
frame rate, guidance, and sampler. Shape/frame-rate mismatches are rejected;
the tool does not silently resize or truncate a pair. For horizontal image
grids, specify `--image-grid-count` and `--image-grid-padding` explicitly.
Reference fidelity is not semantic quality or an official VBench Total Score.

## Dataset and candidate checks

- `validate_video_set.py` checks expected video count, dimensions, frames,
  and FPS. Inspect `--help` for the required fields.
- `generate_uncached_reference.py` generates small unpatched DiT/FLUX
  references for zero-cache checks; supply your checkpoint/model paths.
- `run_hunyuan_module_candidate.py`, `run_hunyuan_step_layer_candidate.py`,
  and `reload_hunyuan_candidates.py` support candidate calibration, generation,
  and paired diagnostics. Their short-video screening defaults are not a
  substitute for native-length validation.
- `hunyuan_ccmr_pair.py` and `aggregate_hunyuan_ccmr.py` collect/aggregate
  read-only mechanism statistics. `aggregate_paired_delta.py` performs paired
  comparisons; video-level summaries treat videos, not frames, as samples.

For Hunyuan helpers, set `HUNYUAN_REPO` and `HUNYUAN_MODEL` to your upstream
checkout and model directory. Optional `HUNYUAN_PYTHON` and `METRIC_PYTHON`
select separate Python environments; both default to the current interpreter.
See [`../HunyuanVideo/README.md`](../HunyuanVideo/README.md) for model setup.

Install the optional test and plotting dependencies:

```bash
pip install pytest pandas PyYAML matplotlib
python -m pytest evaluation/tests -q
```

Some integration checks require upstream model packages; no pretrained
weights or private experiment reports are included in this directory.
