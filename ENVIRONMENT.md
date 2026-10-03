# Tested environments

Use separate model environments when their upstream dependency requirements
differ. These versions describe the existing release-validation environments;
they are not a claim that every newer package version is compatible.

| Package | DiT / FLUX / Wan environment | HunyuanVideo-1.5 environment |
| --- | --- | --- |
| Python | 3.11.16 | 3.11.16 |
| PyTorch | 2.4.1+cu121 | 2.6.0 |
| diffusers | 0.31.0 | 0.35.0 |
| transformers | 4.49.0 | 4.57.1 |
| accelerate | 1.15.0 | 1.15.0 |
| NumPy | 1.26.4 | 1.26.4 |
| timm | 1.0.15 (DiT) | Not required by these entrypoints |

Install the CUDA-compatible PyTorch build for your system, then install the
model's upstream requirements and use the matching versions above as a
compatibility reference. Model weights must be downloaded separately under
their providers' terms. FLUX uses diffusers Transformer internals, so validate
an upgrade before reusing a released Cache Book.

## Upstream source

- DiT model and diffusion source is included; its license is separate.
- FLUX requires `diffusers` and model access for FLUX.1-dev.
- Wan requires the official [Wan2.1](https://github.com/Wan-Video/Wan2.1)
  source package on `PYTHONPATH`; hybrid provenance uses revision `9737cba`.
- Hunyuan requires the official
  [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
  source and dependencies. Hybrid provenance uses revision `60783e7`.
  Put that checkout beside this repository, or set `HUNYUAN_REPO` to its path.

User-supplied `HF_ENDPOINT` settings are respected. No mirror is selected by
the samplers. Use your normal Hugging Face authentication for gated weights;
never commit tokens or model files.

## Optional evaluation and visualization

Install `pytest`, pandas, PyYAML, matplotlib, Pillow, OpenCV and `lpips`, in
addition to the relevant model dependencies. The full tests also import model
helpers; lightweight CLI-contract tests do not load pretrained weights.
The tested base environment uses pytest 9.1.1, lpips 0.1.4 and OpenCV 4.10.0.84.
LPIPS weights may require an initial download.

Run from the repository root:

```bash
python -m pytest -q
```

Generated media and reports belong under `outputs/` or `runs/` and are ignored
by Git. See [evaluation](evaluation/README.md) and
[visualization](assets/visualization/ccmr/README.md) for protocols.
