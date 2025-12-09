# InvarDiff for FLUX

Fine-grained caching acceleration for FLUX.1-dev.

## Installation

```bash
pip install torch torchvision
pip install diffusers transformers accelerate safetensors
pip install Pillow numpy tqdm
```

## Quick Start

### Basic Usage

```bash
python sample_InvarDiff_flux.py
```

The script will generate images using pre-calibrated cache books. Generated images will be saved in `./images/` directory.

### Speed Modes Configuration

Pre-configured speed modes with different quality-speed trade-offs:

| Mode | Speedup | nonskip_rate | step_thres | attn_thres | ff_thres | context_ff_thres | Single_attn_thres | Single_mlp_thres |
|------|-------|--------------|------------|------------|----------|------------------|-------------------|------------------|
| Fast | 3.3× | 0.1 | 0.7 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |
| Medium-1 | 2.9× | 0.1 | 0.7 | 0.68 | 0.0 | 0.0 | 0.68 | 0.0 |
| Medium-2 | 2.6× | 0.15 | 0.7 | 0.68 | 0.0 | 0.0 | 0.7 | 0.0 |
| Slow | 2.5× | 0.22 | 0.72 | 0.68 | 0.66 | 0.0 | 0.68 | 0.62 |

### Comparison

![FLUX_compare](../assets/FLUX_compare.jpg)
