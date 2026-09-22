#!/usr/bin/env python3
"""Reference-based LPIPS evaluation for aligned images or videos.

This tool is intentionally pairwise: the reference and candidate must come
from the same model, prompt, seed, scheduler, geometry, and sampling steps.
For videos it supplements frame-wise LPIPS with two inexpensive temporal-drift
signals; it is not a replacement for a full video-quality benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_media(path: Path):
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return np.expand_dims(frame, 0), {"kind": "image", "fps": None}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open media file: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No decodable video frames: {path}")
    return np.stack(frames), {"kind": "video", "fps": fps}


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    # PIL-backed arrays can be read-only; copy before torch conversion so no
    # in-place normalization can alias immutable storage.
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
    return tensor.div_(127.5).sub_(1.0)


def _batched_lpips(model, left, right, device, batch_size):
    values = []
    for start in range(0, left.shape[0], batch_size):
        stop = min(start + batch_size, left.shape[0])
        with torch.inference_mode():
            score = model(
                left[start:stop].to(device), right[start:stop].to(device)
            )
        values.extend(score.flatten().float().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray):
    if values.size == 0:
        return None
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _frame_ssim_rgb(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Canonical 11x11 Gaussian-window SSIM, averaged over RGB channels."""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values = []
    for x, y in zip(left.astype(np.float64), right.astype(np.float64)):
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
            sigma_x + sigma_y + c2
        )
        score_map = numerator / np.maximum(denominator, 1e-12)
        # Ignore the five-pixel Gaussian-padding boundary when possible.
        if score_map.shape[0] > 10 and score_map.shape[1] > 10:
            score_map = score_map[5:-5, 5:-5]
        values.append(float(score_map.mean()))
    return np.asarray(values, dtype=np.float64)


def _split_horizontal_grid(frames, count, padding):
    if frames.shape[0] != 1:
        raise ValueError("Image-grid splitting applies only to still images")
    height, width = frames.shape[1:3]
    content_width = width - padding * (count + 1)
    content_height = height - 2 * padding
    if content_width <= 0 or content_height <= 0 or content_width % count:
        raise ValueError(
            f"Invalid horizontal grid geometry: {width}x{height}, "
            f"count={count}, padding={padding}"
        )
    cell_width = content_width // count
    return np.stack([
        frames[0, padding:padding + content_height,
               padding + index * (cell_width + padding):
               padding + index * (cell_width + padding) + cell_width]
        for index in range(count)
    ])


def evaluate(
    reference,
    candidate,
    device,
    net,
    batch_size,
    image_grid_count=1,
    image_grid_padding=0,
    metric=None,
):
    ref_frames, ref_meta = _read_media(reference)
    cand_frames, cand_meta = _read_media(candidate)
    if image_grid_count > 1:
        if ref_meta["kind"] != "image" or cand_meta["kind"] != "image":
            raise ValueError("--image-grid-count applies only to images")
        ref_frames = _split_horizontal_grid(
            ref_frames, image_grid_count, image_grid_padding
        )
        cand_frames = _split_horizontal_grid(
            cand_frames, image_grid_count, image_grid_padding
        )
    if ref_frames.shape != cand_frames.shape:
        raise ValueError(
            f"Media shapes differ: {ref_frames.shape} vs {cand_frames.shape}"
        )
    if ref_meta["kind"] != cand_meta["kind"]:
        raise ValueError("Reference and candidate media kinds differ")
    if ref_meta["kind"] == "video" and not math.isclose(
        ref_meta["fps"], cand_meta["fps"], rel_tol=0.0, abs_tol=1e-3
    ):
        raise ValueError(
            f"Video FPS differs: {ref_meta['fps']} vs {cand_meta['fps']}"
        )

    ref = _to_tensor(ref_frames)
    cand = _to_tensor(cand_frames)
    if metric is None:
        metric = lpips.LPIPS(net=net, verbose=False).eval().to(device)
    frame_lpips = _batched_lpips(metric, ref, cand, device, batch_size)

    pixel_delta = (ref - cand).float()
    frame_l1 = pixel_delta.abs().mean(dim=(1, 2, 3)).numpy() / 2.0
    frame_mse = pixel_delta.square().mean(dim=(1, 2, 3)).numpy() / 4.0
    # Clamp exact matches at 120 dB so the JSON remains standards-compliant.
    frame_psnr = -10.0 * np.log10(np.maximum(frame_mse, 1e-12))
    frame_ssim = _frame_ssim_rgb(ref_frames, cand_frames)

    temporal = None
    if ref.shape[0] > 1:
        ref_motion = _batched_lpips(
            metric, ref[:-1], ref[1:], device, batch_size
        )
        cand_motion = _batched_lpips(
            metric, cand[:-1], cand[1:], device, batch_size
        )
        motion_delta = np.abs(ref_motion - cand_motion)
        delta_l1 = (
            ((cand[1:] - cand[:-1]) - (ref[1:] - ref[:-1]))
            .abs().mean(dim=(1, 2, 3)).numpy() / 2.0
        )
        temporal = {
            "adjacent_lpips_reference": _summary(ref_motion),
            "adjacent_lpips_candidate": _summary(cand_motion),
            "adjacent_lpips_abs_delta": _summary(motion_delta),
            "temporal_delta_l1": _summary(delta_l1),
            "per_transition_lpips_abs_delta": motion_delta.tolist(),
            "per_transition_delta_l1": delta_l1.tolist(),
        }

    return {
        "protocol_version": 2,
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "reference_sha256": _sha256(reference),
        "candidate_sha256": _sha256(candidate),
        "media": {
            "kind": "image_grid" if image_grid_count > 1 else ref_meta["kind"],
            "frames": int(ref_frames.shape[0]),
            "height": int(ref_frames.shape[1]),
            "width": int(ref_frames.shape[2]),
            "fps": ref_meta["fps"],
        },
        "lpips_backbone": net,
        "frame_lpips": _summary(frame_lpips),
        "frame_l1": _summary(frame_l1),
        "frame_psnr_db": _summary(frame_psnr),
        "frame_ssim_rgb": _summary(frame_ssim),
        "temporal": temporal,
        "per_frame_lpips": frame_lpips.tolist(),
        "per_frame_l1": frame_l1.tolist(),
        "per_frame_psnr_db": frame_psnr.tolist(),
        "per_frame_ssim_rgb": frame_ssim.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-grid-count", type=int, default=1)
    parser.add_argument("--image-grid-padding", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.image_grid_count <= 0:
        parser.error("--batch-size and --image-grid-count must be positive")
    if args.image_grid_padding < 0:
        parser.error("--image-grid-padding must be non-negative")

    result = evaluate(
        args.reference,
        args.candidate,
        args.device,
        args.net,
        args.batch_size,
        args.image_grid_count,
        args.image_grid_padding,
    )
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
