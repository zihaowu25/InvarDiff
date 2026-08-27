"""Standalone HunyuanVideo-1.5 SeaCache + Finegrained Cache sampler.

The HunyuanVideo-1.5 pipeline integration is based on Tencent-Hunyuan commit
60783e7. SeaCache policy details are ported from jiwoogit/SeaCache commit
8dcf490. No executable sampling module is imported.
"""

import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import copy
import datetime
import gc
import json
import logging
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger as _logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _logger = logging.getLogger("finegrained.hunyuan")
    loguru_compat = types.ModuleType("loguru")
    loguru_compat.logger = _logger
    sys.modules["loguru"] = loguru_compat

OFFICIAL_ROOT = Path(__file__).resolve().parents[3] / "HunyuanVideo-1.5"
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

import einops
import imageio
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from hyvideo.commons import PIPELINE_CONFIGS
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.models.transformers.modules.attention import parallel_attention
from hyvideo.models.transformers.modules.mlp_layers import LinearWarpforSingle
from hyvideo.models.transformers.modules.modulate_layers import apply_gate, modulate
from hyvideo.models.transformers.modules.posemb_layers import apply_rotary_emb
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.utils.communications import all_gather


CACHE_SCOPE = "hybrid"
POLICY_VARIANT = "hybrid_seacache"
STEP_POLICY = "seacache"
RATE_METHOD = "three_point_l1"
CACHE_BOOK_VERSION = 2
SOURCE_COMMIT = "8dcf490"
HUNYUAN_COMMIT = "60783e7"
SEACACHE_POWER_EXP = 3.0
SEACACHE_DIMS = (-2, -3, -4)
SEACACHE_NORM_MODE = "mean"
RATE_CHUNK_SIZE = 1_048_576
MODULES = (
    "double.img_attn",
    "double.txt_attn",
    "double.img_mlp",
    "double.txt_mlp",
    "single.attn",
    "single.mlp",
)


def str_to_bool(value):
    if value is None or isinstance(value, bool):
        return True if value is None else value
    value = value.lower().strip()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_log(message: str, level: str = "INFO"):
    if rank0():
        method = getattr(_logger, level.lower(), _logger.info)
        method(message)


def _broadcast_object(value):
    if not dist.is_initialized():
        return value
    values = [value if rank0() else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        state = get_parallel_state()
        group = state.sp_group if state.sp_enabled else None
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def _history_tensor(feature: torch.Tensor, storage: str) -> torch.Tensor:
    feature = feature.detach()
    if storage == "gpu" or feature.device.type == "cpu":
        return feature
    try:
        result = torch.empty_like(feature, device="cpu", pin_memory=True)
    except RuntimeError:
        result = torch.empty_like(feature, device="cpu")
    result.copy_(feature, non_blocking=False)
    return result


def compute_l1_distance(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    chunk_size: int = RATE_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ||x_end-x_start+1e-8||_1 with chunked FP32 GPU sums."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    start_flat = x_start.detach().reshape(-1)
    end_flat = x_end.detach().reshape(-1)
    if start_flat.numel() != end_flat.numel():
        raise ValueError(
            f"Rate feature sizes differ: {start_flat.numel()} and {end_flat.numel()}"
        )
    if end_flat.device.type == "cuda":
        device = end_flat.device
    elif start_flat.device.type == "cuda":
        device = start_flat.device
    else:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else end_flat.device
        )
    total = torch.zeros((), device=device, dtype=torch.float32)
    for begin in range(0, start_flat.numel(), chunk_size):
        end = min(begin + chunk_size, start_flat.numel())
        lhs = start_flat[begin:end].to(device=device, non_blocking=True)
        rhs = end_flat[begin:end].to(device=device, non_blocking=True)
        total += (rhs.float() - lhs.float() + 1e-8).abs().sum()
    return _reduce_sum(total)


def compute_rate(
    current_norm: torch.Tensor,
    previous_norm: torch.Tensor,
) -> torch.Tensor:
    return current_norm / previous_norm.clamp_min(1e-8)


def _conditional_feature(feature: torch.Tensor, do_cfg: bool) -> torch.Tensor:
    if not do_cfg:
        return feature
    if feature.shape[0] % 2:
        raise ValueError(
            "CFG feature batch must contain equal unconditional/conditional halves"
        )
    return feature.chunk(2, dim=0)[1]


class FeatureChangeAnalyzer:
    """Three-point analyzer for six fine-grained module trajectories."""

    def __init__(
        self,
        num_steps: int,
        depths: Dict[str, int],
        active_modules: Tuple[str, ...],
        feature_device: str,
        do_cfg: bool,
        correction_mode: bool = False,
        step_mask: Optional[List[bool]] = None,
        module_books: Optional[Dict[str, List[List[bool]]]] = None,
    ):
        self.num_steps = num_steps
        self.depths = depths
        self.active_modules = active_modules
        self.feature_device = feature_device
        self.do_cfg = do_cfg
        self.correction_mode = correction_mode
        self.step_mask = step_mask
        self.module_books = module_books
        if correction_mode and (step_mask is None or module_books is None):
            raise ValueError("Correction calibration requires provisional books")
        self.module_scores = {
            name: torch.ones((num_steps, depths[name]), dtype=torch.float32)
            for name in MODULES
        }
        self._active = {
            name: [False] * depths[name] for name in active_modules
        }
        self._current = {name: [None] * depths[name] for name in active_modules}
        self._norm = {name: [None] * depths[name] for name in active_modules}

    @staticmethod
    def _refresh_state(is_cached: bool, active: bool) -> Tuple[bool, bool]:
        if not is_cached:
            return True, False
        if not active:
            return True, True
        return False, True

    def _module_cached(self, name: str, score_idx: int, layer_idx: int) -> bool:
        if self.step_mask[score_idx]:
            return True
        if name in ("double.img_attn", "double.txt_attn"):
            return bool(
                self.module_books["double.img_attn"][score_idx][layer_idx]
                and self.module_books["double.txt_attn"][score_idx][layer_idx]
            )
        return bool(self.module_books[name][score_idx][layer_idx])

    def _rotate_module(
        self, name: str, layer_idx: int, feature: torch.Tensor,
        next_norm: Optional[torch.Tensor] = None,
    ):
        current = self._current[name][layer_idx]
        if next_norm is None:
            next_norm = compute_l1_distance(current, feature)
        self._current[name][layer_idx] = _history_tensor(
            feature, self.feature_device
        )
        self._norm[name][layer_idx] = next_norm

    def update_module(
        self, step_idx: int, name: str, layer_idx: int, feature: torch.Tensor
    ):
        if name not in self.active_modules:
            return
        feature = _conditional_feature(feature.detach(), self.do_cfg)
        current = self._current[name][layer_idx]
        if step_idx == 0 or current is None:
            self._current[name][layer_idx] = _history_tensor(
                feature, self.feature_device
            )
            return
        previous_norm = self._norm[name][layer_idx]
        if step_idx == 1 or previous_norm is None:
            self._current[name][layer_idx] = _history_tensor(
                feature, self.feature_device
            )
            self._norm[name][layer_idx] = compute_l1_distance(current, feature)
            return
        score_idx = step_idx - 1
        current_norm = compute_l1_distance(current, feature)
        self.module_scores[name][score_idx, layer_idx] = float(
            compute_rate(
                current_norm,
                self._norm[name][layer_idx],
            ).item()
        )
        if not self.correction_mode:
            self._rotate_module(name, layer_idx, feature, current_norm)
            return
        refresh, active = self._refresh_state(
            self._module_cached(name, score_idx, layer_idx),
            self._active[name][layer_idx],
        )
        self._active[name][layer_idx] = active
        if refresh:
            self._rotate_module(name, layer_idx, feature, current_norm)

    def release(self):
        self._current.clear()
        self._norm.clear()


def _relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    current_flat = current.detach().reshape(-1)
    previous_flat = previous.detach().reshape(-1)
    if current_flat.numel() != previous_flat.numel():
        raise ValueError("Relative-L1 feature sizes differ")
    device = current.device
    numerator = torch.zeros((), device=device, dtype=torch.float32)
    denominator = torch.zeros((), device=device, dtype=torch.float32)
    for begin in range(0, current_flat.numel(), RATE_CHUNK_SIZE):
        end = min(begin + RATE_CHUNK_SIZE, current_flat.numel())
        lhs = current_flat[begin:end].to(device, non_blocking=True).float()
        rhs = previous_flat[begin:end].to(device, non_blocking=True).float()
        numerator += (lhs - rhs).abs().sum()
        denominator += rhs.abs().sum()
    pair = _reduce_sum(torch.stack((numerator, denominator)))
    return float((pair[0] / pair[1].clamp_min(1e-16)).item())

def _ab_from_scheduler(scheduler, idx: int) -> Tuple[float, float]:
    def clamp01(value):
        return max(1e-6, min(1.0 - 1e-6, float(value)))

    if hasattr(scheduler, "sigmas"):
        sigma = float(scheduler.sigmas[idx])
    else:
        count = getattr(scheduler, "num_inference_steps", idx + 1)
        sigma = 1.0 - (idx + 1) / float(count)
    sigma = clamp01(sigma)
    return 1.0 - sigma, sigma


def _apply_sea_from_ab(
    x: torch.Tensor, a: float, b: float, power_exp: float = 3.0,
    dims=SEACACHE_DIMS, eps: float = 1e-16,
) -> torch.Tensor:
    original_dtype = x.dtype
    x32 = x.contiguous().float()
    spectrum = torch.fft.fftn(x32, dim=dims)
    filt = None
    for axis in dims:
        freq = torch.fft.fftfreq(
            x32.shape[axis], device=x32.device, dtype=torch.float32
        )
        radius = freq.abs()
        signal = 1.0 / (radius.pow(power_exp) + eps)
        gain = (a * signal) / (a * a * signal + b * b + eps)
        shape = [1] * x32.ndim
        shape[axis] = gain.shape[0]
        gain = gain.reshape(shape)
        filt = gain if filt is None else filt * gain
    mean = filt.mean()
    if torch.isfinite(mean) and mean > 0:
        filt = filt / mean
    return torch.fft.ifftn(spectrum * filt, dim=dims).real.to(original_dtype)


class SeaCachePolicy:
    """Official HunyuanVideo SeaCache dynamic accumulator."""

    def __init__(
        self, num_steps, threshold, scheduler, enabled=True,
        feature_device="gpu",
    ):
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.scheduler = scheduler
        self.enabled = bool(enabled)
        self.feature_device = feature_device
        self.reset()

    def reset(self):
        self.previous_modulated_input = None
        self.accumulated_distance = 0.0

    def decide(self, step_idx, modulated_input, thw=None):
        if thw is None:
            raise ValueError("SeaCache requires the current (T,H,W) token grid")
        current = modulated_input.reshape(
            modulated_input.shape[0], *thw, modulated_input.shape[-1]
        )
        a, b = _ab_from_scheduler(self.scheduler, step_idx)
        current = _apply_sea_from_ab(
            current, a, b, power_exp=SEACACHE_POWER_EXP,
            dims=SEACACHE_DIMS,
        ).reshape_as(modulated_input)
        first_or_last = step_idx == 0 or step_idx == self.num_steps - 1
        if self.previous_modulated_input is None or first_or_last:
            should_skip = False
            self.accumulated_distance = 0.0
        else:
            self.accumulated_distance += _relative_l1(
                current, self.previous_modulated_input
            )
            should_skip = self.accumulated_distance < self.threshold
            if not should_skip:
                self.accumulated_distance = 0.0
        self.previous_modulated_input = _history_tensor(
            current, self.feature_device
        )
        return bool(self.enabled and should_skip)

class RuntimeCache:
    def __init__(self, mode: str, reserve_gib: float):
        self.mode = mode
        self.reserve_bytes = int(reserve_gib * 1024**3)
        self.values: Dict[Tuple[str, int], torch.Tensor] = {}

    def _use_gpu(self, tensor: torch.Tensor) -> bool:
        if self.mode == "gpu":
            return True
        if self.mode == "cpu" or tensor.device.type != "cuda":
            return False
        free_bytes, _ = torch.cuda.mem_get_info(tensor.device)
        return (
            free_bytes - tensor.numel() * tensor.element_size()
            >= self.reserve_bytes
        )

    def put(
        self, key: Tuple[str, int], tensor: torch.Tensor, has_future: bool
    ):
        if not has_future:
            self.values.pop(key, None)
            return
        tensor = tensor.detach()
        if self._use_gpu(tensor):
            self.values[key] = tensor
        else:
            try:
                stored = torch.empty_like(tensor, device="cpu", pin_memory=True)
            except RuntimeError:
                stored = torch.empty_like(tensor, device="cpu")
            stored.copy_(tensor, non_blocking=False)
            self.values[key] = stored

    def has(self, key: Tuple[str, int]) -> bool:
        return key in self.values

    def get(
        self, key: Tuple[str, int], device: torch.device, last_use: bool
    ) -> torch.Tensor:
        stored = self.values[key]
        result = (
            stored
            if stored.device == device
            else stored.to(device, non_blocking=True)
        )
        if last_use:
            del self.values[key]
        return result

    def clear(self):
        self.values.clear()


def _cache_key(name: str, layer_idx: int, slot: int):
    return name, layer_idx, slot


def _cache_has(model, name: str, layer_idx: int) -> bool:
    return all(
        model.finegrained_runtime_cache.has(_cache_key(name, layer_idx, slot))
        for slot in model.finegrained_active_slots
    )


def _cache_get(model, name, layer_idx, device, last_use):
    values = [
        model.finegrained_runtime_cache.get(
            _cache_key(name, layer_idx, slot), device, last_use
        )
        for slot in model.finegrained_active_slots
    ]
    return values[0] if len(values) == 1 else torch.cat(values, dim=0)


def _cache_put(model, name, layer_idx, tensor, has_future):
    slots = model.finegrained_active_slots
    chunks = tensor.chunk(len(slots), dim=0)
    for slot, chunk in zip(slots, chunks):
        model.finegrained_runtime_cache.put(
            _cache_key(name, layer_idx, slot), chunk, has_future
        )


def _thresholds(args) -> Dict[str, float]:
    return {
        "double.img_attn": args.double_img_attn_thres,
        "double.txt_attn": args.double_txt_attn_thres,
        "double.img_mlp": args.double_img_mlp_thres,
        "double.txt_mlp": args.double_txt_mlp_thres,
        "single.attn": args.single_attn_thres,
        "single.mlp": args.single_mlp_thres,
    }


def _quantile_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    valid = values[~torch.isnan(values)]
    if valid.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return values < torch.quantile(valid, q)


def _books_from_scores(
    module_scores: Dict[str, torch.Tensor],
    nonskip_rate: float,
    thresholds: Dict[str, float],
):
    num_steps = next(iter(module_scores.values())).shape[0]
    protected = max(1, int(nonskip_rate * num_steps))
    module_books = {}
    for name in MODULES:
        scores = module_scores[name]
        book = torch.zeros_like(scores, dtype=torch.bool)
        if thresholds[name] > 0:
            book[1:-1] = _quantile_mask(
                scores[1:-1].reshape(-1), thresholds[name]
            ).reshape(num_steps - 2, scores.shape[1])
        book[:protected] = False
        book[-1] = False
        module_books[name] = book
    return {
        name: book.tolist() for name, book in module_books.items()
    }


def _future_hit(model, name: str, layer_idx: int, step_idx: int) -> bool:
    books = model.finegrained_module_books
    if name in ("double.img_attn", "double.txt_attn"):
        return any(
            books["double.img_attn"][future][layer_idx]
            and books["double.txt_attn"][future][layer_idx]
            for future in range(step_idx + 1, model.finegrained_num_steps)
        )
    return any(
        books[name][future][layer_idx]
        for future in range(step_idx + 1, model.finegrained_num_steps)
    )


def _last_hit(model, name: str, layer_idx: int, step_idx: int) -> bool:
    return not _future_hit(model, name, layer_idx, step_idx)


def _plan(model, name: str, layer_idx: int) -> bool:
    if (
        not model.finegrained_use
        or model.finegrained_calibrating
        or model.finegrained_force_refresh
    ):
        return False
    return bool(
        model.finegrained_module_books[name][model.finegrained_step_idx][layer_idx]
    )


def _analyse(block, name: str, feature: torch.Tensor):
    model = block.finegrained_owner
    if model.finegrained_analyzer is not None:
        model.finegrained_analyzer.update_module(
            model.finegrained_step_idx,
            name,
            block.finegrained_layer_idx,
            feature,
        )


def finegrained_double_block_forward(
    self,
    img,
    txt,
    vec,
    freqs_cis=None,
    text_mask=None,
    attn_param=None,
    is_flash=False,
    block_idx=None,
):
    model = self.finegrained_owner
    idx = self.finegrained_layer_idx
    step_idx = model.finegrained_step_idx
    (
        img_shift1,
        img_scale1,
        img_gate1,
        img_shift2,
        img_scale2,
        img_gate2,
    ) = self.img_mod(vec).chunk(6, dim=-1)
    (
        txt_shift1,
        txt_scale1,
        txt_gate1,
        txt_shift2,
        txt_scale2,
        txt_gate2,
    ) = self.txt_mod(vec).chunk(6, dim=-1)

    joint_hit = (
        _plan(model, "double.img_attn", idx)
        and _plan(model, "double.txt_attn", idx)
        and _cache_has(model, "double.img_attn", idx)
        and _cache_has(model, "double.txt_attn", idx)
    )
    if joint_hit:
        model.finegrained_layer_hits += 2 * len(
            getattr(model, "finegrained_active_slots", (0,))
        )
        last = _last_hit(model, "double.img_attn", idx, step_idx)
        img_attn_out = _cache_get(
            model, "double.img_attn", idx, img.device, last
        )
        txt_attn_out = _cache_get(
            model, "double.txt_attn", idx, txt.device, last
        )
    else:
        img_modulated = modulate(
            self.img_norm1(img), shift=img_shift1, scale=img_scale1
        )
        txt_modulated = modulate(
            self.txt_norm1(txt), shift=txt_shift1, scale=txt_scale1
        )
        img_q = rearrange(
            self.img_attn_q(img_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        img_k = rearrange(
            self.img_attn_k(img_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        img_v = rearrange(
            self.img_attn_v(img_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        txt_q = rearrange(
            self.txt_attn_q(txt_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        txt_k = rearrange(
            self.txt_attn_k(txt_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        txt_v = rearrange(
            self.txt_attn_v(txt_modulated),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(
                img_q, img_k, freqs_cis, head_first=False
            )
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode="flash" if is_flash else self.attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_raw = attn[:, : img_q.shape[1]].contiguous()
        txt_raw = attn[:, img_q.shape[1] :].contiguous()
        img_attn_out = self.img_attn_proj(img_raw)
        txt_attn_out = self.txt_attn_proj(txt_raw)
        if model.finegrained_use and not model.finegrained_calibrating:
            future = _future_hit(
                model, "double.img_attn", idx, step_idx
            )
            _cache_put(model, "double.img_attn", idx, img_attn_out, future)
            _cache_put(model, "double.txt_attn", idx, txt_attn_out, future)
    _analyse(self, "double.img_attn", img_attn_out)
    _analyse(self, "double.txt_attn", txt_attn_out)

    img = img + apply_gate(img_attn_out, gate=img_gate1)
    if _plan(model, "double.img_mlp", idx) and _cache_has(
        model, "double.img_mlp", idx
    ):
        model.finegrained_layer_hits += len(model.finegrained_active_slots)
        img_mlp_out = _cache_get(
            model, "double.img_mlp", idx, img.device,
            _last_hit(model, "double.img_mlp", idx, step_idx),
        )
    else:
        img_mlp_out = self.img_mlp(
            modulate(self.img_norm2(img), shift=img_shift2, scale=img_scale2)
        )
        if model.finegrained_use and not model.finegrained_calibrating:
            _cache_put(
                model, "double.img_mlp", idx, img_mlp_out,
                _future_hit(model, "double.img_mlp", idx, step_idx),
            )
    _analyse(self, "double.img_mlp", img_mlp_out)
    img = img + apply_gate(img_mlp_out, gate=img_gate2)

    txt = txt + apply_gate(txt_attn_out, gate=txt_gate1)
    if _plan(model, "double.txt_mlp", idx) and _cache_has(
        model, "double.txt_mlp", idx
    ):
        model.finegrained_layer_hits += len(model.finegrained_active_slots)
        txt_mlp_out = _cache_get(
            model, "double.txt_mlp", idx, txt.device,
            _last_hit(model, "double.txt_mlp", idx, step_idx),
        )
    else:
        txt_mlp_out = self.txt_mlp(
            modulate(self.txt_norm2(txt), shift=txt_shift2, scale=txt_scale2)
        )
        if model.finegrained_use and not model.finegrained_calibrating:
            _cache_put(
                model, "double.txt_mlp", idx, txt_mlp_out,
                _future_hit(model, "double.txt_mlp", idx, step_idx),
            )
    _analyse(self, "double.txt_mlp", txt_mlp_out)
    txt = txt + apply_gate(txt_mlp_out, gate=txt_gate2)
    return img, txt


def _single_linear_parts(block, attn=None, mlp=None):
    fc = block.linear2.fc
    if type(fc) is not nn.Linear:
        raise RuntimeError(
            "single.linear2.fc must remain torch.nn.Linear for fine-grained "
            "Finegrained Cache; disable single-block FP8/LoRA wrapping."
        )
    hidden = block.hidden_size
    attn_out = (
        F.linear(attn, fc.weight[:, :hidden], None)
        if attn is not None
        else None
    )
    mlp_out = (
        F.linear(mlp, fc.weight[:, hidden:], None)
        if mlp is not None
        else None
    )
    return attn_out, mlp_out, fc.bias


def finegrained_single_block_forward(
    self,
    x,
    vec,
    txt_len,
    freqs_cis=None,
    text_mask=None,
    attn_param=None,
    is_flash=False,
):
    model = self.finegrained_owner
    idx = self.finegrained_layer_idx
    step_idx = model.finegrained_step_idx
    attn_hit = _plan(model, "single.attn", idx) and _cache_has(
        model, "single.attn", idx
    )
    mlp_hit = _plan(model, "single.mlp", idx) and _cache_has(
        model, "single.mlp", idx
    )
    shift, scale, gate = self.modulation(vec).chunk(3, dim=-1)
    x_mod = (
        None
        if attn_hit and mlp_hit
        else modulate(self.pre_norm(x), shift=shift, scale=scale)
    )

    if attn_hit:
        model.finegrained_layer_hits += len(
            getattr(model, "finegrained_active_slots", (0,))
        )
        attn_out = _cache_get(
            model, "single.attn", idx, x.device,
            _last_hit(model, "single.attn", idx, step_idx),
        )
    else:
        q = rearrange(
            self.linear1_q(x_mod),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        k = rearrange(
            self.linear1_k(x_mod),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        v = rearrange(
            self.linear1_v(x_mod),
            "B L (H D) -> B L H D",
            H=self.heads_num,
        )
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_v, txt_v = v[:, :-txt_len], v[:, -txt_len:]
        img_q, img_k = apply_rotary_emb(
            img_q, img_k, freqs_cis, head_first=False
        )
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode="flash" if is_flash else self.attn_mode,
            attn_param=attn_param,
        )
        attn_out, _, _ = _single_linear_parts(self, attn=attn)
        if model.finegrained_use and not model.finegrained_calibrating:
            _cache_put(
                model, "single.attn", idx, attn_out,
                _future_hit(model, "single.attn", idx, step_idx),
            )

    if mlp_hit:
        model.finegrained_layer_hits += len(
            getattr(model, "finegrained_active_slots", (0,))
        )
        mlp_out = _cache_get(
            model, "single.mlp", idx, x.device,
            _last_hit(model, "single.mlp", idx, step_idx),
        )
    else:
        mlp = self.mlp_act(self.linear1_mlp(x_mod))
        _, mlp_out, _ = _single_linear_parts(self, mlp=mlp)
        if model.finegrained_use and not model.finegrained_calibrating:
            _cache_put(
                model, "single.mlp", idx, mlp_out,
                _future_hit(model, "single.mlp", idx, step_idx),
            )
    _analyse(self, "single.attn", attn_out)
    _analyse(self, "single.mlp", mlp_out)
    output = attn_out + mlp_out
    if self.linear2.fc.bias is not None:
        output = output + self.linear2.fc.bias
    return x + apply_gate(output, gate=gate)


def _reset_runtime(model):
    model.finegrained_step_idx = 0
    model.finegrained_runtime_cache.clear()
    model.finegrained_step_residual = None
    model.finegrained_previous_step_cached = False
    model.finegrained_force_refresh = False
    if model.finegrained_step_policy is not None:
        model.finegrained_step_policy.reset()


def finegrained_transformer_forward(
    self,
    hidden_states,
    timestep,
    text_states,
    text_states_2,
    encoder_attention_mask,
    timestep_r=None,
    vision_states=None,
    output_features=False,
    output_features_stride=8,
    attention_kwargs=None,
    freqs_cos=None,
    freqs_sin=None,
    return_dict=False,
    guidance=None,
    mask_type="t2v",
    extra_kwargs=None,
):
    if torch.cuda.is_available():
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_start.record()
    else:
        timing_start = time.perf_counter()
        timing_end = None
    if guidance is None:
        guidance = torch.tensor(
            [6016.0], device=hidden_states.device, dtype=torch.bfloat16
        )
    img = hidden_states
    text_mask = encoder_attention_mask
    txt = text_states
    t = timestep
    bs, _, ot, oh, ow = img.shape
    tt = ot // self.patch_size[0]
    th = oh // self.patch_size[1]
    tw = ow // self.patch_size[2]
    self.attn_param["thw"] = [tt, th, tw]
    if freqs_cos is None and freqs_sin is None:
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
    img = self.img_in(img)
    parallel_dims = get_parallel_state()
    if parallel_dims.sp_enabled:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
        if img.shape[1] % sp_size != 0:
            n_token = img.shape[1]
            assert n_token > (n_token // sp_size + 1) * (sp_size - 1)
        img = torch.chunk(img, sp_size, dim=1)[sp_rank]
        freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
        freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

    vec = self.time_in(t)
    if text_states_2 is not None:
        vec = vec + self.vector_in(text_states_2)
    if self.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Missing guidance strength for guidance-distilled model"
            )
        vec = vec + self.guidance_in(guidance)
    if timestep_r is not None:
        vec = vec + self.time_r_in(timestep_r)

    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(
            txt, t, text_mask if self.use_attention_mask else None
        )
    else:
        raise NotImplementedError(
            f"Unsupported text_projection: {self.text_projection}"
        )
    if self.cond_type_embedding is not None:
        txt = txt + self.cond_type_embedding(
            torch.zeros_like(
                txt[:, :, 0], device=text_mask.device, dtype=torch.long
            )
        )
    extra_kwargs = extra_kwargs or {}
    if self.glyph_byT5_v2:
        byt5_txt = self.byt5_in(extra_kwargs["byt5_text_states"])
        byt5_mask = extra_kwargs["byt5_text_mask"]
        if self.cond_type_embedding is not None:
            byt5_txt = byt5_txt + self.cond_type_embedding(
                torch.ones_like(
                    byt5_txt[:, :, 0],
                    device=byt5_txt.device,
                    dtype=torch.long,
                )
            )
        txt, text_mask = self.reorder_txt_token(
            byt5_txt, txt, byt5_mask, text_mask, zero_feat=True
        )
    if self.vision_in is not None and vision_states is not None:
        extra_txt = self.vision_in(vision_states)
        if mask_type == "t2v" and torch.all(vision_states == 0):
            extra_mask = torch.zeros(
                (bs, extra_txt.shape[1]),
                dtype=text_mask.dtype,
                device=text_mask.device,
            )
            extra_txt = extra_txt * 0.0
        else:
            extra_mask = torch.ones(
                (bs, extra_txt.shape[1]),
                dtype=text_mask.dtype,
                device=text_mask.device,
            )
        if self.cond_type_embedding is not None:
            extra_txt = extra_txt + self.cond_type_embedding(
                2
                * torch.ones_like(
                    extra_txt[:, :, 0],
                    dtype=torch.long,
                    device=extra_txt.device,
                )
            )
        txt, text_mask = self.reorder_txt_token(
            extra_txt, txt, extra_mask, text_mask
        )

    freqs_cis = (
        (freqs_cos, freqs_sin) if freqs_cos is not None else None
    )
    step_idx = self.finegrained_step_idx
    policy = self.finegrained_step_policy
    would_skip = False
    if policy is not None:
        first_block = self.double_blocks[0]
        img_shift1, img_scale1, _, _, _, _ = first_block.img_mod(vec).chunk(
            6, dim=-1
        )
        modulated_input = modulate(
            first_block.img_norm1(img), shift=img_shift1, scale=img_scale1
        )
        would_skip = policy.decide(step_idx, modulated_input, (tt, th, tw))
    if self.finegrained_observer_mask is not None:
        self.finegrained_observer_mask[step_idx] = bool(would_skip)

    step_slots = self.finegrained_active_slots
    step_hit = bool(
        self.finegrained_execute_step_cache
        and would_skip
        and all(
            self.finegrained_runtime_cache.has(("step", -1, slot))
            for slot in step_slots
        )
        and not output_features
    )
    features_list = [] if output_features else None
    img_before_blocks = img
    if step_hit:
        residuals = [
            self.finegrained_runtime_cache.get(
                ("step", -1, slot), img.device, last_use=False
            )
            for slot in step_slots
        ]
        residual = residuals[0] if len(residuals) == 1 else torch.cat(
            residuals, dim=0
        )
        img = img + residual
        self.finegrained_step_hits += 1
    else:
        self.finegrained_force_refresh = bool(
            self.finegrained_previous_step_cached
        )
        for index, block in enumerate(self.double_blocks):
            force_full = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or index + 1 == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            img, txt = block(
                img, txt, vec, freqs_cis, text_mask,
                self.attn_param, force_full, index,
            )
        txt_len = txt.shape[1]
        img_len = img.shape[1]
        x = torch.cat((img, txt), dim=1)
        for index, block in enumerate(self.single_blocks):
            force_full = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or index + 1 == len(self.single_blocks)
                )
            )
            self.attn_param["layer-name"] = f"single_block_{index + 1}"
            x = block(
                x, vec, txt_len, (freqs_cos, freqs_sin), text_mask,
                self.attn_param, force_full,
            )
            if output_features and index % output_features_stride == 0:
                features_list.append(x[:, :img_len])
        img = x[:, :img_len]
        if self.finegrained_execute_step_cache:
            residual = (img - img_before_blocks).detach()
            for slot, chunk in zip(
                step_slots, residual.chunk(len(step_slots), dim=0)
            ):
                self.finegrained_runtime_cache.put(
                    ("step", -1, slot), chunk, has_future=True
                )
        self.finegrained_force_refresh = False
        self.finegrained_computed_steps += 1
    self.finegrained_previous_step_cached = step_hit
    img = self.final_layer(img, vec)
    if parallel_dims.sp_enabled:
        img = all_gather(img, dim=1, group=parallel_dims.sp_group)
    img = self.unpatchify(img, tt, th, tw)
    if output_features:
        features_list = torch.stack(features_list, dim=0)
        if parallel_dims.sp_enabled:
            features_list = all_gather(
                features_list, dim=2, group=parallel_dims.sp_group
            )
    else:
        features_list = None
    assert return_dict is False, "return_dict is not supported"
    if timing_end is not None:
        timing_end.record()
        self.finegrained_timing_events.append((timing_start, timing_end))
    else:
        self.finegrained_transformer_seconds += time.perf_counter() - timing_start
    self.finegrained_step_idx += 1
    if self.finegrained_step_idx >= self.finegrained_num_steps:
        _reset_runtime(self)
    return img, features_list


def _patch_model(model):
    if not all(
        type(block.linear2) is LinearWarpforSingle
        and type(block.linear2.fc) is nn.Linear
        for block in model.single_blocks
    ):
        raise RuntimeError(
            "Fine-grained single-stream Cache requires unwrapped "
            "nn.Linear linear2.fc modules"
        )
    model.forward = types.MethodType(finegrained_transformer_forward, model)
    for idx, block in enumerate(model.double_blocks):
        block.finegrained_owner = model
        block.finegrained_layer_idx = idx
        block.forward = types.MethodType(
            finegrained_double_block_forward, block
        )
    for idx, block in enumerate(model.single_blocks):
        block.finegrained_owner = model
        block.finegrained_layer_idx = idx
        block.forward = types.MethodType(
            finegrained_single_block_forward, block
        )


def _empty_module_books(model, num_steps):
    return {
        name: [
            [False]
            * (
                len(model.double_blocks)
                if name.startswith("double")
                else len(model.single_blocks)
            )
            for _ in range(num_steps)
        ]
        for name in MODULES
    }


def _init_runtime(
    model,
    args,
    num_steps,
    do_cfg,
    books=None,
    analyzer=None,
    use=False,
    step_policy=None,
    execute_step_cache=False,
    observer_mask=None,
):
    model.finegrained_num_steps = num_steps
    model.finegrained_step_idx = 0
    model.finegrained_do_cfg = do_cfg
    model.finegrained_use = use
    model.finegrained_calibrating = analyzer is not None
    model.finegrained_analyzer = analyzer
    model.finegrained_module_books = (
        books if books else _empty_module_books(model, num_steps)
    )
    model.finegrained_runtime_cache = RuntimeCache(
        args.runtime_cache_device, args.runtime_cache_gpu_reserve_gib
    )
    model.finegrained_step_policy = step_policy
    model.finegrained_execute_step_cache = bool(execute_step_cache)
    model.finegrained_observer_mask = observer_mask
    model.finegrained_step_residual = None
    model.finegrained_previous_step_cached = False
    model.finegrained_force_refresh = False
    model.finegrained_step_hits = 0
    model.finegrained_computed_steps = 0
    model.finegrained_layer_hits = 0
    model.finegrained_timing_events = []
    model.finegrained_transformer_seconds = 0.0
    model.finegrained_active_slots = (0, 1) if do_cfg else (0,)


def _transformer_seconds(model) -> float:
    if model.finegrained_timing_events:
        return sum(
            start.elapsed_time(end)
            for start, end in model.finegrained_timing_events
        ) / 1000.0
    return float(model.finegrained_transformer_seconds)


def _release_stage(model, analyzer):
    analyzer.release()
    model.finegrained_analyzer = None
    model.finegrained_calibrating = False
    model.finegrained_runtime_cache.clear()
    model.finegrained_step_residual = None
    model.finegrained_step_policy = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect_stage(
    run_once,
    model,
    args,
    num_steps,
    depths,
    do_cfg,
    description,
    books=None,
    step_mask=None,
):
    active = tuple(
        name
        for name, value in _thresholds(args).items()
        if value > 0
    )
    analyzer = FeatureChangeAnalyzer(
        num_steps,
        depths,
        active,
        args.calibration_feature_device,
        do_cfg,
        correction_mode=books is not None,
        step_mask=step_mask,
        module_books=books,
    )
    observed_mask = [False] * num_steps
    _init_runtime(
        model,
        args,
        num_steps,
        do_cfg,
        analyzer=analyzer,
        use=False,
        step_policy=SeaCachePolicy(
            num_steps, args.seacache_thresh, model.finegrained_scheduler,
            enabled=not args.disable_step_cache,
            feature_device=args.calibration_feature_device,
        ),
        execute_step_cache=False,
        observer_mask=observed_mask,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = run_once()
        del result
    except Exception:
        _release_stage(model, analyzer)
        raise
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak = 0.0
    rank0_log(
        f"{description}: {time.perf_counter() - started:.2f}s total, "
        f"{_transformer_seconds(model):.2f}s Transformer, "
        f"peak allocated {peak:.2f} GiB"
    )
    scores = {
        name: value.clone()
        for name, value in analyzer.module_scores.items()
    }
    _release_stage(model, analyzer)
    return scores, observed_mask, peak


def _cache_config(args, version, task, width, height, steps, depths):
    return {
        "transformer_version": version,
        "task": task,
        "width": width,
        "height": height,
        "frames": args.video_length,
        "steps": steps,
        "nonskip": args.nonskip_rate,
        "module_thresholds": _thresholds(args),
        "seed": args.seed,
        "cfg_distilled": args.cfg_distilled,
        "step_distilled": args.enable_step_distill,
        "sparse_attention": args.sparse_attn,
        "scheduler_class": args.resolved_scheduler_class,
        "scheduler_shift": args.resolved_scheduler_shift,
        "guidance_scale": args.resolved_guidance_scale,
        "embedded_guidance_scale": args.resolved_embedded_guidance_scale,
        "double_blocks": depths["double.img_attn"],
        "single_blocks": depths["single.attn"],
        "seacache_thresh": args.seacache_thresh,
        "seacache_power_exp": SEACACHE_POWER_EXP,
        "seacache_dims": list(SEACACHE_DIMS),
        "seacache_norm_mode": SEACACHE_NORM_MODE,
        "step_policy_source": "official_hunyuanvideo_port",
        "source_commit": SOURCE_COMMIT,
        "hunyuanvideo_1_5_commit": HUNYUAN_COMMIT,
    }


def _cache_filename(args, version, width, height, steps):
    fmt = lambda value: format(float(value), "g")
    return (
        f"cache_book_hybrid_seacache_{version}_{width}x{height}"
        f"_f{args.video_length}_steps{steps}"
        f"_ns{fmt(args.nonskip_rate)}_seath{fmt(args.seacache_thresh)}"
        f"_diath{fmt(args.double_img_attn_thres)}"
        f"_dtath{fmt(args.double_txt_attn_thres)}"
        f"_dimth{fmt(args.double_img_mlp_thres)}"
        f"_dtmth{fmt(args.double_txt_mlp_thres)}"
        f"_sath{fmt(args.single_attn_thres)}"
        f"_smth{fmt(args.single_mlp_thres)}.json"
    )


def _save_books(path, books, config, observer_mask, peaks):
    payload = {
        "cache_version": CACHE_BOOK_VERSION,
        "cache_scope": CACHE_SCOPE,
        "policy": POLICY_VARIANT,
        "step_policy": STEP_POLICY,
        "rate_method": RATE_METHOD,
        "config": config,
        "finegrained_cache": {"module_cache_book": books},
        "step_policy_artifacts": {
            "observer_mask": observer_mask,
            "observer_hit_rate": (
                sum(observer_mask) / len(observer_mask) if observer_mask else 0.0
            ),
        },
        "calibration_peak_allocated_gib": peaks,
    }
    if rank0():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    if dist.is_initialized():
        dist.barrier()


def _load_books(path, expected, steps, depths):
    payload = None
    if rank0():
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload = _broadcast_object(payload)
    if payload.get("cache_version") != CACHE_BOOK_VERSION:
        raise ValueError("Incompatible cache book; re-run calibration.")
    if (
        payload.get("cache_scope") != CACHE_SCOPE
        or payload.get("policy") != POLICY_VARIANT
        or payload.get("step_policy") != STEP_POLICY
        or payload.get("rate_method") != RATE_METHOD
    ):
        raise ValueError(
            "Legacy or incompatible Cache Book; rerun calibration with this script"
        )
    config = payload.get("config", {})
    required = (
        "transformer_version", "task", "width", "height", "frames",
        "steps", "nonskip", "module_thresholds", "cfg_distilled",
        "step_distilled", "sparse_attention", "scheduler_class",
        "scheduler_shift", "guidance_scale", "embedded_guidance_scale",
        "double_blocks",
        "single_blocks", "seacache_thresh", "seacache_power_exp",
        "seacache_dims", "seacache_norm_mode",
        "source_commit", "hunyuanvideo_1_5_commit",
    )
    for key in required:
        if config.get(key) != expected.get(key):
            raise ValueError(
                f"Cache Book mismatch for {key}: expected "
                f"{expected.get(key)!r}, got {config.get(key)!r}"
            )
    module_books = payload.get("finegrained_cache", {}).get(
        "module_cache_book"
    )
    if not isinstance(module_books, dict) or set(module_books) != set(MODULES):
        raise ValueError("Invalid module_cache_book keys")
    for name in MODULES:
        book = module_books[name]
        depth = depths[name]
        if (
            len(book) != steps
            or any(
                not isinstance(row, list)
                or len(row) != depth
                or any(type(value) is not bool for value in row)
                for row in book
            )
        ):
            raise ValueError(f"Invalid Cache Book shape for {name}")
    return module_books


def save_video(video, path):
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("Video saving currently expects batch size 1")
        video = video[0]
    video = (video * 255).clamp(0, 255).to(torch.uint8)
    video = einops.rearrange(video, "c f h w -> f h w c")
    imageio.mimwrite(path, video, fps=24)


def save_config(args, output_path, task, version):
    data = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    path = f"{os.path.splitext(output_path)[0]}_config.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "task": task,
                "transformer_version": version,
                "output_path": output_path,
                "arguments": data,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def load_checkpoint(pipe, path):
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint path does not exist: {path}")
    state = get_model_state_dict(pipe.transformer)
    dcp.load(state_dict={"model": state}, checkpoint_id=path)


def _resolve_prompt(args, task):
    if not args.rewrite:
        return args.prompt
    from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

    image = (
        Image.open(args.image_path).convert("RGB")
        if task == "i2v"
        else None
    )
    prompt = None
    if rank0():
        try:
            prompt = run_prompt_rewrite(args.prompt, image, task)
        except Exception as exc:
            rank0_log(
                f"Prompt rewrite failed, using original prompt: {exc}",
                "WARNING",
            )
            prompt = args.prompt
    return _broadcast_object(prompt)


def _geometry(pipe, args, task):
    if task == "i2v":
        image = Image.open(args.image_path).convert("RGB")
        height, width = pipe.get_closest_resolution_given_reference_image(
            image, pipe.ideal_resolution
        )
    else:
        width_ratio, height_ratio = map(
            int, args.aspect_ratio.split(":")
        )
        height, width = pipe.get_closest_resolution_given_original_size(
            (width_ratio, height_ratio), pipe.ideal_resolution
        )
    return int(width), int(height)


def _validate_args(args):
    enabled = (
        not args.disable_step_cache
        or args.use_finegrained_cache
        or args.finegrained_calibration
    )
    if args.sparse_attn and args.use_sageattn:
        raise ValueError(
            "sparse_attn and use_sageattn cannot be enabled simultaneously"
        )
    if args.enable_step_distill and args.enable_cache:
        raise ValueError(
            "The official cache is not supported with step-distilled models"
        )
    if enabled and args.enable_cache:
        raise ValueError(
            "Official --enable_cache cannot be combined with Finegrained Cache"
        )
    if enabled and args.enable_torch_compile:
        raise ValueError(
            "--enable_torch_compile cannot be combined with dynamic "
            "Finegrained Cache control flow"
        )
    if (
        enabled
        and args.use_fp8_gemm
        and "single_blocks" in args.include_patterns
    ):
        raise ValueError(
            "Single-block FP8 is incompatible with split single-stream "
            "Cache contributions"
        )
    if not 0 <= args.nonskip_rate <= 1:
        raise ValueError("nonskip_rate must be in [0, 1]")
    if any(
        not 0 <= value <= 1 for value in _thresholds(args).values()
    ):
        raise ValueError("All module thresholds must be in [0, 1]")
    if args.runtime_cache_gpu_reserve_gib < 0:
        raise ValueError(
            "runtime_cache_gpu_reserve_gib must be non-negative"
        )
    if args.seacache_thresh < 0:
        raise ValueError("seacache_thresh must be non-negative")
    if args.use_fp8_gemm and "sgl" in args.quant_type:
        try:
            __import__("sgl_kernel")
        except ImportError as exc:
            raise ValueError(
                "sgl_kernel is required by the selected FP8 quant_type"
            ) from exc


def generate(args):
    _validate_args(args)
    task = "i2v" if args.image_path else "t2v"
    version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution,
        task,
        args.cfg_distilled,
        args.enable_step_distill,
        args.sparse_attn,
    )
    if version not in PIPELINE_CONFIGS:
        raise ValueError(
            f"Unsupported official transformer configuration: {version}"
        )
    dtype = (
        torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    enable_group = (
        HunyuanVideo_1_5_Pipeline.get_offloading_config()[
            "enable_group_offloading"
        ]
        if args.group_offloading is None
        else args.group_offloading
    )
    if enable_group and not args.offloading:
        raise ValueError("group_offloading requires offloading")
    device = torch.device("cpu") if args.offloading else torch.device("cuda")
    init_device = torch.device("cpu") if enable_group else device
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        args.model_path,
        version,
        create_sr_pipeline=args.sr,
        transformer_dtype=dtype,
        device=device,
        transformer_init_device=init_device,
    )
    infer_state = initialize_infer_state(args)
    pipe.apply_infer_optimization(
        infer_state,
        args.offloading,
        enable_group,
        args.overlap_group_offloading,
    )
    if args.checkpoint_path:
        load_checkpoint(pipe, args.checkpoint_path)
    if args.lora_path:
        pipe.transformer.load_lora_adapter(
            args.lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
    if args.sr and hasattr(pipe, "sr_pipeline"):
        sr_state = copy.deepcopy(infer_state)
        sr_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            sr_state,
            args.offloading,
            enable_group,
            args.overlap_group_offloading,
        )

    enabled = (
        not args.disable_step_cache
        or args.use_finegrained_cache
        or args.finegrained_calibration
    )
    if enabled:
        _patch_model(pipe.transformer)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    steps = (
        int(pipe.config.num_inference_steps)
        if args.num_inference_steps is None
        else args.num_inference_steps
    )
    if steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if args.finegrained_calibration and steps < 3:
        raise ValueError("Three-point calibration requires at least 3 inference steps")
    do_cfg = float(pipe.config.guidance_scale) > 1.0
    scheduler_config = getattr(pipe.scheduler, "config", {})
    args.resolved_scheduler_class = type(pipe.scheduler).__name__
    scheduler_shift = (
        scheduler_config.get("shift")
        if hasattr(scheduler_config, "get")
        else getattr(scheduler_config, "shift", None)
    )
    args.resolved_scheduler_shift = (
        None if scheduler_shift is None else float(scheduler_shift)
    )
    args.resolved_guidance_scale = float(pipe.config.guidance_scale)
    embedded = getattr(pipe.config, "embedded_guidance_scale", None)
    args.resolved_embedded_guidance_scale = (
        None if embedded is None else float(embedded)
    )
    if enabled:
        pipe.transformer.finegrained_scheduler = pipe.scheduler
    width, height = _geometry(pipe, args, task)
    depths = {
        name: (
            len(pipe.transformer.double_blocks)
            if name.startswith("double")
            else len(pipe.transformer.single_blocks)
        )
        for name in MODULES
    }
    if args.seed == -1:
        args.seed = _broadcast_object(
            random.randint(100000, 999999) if rank0() else None
        )
    prompt = _resolve_prompt(args, task)
    call_kwargs = {
        "prompt": prompt,
        "aspect_ratio": args.aspect_ratio,
        "num_inference_steps": args.num_inference_steps,
        "sr_num_inference_steps": None,
        "video_length": args.video_length,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "prompt_rewrite": False,
        "return_pre_sr_video": args.save_pre_sr_video,
    }
    if task == "i2v":
        call_kwargs["reference_image"] = args.image_path

    def run_once(output_type="latent", enable_sr=False):
        return pipe(
            enable_sr=enable_sr,
            output_type=output_type,
            **call_kwargs,
        )

    books = None
    config = _cache_config(
        args, version, task, width, height, steps, depths
    )
    cache_file = args.cache_book_file or _cache_filename(
        args, version, width, height, steps
    )
    cache_path = os.path.join(args.cache_book_path, cache_file)
    if args.finegrained_calibration:
        raw_modules, observer_mask, raw_peak = _collect_stage(
            run_once, pipe.transformer, args, steps, depths, do_cfg,
            "raw SeaCache observer and layer calibration",
        )
        provisional = _books_from_scores(
            raw_modules, args.nonskip_rate, _thresholds(args)
        )
        del raw_modules
        corrected_modules, _, correction_peak = _collect_stage(
            run_once, pipe.transformer, args, steps, depths, do_cfg,
            "SeaCache-aware correction calibration",
            books=provisional, step_mask=observer_mask,
        )
        books = _books_from_scores(
            corrected_modules, args.nonskip_rate, _thresholds(args)
        )
        del corrected_modules
        _save_books(
            cache_path, books, config, observer_mask,
            {"raw": raw_peak, "correction": correction_peak},
        )
        rank0_log(f"Cache Book saved to {cache_path}")
    elif args.use_finegrained_cache:
        books = _load_books(cache_path, config, steps, depths)

    if not args.use_finegrained_cache and args.finegrained_calibration:
        return
    if enabled:
        _init_runtime(
            pipe.transformer, args, steps, do_cfg, books=books,
            use=args.use_finegrained_cache,
            step_policy=SeaCachePolicy(
                steps, args.seacache_thresh, pipe.scheduler,
                enabled=not args.disable_step_cache,
            ),
            execute_step_cache=not args.disable_step_cache,
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = run_once(output_type="pt", enable_sr=args.sr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        rank0_log(
            "Generation peak allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
        )
    rank0_log(
        f"Generation elapsed: {time.perf_counter() - started:.2f}s"
    )
    if enabled:
        rank0_log(
            f"Synchronized Transformer sampling time: "
            f"{_transformer_seconds(pipe.transformer):.2f}s"
        )
    if enabled:
        total = max(1, pipe.transformer.finegrained_num_steps)
        step_hits = pipe.transformer.finegrained_step_hits
        module_count = 4 * len(pipe.transformer.double_blocks) + 2 * len(
            pipe.transformer.single_blocks
        )
        computed = max(1, pipe.transformer.finegrained_computed_steps)
        layer_hits = pipe.transformer.finegrained_layer_hits
        rank0_log(
            f"SeaCache step hits: {step_hits}/{total} ({step_hits / total:.2%})"
        )
        rank0_log(
            f"Computed-step layer hits: {layer_hits}/"
            f"{computed * module_count} "
            f"({layer_hits / (computed * module_count):.2%})"
        )
        effective = (step_hits * module_count + layer_hits) / (
            total * module_count
        )
        rank0_log(f"Combined effective skip ratio: {effective:.2%}")
    if rank0():
        output_path = args.output_path or (
            f"./outputs/output_{version}_"
            f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if args.sr and hasattr(out, "sr_videos"):
            save_video(out.sr_videos, output_path)
            if args.save_pre_sr_video:
                base, ext = os.path.splitext(output_path)
                save_video(out.videos, f"{base}_before_sr{ext}")
        else:
            save_video(out.videos, output_path)
        if args.save_generation_config:
            save_config(args, output_path, task, version)
        print(f"Saved video to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "HunyuanVideo-1.5 SeaCache + Finegrained Cache hybrid sampler"
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--resolution", required=True, choices=("480p", "720p")
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--aspect_ratio", default="16:9")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--video_length", type=int, default=121)
    parser.add_argument(
        "--sr", type=str_to_bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--save_pre_sr_video",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--rewrite",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--cfg_distilled",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--enable_step_distill",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--sparse_attn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--overlap_group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--dtype", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--use_sageattn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--sage_blocks_range", default="0-53")
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--enable_cache",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--cache_type", default="deepcache")
    parser.add_argument("--no_cache_block_id", default="53")
    parser.add_argument("--cache_start_step", type=int, default=11)
    parser.add_argument("--cache_end_step", type=int, default=45)
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--cache_step_interval", type=int, default=4)
    parser.add_argument(
        "--save_generation_config",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument(
        "--use_fp8_gemm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--quant_type", default="fp8-per-token-sgl")
    parser.add_argument("--include_patterns", default="double_blocks")

    parser.add_argument("--use_finegrained_cache", action="store_true")
    parser.add_argument("--finegrained_calibration", action="store_true")
    parser.add_argument("--disable_step_cache", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--seacache_thresh", type=float, default=0.20)
    parser.add_argument("--cache_book_path", default="./cache_books")
    parser.add_argument("--cache_book_file", default=None)
    parser.add_argument("--nonskip_rate", type=float, default=0.1)
    parser.add_argument(
        "--double_img_attn_thres", type=float, default=0.5
    )
    parser.add_argument(
        "--double_txt_attn_thres", type=float, default=0.5
    )
    parser.add_argument(
        "--double_img_mlp_thres", type=float, default=0.5
    )
    parser.add_argument(
        "--double_txt_mlp_thres", type=float, default=0.5
    )
    parser.add_argument("--single_attn_thres", type=float, default=0.5)
    parser.add_argument("--single_mlp_thres", type=float, default=0.5)
    parser.add_argument(
        "--calibration_feature_device",
        choices=("cpu", "gpu"),
        default="cpu",
    )
    parser.add_argument(
        "--runtime_cache_device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--runtime_cache_gpu_reserve_gib", type=float, default=2.0
    )
    return parser


def main():
    args = build_parser().parse_args()
    if (
        args.image_path is not None
        and args.image_path.lower().strip() == "none"
    ):
        args.image_path = None
    initialize_parallel_state(
        sp=int(os.environ.get("WORLD_SIZE", "1"))
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    try:
        generate(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
