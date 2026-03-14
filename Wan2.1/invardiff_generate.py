# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from PIL import Image
import time

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

WAN_CACHE_MODULES = ("self_attn", "cross_attn", "ffn")

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "A graceful portrait of a lady in simple elegant style.",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "image": "examples/i2v_input.JPG",
    },
    "flf2v-14B": {
        "prompt": "A small blue bird takes off from the ground and flaps its wings into a bright blue sky.",
        "first_frame": "examples/flf2v_input_first_frame.png",
        "last_frame": "examples/flf2v_input_last_frame.png",
    },
    "vace-1.3B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "A festive scene where a little girl in red spring clothing happily plays with a cute cartoon snake.",
    },
    "vace-14B": {
        "src_ref_images": "examples/girl.png,examples/snake.png",
        "prompt": "A festive scene where a little girl in red spring clothing happily plays with a cute cartoon snake.",
    },
}


class FeatureChangeAnalyzer:
    def __init__(
        self,
        num_steps: int,
        num_layers: int,
        correction_mode: bool = False,
        step_cache_state: Optional[List[bool]] = None,
        module_cache_state: Optional[Dict[str, List[List[bool]]]] = None,
    ):
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.correction_mode = correction_mode
        self.step_cache_state = step_cache_state
        self.module_cache_state = module_cache_state

        self.step_scores = torch.ones(num_steps, dtype=torch.float32)
        self.module_scores = {
            k: torch.ones((num_steps, num_layers), dtype=torch.float32)
            for k in WAN_CACHE_MODULES
        }

        self._step_state = False
        self._current_hidden_states = None
        self._prev_step_norm = None

        self._module_state = {k: [False] * num_layers for k in WAN_CACHE_MODULES}
        self._current_module_feat = {k: [None] * num_layers for k in WAN_CACHE_MODULES}
        self._prev_module_norm = {k: [None] * num_layers for k in WAN_CACHE_MODULES}

    @staticmethod
    def _compute_norm(x_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (x - x_prev + 1e-8).norm(p=1)

    def _update_step_score(self, step_idx: int, hidden_states: torch.Tensor):
        if step_idx == 0:
            self._current_hidden_states= hidden_states
            return

        if step_idx == 1:
            self._prev_step_norm = self._compute_norm(self._current_hidden_states, hidden_states)
            self._current_hidden_states = hidden_states
            return

        curr_norm = self._compute_norm(self._current_hidden_states, hidden_states)
        score = float((curr_norm / self._prev_step_norm).item())
        self.step_scores[step_idx-1] = score

        if not self.correction_mode:
            self._current_hidden_states = hidden_states
            self._prev_step_norm = curr_norm
            return

        if self.step_cache_state is None:
            self._current_hidden_states = hidden_states
            self._prev_step_norm = curr_norm
            return

        if not self.step_cache_state[step_idx - 1]:
            self._current_hidden_states = hidden_states
            self._prev_step_norm = curr_norm
            self._step_state = False
        elif not self._step_state:
            self._current_hidden_states = hidden_states
            self._prev_step_norm = curr_norm
            self._step_state = True

    def _update_module_scores(
        self,
        step_idx: int,
        module_outputs: Dict[str, List[torch.Tensor]],
    ):
        for mod_name in WAN_CACHE_MODULES:
            for layer_idx, feat in enumerate(module_outputs[mod_name]):
                if feat is None:
                    continue

                if step_idx == 0:
                    self._current_module_feat[mod_name][layer_idx] = feat
                    continue

                if step_idx == 1:
                    self._prev_module_norm[mod_name][layer_idx] = self._compute_norm(
                        self._current_module_feat[mod_name][layer_idx],
                        feat,
                    )
                    self._current_module_feat[mod_name][layer_idx] = feat
                    continue

                curr_norm = self._compute_norm(self._current_module_feat[mod_name][layer_idx], feat)
                prev_norm = self._prev_module_norm[mod_name][layer_idx]
                score = float((curr_norm / prev_norm ).item())
                self.module_scores[mod_name][step_idx - 1, layer_idx] = score

                if not self.correction_mode:
                    self._current_module_feat[mod_name][layer_idx] = feat
                    self._prev_module_norm[mod_name][layer_idx] = curr_norm
                    continue

                if self.module_cache_state is None:
                    self._current_module_feat[mod_name][layer_idx] = feat
                    self._prev_module_norm[mod_name][layer_idx] = curr_norm
                    continue

                if not self.module_cache_state[mod_name][step_idx - 1][layer_idx]:
                    self._current_module_feat[mod_name][layer_idx] = feat
                    self._prev_module_norm[mod_name][layer_idx] = curr_norm
                    self._module_state[mod_name][layer_idx] = False
                elif not self._module_state[mod_name][layer_idx]:
                    self._current_module_feat[mod_name][layer_idx] = feat
                    self._prev_module_norm[mod_name][layer_idx] = curr_norm
                    self._module_state[mod_name][layer_idx] = True

    def update(self, step_idx: int, hidden_states: torch.Tensor, module_outputs: Dict[str, List[torch.Tensor]]):
        # Only record conditional branch (CFG cond call comes first in Wan pipelines).
        if step_idx % 2 != 0:
            return

        policy_step_idx = step_idx // 2
        if policy_step_idx >= self.num_steps:
            return
        self._update_step_score(policy_step_idx, hidden_states)
        self._update_module_scores(policy_step_idx, module_outputs)

    def reset(self):
        pass

def _to_bool_list(x: torch.Tensor):
    return x.to(dtype=torch.bool).cpu().tolist()

def _compute_quantile_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    valid = values[~torch.isnan(values)]
    if valid.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    threshold = torch.quantile(valid, q)
    return values < threshold

def _books_from_scores(
    step_scores: torch.Tensor,
    module_scores: Dict[str, torch.Tensor],
    nonskip_rate: float,
    step_thres: float,
    module_thres: Dict[str, float],
) -> Tuple[List[bool], Dict[str, List[List[bool]]]]:
    num_steps = int(step_scores.shape[0])
    num_layers = int(next(iter(module_scores.values())).shape[1])

    num_nonskip = max(1, int(nonskip_rate * num_steps))

    step_mask = _compute_quantile_mask(step_scores[1:-1], step_thres)
    step_cache_bool = torch.zeros(num_steps, dtype=torch.bool)
    step_cache_bool[1:-1] = step_mask
    step_cache_bool[:num_nonskip] = False
    step_cache_bool[-1] = False

    module_cache_book: Dict[str, torch.Tensor] = {}
    for mod_name in WAN_CACHE_MODULES:
        scores = module_scores[mod_name]
        flat = scores[1:-1].reshape(-1)
        mod_mask = _compute_quantile_mask(flat, module_thres[mod_name])
        mask_2d = mod_mask.reshape(num_steps - 2, num_layers)

        book = torch.zeros_like(scores, dtype=torch.bool)
        book[1:-1] = mask_2d
        book[:num_nonskip, :] = False
        book[-1, :] = False
        module_cache_book[mod_name] = book

    # Step-first caching: if a whole step is cached, all modules at that step are cached.
    # Disable the next step to reduce consecutive drift bursts.
    for t in range(num_steps):
        if step_cache_bool[t]:
            for mod_name in WAN_CACHE_MODULES:
                module_cache_book[mod_name][t, :] = True
            if t + 1 < num_steps:
                for mod_name in WAN_CACHE_MODULES:
                    module_cache_book[mod_name][t + 1, :] = False

    return _to_bool_list(step_cache_bool), {k: _to_bool_list(v) for k, v in module_cache_book.items()}


def _save_cache_books(file_path: str, step_cache_bool: List[bool], module_cache_book: Dict[str, List[List[bool]]] ):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    payload = {
        "step_cache_bool": step_cache_bool,
        "module_cache_book": module_cache_book,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_cache_books(file_path: str) -> Tuple[List[bool], Dict[str, List[List[bool]]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    step_cache_bool = payload.get("step_cache_bool", payload.get("step_cache_book", []))
    module_cache_book = payload.get("module_cache_book", {})

    if not module_cache_book:
        raise ValueError(f"Invalid cache book format: {file_path}")

    for key in WAN_CACHE_MODULES:
        if key not in module_cache_book:
            raise ValueError(f"Missing module cache key '{key}' in {file_path}")

    return step_cache_bool, module_cache_book


def _build_default_cache_book_filename(args) -> str:
    return (
        f"invardiff_{args.task}_stp{args.sample_steps}"
        f"_n{args.nonskip_rate}_sth{args.step_thres}"
        f"_sa{args.self_attn_thres}_ca{args.cross_attn_thres}_ff{args.ffn_thres}.json"
    )


def _prepare_common_inputs(self, x, t, context, seq_len, clip_fea=None, y=None):
    if self.model_type in ("i2v", "flf2v"):
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )
    return x, e, kwargs, grid_sizes


def _wan_block_forward_with_cache(
    block,
    x,
    kwargs,
    module_plan: Dict[str, bool],
    module_cache: Dict[str, Optional[torch.Tensor]],
):
    with amp.autocast(dtype=torch.float32):
        e = (block.modulation + kwargs["e"]).chunk(6, dim=1)

    if module_plan["self_attn"] and module_cache["self_attn"] is not None:
        y_self = module_cache["self_attn"]
    else:
        y_self = block.self_attn(
            block.norm1(x).float() * (1 + e[1]) + e[0],
            kwargs["seq_lens"],
            kwargs["grid_sizes"],
            kwargs["freqs"],
        )
    with amp.autocast(dtype=torch.float32):
        x = x + y_self * e[2]

    if module_plan["cross_attn"] and module_cache["cross_attn"] is not None:
        y_cross = module_cache["cross_attn"]
    else:
        y_cross = block.cross_attn(block.norm3(x), kwargs["context"], kwargs["context_lens"])
    x = x + y_cross

    if module_plan["ffn"] and module_cache["ffn"] is not None:
        y_ffn = module_cache["ffn"]
    else:
        y_ffn = block.ffn(block.norm2(x).float() * (1 + e[4]) + e[3])
    with amp.autocast(dtype=torch.float32):
        x = x + y_ffn * e[5]

    # VACE base block adds hint skip after parent forward.
    if hasattr(block, "block_id") and block.block_id is not None and "hints" in kwargs:
        x = x + kwargs["hints"][block.block_id] * kwargs.get("context_scale", 1.0)

    return x, {"self_attn": y_self, "cross_attn": y_cross, "ffn": y_ffn}


def _maybe_reset_runtime(self):
    if self.invardiff_timestep_idx >= self.invardiff_num_steps:
        self.invardiff_timestep_idx = 0
        self.step_cache = [None, None]
        for mod_name in WAN_CACHE_MODULES:
            self.module_feature_cache[mod_name] = [[None] * self.invardiff_num_layers for _ in range(2)]


def _invardiff_core_forward(self, x, e, kwargs, grid_sizes):
    step_idx = self.invardiff_timestep_idx
    slot = step_idx % 2
    policy_step_idx = step_idx // 2

    use_step_cache = (
        self.invardiff_use and (not self.invardiff_calib_running)
        and policy_step_idx > 0 and policy_step_idx < len(self.step_cache_bool)
        and self.step_cache_bool[policy_step_idx]
        and self.step_cache[slot] is not None
    )

    module_outputs = {k: [None] * self.invardiff_num_layers for k in WAN_CACHE_MODULES}

    if use_step_cache:
        x = self.step_cache[slot]
    else:
        for layer_idx, block in enumerate(self.blocks):
            if self.invardiff_use and (not self.invardiff_calib_running):
                module_plan = {
                    k: bool(self.module_cache_book[k][policy_step_idx][layer_idx])
                    if policy_step_idx < len(self.module_cache_book[k]) else False
                    for k in WAN_CACHE_MODULES
                }
            else:
                module_plan = {k: False for k in WAN_CACHE_MODULES}

            layer_cache = {
                k: self.module_feature_cache[k][slot][layer_idx]
                for k in WAN_CACHE_MODULES
            }

            x, outs = _wan_block_forward_with_cache(block, x, kwargs, module_plan, layer_cache)

            for mod_name in WAN_CACHE_MODULES:
                module_outputs[mod_name][layer_idx] = outs[mod_name].detach()
                self.module_feature_cache[mod_name][slot][layer_idx] = outs[mod_name].detach()

        self.step_cache[slot] = x.detach()

    if self.feature_change_analyzer is not None:
        self.feature_change_analyzer.update(step_idx, x, module_outputs)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.invardiff_timestep_idx += 1
    _maybe_reset_runtime(self)

    return [u.float() for u in x]


def invardiff_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    x, e, kwargs, grid_sizes = _prepare_common_inputs(self, x, t, context, seq_len, clip_fea, y)
    return _invardiff_core_forward(self, x, e, kwargs, grid_sizes)


def invardiff_vace_forward(
    self,
    x,
    t,
    vace_context,
    context,
    seq_len,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    # VACE does not use clip/y path in current implementation.
    x, e, kwargs, grid_sizes = _prepare_common_inputs(self, x, t, context, seq_len, None, None)

    step_idx = self.invardiff_timestep_idx
    policy_step_idx = step_idx // 2
    use_step_cache = (
        self.invardiff_use and (not self.invardiff_calib_running)
        and policy_step_idx > 0 and policy_step_idx < len(self.step_cache_bool)
        and self.step_cache_bool[policy_step_idx]
        and self.step_cache[step_idx % 2] is not None
    )

    if not use_step_cache:
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs["hints"] = hints
        kwargs["context_scale"] = vace_context_scale

    return _invardiff_core_forward(self, x, e, kwargs, grid_sizes)


def _init_invardiff_runtime(
    model,
    num_steps: int,
    step_cache_bool: Optional[List[bool]] = None,
    module_cache_book: Optional[Dict[str, List[List[bool]]]] = None,
    use_invardiff: bool = False,
    analyzer: Optional[FeatureChangeAnalyzer] = None,
):
    model.invardiff_num_steps = num_steps
    model.invardiff_num_layers = len(model.blocks)
    model.invardiff_use = use_invardiff
    model.invardiff_calib_running = analyzer is not None
    model.invardiff_timestep_idx = 0
    model.step_cache = [None, None]
    model.module_feature_cache = {
        k: [[None] * model.invardiff_num_layers for _ in range(2)]
        for k in WAN_CACHE_MODULES
    }
    policy_steps = num_steps // 2
    model.step_cache_bool = step_cache_bool if step_cache_bool is not None else [False] * policy_steps
    model.module_cache_book = module_cache_book if module_cache_book is not None else {
        k: [[False] * model.invardiff_num_layers for _ in range(policy_steps)]
        for k in WAN_CACHE_MODULES
    }
    model.feature_change_analyzer = analyzer


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, "
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )



def _parse_args(cli_args=None):
    parser = argparse.ArgumentParser(description="Generate image/video using Wan with InvarDiff acceleration")
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--save_file", type=str, default=None)

    parser.add_argument("--src_video", type=str, default=None)
    parser.add_argument("--src_mask", type=str, default=None)
    parser.add_argument("--src_ref_images", type=str, default=None)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="zh", choices=["zh", "en"])

    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--first_frame", type=str, default=None)
    parser.add_argument("--last_frame", type=str, default=None)

    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    # InvarDiff options
    parser.add_argument("--use_invardiff", action="store_true", default=False)
    parser.add_argument("--invardiff_calibration", action="store_true", default=False)
    parser.add_argument("--cache_book_path", type=str, default="./cache_books")
    parser.add_argument("--cache_book_file", type=str, default=None)
    parser.add_argument("--nonskip_rate", type=float, default=0.1)
    parser.add_argument("--step_thres", type=float, default=0.7)
    parser.add_argument("--self_attn_thres", type=float, default=0.68)
    parser.add_argument("--cross_attn_thres", type=float, default=0.68)
    parser.add_argument("--ffn_thres", type=float, default=0.68)

    args = parser.parse_args(cli_args)
    _validate_args(args)
    return args



def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)



def _setup_distributed(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_id = local_rank

    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp/dit_fsdp require distributed run"
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel requires distributed run"

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    return rank, world_size, device_id



def _make_prompt_expander(args, rank):
    if not args.use_prompt_extend:
        return None

    if args.prompt_extend_method == "dashscope":
        return DashScopePromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task or "flf2v" in args.task,
        )
    if args.prompt_extend_method == "local_qwen":
        return QwenPromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task,
            device=rank,
        )
    raise NotImplementedError(args.prompt_extend_method)



def _calibrate_invardiff(run_once, model, args):
    policy_steps = args.sample_steps
    runtime_steps = args.sample_steps * 2
    num_layers = len(model.blocks)

    module_thres = {
        "self_attn": args.self_attn_thres,
        "cross_attn": args.cross_attn_thres,
        "ffn": args.ffn_thres,
    }

    # Phase 1: full-run statistics.
    analyzer1 = FeatureChangeAnalyzer(policy_steps, num_layers, correction_mode=False)
    _init_invardiff_runtime(model, num_steps=runtime_steps, use_invardiff=False, analyzer=analyzer1)
    _ = run_once()

    provisional_step_cache_bool, provisional_module_cache_book = _books_from_scores(
        analyzer1.step_scores,
        analyzer1.module_scores,
        nonskip_rate=args.nonskip_rate,
        step_thres=args.step_thres,
        module_thres=module_thres,
    )

    # Phase 2: correction pass with cache-state-aware baseline updates.
    analyzer2 = FeatureChangeAnalyzer(
        policy_steps,
        num_layers,
        correction_mode=True,
        step_cache_state=provisional_step_cache_bool,
        module_cache_state=provisional_module_cache_book,
    )
    _init_invardiff_runtime(model, num_steps=runtime_steps, use_invardiff=False, analyzer=analyzer2)
    _ = run_once()

    final_step_cache_bool, final_module_cache_book = _books_from_scores(
        analyzer2.step_scores,
        analyzer2.module_scores,
        nonskip_rate=args.nonskip_rate,
        step_thres=args.step_thres,
        module_thres=module_thres,
    )

    cache_file = args.cache_book_file or _build_default_cache_book_filename(args)
    cache_path = os.path.join(args.cache_book_path, cache_file)
    _save_cache_books(cache_path, final_step_cache_bool, final_module_cache_book)
    logging.info(f"InvarDiff cache books saved to: {cache_path}")

    return final_step_cache_bool, final_module_cache_book



def _prepare_invardiff_books(model, args):
    step_cache_bool = None
    module_cache_book = None

    if args.use_invardiff and not args.invardiff_calibration:
        cache_file = args.cache_book_file or _build_default_cache_book_filename(args)
        cache_path = os.path.join(args.cache_book_path, cache_file)
        step_cache_bool, module_cache_book = _load_cache_books(cache_path)
        logging.info(f"Loaded InvarDiff cache books: {cache_path}")

    _init_invardiff_runtime(
        model,
        num_steps=args.sample_steps * 2,
        step_cache_bool=step_cache_bool,
        module_cache_book=module_cache_book,
        use_invardiff=args.use_invardiff and step_cache_bool is not None,
        analyzer=None,
    )



def generate(args):
    video = None
    rank, world_size, device = _setup_distributed(args)

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    prompt_expander = _make_prompt_expander(args, rank)

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        enable_invardiff = args.use_invardiff or args.invardiff_calibration

        if enable_invardiff:
            pipeline.model.forward = invardiff_forward.__get__(pipeline.model, pipeline.model.__class__)

        def run_once():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = pipeline.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if rank == 0:
                logging.info(f"[Latency] {args.task} generate: {dt:.4f}s")

            return out

        if args.invardiff_calibration:
            step_cache_bool, module_cache_book = _calibrate_invardiff(run_once, pipeline.model, args)

            if args.use_invardiff:
                _init_invardiff_runtime(
                    pipeline.model,
                    num_steps=args.sample_steps * 2,
                    step_cache_bool=step_cache_bool,
                    module_cache_book=module_cache_book,
                    use_invardiff=args.use_invardiff,
                    analyzer=None,
                )
                video = run_once()
        elif args.use_invardiff:
            _prepare_invardiff_books(pipeline.model, args)
            video = run_once()
        else:
            video = run_once()

    elif "i2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

        img = Image.open(args.image).convert("RGB")

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed,
                )
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        enable_invardiff = args.use_invardiff or args.invardiff_calibration

        if enable_invardiff:
            pipeline.model.forward = invardiff_forward.__get__(pipeline.model, pipeline.model.__class__)

        def run_once():
            return pipeline.generate(
                args.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        if args.invardiff_calibration:
            step_cache_bool, module_cache_book = _calibrate_invardiff(run_once, pipeline.model, args)
            
            if args.use_invardiff:
                _init_invardiff_runtime(
                    pipeline.model,
                    num_steps=args.sample_steps * 2,
                    step_cache_bool=step_cache_bool,
                    module_cache_book=module_cache_book,
                    use_invardiff=args.use_invardiff,
                    analyzer=None,
                )
                video = run_once()
        elif args.use_invardiff:
            _prepare_invardiff_books(pipeline.model, args)
            video = run_once()
        else:
            video = run_once()

    elif "flf2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.first_frame is None or args.last_frame is None:
            args.first_frame = EXAMPLE_PROMPT[args.task]["first_frame"]
            args.last_frame = EXAMPLE_PROMPT[args.task]["last_frame"]

        first_frame = Image.open(args.first_frame).convert("RGB")
        last_frame = Image.open(args.last_frame).convert("RGB")

        if args.use_prompt_extend:
            if rank == 0:
                out = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=[first_frame, last_frame],
                    seed=args.base_seed,
                )
                prompt_val = out.prompt if out.status else args.prompt
                shared = [prompt_val]
            else:
                shared = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(shared, src=0)
            args.prompt = shared[0]

        pipeline = wan.WanFLF2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # FLF2V shares WanModel forward signature, so InvarDiff is directly reusable.
        enable_invardiff = args.use_invardiff or args.invardiff_calibration

        if enable_invardiff:
            pipeline.model.forward = invardiff_forward.__get__(pipeline.model, pipeline.model.__class__)

        def run_once():
            return pipeline.generate(
                args.prompt,
                first_frame,
                last_frame,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        if args.invardiff_calibration:
            step_cache_bool, module_cache_book = _calibrate_invardiff(run_once, pipeline.model, args)
            
            if args.use_invardiff:
                _init_invardiff_runtime(
                    pipeline.model,
                    num_steps=args.sample_steps * 2,
                    step_cache_bool=step_cache_bool,
                    module_cache_book=module_cache_book,
                    use_invardiff=args.use_invardiff,
                    analyzer=None,
                )
                video = run_once()
        elif args.use_invardiff:
            _prepare_invardiff_books(pipeline.model, args)
            video = run_once()
        else:
            video = run_once()

    elif "vace" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            args.src_video = EXAMPLE_PROMPT[args.task].get("src_video", None)
            args.src_mask = EXAMPLE_PROMPT[args.task].get("src_mask", None)
            args.src_ref_images = EXAMPLE_PROMPT[args.task].get("src_ref_images", None)

        pipeline = wan.WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        enable_invardiff = args.use_invardiff or args.invardiff_calibration

        if enable_invardiff:
            pipeline.model.forward = invardiff_vace_forward.__get__(pipeline.model, pipeline.model.__class__)

        src_video, src_mask, src_ref_images = pipeline.prepare_source(
            [args.src_video],
            [args.src_mask],
            [None if args.src_ref_images is None else args.src_ref_images.split(",")],
            args.frame_num,
            SIZE_CONFIGS[args.size],
            device,
        )

        def run_once():
            return pipeline.generate(
                args.prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        if args.invardiff_calibration:
            step_cache_bool, module_cache_book = _calibrate_invardiff(run_once, pipeline.model, args)
            
            if args.use_invardiff:
                _init_invardiff_runtime(
                    pipeline.model,
                    num_steps=args.sample_steps * 2,
                    step_cache_bool=step_cache_bool,
                    module_cache_book=module_cache_book,
                    use_invardiff=args.use_invardiff,
                    analyzer=None,
                )
                video = run_once()
        elif args.use_invardiff:
            _prepare_invardiff_books(pipeline.model, args)
            video = run_once()
        else:
            video = run_once()

    else:
        raise ValueError(f"Unknown task type: {args.task}")

    if rank == 0 and video is not None:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".png" if "t2i" in args.task else ".mp4"
            inv_params = "_invar" if args.use_invardiff else "_full"
            args.save_file = (
                f"invardiff_{args.task}{inv_params}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_"
                f"{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"
            )
        save_path = os.path.abspath(args.save_file)
        if "t2i" in args.task:
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=save_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            cache_video(
                tensor=video[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        logging.info(f"Output will be saved to: {save_path}")

    logging.info("Finished.")


if __name__ == "__main__":

    debug_args = [
    ## Core task/model
    "--task", "t2v-1.3B",
    "--ckpt_dir", "./Wan2.1-T2V-1.3B",

    ## Output controls
    "--size", "832*480",
    "--frame_num", "81",
    # "--save_file", "debug_t2v_1_3b.mp4",

    ## Sampling controls
    "--sample_solver", "unipc",      # or dpm++
    "--sample_steps", "30",
    "--sample_shift", "5.0",
    "--sample_guide_scale", "5.0",
    "--base_seed", "0",

    ## Prompt
    "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    # "--use_prompt_extend",
    # "--prompt_extend_method", "local_qwen",   # or dashscope
    # "--prompt_extend_model", "Qwen2.5-7B-Instruct",
    # "--prompt_extend_target_lang", "en",

    ## Runtime / parallel
    # "--offload_model", "True",      # True/False (string)
    "--ulysses_size", "1",
    "--ring_size", "1",
    # "--t5_fsdp",
    # "--dit_fsdp",
    # "--t5_cpu",

    ## InvarDiff controls

    "--invardiff_calibration",
    "--use_invardiff",

    "--cache_book_path", "./cache_books",
    # "--cache_book_file", "invardiff_t2v-1.3B_stp50_n0.1_sth0.7_sa0.68_ca0.68_ff0.68.json",

    "--nonskip_rate", "0.1",
    "--step_thres", "0.58",
    "--self_attn_thres", "0.5",
    "--cross_attn_thres", "0.5",
    "--ffn_thres", "0.5",
    ]
    cli_args = _parse_args(debug_args)
    # cli_args = _parse_args()
    generate(cli_args)
