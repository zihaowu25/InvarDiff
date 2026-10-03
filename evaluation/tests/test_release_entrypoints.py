"""Verify release CLI contracts without loading model backends or weights."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

import cache_presets

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = {
    "dit_module": "DiT/sample_dit.py",
    "dit_step_layer": "DiT/sample_dit_step_layer.py",
    "flux_module": "FLUX/sample_flux.py",
    "flux_step_layer": "FLUX/sample_flux_step_layer.py",
    "wan_module": "Wan2.1/sample_wan.py",
    "wan_step_layer": "Wan2.1/sample_wan_step_layer.py",
    "hunyuan_module": "HunyuanVideo/sample_hunyuan.py",
    "hunyuan_step_layer": "HunyuanVideo/sample_hunyuan_step_layer.py",
    "flux_magcache_hybrid": "FLUX/hybrid_cache/sample_flux_magcache_hybrid.py",
    "flux_seacache_hybrid": "FLUX/hybrid_cache/sample_flux_seacache_hybrid.py",
    "wan_magcache_hybrid": "Wan2.1/hybrid_cache/sample_wan_magcache_hybrid.py",
    "wan_seacache_hybrid": "Wan2.1/hybrid_cache/sample_wan_seacache_hybrid.py",
    "hunyuan_magcache_hybrid": "HunyuanVideo/hybrid_cache/sample_hunyuan_magcache_hybrid.py",
    "hunyuan_seacache_hybrid": "HunyuanVideo/hybrid_cache/sample_hunyuan_seacache_hybrid.py",
}
ATTRIBUTES = {
    "cache_book_path", "cache_book_file", "use_invardiff", "use_finegrained_cache",
    "generate_cache_books", "invardiff_calibration", "finegrained_calibration",
    "image_size", "model", "num_classes", "num_timesteps", "cfg_scale",
    "height", "width", "num_inference_steps", "guidance_scale", "nonskip_rate",
    "task", "size", "frame_num", "sample_steps", "sample_solver", "sample_shift",
    "sample_guide_scale", "resolution", "aspect_ratio", "video_length", "dtype",
    "deterministic_attention", "cfg_distilled", "enable_step_distill", "sparse_attn",
    "magcache_thresh", "magcache_K", "retention_ratio", "magcache_ratio_source",
    "seacache_thresh", "seacache_threshold", "disable_step_cache", "num_analysis",
}


def cli_namespace(policy, extra=()):
    path = ROOT / ENTRIES[policy]
    tree = ast.parse(path.read_text())
    parser = argparse.ArgumentParser()
    flag_map = {}
    preset_policy = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "apply_preset":
            preset_policy = ast.literal_eval(node.args[1])
            flag_map = ast.literal_eval(node.args[2])
    assert preset_policy
    attributes = ATTRIBUTES | set(cache_presets.BOOK_CONFIG_FIELDS) | set(flag_map.values())
    namespace = {"parser": parser, "argparse": argparse, "os": os,
                 "__file__": str(path), "str_to_bool": lambda v: str(v).lower() == "true",
                 "str2bool": lambda v: str(v).lower() == "true",
                 "WAN_CONFIGS": {"t2v-1.3B": None}, "SIZE_CONFIGS": {"832*480": None}}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        attr = flag.lstrip("-").replace("-", "_")
        if attr not in attributes:
            continue
        expr = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=node)], type_ignores=[]))
        exec(compile(expr, str(path), "exec"), namespace)
    cache_presets.add_preset_argument(parser, preset_policy)
    argv = (["--resolution", "720p"] if policy.startswith("hunyuan") else []) + list(extra)
    args = parser.parse_args(argv)
    return cache_presets.apply_preset(args, preset_policy, flag_map, argv)


@pytest.mark.parametrize("policy", ENTRIES)
def test_each_sampler_resolves_its_own_default_book(policy):
    args = cli_namespace(policy)
    assert Path(args.cache_book_path) == cache_presets.cache_book_directory(policy)
    assert args.cache_book_file == cache_presets.cache_book_filename(args, policy)
    assert len(args.cache_book_file.encode()) < 240
    if hasattr(args, "use_invardiff"):
        assert args.use_invardiff is True
    if hasattr(args, "use_finegrained_cache"):
        assert args.use_finegrained_cache is True


@pytest.mark.parametrize("policy", ENTRIES)
def test_each_sampler_calibration_and_inference_share_default_path(policy):
    model = policy.split("_")[0]
    if policy.endswith("hybrid"):
        flag = "--finegrained-calibration" if model == "flux" else "--finegrained_calibration"
    else:
        flag = "--generate-cache-books" if model in ("dit", "flux") else "--invardiff_calibration"
    args = cli_namespace(policy, [flag])
    assert Path(args.cache_book_path) == cache_presets.cache_book_directory(policy)
    assert args.cache_book_file == cache_presets.cache_book_filename(args, policy)
    inference = cli_namespace(policy)
    assert args.cache_book_path == inference.cache_book_path
    assert args.cache_book_file == inference.cache_book_file
    if hasattr(args, "use_invardiff"):
        assert args.use_invardiff is False
    if hasattr(args, "use_finegrained_cache"):
        assert args.use_finegrained_cache is False


def test_public_books_match_manifest_and_default_execution_settings():
    file = cache_presets.CACHE_BOOKS / "manifest.json"
    if not file.exists():
        pytest.skip("Release assets have not been packaged yet")
    manifest = json.loads(file.read_text())
    assert set(manifest["policies"]) == set(ENTRIES)
    for policy, item in manifest["policies"].items():
        path = cache_presets.CACHE_BOOKS / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        book = json.loads(path.read_text())
        assert book["release"]["policy"] == policy
        assert "seed" not in book["release"]["execution_args"]
        assert "base_seed" not in book["release"]["execution_args"]
        args = cli_namespace(policy)
        assert Path(item["file"]).name == args.cache_book_file


def test_public_masks_are_boolean_native_length_and_protected():
    file = cache_presets.CACHE_BOOKS / "manifest.json"
    if not file.exists():
        pytest.skip("Release assets have not been packaged yet")
    manifest = json.loads(file.read_text())

    def arrays(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("cache_book"):
                    if isinstance(item, dict):
                        yield from item.values()
                    else:
                        yield item
                else:
                    yield from arrays(item)

    def flattened(value):
        if isinstance(value, list):
            for child in value:
                yield from flattened(child)
        else:
            yield value

    for policy, item in manifest["policies"].items():
        book = json.loads((cache_presets.CACHE_BOOKS / item["file"]).read_text())
        args = cli_namespace(policy)
        steps = 28 if policy.startswith("flux") else 50
        protected = max(1, int(args.nonskip_rate * steps))
        masks = list(arrays(book))
        assert masks
        for mask in masks:
            assert len(mask) == steps
            assert all(type(v) is bool for v in flattened(mask))
            assert not any(flattened(mask[:protected]))
            # The native FLUX MagCache hybrid permits final-step reuse in its
            # serialized step and module artifacts. Keep that existing policy;
            # do not edit its decisions to impose the standalone contract.
            if policy != "flux_magcache_hybrid":
                assert not any(flattened(mask[-1:]))
