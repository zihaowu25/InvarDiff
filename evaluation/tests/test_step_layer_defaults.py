"""Check CLI defaults and Cache Book guards without importing model backends."""
import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cache_presets


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "dit": ("DiT/sample_dit_step_layer.py", .04, {"step_thres": .55, "msa_thres": .50, "mlp_thres": .15}),
    "flux": ("FLUX/sample_flux_step_layer.py", .08, {"step_thres": .52, "attn_thres": 1., "context_attn_thres": 1., "ff_thres": 1., "context_ff_thres": 1., "single_attn_thres": 1., "single_mlp_thres": 1.}),
    "wan": ("Wan2.1/sample_wan_step_layer.py", .04, {"step_thres": .63, "self_attn_thres": .82, "cross_attn_thres": 1., "ffn_thres": .82}),
    "hunyuan": ("HunyuanVideo/sample_hunyuan_step_layer.py", .06, {"step_thres": .70, "double_img_attn_thres": .40, "double_txt_attn_thres": .01, "double_img_mlp_thres": .20, "double_txt_mlp_thres": .32, "single_attn_thres": 0., "single_mlp_thres": 0.}),
}


def isolated_functions(model, *names):
    path = ROOT / MODELS[model][0]
    tree = ast.parse(path.read_text())
    constants = {"CACHE_BOOK_VERSION", "CACHE_SCOPE", "POLICY_VARIANT", "RATE_METHOD", "WAN_CACHE_MODULES", "MODULES"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if
              (isinstance(node, ast.FunctionDef) and node.name in names) or
              (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    namespace = {"json": json, "os": os, "torch": torch,
                 "same_execution_config": cache_presets.same_execution_config,
                 "rank0": lambda: True, "_broadcast_object": lambda x: x}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("model", MODELS)
def test_single_default_cli_prefix_and_explicit_overrides(model):
    path, prefix, expected = MODELS[model]
    parser = argparse.ArgumentParser()
    mapping = {}
    tree = ast.parse((ROOT / path).read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str):
            continue
        attr = flag.lstrip("-").replace("-", "_")
        if attr in expected or attr == "nonskip_rate":
            expr = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=node)], type_ignores=[]))
            exec(compile(expr, str(ROOT / path), "exec"), {"parser": parser})
            if attr in expected:
                mapping[flag] = attr
    policy = model + "_step_layer"
    assert list(cache_presets.load_presets()[policy]["presets"]) == ["default"]
    cache_presets.add_preset_argument(parser, policy)
    args = cache_presets.apply_preset(parser.parse_args([]), policy, mapping, [])
    assert args.cache_preset == "default"
    assert args.nonskip_rate == prefix
    assert {k: getattr(args, k) for k in expected} == expected
    flag = next(iter(mapping))
    prefix_flag = "--nonskip-rate" if model in ("dit", "flux") else "--nonskip_rate"
    argv = [flag + "=0.17", prefix_flag, "0.10"]
    override = cache_presets.apply_preset(parser.parse_args(argv), policy, mapping, argv)
    assert getattr(override, mapping[flag]) == .17
    assert override.nonskip_rate == .10


def test_dit_book_rejects_changed_prefix(tmp_path):
    ns = isolated_functions("dit", "load_cache_books")
    conf = {"steps": 3, "nonskip": .04, "step": .55, "msa": .50, "mlp": .15}
    payload = {"cache_version": ns["CACHE_BOOK_VERSION"], "cache_scope": ns["CACHE_SCOPE"],
               "rate_method": ns["RATE_METHOD"], "config": conf, "step_cache_book": [False] * 3,
               "msa_cache_book": [[False] * 2 for _ in range(3)], "mlp_cache_book": [[False] * 2 for _ in range(3)]}
    (tmp_path / "book.json").write_text(json.dumps(payload))
    ns["load_cache_books"](str(tmp_path), "book.json", conf)
    with pytest.raises(ValueError, match="configuration mismatch"):
        ns["load_cache_books"](str(tmp_path), "book.json", {**conf, "nonskip": .06})


def test_flux_book_allows_label_change_but_rejects_threshold_change(tmp_path):
    ns = isolated_functions("flux", "load_cache_books")
    conf = {"cache_preset": "legacy", "step": .52, "nonskip": .08}
    payload = {"cache_version": ns["CACHE_BOOK_VERSION"], "rate_method": ns["RATE_METHOD"], "config": conf,
               "step_cache_book": [False] * 3, "transformer_cache_book": {}, "single_transformer_cache_book": {}}
    (tmp_path / "book.json").write_text(json.dumps(payload))
    ns["load_cache_books"](str(tmp_path), "book.json", expected_config={**conf, "cache_preset": "default"})
    with pytest.raises(ValueError, match="configuration mismatch"):
        ns["load_cache_books"](str(tmp_path), "book.json", expected_config={**conf, "step": .62})


def test_wan_book_ignores_seed_but_checks_thresholds_and_prefix(tmp_path):
    ns = isolated_functions("wan", "_cache_book_config", "_load_cache_books")
    args = SimpleNamespace(task="t2v-1.3B", size="832*480", frame_num=81, sample_steps=3,
                           nonskip_rate=.04, base_seed=1, **MODELS["wan"][2])
    conf = ns["_cache_book_config"](args, 2)
    payload = {"cache_version": ns["CACHE_BOOK_VERSION"], "cache_scope": ns["CACHE_SCOPE"],
               "policy": ns["POLICY_VARIANT"], "rate_method": ns["RATE_METHOD"], "config": conf,
               "step_cache_book": [False] * 3, "module_cache_book": {k: [[False] * 2 for _ in range(3)] for k in ns["WAN_CACHE_MODULES"]}}
    path = tmp_path / "book.json"
    path.write_text(json.dumps(payload))
    args.base_seed = 2
    ns["_load_cache_books"](str(path), args, 2)
    args.nonskip_rate = .06
    with pytest.raises(ValueError, match="config mismatch"):
        ns["_load_cache_books"](str(path), args, 2)
    args.nonskip_rate = .04
    args.ffn_thres = .72
    with pytest.raises(ValueError, match="config mismatch"):
        ns["_load_cache_books"](str(path), args, 2)


def test_hunyuan_book_accepts_unseen_seed_and_rejects_changed_module_quantile(tmp_path):
    ns = isolated_functions("hunyuan", "_load_books")
    conf = {"transformer_version": "test", "task": "t2v", "width": 1280, "height": 720,
            "frames": 121, "steps": 3, "nonskip": .06, "step_threshold": .70,
            "module_thresholds": MODELS["hunyuan"][2], "deterministic_attention": True,
            "double_blocks": 2, "single_blocks": 0, "seed": 1}
    depths = {k: 0 if k.startswith("single.") else 2 for k in ns["MODULES"]}
    payload = {"cache_version": ns["CACHE_BOOK_VERSION"], "cache_scope": ns["CACHE_SCOPE"],
               "policy": ns["POLICY_VARIANT"], "rate_method": ns["RATE_METHOD"], "config": conf,
               "step_cache_book": [False] * 3, "module_cache_book": {k: [[False] * depth for _ in range(3)] for k, depth in depths.items()}}
    path = tmp_path / "book.json"
    path.write_text(json.dumps(payload))
    ns["_load_books"](str(path), {**conf, "seed": 2}, 3, depths)
    changed = {**conf, "module_thresholds": {**conf["module_thresholds"], "double_img_mlp_thres": .30}}
    with pytest.raises(ValueError, match="Cache Book mismatch"):
        ns["_load_books"](str(path), changed, 3, depths)
