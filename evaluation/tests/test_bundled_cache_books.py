import argparse
import json
from pathlib import Path

import pytest

import cache_presets


@pytest.fixture(autouse=True)
def isolated_cache_directory(tmp_path, monkeypatch):
    # These tests use minimal namespaces; complete sampler/header compatibility
    # is exercised against the real release assets in test_release_entrypoints.
    monkeypatch.setattr(cache_presets, "CACHE_BOOKS", tmp_path / "cache_books")


def namespace(**overrides):
    values = dict(cache_book_path=None, cache_book_file=None,
                  invardiff_calibration=False, use_invardiff=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_inference_loads_released_policy_by_default():
    args = namespace()
    cache_presets.configure_cache_book(args, "wan_module")
    assert Path(args.cache_book_path) == cache_presets.CACHE_BOOKS / "Wan2.1"
    assert args.cache_book_file == cache_presets.cache_book_filename(args, "wan_module")
    assert args.use_invardiff is True


def test_calibration_and_inference_use_the_same_default_file():
    args = namespace(invardiff_calibration=True)
    cache_presets.configure_cache_book(args, "wan_module")
    assert Path(args.cache_book_path) == cache_presets.CACHE_BOOKS / "Wan2.1"
    assert args.cache_book_file == cache_presets.cache_book_filename(args, "wan_module")
    assert args.use_invardiff is False
    inference = namespace()
    cache_presets.configure_cache_book(inference, "wan_module")
    assert inference.cache_book_path == args.cache_book_path
    assert inference.cache_book_file == args.cache_book_file


def test_explicit_folder_filename_and_opt_out_are_preserved(tmp_path):
    args = namespace(cache_book_path=str(tmp_path), cache_book_file="custom.json", use_invardiff=False)
    cache_presets.configure_cache_book(args, "wan_module")
    assert args.cache_book_path == str(tmp_path)
    assert args.cache_book_file == "custom.json"
    assert args.use_invardiff is False


def test_hybrid_default_is_module_plus_step_and_calibration_can_be_only():
    args = namespace(use_finegrained_cache=None)
    cache_presets.configure_cache_book(args, "wan_magcache_hybrid")
    assert args.use_finegrained_cache is True
    args = namespace(use_finegrained_cache=None, finegrained_calibration=True)
    cache_presets.configure_cache_book(args, "wan_magcache_hybrid")
    assert args.use_finegrained_cache is False


def test_released_book_rejects_wrong_execution_setting_and_policy(tmp_path):
    file = tmp_path / "book.json"
    file.write_text(json.dumps({"release": {"policy": "wan_module", "execution_args": {"frame_num": 81}}}))
    args = namespace(cache_book_path=str(tmp_path), cache_book_file="book.json", frame_num=49)
    with pytest.raises(ValueError, match="frame_num"):
        cache_presets.configure_cache_book(args, "wan_module")
    args.frame_num = 81
    cache_presets.configure_cache_book(args, "wan_module")
    with pytest.raises(ValueError, match="belongs"):
        cache_presets.configure_cache_book(args, "wan_step_layer")


def test_custom_legacy_book_is_still_supported(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"cache_version": 2}))
    args = namespace(cache_book_path=str(tmp_path), cache_book_file="old.json")
    cache_presets.configure_cache_book(args, "wan_module")
    assert args.use_invardiff is True


def test_historical_hunyuan_preset_resolves_to_magcache_book():
    args = namespace(use_finegrained_cache=None)
    cache_presets.configure_cache_book(args, "hunyuan_hybrid")
    assert Path(args.cache_book_path) == cache_presets.CACHE_BOOKS / "HunyuanVideo"
    assert args.use_finegrained_cache is True


def test_calibration_allows_an_explicit_absolute_filename(tmp_path):
    args = namespace(invardiff_calibration=True, cache_book_path=str(tmp_path),
                     cache_book_file=str(cache_presets.CACHE_BOOKS / "Wan2.1" / "book.json"))
    cache_presets.configure_cache_book(args, "wan_module")
    assert Path(args.cache_book_file) == cache_presets.CACHE_BOOKS / "Wan2.1" / "book.json"
    assert args.use_invardiff is False


@pytest.mark.parametrize("policy,model", [
    ("dit_module", "DiT"), ("dit_step_layer", "DiT"),
    ("flux_module", "FLUX"), ("flux_magcache_hybrid", "FLUX"),
    ("wan_module", "Wan2.1"), ("wan_seacache_hybrid", "Wan2.1"),
    ("hunyuan_module", "HunyuanVideo"), ("hunyuan_step_layer", "HunyuanVideo"),
])
def test_default_books_are_grouped_by_model(policy, model):
    args = namespace()
    cache_presets.configure_cache_book(args, policy)
    assert Path(args.cache_book_path) == cache_presets.CACHE_BOOKS / model


@pytest.mark.parametrize("key,value", [
    ("sample_steps", 30), ("size", "1280*720"), ("frame_num", 121),
    ("nonskip_rate", .06), ("self_attn_thres", .6), ("ffn_thres", .4),
    ("sample_solver", "dpm++"), ("sample_shift", 7), ("sample_guide_scale", 6),
    ("num_analysis", 2), ("magcache_K", 5),
])
def test_different_configurations_have_different_default_names(key, value):
    original = namespace(sample_steps=50, size="832*480", frame_num=81,
                         nonskip_rate=.04, self_attn_thres=.5, ffn_thres=.2,
                         sample_solver="unipc", sample_shift=5,
                         sample_guide_scale=5, num_analysis=1, magcache_K=4)
    changed = argparse.Namespace(**vars(original))
    setattr(changed, key, value)
    assert cache_presets.cache_book_filename(original, "wan_module") != cache_presets.cache_book_filename(changed, "wan_module")


def test_demo_seed_prompt_and_output_do_not_change_policy_name():
    first = namespace(seed=42, prompt="dog", output_dir="outputs/a", sample_steps=50)
    second = namespace(seed=43, prompt="cat", output_dir="outputs/b", sample_steps=50)
    assert cache_presets.cache_book_filename(first, "wan_module") == cache_presets.cache_book_filename(second, "wan_module")
