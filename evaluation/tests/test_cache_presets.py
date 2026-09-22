import argparse
import json

import cache_presets


def test_explicit_threshold_overrides_selected_preset(tmp_path, monkeypatch):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "policies": {
            "example": {
                "presets": {
                    "default": {"thresholds": {"attn_thres": 0.6, "ff_thres": 0.2}}
                }
            }
        }
    }))
    monkeypatch.setattr(cache_presets, "PRESET_PATH", path)
    monkeypatch.setattr(cache_presets, "load_presets", lambda path=path: json.loads(path.read_text())["policies"])
    args = argparse.Namespace(
        cache_preset="default",
        attn_thres=0.7,
        ff_thres=0.0,
        magcache_thresh=0.24,
    )
    cache_presets.apply_preset(
        args,
        "example",
        {"--attn-thres": "attn_thres", "--ff-thres": "ff_thres"},
        ["--attn-thres", "0.7"],
    )
    assert args.attn_thres == 0.7
    assert args.ff_thres == 0.2
    assert args.magcache_thresh == 0.24
    assert args.cache_resolved_thresholds == {
        "attn_thres": 0.7,
        "ff_thres": 0.2,
    }


def test_repository_module_presets_keep_valid_active_thresholds():
    policies = cache_presets.load_presets()
    for policy in ("dit_module", "flux_module", "wan_module", "hunyuan_module"):
        presets = policies[policy]["presets"]
        for tier, item in presets.items():
            active = [value for value in item["thresholds"].values() if value > 0]
            assert all(0 < value <= 1 for value in active)
            if policy != "dit_module":
                assert len(set(active)) > 1, f"uniform active thresholds in {policy}:{tier}"


def test_default_module_presets_match_selected_configs():
    policies = cache_presets.load_presets()
    expected = {
        "dit_module": {"msa_thres": 0.55, "mlp_thres": 0.55},
        "flux_module": {
            "attn_thres": 0.70,
            "context_attn_thres": 0.70,
            "ff_thres": 0.30,
            "context_ff_thres": 0.23,
            "single_attn_thres": 0.50,
            "single_mlp_thres": 0.02,
        },
        "wan_module": {
            "step_thres": 0.00,
            "self_attn_thres": 0.50,
            "cross_attn_thres": 0.35,
            "ffn_thres": 0.20,
        },
        "hunyuan_module": {
            "double_img_attn_thres": 0.90,
            "double_txt_attn_thres": 0.45,
            "double_img_mlp_thres": 0.04,
            "double_txt_mlp_thres": 0.12,
            "single_attn_thres": 0.00,
            "single_mlp_thres": 0.00,
        },
    }
    for policy, thresholds in expected.items():
        assert list(policies[policy]["presets"]) == ["default"]
        assert policies[policy]["presets"]["default"]["thresholds"] == thresholds


def test_preset_label_is_not_a_cache_book_execution_setting():
    saved = {"cache_preset": "legacy", "attn_thres": 0.7, "steps": 28}
    expected = {"cache_preset": "default", "attn_thres": 0.7, "steps": 28}
    assert cache_presets.same_execution_config(saved, expected)
    assert not cache_presets.same_execution_config(
        saved, {**expected, "attn_thres": 0.8}
    )


def test_single_module_configuration_is_the_cli_default():
    parser = argparse.ArgumentParser()
    cache_presets.add_preset_argument(
        parser, "dit_module", tiers=("default",)
    )
    assert parser.parse_args([]).cache_preset == "default"


def test_external_hybrid_policy_is_one_fixed_tier():
    policies = cache_presets.load_presets()
    for policy in (
        "flux_hybrid",
        "wan_hybrid",
        "wan_teacache_hybrid",
        "hunyuan_hybrid",
    ):
        assert list(policies[policy]["presets"]) == ["hybrid"]


def test_native_validated_hybrid_defaults_are_policy_specific():
    policies = cache_presets.load_presets()
    expected = {
        "flux_magcache_hybrid": {
            "attn_thres": 0.30,
            "context_attn_thres": 0.30,
            "ff_thres": 0.00,
            "context_ff_thres": 0.00,
            "single_attn_thres": 0.30,
            "single_mlp_thres": 0.00,
        },
        "flux_seacache_hybrid": {
            "attn_thres": 0.30,
            "context_attn_thres": 0.30,
            "ff_thres": 0.20,
            "context_ff_thres": 0.20,
            "single_attn_thres": 0.30,
            "single_mlp_thres": 0.20,
        },
        "wan_magcache_hybrid": {
            "self_attn_thres": 0.30,
            "cross_attn_thres": 0.30,
            "ffn_thres": 0.00,
        },
        "wan_seacache_hybrid": {
            "self_attn_thres": 0.30,
            "cross_attn_thres": 0.30,
            "ffn_thres": 0.00,
        },
        "hunyuan_seacache_hybrid": {
            "double_img_attn_thres": 0.60,
            "double_txt_attn_thres": 0.50,
            "double_img_mlp_thres": 0.00,
            "double_txt_mlp_thres": 0.00,
            "single_attn_thres": 0.00,
            "single_mlp_thres": 0.00,
        },
    }
    for policy, thresholds in expected.items():
        assert policies[policy]["presets"]["hybrid"]["thresholds"] == thresholds


def test_wan_teacache_uses_a_method_specific_module_vector():
    policies = cache_presets.load_presets()
    tea = policies["wan_teacache_hybrid"]["presets"]["hybrid"]["thresholds"]
    shared = policies["wan_hybrid"]["presets"]["hybrid"]["thresholds"]
    assert tea == {
        "self_attn_thres": 0.40,
        "cross_attn_thres": 0.50,
        "ffn_thres": 0.60,
    }
    assert shared == {
        "self_attn_thres": 0.10,
        "cross_attn_thres": 0.11,
        "ffn_thres": 0.12,
    }
