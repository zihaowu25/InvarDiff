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
                    "fast": {"thresholds": {"attn_thres": 0.6, "ff_thres": 0.2}}
                }
            }
        }
    }))
    monkeypatch.setattr(cache_presets, "PRESET_PATH", path)
    monkeypatch.setattr(cache_presets, "load_presets", lambda path=path: json.loads(path.read_text())["policies"])
    args = argparse.Namespace(
        cache_preset="fast",
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


def test_repository_module_presets_keep_independent_active_thresholds():
    policies = cache_presets.load_presets()
    for policy in ("dit_module", "flux_module", "wan_module", "hunyuan_module"):
        presets = policies[policy]["presets"]
        for tier, item in presets.items():
            active = [value for value in item["thresholds"].values() if value > 0]
            assert len(active) == len(set(active)), f"uniform active thresholds in {policy}:{tier}"


def test_external_hybrid_policy_is_one_fixed_tier():
    policies = cache_presets.load_presets()
    for policy in (
        "flux_hybrid",
        "wan_hybrid",
        "wan_teacache_hybrid",
        "hunyuan_hybrid",
    ):
        assert list(policies[policy]["presets"]) == ["hybrid"]


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
