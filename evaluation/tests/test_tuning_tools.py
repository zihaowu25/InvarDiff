import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coarse_elbow_requires_absolute_and_relative_jump():
    module = _load("analyze_threshold_sweep", "evaluation/analyze_threshold_sweep.py")
    frame = pd.DataFrame(
        {
            "threshold": [0.0, 0.1, 0.2, 0.3, 0.4],
            "lpips": [0.0, 0.01, 0.02, 0.03, 0.08],
        }
    )
    assert module.detect_elbow(frame, 0.1) == 0.4


def test_fine_elbow_uses_fine_absolute_increment():
    module = _load("analyze_threshold_sweep_fine", "evaluation/analyze_threshold_sweep.py")
    frame = pd.DataFrame(
        {
            "threshold": [0.10, 0.11, 0.12, 0.13],
            "lpips": [0.098, 0.087, 0.095, 0.123],
        }
    )
    assert module.detect_elbow(frame, 0.01) == 0.12


def test_pairwise_analyzer_matches_two_condition_variance():
    module = _load("hunyuan_ccmr_pair", "evaluation/hunyuan_ccmr_pair.py")
    analyzer = module.PairwiseCCMRAnalyzer(
        steps=2, depths={"double.img_attn": 1}, do_cfg=False
    )
    first = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    second = torch.tensor([[2.0, 4.0], [2.0, 0.0]])
    analyzer.update_module(0, "double.img_attn", 0, first)
    analyzer.update_module(1, "double.img_attn", 0, second)
    row = analyzer.rows[0]
    assert abs(row["v_raw_prev"] - 1.0) < 1e-6
    assert abs(row["v_raw"] - 2.0) < 1e-6
    assert abs(row["v_diff"] - 1.0) < 1e-6


def test_paired_bootstrap_operates_on_deltas():
    module = _load("aggregate_paired_delta", "evaluation/aggregate_paired_delta.py")
    baseline = torch.tensor([0.10, 0.20, 0.30]).numpy()
    candidate = torch.tensor([0.15, 0.25, 0.35]).numpy()
    mean, upper = module.paired_bootstrap_upper(
        baseline, candidate, samples=2000, seed=7
    )
    assert abs(mean - 0.05) < 1e-6
    assert abs(upper - 0.05) < 1e-6


def test_video_confirmation_statistics_include_tail():
    module = _load(
        "aggregate_video_confirmation",
        "evaluation/aggregate_video_confirmation.py",
    )
    result = module._stat([0.0, 0.1, 0.2, 0.9])
    assert abs(result["mean"] - 0.3) < 1e-12
    assert result["p95"] > 0.7
    assert result["max"] == 0.9


def test_horizontal_grid_split_removes_torchvision_padding():
    module = _load("paired_lpips_grid", "evaluation/paired_lpips.py")
    grid = torch.zeros(1, 4, 10, 3, dtype=torch.uint8).numpy()
    grid[:, 1:3, 1:3] = 10
    grid[:, 1:3, 4:6] = 20
    grid[:, 1:3, 7:9] = 30
    cells = module._split_horizontal_grid(grid, count=3, padding=1)
    assert cells.shape == (3, 2, 2, 3)
    assert cells[:, 0, 0, 0].tolist() == [10, 20, 30]


def test_reload_command_preserves_external_candidate_settings(tmp_path):
    module = _load(
        "reload_hunyuan_candidates",
        "evaluation/reload_hunyuan_candidates.py",
    )
    candidate = tmp_path / "step_q03"
    candidate.mkdir()
    thresholds = {
        "double.img_attn": 0.9,
        "double.txt_attn": 0.01,
        "double.img_mlp": 0.0,
        "double.txt_mlp": 0.32,
        "single.attn": 0.0,
        "single.mlp": 0.0,
    }
    (candidate / "summary.json").write_text(json.dumps({
        "policy": "step-layer",
        "thresholds": thresholds,
        "step_threshold": 0.03,
        "frames": 17,
        "steps": 50,
        "seed": 2027,
        "prompt_set": "screen",
    }))
    command = module._candidate_command(candidate, tmp_path / "reference", "6")
    assert "--reuse-cache-book" in command
    assert command[command.index("--gpu") + 1] == "6"
    assert command[command.index("--step-threshold") + 1] == "0.03"
    assert json.loads(command[command.index("--thresholds") + 1]) == thresholds


def test_hunyuan_cache_book_rejects_threshold_mismatch(tmp_path, monkeypatch):
    """A book calibrated for one vector must never load for another vector."""
    official = ROOT.parent / "open-source" / "HunyuanVideo-1.5"
    if not official.is_dir():
        pytest.skip("official HunyuanVideo-1.5 checkout is not available")
    sys.path.insert(0, str(official))
    try:
        module = _load("hunyuan_cache_mismatch", "HunyuanVideo/sample_hunyuan.py")
    finally:
        sys.path.pop(0)
    expected = {
        "transformer_version": "v1", "task": "t2v", "width": 1280,
        "height": 720, "frames": 17, "steps": 50, "nonskip": 0.1,
        "module_thresholds": {name: 0.1 for name in module.MODULES},
        "seed": 2027, "deterministic_attention": False,
        "double_blocks": 1, "single_blocks": 1,
    }
    payload = {
        "cache_version": module.CACHE_BOOK_VERSION,
        "cache_scope": module.CACHE_SCOPE,
        "policy": module.POLICY_VARIANT,
        "rate_method": module.RATE_METHOD,
        "config": dict(expected, module_thresholds={name: 0.2 for name in module.MODULES}),
    }
    path = tmp_path / "book.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(module, "rank0", lambda: True)
    monkeypatch.setattr(module, "_broadcast_object", lambda value: value)
    with pytest.raises(ValueError, match="module_thresholds"):
        module._load_books(path, expected, steps=50,
                           depths={name: 1 for name in module.MODULES})
