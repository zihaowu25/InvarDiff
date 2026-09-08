from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_ccmr import _recompute_pairwise_ccmr, _recompute_pairwise_time_gap, _rho_stability, validate_aggregate_inputs  # noqa: E402
from common import all_unordered_pairs, centered_variance, compact_gap_delta, pair_energy  # noqa: E402


def test_compact_gap_delta_does_not_treat_tokens_as_conditions():
    current = torch.tensor([[3.0, 5.0], [7.0, 9.0]])
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(compact_gap_delta(current, previous), current - previous)


def test_pairwise_aggregate_is_one_row_and_matches_exact_variance():
    previous = torch.tensor([[0.0, 1.0], [1.0, 3.0], [4.0, 2.0], [8.0, 5.0]])
    current = previous + torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, -1.0], [-1.0, 1.0]])
    rows = []
    distance = []
    for i, j in all_unordered_pairs(4):
        rows.append({
            "source_run": "flux", "model": "flux", "seed": 0,
            "estimator": "pairwise", "num_conditions": 4,
            "module_family": "double", "module_name": "attn", "layer_idx": 0, "step_idx": 2,
            "v_raw_prev": float(pair_energy(previous[i], previous[j]) / 2.0),
        })
        distance.append({
            "source_run": "flux", "model": "flux", "seed": 0,
            "module_family": "double", "module_name": "attn", "layer_idx": 0, "step_idx": 2,
            "raw_pair_distance": float(pair_energy(current[i], current[j])),
            "raw_prev_pair_distance": float(pair_energy(previous[i], previous[j])),
            "diff_pair_distance": float(pair_energy(current[i] - previous[i], current[j] - previous[j])),
        })
    result = _recompute_pairwise_ccmr(rows, distance)
    assert len(result) == 1
    assert abs(result[0]["v_raw"] - float(centered_variance(current))) < 1e-7
    assert abs(result[0]["v_raw_prev"] - float(centered_variance(previous))) < 1e-7
    assert abs(result[0]["v_diff"] - float(centered_variance(current - previous))) < 1e-7
    assert result[0]["input_pair_rows"] == 6


def test_subset_selection_uses_one_fixed_condition_set_per_trial():
    rho_rows = []
    for condition in ("a", "b", "c"):
        offset = {"a": 0.0, "b": 0.2, "c": 0.4}[condition]
        for step in (1, 2):
            rho_rows.append({
                "source_run": "dit_run", "model": "dit", "seed": 0,
                "condition_id": condition, "rho_scope": "condition",
                "module_family": "dit", "module_name": "msa", "layer_idx": 0,
                "score_step_idx": step, "rho_clean": 1.0 + offset + step * 0.01,
                "valid": True,
            })
    stability, subsets = _rho_stability(
        rho_rows,
        [],
        {"subset_sizes": {"dit": [2]}, "subset_trials": 4, "cache_fractions": [0.1, 0.2], "shuffle_seed": 7},
    )
    assert len(stability) == 2
    # C(3, 2)=3 unique subsets; mean/median each emit one rank row and two
    # overlap rows. No duplicate random trials are manufactured.
    assert len(subsets) == 18
    selected_by_trial = {}
    for row in subsets:
        selected_by_trial.setdefault(row["trial"], row["selected_condition_ids"])
        assert selected_by_trial[row["trial"]] == row["selected_condition_ids"]
        assert row["num_reference_conditions"] == 3


def test_pairwise_time_gap_aggregates_energy_before_gain():
    rows = []
    for current, previous, difference in ((2.0, 4.0, 1.0), (6.0, 8.0, 3.0), (10.0, 12.0, 5.0)):
        rows.append({
            "source_run": "flux_run", "model": "flux", "seed": 0,
            "estimator": "pairwise", "num_conditions": 4,
            "module_family": "double", "module_name": "attn", "layer_idx": 0,
            "step_idx": 3, "time_gap": 1,
            "raw_current_pair_distance": current,
            "raw_previous_pair_distance": previous,
            "diff_pair_distance": difference,
            "v_raw_base": 0.0, "v_diff_gap": 0.0,
            "r_ccmr_gap": 0.0, "g_ccmr_gap_db": 0.0,
        })
    result = _recompute_pairwise_time_gap(rows)
    assert len(result) == 1
    v_current = (3.0 / 4.0) * (2.0 + 6.0 + 10.0) / 3.0
    v_previous = (3.0 / 4.0) * (4.0 + 8.0 + 12.0) / 3.0
    v_difference = (3.0 / 4.0) * (1.0 + 3.0 + 5.0) / 3.0
    v_base = 0.5 * (v_current + v_previous)
    assert abs(result[0]["v_raw_base"] - v_base) < 1e-12
    assert abs(result[0]["v_diff_gap"] - v_difference) < 1e-12
    assert abs(result[0]["g_ccmr_gap_db"] - 10.0 * math.log10(v_base / v_difference)) < 1e-9


def test_formal_aggregate_rejects_aborted_or_unregistered_runs(tmp_path):
    aborted = tmp_path / "flux_pairwise_formal24_aborted_20260908"
    aborted.mkdir()
    (aborted / "aborted_protocol_manifest.json").write_text("{}", encoding="utf-8")
    config = {"allowed_run_ids": ["dit", "pair", "rho"]}
    try:
        validate_aggregate_inputs([aborted], config)
    except ValueError as exc:
        assert "preregistration" in str(exc) or "Aborted" in str(exc)
    else:
        raise AssertionError("aborted aggregate input was accepted")
