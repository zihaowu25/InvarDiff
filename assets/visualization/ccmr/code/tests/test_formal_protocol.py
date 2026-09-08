from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from aggregate_ccmr import _rho_stability  # noqa: E402
from common import (  # noqa: E402
    ccmr_ratio,
    deterministic_derangements,
    deterministic_unique_subsets,
    hierarchical_bootstrap,
    load_yaml,
    online_rho,
)
from collect_dit_ccmr import _detach_conditional_clone  # noqa: E402
from compose_main_figure import _subset_hierarchical_summary  # noqa: E402
from formal_protocol import (  # noqa: E402
    rho_consistency,
    tolerance_check,
    validate_boundary,
    validate_derangements,
    validate_flux_pair_selection,
    validate_test_log,
    validate_valid_rho_finite,
)
from validate_artifacts import validate  # noqa: E402
from validate_smoke_artifacts import validate_smokes  # noqa: E402


def test_yaml_duplicate_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("value: 1\nvalue: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_yaml(path)


def test_one_hundred_unique_derangements_have_no_fixed_points():
    values = deterministic_derangements(16, 100, 2027)
    assert len(values) == len({tuple(value) for value in values}) == 100
    assert all(all(index != target for index, target in enumerate(value)) for value in values)


def test_dit_conditional_clone_does_not_share_cfg_storage():
    cfg_batch = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4)
    sliced = cfg_batch[:2].detach().contiguous()
    cloned = _detach_conditional_clone(cfg_batch, 2)
    assert sliced.untyped_storage().data_ptr() == cfg_batch.untyped_storage().data_ptr()
    assert cloned.untyped_storage().data_ptr() != cfg_batch.untyped_storage().data_ptr()
    cfg_batch.zero_()
    assert torch.count_nonzero(cloned) > 0


def test_derangement_validation_is_scoped_per_cell():
    rows = []
    permutations = deterministic_derangements(16, 100, 7)
    for step in (1, 2):
        for trial, permutation in enumerate(permutations):
            rows.append({"model": "dit", "seed": 0, "module_family": "dit", "module_name": "msa", "layer_idx": 0, "step_idx": step, "shuffle_trial": trial, "permutation": json.dumps(permutation), "valid": True})
    assert validate_derangements(rows) == []


def test_ccmr_ratio_uses_additive_denominator():
    assert ccmr_ratio(2.0, 1.0, 0.5) == 0.4


def test_online_rho_calls_live_distance_function_twice():
    calls = []
    def live(a, b):
        calls.append((a.clone(), b.clone()))
        return (b - a + 1e-8).abs().sum(dtype=torch.float32)
    a, b, c = torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([3.0])
    assert online_rho(a, b, c, live) == pytest.approx(2.0)
    assert len(calls) == 2


def test_rho_tolerance_executes_pass_and_fail():
    good = [{"rho_clean": 1.0, "rho_code": 1.0 + 1e-7, "valid": True}]
    bad = [{"rho_clean": 1.0, "rho_code": 1.1, "valid": True}]
    assert tolerance_check(good, 1e-4, 1e-6)[0]
    assert not tolerance_check(bad, 1e-4, 1e-6)[0]


def test_valid_rho_requires_both_finite_values():
    assert validate_valid_rho_finite([])
    assert validate_valid_rho_finite([{"rho_clean": 1.0, "rho_code": "", "valid": True}])
    assert validate_valid_rho_finite([{"rho_clean": 1.0, "rho_code": 1.0, "valid": True}]) == []


def test_invalid_boundaries_are_explicit_and_excluded():
    rows = []
    for index in range(5):
        rows.append({"seed": 0, "condition_id": "a", "module_family": "dit", "module_name": "msa", "layer_idx": 0, "score_step_idx": index, "rho_clean": 1.0, "rho_code": 1.0, "valid": 1 <= index <= 3})
    assert validate_boundary(rows, 5) == []
    assert rho_consistency(rows)["valid_cells"] == 3


def test_empty_boundary_and_derangement_inputs_do_not_pass():
    assert validate_boundary([], 5)
    assert validate_derangements([], 100)


def test_unique_subsets_enumerate_small_spaces_without_duplicates():
    values = deterministic_unique_subsets(["a", "b", "c"], 2, 100, 1)
    assert values == [("a", "b"), ("a", "c"), ("b", "c")]


def _rho_fixture(scope="condition"):
    rows = []
    for condition, offset in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        for step in (1, 2, 3):
            rows.append({"source_run": "run", "model": "dit", "seed": 0, "condition_id": condition, "rho_scope": scope, "module_family": "dit", "module_name": "msa", "layer_idx": 0, "score_step_idx": step, "rho_clean": step + offset, "valid": True})
    return rows


def test_primary_subset_aggregation_is_mean_and_median_is_separate():
    _, rows = _rho_stability(_rho_fixture(), [], {"subset_sizes": {"dit": [1]}, "subset_trials": 100, "cache_fractions": [0.3]})
    assert {row["aggregation_method"] for row in rows} == {"mean", "median"}
    assert all(row["is_primary"] == (row["aggregation_method"] == "mean") for row in rows)


def test_pair_difference_scope_is_rejected_from_panel_d_statistics():
    stability, subsets = _rho_stability(_rho_fixture("pair_difference"), [], {"subset_sizes": {"dit": [1]}})
    assert stability == [] and subsets == []


def test_subset_condition_ids_are_fixed_across_entire_cache_book():
    _, rows = _rho_stability(_rho_fixture(), [], {"subset_sizes": {"dit": [2]}, "subset_trials": 100, "cache_fractions": [0.3], "shuffle_seed": 9})
    by_trial = {}
    for row in rows:
        by_trial.setdefault(row["trial"], row["selected_condition_ids"])
        assert by_trial[row["trial"]] == row["selected_condition_ids"]


def test_rank_metrics_are_not_duplicated_by_cache_fraction():
    _, rows = _rho_stability(_rho_fixture(), [], {"subset_sizes": {"dit": [1]}, "subset_trials": 100, "cache_fractions": [0.1, 0.2, 0.3, 0.5]})
    rank_rows = [row for row in rows if row["metric_scope"] == "rank" and row["aggregation_method"] == "mean"]
    assert len(rank_rows) == 3
    assert all(row["cache_fraction"] is None for row in rank_rows)


def test_hierarchical_bootstrap_uses_seed_then_cluster():
    rows = []
    for seed in range(3):
        for cluster in range(2):
            for _cell in range(100):
                rows.append({"seed": seed, "cluster_id": cluster, "value": seed + cluster})
    result = hierarchical_bootstrap(rows, "value", trials=100, random_seed=2)
    assert len(result["seed_points"]) == 3
    assert result["estimate"] == pytest.approx(1.5)


def test_subset_uncertainty_resamples_seed_then_subset():
    rows = []
    for seed in (0, 1, 2):
        for trial in range(10):
            rows.append({"seed": seed, "spearman": seed * 0.1 + trial * 0.001})
    result = _subset_hierarchical_summary(rows, "spearman", trials=100, seed=9)
    assert len(result["seed_points"]) == 3
    assert result["p025"] <= result["mean"] <= result["p975"]


@pytest.mark.parametrize("text", ["1 failed", "1 error", "no tests ran", "collected 3 items"])
def test_test_log_rejects_missing_or_failed_result(tmp_path, text):
    path = tmp_path / "tests.log"; path.write_text(text, encoding="utf-8")
    assert validate_test_log(path)


def test_test_log_accepts_explicit_pass_count(tmp_path):
    path = tmp_path / "tests.log"; path.write_text("34 passed in 1.2s\n", encoding="utf-8")
    assert validate_test_log(path) == []


def test_flux_pair_selection_detects_seed_hash_duplicates_and_illegal_pairs():
    config = {"pair_selection_seed": 7, "num_condition_pairs": 2}
    conditions = [{"id": value} for value in ("a", "b", "c")]
    bad = {"seed": 8, "pairs": [["a", "a"], ["a", "a"]], "pair_selection_hash": "bad"}
    failures = validate_flux_pair_selection(config, conditions, bad)
    assert len(failures) >= 3


def test_formal_composer_fails_without_complete_coverage(tmp_path):
    script = CODE / "compose_main_figure.py"
    result = subprocess.run([sys.executable, str(script), "--combined", str(tmp_path / "combined"), "--tables-dir", str(tmp_path / "tables"), "--output-dir", str(tmp_path / "paper"), "--dit-run", str(tmp_path / "dit"), "--flux-pair-run", str(tmp_path / "pair"), "--flux-rho-run", str(tmp_path / "rho"), "--tests-log", str(tmp_path / "tests.log")], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "paper" / "fig_ccmr_main.png").exists()


def test_formal_validator_marks_missing_inputs_not_run(tmp_path):
    result = validate(tmp_path / "dit", tmp_path / "pair", tmp_path / "rho", tmp_path / "combined", tmp_path / "tests.log")
    assert not result["passed"]
    assert result["checks"]["dit.atomic"]["status"] == "not_run"
    assert not result["checks"]["dit.boundaries"]["passed"]
    assert not result["checks"]["flux_pairwise.swap"]["passed"]


def test_smoke_validator_fails_all_missing_runs(tmp_path):
    result = validate_smokes(tmp_path / "dit", tmp_path / "pair", tmp_path / "rho")
    assert not result["passed"]
    assert result["checks"]["dit.atomic"]["status"] == "failed"
    # The partial-progress manifest must remain strict-JSON serializable.
    json.dumps(result, allow_nan=False)


def test_formal_command_file_requires_explicit_approval():
    script = CODE.parents[2] / "ccmr_formal" / "commands_v2_formal.sh"
    result = subprocess.run(["bash", str(script)], env={}, capture_output=True, text=True)
    assert result.returncode == 2
    assert "locked" in result.stderr


def test_plot_and_export_modules_import_without_model_packages():
    code = "import compose_main_figure, export_paper_tables, validate_artifacts"
    result = subprocess.run([sys.executable, "-c", code], cwd=CODE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
