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
    balanced_cycle_selection,
    ccmr_ratio,
    deterministic_derangements,
    deterministic_unique_subsets,
    hierarchical_bootstrap,
    load_yaml,
    load_attempt_history,
    online_rho,
)
from collect_dit_ccmr import _detach_conditional_clone  # noqa: E402
from compose_main_figure import _subset_hierarchical_summary  # noqa: E402
from formal_protocol import (  # noqa: E402
    rho_consistency,
    alignment_hierarchical_summary,
    tolerance_check,
    validate_boundary,
    validate_derangements,
    validate_flux_pair_selection,
    validate_flux_swaps,
    subset_hierarchical_summary,
    validate_test_log,
    validate_valid_rho_finite,
)
from validate_artifacts import validate  # noqa: E402
from validate_artifacts import (  # noqa: E402
    FORMAL_COLLECTION_COMMIT_KEYS,
    FORMAL_COMBINED_COUNTS,
    _validate_formal12_shards,
    validate_collection_commit_map,
)
from validate_smoke_artifacts import validate_smokes  # noqa: E402
from validate_smoke_artifacts import _config_hashes, _protocol, _runtime_metadata  # noqa: E402
from export_paper_tables import _rho_rows  # noqa: E402


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


def test_balanced_cycle_is_deterministic_unique_connected_and_degree_two():
    ids = [f"p{i:02d}" for i in range(12)]
    first = balanced_cycle_selection(ids, 2027)
    second = balanced_cycle_selection(ids, 2027)
    other = balanced_cycle_selection(ids, 2028)
    assert first == second
    assert first["prompt_order"] != other["prompt_order"]
    assert len(first["pairs"]) == len({tuple(pair) for pair in first["pairs"]}) == 12
    assert all(left != right for left, right in first["pairs"])
    assert set(item for pair in first["pairs"] for item in pair) == set(ids)
    assert first["prompt_degrees"] == {item: 2 for item in ids}
    assert first["connected"] is True


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


def test_subset_macro_does_not_treat_repeated_family_rows_as_iid():
    rows = []
    for seed in range(3):
        for trial, selected in enumerate(('[' + str(trial) + ']') for trial in range(4)):
            for family, offset in (("msa", 0.0), ("mlp", 0.2)):
                rows.append({
                    "source_run": "run", "seed": seed, "trial": trial,
                    "selected_condition_ids": selected, "module_family": family,
                    "spearman": 0.4 + seed * 0.1 + trial * 0.01 + offset,
                })
    base = subset_hierarchical_summary(rows, "spearman", trials=200, random_seed=11)
    duplicated = subset_hierarchical_summary(rows + [dict(row) for row in rows], "spearman", trials=200, random_seed=11)
    assert base == duplicated
    assert base["num_seeds"] == 3
    assert base["num_unique_subsets"] == 12


def test_paper_rho_table_uses_hierarchical_counts_and_model_macro():
    rows = []
    for seed in range(3):
        for trial in range(2):
            for family in ("dit.msa", "dit.mlp"):
                rows.append({
                    "model": "dit", "source_run": "run", "seed": seed,
                    "module_family": family, "subset_size": 1, "trial": trial,
                    "selected_condition_ids": json.dumps([f"c{trial}"]),
                    "metric_scope": "rank", "cache_fraction": None,
                    "aggregation_method": "mean", "is_primary": True,
                    "valid": True, "spearman": 0.5 + seed * 0.1,
                })
    result = _rho_rows(rows)
    assert {row["aggregation"] for row in result} == {"module_family", "model_macro"}
    assert all(row["num_seeds"] == 3 for row in result)
    assert all(row["num_unique_subsets"] == 6 for row in result)
    assert all("n" not in row for row in result)


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


def test_balanced_cycle_validator_rejects_degree_disconnect_missing_and_hash():
    ids = [f"p{i:02d}" for i in range(12)]
    config = {"pair_selection_mode": "balanced_cycle", "pair_selection_seed": 2027, "num_condition_pairs": 12}
    conditions = [{"id": item} for item in ids]
    good = balanced_cycle_selection(ids, 2027)
    assert validate_flux_pair_selection(config, conditions, good) == []
    bad = dict(good)
    bad["pairs"] = good["pairs"][:-1] + [good["pairs"][0]]
    bad["pair_selection_hash"] = "wrong"
    failures = validate_flux_pair_selection(config, conditions, bad)
    assert any("duplicate" in item for item in failures)
    assert any("degree" in item for item in failures)
    assert any("hash" in item for item in failures)


def test_formal12_validator_requires_36_complete_shards(tmp_path):
    selection = balanced_cycle_selection([f"p{i:02d}" for i in range(12)], 2027)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"shards": {}}), encoding="utf-8")
    failures = _validate_formal12_shards(tmp_path, selection)
    assert any("0/36" in item for item in failures)
    assert any("missing formal12 shard" in item for item in failures)


def test_t02_combined_counts_and_collection_commit_map_are_preregistered():
    assert FORMAL_COMBINED_COUNTS["alignment_control"] == 1_519_744
    assert FORMAL_COMBINED_COUNTS["ccmr_metrics"] == 26_768
    assert FORMAL_COLLECTION_COMMIT_KEYS == {
        "dit_512", "flux_pairwise_formal12", "flux_condition_rho",
    }


def test_flux_swap_requires_one_exact_swap_per_valid_cell():
    base = {"model": "flux", "valid": True, "seed": 0, "pair_id": "a__b",
            "module_family": "double", "module_name": "attn", "layer_idx": 0, "step_idx": 1}
    assert validate_flux_swaps([{**base, "permutation": "[1,0]"}]) == []
    assert validate_flux_swaps([{**base, "permutation": "[0,1]"}])
    assert validate_flux_swaps([{**base, "permutation": "[1,0]"}, {**base, "permutation": "[1,0]"}])


def test_retry_history_recovers_legacy_failure(tmp_path):
    marker = tmp_path / "manifest.json"
    marker.write_text(json.dumps({
        "status": "failed", "resolved_config_hash": "cfg", "started_at": "a",
        "failed_at": "b", "wall_time_s": 1.0, "oom": True,
        "peak_allocated_gib": 2.0, "peak_reserved_gib": 3.0,
        "commit": "abc", "error_type": "OutOfMemoryError", "error": "oom",
    }), encoding="utf-8")
    retries, history = load_attempt_history(marker, "cfg")
    assert retries == 1 and len(history) == 1 and history[0]["status"] == "failed"
    assert load_attempt_history(marker, "different") == (0, [])


def test_smoke_runtime_protocol_and_hash_failures(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({
        "resolved_config_hash": "bad", "shards": {"x": {
            "status": "failed", "dirty_status": " M code.py", "paper_eligible": True,
            "retry_count": 1, "cache_enabled": True, "decode_output": True,
            "oom": True, "wall_time_s": -1, "peak_allocated_gib": -1,
            "peak_reserved_gib": -1,
        }}
    }), encoding="utf-8")
    (run / "memory.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    assert _runtime_metadata(run)
    assert _config_hashes(run)
    assert _protocol({}, "dit")


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


def test_alignment_hierarchy_macro_averages_families_before_resampling():
    rows = [
        {"source_run": "r", "seed": 0, "cluster_id": "c", "module_family": "a",
         "g_aligned_db": 2.0, "g_shuffled_db": 1.0},
        {"source_run": "r", "seed": 0, "cluster_id": "c", "module_family": "b",
         "g_aligned_db": 8.0, "g_shuffled_db": 4.0},
    ]
    baseline = alignment_hierarchical_summary(rows, trials=10)
    duplicated = alignment_hierarchical_summary(rows + [rows[0]] * 20, trials=10)
    assert baseline is not None and duplicated is not None
    assert baseline["aligned"] == duplicated["aligned"] == 5.0
    assert baseline["shuffled"] == duplicated["shuffled"] == 2.5


def test_collection_commit_map_rejects_missing_and_mismatched_entries(tmp_path):
    names = ("dit_512_formal_v2", "flux_pairwise_formal12_v2", "flux_rho_formal_v2")
    runs = []
    for name in names:
        run = tmp_path / name
        run.mkdir()
        (run / "run_manifest.json").write_text(
            json.dumps({"shards": {"0": {"commit": "good"}}}), encoding="utf-8",
        )
        runs.append(run)
    valid = {"collection_commits": {
        "dit_512": "good", "flux_pairwise_formal12": "good", "flux_condition_rho": "good",
    }}
    assert validate_collection_commit_map(valid, runs) == []
    missing = {"collection_commits": {"dit_512": "good"}}
    assert validate_collection_commit_map(missing, runs)
    mismatch = json.loads(json.dumps(valid))
    mismatch["collection_commits"]["flux_pairwise_formal12"] = "bad"
    assert validate_collection_commit_map(mismatch, runs)


def test_plot_and_export_modules_import_without_model_packages():
    code = "import compose_main_figure, export_paper_tables, validate_artifacts"
    result = subprocess.run([sys.executable, "-c", code], cwd=CODE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
