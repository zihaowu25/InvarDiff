#!/usr/bin/env python3
"""Hard gate for CCMR formal artifacts; never falls back to pilot data."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from common import read_rows, save_json_atomic, sha256_file, utc_now
from formal_protocol import (
    DIT_MODULES,
    FLUX_MODULES,
    finite,
    is_true,
    table_rows,
    tolerance_check,
    validate_atomic_manifest,
    validate_boundary,
    validate_derangements,
    validate_flux_pair_selection,
    validate_flux_swaps,
    validate_modules,
    validate_test_log,
    validate_valid_rho_finite,
)

FORMAL_COMBINED_COUNTS = {
    "ccmr_metrics": 26_768,
    "alignment_control": 1_519_744,
    "rho_per_condition": 377_216,
}
FORMAL_COLLECTION_COMMIT_KEYS = {
    "dit_512", "flux_pairwise_formal12", "flux_condition_rho",
}


def _config(run: Path) -> dict[str, Any]:
    path = run / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _finite_valid(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    if not rows:
        return ["finite-value validation has no rows"]
    failures = []
    for index, row in enumerate(rows):
        if not is_true(row.get("valid", True)):
            continue
        for field in fields:
            if not finite(row.get(field)):
                failures.append(f"valid row {index} has non-finite {field}")
                if len(failures) >= 20:
                    return failures
    return failures


def _validate_shard_metadata(run: Path) -> list[str]:
    path = run / "run_manifest.json"
    if not path.is_file():
        return ["missing run manifest for shard metadata validation"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for key, shard in manifest.get("shards", {}).items():
        if shard.get("status") != "complete":
            failures.append(f"shard {key}: status is not complete")
        if str(shard.get("dirty_status", "")).strip():
            failures.append(f"shard {key}: tracked tree was dirty")
        if shard.get("paper_eligible") is not False:
            failures.append(f"shard {key}: paper_eligible must remain false")
        history = shard.get("attempt_history")
        if not isinstance(history, list) or not history:
            failures.append(f"shard {key}: missing attempt_history")
            continue
        failed_attempts = sum(item.get("status") == "failed" for item in history if isinstance(item, dict))
        if int(shard.get("retry_count", -1)) != failed_attempts:
            failures.append(
                f"shard {key}: retry_count {shard.get('retry_count')} != failed attempts {failed_attempts}"
            )
        if history[-1].get("status") != "complete":
            failures.append(f"shard {key}: final attempt is not complete")
        for attempt_idx, attempt in enumerate(history):
            for field in ("wall_time_s", "peak_allocated_gib", "peak_reserved_gib"):
                if not finite(attempt.get(field)) or float(attempt[field]) < 0:
                    failures.append(f"shard {key} attempt {attempt_idx}: invalid {field}")
            if attempt.get("resolved_config_hash") != shard.get("resolved_config_hash"):
                failures.append(f"shard {key} attempt {attempt_idx}: config hash mismatch")
            if attempt.get("commit") != shard.get("commit"):
                failures.append(f"shard {key} attempt {attempt_idx}: commit mismatch")
    return failures


def _validate_formal12_shards(run: Path, pair_selection: dict[str, Any]) -> list[str]:
    manifest = _load_manifest(run)
    shards = manifest.get("shards", {})
    expected_pairs = [f"{left}__{right}" for left, right in pair_selection.get("pairs", [])]
    expected_counts = {
        "ccmr_metrics": 4_256, "alignment_control": 4_104,
        "condition_distance": 4_104, "condition_similarity": 4_104,
        "pair_metrics": 4_104, "temporal_metrics": 8_208,
        "time_gap_metrics": 11_704, "rho_per_condition": 0,
    }
    failures = []
    if len(shards) != 36:
        failures.append(f"formal12 shard coverage {len(shards)}/36")
    for seed in range(3):
        observed = []
        for pair_idx, pair_id in enumerate(expected_pairs):
            key = f"{seed}:{pair_idx}"
            shard = shards.get(key)
            if not shard:
                failures.append(f"missing formal12 shard {key}")
                continue
            observed.append(str(shard.get("pair_id")))
            if shard.get("pair_id") != pair_id:
                failures.append(f"shard {key}: pair ID mismatch")
            if shard.get("pair_selection_hash") != pair_selection.get("pair_selection_hash"):
                failures.append(f"shard {key}: pair-selection hash mismatch")
            for table, count in expected_counts.items():
                if int(shard.get("row_counts", {}).get(table, -1)) != count:
                    failures.append(f"shard {key}: {table} row count mismatch")
        if observed and observed != expected_pairs:
            failures.append(f"seed {seed}: pair order differs from preregistration")
    return failures


def validate_collection_commit_map(
    aggregate_manifest: dict[str, Any], expected_runs: list[Path],
) -> list[str]:
    """Match every aggregate commit entry to its source run exactly."""
    failures: list[str] = []
    commit_map = aggregate_manifest.get("collection_commits", {})
    if set(commit_map) != FORMAL_COLLECTION_COMMIT_KEYS or any(
        not str(value).strip() for value in commit_map.values()
    ):
        failures.append(f"aggregate collection commit map mismatch: {commit_map}")
        return failures
    expected_run_keys = {
        "dit_512_formal_v2": "dit_512",
        "flux_pairwise_formal12_v2": "flux_pairwise_formal12",
        "flux_rho_formal_v2": "flux_condition_rho",
    }
    for run in expected_runs:
        key = expected_run_keys.get(run.name)
        if key is None:
            failures.append(f"unexpected aggregate source run: {run.name}")
            continue
        run_manifest = _load_manifest(run)
        commits = {
            str(item.get("commit"))
            for item in run_manifest.get("shards", {}).values()
            if item.get("commit")
        }
        if len(commits) != 1 or commit_map.get(key) not in commits:
            failures.append(f"aggregate collection commit mismatch for {run.name}")
    return failures


def _validate_combined(combined: Path, expected_runs: list[Path]) -> list[str]:
    required = ["ccmr_metrics.csv.gz", "alignment_control.csv.gz", "alignment_cluster_summary.csv.gz", "rho_per_condition.csv.gz", "rho_stability.csv.gz", "subset_stability.csv.gz", "rho_consistency.json", "hierarchical_bootstrap.json", "summary.json", "aggregate_manifest.json"]
    failures = [f"missing {name}" for name in required if not (combined / name).is_file()]
    if failures:
        return failures
    manifest = json.loads((combined / "aggregate_manifest.json").read_text(encoding="utf-8"))
    actual_runs = {str(Path(item["path"]).resolve()) for item in manifest.get("source_runs", [])}
    wanted_runs = {str(path.resolve()) for path in expected_runs}
    if actual_runs != wanted_runs:
        failures.append(f"aggregate source runs mismatch: {sorted(actual_runs)}")
    for table, count in FORMAL_COMBINED_COUNTS.items():
        if int(manifest.get("tables", {}).get(table, -1)) != count:
            failures.append(f"aggregate {table} row count mismatch")
        rows = read_rows(combined / f"{table}.csv.gz")
        if len(rows) != count:
            failures.append(f"combined {table} actual row count mismatch")
    if manifest.get("rho_scope_counts") != {"condition": 377_216}:
        failures.append(f"aggregate rho scope mismatch: {manifest.get('rho_scope_counts')}")
    expected_revision = "24 random pairs replaced by 12 preregistered balanced-cycle pairs before formal FLUX analysis"
    if manifest.get("protocol_revision") != expected_revision:
        failures.append("aggregate protocol revision mismatch")
    failures.extend(validate_collection_commit_map(manifest, expected_runs))
    artifacts = {item.get("path"): item for item in manifest.get("artifacts", [])}
    for name in required[:-1]:
        artifact = artifacts.get(name)
        path = combined / name
        if not artifact:
            failures.append(f"aggregate manifest lacks artifact {name}")
        elif sha256_file(path) != artifact.get("sha256"):
            failures.append(f"aggregate checksum mismatch: {name}")
    for name in ("rho_stability.csv.gz", "subset_stability.csv.gz", "alignment_cluster_summary.csv.gz"):
        if not read_rows(combined / name):
            failures.append(f"aggregate table is empty: {name}")
    return failures


def _load_manifest(run: Path) -> dict[str, Any]:
    path = run / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _latent_checks(run: Path, kind: str, expected_seeds: int, expected_conditions: int, pairs_per_seed: int = 12) -> list[str]:
    path = run / "latent_hashes.json"
    if not path.is_file():
        return [f"{kind}: missing latent_hashes.json"]
    values = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if kind == "dit":
        if len(values) != expected_seeds:
            failures.append(f"dit latent seed coverage {len(values)}/{expected_seeds}")
        for seed, hashes in values.items():
            if len(hashes) != expected_conditions or len(set(hashes)) != 1:
                failures.append(f"dit seed {seed}: shared-latent hash mismatch")
    elif kind == "flux_pairwise":
        if len(values) != expected_seeds * pairs_per_seed:
            failures.append(f"flux pair latent coverage {len(values)}/{expected_seeds * pairs_per_seed}")
        by_seed = {}
        for shard, hashes in values.items():
            if len(hashes) != 2 or len(set(hashes)) != 1:
                failures.append(f"flux pair {shard}: duplicated-latent hash mismatch")
            seed = str(shard).split(":", 1)[0]
            by_seed.setdefault(seed, set()).update(hashes)
        for seed, hashes in by_seed.items():
            if len(hashes) != 1:
                failures.append(f"flux seed {seed}: latent differs across prompt pairs")
    else:
        if len(values) != expected_seeds:
            failures.append(f"flux rho latent seed coverage {len(values)}/{expected_seeds}")
        manifest_path = run / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        by_seed: dict[str, set[str]] = {}
        for key, shard in manifest.get("shards", {}).items():
            seed = str(key).split(":", 1)[0]
            latent = str(shard.get("latent_hash", ""))
            if latent:
                by_seed.setdefault(seed, set()).add(latent)
        for seed, hashes in by_seed.items():
            if len(hashes) != 1 or values.get(seed) not in hashes:
                failures.append(f"flux rho seed {seed}: latent differs across prompt shards")
    return failures


def validate(
    dit_run: Path,
    flux_pair_run: Path,
    flux_rho_run: Path,
    combined: Path,
    tests_log: Path,
    tables_dir: Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, failures: list[str], not_run: bool = False) -> None:
        status = "not_run" if not_run else ("failed" if failures else "passed")
        checks[name] = {"passed": not failures and not not_run, "status": status, "failures": failures}

    for label, run in (("dit", dit_run), ("flux_pairwise", flux_pair_run), ("flux_rho", flux_rho_run)):
        record(f"{label}.atomic", validate_atomic_manifest(run) if run.exists() else [f"missing run: {run}"], not_run=not run.exists())
        record(f"{label}.shards", _validate_shard_metadata(run) if run.exists() else [f"missing run: {run}"], not_run=not run.exists())

    dit_cfg = _config(dit_run)
    dit_ccmr = table_rows(dit_run, "ccmr_metrics")
    dit_rho = table_rows(dit_run, "rho_per_condition")
    dit_align = table_rows(dit_run, "alignment_control")
    record("dit.protocol", [] if (
        int(dit_cfg.get("image_size", 0)) == 512
        and len(dit_cfg.get("seeds", [])) == 5
        and len(dit_cfg.get("class_ids", [])) == 16
        and int(dit_cfg.get("num_inference_steps", 0)) == 50
        and float(dit_cfg.get("cfg_scale", 0)) == 4.0
        and dit_cfg.get("sampler") == "ddim" and dit_cfg.get("time_gaps") == [1, 2, 4]
        and dit_cfg.get("condition_backend") == "exact_batch" and int(dit_cfg.get("condition_batch_size", 0)) == 16
        and int(dit_cfg.get("alignment_shuffle_trials", 0)) == 100
        and dit_cfg.get("cache_enabled") is False and dit_cfg.get("decode_output") is False
        and dit_cfg.get("experiment_level") == "formal"
        and dit_cfg.get("paper_candidate") is True and dit_cfg.get("paper_eligible") is False
    ) else ["DiT requires formal 512, 5 seeds, 16 classes, 50 steps"])
    record("dit.modules", validate_modules(dit_ccmr, DIT_MODULES))
    record("dit.coverage", [] if (len(dit_ccmr) == 14_000 and len(dit_rho) == 224_000 and len(dit_align) == 1_372_000) else [f"DiT row coverage ccmr={len(dit_ccmr)}/14000 rho={len(dit_rho)}/224000 alignment={len(dit_align)}/1372000"])
    record("dit.boundaries", validate_boundary(dit_rho, 50))
    record("dit.derangements", validate_derangements(dit_align, 100))
    record("dit.finite", _finite_valid(dit_ccmr, ("v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("dit.rho_finite", validate_valid_rho_finite(dit_rho))
    record("dit.latents", _latent_checks(dit_run, "dit", 5, 16))

    pair_cfg = _config(flux_pair_run)
    pair_selection_path = flux_pair_run / "pair_selection.json"
    pair_selection = json.loads(pair_selection_path.read_text(encoding="utf-8")) if pair_selection_path.is_file() else {}
    pair_conditions = json.loads((flux_pair_run / "conditions.json").read_text(encoding="utf-8")) if (flux_pair_run / "conditions.json").is_file() else []
    pair_ccmr = table_rows(flux_pair_run, "ccmr_metrics")
    pair_align = table_rows(flux_pair_run, "alignment_control")
    pair_shards = json.loads((flux_pair_run / "run_manifest.json").read_text()).get("shards", {}) if (flux_pair_run / "run_manifest.json").is_file() else {}
    record("flux_pairwise.protocol", [] if (
        int(pair_cfg.get("height", 0)) == 1024 and int(pair_cfg.get("width", 0)) == 1024
        and len(pair_cfg.get("seeds", [])) == 3 and len(pair_selection.get("pairs", [])) == 12
        and len(pair_shards) == 36 and int(pair_cfg.get("num_inference_steps", 0)) == 28
        and pair_cfg.get("pair_selection_mode") == "balanced_cycle"
        and pair_cfg.get("condition_backend") == "pairwise" and int(pair_cfg.get("condition_batch_size", 0)) == 2
        and int(pair_cfg.get("pair_selection_seed", -1)) == 2027
        and pair_cfg.get("compact_pair_stats") is True and pair_cfg.get("feature_device") == "cpu"
        and pair_cfg.get("model_dtype") == "bfloat16" and pair_cfg.get("statistics_dtype") == "float32"
        and pair_cfg.get("time_gaps") == [1, 2, 4]
        and pair_cfg.get("cache_enabled") is False and pair_cfg.get("decode_output") is False
        and pair_cfg.get("experiment_level") == "formal"
        and pair_cfg.get("paper_candidate") is True and pair_cfg.get("paper_eligible") is False
    ) else ["FLUX pairwise requires formal 1024, 3 seeds x 12 balanced-cycle pairs, 28 steps"])
    record("flux_pairwise.modules", validate_modules(pair_ccmr, FLUX_MODULES))
    record("flux_pairwise.coverage", [] if (len(pair_ccmr) == 153_216 and len(pair_align) == 147_744) else [f"FLUX pair row coverage ccmr={len(pair_ccmr)}/153216 alignment={len(pair_align)}/147744"])
    record("flux_pairwise.swap", validate_flux_swaps(pair_align))
    record("flux_pairwise.selection", validate_flux_pair_selection(pair_cfg, pair_conditions, pair_selection) if pair_cfg and pair_conditions else ["pair-selection inputs are empty"])
    record("flux_pairwise.shard_protocol", _validate_formal12_shards(flux_pair_run, pair_selection))
    record("flux_pairwise.finite", _finite_valid(pair_ccmr, ("v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("flux_pairwise.latents", _latent_checks(flux_pair_run, "flux_pairwise", 3, 2, pairs_per_seed=12))

    rho_cfg = _config(flux_rho_run)
    flux_rho = table_rows(flux_rho_run, "rho_per_condition")
    rho_manifest = json.loads((flux_rho_run / "run_manifest.json").read_text()).get("shards", {}) if (flux_rho_run / "run_manifest.json").is_file() else {}
    record("flux_rho.protocol", [] if (
        int(rho_cfg.get("height", 0)) == 1024 and int(rho_cfg.get("width", 0)) == 1024
        and len(rho_cfg.get("seeds", [])) == 3 and len(rho_cfg.get("prompt_ids", [])) == 12
        and len(rho_manifest) == 36 and int(rho_cfg.get("num_inference_steps", 0)) == 28
        and rho_cfg.get("feature_device") == "cpu"
        and rho_cfg.get("model_dtype") == "bfloat16" and rho_cfg.get("statistics_dtype") == "float32"
        and rho_cfg.get("cache_enabled") is False and rho_cfg.get("decode_output") is False
        and rho_cfg.get("experiment_level") == "formal"
        and rho_cfg.get("paper_candidate") is True and rho_cfg.get("paper_eligible") is False
    ) else ["FLUX rho requires formal 1024, 3 seeds x 12 prompts, 28 steps"])
    record("flux_rho.scope", [] if flux_rho and all(row.get("rho_scope") == "condition" for row in flux_rho) else ["FLUX Panel D requires condition-scope rho only"])
    record("flux_rho.modules", validate_modules(flux_rho, FLUX_MODULES))
    record("flux_rho.coverage", [] if len(flux_rho) == 153_216 else [f"FLUX rho row coverage {len(flux_rho)}/153216"])
    record("flux_rho.boundaries", validate_boundary(flux_rho, 28))
    record("flux_rho.rho_finite", validate_valid_rho_finite(flux_rho))
    record("flux_rho.latents", _latent_checks(flux_rho_run, "flux_rho", 3, 12))

    audits = {}
    for label, rows, cfg in (("dit", dit_rho, dit_cfg), ("flux_rho", flux_rho, rho_cfg)):
        tolerance_failures = []
        if rows:
            try:
                tolerance_passed, audit = tolerance_check(
                    rows, float(cfg.get("rho_repeat_rtol", 1e-4)),
                    float(cfg.get("rho_repeat_atol", 1e-6)),
                )
                if not tolerance_passed:
                    tolerance_failures.append(f"rho equivalence failed for {audit['failure_count']} cells")
            except Exception as exc:
                audit = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                tolerance_failures.append(audit["error"])
        else:
            audit = {"passed": False, "status": "not_run", "failure_count": 0}
            tolerance_failures.append("no condition-level rho rows")
        audits[label] = audit
        record(f"{label}.rho_live_code_equivalence", tolerance_failures)

    record("combined", _validate_combined(combined, [dit_run, flux_pair_run, flux_rho_run]), not_run=not combined.exists())
    record("tests", validate_test_log(tests_log), not_run=not tests_log.exists())
    if tables_dir is not None:
        names = ("table_ccmr_summary.csv", "table_ccmr_summary.tex", "table_ccmr_summary.md", "table_rho_stability.csv", "table_rho_stability.tex", "table_rho_stability.md", "ccmr_paper_numbers.tex", "table_manifest.json")
        record("paper_tables", [f"missing {name}" for name in names if not (tables_dir / name).is_file()])

    passed = all(item["passed"] for item in checks.values())
    return {
        "schema_version": 2,
        "validated_at": utc_now(),
        "passed": passed,
        "paper_eligible": passed,
        "checks": checks,
        "rho_consistency": audits,
        "missing_or_failed": [failure for item in checks.values() for failure in item["failures"]],
        "recovery_commands": [
            "python assets/visualization/ccmr/code/collect_dit_ccmr.py --config assets/ccmr_formal/configs_v2/dit_512_formal.yaml --output-dir assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 --resume",
            "python assets/visualization/ccmr/code/collect_flux_ccmr.py --config assets/ccmr_formal/configs_v2/flux_pairwise_formal12.yaml --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2 --resume",
            "python assets/visualization/ccmr/code/collect_flux_rho.py --config assets/ccmr_formal/configs_v2/flux_rho_formal.yaml --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 --resume",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all CCMR formal gates")
    parser.add_argument("--dit-run", default="assets/ccmr_formal/data/runs_v2/dit_512_formal_v2")
    parser.add_argument("--flux-pair-run", default="assets/ccmr_formal/data/runs_v2/flux_pairwise_formal12_v2")
    parser.add_argument("--flux-rho-run", default="assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2")
    parser.add_argument("--combined", default="assets/ccmr_formal/data/combined/formal_v2")
    parser.add_argument("--tests-log", default="assets/ccmr_formal/logs_v2/tests.log")
    parser.add_argument("--tables-dir")
    parser.add_argument("--output", default="assets/ccmr_formal/formal_manifest_v2.json")
    args = parser.parse_args()
    result = validate(*(Path(value).resolve() for value in (args.dit_run, args.flux_pair_run, args.flux_rho_run, args.combined, args.tests_log)), Path(args.tables_dir).resolve() if args.tables_dir else None)
    save_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
