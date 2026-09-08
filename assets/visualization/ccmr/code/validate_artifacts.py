#!/usr/bin/env python3
"""Hard gate for CCMR formal artifacts; never falls back to pilot data."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from common import load_yaml, read_rows, save_json_atomic, sha256_file, utc_now
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
    validate_modules,
)


def _config(run: Path) -> dict[str, Any]:
    path = run / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _finite_valid(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
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


def _latent_checks(run: Path, kind: str, expected_seeds: int, expected_conditions: int) -> list[str]:
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
        if len(values) != expected_seeds * 24:
            failures.append(f"flux pair latent coverage {len(values)}/{expected_seeds * 24}")
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

    def record(name: str, failures: list[str]) -> None:
        checks[name] = {"passed": not failures, "failures": failures}

    for label, run in (("dit", dit_run), ("flux_pairwise", flux_pair_run), ("flux_rho", flux_rho_run)):
        record(f"{label}.atomic", validate_atomic_manifest(run) if run.exists() else [f"missing run: {run}"])

    dit_cfg = _config(dit_run)
    dit_ccmr = table_rows(dit_run, "ccmr_metrics")
    dit_rho = table_rows(dit_run, "rho_per_condition")
    dit_align = table_rows(dit_run, "alignment_control")
    record("dit.protocol", [] if (
        int(dit_cfg.get("image_size", 0)) == 512
        and len(dit_cfg.get("seeds", [])) == 5
        and len(dit_cfg.get("class_ids", [])) == 16
        and int(dit_cfg.get("num_inference_steps", 0)) == 50
        and dit_cfg.get("experiment_level") == "formal"
    ) else ["DiT requires formal 512, 5 seeds, 16 classes, 50 steps"])
    record("dit.modules", validate_modules(dit_ccmr, DIT_MODULES))
    record("dit.coverage", [] if (len(dit_ccmr) == 14_000 and len(dit_rho) == 224_000 and len(dit_align) == 1_372_000) else [f"DiT row coverage ccmr={len(dit_ccmr)}/14000 rho={len(dit_rho)}/224000 alignment={len(dit_align)}/1372000"])
    record("dit.boundaries", validate_boundary(dit_rho, 50))
    record("dit.derangements", validate_derangements(dit_align, 100))
    record("dit.finite", _finite_valid(dit_ccmr, ("v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("dit.latents", _latent_checks(dit_run, "dit", 5, 16))

    pair_cfg = _config(flux_pair_run)
    pair_selection_path = flux_pair_run / "pair_selection.json"
    pair_selection = json.loads(pair_selection_path.read_text(encoding="utf-8")) if pair_selection_path.is_file() else {}
    pair_ccmr = table_rows(flux_pair_run, "ccmr_metrics")
    pair_align = table_rows(flux_pair_run, "alignment_control")
    pair_shards = json.loads((flux_pair_run / "run_manifest.json").read_text()).get("shards", {}) if (flux_pair_run / "run_manifest.json").is_file() else {}
    record("flux_pairwise.protocol", [] if (
        int(pair_cfg.get("height", 0)) == 1024 and int(pair_cfg.get("width", 0)) == 1024
        and len(pair_cfg.get("seeds", [])) == 3 and len(pair_selection.get("pairs", [])) == 24
        and len(pair_shards) == 72 and int(pair_cfg.get("num_inference_steps", 0)) == 28
        and pair_cfg.get("experiment_level") == "formal"
    ) else ["FLUX pairwise requires formal 1024, 3 seeds x 24 pairs, 28 steps"])
    record("flux_pairwise.modules", validate_modules(pair_ccmr, FLUX_MODULES))
    record("flux_pairwise.coverage", [] if (len(pair_ccmr) == 306_432 and len(pair_align) == 295_488) else [f"FLUX pair row coverage ccmr={len(pair_ccmr)}/306432 alignment={len(pair_align)}/295488"])
    duplicate_swap = []
    grouped_swap = {}
    for row in pair_align:
        key = (row.get("seed"), row.get("pair_id"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("step_idx"))
        grouped_swap[key] = grouped_swap.get(key, 0) + 1
    if any(count != 1 for count in grouped_swap.values()):
        duplicate_swap.append("FLUX pair swap must occur exactly once per valid cell")
    record("flux_pairwise.swap", duplicate_swap)
    record("flux_pairwise.finite", _finite_valid(pair_ccmr, ("v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("flux_pairwise.latents", _latent_checks(flux_pair_run, "flux_pairwise", 3, 2))

    rho_cfg = _config(flux_rho_run)
    flux_rho = table_rows(flux_rho_run, "rho_per_condition")
    rho_manifest = json.loads((flux_rho_run / "run_manifest.json").read_text()).get("shards", {}) if (flux_rho_run / "run_manifest.json").is_file() else {}
    record("flux_rho.protocol", [] if (
        int(rho_cfg.get("height", 0)) == 1024 and int(rho_cfg.get("width", 0)) == 1024
        and len(rho_cfg.get("seeds", [])) == 3 and len(rho_cfg.get("prompt_ids", [])) == 12
        and len(rho_manifest) == 36 and int(rho_cfg.get("num_inference_steps", 0)) == 28
        and rho_cfg.get("experiment_level") == "formal"
    ) else ["FLUX rho requires formal 1024, 3 seeds x 12 prompts, 28 steps"])
    record("flux_rho.scope", [] if flux_rho and all(row.get("rho_scope") == "condition" for row in flux_rho) else ["FLUX Panel D requires condition-scope rho only"])
    record("flux_rho.modules", validate_modules(flux_rho, FLUX_MODULES))
    record("flux_rho.coverage", [] if len(flux_rho) == 153_216 else [f"FLUX rho row coverage {len(flux_rho)}/153216"])
    record("flux_rho.boundaries", validate_boundary(flux_rho, 28))
    record("flux_rho.latents", _latent_checks(flux_rho_run, "flux_rho", 3, 12))

    all_rho = dit_rho + flux_rho
    tolerance_failures = []
    audit = {}
    if all_rho:
        passed, audit = tolerance_check(
            all_rho,
            max(float(dit_cfg.get("rho_repeat_rtol", 1e-4)), float(rho_cfg.get("rho_repeat_rtol", 1e-4))),
            max(float(dit_cfg.get("rho_repeat_atol", 1e-6)), float(rho_cfg.get("rho_repeat_atol", 1e-6))),
        )
        if not passed:
            tolerance_failures.append(f"rho equivalence failed for {audit['failure_count']} cells")
    else:
        tolerance_failures.append("no condition-level rho rows")
    record("rho.live_code_equivalence", tolerance_failures)

    combined_required = ["ccmr_metrics.csv.gz", "alignment_control.csv.gz", "alignment_cluster_summary.csv.gz", "rho_stability.csv.gz", "subset_stability.csv.gz", "rho_consistency.json", "hierarchical_bootstrap.json", "summary.json"]
    record("combined", [f"missing {name}" for name in combined_required if not (combined / name).is_file()])
    record("tests", [] if tests_log.is_file() and "failed" not in tests_log.read_text(encoding="utf-8", errors="replace").lower() else [f"passing test log missing: {tests_log}"])
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
        "rho_consistency": audit,
        "missing_or_failed": [failure for item in checks.values() for failure in item["failures"]],
        "recovery_commands": [
            "python assets/visualization/ccmr/code/collect_dit_ccmr.py --config assets/ccmr_formal/configs_v2/dit_512_formal.yaml --output-dir assets/ccmr_formal/data/runs_v2/dit_512_formal_v2 --resume",
            "python assets/visualization/ccmr/code/collect_flux_ccmr.py --config assets/ccmr_formal/configs_v2/flux_pairwise_formal.yaml --output-dir assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2 --resume",
            "python assets/visualization/ccmr/code/collect_flux_rho.py --config assets/ccmr_formal/configs_v2/flux_rho_formal.yaml --output-dir assets/ccmr_formal/data/runs_v2/flux_rho_formal_v2 --resume",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all CCMR formal gates")
    parser.add_argument("--dit-run", default="assets/ccmr_formal/data/runs_v2/dit_512_formal_v2")
    parser.add_argument("--flux-pair-run", default="assets/ccmr_formal/data/runs_v2/flux_pairwise_formal_v2")
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
