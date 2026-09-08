#!/usr/bin/env python3
"""Validate the three audited CCMR smoke runs independently of formal gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import read_rows, save_json_atomic, utc_now
from formal_protocol import (
    finite,
    is_true,
    table_rows,
    tolerance_check,
    validate_atomic_manifest,
    validate_boundary,
    validate_flux_pair_selection,
    validate_valid_rho_finite,
)


EXPECTED = {
    "dit": {"ccmr_metrics": 2800, "rho_per_condition": 44800, "alignment_control": 274400},
    "flux_pair": {"ccmr_metrics": 4256, "alignment_control": 4104},
    "flux_rho": {"rho_per_condition": 4256},
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _valid_finite(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    if not rows:
        return ["numeric validation has no rows"]
    failures = []
    for index, row in enumerate(rows):
        if not is_true(row.get("valid", True)):
            continue
        for field in fields:
            if not finite(row.get(field)):
                failures.append(f"valid row {index}: non-finite {field}")
                if len(failures) == 20:
                    return failures
    return failures


def _shared_latents(run: Path, kind: str) -> list[str]:
    values = _load(run / "latent_hashes.json")
    if not values:
        return ["missing or empty latent_hashes.json"]
    failures = []
    if kind == "dit":
        for seed, hashes in values.items():
            if len(hashes) != 16 or len(set(hashes)) != 1:
                failures.append(f"DiT seed {seed} does not share one latent across 16 conditions")
    elif kind == "flux_pair":
        for shard, hashes in values.items():
            if len(hashes) != 2 or len(set(hashes)) != 1:
                failures.append(f"FLUX pair {shard} does not duplicate one latent")
    elif len(values) != 1:
        failures.append("FLUX rho smoke must contain exactly one seed latent")
    return failures


def _runtime_metadata(run: Path) -> list[str]:
    failures = []
    manifest = _load(run / "run_manifest.json")
    shards = manifest.get("shards", {})
    if not shards:
        return ["run manifest contains no shards"]
    for key, shard in shards.items():
        if str(shard.get("dirty_status", "")).strip():
            failures.append(f"shard {key}: tracked source tree was dirty")
        if shard.get("cache_enabled") is not False or shard.get("decode_output") is not False:
            failures.append(f"shard {key}: cache/decode flags are not false")
        if shard.get("oom") is not False:
            failures.append(f"shard {key}: OOM or missing OOM status")
        for field in ("wall_time_s", "peak_allocated_gib", "peak_reserved_gib"):
            if not finite(shard.get(field)) or float(shard[field]) < 0:
                failures.append(f"shard {key}: invalid {field}")
    memory = _load(run / "memory.json")
    for field in ("peak_allocated_gib", "peak_reserved_gib"):
        if not finite(memory.get(field)) or float(memory[field]) < 0:
            failures.append(f"memory.json: invalid {field}")
    summary = _load(run / "summary.json")
    if not finite(summary.get("elapsed_s")) or float(summary["elapsed_s"]) <= 0:
        failures.append("summary.json: invalid elapsed_s")
    return failures


def _coverage(run: Path, expected: dict[str, int]) -> list[str]:
    summary = _load(run / "summary.json")
    failures = []
    for table, count in expected.items():
        rows = table_rows(run, table)
        if len(rows) != count:
            failures.append(f"{table}: actual {len(rows)}, expected {count}")
        if int(summary.get("row_counts", {}).get(table, -1)) != count:
            failures.append(f"summary {table}: {summary.get('row_counts', {}).get(table)}, expected {count}")
    return failures


def validate_smokes(dit: Path, flux_pair: Path, flux_rho: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    def record(name: str, failures: list[str]):
        checks[name] = {"passed": not failures, "status": "passed" if not failures else "failed", "failures": failures}

    for label, run in (("dit", dit), ("flux_pair", flux_pair), ("flux_rho", flux_rho)):
        record(f"{label}.atomic", validate_atomic_manifest(run) if run.exists() else [f"missing run {run}"])
        record(f"{label}.runtime", _runtime_metadata(run) if run.exists() else ["not run"])
        record(f"{label}.coverage", _coverage(run, EXPECTED[label]))
        record(f"{label}.latents", _shared_latents(run, label) if run.exists() else ["not run"])

    dit_ccmr = table_rows(dit, "ccmr_metrics")
    pair_ccmr = table_rows(flux_pair, "ccmr_metrics")
    dit_rho = table_rows(dit, "rho_per_condition")
    flux_rho_rows = table_rows(flux_rho, "rho_per_condition")
    record("dit.finite", _valid_finite(dit_ccmr, ("v_raw", "v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("flux_pair.finite", _valid_finite(pair_ccmr, ("v_raw", "v_base", "v_diff", "r_ccmr", "g_ccmr_db")))
    record("dit.rho_finite", validate_valid_rho_finite(dit_rho))
    record("flux_rho.rho_finite", validate_valid_rho_finite(flux_rho_rows))
    record("dit.boundary", validate_boundary(dit_rho, 50))
    record("flux_rho.boundary", validate_boundary(flux_rho_rows, 28))

    audits = {}
    for label, rows, config_path in (("dit", dit_rho, dit / "config.json"), ("flux_rho", flux_rho_rows, flux_rho / "config.json")):
        config = _load(config_path)
        if not rows:
            passed, audit = False, {"passed": False, "status": "not_run", "failure_count": 0}
        else:
            try:
                passed, audit = tolerance_check(rows, float(config.get("rho_repeat_rtol", 1e-4)), float(config.get("rho_repeat_atol", 1e-6)))
            except Exception as exc:
                passed, audit = False, {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        audits[label] = audit
        record(f"{label}.rho_live_code", [] if passed else [audit.get("error", f"tolerance failures: {audit.get('failure_count', 'unknown')}")])

    pair_config = _load(flux_pair / "config.json")
    conditions = _load(flux_pair / "conditions.json")
    selection = _load(flux_pair / "pair_selection.json")
    record("flux_pair.selection", validate_flux_pair_selection(pair_config, conditions if isinstance(conditions, list) else [], selection) if pair_config else ["not run"])

    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 2, "validation_scope": "smoke", "validated_at": utc_now(),
        "passed": passed, "paper_eligible": False, "checks": checks,
        "rho_consistency": audits,
        "runs": {"dit": str(dit), "flux_pair": str(flux_pair), "flux_rho": str(flux_rho)},
        "missing_or_failed": [failure for check in checks.values() for failure in check["failures"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CCMR smoke artifacts")
    parser.add_argument("--dit-run", default="assets/ccmr_formal/data/runs_v2/dit_512_smoke_v2")
    parser.add_argument("--flux-pair-run", default="assets/ccmr_formal/data/runs_v2/flux_pairwise_smoke_v2")
    parser.add_argument("--flux-rho-run", default="assets/ccmr_formal/data/runs_v2/flux_rho_smoke_v2")
    parser.add_argument("--output", default="assets/ccmr_formal/smoke_manifest_v2.json")
    args = parser.parse_args()
    result = validate_smokes(Path(args.dit_run).resolve(), Path(args.flux_pair_run).resolve(), Path(args.flux_rho_run).resolve())
    save_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
