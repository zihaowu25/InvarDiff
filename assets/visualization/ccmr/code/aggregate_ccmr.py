#!/usr/bin/env python3
"""Aggregate CCMR collector shards into portable tables and summaries."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (  # noqa: E402
    flatten_jsonable,
    gain_db,
    jaccard_at_fraction,
    load_yaml,
    percentile_ci,
    rank_corr,
    read_rows,
    save_json_atomic,
    write_rows,
)


TABLES = ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition")
NUMERIC = {
    "ccmr_metrics": ("seed", "num_conditions", "layer_idx", "step_idx", "scheduler_timestep", "feature_numel", "v_raw", "v_raw_prev", "v_base", "v_diff", "r_ccmr", "g_ccmr_db"),
    "condition_distance": ("seed", "layer_idx", "step_idx", "raw_pair_distance", "diff_pair_distance"),
    "rho_per_condition": ("seed", "layer_idx", "score_step_idx", "l1_prev", "l1_next", "rho_clean", "rho_code"),
    "temporal_metrics": ("seed", "layer_idx", "step_idx", "output_rms", "diff_rms", "r_time"),
    "time_gap_metrics": ("seed", "layer_idx", "step_idx", "time_gap", "v_raw_base", "v_diff_gap", "r_ccmr_gap", "g_ccmr_gap_db"),
}


def _read_runs(inputs: list[Path]):
    tables: dict[str, list[dict[str, Any]]] = {key: [] for key in TABLES}
    runs = []
    for root in inputs:
        candidates = [root] if (root / "summary.json").exists() else [p.parent for p in root.rglob("summary.json")]
        for run in candidates:
            try:
                summary = json.loads((run / "summary.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            runs.append({"path": run, "summary": summary})
            for table in TABLES:
                for candidate in (run / "tables" / f"{table}.csv.gz", run / "tables" / f"{table}.csv"):
                    if candidate.exists():
                        rows = read_rows(candidate)
                        for row in rows:
                            row["source_run"] = summary.get("run_id", run.name)
                            for key in NUMERIC.get(table, ()):
                                if row.get(key) not in (None, ""):
                                    try: row[key] = float(row[key])
                                    except (ValueError, TypeError): pass
                        tables[table].extend(rows)
                        break
    return runs, tables


def _recompute_pairwise_ccmr(ccmr: list[dict[str, Any]], distance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use all selected pairs for the finite-population pairwise estimate."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in distance:
        grouped[(row.get("source_run"), row.get("model"), row.get("seed"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("step_idx"))].append(row)
    out = []
    for row in ccmr:
        if row.get("estimator") != "pairwise":
            out.append(row); continue
        key = (row.get("source_run"), row.get("model"), row.get("seed"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("step_idx"))
        pairs = grouped.get(key, [])
        if not pairs:
            out.append(row); continue
        k = max(2.0, float(row.get("num_conditions") or 2.0))
        raw = (k - 1.0) / k * float(np.mean([float(p["raw_pair_distance"]) for p in pairs]))
        diff = (k - 1.0) / k * float(np.mean([float(p["diff_pair_distance"]) for p in pairs]))
        updated = dict(row)
        updated["v_raw"] = raw
        updated["v_diff"] = diff
        prev = float(row.get("v_raw_prev") or raw)
        updated["v_raw_prev"] = prev
        updated["v_base"] = 0.5 * (raw + prev)
        updated["r_ccmr"] = diff / max(updated["v_base"], 1.0e-12)
        updated["g_ccmr_db"] = gain_db(updated["v_base"], diff)
        updated["pair_count"] = len(pairs)
        out.append(updated)
    return out


def _rho_stability(rho_rows: list[dict[str, Any]], ccmr_rows: list[dict[str, Any]], config: dict[str, Any]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rho_rows:
        if str(row.get("valid", "True")).lower() == "false": continue
        groups[(row.get("source_run"), row.get("model"), row.get("seed"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("score_step_idx"))].append(row)
    out = []
    full_values: dict[tuple[Any, ...], float] = {}
    for key, rows in groups.items():
        vals = np.asarray([float(x["rho_clean"]) for x in rows if str(x.get("rho_clean")) not in {"", "nan", "None"}], dtype=float)
        if vals.size == 0: continue
        logs = np.log10(np.maximum(vals, 1.0e-30))
        item = {"source_run": key[0], "model": key[1], "seed": key[2], "module_family": key[3], "module_name": key[4], "layer_idx": key[5], "score_step_idx": key[6], "rho_mean": float(vals.mean()), "rho_median": float(np.median(vals)), "log_rho_mad": float(np.median(np.abs(logs - np.median(logs)))), "log_rho_iqr": float(np.percentile(logs, 75) - np.percentile(logs, 25)), "num_conditions": int(vals.size), "valid": True}
        out.append(item)
        full_values[key] = float(vals.mean())
    sizes = [int(x) for x in config.get("subset_sizes", [])]
    fractions = [float(x) for x in config.get("cache_fractions", [0.1, 0.2, 0.3, 0.5])]
    subset_rows = []
    rng = np.random.default_rng(int(config.get("shuffle_seed", 2027)))
    for size in sizes:
        for trial in range(int(config.get("subset_trials", 50))):
            subset: dict[tuple[Any, ...], float] = {}
            for key, rows in groups.items():
                vals = [float(x["rho_clean"]) for x in rows if x.get("rho_clean") not in (None, "")]
                if len(vals) < size: continue
                idx = rng.choice(len(vals), size=size, replace=False)
                subset[key] = float(np.mean(np.asarray(vals)[idx]))
            common = [key for key in subset if key in full_values]
            if not common: continue
            ref = [full_values[key] for key in common]; est = [subset[key] for key in common]
            rho, tau = rank_corr(ref, est)
            for fraction in fractions:
                subset_rows.append({"subset_size": size, "trial": trial, "cache_fraction": fraction, "spearman": rho, "kendall": tau, "jaccard": jaccard_at_fraction(ref, est, fraction), "num_points": len(common)})
    return out, subset_rows


def _summary(ccmr: list[dict[str, Any]], rho_stability: list[dict[str, Any]], subset_rows: list[dict[str, Any]], runs: list[dict[str, Any]]):
    summary: dict[str, Any] = {"runs": [{"run_id": r["summary"].get("run_id"), "path": str(r["path"]), "summary": r["summary"]} for r in runs], "models": {}, "invalid": {}, "subset_stability": {}}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ccmr:
        model = str(row.get("model")); module = f"{row.get('module_family')}.{row.get('module_name')}"
        value = row.get("g_ccmr_db")
        if value not in (None, ""):
            try: grouped[(model, module, "gain_db")].append(float(value))
            except (ValueError, TypeError): pass
        if str(row.get("valid", "True")).lower() == "false": summary["invalid"][model] = summary["invalid"].get(model, 0) + 1
    for (model, module, metric), values in grouped.items():
        summary["models"].setdefault(model, {}).setdefault(module, {})[metric] = percentile_ci(values, 2000)
    for size in sorted({int(r["subset_size"]) for r in subset_rows}):
        subset = [r for r in subset_rows if int(r["subset_size"]) == size]
        summary["subset_stability"][str(size)] = {metric: percentile_ci([float(r[metric]) for r in subset], 1000) for metric in ("spearman", "kendall", "jaccard")}
    summary["rho_stability_points"] = len(rho_stability)
    summary["ccmr_points"] = len(ccmr)
    return flatten_jsonable(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CCMR tables")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config) if args.config else {}
    runs, tables = _read_runs([Path(x).resolve() for x in args.input])
    tables["ccmr_metrics"] = _recompute_pairwise_ccmr(tables["ccmr_metrics"], tables["condition_distance"])
    stability, subset_rows = _rho_stability(tables["rho_per_condition"], tables["ccmr_metrics"], config)
    for table, rows in tables.items():
        write_rows(output / f"{table}.csv.gz", rows)
    write_rows(output / "rho_stability.csv.gz", stability)
    write_rows(output / "subset_stability.csv.gz", subset_rows)
    save_json_atomic(output / "summary.json", _summary(tables["ccmr_metrics"], stability, subset_rows, runs))
    save_json_atomic(output / "aggregate_manifest.json", {"inputs": [str(Path(x).resolve()) for x in args.input], "runs": [str(r["path"]) for r in runs], "tables": {key: len(value) for key, value in tables.items()}, "rho_stability": len(stability), "subset_stability": len(subset_rows)})
    print(json.dumps({"output": str(output), "runs": len(runs), "tables": {key: len(value) for key, value in tables.items()}}, indent=2))


if __name__ == "__main__":
    main()
