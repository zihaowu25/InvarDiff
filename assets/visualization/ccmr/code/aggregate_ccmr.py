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
    pairwise_population_variance,
    percentile_ci,
    rank_corr,
    read_rows,
    save_json_atomic,
    write_rows,
)


TABLES = ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition")
NUMERIC = {
    "ccmr_metrics": ("seed", "num_conditions", "layer_idx", "step_idx", "scheduler_timestep", "feature_numel", "v_raw", "v_raw_prev", "v_base", "v_diff", "r_ccmr", "g_ccmr_db"),
    "condition_distance": ("seed", "layer_idx", "step_idx", "raw_pair_distance", "raw_prev_pair_distance", "diff_pair_distance"),
    "rho_per_condition": ("seed", "layer_idx", "score_step_idx", "l1_prev", "l1_next", "rho_clean", "rho_code"),
    "temporal_metrics": ("seed", "layer_idx", "step_idx", "output_rms", "diff_rms", "r_time"),
    "time_gap_metrics": ("seed", "layer_idx", "step_idx", "time_gap", "v_raw_base", "v_diff_gap", "r_ccmr_gap", "g_ccmr_gap_db", "raw_current_pair_distance", "raw_previous_pair_distance", "diff_pair_distance"),
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


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_run"), row.get("model"), row.get("seed"),
        row.get("module_family"), row.get("module_name"),
        row.get("layer_idx"), row.get("step_idx"),
    )


def _finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _recompute_pairwise_ccmr(ccmr: list[dict[str, Any]], distance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate pairwise FLUX rows once per cell.

    A pairwise collector emits one ``ccmr_metrics`` row per selected prompt
    pair.  The previous implementation recomputed the same population
    estimate independently for every input row, multiplying the apparent
    sample size in downstream plots.  This version first groups both tables
    by the complete cell key and emits exactly one row for each cell.
    """
    grouped_distance: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in distance:
        grouped_distance[_cell_key(row)].append(row)
    grouped_ccmr: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in ccmr:
        if str(row.get("estimator", "")).lower() == "pairwise":
            grouped_ccmr[_cell_key(row)].append(row)
        else:
            passthrough.append(row)

    out = list(passthrough)
    for key, input_rows in grouped_ccmr.items():
        representative = dict(input_rows[0])
        pairs = grouped_distance.get(key, [])
        if not pairs:
            out.append(representative)
            continue
        raw_values = [v for v in (_finite_float(p.get("raw_pair_distance")) for p in pairs) if v is not None]
        diff_values = [v for v in (_finite_float(p.get("diff_pair_distance")) for p in pairs) if v is not None]
        if not raw_values or not diff_values:
            out.append(representative)
            continue
        k = max(2, int(float(representative.get("num_conditions") or 2)))
        # New collectors persist the pair energy for the previous state.  For
        # old books, recover it from the K=2 centered variance when possible.
        prev_values = [v for v in (_finite_float(p.get("raw_prev_pair_distance")) for p in pairs) if v is not None]
        legacy_prev = [
            2.0 * v for v in
            (_finite_float(row.get("v_raw_prev")) for row in input_rows)
            if v is not None
        ]
        if not prev_values:
            if len(legacy_prev) == len(pairs):
                prev_values = legacy_prev
            elif legacy_prev:
                prev_values = [float(np.mean(legacy_prev))] * len(raw_values)
            else:
                prev_values = [raw_values[0]] * len(raw_values)
        raw = pairwise_population_variance(raw_values, k)
        prev = pairwise_population_variance(prev_values, k)
        diff = pairwise_population_variance(diff_values, k)
        v_base = 0.5 * (raw + prev)
        representative.update({
            "v_raw": raw,
            "v_raw_prev": prev,
            "v_base": v_base,
            "v_diff": diff,
            "r_ccmr": diff / max(v_base, 1.0e-12),
            "g_ccmr_db": gain_db(v_base, diff),
            "pair_count": len(raw_values),
            "input_pair_rows": len(input_rows),
        })
        out.append(representative)
    return out


def _recompute_pairwise_time_gap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate FLUX time-gap energies before taking the dB gain.

    The compact FLUX collector stores one pair energy per selected prompt
    pair.  Averaging per-pair gains is not equivalent to taking the gain of
    the aggregated energies, so reduce current/previous/difference energies
    with the finite-population correction first.  Rows from older collectors
    that do not persist pair energies pass through unchanged.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    required = ("raw_current_pair_distance", "raw_previous_pair_distance", "diff_pair_distance")
    for row in rows:
        is_pairwise = str(row.get("estimator", "")).lower() == "pairwise"
        if is_pairwise and all(_finite_float(row.get(key)) is not None for key in required):
            grouped[_cell_key(row) + (row.get("time_gap"),)].append(row)
        else:
            passthrough.append(row)

    out = list(passthrough)
    for key, input_rows in grouped.items():
        representative = dict(input_rows[0])
        k = max(2, int(float(representative.get("num_conditions") or 2)))
        current = [float(row["raw_current_pair_distance"]) for row in input_rows]
        previous = [float(row["raw_previous_pair_distance"]) for row in input_rows]
        difference = [float(row["diff_pair_distance"]) for row in input_rows]
        v_current = pairwise_population_variance(current, k)
        v_previous = pairwise_population_variance(previous, k)
        v_difference = pairwise_population_variance(difference, k)
        v_base = 0.5 * (v_current + v_previous)
        representative.update({
            "v_raw_base": v_base,
            "v_diff_gap": v_difference,
            "r_ccmr_gap": v_difference / max(v_base, 1.0e-12),
            "g_ccmr_gap_db": gain_db(v_base, v_difference),
            "pair_count": len(input_rows),
            "input_pair_rows": len(input_rows),
        })
        out.append(representative)
    return out


def _rho_stability(rho_rows: list[dict[str, Any]], ccmr_rows: list[dict[str, Any]], config: dict[str, Any]):
    """Aggregate condition rho and evaluate fixed-subset stability.

    A calibration subset is selected once per model/source/seed/trial and is
    then applied to every layer and timestep.  Pair-difference rho rows are
    intentionally excluded because they do not represent individual prompt
    trajectories.
    """
    cell_condition_values: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    population: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rho_rows:
        if str(row.get("valid", "True")).lower() == "false":
            continue
        if str(row.get("rho_scope", "")).lower() == "pair_difference":
            continue
        # Legacy compact FLUX rho rows were mirrored pair values.  Until a
        # condition-scope rho collector supplies real prompt rows, exclude
        # them rather than reporting artificially zero dispersion.
        if str(row.get("model", "")).lower() == "flux" and str(row.get("rho_scope", "")).lower() != "condition":
            continue
        value = _finite_float(row.get("rho_clean"))
        if value is None:
            continue
        run_key = (row.get("source_run"), row.get("model"), row.get("seed"))
        cell_key = run_key + (row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("score_step_idx"))
        condition = str(row.get("condition_id"))
        cell_condition_values[cell_key][condition].append(value)
        population[run_key].add(condition)

    out: list[dict[str, Any]] = []
    reduced_cells: dict[tuple[Any, ...], dict[str, float]] = {}
    for cell_key, condition_values in cell_condition_values.items():
        reduced = {condition: float(np.median(values)) for condition, values in condition_values.items() if values}
        if not reduced:
            continue
        reduced_cells[cell_key] = reduced
        reduced_values = np.asarray(list(reduced.values()), dtype=np.float64)
        reduced_logs = np.log10(np.maximum(reduced_values, 1.0e-30))
        out.append({
            "source_run": cell_key[0], "model": cell_key[1], "seed": cell_key[2],
            "module_family": cell_key[3], "module_name": cell_key[4],
            "layer_idx": cell_key[5], "score_step_idx": cell_key[6],
            "rho_mean": float(np.mean(list(reduced.values()))),
            "rho_median": float(np.median(list(reduced.values()))),
            "log_rho_mad": float(np.median(np.abs(reduced_logs - np.median(reduced_logs)))),
            "log_rho_iqr": float(np.percentile(reduced_logs, 75) - np.percentile(reduced_logs, 25)),
            "num_conditions": len(reduced), "rho_scope": "condition", "valid": True,
        })

    def sizes_for(model: Any) -> list[int]:
        configured = config.get("subset_sizes", [])
        if isinstance(configured, dict):
            configured = configured.get(str(model), [])
        return [int(x) for x in configured]

    fractions = [float(x) for x in config.get("cache_fractions", [0.1, 0.2, 0.3, 0.5])]
    subset_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config.get("shuffle_seed", 2027)))
    for run_key, conditions in sorted(population.items(), key=lambda item: tuple(str(x) for x in item[0])):
        condition_ids = sorted(conditions)
        cells = {key: values for key, values in reduced_cells.items() if key[:3] == run_key}
        if not cells:
            continue
        for size in sizes_for(run_key[1]):
            if size > len(condition_ids):
                continue
            for trial in range(int(config.get("subset_trials", 50))):
                selected = sorted(str(x) for x in rng.choice(condition_ids, size=size, replace=False).tolist())
                common = [key for key, values in cells.items() if all(condition in values for condition in selected)]
                if not common:
                    continue
                ref = [float(np.mean([cells[key][condition] for condition in condition_ids if condition in cells[key]])) for key in common]
                est = [float(np.mean([cells[key][condition] for condition in selected])) for key in common]
                spearman, kendall = rank_corr(ref, est)
                for fraction in fractions:
                    subset_rows.append({
                        "model": run_key[1], "source_run": run_key[0], "seed": run_key[2],
                        "subset_size": size, "trial": trial,
                        "selected_condition_ids": json.dumps(selected, separators=(",", ":")),
                        "cache_fraction": fraction, "spearman": spearman, "kendall": kendall,
                        "jaccard": jaccard_at_fraction(ref, est, fraction),
                        "num_reference_conditions": len(condition_ids), "num_cells": len(common),
                        "num_points": len(common),
                    })
    return out, subset_rows


def _summary(ccmr: list[dict[str, Any]], rho_stability: list[dict[str, Any]], subset_rows: list[dict[str, Any]], runs: list[dict[str, Any]]):
    summary: dict[str, Any] = {"runs": [{"run_id": r["summary"].get("run_id"), "path": str(r["path"]), "summary": r["summary"]} for r in runs], "models": {}, "invalid": {}, "subset_stability": {}}
    cell_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in ccmr:
        model = str(row.get("model")); module = f"{row.get('module_family')}.{row.get('module_name')}"
        value = _finite_float(row.get("g_ccmr_db"))
        if value is not None:
            cell_values[(model, str(row.get("source_run")), str(row.get("seed")), module)].append(value)
        if str(row.get("valid", "True")).lower() == "false":
            summary["invalid"][model] = summary["invalid"].get(model, 0) + 1
    seed_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (model, source, seed, module), values in cell_values.items():
        seed_values[(model, source, module)].append(float(np.mean(values)))
    for (model, source, module), values in seed_values.items():
        stats = percentile_ci(values, 2000)
        stats.update({"std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "min": float(np.min(values)), "max": float(np.max(values)), "num_seeds": len(values)})
        summary["models"].setdefault(model, {}).setdefault(module, {}).setdefault("gain_db", {})[source] = stats
    for row in subset_rows:
        group = (str(row.get("model")), str(row.get("source_run")), int(row.get("subset_size", 0)), float(row.get("cache_fraction", 0.0)))
        key = "|".join(str(x) for x in group)
        summary["subset_stability"].setdefault(key, {})
        for metric in ("spearman", "kendall", "jaccard"):
            value = _finite_float(row.get(metric))
            if value is not None:
                summary["subset_stability"][key].setdefault(metric, []).append(value)
    for key, metrics in list(summary["subset_stability"].items()):
        summary["subset_stability"][key] = {metric: percentile_ci(values, 1000) for metric, values in metrics.items()}
    summary["rho_stability_points"] = len(rho_stability)
    summary["ccmr_points"] = len(ccmr)
    summary["subset_stability_rows"] = len(subset_rows)
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
    tables["time_gap_metrics"] = _recompute_pairwise_time_gap(tables["time_gap_metrics"])
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
