#!/usr/bin/env python3
"""CPU-only prefix stability diagnostic for the preregistered FLUX cycle."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_ccmr import _recompute_pairwise_ccmr
from common import rank_corr, read_rows, save_json_atomic, sha256_file, utc_now, write_rows_atomic


def _module(row):
    return f"{row.get('module_family')}.{row.get('module_name')}"


def _rows_for_prefix(run: Path, seed: int, count: int, table: str):
    rows = []
    for pair_idx in range(count):
        rows.extend(read_rows(run / "shards" / f"seed_{seed}_pair_{pair_idx}" / f"{table}.csv.gz"))
    for row in rows:
        row["source_run"] = run.name
        for key in ("seed", "layer_idx", "step_idx", "num_conditions", "v_raw_prev",
                    "raw_pair_distance", "raw_prev_pair_distance", "diff_pair_distance"):
            if row.get(key) not in (None, ""):
                try: row[key] = float(row[key])
                except (TypeError, ValueError): pass
    return rows


def main():
    parser = argparse.ArgumentParser(description="Diagnose FLUX CCMR stability versus pair count")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-counts", nargs="+", type=int, default=[4, 6, 8, 10, 12])
    args = parser.parse_args()
    run = Path(args.run).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "run_manifest.json").read_text())
    seeds = sorted({int(value["seed"]) for value in manifest["shards"].values()})
    reference = {}
    summaries = []
    for seed in seeds:
        by_count = {}
        for count in args.pair_counts:
            ccmr = _rows_for_prefix(run, seed, count, "ccmr_metrics")
            distance = _rows_for_prefix(run, seed, count, "condition_distance")
            alignment = _rows_for_prefix(run, seed, count, "alignment_control")
            cells = [row for row in _recompute_pairwise_ccmr(ccmr, distance) if str(row.get("valid", "true")).lower() == "true"]
            cell_map = {(_module(row), int(float(row["layer_idx"])), int(float(row["step_idx"]))): float(row["g_ccmr_db"]) for row in cells}
            by_count[count] = cell_map
            family_values = defaultdict(list)
            for key, value in cell_map.items(): family_values[key[0]].append(value)
            family_medians = {family: float(np.median(values)) for family, values in family_values.items()}
            alignment_values = defaultdict(list)
            for row in alignment:
                if row.get("delta_g_db") not in (None, ""):
                    alignment_values[_module(row)].append(float(row["delta_g_db"]))
            alignment_family = {family: float(np.mean(values)) for family, values in alignment_values.items()}
            summaries.append({"seed": seed, "pair_count": count, "aggregation": "pooled", "module_family": "ALL", "median_gain_db": float(np.median(list(cell_map.values()))), "aligned_shuffled_delta_g_db": float(np.mean([float(row["delta_g_db"]) for row in alignment if row.get("delta_g_db") not in (None, "")]))})
            summaries.append({"seed": seed, "pair_count": count, "aggregation": "macro", "module_family": "ALL", "median_gain_db": float(np.mean(list(family_medians.values()))), "aligned_shuffled_delta_g_db": float(np.mean(list(alignment_family.values())))})
            summaries.extend({"seed": seed, "pair_count": count, "aggregation": "module_family", "module_family": family, "median_gain_db": value, "aligned_shuffled_delta_g_db": alignment_family.get(family)} for family, value in sorted(family_medians.items()))
        reference[seed] = by_count[12]
        for count, cell_map in by_count.items():
            keys = sorted(set(cell_map) & set(reference[seed]))
            current = [cell_map[key] for key in keys]; target = [reference[seed][key] for key in keys]
            spearman, _ = rank_corr(current, target)
            delta = float(np.mean(np.abs(np.asarray(current) - np.asarray(target))))
            for row in summaries:
                if row["seed"] == seed and row["pair_count"] == count:
                    row["mean_absolute_difference_vs_12_db"] = delta
                    row["rank_correlation_vs_12"] = spearman
    csv = output / "pair_count_stability.csv"
    write_rows_atomic(csv, summaries)
    save_json_atomic(output / "pair_count_stability_manifest.json", {
        "created_at": utc_now(), "source_run": str(run), "pair_counts": args.pair_counts,
        "note": "Prefixes are correlated diagnostics; the formal estimate always uses all 12 preregistered pairs.",
        "artifact": {"path": csv.name, "sha256": sha256_file(csv), "row_count": len(summaries)},
    })
    print(json.dumps({"rows": len(summaries), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
