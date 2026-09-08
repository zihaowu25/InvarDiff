#!/usr/bin/env python3
"""Export auditable CCMR paper tables from formal scalar aggregates."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import read_rows, save_json_atomic, sha256_file, utc_now, write_rows_atomic
from formal_protocol import is_true
from validate_artifacts import validate


def _module(row: dict[str, Any]) -> str:
    return f"{row.get('module_family')}.{row.get('module_name')}"


def _summary_rows(ccmr: list[dict[str, Any]], alignment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ccmr:
        groups[(str(row.get("model")), _module(row))].append(row)
    alignment_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in alignment:
        alignment_groups[(str(row.get("model")), str(row.get("module_family")))].append(float(row["delta_g_db"]))
    result = []
    model_family_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, family), rows in sorted(groups.items()):
        valid = [row for row in rows if is_true(row.get("valid", True))]
        gains = np.asarray([float(row["g_ccmr_db"]) for row in valid], dtype=np.float64)
        ratios = np.asarray([float(row["r_ccmr"]) for row in valid], dtype=np.float64)
        deltas = np.asarray(alignment_groups.get((model, family), []), dtype=np.float64)
        item = {
            "aggregation": "module_family", "model": model, "module_family": family,
            "valid": len(valid), "invalid": len(rows) - len(valid),
            "degenerate": sum(is_true(row.get("degenerate", False)) for row in valid),
            "nan": int(np.isnan(gains).sum()), "inf": int(np.isinf(gains).sum()),
            "gain_median_db": float(np.median(gains)),
            "gain_iqr_db": float(np.percentile(gains, 75) - np.percentile(gains, 25)),
            "gain_p5_db": float(np.percentile(gains, 5)), "gain_p90_db": float(np.percentile(gains, 90)),
            "gain_p95_db": float(np.percentile(gains, 95)),
            "geometric_mean_r": float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-30))))),
            "p_gain_gt_0": float(np.mean(gains > 0)), "p_gain_gt_10": float(np.mean(gains > 10)),
            "p_gain_gt_20": float(np.mean(gains > 20)), "p_gain_le_0": float(np.mean(gains <= 0)),
            "delta_gain_median_db": float(np.median(deltas)) if deltas.size else float("nan"),
            "delta_gain_iqr_db": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)) if deltas.size else float("nan"),
            "delta_gain_p5_db": float(np.percentile(deltas, 5)) if deltas.size else float("nan"),
            "p_delta_gain_gt_0": float(np.mean(deltas > 0)) if deltas.size else float("nan"),
        }
        result.append(item)
        model_family_stats[model].append(item)
    for model, items in model_family_stats.items():
        for aggregation in ("macro", "pooled"):
            model_rows = [row for row in ccmr if str(row.get("model")) == model and is_true(row.get("valid", True))]
            gains = np.asarray([float(row["g_ccmr_db"]) for row in model_rows])
            if aggregation == "macro":
                result.append({
                    "aggregation": "macro", "model": model, "module_family": "ALL",
                    "valid": sum(int(item["valid"]) for item in items),
                    "invalid": sum(int(item["invalid"]) for item in items),
                    "gain_median_db": float(np.mean([item["gain_median_db"] for item in items])),
                    "gain_iqr_db": float(np.mean([item["gain_iqr_db"] for item in items])),
                    "gain_p5_db": float(np.mean([item["gain_p5_db"] for item in items])),
                    "gain_p90_db": float(np.mean([item["gain_p90_db"] for item in items])),
                    "gain_p95_db": float(np.mean([item["gain_p95_db"] for item in items])),
                    "p_gain_gt_0": float(np.mean([item["p_gain_gt_0"] for item in items])),
                    "p_gain_gt_10": float(np.mean([item["p_gain_gt_10"] for item in items])),
                    "p_gain_gt_20": float(np.mean([item["p_gain_gt_20"] for item in items])),
                    "p_gain_le_0": float(np.mean([item["p_gain_le_0"] for item in items])),
                })
            else:
                result.append({
                    "aggregation": "pooled", "model": model, "module_family": "ALL",
                    "valid": len(model_rows), "invalid": len([row for row in ccmr if str(row.get("model")) == model]) - len(model_rows),
                    "gain_median_db": float(np.median(gains)),
                    "gain_iqr_db": float(np.percentile(gains, 75) - np.percentile(gains, 25)),
                    "gain_p5_db": float(np.percentile(gains, 5)), "gain_p90_db": float(np.percentile(gains, 90)),
                    "gain_p95_db": float(np.percentile(gains, 95)),
                    "p_gain_gt_0": float(np.mean(gains > 0)), "p_gain_gt_10": float(np.mean(gains > 10)),
                    "p_gain_gt_20": float(np.mean(gains > 20)), "p_gain_le_0": float(np.mean(gains <= 0)),
                })
    return result


def _rho_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("is_primary", "true")).lower() != "true" or not is_true(row.get("valid", True)):
            continue
        metric = "rank" if row.get("metric_scope") == "rank" else "jaccard30"
        if metric == "jaccard30" and abs(float(row.get("cache_fraction", 0)) - 0.3) > 1e-9:
            continue
        for name in (("spearman", "kendall") if metric == "rank" else ("jaccard",)):
            if row.get(name) in (None, ""):
                continue
            key = (row.get("model"), row.get("module_family"), int(float(row.get("subset_size"))), name)
            grouped[key].append(float(row[name]))
    return [{
        "model": key[0], "module_family": key[1], "subset_size": key[2], "metric": key[3],
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p025": float(np.percentile(values, 2.5)), "p975": float(np.percentile(values, 97.5)), "n": len(values),
    } for key, values in sorted(grouped.items())]


def _markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows.\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def _latex(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "% No rows.\n"
    keys = list(rows[0])
    def esc(value: Any) -> str:
        return str(value).replace("_", "\\_").replace("%", "\\%")
    lines = ["\\begin{tabular}{" + "l" * len(keys) + "}", " \\toprule", " & ".join(esc(key) for key in keys) + " \\\\", " \\midrule"]
    lines.extend(" & ".join(esc(row.get(key, "")) for key in keys) + " \\\\" for row in rows)
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CCMR paper tables from scalar data")
    parser.add_argument("--combined", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dit-run", required=True)
    parser.add_argument("--flux-pair-run", required=True)
    parser.add_argument("--flux-rho-run", required=True)
    parser.add_argument("--tests-log", required=True)
    args = parser.parse_args()
    combined = Path(args.combined).resolve()
    gate = validate(Path(args.dit_run).resolve(), Path(args.flux_pair_run).resolve(), Path(args.flux_rho_run).resolve(), combined, Path(args.tests_log).resolve())
    if not gate["passed"]:
        print(json.dumps(gate, indent=2))
        raise SystemExit("Formal core validation failed; refusing table export")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(read_rows(combined / "ccmr_metrics.csv.gz"), read_rows(combined / "alignment_cluster_summary.csv.gz"))
    rho = _rho_rows(read_rows(combined / "subset_stability.csv.gz"))
    artifacts = []
    for stem, rows in (("table_ccmr_summary", summary), ("table_rho_stability", rho)):
        csv_path = output / f"{stem}.csv"; write_rows_atomic(csv_path, rows)
        md_path = output / f"{stem}.md"; md_path.write_text(_markdown(rows), encoding="utf-8")
        tex_path = output / f"{stem}.tex"; tex_path.write_text(_latex(rows), encoding="utf-8")
        artifacts.extend([csv_path, md_path, tex_path])
    macros = ["% Generated from formal_v2 scalar aggregates; do not edit manually."]
    for row in summary:
        if row.get("aggregation") == "pooled":
            name = str(row["model"]).title().replace("_", "")
            macros.append(f"\\newcommand{{\\CCMR{name}MedianGain}}{{{float(row['gain_median_db']):.3f}}}")
    macro_path = output / "ccmr_paper_numbers.tex"; macro_path.write_text("\n".join(macros) + "\n", encoding="utf-8"); artifacts.append(macro_path)
    manifest = {"created_at": utc_now(), "source": str(combined), "artifacts": [{"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts]}
    save_json_atomic(output / "table_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
