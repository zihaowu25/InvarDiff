#!/usr/bin/env python3
"""Render CCMR figures from aggregated scalar tables only."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_yaml, read_rows, save_json_atomic


HERE = Path(__file__).resolve().parent
TABLES = ("ccmr_metrics", "condition_similarity", "condition_distance", "temporal_metrics", "time_gap_metrics", "rho_per_condition", "rho_stability", "subset_stability")
MODULE_COLORS = {"attn": "#0072B2", "context_attn": "#CC79A7", "ff": "#E69F00", "context_ff": "#009E73", "mlp": "#D55E00", "msa": "#0072B2"}


def _read_tables(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for name in TABLES:
        path = input_dir / f"{name}.csv.gz"
        if not path.exists():
            path = input_dir / f"{name}.csv"
        out[name] = read_rows(path) if path.exists() else []
        for row in out[name]:
            for key, value in list(row.items()):
                if key in {"seed", "layer_idx", "step_idx", "score_step_idx", "time_gap", "subset_size", "trial", "num_points", "condition_i", "condition_j"}:
                    try: row[key] = int(float(value))
                    except (ValueError, TypeError): pass
                elif key not in {"run_id", "source_run", "model", "module_family", "module_name", "condition_id", "condition_i", "condition_j", "estimator", "valid", "pair_id_or_condition_group"}:
                    try: row[key] = float(value)
                    except (ValueError, TypeError): pass
    return out


def _save(fig, output: Path, name: str, dpi: int = 300):
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _finite(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key) not in (None, "", "nan", "NaN") and np.isfinite(float(r[key]))], dtype=float)


def _module_label(row):
    return f"{row.get('module_family')}.{row.get('module_name')}"


def _heat(rows, key, title, out, name, cmap, center=None, vmin=None, vmax=None, dpi=300):
    if not rows: return
    grouped = defaultdict(list)
    for row in rows:
        if row.get(key) in (None, ""): continue
        try: grouped[(int(row["step_idx"]), int(row["layer_idx"]))].append(float(row[key]))
        except (KeyError, ValueError, TypeError): continue
    if not grouped: return
    steps = sorted({k[0] for k in grouped}); layers = sorted({k[1] for k in grouped})
    matrix = np.full((len(layers), len(steps)), np.nan)
    for (s, l), values in grouped.items(): matrix[layers.index(l), steps.index(s)] = float(np.median(values))
    fig, ax = plt.subplots(figsize=(max(6, len(steps) * .22), max(4, len(layers) * .18)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(title=title, xlabel="inference execution step", ylabel="layer (0 at top)")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, rotation=90, fontsize=7)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    _save(fig, out, name, dpi)


def _ecdf(ax, values, label, color=None):
    values = np.sort(np.asarray(values, dtype=float))
    if values.size: ax.plot(values, np.arange(1, values.size + 1) / values.size, label=label, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CCMR scalar results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    data = _read_tables(Path(args.input_dir).resolve())
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    ccmr = data["ccmr_metrics"]; rho = data["rho_per_condition"]; stability = data["rho_stability"]
    values = {key: _finite(ccmr, key) for key in ("v_raw", "v_diff", "g_ccmr_db", "r_ccmr")}
    rho_values = np.log10(np.maximum(_finite(rho, "rho_clean"), 1e-30))
    resolved = dict(config)
    resolved["global_limits"] = {
        "log_v": [float(np.percentile(np.log10(np.maximum(np.concatenate([values["v_raw"], values["v_diff"]]), 1e-30)), q)) for q in config.get("global_robust_quantiles", [0.005, 0.995])] if values["v_raw"].size and values["v_diff"].size else [None, None],
        "gain_db": [float(np.percentile(values["g_ccmr_db"], q)) for q in config.get("global_robust_quantiles", [0.005, 0.995])] if values["g_ccmr_db"].size else [None, None],
        "log_rho": [float(np.percentile(rho_values, q)) for q in config.get("global_robust_quantiles", [0.005, 0.995])] if rho_values.size else [None, None],
    }
    save_json_atomic(out / "plot_config_resolved.json", resolved)
    dpi = int(config.get("png_dpi", 300))
    gain_lim = resolved["global_limits"]["gain_db"]
    for model in ("dit", "flux"):
        model_rows = [r for r in ccmr if r.get("model") == model]
        _heat(model_rows, "v_raw", f"{model.upper()} log10 V(raw)", out, f"ccmr_variance_heatmaps_{model}_raw", config.get("sequential_cmap", "cividis"), dpi=dpi)
        _heat(model_rows, "v_diff", f"{model.upper()} log10 V(diff)", out, f"ccmr_variance_heatmaps_{model}_diff", config.get("sequential_cmap", "cividis"), dpi=dpi)
        _heat(model_rows, "g_ccmr_db", f"{model.upper()} CCMR gain (dB)", out, f"ccmr_variance_heatmaps_{model}_gain", config.get("gain_cmap", "BrBG"), vmin=gain_lim[0] if gain_lim[0] is not None else None, vmax=gain_lim[1] if gain_lim[1] is not None else None, dpi=dpi)
    if values["v_raw"].size and values["v_diff"].size:
        n = min(values["v_raw"].size, values["v_diff"].size)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, model in zip(axes, ("dit", "flux")):
            rows = [r for r in ccmr if r.get("model") == model and r.get("v_base") not in (None, "") and r.get("v_diff") not in (None, "")]
            x = np.log10(np.maximum(_finite(rows, "v_base"), 1e-30)); y = np.log10(np.maximum(_finite(rows, "v_diff"), 1e-30))
            if len(x): ax.hexbin(x, y, gridsize=35, bins="log", cmap=config.get("density_cmap", "cividis"), mincnt=1)
            lo, hi = -30, 20
            ax.plot([lo, hi], [lo, hi], "k--", lw=.8); ax.plot([lo, hi], [lo-1, hi-1], "k:", lw=.8); ax.plot([lo, hi], [lo-2, hi-2], "k-.", lw=.8)
            ax.set(title=model.upper(), xlabel="log10 V(base)", ylabel="log10 V(diff)"); ax.grid(alpha=.2)
        fig.suptitle("CCMR raw-vs-difference variance"); _save(fig, out, "ccmr_raw_vs_diff_hexbin", dpi)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("dit", "flux"):
        vals = _finite([r for r in ccmr if r.get("model") == model], "g_ccmr_db")
        _ecdf(ax, vals, model.upper())
    for mark in (0, 10, 20): ax.axvline(mark, ls="--", lw=.7, color="gray")
    ax.set(xlabel="CCMR gain (dB)", ylabel="ECDF", title="CCMR gain ECDF"); ax.legend(); ax.grid(alpha=.2); _save(fig, out, "ccmr_gain_ecdf", dpi)
    groups = defaultdict(list)
    for row in ccmr:
        if row.get("g_ccmr_db") not in (None, ""): groups[_module_label(row)].append(float(row["g_ccmr_db"]))
    if groups:
        labels = sorted(groups); fig, ax = plt.subplots(figsize=(max(7, len(labels)*.55), 4.5)); ax.violinplot([groups[x] for x in labels], showmedians=True); ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8); ax.set_ylabel("gain (dB)"); ax.set_title("CCMR gain by module"); ax.grid(alpha=.2); _save(fig, out, "ccmr_gain_violin", dpi)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, ax in zip(("dit", "flux"), axes):
        rows = [r for r in ccmr if r.get("model") == model and r.get("g_ccmr_db") not in (None, "")]; grouped = defaultdict(list)
        for r in rows: grouped[int(r["step_idx"])].append(float(r["g_ccmr_db"]))
        xs = sorted(grouped); ax.plot(xs, [np.median(grouped[x]) for x in xs], "o-"); ax.set(title=model.upper(), xlabel="step", ylabel="median gain (dB)"); ax.grid(alpha=.2)
    fig.suptitle("CCMR gain over time"); _save(fig, out, "ccmr_gain_over_time", dpi)
    sim = data["condition_similarity"]
    fig, ax = plt.subplots(figsize=(7, 4.5)); grouped = defaultdict(list)
    for r in sim:
        if r.get("adjacent_condition_cosine") not in (None, ""): grouped[_module_label(r)].append(float(r["adjacent_condition_cosine"]))
    if grouped:
        ax.boxplot([grouped[x] for x in sorted(grouped)], tick_labels=sorted(grouped), showfliers=False)
        ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("adjacent condition cosine"); ax.set_title("Condition adjacent similarity"); ax.grid(alpha=.2); _save(fig, out, "condition_adjacent_similarity_heatmap", dpi)
    fig, ax = plt.subplots(figsize=(7, 4.5)); aligned = _finite(sim, "aligned_g_ccmr_db"); shuffled = _finite(sim, "shuffled_g_ccmr_db");
    if aligned.size and shuffled.size: ax.boxplot([aligned, shuffled], tick_labels=["aligned", "shuffled"], showfliers=False)
    ax.set_ylabel("gain (dB)"); ax.set_title("Condition alignment shuffle control"); ax.grid(alpha=.2); _save(fig, out, "condition_alignment_shuffle", dpi)
    distances = data["condition_distance"]
    for model in ("dit", "flux"):
        rows = [r for r in distances if r.get("model") == model]
        if not rows: continue
        cond = sorted({str(r.get("condition_i")) for r in rows} | {str(r.get("condition_j")) for r in rows}); matrix = np.full((len(cond), len(cond)), np.nan); idx = {x:i for i,x in enumerate(cond)}
        for r in rows:
            i, j = idx[str(r["condition_i"])], idx[str(r["condition_j"])]
            matrix[i,j] = matrix[j,i] = float(r["raw_pair_distance"])
        np.fill_diagonal(matrix, 0)
        fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(matrix, cmap="cividis"); ax.set_title(f"{model.upper()} condition distance"); ax.set_xticks(range(len(cond)), cond, rotation=90); ax.set_yticks(range(len(cond)), cond); fig.colorbar(im, ax=ax); _save(fig, out, f"condition_distance_matrices_{model}", dpi)
    rho_rows = [r for r in stability if r.get("rho_mean") not in (None, "")]
    if rho_rows:
        fig, ax = plt.subplots(figsize=(7,4.5));
        for model in ("dit", "flux"):
            vals = np.log10(np.maximum(_finite([r for r in rho_rows if r.get("model") == model], "rho_mean"), 1e-30)); _ecdf(ax, vals, model.upper())
        ax.axvline(0, ls="--", color="gray"); ax.set(xlabel="log10 rho", ylabel="ECDF", title="rho distribution"); ax.legend(); ax.grid(alpha=.2); _save(fig, out, "rho_distributions", dpi)
        for key, name in (("rho_mean", "rho_mean_heatmaps"), ("rho_median", "rho_median_heatmaps"), ("log_rho_mad", "rho_condition_dispersion_heatmaps")):
            for model in ("dit", "flux"):
                _heat([r for r in rho_rows if r.get("model") == model], key, f"{model.upper()} {key}", out, f"{name}_{model}", config.get("rho_dispersion_cmap", "magma") if "mad" in key else config.get("rho_cmap", "PuOr"), center=0, dpi=dpi)
    subset = data["subset_stability"]
    if subset:
        fig, ax = plt.subplots(figsize=(7, 4.5)); grouped = defaultdict(list)
        for r in subset:
            if r.get("jaccard") not in (None, ""): grouped[int(r["subset_size"])].append(float(r["jaccard"]))
        xs=sorted(grouped); ax.plot(xs, [np.mean(grouped[x]) for x in xs], "o-"); ax.set(xlabel="subset size", ylabel="Cache Book Jaccard", title="Few-sample subset stability"); ax.grid(alpha=.2); _save(fig, out, "rho_few_sample_vs_reference", dpi)
    temporal = data["temporal_metrics"]
    if temporal:
        x = _finite(temporal, "output_rms"); y = _finite(temporal, "diff_rms");
        fig, ax = plt.subplots(figsize=(6,5)); ax.hexbin(np.log10(np.maximum(x,1e-30)), np.log10(np.maximum(y,1e-30)), gridsize=35, bins="log", cmap=config.get("density_cmap", "cividis"), mincnt=1); lo,hi=-30,10; ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="log10 A", ylabel="log10 D", title="Temporal smoothness"); _save(fig, out, "temporal_smoothness_hexbin", dpi)
        fig, ax = plt.subplots(figsize=(7,4.5)); _ecdf(ax, np.log10(np.maximum(_finite(temporal,"r_time"),1e-30)), "all"); ax.set(xlabel="log10 r_time", ylabel="ECDF", title="Temporal rate ECDF"); ax.grid(alpha=.2); _save(fig, out, "temporal_rate_ecdf", dpi)
    gaps = data["time_gap_metrics"]
    if gaps:
        grouped=defaultdict(list)
        for r in gaps:
            if r.get("g_ccmr_gap_db") not in (None, ""): grouped[int(r["time_gap"])].append(float(r["g_ccmr_gap_db"]))
        fig, ax = plt.subplots(figsize=(7,4.5)); xs=sorted(grouped); ax.plot(xs,[np.median(grouped[x]) for x in xs],"o-"); ax.set(xlabel="time gap",ylabel="median gain (dB)",title="CCMR time-gap ablation"); ax.grid(alpha=.2); _save(fig,out,"ccmr_time_gap_ablation",dpi)
    save_json_atomic(out / "figure_manifest.json", {"output_dir": str(out), "figures": sorted(p.name for p in out.iterdir() if p.suffix in {".pdf", ".png"})})
    print(json.dumps({"output": str(out), "figures": len(list(out.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
