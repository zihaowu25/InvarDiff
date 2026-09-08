#!/usr/bin/env python3
"""Compose the gated ICLR CCMR A-D main figure from formal scalar tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image

from common import read_rows, save_json_atomic, sha256_file, utc_now
from formal_protocol import alignment_hierarchical_summary, is_true, subset_hierarchical_summary
from validate_artifacts import validate


def _module(row):
    return f"{row.get('module_family')}.{row.get('module_name')}"


def _qa_images(png: Path, output: Path) -> list[Path]:
    image = Image.open(png).convert("RGB")
    gray = output / "qa" / "fig_ccmr_main_grayscale.png"
    gray.parent.mkdir(parents=True, exist_ok=True)
    image.convert("L").save(gray, dpi=(300, 300))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    matrix = np.asarray([[0.367, 0.861, -0.228], [0.280, 0.673, 0.047], [-0.012, 0.043, 0.969]])
    simulated = np.clip(rgb @ matrix.T, 0, 1)
    deut = output / "qa" / "fig_ccmr_main_deuteranopia.png"
    Image.fromarray((simulated * 255).astype(np.uint8)).save(deut, dpi=(300, 300))
    return [gray, deut]


def _subset_hierarchical_summary(rows, metric: str, trials: int = 2000, seed: int = 2027):
    return subset_hierarchical_summary(rows, metric, trials=trials, random_seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose gated CCMR main figure")
    parser.add_argument("--combined", required=True)
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dit-run", required=True)
    parser.add_argument("--flux-pair-run", required=True)
    parser.add_argument("--flux-rho-run", required=True)
    parser.add_argument("--tests-log", required=True)
    args = parser.parse_args()
    combined = Path(args.combined).resolve(); tables_dir = Path(args.tables_dir).resolve()
    gate = validate(Path(args.dit_run).resolve(), Path(args.flux_pair_run).resolve(), Path(args.flux_rho_run).resolve(), combined, Path(args.tests_log).resolve(), tables_dir)
    if not gate["passed"]:
        print(json.dumps(gate, indent=2))
        raise SystemExit("Formal validation failed; refusing to create or overwrite paper figure")

    ccmr = [row for row in read_rows(combined / "ccmr_metrics.csv.gz") if is_true(row.get("valid", True)) and not is_true(row.get("degenerate", False))]
    alignment = [row for row in read_rows(combined / "alignment_cluster_summary.csv.gz") if is_true(row.get("valid", True))]
    subsets = [row for row in read_rows(combined / "subset_stability.csv.gz") if is_true(row.get("valid", True)) and str(row.get("is_primary", "true")).lower() == "true"]
    if not ccmr or not alignment or not subsets or {row.get("model") for row in ccmr} != {"dit", "flux"}:
        raise SystemExit("Formal scalar tables are incomplete; no pilot fallback is permitted")

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(6.75, 4.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    # A: separate model facets share axes and reference semantics.
    a_grid = grid[0, 0].subgridspec(1, 2, wspace=0.08)
    ax_a_dit = fig.add_subplot(a_grid[0, 0])
    ax_a_flux = fig.add_subplot(a_grid[0, 1], sharex=ax_a_dit, sharey=ax_a_dit)
    colors = {"dit": "#0072B2", "flux": "#D55E00"}
    positives = [float(row[key]) for row in ccmr for key in ("v_base", "v_diff") if float(row[key]) > 0]
    low, high = np.percentile(positives, [0.1, 99.9]); reference = np.geomspace(low, high, 100)
    for model, ax_a in (("dit", ax_a_dit), ("flux", ax_a_flux)):
        rows = [row for row in ccmr if row.get("model") == model]
        x = np.asarray([float(row["v_base"]) for row in rows]); y = np.asarray([float(row["v_diff"]) for row in rows])
        ax_a.scatter(x, y, s=1.2, alpha=0.08, rasterized=True, color=colors[model])
        for gain, style in ((0, "-"), (10, "--"), (20, ":")):
            ax_a.plot(reference, reference / (10 ** (gain / 10)), color="0.25", lw=.65, ls=style,
                      label=f"{gain} dB" if model == "dit" else None)
        ax_a.set(xscale="log", yscale="log", xlabel=r"$V^{base}$", title=model.upper())
    ax_a_dit.set_ylabel(r"$V^{diff}$")
    ax_a_dit.text(-0.34, 1.08, "A  Condition variance", transform=ax_a_dit.transAxes, fontweight="bold")
    ax_a_dit.legend(frameon=False, loc="lower right", fontsize=6)
    ax_a_flux.tick_params(labelleft=False)

    # B: common 0-centered scale, normalized progress.
    ax_b = fig.add_subplot(grid[0, 1])
    families = sorted({_module(row) for row in ccmr}, key=lambda x: ("flux" not in x, x))
    labels = [] ; matrices = []
    for model in ("dit", "flux"):
        for family in sorted({_module(row) for row in ccmr if row.get("model") == model}):
            bins = [[] for _ in range(50)]
            for row in ccmr:
                if row.get("model") == model and _module(row) == family:
                    index = min(49, max(0, int(round(float(row.get("denoising_progress", 0)) * 49))))
                    bins[index].append(float(row["g_ccmr_db"]))
            matrices.append([np.median(values) if values else np.nan for values in bins]); labels.append(f"{model}:{family}")
    matrix = np.asarray(matrices); limit = max(1.0, float(np.nanpercentile(np.abs(matrix), 98)))
    image = ax_b.imshow(matrix, aspect="auto", cmap="BrBG", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), origin="lower")
    ax_b.set(title="B  Gain over denoising progress", xlabel="normalized denoising progress", yticks=np.arange(len(labels)), yticklabels=labels)
    ax_b.set_xticks([0, 24, 49], ["0", "0.5", "1"]); fig.colorbar(image, ax=ax_b, label="median gain (dB)", fraction=.046)

    # C: cluster-first seed points and paired ranges.
    ax_c = fig.add_subplot(grid[1, 0])
    positions = {"dit": 0, "flux": 1}
    for model in ("dit", "flux"):
        result = alignment_hierarchical_summary(
            [row for row in alignment if row.get("model") == model],
            trials=2000, random_seed=2027 + positions[model],
        )
        x = positions[model]
        for point_index, point in enumerate(result["seed_points"]):
            jitter = (point_index - (len(result["seed_points"]) - 1) / 2) * .025
            ax_c.plot([x + jitter, x + jitter], [point["shuffled"], point["aligned"]], color=colors[model], alpha=.25, lw=.6)
            ax_c.scatter([x + jitter], [point["aligned"]], color=colors[model], s=8, marker="o")
            ax_c.scatter([x + jitter], [point["shuffled"]], facecolors="none", edgecolors=colors[model], s=8, marker="o")
        ax_c.errorbar([x - .11], [result["aligned"]],
                      yerr=[[result["aligned"] - result["aligned_p025"]], [result["aligned_p975"] - result["aligned"]]],
                      fmt="o", color=colors[model], capsize=2, markersize=4, lw=.8)
        ax_c.errorbar([x + .11], [result["shuffled"]],
                      yerr=[[result["shuffled"] - result["shuffled_p025"]], [result["shuffled_p975"] - result["shuffled"]]],
                      fmt="o", markerfacecolor="white", markeredgecolor=colors[model], color=colors[model], capsize=2, markersize=4, lw=.8)
    ax_c.axhline(0, color="0.4", lw=.6); ax_c.set(xticks=[0, 1], xticklabels=["DiT", "FLUX"], ylabel="cluster mean gain (dB)", title="C  Aligned vs. shuffled")
    ax_c.text(.02, .02, "filled: aligned   open: shuffled", transform=ax_c.transAxes, va="bottom")

    # D: condition-mean primary estimator only; no pair-difference rho.
    ax_d = fig.add_subplot(grid[1, 1])
    panel_d_values = []
    full_sizes = {"dit": 16, "flux": 12}
    for model, color in colors.items():
        for metric, marker, linestyle in (("spearman", "o", "-"), ("jaccard", "s", "--")):
            points = {}
            for row in subsets:
                if row.get("model") != model:
                    continue
                if metric == "spearman" and row.get("metric_scope") != "rank":
                    continue
                if metric == "jaccard" and (row.get("metric_scope") != "overlap" or abs(float(row.get("cache_fraction", 0)) - .3) > 1e-9):
                    continue
                value = row.get(metric)
                if value not in (None, ""):
                    points.setdefault(int(float(row["subset_size"])), []).append(row)
            xs = sorted(points)
            summaries = {x: _subset_hierarchical_summary(points[x], metric, seed=2027 + x) for x in xs}
            ys = [summaries[x]["mean"] for x in xs]
            lows = [ys[index] - summaries[x]["p025"] for index, x in enumerate(xs)]
            highs = [summaries[x]["p975"] - ys[index] for index, x in enumerate(xs)]
            ax_d.plot(xs, ys, ls=linestyle, color=color, label=f"{model.upper()} {metric}")
            for index, x in enumerate(xs):
                hollow = x == full_sizes[model]
                ax_d.errorbar([x], [ys[index]], yerr=[[lows[index]], [highs[index]]], fmt=marker,
                              color=color, markerfacecolor="white" if hollow else color,
                              markeredgecolor=color, markersize=4, capsize=1.5, lw=.7)
                panel_d_values.extend([summaries[x]["p025"], summaries[x]["p975"]])
            for x in xs:
                ax_d.scatter([x] * len(summaries[x]["seed_points"]), summaries[x]["seed_points"], s=5, alpha=.25, color=color)
    lower = max(-1.03, min(panel_d_values + [0.0]) - 0.05)
    ax_d.set(xlabel="number of calibration conditions", ylabel="stability", ylim=(lower, 1.03), title="D  Few-sample $\\rho$ stability")
    ax_d.legend(frameon=False, ncol=2); ax_d.text(.02, .02, "condition mean; Jaccard@30%", transform=ax_d.transAxes)

    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    paths = {ext: output / f"fig_ccmr_main.{ext}" for ext in ("pdf", "svg", "png")}
    fig.savefig(paths["pdf"], bbox_inches="tight"); fig.savefig(paths["svg"], bbox_inches="tight"); fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    qa_paths = _qa_images(paths["png"], output)
    caption = output / "fig_ccmr_main_caption.md"
    caption.write_text("**CCMR mechanism evidence.** A, adjacent differencing suppresses condition variance; DiT and FLUX are shown in separate facets with shared axes. B, suppression across module families and normalized denoising progress. C, aligned and shuffled hierarchical means; families are macro-averaged within each seed and pair/derangement cluster, error bars are seed-then-cluster bootstrap 95% intervals, and small paired markers are seed means. FLUX uses 12 preregistered balanced-cycle pairs. D, stability of the condition-mean online cache score; lines show hierarchical means, error bars are 95% intervals from seed-then-subset resampling, faint dots are seed means, and hollow endpoints are full-condition references. Jaccard uses the lowest 30% of scores.\n", encoding="utf-8")
    artifacts = list(paths.values()) + qa_paths + [caption]
    manifest = {"created_at": utc_now(), "formal_gate": gate, "source_combined": str(combined), "pilot_fallback": False, "empty_panels": False, "width_inches": 6.75, "height_inches": 4.15, "minimum_font_pt": 7.0, "qa": {"grayscale": True, "deuteranopia": True, "png_dpi": 300, "pdf_fonttype": 42}, "artifacts": [{"path": str(path.relative_to(output)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts]}
    save_json_atomic(output / "fig_ccmr_main_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
