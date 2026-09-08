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
from formal_protocol import is_true
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
    """Resample seeds first and correlated subset trials within each seed."""
    by_seed = {}
    for row in rows:
        if row.get(metric) not in (None, ""):
            by_seed.setdefault(str(row["seed"]), []).append(float(row[metric]))
    if not by_seed:
        return None
    seed_ids = sorted(by_seed)
    seed_points = [float(np.mean(by_seed[item])) for item in seed_ids]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(trials):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        values = []
        for sampled_seed in sampled_seeds:
            subsets = np.asarray(by_seed[str(sampled_seed)], dtype=np.float64)
            values.append(float(np.mean(rng.choice(subsets, size=len(subsets), replace=True))))
        boot.append(float(np.mean(values)))
    return {
        "mean": float(np.mean(seed_points)), "seed_points": seed_points,
        "p025": float(np.percentile(boot, 2.5)), "p975": float(np.percentile(boot, 97.5)),
    }


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
    fig = plt.figure(figsize=(6.75, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    # A: two inset facets share axes and reference semantics.
    ax_a = fig.add_subplot(grid[0, 0])
    colors = {"dit": "#0072B2", "flux": "#D55E00"}
    for model in ("dit", "flux"):
        rows = [row for row in ccmr if row.get("model") == model]
        x = np.asarray([float(row["v_base"]) for row in rows]); y = np.asarray([float(row["v_diff"]) for row in rows])
        ax_a.scatter(x, y, s=1.2, alpha=0.08, rasterized=True, color=colors[model], label=model.upper())
    positives = [float(row[key]) for row in ccmr for key in ("v_base", "v_diff") if float(row[key]) > 0]
    low, high = np.percentile(positives, [0.1, 99.9]); reference = np.geomspace(low, high, 100)
    for gain, style in ((0, "-"), (10, "--"), (20, ":")):
        ax_a.plot(reference, reference / (10 ** (gain / 10)), color="0.25", lw=.65, ls=style, label=f"{gain} dB" if gain else "0 dB")
    ax_a.set(xscale="log", yscale="log", xlabel=r"$V^{base}$", ylabel=r"$V^{diff}$", title="A  Condition variance")
    ax_a.legend(ncol=2, frameon=False, loc="lower right")

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
    grouped = {}
    for row in alignment:
        # Module families are macro-aggregated inside each model/seed before
        # drawing independent seed-level paired points.
        key = (row["model"], row["seed"])
        grouped.setdefault(key, [[], []]); grouped[key][0].append(float(row["g_aligned_db"])); grouped[key][1].append(float(row["g_shuffled_db"]))
    positions = {"dit": 0, "flux": 1}
    for (model, seed), (aligned_values, shuffled_values) in grouped.items():
        x = positions[model] + (int(float(seed)) - 2) * .025
        aligned_value, shuffled_value = np.mean(aligned_values), np.mean(shuffled_values)
        ax_c.plot([x, x], [shuffled_value, aligned_value], color=colors[model], alpha=.25, lw=.6)
        ax_c.scatter([x], [aligned_value], color=colors[model], s=8, marker="o")
        ax_c.scatter([x], [shuffled_value], facecolors="none", edgecolors=colors[model], s=8, marker="o")
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
    caption.write_text("**CCMR mechanism evidence.** A, adjacent differencing suppresses condition variance. B, suppression across module families and normalized denoising progress. C, aligned and shuffled gains are macro-aggregated across module families and paired at the seed level. D, stability of the condition-mean online cache score; lines show hierarchical means, error bars are 95% intervals from seed-then-subset resampling, faint dots are seed means, and hollow endpoints are full-condition references. Jaccard uses the lowest 30% of scores.\n", encoding="utf-8")
    artifacts = list(paths.values()) + qa_paths + [caption]
    manifest = {"created_at": utc_now(), "formal_gate": gate, "source_combined": str(combined), "pilot_fallback": False, "empty_panels": False, "width_inches": 6.75, "minimum_font_pt": 7.0, "qa": {"grayscale": True, "deuteranopia": True, "png_dpi": 300, "pdf_fonttype": 42}, "artifacts": [{"path": str(path.relative_to(output)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifacts]}
    save_json_atomic(output / "fig_ccmr_main_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
