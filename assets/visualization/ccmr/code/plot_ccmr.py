#!/usr/bin/env python3
"""Render CCMR figures from aggregated scalar tables only.

The plotting layer is deliberately defensive: invalid first-step cells are
expected in the collector output and must be masked consistently in every
figure.  No activation tensor is loaded here.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_yaml, read_rows, save_json_atomic


TABLES = (
    "ccmr_metrics",
    "condition_similarity",
    "condition_distance",
    "temporal_metrics",
    "time_gap_metrics",
    "rho_per_condition",
    "rho_stability",
    "subset_stability",
)

# Okabe-Ito-inspired colors.  The module name is used as the final fallback
# so newly added module families remain visible without a code change.
MODULE_COLORS = {
    "attn": "#0072B2",
    "context_attn": "#CC79A7",
    "ff": "#E69F00",
    "context_ff": "#009E73",
    "mlp": "#D55E00",
    "msa": "#0072B2",
}


def _read_tables(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read scalar tables and cast known numeric fields.

    ``float('nan')`` is intentionally retained in rows so the invalid cells
    remain auditable.  All plotting paths go through ``_finite`` or
    ``_finite_value`` before using numeric values.
    """

    out: dict[str, list[dict[str, Any]]] = {}
    integer_keys = {
        "seed",
        "layer_idx",
        "step_idx",
        "score_step_idx",
        "time_gap",
        "subset_size",
        "trial",
        "num_points",
        "condition_i",
        "condition_j",
    }
    string_keys = {
        "run_id",
        "source_run",
        "model",
        "module_family",
        "module_name",
        "condition_id",
        "condition_i",
        "condition_j",
        "estimator",
        "valid",
        "pair_id_or_condition_group",
    }
    for name in TABLES:
        path = input_dir / f"{name}.csv.gz"
        if not path.exists():
            path = input_dir / f"{name}.csv"
        rows = read_rows(path) if path.exists() else []
        for row in rows:
            for key, value in list(row.items()):
                if key in integer_keys:
                    try:
                        row[key] = int(float(value))
                    except (ValueError, TypeError):
                        pass
                elif key not in string_keys:
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        out[name] = rows
    return out


def _save(fig: Any, output: Path, name: str, dpi: int = 300) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _empty_figure(output: Path, name: str, title: str, message: str, dpi: int) -> None:
    """Write an explicit no-data figure instead of a misleading blank plot."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    _save(fig, output, name, dpi)


def _finite_value(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _finite(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = _finite_value(row.get(key))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def _paired_finite(
    rows: list[dict[str, Any]], x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite pairs; never filter x and y independently."""

    pairs = []
    for row in rows:
        x = _finite_value(row.get(x_key))
        y = _finite_value(row.get(y_key))
        if x is not None and y is not None:
            pairs.append((x, y))
    if not pairs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    values = np.asarray(pairs, dtype=float)
    return values[:, 0], values[:, 1]


def _module_label(row: dict[str, Any]) -> str:
    return f"{row.get('module_family')}.{row.get('module_name')}"


def _module_color(label: str) -> str:
    return MODULE_COLORS.get(label.split(".")[-1], "#999999")


def _cmap(name: str, mask_color: str):
    cmap = plt.get_cmap(name)
    try:
        cmap = cmap.copy()
    except AttributeError:
        pass
    cmap.set_bad(mask_color)
    return cmap


def _parse_index(value: Any) -> int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed)


def _heat_matrix(
    rows: list[dict[str, Any]],
    key: str,
    step_key: str,
    transform: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, list[int], list[int]] | None:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_value(row.get(key))
        step = _parse_index(row.get(step_key))
        layer = _parse_index(row.get("layer_idx"))
        if value is None or step is None or layer is None:
            continue
        if transform is not None:
            value = transform(value)
        if math.isfinite(value):
            grouped[(step, layer)].append(value)
    if not grouped:
        return None
    steps = sorted({item[0] for item in grouped})
    layers = sorted({item[1] for item in grouped})
    matrix = np.full((len(layers), len(steps)), np.nan, dtype=float)
    layer_pos = {value: index for index, value in enumerate(layers)}
    step_pos = {value: index for index, value in enumerate(steps)}
    for (step, layer), values in grouped.items():
        matrix[layer_pos[layer], step_pos[step]] = float(np.median(values))
    return matrix, steps, layers


def _heat_panels(
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    output: Path,
    name: str,
    cmap_name: str,
    step_key: str = "step_idx",
    transform: Callable[[float], float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_color: str = "#BDBDBD",
    dpi: int = 300,
) -> None:
    """Render one module-specific panel per figure rather than collapsing modules.

    The old implementation pooled all module families at the same
    ``(layer, step)`` coordinate.  That hid module differences and propagated
    invalid first-step NaNs into every cell.  This function keeps each module
    trajectory separate and masks missing cells in gray.
    """

    labels = sorted(
        {
            _module_label(row)
            for row in rows
            if _finite_value(row.get(key)) is not None
        }
    )
    if not labels:
        _empty_figure(output, name, title, "No finite data available", dpi)
        return
    columns = min(3, max(1, len(labels)))
    rows_count = int(math.ceil(len(labels) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        squeeze=False,
        figsize=(max(7.0, columns * 4.5), max(4.0, rows_count * 3.5)),
        constrained_layout=True,
    )
    cmap = _cmap(cmap_name, mask_color)
    image = None
    for index, label in enumerate(labels):
        ax = axes.flat[index]
        module_rows = [row for row in rows if _module_label(row) == label]
        result = _heat_matrix(module_rows, key, step_key, transform)
        if result is None:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        matrix, steps, layers = result
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("inference execution step")
        ax.set_ylabel("layer (0 at top)")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=90, fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
    for ax in axes.flat[len(labels) :]:
        ax.set_axis_off()
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    _save(fig, output, name, dpi)


def _ecdf(ax: Any, values: np.ndarray, label: str, color: str | None = None) -> None:
    values = values[np.isfinite(values)]
    if values.size:
        ax.plot(values, np.arange(1, values.size + 1) / values.size, label=label, color=color)


def _set_no_data(ax: Any, message: str = "No finite data available") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _global_limits(values: np.ndarray, quantiles: list[float], fallback: tuple[float, float]) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float(fallback[0]), float(fallback[1])]
    # YAML uses the conventional fractional form (e.g. 0.005 and 0.995),
    # while NumPy's percentile API expects percentages (0.5 and 99.5).
    # Passing the fractions directly selects only the bottom 1% and was the
    # reason many heatmaps collapsed to one color.
    quantiles_array = np.asarray(quantiles, dtype=float)
    if np.all((quantiles_array >= 0.0) & (quantiles_array <= 1.0)):
        quantiles_array = quantiles_array * 100.0
    low, high = np.percentile(values, quantiles_array)
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        return [float(fallback[0]), float(fallback[1])]
    if high <= low:
        padding = max(abs(float(low)) * 0.05, 1.0)
        low -= padding
        high += padding
    return [float(low), float(high)]


def _condition_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def _distance_figure(
    rows: list[dict[str, Any]],
    model: str,
    output: Path,
    dpi: int,
    mask_color: str,
) -> None:
    model_rows = []
    for row in rows:
        if row.get("model") != model:
            continue
        if _finite_value(row.get("raw_pair_distance")) is not None:
            model_rows.append(row)
    name = f"condition_distance_matrices_{model}"
    title = f"{model.upper()} condition distance"
    if not model_rows:
        _empty_figure(output, name, title, "No finite pair distances available", dpi)
        return
    conditions = sorted(
        {str(row.get("condition_i")) for row in model_rows}
        | {str(row.get("condition_j")) for row in model_rows},
        key=_condition_sort_key,
    )
    positions = {value: index for index, value in enumerate(conditions)}
    # A pair occurs at multiple layers and time steps.  Aggregate those
    # observations instead of letting the last CSV row overwrite the matrix.
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in model_rows:
        value = _finite_value(row.get("raw_pair_distance"))
        if value is None:
            continue
        left = str(row.get("condition_i"))
        right = str(row.get("condition_j"))
        pair_values[tuple(sorted((left, right), key=_condition_sort_key))].append(value)
    matrix = np.full((len(conditions), len(conditions)), np.nan, dtype=float)
    for (left, right), values in pair_values.items():
        i = positions[left]
        j = positions[right]
        matrix[i, j] = matrix[j, i] = float(np.median(values))
    np.fill_diagonal(matrix, 0.0)
    fig, ax = plt.subplots(figsize=(max(5.0, len(conditions) * 0.45), max(4.5, len(conditions) * 0.4)))
    cmap = _cmap("cividis", mask_color)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(conditions)), conditions, rotation=90)
    ax.set_yticks(range(len(conditions)), conditions)
    fig.colorbar(image, ax=ax)
    _save(fig, output, name, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CCMR scalar results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data = _read_tables(Path(args.input_dir).resolve())
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("png_dpi", 300))
    mask_color = str(config.get("mask_color", "#BDBDBD"))
    quantiles = [float(value) for value in config.get("global_robust_quantiles", [0.005, 0.995])]
    ccmr = data["ccmr_metrics"]
    rho = data["rho_per_condition"]
    stability = data["rho_stability"]

    v_raw = _finite(ccmr, "v_raw")
    v_diff = _finite(ccmr, "v_diff")
    gain = _finite(ccmr, "g_ccmr_db")
    rho_clean = _finite(rho, "rho_clean")
    log_v = np.log10(np.maximum(np.concatenate([v_raw, v_diff]), 1.0e-30)) if v_raw.size and v_diff.size else np.asarray([], dtype=float)
    log_rho = np.log10(np.maximum(rho_clean, 1.0e-30))
    resolved = dict(config)
    resolved["global_limits"] = {
        "log_v": _global_limits(log_v, quantiles, (-10.0, 1.0)),
        "gain_db": _global_limits(gain, quantiles, (0.0, 1.0)),
        "log_rho": _global_limits(log_rho, quantiles, (-3.0, 1.0)),
    }
    save_json_atomic(output / "plot_config_resolved.json", resolved)

    # Variance and gain heatmaps retain the historic filenames but now use
    # module-specific panels and the promised log10 scale for variances.
    for model in ("dit", "flux"):
        model_rows = [row for row in ccmr if row.get("model") == model]
        _heat_panels(
            model_rows,
            "v_raw",
            f"{model.upper()} log10 V(raw) by module",
            output,
            f"ccmr_variance_heatmaps_{model}_raw",
            str(config.get("sequential_cmap", "cividis")),
            transform=lambda value: math.log10(max(value, 1.0e-30)),
            vmin=resolved["global_limits"]["log_v"][0],
            vmax=resolved["global_limits"]["log_v"][1],
            mask_color=mask_color,
            dpi=dpi,
        )
        _heat_panels(
            model_rows,
            "v_diff",
            f"{model.upper()} log10 V(diff) by module",
            output,
            f"ccmr_variance_heatmaps_{model}_diff",
            str(config.get("sequential_cmap", "cividis")),
            transform=lambda value: math.log10(max(value, 1.0e-30)),
            vmin=resolved["global_limits"]["log_v"][0],
            vmax=resolved["global_limits"]["log_v"][1],
            mask_color=mask_color,
            dpi=dpi,
        )
        _heat_panels(
            model_rows,
            "g_ccmr_db",
            f"{model.upper()} CCMR gain (dB) by module",
            output,
            f"ccmr_variance_heatmaps_{model}_gain",
            str(config.get("gain_cmap", "BrBG")),
            vmin=resolved["global_limits"]["gain_db"][0],
            vmax=resolved["global_limits"]["gain_db"][1],
            mask_color=mask_color,
            dpi=dpi,
        )

    # Raw-vs-difference uses jointly filtered pairs, so x and y cannot become
    # misaligned when only one field contains an invalid value.
    pair_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_log_values = []
    for model in ("dit", "flux"):
        rows = [row for row in ccmr if row.get("model") == model]
        x, y = _paired_finite(rows, "v_base", "v_diff")
        x_log = np.log10(np.maximum(x, 1.0e-30))
        y_log = np.log10(np.maximum(y, 1.0e-30))
        pair_values[model] = x_log, y_log
        all_log_values.extend([x_log, y_log])
    if any(values.size for pair in pair_values.values() for values in pair):
        combined = np.concatenate([values for values in all_log_values if values.size])
        low, high = _global_limits(combined, quantiles, (-10.0, 1.0))
        padding = max((high - low) * 0.04, 0.25)
        low -= padding
        high += padding
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for model, ax in zip(("dit", "flux"), axes):
            x_log, y_log = pair_values[model]
            if x_log.size:
                ax.hexbin(
                    x_log,
                    y_log,
                    gridsize=35,
                    bins="log",
                    cmap=str(config.get("density_cmap", "cividis")),
                    mincnt=1,
                )
                ax.plot([low, high], [low, high], "k--", lw=0.8)
                ax.plot([low, high], [low - 1.0, high - 1.0], "k:", lw=0.8)
                ax.plot([low, high], [low - 2.0, high - 2.0], "k-.", lw=0.8)
                ax.set_xlim(low, high)
                ax.set_ylim(low, high)
            else:
                _set_no_data(ax)
            ax.set(title=model.upper(), xlabel="log10 V(base)", ylabel="log10 V(diff)")
            ax.grid(alpha=0.2)
        fig.suptitle("CCMR raw-vs-difference variance")
        _save(fig, output, "ccmr_raw_vs_diff_hexbin", dpi)
    else:
        _empty_figure(output, "ccmr_raw_vs_diff_hexbin", "CCMR raw-vs-difference variance", "No finite paired data available", dpi)

    # Aggregate gain ECDF and module distributions.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("dit", "flux"):
        values = _finite([row for row in ccmr if row.get("model") == model], "g_ccmr_db")
        _ecdf(ax, values, model.upper())
    if not any(_finite([row for row in ccmr if row.get("model") == model], "g_ccmr_db").size for model in ("dit", "flux")):
        _set_no_data(ax)
    for mark in (0, 10, 20):
        ax.axvline(mark, ls="--", lw=0.7, color="gray")
    ax.set(xlabel="CCMR gain (dB)", ylabel="ECDF", title="CCMR gain ECDF")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    _save(fig, output, "ccmr_gain_ecdf", dpi)

    groups: dict[str, list[float]] = defaultdict(list)
    for row in ccmr:
        value = _finite_value(row.get("g_ccmr_db"))
        if value is not None:
            groups[_module_label(row)].append(value)
    if groups:
        labels = sorted(groups)
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.5))
        parts = ax.violinplot([groups[label] for label in labels], showmedians=True, showextrema=True)
        for body, label in zip(parts["bodies"], labels):
            body.set_facecolor(_module_color(label))
            body.set_edgecolor("black")
            body.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("gain (dB)")
        ax.set_title("CCMR gain by module")
        ax.grid(alpha=0.2)
        _save(fig, output, "ccmr_gain_violin", dpi)
    else:
        _empty_figure(output, "ccmr_gain_violin", "CCMR gain by module", "No finite gain data available", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, ax in zip(("dit", "flux"), axes):
        rows = [row for row in ccmr if row.get("model") == model]
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            step = _parse_index(row.get("step_idx"))
            value = _finite_value(row.get("g_ccmr_db"))
            if step is not None and value is not None:
                grouped[step].append(value)
        if grouped:
            xs = sorted(grouped)
            ax.plot(xs, [float(np.median(grouped[x])) for x in xs], "o-")
        else:
            _set_no_data(ax)
        ax.set(title=model.upper(), xlabel="step", ylabel="median gain (dB)")
        ax.grid(alpha=0.2)
    fig.suptitle("CCMR gain over time")
    _save(fig, output, "ccmr_gain_over_time", dpi)

    # Condition similarity and alignment shuffle controls.
    similarity = data["condition_similarity"]
    grouped_similarity: dict[str, list[float]] = defaultdict(list)
    for row in similarity:
        value = _finite_value(row.get("adjacent_condition_cosine"))
        if value is not None:
            grouped_similarity[_module_label(row)].append(value)
    if grouped_similarity:
        labels = sorted(grouped_similarity)
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.5))
        ax.boxplot([grouped_similarity[label] for label in labels], tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("adjacent condition cosine")
        ax.set_title("Condition adjacent similarity")
        ax.grid(alpha=0.2)
        _save(fig, output, "condition_adjacent_similarity_heatmap", dpi)
    else:
        _empty_figure(output, "condition_adjacent_similarity_heatmap", "Condition adjacent similarity", "No finite similarity data available", dpi)

    aligned = _finite(similarity, "aligned_g_ccmr_db")
    shuffled = _finite(similarity, "shuffled_g_ccmr_db")
    if aligned.size and shuffled.size:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot([aligned, shuffled], tick_labels=["aligned", "shuffled"], showfliers=False)
        ax.set_ylabel("gain (dB)")
        ax.set_title("Condition alignment shuffle control")
        ax.grid(alpha=0.2)
        _save(fig, output, "condition_alignment_shuffle", dpi)
    else:
        _empty_figure(output, "condition_alignment_shuffle", "Condition alignment shuffle control", "No finite aligned/shuffled data available", dpi)

    for model in ("dit", "flux"):
        _distance_figure(data["condition_distance"], model, output, dpi, mask_color)

    # Rho distributions and heatmaps.  rho_stability uses score_step_idx,
    # unlike ccmr_metrics, so the correct index field is passed explicitly.
    rho_rows = [row for row in stability if _finite_value(row.get("rho_mean")) is not None]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("dit", "flux"):
        values = _finite([row for row in rho_rows if row.get("model") == model], "rho_mean")
        _ecdf(ax, np.log10(np.maximum(values, 1.0e-30)), model.upper())
    if not rho_rows:
        _set_no_data(ax)
    ax.axvline(0, ls="--", color="gray")
    ax.set(xlabel="log10 rho", ylabel="ECDF", title="rho distribution")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    _save(fig, output, "rho_distributions", dpi)
    for key, name in (("rho_mean", "rho_mean_heatmaps"), ("rho_median", "rho_median_heatmaps"), ("log_rho_mad", "rho_condition_dispersion_heatmaps")):
        for model in ("dit", "flux"):
            _heat_panels(
                [row for row in stability if row.get("model") == model],
                key,
                f"{model.upper()} {key} by module",
                output,
                f"{name}_{model}",
                str(config.get("rho_dispersion_cmap", "magma")) if "mad" in key else str(config.get("rho_cmap", "PuOr")),
                step_key="score_step_idx",
                transform=(lambda value: math.log10(max(value, 1.0e-30))) if key in {"rho_mean", "rho_median"} else None,
                mask_color=mask_color,
                dpi=dpi,
            )

    # Optional subset stability is still rendered when unavailable, with an
    # explicit explanation rather than an empty axes-only image.
    subset = data["subset_stability"]
    if subset:
        grouped_subset: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            size = _parse_index(row.get("subset_size"))
            value = _finite_value(row.get("jaccard"))
            if size is not None and value is not None:
                grouped_subset[size].append(value)
        if grouped_subset:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            xs = sorted(grouped_subset)
            ax.plot(xs, [float(np.mean(grouped_subset[x])) for x in xs], "o-")
            ax.set(xlabel="subset size", ylabel="Cache Book Jaccard", title="Few-sample subset stability")
            ax.grid(alpha=0.2)
            _save(fig, output, "rho_few_sample_vs_reference", dpi)
        else:
            _empty_figure(output, "rho_few_sample_vs_reference", "Few-sample subset stability", "No finite subset stability data available", dpi)
    else:
        _empty_figure(output, "rho_few_sample_vs_reference", "Few-sample subset stability", "Subset stability was not available for this aggregate", dpi)

    temporal = data["temporal_metrics"]
    temporal_x, temporal_y = _paired_finite(temporal, "output_rms", "diff_rms")
    if temporal_x.size:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hexbin(
            np.log10(np.maximum(temporal_x, 1.0e-30)),
            np.log10(np.maximum(temporal_y, 1.0e-30)),
            gridsize=35,
            bins="log",
            cmap=str(config.get("density_cmap", "cividis")),
            mincnt=1,
        )
        ax.set(xlabel="log10 A", ylabel="log10 D", title="Temporal smoothness")
        ax.grid(alpha=0.2)
        _save(fig, output, "temporal_smoothness_hexbin", dpi)
    else:
        _empty_figure(output, "temporal_smoothness_hexbin", "Temporal smoothness", "No finite temporal pairs available", dpi)
    temporal_rate = _finite(temporal, "r_time")
    if temporal_rate.size:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _ecdf(ax, np.log10(np.maximum(temporal_rate, 1.0e-30)), "all")
        ax.set(xlabel="log10 r_time", ylabel="ECDF", title="Temporal rate ECDF")
        ax.grid(alpha=0.2)
        _save(fig, output, "temporal_rate_ecdf", dpi)
    else:
        _empty_figure(output, "temporal_rate_ecdf", "Temporal rate ECDF", "No finite temporal rate data available", dpi)

    gaps = data["time_gap_metrics"]
    grouped_gaps: dict[int, list[float]] = defaultdict(list)
    for row in gaps:
        gap = _parse_index(row.get("time_gap"))
        value = _finite_value(row.get("g_ccmr_gap_db"))
        if gap is not None and value is not None:
            grouped_gaps[gap].append(value)
    if grouped_gaps:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = sorted(grouped_gaps)
        ax.plot(xs, [float(np.median(grouped_gaps[x])) for x in xs], "o-")
        ax.set(xlabel="time gap", ylabel="median gain (dB)", title="CCMR time-gap ablation")
        ax.grid(alpha=0.2)
        _save(fig, output, "ccmr_time_gap_ablation", dpi)
    else:
        _empty_figure(output, "ccmr_time_gap_ablation", "CCMR time-gap ablation", "No finite time-gap data available", dpi)

    figure_names = sorted(path.name for path in output.iterdir() if path.suffix in {".pdf", ".png"})
    save_json_atomic(output / "figure_manifest.json", {"output_dir": str(output), "figures": figure_names})
    print({"output": str(output), "figures": len(list(output.glob("*.png")))})


if __name__ == "__main__":
    main()
