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
from matplotlib import colors as mpl_colors
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
        "rho_scope",
        "pair_id",
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
    ax.set_facecolor("#F2F2F2")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#777777"},
    )
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
    colorbar_label: str | None = None,
    center: float | None = None,
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
        norm = None
        if center is not None and vmin is not None and vmax is not None and vmin < center < vmax:
            norm = mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        image = ax.imshow(masked, aspect="auto", origin="upper", cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
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
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        if colorbar_label:
            colorbar.set_label(colorbar_label)
    _save(fig, output, name, dpi)


def _ecdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x values and a monotone empirical CDF."""
    x = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if x.size == 0:
        return x, np.asarray([], dtype=float)
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _ecdf(ax: Any, values: np.ndarray, label: str, color: str | None = None) -> None:
    x, y = _ecdf_points(values)
    if x.size:
        ax.step(x, y, where="post", label=label, color=color)


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
    model_rows = [
        row for row in rows
        if row.get("model") == model
        and _finite_value(row.get("raw_pair_distance")) is not None
        and _finite_value(row.get("diff_pair_distance")) is not None
    ]
    name = f"condition_distance_matrices_{model}"
    title = f"{model.upper()} condition distance"
    if not model_rows:
        _empty_figure(output, name, title, "No finite condition-pair distances available", dpi)
        return
    unique_pairs = {
        tuple(sorted((str(row.get("condition_i")), str(row.get("condition_j"))), key=_condition_sort_key))
        for row in model_rows
    }
    if len(unique_pairs) < 2:
        _empty_figure(output, name, title, "At least two observed condition pairs are required", dpi)
        return
    conditions = sorted(
        {str(row.get("condition_i")) for row in model_rows}
        | {str(row.get("condition_j")) for row in model_rows},
        key=_condition_sort_key,
    )
    positions = {value: index for index, value in enumerate(conditions)}
    # A pair occurs at multiple layers and time steps.  Aggregate those
    # observations instead of letting the last CSV row overwrite the matrix.
    pair_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in model_rows:
        raw = _finite_value(row.get("raw_pair_distance"))
        diff = _finite_value(row.get("diff_pair_distance"))
        if raw is None or diff is None or raw <= 0.0:
            continue
        left = str(row.get("condition_i"))
        right = str(row.get("condition_j"))
        pair = tuple(sorted((left, right), key=_condition_sort_key))
        pair_values[pair]["raw"].append(raw)
        pair_values[pair]["diff"].append(diff)
    matrices = {name: np.full((len(conditions), len(conditions)), np.nan, dtype=float) for name in ("raw", "diff", "gain")}
    for (left, right), values in pair_values.items():
        i = positions[left]
        j = positions[right]
        raw_value = float(np.median(values["raw"]))
        diff_value = float(np.median(values["diff"]))
        matrices["raw"][i, j] = matrices["raw"][j, i] = raw_value
        matrices["diff"][i, j] = matrices["diff"][j, i] = diff_value
        matrices["gain"][i, j] = matrices["gain"][j, i] = 10.0 * math.log10(raw_value / max(diff_value, 1.0e-30))
    # The diagonal is not an observed condition pair.  Keep it masked rather
    # than displaying a zero that would dominate the logarithmic distance
    # scale and make the off-diagonal structure unreadable.
    np.fill_diagonal(matrices["raw"], np.nan)
    np.fill_diagonal(matrices["diff"], np.nan)
    np.fill_diagonal(matrices["gain"], np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(max(13.0, len(conditions) * 1.2), max(4.5, len(conditions) * 0.45)), constrained_layout=True)
    panel_specs = (
        ("raw", "log10 raw condition distance", "cividis", "log10 distance"),
        ("diff", "log10 difference condition distance", "cividis", "log10 distance"),
        ("gain", "diff/raw contraction (dB)", "BrBG", "dB"),
    )
    images = []
    for ax, (matrix_key, panel_title, cmap_name, label) in zip(axes, panel_specs):
        matrix = matrices[matrix_key]
        display = matrix
        if matrix_key in {"raw", "diff"}:
            display = np.log10(np.maximum(matrix, 1.0e-30))
        image = ax.imshow(np.ma.masked_invalid(display), cmap=_cmap(cmap_name, mask_color))
        images.append(image)
        ax.set_title(panel_title)
        ax.set_xticks(range(len(conditions)), conditions, rotation=90)
        ax.set_yticks(range(len(conditions)), conditions)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(label)
    fig.suptitle(title)
    _save(fig, output, name, dpi)


def _pair_rho_summary(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Reduce compact pair-difference rho rows to diagnostic heatmap cells.

    A compact FLUX row represents ``rho(Z_i-Z_j)``, not a prompt-level rho.
    It is still useful for a descriptive mean/median trajectory, provided the
    scope is kept explicit in the caller and it is never fed to condition
    dispersion or subset-stability statistics.
    """
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        scope = str(row.get("rho_scope") or "").lower()
        # Older smoke files predate the rho_scope column.  Their FLUX rho was
        # produced by the compact two-prompt path, so it is safe to treat an
        # absent scope as a legacy pair-difference diagnostic here.  The
        # aggregator still excludes it from condition-level stability.
        legacy_flux_pair = model == "flux" and scope == ""
        if row.get("model") != model or (scope != "pair_difference" and not legacy_flux_pair):
            continue
        if str(row.get("valid", "True")).lower() == "false":
            continue
        value = _finite_value(row.get("rho_clean"))
        if value is None:
            continue
        key = (
            row.get("source_run"), row.get("model"), row.get("seed"),
            row.get("module_family"), row.get("module_name"),
            row.get("layer_idx"), row.get("score_step_idx"),
        )
        grouped[key].append(value)
    result = []
    for key, values in grouped.items():
        result.append({
            "source_run": key[0], "model": key[1], "seed": key[2],
            "module_family": key[3], "module_name": key[4],
            "layer_idx": key[5], "score_step_idx": key[6],
            "rho_mean": float(np.mean(values)),
            "rho_median": float(np.median(values)),
            "rho_scope": "pair_difference", "valid": True,
        })
    return result


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
    log_v_values = [values for values in (v_raw, v_diff) if values.size]
    log_v = np.log10(np.maximum(np.concatenate(log_v_values), 1.0e-30)) if log_v_values else np.asarray([], dtype=float)
    log_rho = np.log10(np.maximum(rho_clean, 1.0e-30))
    resolved = dict(config)
    resolved["global_limits"] = {
        "log_v": _global_limits(log_v, quantiles, (-10.0, 1.0)),
        "gain_db": _global_limits(gain, quantiles, (0.0, 1.0)),
        "log_rho": _global_limits(log_rho, quantiles, (-3.0, 1.0)),
    }
    resolved["scale_notes"] = {
        "variance_heatmaps": "log10 variance",
        "gain_heatmaps": "diverging dB centered at gain_center_db",
        "condition_distance": "log10 raw/difference distance and dB contraction",
        "rho_mean_median_heatmaps": "log10 rho centered at log_rho_center",
        "rho_dispersion_heatmaps": "nonnegative log-rho MAD with vmin=0",
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
            colorbar_label="log10 variance",
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
            colorbar_label="log10 variance",
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
            colorbar_label="gain (dB)",
            center=float(config.get("gain_center_db", 0.0)),
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
                x_label = low + 0.04 * (high - low)
                ax.text(x_label, x_label, "0 dB: Vdiff = Vbase", fontsize=8, va="bottom")
                ax.text(x_label, x_label - 1.0, "10 dB: Vdiff = 0.1 Vbase", fontsize=8, va="bottom")
                ax.text(x_label, x_label - 2.0, "20 dB: Vdiff = 0.01 Vbase", fontsize=8, va="bottom")
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

    # Keep the alignment control separate for each model.  Mixing DiT and
    # FLUX into one pair of boxes hides model-specific scale differences.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    has_alignment = False
    for model, ax in zip(("dit", "flux"), axes):
        model_rows = [row for row in similarity if row.get("model") == model]
        # Keep the aligned/shuffled values paired by cell.  Independent
        # filtering would shift the arrays whenever one side contains an
        # invalid value and would make the displayed median delta meaningless.
        aligned, shuffled = _paired_finite(model_rows, "aligned_g_ccmr_db", "shuffled_g_ccmr_db")
        if aligned.size and shuffled.size:
            ax.boxplot([aligned, shuffled], tick_labels=["aligned", "shuffled"], showfliers=False)
            difference = aligned - shuffled
            ax.text(0.98, 0.04, f"median Δ={float(np.median(difference)):.2f} dB", transform=ax.transAxes, ha="right", fontsize=8)
            has_alignment = True
        else:
            _set_no_data(ax)
        ax.set_title(model.upper())
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("gain (dB)")
    fig.suptitle("Condition alignment shuffle control")
    if has_alignment:
        _save(fig, output, "condition_alignment_shuffle", dpi)
    else:
        plt.close(fig)
        _empty_figure(output, "condition_alignment_shuffle", "Condition alignment shuffle control", "No finite aligned/shuffled data available", dpi)

    for model in ("dit", "flux"):
        _distance_figure(data["condition_distance"], model, output, dpi, mask_color)

    # Rho distributions and heatmaps.  rho_stability uses score_step_idx,
    # unlike ccmr_metrics, so the correct index field is passed explicitly.
    rho_rows = [row for row in stability if _finite_value(row.get("rho_mean")) is not None]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("dit", "flux"):
        values = _finite([row for row in rho_rows if row.get("model") == model], "rho_mean")
        label = model.upper()
        if not values.size and model == "flux":
            pair_values = _finite(
                [
                    row for row in rho
                    if row.get("model") == model
                    and str(row.get("rho_scope") or "").lower() in {"", "pair_difference"}
                ],
                "rho_clean",
            )
            values = pair_values
            label = "FLUX pair-difference"
        _ecdf(ax, np.log10(np.maximum(values, 1.0e-30)), label)
    if not rho_rows:
        _set_no_data(ax)
    ax.axvline(0, ls="--", color="gray")
    ax.set(xlabel="log10 rho", ylabel="ECDF", title="rho distribution")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    _save(fig, output, "rho_distributions", dpi)
    for key, name in (("rho_mean", "rho_mean_heatmaps"), ("rho_median", "rho_median_heatmaps"), ("log_rho_mad", "rho_condition_dispersion_heatmaps")):
        for model in ("dit", "flux"):
            model_stability = [row for row in stability if row.get("model") == model]
            diagnostic_scope = False
            if not model_stability and model == "flux" and key in {"rho_mean", "rho_median"}:
                model_stability = _pair_rho_summary(rho, model)
                diagnostic_scope = bool(model_stability)
            if key == "log_rho_mad":
                mad_values = _finite(model_stability, key)
                if mad_values.size == 0 or float(np.nanmax(np.abs(mad_values))) <= 0.0:
                    _empty_figure(output, f"{name}_{model}", f"{model.upper()} rho dispersion", "Unavailable for mirrored pair diagnostics", dpi)
                    continue
                mad_vmax = max(float(np.nanmax(mad_values)), 1.0e-12)
            else:
                mad_vmax = None
            panel_title = {
                "rho_mean": f"{model.upper()} log10 mean rho by module",
                "rho_median": f"{model.upper()} log10 median rho by module",
                "log_rho_mad": f"{model.upper()} log10 rho MAD by module",
            }[key]
            if diagnostic_scope:
                panel_title += " (pair-difference diagnostic)"
            _heat_panels(
                model_stability,
                key,
                panel_title,
                output,
                f"{name}_{model}",
                str(config.get("rho_dispersion_cmap", "magma")) if "mad" in key else str(config.get("rho_cmap", "PuOr")),
                step_key="score_step_idx",
                transform=(lambda value: math.log10(max(value, 1.0e-30))) if key in {"rho_mean", "rho_median"} else None,
                vmin=0.0 if key == "log_rho_mad" else None,
                vmax=mad_vmax,
                colorbar_label="log10 rho" if key in {"rho_mean", "rho_median"} else "log10 rho MAD",
                center=float(config.get("log_rho_center", 0.0)) if key in {"rho_mean", "rho_median"} else None,
                mask_color=mask_color,
                dpi=dpi,
            )

    # Optional subset stability is rendered as a low-rho selection-overlap
    # diagnostic. It is not a complete two-stage Cache Book claim unless the
    # upstream table explicitly contains that experiment.
    subset = data["subset_stability"]
    if subset:
        grouped_subset: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in subset:
            size = _parse_index(row.get("subset_size"))
            fraction = _finite_value(row.get("cache_fraction"))
            if size is None or fraction is None:
                continue
            for metric in ("spearman", "kendall", "jaccard"):
                value = _finite_value(row.get(metric))
                if value is not None:
                    grouped_subset[(size, fraction)][metric].append(value)
        if grouped_subset:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            sizes = sorted({key[0] for key in grouped_subset})
            ax = axes[0]
            for metric, color in (("spearman", "#0072B2"), ("kendall", "#009E73")):
                medians = []; lows = []; highs = []; xs = []
                for size in sizes:
                    values = np.asarray([value for key, metrics in grouped_subset.items() if key[0] == size for value in metrics.get(metric, [])], dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        xs.append(size); medians.append(float(np.median(values)))
                        lows.append(float(np.percentile(values, 25))); highs.append(float(np.percentile(values, 75)))
                if xs:
                    ax.plot(xs, medians, "o-", color=color, label=metric)
                    ax.fill_between(xs, lows, highs, color=color, alpha=0.15)
            ax.set(xlabel="calibration sample count", ylabel="rank correlation", title="Rank stability")
            ax.set_ylim(-1.05, 1.05)
            ax.legend(loc="best")
            ax.grid(alpha=0.2)
            ax = axes[1]
            fractions = sorted({key[1] for key in grouped_subset})
            colors = ["#D55E00", "#E69F00", "#56B4E9", "#CC79A7"]
            for fraction, color in zip(fractions, colors):
                medians = []; lows = []; highs = []; xs = []
                for size in sizes:
                    values = np.asarray(grouped_subset.get((size, fraction), {}).get("jaccard", []), dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        xs.append(size); medians.append(float(np.median(values)))
                        lows.append(float(np.percentile(values, 25))); highs.append(float(np.percentile(values, 75)))
                if xs:
                    ax.plot(xs, medians, "o-", color=color, label=f"Jaccard@{fraction:g}")
                    ax.fill_between(xs, lows, highs, color=color, alpha=0.15)
            ax.set(xlabel="calibration sample count", ylabel="low-rho overlap", title="Low-rho selection overlap")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.2)
            fig.suptitle("Low-rho selection overlap")
            _save(fig, output, "rho_few_sample_vs_reference", dpi)
        else:
            _empty_figure(output, "rho_few_sample_vs_reference", "Low-rho selection overlap", "No finite subset stability data available", dpi)
    else:
        _empty_figure(output, "rho_few_sample_vs_reference", "Low-rho selection overlap", "Subset stability was not available for this aggregate", dpi)

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
    grouped_gaps: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in gaps:
        model = str(row.get("model", "unknown"))
        gap = _parse_index(row.get("time_gap"))
        value = _finite_value(row.get("g_ccmr_gap_db"))
        if gap is not None and value is not None:
            grouped_gaps[(model, _module_label(row), gap)].append(value)
    if grouped_gaps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
        has_gap = False
        for model, ax in zip(("dit", "flux"), axes):
            model_groups = {(module, gap): values for (row_model, module, gap), values in grouped_gaps.items() if row_model == model}
            modules = sorted({module for module, _ in model_groups})
            for module in modules:
                points = [(gap, values) for (name, gap), values in model_groups.items() if name == module]
                points.sort()
                if not points:
                    continue
                xs = [gap for gap, _ in points]
                medians = [float(np.median(values)) for _, values in points]
                lows = [float(np.percentile(values, 25)) for _, values in points]
                highs = [float(np.percentile(values, 75)) for _, values in points]
                color = _module_color(module)
                ax.plot(xs, medians, "o-", color=color, label=module)
                ax.fill_between(xs, lows, highs, color=color, alpha=0.15)
                has_gap = True
            if not modules:
                _set_no_data(ax)
            ax.set_title(model.upper())
            ax.set_xlabel("time gap")
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)
        axes[0].set_ylabel("gain (dB), median with IQR")
        fig.suptitle("CCMR time-gap ablation by model and module")
        if has_gap:
            _save(fig, output, "ccmr_time_gap_ablation", dpi)
        else:
            plt.close(fig)
            _empty_figure(output, "ccmr_time_gap_ablation", "CCMR time-gap ablation", "No finite time-gap data available", dpi)
    else:
        _empty_figure(output, "ccmr_time_gap_ablation", "CCMR time-gap ablation", "No finite time-gap data available", dpi)

    figure_names = sorted(path.name for path in output.iterdir() if path.suffix in {".pdf", ".png"})
    save_json_atomic(output / "figure_manifest.json", {"output_dir": str(output), "figures": figure_names})
    print({"output": str(output), "figures": len(list(output.glob("*.png")))})


if __name__ == "__main__":
    main()
