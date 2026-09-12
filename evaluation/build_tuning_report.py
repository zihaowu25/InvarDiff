#!/usr/bin/env python3
"""Build compact CSV/JSON summaries and plots from low-cost cache sweeps.

The builder deliberately tolerates incomplete optional runs: a missing run is
reported as ``status=missing`` instead of fabricating a quality number.  Raw
videos and Cache Books remain under the ignored ``runs/cache_tuning`` tree.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _read(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _row(model, policy, tier, thresholds, lpips=None, p95=None, hit=None,
         latency=None, peak=None, status="measured", source=None):
    return {
        "model": model,
        "policy": policy,
        "tier": tier,
        "thresholds": thresholds,
        "lpips": lpips,
        "p95": p95,
        "hit_rate": hit,
        "latency_s": latency,
        "peak_gib": peak,
        "status": status,
        "source": source,
    }


def _summary_row(model, policy, tier, path, thresholds=None):
    payload = _read(path)
    if payload is None:
        return _row(model, policy, tier, thresholds or {}, status="missing", source=str(path))
    means = payload.get("lpips_prompt_means") or payload.get("values")
    p95s = payload.get("lpips_prompt_p95")
    hit = payload.get("module_hit_rates") or payload.get("cache_hit_rates")
    if isinstance(hit, dict):
        numeric = [float(v) for v in hit.values() if isinstance(v, (float, int))]
        hit = sum(numeric) / len(numeric) if numeric else None
    # paired_lpips.py stores image/video statistics under frame_lpips, while
    # candidate runners use the flat lpips_macro_mean/values schema.
    frame_stats = payload.get("frame_lpips") or {}
    lpips_value = payload.get("lpips_macro_mean", payload.get("mean", payload.get("mean_lpips")))
    if lpips_value is None:
        lpips_value = frame_stats.get("mean")
    p95_value = max(p95s) if p95s else payload.get("max", payload.get("p95_lpips", payload.get("lpips_p95_max")))
    if p95_value is None:
        p95_value = frame_stats.get("p95")
    if p95_value is None:
        p95_value = frame_stats.get("max")
    return _row(
        model, policy, tier,
        thresholds or payload.get("thresholds", {}),
        lpips=lpips_value,
        p95=p95_value,
        hit=hit,
        latency=payload.get("generation_wall_s_mean"),
        peak=payload.get("generation_peak_allocated_gib_max"),
        source=str(path),
    )


def _paired_hybrid_row(model, method, root, thresholds, *, expected_pairs=3):
    """Summarize paired step-only/hybrid video LPIPS observations.

    ``paired_lpips.py`` writes one JSON per prompt.  The hybrid budget is the
    mean of those paired frame means; the bootstrap upper bound is computed at
    the prompt level so the report does not over-count correlated video
    frames.
    """
    files = sorted(root.glob("p*.json")) if root.is_dir() else []
    values = []
    p95_values = []
    for path in files:
        payload = _read(path) or {}
        frame = payload.get("frame_lpips") or {}
        if isinstance(frame.get("mean"), (int, float)):
            values.append(float(frame["mean"]))
            if isinstance(frame.get("p95"), (int, float)):
                p95_values.append(float(frame["p95"]))
    if len(values) != expected_pairs:
        return _row(model, method + " hybrid", "hybrid", thresholds,
                    status="missing", source=str(root))
    observations = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2027)
    samples = rng.integers(0, observations.size,
                           size=(100_000, observations.size))
    upper = float(np.percentile(observations[samples].mean(axis=1), 95))
    row = _row(model, method + " hybrid", "hybrid", thresholds,
               lpips=float(observations.mean()),
               p95=(max(p95_values) if p95_values else None),
               status="measured", source=str(root))
    row["bootstrap_95_upper"] = upper
    row["paired_values"] = values
    return row


def collect():
    rows = []
    # Three-sample DiT module-only screen (measured in the prior low-cost run).
    dit = {
        "fast": ({"msa_thres": .60, "mlp_thres": .65}, .4048563),
        "balanced": ({"msa_thres": .60, "mlp_thres": .62}, .2136620),
        "slow": ({"msa_thres": .62, "mlp_thres": .59}, .1073058),
    }
    for tier, (thresholds, value) in dit.items():
        rows.append(_row("DiT", "module-only", tier, thresholds, lpips=value,
                         status="measured", source="runs/cache_tuning/dit_lowcost/module_target"))

    # FLUX images are stored as a 3-panel horizontal grid; lpips_grid.json
    # splits panels before averaging, which is the requested 3-sample mean.
    flux_thresholds = {
        "fast": ("flux_fast2", {"attn_thres": .70, "context_attn_thres": .01, "ff_thres": .20,
                 "context_ff_thres": .05, "single_attn_thres": .40, "single_mlp_thres": .02}),
        "balanced": ("flux_balanced2", {"attn_thres": .30, "context_attn_thres": .00, "ff_thres": .04,
                     "context_ff_thres": .03, "single_attn_thres": .10, "single_mlp_thres": .02}),
        "slow": ("flux_slow_mid", {"attn_thres": .08, "context_attn_thres": .00, "ff_thres": .01,
                 "context_ff_thres": .02, "single_attn_thres": .03, "single_mlp_thres": .00}),
    }
    for tier, (run_id, thresholds) in flux_thresholds.items():
        rows.append(_summary_row("FLUX.1-dev", "module-only", tier,
                                 REPO / "runs/cache_tuning/flux_lowcost/targets" / run_id / "lpips_grid.json",
                                 thresholds))

    # Wan target-band probe and the conservative first probe are both kept;
    # the former is used for preset selection when it reaches a target band.
    # Prefer the narrow intermediate slow probe when it has completed; the
    # fallback keeps the report reproducible if a caller only has the earlier
    # band-3 run.
    wan_slow_id = "wan_slow_mid2" if (REPO / "runs/cache_tuning/wan_lowcost/targets/wan_slow_mid2/lpips_summary.json").is_file() else "wan_slow_band3"
    wan_thresholds = {
        "fast": ("wan_fast_band2", {"self_attn_thres": .25, "cross_attn_thres": .30, "ffn_thres": .35}),
        "balanced": ("wan_balanced_band2", {"self_attn_thres": .10, "cross_attn_thres": .15, "ffn_thres": .20}),
        "slow": (wan_slow_id, {"self_attn_thres": .04, "cross_attn_thres": .05, "ffn_thres": .07} if wan_slow_id == "wan_slow_mid2" else {"self_attn_thres": .05, "cross_attn_thres": .07, "ffn_thres": .09}),
    }
    for tier, (run_id, thresholds) in wan_thresholds.items():
        rows.append(_summary_row("Wan2.1-1.3B", "module-only", tier,
                                 REPO / "runs/cache_tuning/wan_lowcost/targets" / run_id / "lpips_summary.json",
                                 thresholds))

    h_thresholds = {
        "fast": ("h_fast", {"double.img_attn": .40, "double.txt_attn": .01, "double.img_mlp": .20, "double.txt_mlp": .32}),
        "balanced": ("h_balanced", {"double.img_attn": .40, "double.txt_attn": .01, "double.img_mlp": .04, "double.txt_mlp": .32}),
        "slow": ("h_slow_alt", {"double.img_attn": .40, "double.txt_attn": .01, "double.img_mlp": .02, "double.txt_mlp": .00}),
    }
    for tier, (run_id, thresholds) in h_thresholds.items():
        rows.append(_summary_row("HunyuanVideo-1.5", "module-only", tier,
                                 REPO / "runs/cache_tuning/hunyuan_lowcost_module/targets" / run_id / "summary.json",
                                 thresholds))
    h_step_thresholds = {
        "fast": {"step_thres": .20, "double_img_attn_thres": .40, "double_txt_attn_thres": .01,
                 "double_img_mlp_thres": .20, "double_txt_mlp_thres": .32},
        "balanced": {"step_thres": .03, "double_img_attn_thres": .40, "double_txt_attn_thres": .01,
                     "double_img_mlp_thres": .04, "double_txt_mlp_thres": .12},
        "slow": {"step_thres": .02, "double_img_attn_thres": .40, "double_txt_attn_thres": .01,
                 "double_img_mlp_thres": .00, "double_txt_mlp_thres": .00},
    }
    for tier in ("fast", "balanced", "slow"):
        rows.append(_summary_row("HunyuanVideo-1.5", "step+layer", tier,
                                 REPO / "runs/cache_tuning/hunyuan_lowcost_step/targets" / f"h_{tier}" / "summary.json",
                                 h_step_thresholds[tier]))

    hybrid = {
        "MagCache": ("magcache", "q003"),
        "TeaCache": ("teacache", "q020"),
        "SeaCache": ("seacache", "q020"),
    }
    for method, (folder, q) in hybrid.items():
        path = REPO / "runs/cache_tuning/hunyuan_lowcost_hybrid" / folder / f"hybrid_{q}" / "lpips_summary.json"
        payload = _read(path)
        rows.append(_row("HunyuanVideo-1.5", method + " hybrid", "hybrid",
                         {"double.img_attn": .90, "double.txt_attn": .20,
                          "double.img_mlp": .00, "double.txt_mlp": .00},
                         lpips=payload.get("mean") if payload else None,
                         p95=payload.get("max") if payload else None,
                         status=("measured" if payload else "missing"), source=str(path)))

    # FLUX external-hybrid screen: the baseline is the same external step
    # policy with layer cache disabled; only the module vector is changed.
    flux_hybrid = {
        "MagCache": ("magcache", "hybrid", {"attn_thres": .20, "context_attn_thres": .10,
                    "ff_thres": .05, "context_ff_thres": .03, "single_attn_thres": .08, "single_mlp_thres": .01}),
        "TeaCache": ("teacache", "hybrid_low", {"attn_thres": .20, "context_attn_thres": .10,
                    "ff_thres": .05, "context_ff_thres": .03, "single_attn_thres": .08, "single_mlp_thres": .01}),
        "SeaCache": ("seacache", "hybrid", {"attn_thres": .20, "context_attn_thres": .10,
                    "ff_thres": .05, "context_ff_thres": .03, "single_attn_thres": .08, "single_mlp_thres": .01}),
    }
    for method, (folder, run, thresholds) in flux_hybrid.items():
        path = REPO / "runs/cache_tuning/flux_hybrid_lowcost" / folder / run / "lpips_summary.json"
        payload = _read(path)
        rows.append(_row("FLUX.1-dev", "FLUX " + method + " hybrid", "hybrid", thresholds,
                         lpips=payload.get("frame_lpips", {}).get("mean") if payload else None,
                         p95=payload.get("frame_lpips", {}).get("p95") if payload else None,
                         status=("measured" if payload else "missing"), source=str(path)))

    # Wan external-hybrid screen.  Each method keeps its upstream default
    # cache parameters; only the module-level vector from the ``hybrid``
    # preset is applied.  The source contains three paired prompt videos.
    wan_hybrid_thresholds = {
        "MagCache": {"self_attn_thres": .10, "cross_attn_thres": .11, "ffn_thres": .12},
        "TeaCache": {"self_attn_thres": .10, "cross_attn_thres": .11, "ffn_thres": .12},
        "SeaCache": {"self_attn_thres": .10, "cross_attn_thres": .11, "ffn_thres": .12},
    }
    for method, thresholds in wan_hybrid_thresholds.items():
        row = _paired_hybrid_row(
            "Wan2.1-1.3B", method,
            REPO / "runs/cache_tuning/wan_hybrid_lowcost" / method.lower() / "paired",
            thresholds,
        )
        row["policy"] = "Wan " + row["policy"]
        rows.append(row)
    return rows


def write_outputs(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_version": 1, "rows": rows}
    (output_dir / "threshold_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "policy", "tier", "lpips", "p95", "hit_rate", "latency_s", "peak_gib", "status", "thresholds", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["thresholds"] = json.dumps(item["thresholds"], sort_keys=True)
            writer.writerow({field: item.get(field) for field in fields})

    measured = [r for r in rows if r["status"] == "measured" and r["lpips"] is not None]
    plt.figure(figsize=(11, 6))
    for model in sorted({r["model"] for r in measured}):
        group = [r for r in measured if r["model"] == model and r["policy"] == "module-only"]
        if group:
            xs = list(range(len(group)))
            plt.plot(xs, [r["lpips"] for r in group], marker="o", label=model)
            plt.xticks(xs, [r["tier"] for r in group])
    for y, label in ((.10, "slow target"), (.20, "balanced target"), (.40, "fast target")):
        plt.axhline(y, linestyle="--", linewidth=.8, label=label)
    plt.ylabel("LPIPS vs uncached reference (3-sample mean)")
    plt.title("Low-cost module-only tier screening")
    plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(output_dir / "threshold_curves.png", dpi=160); plt.close()

    # Per-module diagnostic curves collected from the causal sweeps.
    curves = {
        "Hunyuan double.txt_attn": ([0, .01, .02, .03, .04], [0, .07216, .08183, .08294, .08457]),
        "Hunyuan double.img_mlp": ([0, .01, .02, .03, .04], [.1434, .1428, .1428, .1856, .2097]),
        "DiT MSA": ([.50, .60, .61, .62, .70], [.0, .0, .02, .10, .4]),
        "FLUX context_ff": ([0, .04, .05, .06, .10], [0, .00278, .00495, .01232, .04918]),
        "Wan self_attn": ([.01, .10], [.01881, .09370]),
    }
    with (output_dir / "module_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["module", "threshold", "lpips"])
        for module, (xs, ys) in curves.items():
            writer.writerows((module, x, y) for x, y in zip(xs, ys))
    plt.figure(figsize=(11, 6))
    for module, (xs, ys) in curves.items():
        plt.plot(xs, ys, marker="o", label=module)
    plt.xlabel("module threshold"); plt.ylabel("LPIPS (paired screen)")
    plt.title("Independent module threshold curves and observed elbows")
    plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(output_dir / "module_lpips_curves.png", dpi=160); plt.close()

    hybrids = [r for r in rows if "hybrid" in r["policy"] and r["lpips"] is not None]
    # Paired-bootstrap upper bounds are computed by aggregate_paired_delta.py;
    # retain the values in the compact report even when the raw videos remain
    # ignored under runs/cache_tuning/.
    bootstrap_upper = {"MagCache hybrid": .145656, "TeaCache hybrid": .162144,
                       "SeaCache hybrid": .330149,
                       "FLUX MagCache hybrid": .180054,
                       "FLUX TeaCache hybrid": .159244,
                       "FLUX SeaCache hybrid": .164450}
    with (output_dir / "hybrid_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "delta_lpips", "bootstrap_95_upper", "budget", "passes_budget"])
        for row in hybrids:
            upper = row.get("bootstrap_95_upper", bootstrap_upper.get(row["policy"]))
            writer.writerow((row["policy"], row["lpips"], upper, .2,
                             bool(upper is not None and upper <= .2)))
    if hybrids:
        plt.figure(figsize=(8, 5))
        labels = [r["policy"] for r in hybrids]; values = [r["lpips"] for r in hybrids]
        plt.bar(labels, values); plt.axhline(.2, color="crimson", linestyle="--", label="budget")
        plt.ylabel("LPIPS(step-only, hybrid)"); plt.title("Hybrid layer quality budget")
        plt.xticks(rotation=15); plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / "hybrid_lpips_delta.png", dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "agent_skills" / "report")
    args = parser.parse_args()
    rows = collect(); write_outputs(rows, args.output_dir)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
