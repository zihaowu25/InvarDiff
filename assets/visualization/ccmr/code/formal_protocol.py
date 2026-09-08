"""Auditable formal-protocol checks shared by validation and paper exports."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import jaccard_at_fraction, json_hash, rank_corr, read_rows, select_pairs, sha256_file

DIT_MODULES = {"dit.msa": 28, "dit.mlp": 28}
FLUX_MODULES = {
    "double.attn": 19,
    "double.context_attn": 19,
    "double.ff": 19,
    "double.context_ff": 19,
    "single.attn": 38,
    "single.mlp": 38,
}


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rho_consistency(rows: Iterable[dict[str, Any]], fraction: float = 0.3) -> dict[str, Any]:
    valid = [row for row in rows if is_true(row.get("valid", True))]
    clean = np.asarray([float(row["rho_clean"]) for row in valid], dtype=np.float64)
    code = np.asarray([float(row["rho_code"]) for row in valid], dtype=np.float64)
    absolute = np.abs(clean - code)
    relative = absolute / np.maximum(np.abs(clean), 1.0e-12)
    spearman, kendall = rank_corr(clean, code)
    return {
        "valid_cells": int(clean.size),
        "max_absolute_error": float(absolute.max()) if absolute.size else float("nan"),
        "max_relative_error": float(relative.max()) if relative.size else float("nan"),
        "spearman": spearman,
        "kendall": kendall,
        "jaccard_at_30pct": jaccard_at_fraction(clean, code, fraction),
    }


def tolerance_check(rows: Iterable[dict[str, Any]], rtol: float, atol: float) -> tuple[bool, dict[str, Any]]:
    rows = [row for row in rows if is_true(row.get("valid", True))]
    audit = rho_consistency(rows)
    failures = []
    for index, row in enumerate(rows):
        clean = float(row["rho_clean"])
        code = float(row["rho_code"])
        if not math.isclose(clean, code, rel_tol=float(rtol), abs_tol=float(atol)):
            failures.append(index)
    audit["rtol"] = float(rtol)
    audit["atol"] = float(atol)
    audit["failure_count"] = len(failures)
    audit["passed"] = not failures
    return not failures, audit


def table_rows(run: Path, table: str) -> list[dict[str, str]]:
    for suffix in ("csv.gz", "csv"):
        path = run / "tables" / f"{table}.{suffix}"
        if path.exists():
            return read_rows(path)
    return []


def validate_atomic_manifest(run: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = run / "config.json"
    if not config_path.is_file():
        failures.append("run is missing config.json")
        actual_config_hash = None
    else:
        actual_config_hash = json_hash(json.loads(config_path.read_text(encoding="utf-8")))
        if manifest.get("resolved_config_hash") != actual_config_hash:
            failures.append("run manifest does not match the actual config hash")
    if not manifest.get("shards"):
        failures.append("run manifest contains no shards")
    shard_commits: set[str] = set()
    for key, shard in manifest.get("shards", {}).items():
        if shard.get("status") != "complete":
            failures.append(f"shard {key}: status is not complete")
        if shard.get("resolved_config_hash") != manifest.get("resolved_config_hash"):
            failures.append(f"shard {key}: resolved config hash mismatch")
        if actual_config_hash is not None and shard.get("resolved_config_hash") != actual_config_hash:
            failures.append(f"shard {key}: does not match actual config hash")
        commit = str(shard.get("commit", "")).strip()
        if not commit:
            failures.append(f"shard {key}: missing collection commit")
        else:
            shard_commits.add(commit)
        artifacts = shard.get("tables") or ({"rho_per_condition": shard.get("table")} if shard.get("table") else {})
        for name, artifact in artifacts.items():
            path = Path(str(artifact.get("path", "")))
            if not path.is_absolute():
                path = run / "shards" / path
            if not path.is_file():
                failures.append(f"shard {key}/{name}: missing table")
                continue
            if sha256_file(path) != artifact.get("sha256"):
                failures.append(f"shard {key}/{name}: checksum mismatch")
            if len(read_rows(path)) != int(artifact.get("row_count", -1)):
                failures.append(f"shard {key}/{name}: row-count mismatch")
    if len(shard_commits) > 1:
        failures.append(f"run mixes shard commits: {sorted(shard_commits)}")
    return failures


def validate_boundary(rows: list[dict[str, Any]], steps: int) -> list[str]:
    if not rows:
        return ["rho boundary validation has no rows"]
    failures = []
    by_track: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("seed"), row.get("condition_id"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"))
        by_track[key].append(row)
    for key, track in by_track.items():
        indices = Counter(int(float(row["score_step_idx"])) for row in track)
        if set(indices) != set(range(steps)) or any(count != 1 for count in indices.values()):
            failures.append(f"rho track {key}: incomplete or duplicate score indices")
            continue
        for row in track:
            index = int(float(row["score_step_idx"]))
            expected = 1 <= index <= steps - 2
            if is_true(row.get("valid", True)) != expected:
                failures.append(f"rho track {key}: invalid boundary mask at {index}")
    return failures


def validate_modules(rows: list[dict[str, Any]], expected: dict[str, int]) -> list[str]:
    observed: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        observed[f"{row.get('module_family')}.{row.get('module_name')}"].add(int(float(row["layer_idx"])))
    failures = []
    for module, count in expected.items():
        if observed.get(module) != set(range(count)):
            failures.append(f"module coverage {module}: expected layers 0..{count - 1}")
    extra = set(observed) - set(expected)
    if extra:
        failures.append(f"unexpected modules: {sorted(extra)}")
    return failures


def validate_derangements(rows: list[dict[str, Any]], trials: int = 100) -> list[str]:
    if not rows:
        return ["derangement validation has no rows"]
    failures = []
    grouped: dict[tuple[Any, ...], dict[int, tuple[int, ...]]] = defaultdict(dict)
    for row in rows:
        if str(row.get("model")) != "dit" or not is_true(row.get("valid", True)):
            continue
        key = (row.get("seed"), row.get("module_family"), row.get("module_name"), row.get("layer_idx"), row.get("step_idx"))
        permutation = tuple(json.loads(str(row["permutation"])))
        grouped[key][int(float(row["shuffle_trial"]))] = permutation
        if any(index == value for index, value in enumerate(permutation)):
            failures.append(f"fixed point in {key}, trial {row.get('shuffle_trial')}")
    for key, permutations in grouped.items():
        if len(permutations) != trials or len(set(permutations.values())) != trials:
            failures.append(f"derangement coverage {key}: {len(permutations)}/{trials}")
    return failures


def validate_flux_swaps(rows: list[dict[str, Any]]) -> list[str]:
    """Require exactly one [1,0] alignment control for every valid FLUX cell."""
    if not rows:
        return ["FLUX swap validation has no rows"]
    failures: list[str] = []
    grouped: Counter[tuple[Any, ...]] = Counter()
    for index, row in enumerate(rows):
        if str(row.get("model")) != "flux" or not is_true(row.get("valid", True)):
            continue
        key = (
            row.get("seed"), row.get("pair_id"), row.get("module_family"),
            row.get("module_name"), row.get("layer_idx"), row.get("step_idx"),
        )
        grouped[key] += 1
        try:
            permutation = json.loads(str(row.get("permutation")))
        except (TypeError, json.JSONDecodeError):
            permutation = None
        if permutation != [1, 0]:
            failures.append(f"FLUX alignment row {index}: permutation is not [1,0]")
            if len(failures) >= 20:
                return failures
    if not grouped:
        failures.append("FLUX swap validation has no valid cells")
    for key, count in grouped.items():
        if count != 1:
            failures.append(f"FLUX cell {key}: alignment row count {count}, expected 1")
            if len(failures) >= 20:
                break
    return failures


def subset_hierarchical_summary(
    rows: Iterable[dict[str, Any]],
    metric: str,
    trials: int = 2000,
    random_seed: int = 2027,
) -> dict[str, Any] | None:
    """Macro-average families per seed/subset, then bootstrap seed -> subset.

    A selected condition subset is a correlated cluster repeated across module
    families.  Family rows are reduced inside that cluster before any
    resampling, so adding or duplicating family rows cannot masquerade as
    additional IID subset observations.
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        value = row.get(metric)
        if value in (None, "") or not finite(value):
            continue
        seed_key = (str(row.get("source_run", "")), str(row.get("seed")))
        subset_key = str(row.get("selected_condition_ids") or f"trial:{row.get('trial')}")
        family = str(row.get("module_family", "ALL"))
        grouped[seed_key][subset_key][family].append(float(value))
    if not grouped:
        return None

    by_seed: dict[tuple[str, str], list[float]] = {}
    unique_subsets: set[tuple[str, str, str]] = set()
    for seed_key, subsets in grouped.items():
        cluster_values = []
        for subset_key, families in subsets.items():
            family_means = [float(np.mean(values)) for values in families.values() if values]
            if family_means:
                cluster_values.append(float(np.mean(family_means)))
                unique_subsets.add((*seed_key, subset_key))
        if cluster_values:
            by_seed[seed_key] = cluster_values
    if not by_seed:
        return None

    seed_ids = sorted(by_seed)
    seed_points = [float(np.mean(by_seed[item])) for item in seed_ids]
    rng = np.random.default_rng(random_seed)
    boot = []
    for _ in range(int(trials)):
        sampled_seeds = rng.choice(len(seed_ids), size=len(seed_ids), replace=True)
        values = []
        for seed_index in sampled_seeds:
            clusters = np.asarray(by_seed[seed_ids[int(seed_index)]], dtype=np.float64)
            values.append(float(np.mean(rng.choice(clusters, size=len(clusters), replace=True))))
        boot.append(float(np.mean(values)))
    estimate = float(np.mean(seed_points))
    return {
        "estimate": estimate,
        "mean": estimate,
        "p025": float(np.percentile(boot, 2.5)),
        "p975": float(np.percentile(boot, 97.5)),
        "seed_points": seed_points,
        "num_seeds": len(seed_ids),
        "num_unique_subsets": len(unique_subsets),
    }


def validate_valid_rho_finite(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["rho validation has no rows"]
    failures = []
    valid_count = 0
    for index, row in enumerate(rows):
        if not is_true(row.get("valid", True)):
            continue
        valid_count += 1
        for field in ("rho_clean", "rho_code"):
            if not finite(row.get(field)):
                failures.append(f"valid rho row {index} has non-finite {field}")
                if len(failures) >= 20:
                    return failures
    if valid_count == 0:
        failures.append("rho validation has no valid rows")
    return failures


def validate_test_log(path: Path) -> list[str]:
    if not path.is_file():
        return [f"missing test log: {path}"]
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if any(token in lowered for token in (" failed", " error", "no tests ran")):
        return ["test log reports failed/error/no tests ran"]
    match = re.search(r"(?m)^\s*(\d+) passed(?:\s|$)", text)
    if not match or int(match.group(1)) <= 0:
        return ["test log does not contain an explicit positive 'N passed' result"]
    return []


def validate_flux_pair_selection(config: dict[str, Any], conditions: list[dict[str, Any]], selection: dict[str, Any]) -> list[str]:
    failures = []
    ids = [str(item["id"]) for item in conditions]
    requested_seed = int(config.get("pair_selection_seed", -1))
    requested_count = int(config.get("num_condition_pairs", -1))
    expected_indices = select_pairs(len(ids), requested_count, requested_seed)
    expected = [[ids[i], ids[j]] for i, j in expected_indices]
    actual = [[str(pair[0]), str(pair[1])] for pair in selection.get("pairs", [])]
    if int(selection.get("seed", -2)) != requested_seed:
        failures.append("pair-selection seed mismatch")
    if actual != expected:
        failures.append("pair-selection list does not match deterministic preregistration")
    canonical = [tuple(sorted(pair)) for pair in actual]
    if len(canonical) != len(set(canonical)):
        failures.append("pair-selection contains duplicate unordered pairs")
    if any(a == b or a not in ids or b not in ids for a, b in actual):
        failures.append("pair-selection contains illegal pairs")
    expected_hash = json_hash(expected)
    if selection.get("pair_selection_hash") != expected_hash:
        failures.append("pair-selection hash mismatch")
    return failures
