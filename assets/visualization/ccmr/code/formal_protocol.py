"""Auditable formal-protocol checks shared by validation and paper exports."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import jaccard_at_fraction, rank_corr, read_rows, sha256_file

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
    for key, shard in manifest.get("shards", {}).items():
        if shard.get("status") != "complete":
            failures.append(f"shard {key}: status is not complete")
        if shard.get("resolved_config_hash") != manifest.get("resolved_config_hash"):
            failures.append(f"shard {key}: resolved config hash mismatch")
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
    return failures


def validate_boundary(rows: list[dict[str, Any]], steps: int) -> list[str]:
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
