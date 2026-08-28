"""Shared CCMR statistics, persistence and reproducibility helpers.

The collector scripts deliberately keep model-specific code local to their
files.  This module contains only model-agnostic numerical operations and the
on-disk schema used by the aggregator and plotting code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

EPS = 1.0e-12
DEGENERATE_THRESHOLD = 1.0e-8


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def save_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    """Write JSON using a same-directory temporary file and atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    # NumPy has no bfloat16 dtype.  Hash the contiguous raw bytes instead of
    # converting through NumPy's scalar conversion, preserving dtype/shape
    # metadata above and making hashes stable for BF16 FLUX latents.
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed))
    return generator


def l1_distance(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 1_048_576) -> torch.Tensor:
    """FP32 chunked L1 distance; inputs retain their native dtype/device."""
    if a.numel() != b.numel():
        raise ValueError(f"L1 inputs have different sizes: {a.shape} vs {b.shape}")
    af = a.detach().reshape(-1)
    bf = b.detach().reshape(-1)
    result = torch.zeros((), device=af.device, dtype=torch.float32)
    for start in range(0, af.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), af.numel())
        result += (af[start:stop].float() - bf[start:stop].float()).abs().sum()
    return result


def rho_from_l1(l1_next: torch.Tensor | float, l1_prev: torch.Tensor | float, eps: float = EPS) -> float:
    numerator = float(l1_next.detach().float().cpu()) if torch.is_tensor(l1_next) else float(l1_next)
    denominator = float(l1_prev.detach().float().cpu()) if torch.is_tensor(l1_prev) else float(l1_prev)
    return numerator / max(denominator, float(eps))


def clean_rho(l1_prev: torch.Tensor, l1_next: torch.Tensor, eps: float = EPS) -> float:
    """Two-point-window rho: ||Z[t+1]-Z[t]||_1 / ||Z[t]-Z[t-1]||_1."""
    return rho_from_l1(l1_next, l1_prev, eps)


def code_rho(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1.0e-8) -> float:
    """Reference for the legacy implementation's elementwise-epsilon form."""
    numerator = float((c.detach().float() - b.detach().float() + eps).abs().sum().cpu())
    denominator = float((b.detach().float() - a.detach().float() + eps).abs().sum().cpu())
    return numerator / max(denominator, eps)


def centered_variance(x: torch.Tensor) -> torch.Tensor:
    """Population condition variance normalized by feature element count."""
    xf = x.detach().float()
    if xf.ndim < 2:
        raise ValueError("condition dimension must be the first dimension")
    mean = xf.mean(dim=0, keepdim=True)
    return ((xf - mean) ** 2).sum() / float(xf.shape[0] * xf[0].numel())


def pair_energy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return ((af - bf) ** 2).mean() / 2.0


def pairwise_population_variance(energies: Sequence[float], num_conditions: int) -> float:
    if not energies:
        return float("nan")
    k = int(num_conditions)
    return ((k - 1.0) / k) * float(np.mean(np.asarray(energies, dtype=np.float64)))


def rms_feature(x: torch.Tensor) -> float:
    return float(torch.sqrt((x.detach().float() ** 2).mean()).cpu())


def temporal_metrics(previous: torch.Tensor, current: torch.Tensor, eps: float = EPS) -> tuple[float, float, float]:
    a = math.sqrt(0.5 * (float((previous.detach().float() ** 2).mean()) + float((current.detach().float() ** 2).mean())))
    d = math.sqrt(float(((current.detach().float() - previous.detach().float()) ** 2).mean()))
    return a, d, d / (a + eps)


def safe_log10(value: float, floor: float = 1.0e-30) -> float:
    return math.log10(max(float(value), floor))


def gain_db(v_base: float, v_diff: float, eps: float = EPS) -> float:
    return 10.0 * safe_log10((float(v_base) + eps) / (float(v_diff) + eps))


def valid_rho_index(step_idx: int, num_steps: int) -> bool:
    return 1 <= int(step_idx) <= int(num_steps) - 2


def all_unordered_pairs(num_conditions: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_conditions) for j in range(i + 1, num_conditions)]


def select_pairs(num_conditions: int, count: int, seed: int) -> list[tuple[int, int]]:
    pairs = all_unordered_pairs(num_conditions)
    if count > len(pairs):
        raise ValueError(f"Requested {count} pairs but only {len(pairs)} exist")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(pairs), size=int(count), replace=False)
    return [pairs[int(i)] for i in sorted(indices)]


def flatten_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): flatten_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [flatten_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_rows(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Write CSV (gzip when requested by suffix); returns the actual path."""
    import csv
    import gzip

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(flatten_jsonable(row)) for row in rows]
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key); seen.add(key)
    opener = gzip.open if target.suffix == ".gz" else open
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": ""} if target.suffix == ".gz" else {"mode": "w", "encoding": "utf-8", "newline": ""}
    with opener(target, **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return str(target)


def read_rows(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    import csv
    import gzip

    target = Path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    kwargs = {"mode": "rt", "encoding": "utf-8", "newline": ""}
    with opener(target, **kwargs) as handle:
        return list(csv.DictReader(handle))


def cast_row(row: Mapping[str, Any], numeric: Iterable[str] = ()) -> dict[str, Any]:
    result = dict(row)
    for key in numeric:
        if key not in result or result[key] in (None, ""):
            continue
        try:
            result[key] = float(result[key])
        except (ValueError, TypeError):
            pass
    return result


def percentile_ci(values: Sequence[float], trials: int = 2000, seed: int = 2027) -> dict[str, float]:
    clean = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p025": float("nan"), "p975": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    if clean.size == 1:
        boot = np.repeat(clean, max(1, int(trials)))
    else:
        boot = np.empty(max(1, int(trials)), dtype=np.float64)
        for idx in range(boot.size):
            boot[idx] = rng.choice(clean, size=clean.size, replace=True).mean()
    return {"mean": float(clean.mean()), "median": float(np.median(clean)), "p025": float(np.percentile(boot, 2.5)), "p975": float(np.percentile(boot, 97.5)), "n": int(clean.size)}


def rank_corr(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    try:
        from scipy.stats import kendalltau, spearmanr
        a = np.asarray(x, dtype=np.float64); b = np.asarray(y, dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan")
        return float(spearmanr(a, b).statistic), float(kendalltau(a, b).statistic)
    except Exception:
        return float("nan"), float("nan")


def jaccard_at_fraction(reference: Sequence[float], candidate: Sequence[float], fraction: float) -> float:
    n = len(reference)
    if n == 0:
        return float("nan")
    k = max(1, min(n, int(round(n * float(fraction)))))
    ref_idx = set(np.argsort(np.asarray(reference))[:k].tolist())
    cand_idx = set(np.argsort(np.asarray(candidate))[:k].tolist())
    union = ref_idx | cand_idx
    return len(ref_idx & cand_idx) / len(union) if union else 1.0


def ensure_finite(rows: Sequence[Mapping[str, Any]], row_keys: Sequence[str]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        for key in row_keys:
            value = row.get(key)
            if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                failures.append({"key": key, "row": dict(row)})
    return failures


def environment_snapshot(repo_root: Path, checkpoint_paths: Sequence[Path] = ()) -> dict[str, Any]:
    import platform
    import subprocess
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
    }
    try:
        snapshot["git_commit"] = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        snapshot["git_status"] = subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True)
    except Exception as exc:
        snapshot["git_error"] = repr(exc)
    if torch.cuda.is_available():
        snapshot["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    snapshot["checkpoints"] = []
    for checkpoint in checkpoint_paths:
        item = {"path": str(checkpoint), "exists": checkpoint.exists()}
        if checkpoint.is_file():
            item["size_bytes"] = checkpoint.stat().st_size
            item["sha256"] = sha256_file(checkpoint)
        elif checkpoint.is_dir():
            files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
            item["files"] = [{"path": str(p.relative_to(checkpoint)), "size_bytes": p.stat().st_size} for p in files]
            manifest = hashlib.sha256()
            for p in files:
                manifest.update(str(p.relative_to(checkpoint)).encode())
                manifest.update(str(p.stat().st_size).encode())
            item["size_manifest_sha256"] = manifest.hexdigest()
        snapshot["checkpoints"].append(item)
    return snapshot


def json_hash(value: Any) -> str:
    encoded = json.dumps(flatten_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
