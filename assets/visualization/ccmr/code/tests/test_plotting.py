from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from plot_ccmr import _ecdf_points, _finite, _global_limits, _heat_matrix, _paired_finite  # noqa: E402


def test_plot_helpers_drop_nan_and_inf_values():
    rows = [
        {"value": "nan", "x": 1.0, "y": 2.0},
        {"value": 3.0, "x": "inf", "y": 4.0},
        {"value": 5.0, "x": 5.0, "y": 6.0},
    ]
    assert np.array_equal(_finite(rows, "value"), np.asarray([3.0, 5.0]))
    x, y = _paired_finite(rows, "x", "y")
    assert np.array_equal(x, np.asarray([1.0, 5.0]))
    assert np.array_equal(y, np.asarray([2.0, 6.0]))


def test_heat_matrix_uses_score_step_key_and_masks_invalid_rows():
    rows = [
        {"module_family": "single", "module_name": "mlp", "score_step_idx": 0, "layer_idx": 0, "rho_mean": "nan"},
        {"module_family": "single", "module_name": "mlp", "score_step_idx": 1, "layer_idx": 0, "rho_mean": 0.25},
        {"module_family": "single", "module_name": "mlp", "score_step_idx": 2, "layer_idx": 0, "rho_mean": 0.5},
    ]
    result = _heat_matrix(rows, "rho_mean", "score_step_idx", transform=lambda value: math.log10(value))
    assert result is not None
    matrix, steps, layers = result
    assert steps == [1, 2]
    assert layers == [0]
    assert np.allclose(matrix, np.asarray([[math.log10(0.25), math.log10(0.5)]]))


def test_global_limits_interpret_fractional_robust_quantiles():
    values = np.arange(1000, dtype=float)
    limits = _global_limits(values, [0.005, 0.995], (0.0, 1.0))
    assert limits[0] < 10.0
    assert limits[1] > 990.0


def test_ecdf_is_sorted_and_monotone():
    x, y = _ecdf_points(np.asarray([3.0, np.nan, 1.0, 2.0]))
    assert np.array_equal(x, np.asarray([1.0, 2.0, 3.0]))
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == 1.0
