import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import centered_variance, gain_db, l1_distance, temporal_metrics


def test_common_mode_is_removed_from_adjacent_difference():
    conditions = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    z0 = torch.ones(8, 4, 3) * 2.0 + conditions
    z1 = torch.ones(8, 4, 3) * 7.0 + conditions
    assert centered_variance(z0 - z1).item() < 1e-8
    assert gain_db((centered_variance(z0) + centered_variance(z1)).item() / 2, centered_variance(z0-z1).item()) > 80


def test_condition_amplitude_change_increases_difference_variance():
    conditions = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    z0 = conditions
    z1 = 1.1 * conditions
    z2 = 2.0 * conditions
    assert centered_variance(z1 - z0) < centered_variance(z2 - z1)


def test_random_data_does_not_force_ccmr():
    generator = torch.Generator().manual_seed(3)
    a = torch.randn(12, 4, 5, generator=generator)
    b = torch.randn(12, 4, 5, generator=generator)
    c = torch.randn(12, 4, 5, generator=generator)
    base = (centered_variance(a) + centered_variance(b)) / 2
    diff = centered_variance(c - b)
    assert bool(diff >= 0)
    assert np.isfinite(gain_db(float(base), float(diff)))


def test_gain_is_zero_when_raw_and_difference_variance_match():
    assert abs(gain_db(3.25, 3.25)) < 1e-12


def test_aligned_difference_is_smaller_than_shuffled_difference():
    # Two conditions share the same temporal displacement when aligned; a
    # permutation destroys that cancellation.
    previous = torch.tensor([[0.0, 1.0], [4.0, 5.0]])
    current = previous + torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    aligned = centered_variance(current - previous).item()
    shuffled = centered_variance(current - previous.flip(0)).item()
    assert aligned < shuffled


def test_float16_inputs_use_float32_statistics():
    generator = torch.Generator().manual_seed(4)
    a = torch.randn(3, 5, generator=generator)
    b = torch.randn(3, 5, generator=generator)
    value16 = l1_distance(a.half(), b.half()).item()
    value32 = l1_distance(a, b).item()
    assert abs(value16 - value32) / max(value32, 1e-8) < 2e-3
    assert temporal_metrics(a[0], b[0])[2] >= 0
