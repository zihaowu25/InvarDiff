import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import all_unordered_pairs, pair_energy, pairwise_population_variance, select_pairs


def test_all_pair_estimator_matches_population_variance():
    x = torch.tensor([[0.0], [1.0], [3.0], [8.0]])
    energies = [float(pair_energy(x[i], x[j])) for i, j in all_unordered_pairs(len(x))]
    exact = ((x - x.mean(0)) ** 2).mean().item()
    assert abs(pairwise_population_variance(energies, len(x)) - exact) < 1e-7


def test_pair_selection_is_reproducible_and_without_replacement():
    first = select_pairs(8, 10, 2027)
    second = select_pairs(8, 10, 2027)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(i < j for i, j in first)
