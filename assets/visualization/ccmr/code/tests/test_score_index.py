import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import valid_rho_index, rho_from_l1


def test_rho_is_scored_at_middle_observation():
    # Z=[0, 1, 3, 6] gives l1=[1,2,3], so score indices 1 and 2 are 2 and 1.5.
    l1 = [1.0, 2.0, 3.0]
    assert rho_from_l1(l1[1], l1[0]) == 2.0
    assert rho_from_l1(l1[2], l1[1]) == 1.5


def test_rho_boundaries_are_invalid():
    assert not valid_rho_index(0, 4)
    assert valid_rho_index(1, 4)
    assert valid_rho_index(2, 4)
    assert not valid_rho_index(3, 4)
