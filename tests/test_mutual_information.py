import numpy as np
import pytest

from falcondale.helpers.mutual_information import (
    conditional_mutual_information,
    mutual_information,
    prob,
    shannon_entropy,
)


def test_entropy_of_fair_coin():
    assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)


def test_mutual_information_of_copies_is_entropy():
    x = np.tile([0, 1], 50)
    assert mutual_information(prob(np.column_stack([x, x])), 1) == pytest.approx(1.0)


def test_mutual_information_of_independent_variables():
    x = np.tile([0, 1], 50)
    y = np.repeat([0, 1], 50)
    assert mutual_information(prob(np.column_stack([x, y])), 1) == pytest.approx(0.0)


def test_conditioning_on_a_copy_removes_information():
    x = np.tile([0, 1], 50)
    assert conditional_mutual_information(prob(np.column_stack([x, x, x])), 1, 2) == pytest.approx(0.0)
