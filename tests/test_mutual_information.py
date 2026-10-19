import numpy as np
import pytest

from falcondale.helpers.mutual_information import (
    conditional_mutual_information,
    discretize,
    joint_prob,
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


@pytest.mark.parametrize(
    ("values", "counts"),
    [
        (np.tile([0, 1], 50), [50, 50]),  # binary columns keep their levels
        (np.repeat([3, 1, 4, 0, 2], 20), [20, 20, 20, 20, 20]),  # ... as ordinal ones with few levels
        (np.arange(17).repeat(10), [40, 40, 30, 30, 30]),  # many levels are grouped
        (np.r_[np.zeros(800), np.arange(1, 201)], [800, 50, 50, 50, 50]),  # a mass of zeros gets its own bin
        (np.random.default_rng(0).normal(size=1000), [200] * 5),
    ],
)
def test_quantile_discretization(values, counts):
    assert np.bincount(discretize(values, bins=5)).tolist() == counts


def test_non_numeric_columns_keep_their_levels():
    assert np.bincount(discretize(np.array(list("abcdefg") * 2))).tolist() == [2] * 7


def test_unknown_strategy():
    with pytest.raises(ValueError, match="quantile"):
        discretize(np.arange(10), strategy="kmeans")


def test_uniform_strategy_reproduces_equal_width_histograms():
    rng = np.random.default_rng(1)
    data = np.column_stack([rng.normal(size=300), rng.integers(0, 15, 300), rng.integers(0, 2, 300)])
    bins = [min(len(np.unique(column)), 10) for column in data.T]
    freq, _ = np.histogramdd(data, bins)
    assert shannon_entropy(prob(data)) == pytest.approx(shannon_entropy(freq / freq.sum()))


def test_quantile_bins_reduce_the_bias_of_many_levels():
    """
    Independent noise with 17 levels looks informative to a binary target, the plug-in bias being
    about (cells - 1) / (2 n ln 2) bits. Ten equal width bins give 9 extra cells, five quantile bins 4.
    """
    rng = np.random.default_rng(2)
    n, biased, unbiased = 700, [], []
    for _ in range(50):
        target, noise = rng.integers(0, 2, n), rng.integers(0, 17, n)
        biased.append(mutual_information(prob(np.column_stack([target, noise])), 1))
        unbiased.append(mutual_information(joint_prob(discretize(target), discretize(noise)), 1))

    assert np.mean(biased) == pytest.approx(9 / (2 * n * np.log(2)), rel=0.25)
    assert np.mean(unbiased) == pytest.approx(4 / (2 * n * np.log(2)), rel=0.25)
