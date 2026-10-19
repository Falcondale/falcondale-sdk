"""
Information theory helpers used to compose the feature selection QUBO.

Plug-in estimates of mutual information are biased upwards by the number of cells of the
histogram, so the way variables are discretised matters. Two strategies are available:

* **quantile**: columns with at most `bins` distinct values keep their levels, the rest are
  split by quantiles so that every bin holds a similar number of samples. Repeated values are
  never split, so a mass of zeros becomes one bin and the tail is shared by the other bins.
* **uniform**: `min(distinct values, bins)` equal width bins, the behaviour of falcondale 0.3.0.
"""

import numpy as np

STRATEGIES = ("quantile", "uniform")


def discretize(values: np.ndarray, bins: int = 5, strategy: str = "quantile") -> np.ndarray:
    """Maps a column to integer bin codes starting at 0.

    Args:
        values (np.ndarray): One dimensional array
        bins (int): Maximal number of bins
        strategy (str): "quantile" or "uniform"

    Returns:
        np.ndarray: Bin code of each sample
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown discretization strategy '{strategy}'. Available ones: {list(STRATEGIES)}")

    values = np.asarray(values)
    levels, codes, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Few levels (binary, ordinal) and non numeric columns are already discrete
    if not np.issubdtype(values.dtype, np.number) or (strategy == "quantile" and len(levels) <= bins):
        return codes

    if strategy == "uniform":
        edges = np.histogram_bin_edges(values, bins=min(len(levels), bins))
        return np.digitize(values, edges[1:-1])

    # Equal frequency bins that never split a level. A large mass of repeated values (e.g. zero
    # days in hospital) takes a bin of its own and the remaining bins are shared by the other samples.
    level_bin = np.empty(len(levels), dtype=int)
    current, filled, remaining = 0, 0, len(values)
    for i, count in enumerate(counts):
        level_bin[i] = current
        filled += count
        if current < bins - 1 and filled >= remaining / (bins - current):
            current, filled, remaining = current + 1, 0, remaining - filled
    return level_bin[codes]


def joint_prob(*codes: np.ndarray) -> np.ndarray:
    """Joint probability distribution of discretised variables, one axis per variable."""
    freq = np.zeros([int(code.max()) + 1 for code in codes])
    np.add.at(freq, tuple(codes), 1)
    return freq / len(codes[0])


def prob(dataset: np.ndarray, max_bins: int = 10, strategy: str = "uniform") -> np.ndarray:
    """Joint probability distribution P(X) for the given data.

    Args:
        dataset (np.ndarray): Samples x variables array
        max_bins (int): Maximal number of bins to discretize each variable
        strategy (str): Discretization strategy, see `discretize`

    Returns:
        np.ndarray: Joint probability distribution, one axis per variable
    """
    return joint_prob(*(discretize(column, max_bins, strategy) for column in np.asarray(dataset).T))


def shannon_entropy(joint_prob: np.ndarray) -> float:
    """Shannon entropy H(X) is minus the sum of P(X)log(P(X)) for probability distribution P(X)."""
    probs = joint_prob[joint_prob > 0]
    return float(-np.sum(probs * np.log2(probs)))


def conditional_shannon_entropy(p: np.ndarray, *conditional_indices: int) -> float:
    """Shannon entropy of P(X) conditional on the given variables."""
    axis = tuple(i for i in range(p.ndim) if i not in conditional_indices)

    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))


def mutual_information(p: np.ndarray, j: int) -> float:
    """Mutual information between all variables and variable j."""
    return shannon_entropy(np.sum(p, axis=j)) - conditional_shannon_entropy(p, j)


def conditional_mutual_information(p: np.ndarray, j: int, *conditional_indices: int) -> float:
    """Mutual information between variables X and variable j conditional on the given variables."""
    marginal_conditional_indices = [i - 1 if i > j else i for i in conditional_indices]

    return conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices) - conditional_shannon_entropy(
        p, j, *conditional_indices
    )
