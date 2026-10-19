"""
Information theory helpers used to compose the feature selection QUBO.
"""

import numpy as np


def prob(dataset: np.ndarray, max_bins: int = 10) -> np.ndarray:
    """Joint probability distribution P(X) for the given data.

    Args:
        dataset (np.ndarray): Samples x variables array
        max_bins (int): Maximal number of bins to discretize each variable

    Returns:
        np.ndarray: Joint probability distribution, one axis per variable
    """
    # bin by the number of different values per feature
    bins = [min(len(np.unique(column)), max_bins) for column in dataset.T]

    freq, _ = np.histogramdd(dataset, bins)
    return freq / np.sum(freq)


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
