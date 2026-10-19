"""
Schrödinger potential of a Parzen wave function, the core of Quantum Clustering
(Horn and Gottlieb, "The Method of Quantum Clustering", NIPS 2001, and Physical Review
Letters 88, 018702, 2002).

The wave function is a sum of Gaussians centred on the data points,

    psi(x) = sum_i exp(-|x - x_i|^2 / (2 sigma^2))

and, asking psi to be the ground state of a Schrödinger equation, the potential is

    V(x) = E - d / 2 + 1 / (2 sigma^2 psi(x)) sum_i |x - x_i|^2 exp(-|x - x_i|^2 / (2 sigma^2))

Its minima play the role of cluster centres. Writing p_i(x) for the normalised Gaussian
weights (a softmax of -|x - x_i|^2 / (2 sigma^2)), the non constant part of V is
E_p[|r|^2] / (2 sigma^2) with r = x - x_i, which keeps the computation stable.
"""

import numpy as np
from scipy.special import softmax

# Query points evaluated at once, bounding memory to CHUNK x samples x dimensions
CHUNK = 256


def _weights(points: np.ndarray, data: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = points[:, None, :] - data[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", r, r)
    return r, d2, softmax(-d2 / (2 * sigma**2), axis=1)


def potential(points: np.ndarray, data: np.ndarray, sigma: float) -> np.ndarray:
    """Quantum potential at the given points, up to the constant E - d / 2."""
    values = []
    for start in range(0, len(points), CHUNK):
        _, d2, p = _weights(points[start : start + CHUNK], data, sigma)
        values.append(np.sum(p * d2, axis=1) / (2 * sigma**2))
    return np.concatenate(values) if values else np.empty(0)


def potential_gradient(points: np.ndarray, data: np.ndarray, sigma: float) -> np.ndarray:
    """Gradient of the quantum potential at the given points."""
    s2 = sigma**2
    gradients = []
    for start in range(0, len(points), CHUNK):
        r, d2, p = _weights(points[start : start + CHUNK], data, sigma)
        m1 = np.einsum("ij,ijk->ik", p, r)
        m2 = np.sum(p * d2, axis=1, keepdims=True)
        m3 = np.einsum("ij,ijk->ik", p * d2, r)
        gradients.append((2 * m1 - (m3 - m2 * m1) / s2) / (2 * s2))
    return np.concatenate(gradients) if gradients else np.empty((0, data.shape[1]))


def descend(
    points: np.ndarray,
    data: np.ndarray,
    sigma: float,
    learning_rate: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """
    Gradient descent of the points on the potential built from `data`.

    Steps are scaled by sigma^2, so that the learning rate does not depend on the units of the data.

    Returns:
        tuple: Final positions and number of iterations
    """
    positions = np.array(points, dtype=float, copy=True)
    for iteration in range(1, max_iter + 1):
        step = learning_rate * sigma**2 * potential_gradient(positions, data, sigma)
        positions -= step
        if np.max(np.abs(step), initial=0.0) < tol * sigma:
            return positions, iteration
    return positions, max_iter


def membership(points: np.ndarray, data: np.ndarray, sigma: float, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Share of the wave function at each point contributed by the samples of each cluster."""
    proba = []
    for start in range(0, len(points), CHUNK):
        _, _, p = _weights(points[start : start + CHUNK], data, sigma)
        proba.append(np.stack([p[:, labels == c].sum(axis=1) for c in range(n_clusters)], axis=1))
    # Rounding can push the sums slightly above one
    return np.clip(np.concatenate(proba), 0, 1) if proba else np.empty((0, n_clusters))
