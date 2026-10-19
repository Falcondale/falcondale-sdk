"""
Euclidean distances estimated by quantum interference, as used by quantum k-means and k-medians
(Khan et al., arXiv:1909.12183; Poggiali et al., arXiv:2212.06691; Belis et al., arXiv:2301.10780).

Two vectors u and v are amplitude encoded together, |psi> = (|0>|u> + |1>|v>) / sqrt(Z) with
Z = |u|^2 + |v|^2. A Hadamard gate on the most significant qubit makes both halves interfere and
the probability of measuring it in |1> is

    P(1) = |u - v|^2 / (2 Z)

so the squared distance is estimated as 2 Z P(1). With a finite number of shots P(1) is the
frequency of ones, which adds the sampling noise of a real device.
"""

import numpy as np

from .._optional import optional_dependencies


def _encode(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint states (rows) of every pair, with each half padded to a power of two."""
    size = 1 << max(int(np.ceil(np.log2(max(u.shape[-1], 1)))), 0)
    halves = np.zeros((len(u), 2, size))
    halves[:, 0, : u.shape[-1]] = u
    halves[:, 1, : v.shape[-1]] = v
    norm = np.einsum("ijk,ijk->i", halves, halves)
    return halves.reshape(len(u), 2 * size) / np.sqrt(np.where(norm > 0, norm, 1))[:, None], norm


def interference_circuit(u: np.ndarray, v: np.ndarray):
    """
    Circuit estimating the distance between two vectors, to run it on any Qiskit backend.

    The most significant qubit is measured, P(1) times 2 (|u|^2 + |v|^2) is the squared distance.
    Requires Qiskit, available with `pip install "falcondale[clustering]"`.

    Args:
        u (np.ndarray): First vector
        v (np.ndarray): Second vector

    Returns:
        QuantumCircuit: State preparation, Hadamard on the top qubit and its measurement
    """
    with optional_dependencies("clustering"):
        from qiskit import ClassicalRegister, QuantumCircuit
        from qiskit.circuit.library import StatePreparation

    state, norm = _encode(np.atleast_2d(u), np.atleast_2d(v))
    if norm[0] == 0:
        raise ValueError("Both vectors are zero, their distance is trivially zero.")
    qubits = int(np.log2(state.shape[1]))
    circuit = QuantumCircuit(qubits)
    circuit.add_register(ClassicalRegister(1, "meas"))
    circuit.append(StatePreparation(state[0]), range(qubits))
    circuit.h(qubits - 1)
    circuit.measure(qubits - 1, 0)
    return circuit


def interference_probability(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Probability of measuring the top qubit in |1> for every pair of rows of u and v.

    The statevector of the circuit is simulated directly: amplitudes of the joint state are
    combined by the Hadamard gate as (top half - bottom half) / sqrt(2).
    """
    state, _ = _encode(u, v)
    half = state.shape[1] // 2
    ones = (state[:, :half] - state[:, half:]) / np.sqrt(2)
    return np.einsum("ij,ij->i", ones, ones)


def interference_distances(
    x: np.ndarray, centers: np.ndarray, shots: int | None = 1024, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Squared Euclidean distances between samples and centers estimated by interference.

    Args:
        x (np.ndarray): Samples x features
        centers (np.ndarray): Centers x features
        shots (int): Measurements per pair, None for the exact probabilities
        rng (np.random.Generator): Random generator for the measurements

    Returns:
        np.ndarray: Samples x centers squared distances
    """
    u = np.repeat(x, len(centers), axis=0)
    v = np.tile(centers, (len(x), 1))
    p = interference_probability(u, v)
    if shots is not None:
        p = (rng or np.random.default_rng()).binomial(shots, np.clip(p, 0, 1)) / shots
    z = np.einsum("ij,ij->i", u, u) + np.einsum("ij,ij->i", v, v)
    return (2 * z * p).reshape(len(x), len(centers))
