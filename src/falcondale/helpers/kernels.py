"""
Quantum kernels implemented with PennyLane
"""

import numpy as np
import pennylane as qml


class AngleKernel:
    """
    Fidelity kernel K(a, b) = |<phi(a)|phi(b)>|^2 where phi is an angle embedding.

    Instead of running a compute-uncompute circuit per pair of samples, the embedded
    statevector of each sample is simulated once and the kernel is obtained as
    an inner product. This needs O(samples) simulations instead of O(samples^2).

    The object can be used as a scikit-learn SVC kernel and, unlike a local function,
    it can be pickled (the QNode is rebuilt when needed).
    """

    def __init__(self):
        self._qnode = None
        self._num_qubits = 0

    def _states(self, x: np.ndarray) -> np.ndarray:
        num_qubits = x.shape[1]
        if self._qnode is None or self._num_qubits != num_qubits:

            def circuit(features):
                qml.AngleEmbedding(features, wires=range(num_qubits))
                return qml.state()

            self._qnode = qml.QNode(circuit, qml.device("default.qubit", wires=num_qubits))
            self._num_qubits = num_qubits

        return np.asarray(self._qnode(x)).reshape(len(x), -1)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Kernel matrix between the samples in a and b."""
        states_a = self._states(np.asarray(a, dtype=float))
        states_b = self._states(np.asarray(b, dtype=float))
        return np.abs(states_a.conj() @ states_b.T) ** 2

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()
