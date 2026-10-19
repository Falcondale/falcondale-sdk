"""
This module focuses on feature selection methods to diminish the dimensionality of a target dataset.

Feature selection is posed as a QUBO problem (see `build_qubo`) that can be solved by
different (quantum and quantum-inspired) solvers:

* **sa**: Simulated Annealing, using D-Wave's open source samplers locally
* **sb**: Simulated Bifurcation, a quantum-inspired algorithm running on PyTorch
* **qaoa**: Quantum Approximate Optimization Algorithm simulated with Qiskit

Ref:
- https://dl.acm.org/citation.cfm?id=2623611
- https://arxiv.org/pdf/2203.13261.pdf
"""

import itertools
import warnings
from collections.abc import Callable, Hashable

import dimod
import numpy as np
import simulated_bifurcation as sb
import torch
from dimod.generators import combinations
from dwave.samplers import SimulatedAnnealingSampler
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from .data import Dataset
from .exceptions import TooManyQubitsNeeded
from .helpers.mutual_information import conditional_mutual_information, mutual_information, prob

Sample = dict[Hashable, int]

# Statevector simulation needs 2^n amplitudes, beyond this it will not fit in a laptop
MAX_QAOA_QUBITS = 20


def build_qubo(input_ds: Dataset, max_cols: int | None = None) -> dimod.BinaryQuadraticModel:
    """
    Builds the QUBO required for Quantum Feature Selection (QFS).

    Each binary variable represents a feature. Linear terms reward features informative
    about the target (mutual information) and quadratic terms reward pairs of features whose
    information is complementary (conditional mutual information). A penalty term forces
    the selection of exactly `max_cols` features.

    Args:
        input_ds (Dataset): Falcondale Dataset data type
        max_cols (int): Maximum number of features to be selected

    Returns:
        BinaryQuadraticModel: Dimod model holding the QUBO information
    """
    features, target = input_ds.as_arrays()
    col_list = input_ds.columns()

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Importance
    for i, column in enumerate(col_list):
        bqm.add_variable(column, -mutual_information(prob(np.column_stack([target, features[:, i]])), 1))

    # Redundancy
    for (i, col_i), (j, col_j) in itertools.combinations(enumerate(col_list), 2):
        cmi_ij = conditional_mutual_information(prob(np.column_stack([target, features[:, i], features[:, j]])), 1, 2)
        cmi_ji = conditional_mutual_information(prob(np.column_stack([target, features[:, j], features[:, i]])), 1, 2)
        bqm.add_interaction(col_i, col_j, -(cmi_ij + cmi_ji))

    # Penalty
    if max_cols and max_cols < len(col_list):
        bqm.update(combinations(bqm.variables, max_cols, strength=10 * max_cols))

    bqm.normalize()  # scale the BQM to (-1, 1) biases

    return bqm


def solve_sa(bqm: dimod.BinaryQuadraticModel, seed: int | None = None, **kwargs) -> Sample:
    """
    Solves the QUBO using simulated annealing.

    Args:
        bqm (BinaryQuadraticModel): Problem to be solved
        seed (int): Random seed
        **kwargs: Additional arguments for `SimulatedAnnealingSampler.sample` (num_reads, num_sweeps,...)

    Returns:
        dict: Lowest energy assignment found
    """
    sampleset = SimulatedAnnealingSampler().sample(bqm, seed=seed, **kwargs)
    return dict(sampleset.first.sample)


def solve_sb(bqm: dimod.BinaryQuadraticModel, seed: int | None = None, **kwargs) -> Sample:
    """
    Solves the QUBO using the simulated bifurcation algorithm.

    Ref:
    - https://github.com/bqth29/simulated-bifurcation-algorithm

    Args:
        bqm (BinaryQuadraticModel): Problem to be solved
        seed (int): Random seed
        **kwargs: Additional arguments for `simulated_bifurcation.minimize` (agents, max_steps,...)

    Returns:
        dict: Lowest energy assignment found
    """
    variables = list(bqm.variables)
    index = {var: i for i, var in enumerate(variables)}

    # x^T Q x + l^T x form
    matrix = torch.zeros((len(variables), len(variables)))
    for (u, v), bias in bqm.quadratic.items():
        matrix[index[u], index[v]] = bias
    linear = torch.tensor([bqm.linear[var] for var in variables])

    if seed is not None:
        torch.manual_seed(seed)

    kwargs.setdefault("verbose", False)
    solution, _ = sb.minimize(
        matrix,  # ty: ignore[invalid-argument-type]  (tensors are valid polynomial data)
        linear,  # ty: ignore[invalid-argument-type]
        domain="binary",
        best_only=True,
        **kwargs,
    )

    return {var: int(value) for var, value in zip(variables, solution.tolist(), strict=True)}


def ising_operator(bqm: dimod.BinaryQuadraticModel) -> tuple[SparsePauliOp, list[Hashable]]:
    """
    Maps the QUBO to an Ising Hamiltonian made of Pauli Z operators.

    A binary variable x maps to the Z eigenvalue z as x = (1 - z) / 2, so that measuring
    a qubit in state |1> means the corresponding feature is selected.

    Args:
        bqm (BinaryQuadraticModel): Problem to be mapped

    Returns:
        tuple: Hamiltonian as a SparsePauliOp and the variable associated to each qubit
    """
    variables = list(bqm.variables)
    index = {var: i for i, var in enumerate(variables)}
    h, J, _ = bqm.to_ising()

    # dimod spins are s = 2x - 1 = -z
    terms = [("Z", [index[v]], -bias) for v, bias in h.items()]
    terms += [("ZZ", [index[u], index[v]], bias) for (u, v), bias in J.items()]

    return SparsePauliOp.from_sparse_list(terms, num_qubits=len(variables)).simplify(), variables


def solve_qaoa(
    bqm: dimod.BinaryQuadraticModel, seed: int | None = None, reps: int = 2, shots: int = 1024, maxiter: int = 200
) -> Sample:
    """
    Solves the QUBO using QAOA on a local statevector simulator.

    Parameters are optimized with COBYLA and the final circuit is sampled. Among the
    sampled bitstrings, the one with the lowest QUBO energy is returned.

    Args:
        bqm (BinaryQuadraticModel): Problem to be solved
        seed (int): Random seed
        reps (int): Number of QAOA layers
        shots (int): Number of samples taken from the optimized circuit
        maxiter (int): Maximum number of optimizer iterations

    Returns:
        dict: Lowest energy assignment found
    """
    operator, variables = ising_operator(bqm)
    if operator.num_qubits > MAX_QAOA_QUBITS:
        raise TooManyQubitsNeeded(operator.num_qubits, MAX_QAOA_QUBITS)

    ansatz = qaoa_ansatz(operator, reps=reps)
    estimator = StatevectorEstimator(seed=seed)

    def cost(params: np.ndarray) -> float:
        pub = (ansatz, operator, params)
        return float(estimator.run([pub]).result()[0].data.evs)  # ty: ignore[invalid-argument-type, unresolved-attribute]

    rng = np.random.default_rng(seed)
    result = minimize(cost, rng.uniform(0, np.pi, ansatz.num_parameters), method="COBYLA", options={"maxiter": maxiter})

    circuit = ansatz.copy()
    circuit.measure_all()
    sampler = StatevectorSampler(seed=seed)
    counts = sampler.run([(circuit, result.x)], shots=shots).result()[0].data.meas.get_counts()  # ty: ignore[unresolved-attribute]

    # Qiskit bitstrings place qubit 0 on the right
    samples = [dict(zip(variables, map(int, reversed(bits)), strict=True)) for bits in counts]
    return min(samples, key=bqm.energy)


SOLVERS: dict[str, Callable[..., Sample]] = {
    "sa": solve_sa,
    "sb": solve_sb,
    "qaoa": solve_qaoa,
}


def select_features(input_ds: Dataset, max_cols: int, method: str = "sa", **kwargs) -> list[str]:
    """
    Selects the best `max_cols` features for the target using the requested solver.

    Args:
        input_ds (Dataset): Falcondale Dataset data type
        max_cols (int): Maximum number of features to be selected
        method (str): One of the keys in `SOLVERS`
        **kwargs: Additional solver arguments (seed, num_reads, reps,...)

    Returns:
        list[str]: Selected column list
    """
    method = method.lower()
    if method not in SOLVERS:
        raise ValueError(f"Unknown feature selection method '{method}'. Available ones: {sorted(SOLVERS)}")

    if method == "qaoa" and len(input_ds.columns()) > 12:
        warnings.warn(
            f"QAOA with {len(input_ds.columns())} qubits may take a long time to simulate.",
            RuntimeWarning,
            stacklevel=2,
        )

    bqm = build_qubo(input_ds, max_cols)
    selection = SOLVERS[method](bqm, **kwargs)

    return [str(col) for col in input_ds.columns() if selection[col] == 1]
