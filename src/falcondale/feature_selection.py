"""
This module focuses on feature selection methods to diminish the dimensionality of a target dataset.

Feature selection is posed as a QUBO problem (see `build_qubo`) that can be solved by
different (quantum and quantum-inspired) solvers:

* **sa**: Simulated Annealing, using D-Wave's open source samplers locally
* **sb**: Simulated Bifurcation, a quantum-inspired algorithm running on PyTorch
* **qaoa**: Quantum Approximate Optimization Algorithm simulated with Qiskit
* **exact**: Enumeration of every subset, the reference for small problems

Requires the optional `feature-selection` dependencies: `pip install "falcondale[feature-selection]"`.

Ref:
- https://dl.acm.org/citation.cfm?id=2623611
- https://arxiv.org/pdf/2203.13261.pdf
- https://doi.org/10.1007/s42484-023-00099-z
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._optional import optional_dependencies
from .data import Dataset
from .exceptions import TooManyQubitsNeeded
from .helpers.mutual_information import (
    conditional_mutual_information,
    discretize,
    joint_prob,
    mutual_information,
)

if TYPE_CHECKING:
    import dimod
    from qiskit.quantum_info import SparsePauliOp

Sample = dict[Hashable, int]

FORMULATIONS = ("mrmr", "cmi")

# Statevector simulation needs 2^n amplitudes, beyond this it will not fit in a laptop
MAX_QAOA_QUBITS = 20

# About a few seconds of NumPy, C(30, 6) is 593,775 subsets
MAX_EXACT_SUBSETS = 5_000_000


def _discretized(input_ds: Dataset, bins: int, strategy: str) -> tuple[list[np.ndarray], np.ndarray]:
    features, target = input_ds.as_arrays()
    codes = [discretize(column, bins, strategy) for column in features.T]
    return codes, discretize(target, bins, strategy)


def relevance(input_ds: Dataset, bins: int = 5, strategy: str = "quantile") -> pd.Series:
    """
    Mutual information (in bits) between each selected feature and the target.

    Args:
        input_ds (Dataset): Falcondale Dataset data type
        bins (int): Maximal number of bins per variable
        strategy (str): Discretization strategy, "quantile" or "uniform"

    Returns:
        pd.Series: Relevance of each feature, sorted from the most to the least relevant
    """
    codes, target = _discretized(input_ds, bins, strategy)
    scores = [mutual_information(joint_prob(target, column), 1) for column in codes]
    return pd.Series(scores, index=input_ds.columns(), name="relevance").sort_values(ascending=False, kind="stable")


def build_qubo(
    input_ds: Dataset,
    max_cols: int | None = None,
    formulation: str = "mrmr",
    bins: int = 5,
    strategy: str = "quantile",
) -> dimod.BinaryQuadraticModel:
    """
    Builds the QUBO required for Quantum Feature Selection (QFS).

    Each binary variable represents a feature and linear terms reward the features informative
    about the target, I(x_i; y). Quadratic terms depend on the formulation:

    * **mrmr** (minimum redundancy, maximum relevance): pairs of features sharing information,
      I(x_i; x_j), are penalized with weight 1 / (max_cols - 1). Only pairwise histograms are needed.
    * **cmi**: pairs of features whose information is complementary given the target are rewarded
      with the conditional mutual information. It needs target x feature x feature histograms,
      which are sparse unless there are many samples. It was the only formulation in falcondale 0.3.0.

    A penalty term forces the selection of exactly `max_cols` features. Its strength is the largest
    change a single feature can make on the objective of a nearly feasible selection, so that the lowest
    energy is always reached by a selection of `max_cols` features while keeping as much resolution as
    possible for the objective.

    `formulation="cmi", strategy="uniform", bins=10` reproduces the information terms of falcondale 0.3.0.

    Args:
        input_ds (Dataset): Falcondale Dataset data type
        max_cols (int): Maximum number of features to be selected
        formulation (str): "mrmr" or "cmi"
        bins (int): Maximal number of bins per variable
        strategy (str): Discretization strategy, "quantile" or "uniform" (see `discretize`)

    Returns:
        BinaryQuadraticModel: Dimod model holding the QUBO information
    """
    with optional_dependencies("feature-selection"):
        import dimod
        from dimod.generators import combinations

    formulation = formulation.lower()
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown QUBO formulation '{formulation}'. Available ones: {list(FORMULATIONS)}")

    codes, target = _discretized(input_ds, bins, strategy)
    col_list = input_ds.columns()

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Importance
    for column, code in zip(col_list, codes, strict=True):
        bqm.add_variable(column, -mutual_information(joint_prob(target, code), 1))

    pairs = itertools.combinations(zip(col_list, codes, strict=True), 2)
    if formulation == "mrmr":
        # Redundancy, averaged over the pairs of a selection of size k
        weight = 1 / max((max_cols or len(col_list)) - 1, 1)
        for (col_i, code_i), (col_j, code_j) in pairs:
            bqm.add_interaction(col_i, col_j, weight * mutual_information(joint_prob(code_i, code_j), 1))
    else:
        # Complementarity
        for (col_i, code_i), (col_j, code_j) in pairs:
            cmi_ij = conditional_mutual_information(joint_prob(target, code_i, code_j), 1, 2)
            cmi_ji = conditional_mutual_information(joint_prob(target, code_j, code_i), 1, 2)
            bqm.add_interaction(col_i, col_j, -(cmi_ij + cmi_ji))

    # Penalty
    if max_cols and max_cols < len(col_list):
        bqm.update(combinations(bqm.variables, max_cols, strength=_penalty_strength(bqm, max_cols)))

    bqm.normalize()  # scale the BQM to (-1, 1) biases

    return bqm


def _penalty_strength(bqm: dimod.BinaryQuadraticModel, max_cols: int) -> float:
    """
    Smallest strength keeping the ground state feasible, plus a margin.

    A selection with one feature too many (or too few) has at most `max_cols` other selected features,
    so adding or removing feature i changes the objective by at most |h_i| plus its `max_cols` largest
    |J_ij|. The penalty grows by (2d - 1) times the strength for d extra features, which covers the rest.
    """
    neighbours: dict[Hashable, list[float]] = {var: [] for var in bqm.variables}
    for (u, v), bias in bqm.quadratic.items():
        neighbours[u].append(abs(bias))
        neighbours[v].append(abs(bias))
    change = [abs(bqm.linear[var]) + sum(sorted(js, reverse=True)[:max_cols]) for var, js in neighbours.items()]
    return 1.1 * float(max(max(change, default=0.0), 1e-6))


def solve_sa(bqm: dimod.BinaryQuadraticModel, seed: int | None = None, **kwargs: Any) -> Sample:
    """
    Solves the QUBO using simulated annealing.

    Args:
        bqm (BinaryQuadraticModel): Problem to be solved
        seed (int): Random seed
        **kwargs: Additional arguments for `SimulatedAnnealingSampler.sample` (num_reads, num_sweeps,...)

    Returns:
        dict: Lowest energy assignment found
    """
    with optional_dependencies("feature-selection"):
        from dwave.samplers import SimulatedAnnealingSampler

    # Swapping a selected feature crosses the max_cols penalty, many long reads find better selections
    kwargs.setdefault("num_reads", 100)
    kwargs.setdefault("num_sweeps", 10_000)
    sampleset = SimulatedAnnealingSampler().sample(bqm, seed=seed, **kwargs)
    return dict(sampleset.first.sample)


def solve_sb(bqm: dimod.BinaryQuadraticModel, seed: int | None = None, **kwargs: Any) -> Sample:
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
    with optional_dependencies("feature-selection"):
        import simulated_bifurcation as sb
        import torch

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
    kwargs.setdefault("agents", 1024)
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
    with optional_dependencies("feature-selection"):
        from qiskit.quantum_info import SparsePauliOp

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
    with optional_dependencies("feature-selection"):
        from qiskit.circuit.library import qaoa_ansatz
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler

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


def solve_exact(bqm: dimod.BinaryQuadraticModel, max_cols: int | None = None, seed: int | None = None) -> Sample:
    """
    Solves the QUBO by evaluating the energy of every candidate selection.

    It is the reference any heuristic or quantum solver should be compared against and it is
    fast for small problems (C(26, 6) is about 230k subsets). Only selections of `max_cols`
    features are evaluated when it is given, every subset otherwise.

    Args:
        bqm (BinaryQuadraticModel): Problem to be solved
        max_cols (int): Number of selected features, `select_features` passes it along
        seed (int): Ignored, the solution is deterministic

    Returns:
        dict: Lowest energy assignment
    """
    variables = list(bqm.variables)
    n = len(variables)
    sizes = [max_cols] if max_cols is not None else list(range(n + 1))
    total = sum(math.comb(n, k) for k in sizes)
    if total > MAX_EXACT_SUBSETS:
        raise ValueError(
            f"The exact solver would evaluate {total:,} selections (limit {MAX_EXACT_SUBSETS:,}). "
            "Use a heuristic solver or reduce the candidates with pool=..."
        )

    index = {var: i for i, var in enumerate(variables)}
    linear = np.array([bqm.linear[var] for var in variables])
    quadratic = np.zeros((n, n))
    for (u, v), bias in bqm.quadratic.items():
        quadratic[index[u], index[v]] = quadratic[index[v], index[u]] = bias

    best_energy, best = np.inf, ()
    for k in sizes:
        combos = itertools.combinations(range(n), k)
        while chunk := list(itertools.islice(combos, 100_000)):
            subsets = np.array(chunk, dtype=int).reshape(len(chunk), k)
            energy = linear[subsets].sum(axis=1)
            for a, b in itertools.combinations(range(k), 2):
                energy += quadratic[subsets[:, a], subsets[:, b]]
            i = int(np.argmin(energy))
            if energy[i] < best_energy:
                best_energy, best = energy[i], tuple(subsets[i])

    return {var: int(i in best) for i, var in enumerate(variables)}


SOLVERS: dict[str, Callable[..., Sample]] = {
    "sa": solve_sa,
    "sb": solve_sb,
    "qaoa": solve_qaoa,
    "exact": solve_exact,
}


def select_features(
    input_ds: Dataset,
    max_cols: int,
    method: str = "sa",
    formulation: str = "mrmr",
    bins: int = 5,
    strategy: str = "quantile",
    pool: int | None = None,
    **kwargs: Any,
) -> list[str]:
    """
    Selects the best `max_cols` features for the target using the requested solver.

    Args:
        input_ds (Dataset): Falcondale Dataset data type
        max_cols (int): Maximum number of features to be selected
        method (str): One of the keys in `SOLVERS`
        formulation (str): QUBO formulation, "mrmr" or "cmi" (see `build_qubo`)
        bins (int): Maximal number of bins per variable
        strategy (str): Discretization strategy, "quantile" or "uniform"
        pool (int): (Optional) Only the `pool` most relevant features (see `relevance`) enter the
            QUBO, so that QAOA or the exact solver can be used on datasets with many features
        **kwargs: Additional solver arguments (seed, num_reads, reps,...)

    Returns:
        list[str]: Selected column list
    """
    method = method.lower()
    if method not in SOLVERS:
        raise ValueError(f"Unknown feature selection method '{method}'. Available ones: {sorted(SOLVERS)}")

    if pool is not None:
        if pool < max_cols:
            raise ValueError(f"The pool ({pool}) must hold at least max_cols ({max_cols}) features.")
        candidates = relevance(input_ds, bins, strategy).index[:pool]
        input_ds = copy.copy(input_ds)  # the caller's selection of features is kept untouched
        input_ds.set_features([col for col in input_ds.columns() if col in candidates])

    if method == "qaoa" and len(input_ds.columns()) > 12:
        warnings.warn(
            f"QAOA with {len(input_ds.columns())} qubits may take a long time to simulate.",
            RuntimeWarning,
            stacklevel=2,
        )
    if method == "exact":
        kwargs.setdefault("max_cols", min(max_cols, len(input_ds.columns())))

    bqm = build_qubo(input_ds, max_cols, formulation=formulation, bins=bins, strategy=strategy)
    selection = SOLVERS[method](bqm, **kwargs)

    return [str(col) for col in input_ds.columns() if selection[col] == 1]
