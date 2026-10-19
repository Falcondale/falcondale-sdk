import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falcondale import feature_selection
from falcondale.data import Dataset
from falcondale.exceptions import TooManyQubitsNeeded
from falcondale.feature_selection import (
    MAX_QAOA_QUBITS,
    build_qubo,
    ising_operator,
    relevance,
    select_features,
    solve_exact,
    solve_qaoa,
    solve_sa,
    solve_sb,
)
from falcondale.helpers.mutual_information import (
    conditional_mutual_information,
    discretize,
    joint_prob,
    mutual_information,
    prob,
)

FIXTURE = Path(__file__).parent / "data" / "risk_factors.csv"
CAUSAL = {"age", "smoker", "severity", "hospital_days", "biomarker_a", "biomarker_b"}


@pytest.fixture
def dataset(breast_cancer) -> Dataset:
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess()
    return ds


def test_qubo_has_one_variable_per_feature(dataset):
    bqm = build_qubo(dataset)
    assert set(bqm.variables) == set(dataset.columns())
    assert bqm.num_interactions == len(dataset.columns()) * (len(dataset.columns()) - 1) // 2


def test_ising_operator_reproduces_qubo_energies(dataset):
    dataset.set_features(dataset.columns()[:4])
    bqm = build_qubo(dataset, max_cols=2)
    operator, variables = ising_operator(bqm)
    _, _, offset = bqm.to_ising()

    # Basis state k has qubit i in state (k >> i) & 1, i.e. feature i selected
    for k, energy in enumerate(np.real(operator.to_matrix().diagonal())):
        sample = {var: (k >> i) & 1 for i, var in enumerate(variables)}
        assert energy + offset == pytest.approx(bqm.energy(sample), abs=1e-9)


@pytest.mark.parametrize("method", ["sa", "sb"])
@pytest.mark.parametrize("max_cols", [2, 3])
def test_selects_requested_number_of_features(dataset, method, max_cols):
    selected = select_features(dataset, max_cols=max_cols, method=method, seed=0)
    assert len(selected) == max_cols
    assert set(selected) <= set(dataset.columns())


@pytest.mark.slow
def test_qaoa(dataset):
    dataset.set_features(dataset.columns()[:4])
    assert len(select_features(dataset, max_cols=2, method="qaoa", seed=0)) == 2


def test_qaoa_qubit_limit(dataset):
    bqm = build_qubo(dataset)
    for i in range(MAX_QAOA_QUBITS):
        bqm.add_variable(f"extra{i}", 0.1)
    with pytest.raises(TooManyQubitsNeeded):
        solve_qaoa(bqm)


def test_unknown_method(dataset):
    with pytest.raises(ValueError, match="Available"):
        select_features(dataset, max_cols=2, method="qa")


def test_sa_is_reproducible(dataset):
    first = select_features(dataset, max_cols=3, method="sa", seed=7)
    assert first == select_features(dataset, max_cols=3, method="sa", seed=7)
    assert np.all([isinstance(col, str) for col in first])


def test_unknown_formulation(dataset):
    with pytest.raises(ValueError, match="mrmr"):
        build_qubo(dataset, formulation="jmi")


def test_mrmr_terms(dataset):
    """Relevance on the diagonal and pairwise redundancy averaged over the max_cols - 1 pairs of a feature."""
    bqm = build_qubo(dataset, formulation="mrmr")
    features, target = dataset.as_arrays()
    codes = [discretize(column) for column in features.T]
    columns = dataset.columns()

    linear = [-mutual_information(joint_prob(discretize(target), code), 1) for code in codes]
    quadratic = [
        mutual_information(joint_prob(codes[i], codes[j]), 1) / (len(columns) - 1)
        for i, j in itertools.combinations(range(len(columns)), 2)
    ]
    scale = max(map(abs, linear + quadratic))  # normalize() scales biases into (-1, 1)
    np.testing.assert_allclose([bqm.linear[col] for col in columns], np.divide(linear, scale))
    np.testing.assert_allclose(
        [bqm.quadratic[columns[i], columns[j]] for i, j in itertools.combinations(range(len(columns)), 2)],
        np.divide(quadratic, scale),
    )


def test_cmi_reproduces_previous_release(dataset):
    """formulation="cmi" with ten equal width bins is the QUBO built by falcondale 0.3.0."""
    bqm = build_qubo(dataset, formulation="cmi", strategy="uniform", bins=10)
    features, target = dataset.as_arrays()
    columns = dataset.columns()

    linear = [-mutual_information(prob(np.column_stack([target, column])), 1) for column in features.T]
    quadratic = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        cmi_ij = conditional_mutual_information(prob(np.column_stack([target, features[:, i], features[:, j]])), 1, 2)
        cmi_ji = conditional_mutual_information(prob(np.column_stack([target, features[:, j], features[:, i]])), 1, 2)
        quadratic.append(-(cmi_ij + cmi_ji))
    scale = max(map(abs, linear + quadratic))
    np.testing.assert_allclose([bqm.linear[col] for col in columns], np.divide(linear, scale))
    np.testing.assert_allclose(
        [bqm.quadratic[columns[i], columns[j]] for i, j in itertools.combinations(range(len(columns)), 2)],
        np.divide(quadratic, scale),
    )


@pytest.mark.parametrize("formulation", ["mrmr", "cmi"])
@pytest.mark.parametrize("max_cols", [1, 2, 4, 6])
def test_penalty_keeps_the_ground_state_feasible(dataset, formulation, max_cols):
    # Without max_cols the exact solver evaluates every subset, including the unfeasible ones
    bqm = build_qubo(dataset, max_cols, formulation=formulation)
    assert sum(solve_exact(bqm).values()) == max_cols


@pytest.mark.parametrize("formulation", ["mrmr", "cmi"])
@pytest.mark.parametrize("solver", [solve_sa, solve_sb])
def test_heuristics_reach_the_exact_energy(dataset, formulation, solver):
    bqm = build_qubo(dataset, 3, formulation=formulation)
    assert bqm.energy(solver(bqm, seed=0)) == pytest.approx(bqm.energy(solve_exact(bqm, 3)))


def test_exact_selects_max_cols(dataset):
    selected = select_features(dataset, max_cols=3, method="exact")
    assert len(selected) == 3
    assert selected == select_features(dataset, max_cols=3, method="exact", seed=1)


def test_exact_limit(dataset, monkeypatch):
    monkeypatch.setattr(feature_selection, "MAX_EXACT_SUBSETS", 10)
    with pytest.raises(ValueError, match="pool"):
        select_features(dataset, max_cols=3, method="exact")


def test_relevance(dataset):
    scores = relevance(dataset)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == set(dataset.columns())
    assert scores.is_monotonic_decreasing
    assert (scores >= 0).all()


def test_pool_prescreens_the_most_relevant_features(dataset):
    columns = dataset.columns()
    selected = select_features(dataset, max_cols=2, method="exact", pool=4)
    assert set(selected) <= set(relevance(dataset).index[:4])
    assert dataset.columns() == columns  # the dataset selection is left untouched


def test_pool_smaller_than_max_cols(dataset):
    with pytest.raises(ValueError, match="pool"):
        select_features(dataset, max_cols=3, pool=2)


@pytest.mark.slow
def test_qaoa_with_pool(dataset):
    assert len(select_features(dataset, max_cols=2, method="qaoa", pool=4, seed=0)) == 2


@pytest.fixture(scope="module")
def risk_factors() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    return df[df["split"] == "train"].drop(columns="split")


def _risk_dataset(df: pd.DataFrame, target: str) -> Dataset:
    other = "progress" if target == "diagnosis" else "diagnosis"
    ds = Dataset(df.drop(columns=[other]), target=target)
    ds.preprocess()
    return ds


@pytest.mark.parametrize("target", ["diagnosis", "progress"])
def test_risk_factors_regression(risk_factors, target):
    """
    Synthetic cohort with six causal factors, their proxies and noise (see tests/data). The settings
    of falcondale 0.3.0 rank the high cardinality noise column `region` among the relevant ones.
    """
    ds = _risk_dataset(risk_factors, target)
    selected = select_features(ds, max_cols=6, method="exact")
    assert "region" not in selected
    assert len(CAUSAL.intersection(selected)) >= 4

    legacy = select_features(ds, max_cols=6, method="exact", formulation="cmi", strategy="uniform", bins=10)
    assert "region" in legacy


@pytest.mark.parametrize("method", ["sa", "sb"])
def test_risk_factors_heuristics(risk_factors, method):
    selected = select_features(_risk_dataset(risk_factors, "diagnosis"), max_cols=6, method=method, seed=0)
    assert "region" not in selected
    assert len(CAUSAL.intersection(selected)) >= 4
