import numpy as np
import pytest

from falcondale.data import Dataset
from falcondale.exceptions import TooManyQubitsNeeded
from falcondale.feature_selection import MAX_QAOA_QUBITS, build_qubo, ising_operator, select_features, solve_qaoa


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
