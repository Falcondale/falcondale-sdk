import pickle

import numpy as np
import pytest

from falcondale.classifiers import MAX_QSVC_QUBITS, build_qsvc, compute_metrics, train_and_evaluate
from falcondale.data import Dataset
from falcondale.exceptions import TooManyQubitsNeeded


@pytest.fixture
def dataset(breast_cancer) -> Dataset:
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess(reduced_dimension=4)
    return ds


def test_metrics_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced accuracy"] == 0.75
    assert metrics["auc"] == 1.0


def test_unknown_model(dataset):
    with pytest.raises(ValueError, match="Available"):
        train_and_evaluate("rbm", dataset)


def test_qsvc_qubit_limit():
    with pytest.raises(TooManyQubitsNeeded):
        build_qsvc(MAX_QSVC_QUBITS + 1)


def test_qsvc_unknown_backend():
    with pytest.raises(ValueError):
        build_qsvc(2, backend="cirq")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        ("svc", {}),
        ("qsvc", {"backend": "qiskit"}),
        ("qsvc", {"backend": "pennylane"}),
        ("qnn", {"epochs": 3, "random_state": 0}),
    ],
)
def test_models_train_and_pickle(dataset, model, kwargs):
    estimator, report, metrics = train_and_evaluate(model, dataset, test_size=0.25, **kwargs)
    assert "accuracy" in report
    assert 0 <= metrics["auc"] <= 1

    features = dataset.get_features().to_numpy()[:5]
    probabilities = estimator.predict_proba(features)
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)

    # Every model must survive save and load (GitHub issue #25)
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(restored.predict(features), estimator.predict(features))
