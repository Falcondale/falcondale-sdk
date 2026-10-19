"""
Module implementing the classification methods.

Every model is a scikit-learn compatible estimator, so they can also be used outside
Falcondale (pipelines, cross validation, grid search,...):

* **svc**: Classical Support Vector Classifier, a useful baseline
* **qsvc**: Quantum Support Vector Classifier (Qiskit or PennyLane backends)
* **qnn**: Quantum Neural Network
"""

from collections.abc import Callable
from typing import Protocol, Self

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from .data import Dataset
from .exceptions import TooManyQubitsNeeded
from .helpers.kernels import AngleKernel
from .helpers.qnn import QNNClassifier

# Statevector kernels keep 2^n amplitudes per sample in memory
MAX_QSVC_QUBITS = 16


def _with_probabilities(svc: SVC) -> CalibratedClassifierCV:
    """SVCs only provide decision values, probabilities are obtained by Platt scaling."""
    return CalibratedClassifierCV(svc, ensemble=False)


def build_svc(num_features: int, **kwargs) -> CalibratedClassifierCV:
    """Classical SVC with an RBF kernel."""
    return _with_probabilities(SVC(**kwargs))


def build_qsvc(num_features: int, backend: str = "qiskit", verbose: bool = False, **kwargs) -> CalibratedClassifierCV:
    """
    Quantum Support Vector Classifier, one qubit per feature.

    Kernels are computed from statevectors simulated once per sample, which is much lighter
    than running a circuit for every pair of samples.

    Args:
        num_features (int): Number of features (qubits) to encode
        backend (str): "qiskit" (ZZ feature map) or "pennylane" (angle embedding)
        verbose (bool): Verbose SVC training
        **kwargs: Additional SVC arguments (C, class_weight,...)
    """
    if num_features > MAX_QSVC_QUBITS:
        raise TooManyQubitsNeeded(num_features, MAX_QSVC_QUBITS)

    backend = backend.lower()
    if backend == "qiskit":
        kernel = FidelityStatevectorKernel(feature_map=zz_feature_map(feature_dimension=num_features, reps=2))
        kwargs.setdefault("C", 100)
        return _with_probabilities(QSVC(quantum_kernel=kernel, verbose=verbose, **kwargs))
    if backend == "pennylane":
        kwargs.setdefault("class_weight", "balanced")
        return _with_probabilities(SVC(kernel=AngleKernel(), verbose=verbose, **kwargs))

    raise ValueError(f"Unknown QSVC backend '{backend}'. Available ones: ['pennylane', 'qiskit']")


def build_qnn(num_features: int, **kwargs) -> QNNClassifier:
    """Quantum Neural Network, see `QNNClassifier` for the available arguments."""
    return QNNClassifier(**kwargs)


class Classifier(Protocol):
    """Scikit-learn like classifier"""

    def fit(self, X, y) -> Self: ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...


MODELS: dict[str, Callable[..., Classifier]] = {
    "svc": build_svc,
    "qsvc": build_qsvc,
    "qnn": build_qnn,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """
    Classification metrics. Binary problems use the second class as the positive one,
    multiclass problems use macro averages.

    Args:
        y_true (np.ndarray): Actual labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray): Predicted probabilities (samples x classes)

    Returns:
        dict[str, float]: Metric name and value
    """
    labels = np.unique(y_true)
    binary = len(labels) == 2
    average = "binary" if binary else "macro"
    pos_label = labels[-1] if binary else 1

    metrics = {
        "recall": recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba[:, -1]) if binary else roc_auc_score(y_true, y_proba, multi_class="ovr"),
    }
    if binary:
        metrics["specificity"] = recall_score(y_true, y_pred, pos_label=labels[0], zero_division=0)

    return {name: float(value) for name, value in metrics.items()}


def train_and_evaluate(model: str, dataset: Dataset, test_size: float = 0.3, **kwargs):
    """
    Builds the requested model, trains it on a train split and evaluates it on the test split.

    Args:
        model (str): One of the keys in `MODELS`
        dataset (Dataset): Preprocessed dataset
        test_size (float): Ratio of samples kept for testing
        **kwargs: Model specific arguments

    Returns:
        tuple: fitted estimator, classification report and metrics dictionary
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available ones: {sorted(MODELS)}")

    x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=test_size)
    estimator = MODELS[model](x_train.shape[1], **kwargs)

    estimator.fit(x_train.to_numpy(), y_train)

    y_pred = estimator.predict(x_test.to_numpy())
    y_proba = estimator.predict_proba(x_test.to_numpy())

    return estimator, classification_report(y_test, y_pred), compute_metrics(y_test, y_pred, y_proba)
