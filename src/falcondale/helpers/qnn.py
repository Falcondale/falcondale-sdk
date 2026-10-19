"""
Module implementing a generic Quantum Neural Network classifier
"""

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .._optional import optional_dependencies

with optional_dependencies("qml"):
    import pennylane as qml
    import torch


class QNNClassifier(ClassifierMixin, BaseEstimator):
    """
    PennyLane based Quantum Neural Network following the scikit-learn estimator API.

    Features are amplitude encoded, so n features need only ceil(log2(n)) qubits. A
    strongly entangling ansatz is trained per class (one-vs-rest scores) and the class
    scores are turned into probabilities with a softmax.
    """

    def __init__(
        self,
        layers: int = 3,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.05,
        early_stop: float = 0.98,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.random_state = random_state
        self.verbose = verbose

    def _build_qnode(self):
        num_qubits = self.num_qubits_

        def circuit(weights, features):
            qml.AmplitudeEmbedding(features, wires=range(num_qubits), pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return qml.expval(qml.PauliZ(0))

        self._qnode = qml.QNode(circuit, qml.device("default.qubit", wires=num_qubits), interface="torch")

    def _logits(self, features: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_qnode", None) is None:
            self._build_qnode()
        assert self._qnode is not None
        scores = [self._qnode(self.weights_[c], features) for c in range(len(self.classes_))]
        return torch.stack(scores, dim=-1) + self.bias_

    def fit(self, X, y):
        """
        Trains the classifier.

        Args:
            X (array-like): Training features
            y (array-like): Training labels

        Returns:
            QNNClassifier: The fitted estimator
        """
        features = np.asarray(X, dtype=float)
        self.classes_, labels = np.unique(np.asarray(y), return_inverse=True)
        self.n_features_in_ = features.shape[1]
        self.num_qubits_ = max(1, math.ceil(math.log2(self.n_features_in_)))
        self._build_qnode()

        generator = torch.Generator().manual_seed(self.random_state) if self.random_state is not None else None
        shape = (len(self.classes_), self.layers, self.num_qubits_, 3)
        self.weights_ = torch.nn.Parameter(0.1 * torch.randn(shape, generator=generator, dtype=torch.float64))
        self.bias_ = torch.nn.Parameter(torch.zeros(len(self.classes_), dtype=torch.float64))
        optimizer = torch.optim.Adam([self.weights_, self.bias_], lr=self.learning_rate)

        x_t = torch.tensor(features, dtype=torch.float64)
        y_t = torch.tensor(labels, dtype=torch.long)
        self.history_: list[tuple[float, float]] = []
        for epoch in range(self.epochs):
            for batch in torch.randperm(len(x_t), generator=generator).split(self.batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self._logits(x_t[batch]), y_t[batch])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                logits = self._logits(x_t)
                cost = torch.nn.functional.cross_entropy(logits, y_t).item()
                accuracy = (logits.argmax(dim=-1) == y_t).double().mean().item()
            self.history_.append((cost, accuracy))
            if self.verbose:
                print(f"Epoch: {epoch + 1:5d} | Cost: {cost:0.7f} | Acc train: {accuracy:0.7f}")

            if accuracy > self.early_stop:
                break

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Probability of each class for the given samples.

        Args:
            X (array-like): Features

        Returns:
            np.ndarray: samples x classes probabilities
        """
        with torch.no_grad():
            logits = self._logits(torch.tensor(np.asarray(X, dtype=float), dtype=torch.float64))
            return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X) -> np.ndarray:
        """
        Predicted class for the given samples.

        Args:
            X (array-like): Features

        Returns:
            np.ndarray: Predicted labels
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_qnode"] = None
        return state
