"""
Aiming for clustering dataset samples using classical, quantum-inspired and quantum techniques.

`QuantumClustering` wraps Accelerated Probabilistic Quantum Clustering (APQC, Casaña-Eslava et al.,
arXiv:1902.05578) as a scikit-learn estimator. It searches a grid of bandwidths (and, optionally,
hierarchical energy merges of the potential wells found) and keeps the partition with the best
training silhouette among the admissible ones. Unseen samples are assigned with their membership
probabilities, so it can be fitted on a train split and evaluated on a test split.

Requires the optional `clustering` dependencies: `pip install "falcondale[clustering]"`.
"""

import gc
import os
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from ._optional import optional_dependencies
from .data import Dataset

Bandwidth = tuple[str, float]

# Adaptive kNN fractions and explicit scalar bandwidths (in APQC normalised units)
DEFAULT_BANDWIDTHS: tuple[Bandwidth, ...] = (
    ("knn", 0.05),
    ("knn", 0.10),
    ("knn", 0.20),
    ("knn", 0.30),
    ("scalar", 0.05),
    ("scalar", 0.10),
    ("scalar", 0.20),
    ("scalar", 0.40),
)


def _parse_bandwidth(bandwidth: Bandwidth | str) -> Bandwidth:
    """Accepts ("knn", 0.2) as well as "knn:0.2"."""
    if isinstance(bandwidth, str):
        kind, _, value = bandwidth.partition(":")
        bandwidth = (kind, float(value))
    kind, value = bandwidth
    if kind not in {"knn", "scalar"}:
        raise ValueError(f"Unknown bandwidth kind '{kind}', use 'knn' or 'scalar'.")
    return kind, float(value)


def _load_apqc():
    # TensorFlow reads these when imported. oneDNN is disabled for reproducible results.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    with optional_dependencies("clustering"):
        import tensorflow as tf
        from apqc.pqc import PQC
        from apqc.utils import reduce_sum

    # Independent fits retrace the library's tf.functions, those warnings are expected
    tf.get_logger().setLevel("ERROR")
    return tf, PQC, reduce_sum


class QuantumClustering(ClusterMixin, BaseEstimator):
    """
    Probabilistic Quantum Clustering estimator following the scikit-learn API.

    For each bandwidth APQC descends every sample to a well of the quantum potential. The raw
    wells and, when `merge=True`, their hierarchical energy merges are candidate partitions.
    Candidates are ranked by training silhouette and only admissible ones can be selected:
    between `min_clusters` and `max_clusters` clusters, the smallest one holding at least
    `max(min_cluster_size, min_cluster_fraction * n_samples)` samples.

    Membership probabilities have one column per cluster of the selected partition, the
    potential wells or their merges. Some of them may not win any training sample, so
    `n_clusters_`, the number of clusters training samples are assigned to, can be smaller.

    Attributes:
        labels_ (np.ndarray): Cluster of each training sample
        n_clusters_ (int): Number of clusters training samples are assigned to
        source_labels_ (np.ndarray): Well (or merged well) each training sample descended to,
            defining the clusters and the columns of `predict_proba`
        bandwidth_ (tuple[str, float]): Selected bandwidth
        stage_ (str): Selected stage, "base" for the raw wells or "energy_<i>" for a merge level
        silhouette_ (float): Training silhouette of the selected partition
        candidates_ (list[dict]): Every candidate evaluated, admissible or not
        model_ (apqc.pqc.PQC): Fitted APQC model of the selected bandwidth
    """

    def __init__(
        self,
        bandwidths: Sequence[Bandwidth | str] = DEFAULT_BANDWIDTHS,
        merge: bool = True,
        steps: int = 1500,
        patience: int = 8,
        learning_rate: float = 0.01,
        min_clusters: int = 2,
        max_clusters: int = 8,
        min_cluster_size: int = 3,
        min_cluster_fraction: float = 0.03,
        max_merge_wells: int = 12,
        random_state: int | None = None,
    ):
        self.bandwidths = bandwidths
        self.merge = merge
        self.steps = steps
        self.patience = patience
        self.learning_rate = learning_rate
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.min_cluster_fraction = min_cluster_fraction
        self.max_merge_wells = max_merge_wells
        self.random_state = random_state

    def _admissible(self, labels: np.ndarray) -> bool:
        _, counts = np.unique(labels, return_counts=True)
        min_size = max(self.min_cluster_size, int(np.ceil(self.min_cluster_fraction * len(labels))))
        return bool(self.min_clusters <= len(counts) <= self.max_clusters and np.min(counts) >= min_size)

    @staticmethod
    def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
        k = len(np.unique(labels))
        return float(silhouette_score(x, labels)) if 1 < k < len(x) else float("nan")

    def _fit_bandwidth(self, x: np.ndarray, bandwidth: Bandwidth):
        """Fits APQC for one bandwidth and returns the model with its candidate partitions."""
        tf, PQC, _ = self._apqc
        if self.random_state is not None:
            tf.keras.utils.set_random_seed(self.random_state)

        kind, value = bandwidth
        model = PQC(
            x,
            batch=len(x),
            force_cpu=True,
            max_num_clusters=self.max_merge_wells,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        model.set_sigmas(**({"knn_ratio": value} if kind == "knn" else {"sigma_value": value}))
        model.cluster_allocation_by_sgd(steps=self.steps, patience=self.patience)
        model.cluster_allocation_by_probability()

        candidates = [("base", model.sgd_labels.copy())]
        if self.merge and 2 <= model.k_num <= self.max_merge_wells:
            model.hierarchical_energy_merge()
            result = model.energy_merge_results.get(model.sigmas_id)
            if result is not None:
                merged = result["merged_sgd_labels"]
                # Column 0 repeats the base wells, later columns merge progressively
                candidates += [(f"energy_{i}", merged[:, i].astype(int)) for i in range(1, merged.shape[1])]
        return model, candidates

    def _membership(self, model, x: np.ndarray, source_labels: np.ndarray) -> np.ndarray:
        """Membership probability of each sample (rows) to each cluster (columns) of a fitted model."""
        tf, _, reduce_sum = self._apqc
        n_clusters = len(np.unique(source_labels))
        if len(x) == 0:
            return np.empty((0, n_clusters))

        # The library works in its internally normalised coordinates
        queries = np.asarray(x / float(model.scale.numpy()), dtype=np.float32)
        probas = []
        with warnings.catch_warnings(), tf.device(model.device):
            warnings.simplefilter("ignore")
            for batch in tf.data.Dataset.from_tensor_slices(queries).batch(model.batch):
                joint, _ = reduce_sum(data=model.qc_wave_function_per_sample(batch).numpy(), vector=source_labels)
                probas.append(joint / np.sum(joint, axis=0, keepdims=True))

        proba = np.concatenate(probas, axis=1).T
        if not np.isfinite(proba).all() or np.any(proba < 0) or np.any(proba > 1 + 1e-5):
            raise FloatingPointError("Invalid APQC membership probabilities")
        return proba

    def fit(self, X, y: None = None) -> Self:
        """
        Searches the bandwidth grid and keeps the best admissible partition.

        Args:
            X (array-like): Training samples
            y: Ignored, present for API consistency

        Returns:
            QuantumClustering: The fitted estimator

        Raises:
            RuntimeError: If no candidate on the grid is admissible
        """
        x = validate_data(self, X, dtype=np.float32)
        self._apqc = _load_apqc()

        best, self.candidates_ = -np.inf, []
        for bandwidth in map(_parse_bandwidth, self.bandwidths):
            name = f"{bandwidth[0]}:{bandwidth[1]}"
            start = time.perf_counter()
            try:
                model, options = self._fit_bandwidth(x, bandwidth)
                steps = int(model.step.numpy())
                for stage, source_labels in options:
                    labels = self._membership(model, x, source_labels).argmax(axis=1)
                    k = len(np.unique(labels))
                    score, admissible = self._silhouette(x, labels), self._admissible(labels)
                    self.candidates_.append(
                        {
                            "bandwidth": name,
                            "stage": stage,
                            "k": k,
                            "silhouette": score,
                            "admissible": admissible,
                            "steps": steps,
                            "capped": steps >= self.steps,
                            "seconds": time.perf_counter() - start,
                            "error": "",
                        }
                    )
                    if admissible and score > best:
                        best = score
                        self.model_, self.source_labels_, self.labels_ = model, source_labels.copy(), labels
                        self.bandwidth_, self.stage_, self.n_clusters_, self.silhouette_ = bandwidth, stage, k, score
                        self.steps_, self.capped_ = steps, steps >= self.steps
            except Exception as exc:
                self.candidates_.append({"bandwidth": name, "error": repr(exc)})
            gc.collect()

        if not np.isfinite(best):
            raise RuntimeError(
                "No admissible partition found on the bandwidth grid. Inspect candidates_ "
                "and try other bandwidths or relax the admissibility rules."
            )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Membership probabilities of the samples to the clusters found during training.

        These are the algorithmic probabilities of APQC, not calibrated ones.

        Args:
            X (array-like): Samples, in the same representation used for training

        Returns:
            np.ndarray: samples x clusters probabilities, one column per cluster in `source_labels_`
        """
        check_is_fitted(self, "model_")
        x = validate_data(self, X, dtype=np.float32, reset=False)
        return self._membership(self.model_, x, self.source_labels_)

    def predict(self, X) -> np.ndarray:
        """
        Assigns samples to the clusters found during training.

        Args:
            X (array-like): Samples, in the same representation used for training

        Returns:
            np.ndarray: Cluster of each sample
        """
        return self.predict_proba(X).argmax(axis=1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_apqc", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "model_" in state:
            self._apqc = _load_apqc()


def make_clustering_pipeline(components: int | None = 6, **kwargs: Any) -> Pipeline:
    """
    Pipeline with median imputation, standard scaling, PCA and `QuantumClustering`.

    Every step is fitted on the data given to `fit`, so fitting on a train split and
    predicting on a test split does not leak information from the test samples.

    Args:
        components (int): Number of PCA components, None to skip the reduction
        **kwargs: `QuantumClustering` arguments

    Returns:
        Pipeline: Unfitted scikit-learn pipeline
    """
    steps: list = [SimpleImputer(strategy="median"), StandardScaler()]
    if components:
        steps.append(PCA(n_components=components, svd_solver="full"))
    return make_pipeline(*steps, QuantumClustering(**kwargs))


def prob_q_clustering(input_ds: Dataset, sigma: float | None = None, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilistic Quantum Clustering of the selected features of a dataset.

    Args:
        input_ds (Dataset): Preprocessed dataset
        sigma (float): (Optional) Single scalar bandwidth to use instead of the default grid
        **kwargs: `QuantumClustering` arguments

    Returns:
        tuple: Label per sample and the probability of that label
    """
    if sigma is not None:
        kwargs["bandwidths"] = [("scalar", sigma)]
    estimator = QuantumClustering(**kwargs)
    proba = estimator.fit(input_ds.get_features().to_numpy()).predict_proba(input_ds.get_features().to_numpy())
    return estimator.labels_, proba.max(axis=1)


CLUSTERING: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "pqc": prob_q_clustering,
}
