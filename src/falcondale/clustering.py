"""
Aiming for clustering dataset samples using classical, quantum-inspired and quantum techniques.
Every method is a scikit-learn estimator, so it can be fitted on a train split and assign the
samples of a test split.

* `QuantumClustering` wraps Accelerated Probabilistic Quantum Clustering (APQC, Casaña-Eslava et al.,
  arXiv:1902.05578). It searches a grid of bandwidths (and, optionally, hierarchical energy merges of
  the potential wells found) and keeps the partition with the best training silhouette among the
  admissible ones. Requires the optional `clustering` dependencies: `pip install "falcondale[clustering]"`.
* `SchrodingerClustering` is the original Quantum Clustering by Horn and Gottlieb: samples descend
  the Schrödinger potential of a Parzen wave function and those reaching the same minimum form a
  cluster. It searches a grid of widths with the same selection rules.
* `QuantumKMeans` is k-means (or k-medians) where sample to center distances are estimated by
  quantum interference of amplitude encoded vectors, with the sampling noise of a finite number of shots.

The last two only need the core dependencies.
"""

import gc
import os
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from ._optional import optional_dependencies
from .data import Dataset
from .helpers.interference import interference_distances
from .helpers.potential import descend, membership

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


def _admissible(
    labels: np.ndarray, min_clusters: int, max_clusters: int, min_cluster_size: int, min_cluster_fraction: float
) -> bool:
    """Between min and max clusters, the smallest holding at least max(size, fraction of the samples)."""
    _, counts = np.unique(labels, return_counts=True)
    min_size = max(min_cluster_size, int(np.ceil(min_cluster_fraction * len(labels))))
    return bool(min_clusters <= len(counts) <= max_clusters and np.min(counts) >= min_size)


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    k = len(np.unique(labels))
    return float(silhouette_score(x, labels)) if 1 < k < len(x) else float("nan")


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
        return _admissible(
            labels, self.min_clusters, self.max_clusters, self.min_cluster_size, self.min_cluster_fraction
        )

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
                    score, admissible = _silhouette(x, labels), self._admissible(labels)
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


class SchrodingerClustering(ClusterMixin, BaseEstimator):
    """
    Quantum Clustering by Horn and Gottlieb, following the scikit-learn API.

    The data defines a wave function, a sum of Gaussians of width sigma centred on the samples,
    and the Schrödinger equation it solves defines a potential whose minima are the cluster centres.
    Every sample descends the potential by gradient descent and samples ending closer than
    `merge_distance * sigma` from each other form a cluster.

    Widths are given relative to the spread of the training data (the root mean square of the
    feature standard deviations). Each width on the grid gives a candidate partition and, as with
    `QuantumClustering`, the one with the best training silhouette among the admissible candidates
    is kept: between `min_clusters` and `max_clusters` clusters, the smallest one holding at least
    `max(min_cluster_size, min_cluster_fraction * n_samples)` samples.

    Ref:
    - Horn and Gottlieb, The Method of Quantum Clustering, NIPS 2001
    - Horn and Gottlieb, Physical Review Letters 88, 018702 (2002)

    Attributes:
        labels_ (np.ndarray): Cluster of each training sample
        n_clusters_ (int): Number of clusters
        cluster_centers_ (np.ndarray): Minima of the potential, one per cluster
        sigma_ (float): Selected width, in the units of the data
        silhouette_ (float): Training silhouette of the selected partition
        candidates_ (list[dict]): Every width evaluated, admissible or not
    """

    def __init__(
        self,
        sigmas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        learning_rate: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-4,
        merge_distance: float = 0.5,
        min_clusters: int = 2,
        max_clusters: int = 8,
        min_cluster_size: int = 3,
        min_cluster_fraction: float = 0.03,
    ):
        self.sigmas = sigmas
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.merge_distance = merge_distance
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.min_cluster_fraction = min_cluster_fraction

    def _fit_sigma(self, x: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, int]:
        positions, iterations = descend(x, x, sigma, self.learning_rate, self.max_iter, self.tol)
        if len(x) == 1:
            return np.zeros(1, dtype=int), positions, iterations
        groups = fcluster(linkage(positions, "single"), t=self.merge_distance * sigma, criterion="distance")
        # Number clusters by decreasing size
        _, labels, counts = np.unique(groups, return_inverse=True, return_counts=True)
        labels = np.argsort(np.argsort(-counts, kind="stable"))[labels]
        centers = np.stack([positions[labels == c].mean(axis=0) for c in range(len(counts))])
        return labels, centers, iterations

    def fit(self, X, y: None = None) -> Self:
        """
        Descends the training samples for every width and keeps the best admissible partition.

        Args:
            X (array-like): Training samples
            y: Ignored, present for API consistency

        Returns:
            SchrodingerClustering: The fitted estimator

        Raises:
            RuntimeError: If no width on the grid gives an admissible partition
        """
        x = validate_data(self, X, dtype=np.float64)
        self.scale_ = float(np.sqrt(np.mean(np.var(x, axis=0)))) or 1.0

        best, self.candidates_ = -np.inf, []
        for relative in self.sigmas:
            sigma = float(relative) * self.scale_
            start = time.perf_counter()
            labels, centers, iterations = self._fit_sigma(x, sigma)
            score = _silhouette(x, labels)
            admissible = _admissible(
                labels, self.min_clusters, self.max_clusters, self.min_cluster_size, self.min_cluster_fraction
            )
            self.candidates_.append(
                {
                    "sigma": float(relative),
                    "k": len(centers),
                    "silhouette": score,
                    "admissible": admissible,
                    "steps": iterations,
                    "capped": iterations >= self.max_iter,
                    "seconds": time.perf_counter() - start,
                }
            )
            if admissible and score > best:
                best = score
                self.labels_, self.cluster_centers_, self.sigma_ = labels, centers, sigma
                self.n_clusters_, self.silhouette_ = len(centers), score

        if not np.isfinite(best):
            raise RuntimeError(
                "No admissible partition found on the width grid. Inspect candidates_ "
                "and try other sigmas or relax the admissibility rules."
            )
        self.train_ = x
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Share of the wave function at each sample contributed by the training samples of each cluster.

        It is a membership weight given by the same Gaussians defining the potential, not a calibrated probability.

        Args:
            X (array-like): Samples, in the same representation used for training

        Returns:
            np.ndarray: samples x clusters membership weights
        """
        check_is_fitted(self, "cluster_centers_")
        x = validate_data(self, X, dtype=np.float64, reset=False)
        return membership(x, self.train_, self.sigma_, self.labels_, self.n_clusters_)

    def predict(self, X) -> np.ndarray:
        """
        Descends the samples on the training potential and assigns them to the closest minimum.

        Args:
            X (array-like): Samples, in the same representation used for training

        Returns:
            np.ndarray: Cluster of each sample
        """
        check_is_fitted(self, "cluster_centers_")
        x = validate_data(self, X, dtype=np.float64, reset=False)
        positions, _ = descend(x, self.train_, self.sigma_, self.learning_rate, self.max_iter, self.tol)
        distances = np.sum((positions[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=-1)
        return distances.argmin(axis=1)


def _geometric_median(points: np.ndarray, start: np.ndarray, max_iter: int = 100, eps: float = 1e-8) -> np.ndarray:
    """Weiszfeld iterations with the Vardi and Zhang correction when the estimate hits a sample."""
    median = start.copy()
    for _ in range(max_iter):
        distances = np.linalg.norm(points - median, axis=1)
        at_sample = distances < eps
        weights = 1 / np.maximum(distances, eps)
        weights[at_sample] = 0
        if weights.sum() == 0:
            return median
        update = (weights[:, None] * points).sum(axis=0) / weights.sum()
        if at_sample.any():
            pull = np.linalg.norm((weights[:, None] * (points - median)).sum(axis=0))
            share = min(1.0, at_sample.sum() / pull) if pull > 0 else 1.0
            update = (1 - share) * update + share * median
        if np.linalg.norm(update - median) < eps:
            return update
        median = update
    return median


class QuantumKMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """
    k-means (or k-medians) with distances estimated by quantum interference.

    Each sample and center are amplitude encoded together in a single state of log2(2 d) qubits
    (rounded up), and a Hadamard gate on the most significant qubit makes them interfere. The
    probability of measuring a one is proportional to their squared Euclidean distance, so samples
    are assigned to the center with the smallest estimated distance. Centers are then updated
    classically, with the mean of their samples or, for `center="median"`, their geometric median,
    which is less sensitive to outliers (the quantum k-medians of Belis et al.).

    The circuits are simulated exactly and `shots` measurements are drawn from them, so the
    estimates carry the sampling noise of a real device (`shots=None` gives the exact distances).
    `falcondale.helpers.interference.interference_circuit` builds the circuit of a pair, to run it
    on hardware. Initial centers are chosen with k-means++.

    Ref:
    - Khan et al., K-Means Clustering on Noisy Intermediate Scale Quantum Computers, arXiv:1909.12183
    - Poggiali et al., Quantum Clustering with k-Means: a Hybrid Approach, arXiv:2212.06691
    - Belis et al., Quantum anomaly detection in the latent space of proton collision events at the LHC,
      arXiv:2301.10780

    Attributes:
        cluster_centers_ (np.ndarray): Center of each cluster
        labels_ (np.ndarray): Cluster of each training sample
        n_clusters_ (int): Number of clusters
        inertia_ (float): Sum of the exact squared distances of the samples to their center
        n_iter_ (int): Number of iterations run
    """

    def __init__(
        self,
        n_clusters: int = 2,
        center: str = "mean",
        shots: int | None = 1024,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.center = center
        self.shots = shots
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _distances(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.maximum(interference_distances(x, self.cluster_centers_, self.shots, rng), 0)

    def fit(self, X, y: None = None) -> Self:
        """
        Alternates quantum assignments and classical center updates until the centers settle.

        Args:
            X (array-like): Training samples
            y: Ignored, present for API consistency

        Returns:
            QuantumKMeans: The fitted estimator
        """
        if self.center not in {"mean", "median"}:
            raise ValueError(f"Unknown center '{self.center}', use 'mean' or 'median'.")
        x = validate_data(self, X, dtype=np.float64)
        if not 1 <= self.n_clusters <= len(x):
            raise ValueError(f"n_clusters={self.n_clusters} must be between 1 and the number of samples ({len(x)}).")

        rng = np.random.default_rng(self.random_state)
        seed = int(rng.integers(2**31))
        self.cluster_centers_, _ = kmeans_plusplus(x, self.n_clusters, random_state=seed)
        tol = self.tol * float(np.mean(np.var(x, axis=0)))

        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration
            distances = self._distances(x, rng)
            labels = distances.argmin(axis=1)
            centers = self.cluster_centers_.copy()
            for c in range(self.n_clusters):
                members = x[labels == c]
                if len(members) == 0:
                    # An empty cluster restarts from the sample farthest from its center
                    centers[c] = x[int(np.argmax(distances[np.arange(len(x)), labels]))]
                elif self.center == "mean":
                    centers[c] = members.mean(axis=0)
                else:
                    centers[c] = _geometric_median(members, centers[c])
            shift = float(np.sum((centers - self.cluster_centers_) ** 2))
            self.cluster_centers_ = centers
            if shift <= tol:
                break

        self.labels_ = self._distances(x, rng).argmin(axis=1)
        self.n_clusters_ = self.n_clusters
        self.inertia_ = float(np.sum((x - self.cluster_centers_[self.labels_]) ** 2))
        return self

    def transform(self, X) -> np.ndarray:
        """
        Distances of the samples to every center, estimated by interference.

        The smallest one is a natural anomaly score, samples far from every center being atypical.

        Args:
            X (array-like): Samples

        Returns:
            np.ndarray: samples x clusters distances
        """
        check_is_fitted(self, "cluster_centers_")
        x = validate_data(self, X, dtype=np.float64, reset=False)
        return np.sqrt(self._distances(x, np.random.default_rng(self.random_state)))

    def predict(self, X) -> np.ndarray:
        """
        Assigns samples to the center with the smallest estimated distance.

        Args:
            X (array-like): Samples

        Returns:
            np.ndarray: Cluster of each sample
        """
        return self.transform(X).argmin(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """
        Fuzzy c-means memberships (fuzzifier 2) computed from the estimated distances.

        Args:
            X (array-like): Samples

        Returns:
            np.ndarray: samples x clusters memberships, each row adds up to one
        """
        distances = self.transform(X)
        at_center = distances <= 1e-12
        inverse = 1 / np.maximum(distances, 1e-12) ** 2
        inverse[at_center.any(axis=1)] = at_center[at_center.any(axis=1)]
        return inverse / inverse.sum(axis=1, keepdims=True)


def make_clustering_pipeline(
    components: int | None = 6, estimator: BaseEstimator | None = None, **kwargs: Any
) -> Pipeline:
    """
    Pipeline with median imputation, standard scaling, PCA and a clustering estimator.

    Every step is fitted on the data given to `fit`, so fitting on a train split and
    predicting on a test split does not leak information from the test samples.

    Args:
        components (int): Number of PCA components, None to skip the reduction
        estimator (BaseEstimator): (Optional) Clustering step, `QuantumClustering(**kwargs)` by default
        **kwargs: `QuantumClustering` arguments, when no estimator is given

    Returns:
        Pipeline: Unfitted scikit-learn pipeline
    """
    if estimator is not None and kwargs:
        raise ValueError("Pass the options to the estimator itself when an estimator is given.")
    steps: list = [SimpleImputer(strategy="median"), StandardScaler()]
    if components:
        steps.append(PCA(n_components=components, svd_solver="full"))
    return make_pipeline(*steps, estimator if estimator is not None else QuantumClustering(**kwargs))


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


def _fit_and_score(estimator: BaseEstimator, input_ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    x = input_ds.get_features().to_numpy()
    proba = estimator.fit(x).predict_proba(x)  # ty: ignore[unresolved-attribute]
    return estimator.labels_, proba.max(axis=1)  # ty: ignore[unresolved-attribute]


def schrodinger_clustering(input_ds: Dataset, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Horn and Gottlieb Quantum Clustering of the selected features of a dataset.

    Args:
        input_ds (Dataset): Preprocessed dataset
        **kwargs: `SchrodingerClustering` arguments

    Returns:
        tuple: Label per sample and its membership weight
    """
    return _fit_and_score(SchrodingerClustering(**kwargs), input_ds)


def quantum_kmeans(input_ds: Dataset, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantum k-means of the selected features of a dataset.

    Args:
        input_ds (Dataset): Preprocessed dataset
        **kwargs: `QuantumKMeans` arguments, such as `n_clusters`, `center` or `shots`

    Returns:
        tuple: Label per sample and its fuzzy membership
    """
    return _fit_and_score(QuantumKMeans(**kwargs), input_ds)


CLUSTERING: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "pqc": prob_q_clustering,
    "qc": schrodinger_clustering,
    "qkmeans": quantum_kmeans,
}
