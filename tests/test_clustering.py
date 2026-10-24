import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from falcondale import Project
from falcondale.clustering import (
    APQC_SUPPORTED,
    QuantumClustering,
    QuantumKMeans,
    SchrodingerClustering,
    _parse_bandwidth,
    make_clustering_pipeline,
)
from falcondale.helpers.interference import interference_circuit, interference_distances, interference_probability
from falcondale.helpers.potential import potential, potential_gradient

FIXTURE = Path(__file__).parent / "data" / "four_profiles.csv"

needs_apqc = pytest.mark.skipif(
    importlib.util.find_spec("apqc") is None,
    reason="requires falcondale[clustering]" if APQC_SUPPORTED else "APQC does not support this Python version",
)


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(910)
    x = np.r_[rng.normal(-2, 0.3, (30, 4)), rng.normal(2, 0.3, (30, 4))]
    return x, np.repeat([0, 1], 30)


@pytest.fixture(scope="module")
def fitted(blobs) -> QuantumClustering:
    x, _ = blobs
    return QuantumClustering(bandwidths=["scalar:0.2"], steps=200, random_state=0).fit(x[::2])


@pytest.fixture(scope="module")
def four_profiles():
    """Train/test representation of the synthetic four profile cohort (see tests/data)."""
    df = pd.read_csv(FIXTURE)
    features = [col for col in df.columns if col.startswith("f")]
    train = (df["split"] == "train").to_numpy()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=6, svd_solver="full"))
    train_x = pipe.fit_transform(df.loc[train, features])
    test_x = pipe.transform(df.loc[~train, features])
    return train_x, test_x, df["profile"].to_numpy()[train], df["profile"].to_numpy()[~train]


@pytest.mark.parametrize(
    ("bandwidth", "expected"),
    [("knn:0.2", ("knn", 0.2)), (("scalar", 1), ("scalar", 1.0))],
)
def test_parse_bandwidth(bandwidth, expected):
    assert _parse_bandwidth(bandwidth) == expected


def test_unknown_bandwidth_kind():
    with pytest.raises(ValueError, match="knn"):
        _parse_bandwidth("gaussian:0.2")


@pytest.mark.parametrize(
    ("labels", "admissible"),
    [
        (np.repeat([0, 1], 50), True),
        (np.zeros(100), False),  # a single cluster
        (np.repeat(np.arange(10), 10), False),  # more than 8 clusters
        (np.r_[np.zeros(98), np.ones(2)], False),  # the smallest cluster is below 3 samples
        (np.r_[np.zeros(195), np.ones(5)], False),  # ... or below 3% of the samples
    ],
)
def test_admissibility_rules(labels, admissible):
    assert QuantumClustering()._admissible(labels) is admissible


def test_pipeline_structure():
    pipe = make_clustering_pipeline(components=3, merge=False)
    assert [name for name, _ in pipe.steps] == ["simpleimputer", "standardscaler", "pca", "quantumclustering"]
    assert pipe[-1].merge is False
    assert "pca" not in dict(make_clustering_pipeline(components=None).steps)


@needs_apqc
@pytest.mark.slow
def test_fit_predict(fitted, blobs):
    x, truth = blobs
    assert fitted.n_clusters_ == 2
    assert adjusted_rand_score(truth[::2], fitted.labels_) == pytest.approx(1.0)
    assert adjusted_rand_score(truth[1::2], fitted.predict(x[1::2])) == pytest.approx(1.0)
    assert {"bandwidth", "stage", "k", "silhouette", "admissible", "steps", "capped"} <= set(fitted.candidates_[0])

    # One column per well (or merged well), some of them may not win any sample
    proba = fitted.predict_proba(x[1::2])
    assert proba.shape == (30, len(np.unique(fitted.source_labels_)))
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


@needs_apqc
@pytest.mark.slow
def test_probabilities_match_apqc(fitted, blobs):
    """predict_proba builds the whole matrix, its winners must be the ones APQC reports."""
    x, _ = blobs
    queries = np.asarray(x / float(fitted.model_.scale.numpy()), dtype=np.float32)
    winner_proba, winner, _ = fitted.model_.cluster_probability_per_sample_batched(queries, fitted.source_labels_)

    proba = fitted.predict_proba(x)
    np.testing.assert_array_equal(proba.argmax(axis=1), winner)
    # APQC computes the wave function in float32
    np.testing.assert_allclose(proba.max(axis=1), np.ravel(winner_proba), rtol=1e-5)


@needs_apqc
@pytest.mark.slow
def test_pickle(fitted, blobs):
    x, _ = blobs
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(x), fitted.predict(x))


@needs_apqc
@pytest.mark.slow
def test_no_admissible_partition(blobs):
    x, _ = blobs
    estimator = QuantumClustering(bandwidths=["scalar:0.2"], steps=200, min_clusters=3, random_state=0)
    with pytest.raises(RuntimeError, match="No admissible"):
        estimator.fit(x)
    assert estimator.candidates_


@needs_apqc
@pytest.mark.slow
def test_repeated_independent_fits(four_profiles):
    """Consecutive independent fits in one process used to fail (GitHub issue #26), now they agree."""
    train_x, test_x, _, _ = four_profiles
    results = []
    for _ in range(2):
        estimator = QuantumClustering(bandwidths=["knn:0.2", "scalar:0.4"], random_state=101).fit(train_x)
        results.append((estimator.candidates_, estimator.predict_proba(test_x)))

    # assert_array_equal treats the NaN silhouettes of single cluster candidates as equal
    (first, _), (second, _) = results
    np.testing.assert_array_equal([c["silhouette"] for c in first], [c["silhouette"] for c in second])
    np.testing.assert_array_equal(results[0][1], results[1][1])


@needs_apqc
@pytest.mark.slow
def test_four_profiles_regression(four_profiles):
    """
    Full default grid on the synthetic cohort. The expected values were obtained with the
    reference protocol implementation, which this estimator reproduces candidate by candidate.
    """
    train_x, test_x, train_truth, test_truth = four_profiles
    estimator = QuantumClustering(random_state=101).fit(train_x)

    assert (estimator.bandwidth_, estimator.stage_, estimator.n_clusters_) == (("knn", 0.2), "base", 4)
    assert estimator.silhouette_ == pytest.approx(0.462648, abs=1e-6)
    assert len(estimator.candidates_) == 17
    assert sum(c["admissible"] for c in estimator.candidates_) == 9
    assert adjusted_rand_score(train_truth, estimator.labels_) == pytest.approx(1.0)
    assert adjusted_rand_score(test_truth, estimator.predict(test_x)) == pytest.approx(1.0)


@needs_apqc
@pytest.mark.slow
def test_pipeline_train_test(blobs):
    x, truth = blobs
    pipe = make_clustering_pipeline(components=2, bandwidths=["scalar:0.2"], steps=200, random_state=0)
    pipe.fit(x[::2])
    assert adjusted_rand_score(truth[1::2], pipe.predict(x[1::2])) == pytest.approx(1.0)
    assert pipe.predict_proba(x[1::2]).shape == (30, len(np.unique(pipe[-1].source_labels_)))


@needs_apqc
@pytest.mark.slow
def test_project_without_target(blobs):
    x, truth = blobs
    project = Project(pd.DataFrame(x, columns=list("abcd")))
    project.preprocess()
    labels, probabilities = project.cluster("pqc", sigma=0.2, steps=200, random_state=0)
    assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


# Horn and Gottlieb Quantum Clustering and quantum k-means only need the core dependencies


def test_potential_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    data, points, sigma, h = rng.normal(size=(20, 3)), rng.normal(size=(4, 3)), 0.7, 1e-6
    numeric = np.stack(
        [
            (potential(points + h * e, data, sigma) - potential(points - h * e, data, sigma)) / (2 * h)
            for e in np.eye(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(potential_gradient(points, data, sigma), numeric, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("dimension", [1, 3, 4, 6])
def test_interference_probability_is_the_circuit_one(dimension):
    rng = np.random.default_rng(dimension)
    u, v = rng.normal(size=(5, dimension)), rng.normal(size=(5, dimension))
    expected = np.sum((u - v) ** 2, axis=1) / (2 * (np.sum(u**2, axis=1) + np.sum(v**2, axis=1)))
    np.testing.assert_allclose(interference_probability(u, v), expected)


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="requires qiskit")
def test_interference_circuit_runs_on_qiskit():
    from qiskit.primitives import StatevectorSampler

    u, v = np.array([1.0, -0.5, 2.0]), np.array([0.2, 0.4, -1.0])
    circuit = interference_circuit(u, v)
    assert circuit.num_qubits == 3  # two halves of 4 amplitudes
    counts = StatevectorSampler(seed=0).run([circuit], shots=20_000).result()[0].data.meas.get_counts()  # ty: ignore[unresolved-attribute]
    assert counts.get("1", 0) / 20_000 == pytest.approx(interference_probability(u[None], v[None])[0], abs=0.01)


def test_interference_distances():
    rng = np.random.default_rng(4)
    x, centers = rng.normal(size=(30, 3)), rng.normal(size=(2, 3))
    exact = np.sum((x[:, None] - centers[None]) ** 2, axis=-1)
    np.testing.assert_allclose(interference_distances(x, centers, shots=None), exact)
    noisy = interference_distances(x, centers, shots=4096, rng=rng)
    assert np.max(np.abs(noisy - exact)) < 0.25 * np.max(exact)


@pytest.mark.parametrize("estimator", [SchrodingerClustering(), QuantumKMeans(2, random_state=0)])
def test_core_estimators_fit_and_predict(estimator, blobs):
    x, truth = blobs
    estimator.fit(x[::2])
    assert estimator.n_clusters_ == 2
    assert adjusted_rand_score(truth[::2], estimator.labels_) == pytest.approx(1.0)
    assert adjusted_rand_score(truth[1::2], estimator.predict(x[1::2])) == pytest.approx(1.0)

    proba = estimator.predict_proba(x[1::2])
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(estimator)).predict(x), estimator.predict(x))


def test_schrodinger_candidates(blobs):
    x, _ = blobs
    estimator = SchrodingerClustering(sigmas=[0.05, 0.5]).fit(x)
    assert [c["sigma"] for c in estimator.candidates_] == [0.05, 0.5]
    assert not estimator.candidates_[0]["admissible"]  # a tiny width leaves every sample on its own
    assert estimator.sigma_ == pytest.approx(0.5 * estimator.scale_)
    assert estimator.cluster_centers_.shape == (2, 4)


def test_schrodinger_no_admissible_partition(blobs):
    x, _ = blobs
    with pytest.raises(RuntimeError, match="No admissible"):
        SchrodingerClustering(min_clusters=3).fit(x)


def test_quantum_kmeans_exact_distances(blobs):
    x, _ = blobs
    estimator = QuantumKMeans(2, shots=None, random_state=0).fit(x)
    exact = np.sqrt(np.sum((x[:, None] - estimator.cluster_centers_[None]) ** 2, axis=-1))
    np.testing.assert_allclose(estimator.transform(x), exact)
    assert estimator.inertia_ == pytest.approx(np.sum(exact.min(axis=1) ** 2))


def test_quantum_kmedians_is_robust_to_outliers(blobs):
    x, _ = blobs
    x = np.r_[x[30:], np.full((3, 4), 40.0)]  # one blob around 2 and a few far away outliers
    mean = QuantumKMeans(1, shots=None, random_state=0).fit(x).cluster_centers_[0]
    median = QuantumKMeans(1, center="median", shots=None, random_state=0).fit(x).cluster_centers_[0]
    assert np.linalg.norm(median - 2) < 0.5 < 5 < np.linalg.norm(mean - 2)


def test_quantum_kmeans_is_reproducible(blobs):
    x, _ = blobs
    first = QuantumKMeans(3, random_state=5).fit(x)
    second = QuantumKMeans(3, random_state=5).fit(x)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)


@pytest.mark.parametrize(("kwargs", "match"), [({"center": "mode"}, "median"), ({"n_clusters": 100}, "n_clusters")])
def test_quantum_kmeans_invalid_arguments(blobs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        QuantumKMeans(**kwargs).fit(blobs[0])


@pytest.mark.parametrize("estimator", [SchrodingerClustering(), QuantumKMeans(4, random_state=0)])
def test_four_profiles_with_core_estimators(estimator):
    df = pd.read_csv(FIXTURE)
    features = [col for col in df.columns if col.startswith("f")]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    pipe = make_clustering_pipeline(components=6, estimator=estimator).fit(train[features])
    assert pipe[-1].n_clusters_ == 4
    assert adjusted_rand_score(train["profile"], pipe[-1].labels_) == pytest.approx(1.0)
    assert adjusted_rand_score(test["profile"], pipe.predict(test[features])) == pytest.approx(1.0)


def test_pipeline_rejects_options_with_an_estimator():
    with pytest.raises(ValueError, match="estimator"):
        make_clustering_pipeline(estimator=QuantumKMeans(), n_clusters=3)


@pytest.mark.parametrize(("ctype", "kwargs"), [("qc", {}), ("qkmeans", {"n_clusters": 2, "random_state": 0})])
def test_project_core_clustering(blobs, ctype, kwargs):
    x, truth = blobs
    project = Project(pd.DataFrame(x, columns=list("abcd")))
    project.preprocess()
    labels, weights = project.cluster(ctype, **kwargs)
    assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)
    assert ((weights >= 0) & (weights <= 1)).all()
