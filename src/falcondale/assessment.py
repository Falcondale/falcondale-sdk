"""
Module assessing how well suited a classification dataset is for quantum machine learning.

Quantum models are not a free upgrade over classical ones. Whether they can bring something
to the table depends on how the decision boundary looks like, how hard the problem is for
classical models and how many qubits are available. `QMLAssessor` combines several
diagnostics into a single suitability score:

* **Fourier approximability**: angle encoded circuits express truncated Fourier series of
  their inputs, so a decision boundary captured by a few frequencies is a good sign
* **Trainability**: frequencies above 1 cannot be reached from zero initialised encodings
  and those above 2 not even with ternary {-1, 0, 1} initialisations
  ([Pöppel et al.](https://arxiv.org/abs/2602.23409))
* **Circuit depth**: depth estimate of a fixed frequency encoding reaching the spectrum
* **Classical hardness**: a linear model solving the task leaves little room for quantum
  kernels, a large gap between a decision tree and a linear model points to nonlinearities
* **Class balance** and **dimensionality** compared to the qubit budget

The diagnostics are then used to rank the models available in Falcondale, so that
`Project.evaluate("auto")` can pick the most promising one.

    >>> assessor = QMLAssessor(n_qubits=8).fit(X, y)
    >>> assessor.report()
    >>> assessor.best_model
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from .classifiers import MAX_QSVC_QUBITS, compute_metrics

# Weights of each sub-score in the final score
WEIGHTS = {
    "fourier": 0.25,
    "trainability": 0.25,
    "depth": 0.15,
    "difficulty": 0.20,
    "imbalance": 0.10,
    "dimensionality": 0.05,
}

CRITERIA = {
    "fourier": "Fourier approx.",
    "trainability": "Trainability",
    "depth": "Circuit depth",
    "difficulty": "Classical hardness",
    "imbalance": "Class balance",
    "dimensionality": "Dimensionality",
}

VERDICTS = {
    "HIGH": "Quantum models are worth a try",
    "MEDIUM": "Quantum models might help, with care",
    "LOW": "Classical models are recommended",
}

BASELINES = {"lr": "Logistic regression", "dt": "Decision tree"}

# Same block rich uses in notebooks: no wrapping and fonts with fixed width box drawing characters
PRE_STYLE = (
    '<pre style="white-space:pre;overflow-x:auto;line-height:normal;'
    "font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">{code}</pre>"
)


def _colour(score: float) -> str:
    return "green" if score >= 70 else "yellow" if score >= 45 else "red"


# Fourier helpers


def _dft_magnitudes(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One sided DFT magnitudes (energy of the negative frequencies included) and coefficients."""
    coeffs = np.fft.rfft(signal)
    mags = np.abs(coeffs) / len(signal)
    mags[1:-1] *= 2
    return mags, coeffs


def _top_frequencies(
    power: np.ndarray, noise_floor: float, energy_threshold: float = 0.95, max_k: int = 16
) -> np.ndarray:
    """
    Smallest set of non constant frequencies above the noise floor carrying `energy_threshold`
    of the power above that floor. Empty when nothing stands out from the noise.
    """
    candidates = [idx for idx in np.argsort(power)[::-1] if idx != 0 and power[idx] > noise_floor]
    total = sum(power[idx] for idx in candidates)
    selected, cumulative = [], 0.0
    for idx in candidates[:max_k]:
        selected.append(idx)
        cumulative += power[idx] / total
        if cumulative >= energy_threshold:
            break
    return np.array(selected, dtype=int)


def _fourier_r2(signal: np.ndarray, frequencies: np.ndarray) -> float:
    """R² of the reconstruction of the signal keeping only the given frequencies (and the constant term)."""
    _, coeffs = _dft_magnitudes(signal)
    kept = np.zeros_like(coeffs)
    kept[0] = coeffs[0]
    kept[frequencies] = coeffs[frequencies]
    recon = np.fft.irfft(kept, n=len(signal))
    ss_tot = np.var(signal) * len(signal) + 1e-12
    return float(np.clip(1 - np.sum((signal - recon) ** 2) / ss_tot, 0.0, 1.0))


@dataclass(frozen=True)
class FeatureSpectrum:
    """
    Spectral complexity of the decision boundary along one feature.

    Attributes:
        name (str): Feature name
        omega_max (int): Highest significant frequency, 0 if the feature shows no signal
        frequencies (int): Number of significant frequencies (K)
        fourier_r2 (float): R² of the label signal rebuilt from the significant frequencies
        unreachable_zero (int): Frequencies out of reach of zero initialised encodings (above 1)
        unreachable_ternary (int): Frequencies out of reach of ternary initialised encodings (above 2)
    """

    name: str
    omega_max: int
    frequencies: int
    fourier_r2: float
    unreachable_zero: int
    unreachable_ternary: int

    @property
    def informative(self) -> bool:
        """Whether the labels show any structure along the feature."""
        return self.frequencies > 0


def feature_spectrum(
    values: np.ndarray,
    labels: np.ndarray,
    name: str = "x",
    energy_threshold: float = 0.95,
    max_k: int = 16,
    random_state: int | np.random.Generator | None = 42,
) -> FeatureSpectrum:
    """
    Analyses how fast the labels oscillate along a feature.

    Samples are sorted by the feature value and the labels, read in that order, form a signal
    whose spectrum tells how many oscillations the model has to express along that axis. The
    frequency index counts oscillations over the whole range of the feature, so frequency 1
    is what a single rotation expresses when the feature is scaled to a full period.

    Overlapping classes and irrelevant features produce noisy signals with power spread over
    all frequencies, so only frequencies stronger than any of the same labels shuffled count.
    Multiclass labels are analysed one class against the rest.

    Args:
        values (np.ndarray): Feature values
        labels (np.ndarray): Class labels
        name (str): Feature name
        energy_threshold (float): Share of the signal power the significant frequencies must carry
        max_k (int): Maximum number of significant frequencies
        random_state (int | np.random.Generator): Seed of the shuffles estimating the noise floor

    Returns:
        FeatureSpectrum: Spectral summary of the feature
    """
    rng = np.random.default_rng(random_state)
    sorted_labels = labels[np.argsort(values, kind="stable")]
    classes = np.unique(sorted_labels)
    # Binary problems only need the positive class, the negative one mirrors it
    classes = classes[-1:] if len(classes) == 2 else classes

    window = max(1, len(sorted_labels) // 32)

    def power(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # A short moving average reduces the quantisation noise of 0/1 signals
        smooth = np.convolve(signal.astype(float), np.ones(window) / window, mode="same")
        return smooth, _dft_magnitudes(smooth)[0] ** 2

    selected, r2s = set(), []
    for cls in classes:
        signal = sorted_labels == cls
        smooth, spectrum = power(signal)
        # The strongest frequency of a few shuffles, so that chance peaks over hundreds of frequencies rarely pass
        noise_floor = max(power(rng.permutation(signal))[1][1:].max() for _ in range(5))
        top = _top_frequencies(spectrum, float(noise_floor), energy_threshold, max_k)
        if len(top):
            selected.update(int(k) for k in top)
            r2s.append(_fourier_r2(smooth, top))

    freqs = np.array(sorted(selected), dtype=int)
    return FeatureSpectrum(
        name=name,
        omega_max=int(freqs.max()) if len(freqs) else 0,
        frequencies=len(freqs),
        fourier_r2=float(np.mean(r2s)) if r2s else 0.0,
        unreachable_zero=int(np.sum(freqs > 1)),
        unreachable_ternary=int(np.sum(freqs > 2)),
    )


# Other diagnostics


def class_balance(y: np.ndarray) -> tuple[float, str]:
    """Ratio between the smallest and the largest class (1 means balanced) and its label."""
    counts = np.bincount(LabelEncoder().fit_transform(y)).tolist()
    ratio = min(counts) / max(counts)
    label = "balanced" if ratio >= 0.8 else "mild" if ratio >= 0.4 else "moderate" if ratio >= 0.1 else "severe"
    return ratio, label


def classical_baseline(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> dict[str, dict[str, float]]:
    """
    Out of fold (3 folds) metrics of a logistic regression and a shallow decision tree.

    Returns:
        dict: Metrics per baseline (`lr` and `dt`), as in `falcondale.classifiers.compute_metrics`
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    models = {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "dt": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=random_state),
    }
    results = {}
    for key, model in models.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        y_pred = np.unique(y)[proba.argmax(axis=1)]
        results[key] = compute_metrics(y, y_pred, proba)
    return results


def kernel_spectrum(
    X: np.ndarray, gamma: float | None = None, max_samples: int = 2000, random_state: int = 42
) -> dict[str, float]:
    """
    Eigenspectrum diagnostics of the centred RBF kernel matrix.

    A low effective rank means the data lives in a few kernel directions, a large condition
    number warns about ill conditioned kernel matrices.

    Returns:
        dict: samples used, spectral gap, condition number and effective rank
    """
    if X.shape[0] > max_samples:
        X = X[np.random.default_rng(random_state).choice(X.shape[0], size=max_samples, replace=False)]

    K = rbf_kernel(X, gamma=gamma)
    centred = K - K.mean(axis=0) - K.mean(axis=1)[:, None] + K.mean()
    eigvals = np.clip(np.sort(np.linalg.eigvalsh(centred))[::-1], 0.0, None)
    positive = eigvals[eigvals > 1e-12]

    p = positive / positive.sum() if positive.size else positive
    return {
        "samples": int(X.shape[0]),
        "spectral_gap": float(eigvals[0] - eigvals[1]) if len(eigvals) > 1 else float(eigvals[0]),
        "condition_number": float(positive[0] / positive[-1]) if positive.size else math.inf,
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))) if positive.size else 0.0,
    }


@dataclass(frozen=True)
class ModelRecommendation:
    """
    Suitability of one of Falcondale's models for the assessed dataset.

    Attributes:
        model (str): Model name, as accepted by `Project.evaluate`
        options (dict): Model arguments to pass to `Project.evaluate`
        score (float): Suitability score, from 0 to 100
        qubits (int): Qubits the model needs, 0 for classical models
        feasible (bool): Whether the model fits the qubit budget
        reasons (list[str]): Short explanations of the score
    """

    model: str
    options: dict[str, Any]
    score: float
    qubits: int
    feasible: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Model name and options as one would write them."""
        return self.model + "".join(f" ({k}={v})" for k, v in self.options.items())


class QMLAssessor(BaseEstimator):
    """
    Assesses how well suited a classification dataset is for quantum machine learning and
    ranks Falcondale's models accordingly.

    Follows the scikit-learn estimator API, so it is fitted with `fit(X, y)`. Features are
    scaled into [0, 1] before the analysis.

    Attributes:
        score_ (float): Overall suitability score, from 0 to 100
        verdict_ (str): HIGH, MEDIUM or LOW
        sub_scores_ (dict[str, float]): Score of each criterion, from 0 to 100
        details_ (dict): Values behind the sub-scores (frequencies, qubits needed, depth,...)
        feature_spectra_ (list[FeatureSpectrum]): Spectral summary of each analysed feature
        balance_ (tuple[float, str]): Class balance ratio and its label
        baseline_ (dict | None): Out of fold metrics of the classical baselines
        kernel_spectrum_ (dict | None): RBF kernel eigenspectrum diagnostics
        recommendations_ (list[ModelRecommendation]): Falcondale models, most suitable first
    """

    def __init__(
        self,
        n_qubits: int = MAX_QSVC_QUBITS,
        epsilon: float = 0.1,
        energy_threshold: float = 0.95,
        max_frequencies: int = 16,
        baseline: bool = True,
        kernel: bool = True,
        kernel_gamma: float | None = None,
        max_samples: int | None = None,
        max_features: int | None = None,
        random_state: int = 42,
    ):
        """
        Args:
            n_qubits (int): Qubit budget
            epsilon (float): Approximation precision of the circuit depth estimate
            energy_threshold (float): Share of the label signal power the significant frequencies must carry
            max_frequencies (int): Maximum number of significant frequencies per feature
            baseline (bool): Fit the classical baselines (logistic regression and decision tree)
            kernel (bool): Compute the RBF kernel eigenspectrum diagnostics
            kernel_gamma (float): RBF kernel width, scikit-learn's default if not provided
            max_samples (int): (Optional) Stratified subsample used for the assessment of large datasets
            max_features (int): (Optional) Only analyse the features with the highest variance
            random_state (int): Seed for subsampling and cross validation
        """
        self.n_qubits = n_qubits
        self.epsilon = epsilon
        self.energy_threshold = energy_threshold
        self.max_frequencies = max_frequencies
        self.baseline = baseline
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y) -> QMLAssessor:
        """
        Runs the assessment.

        Args:
            X (array-like): Features (samples x features), a dataframe keeps the column names
            y (array-like): Class labels

        Returns:
            QMLAssessor: The fitted assessor
        """
        names = [str(c) for c in X.columns] if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        X = np.asarray(X, dtype=float)
        y = LabelEncoder().fit_transform(np.asarray(y))
        if len(np.unique(y)) < 2:
            raise ValueError("At least two classes are needed to assess a classification task.")

        X = np.nan_to_num(np.where(np.isfinite(X), X, np.nan), nan=0.0)
        X = MinMaxScaler().fit_transform(X)
        self.n_samples_in_, self.n_features_in_ = X.shape
        self.n_classes_ = len(np.unique(y))

        if self.max_samples is not None and self.max_samples < len(y):
            split = StratifiedShuffleSplit(n_splits=1, train_size=self.max_samples, random_state=self.random_state)
            rows = np.sort(next(split.split(X, y))[0])
            X, y = X[rows], y[rows]
        if self.max_features is not None and self.max_features < X.shape[1]:
            cols = np.sort(np.argsort(X.var(axis=0))[::-1][: self.max_features])
            X, names = X[:, cols], [names[i] for i in cols]

        rng = np.random.default_rng(self.random_state)
        self.feature_spectra_ = [
            feature_spectrum(X[:, i], y, names[i], self.energy_threshold, self.max_frequencies, rng)
            for i in range(X.shape[1])
        ]
        self.balance_ = class_balance(y)
        self.baseline_ = classical_baseline(X, y, self.random_state) if self.baseline else None
        self.kernel_spectrum_ = (
            kernel_spectrum(X, gamma=self.kernel_gamma, random_state=self.random_state) if self.kernel else None
        )
        self.samples_used_, self.features_used_ = X.shape

        self._score()
        self.recommendations_ = self.recommend_models()
        return self

    def _score(self):
        # Features without any signal do not constrain the circuit, they only waste qubits
        spectra = [s for s in self.feature_spectra_ if s.informative]
        if spectra:
            omega_max = max(s.omega_max for s in spectra)
            # Features left out by max_features are assumed to be like the analysed ones
            n_freqs = sum(s.frequencies for s in spectra) * self.n_features_in_ / len(self.feature_spectra_)
            unreachable_zero = float(np.mean([s.unreachable_zero / s.frequencies for s in spectra]))
            unreachable_ternary = float(np.mean([s.unreachable_ternary / s.frequencies for s in spectra]))
            avg_r2 = float(np.mean([s.fourier_r2 for s in spectra]))
        else:
            omega_max, n_freqs, unreachable_zero, unreachable_ternary, avg_r2 = 0, 0.0, 1.0, 1.0, 0.0

        # Depth of a fixed frequency encoding, O(ω (ω + ε⁻²))
        depth = int(omega_max * (omega_max + round(self.epsilon**-2)))
        qubits_needed = max(1, math.ceil(math.log2(n_freqs + 1)))

        if self.baseline_ is not None:
            lr_bacc = self.baseline_["lr"]["balanced accuracy"]
            gap = self.baseline_["dt"]["balanced accuracy"] - lr_bacc
        else:
            lr_bacc, gap = 0.5, 0.0

        difficulty = 20.0 if lr_bacc > 0.9 else 60.0 if lr_bacc > 0.75 else 100.0
        subs = {
            "fourier": min(100.0, avg_r2 * 120),
            "trainability": max(0.0, (1 - unreachable_ternary) * 100),
            "depth": 100.0 if depth <= 200 else 60.0 if depth <= 2000 else 25.0 if depth <= 10_000 else 5.0,
            # A decision tree beating a linear model hints at nonlinearities a kernel could capture
            "difficulty": float(np.clip(difficulty + gap * 50, 0.0, 100.0)),
            "imbalance": min(100.0, self.balance_[0] * 120),
            "dimensionality": 100.0 if self.n_qubits >= qubits_needed else max(5.0, self.n_qubits / qubits_needed * 80),
        }
        self.sub_scores_ = {k: round(v, 1) for k, v in subs.items()}
        self.score_ = round(float(np.clip(sum(WEIGHTS[k] * v for k, v in subs.items()), 0.0, 100.0)), 1)
        self.verdict_ = "HIGH" if self.score_ >= 70 else "MEDIUM" if self.score_ >= 45 else "LOW"
        self.details_ = {
            "informative_features": len(spectra),
            "omega_max": omega_max,
            "frequencies": round(n_freqs),
            "avg_fourier_r2": round(avg_r2, 4),
            "unreachable_zero_pct": round(unreachable_zero * 100, 1),
            "unreachable_ternary_pct": round(unreachable_ternary * 100, 1),
            "depth_estimate": depth,
            "qubits_needed": qubits_needed,
            "lr_balanced_accuracy": lr_bacc,
            "nonlinearity_gap": gap,
        }

    def _check_fitted(self):
        if not hasattr(self, "score_"):
            raise RuntimeError("The assessor has not been fitted yet! Call fit(X, y) first.")

    def recommend_models(self) -> list[ModelRecommendation]:
        """
        Ranks the models available in `Project.evaluate` for the assessed dataset.

        * **qsvc (backend="pennylane")**: angle embedding, each feature enters through a single
          rotation so only frequency 1 is expressed. Suits smooth boundaries
        * **qsvc (backend="qiskit")**: the ZZ feature map repeats the encoding and adds feature
          products, reaching higher frequencies and interactions
        * **qnn**: amplitude embedding, n features need ceil(log2 n) qubits, so it fits large
          feature spaces in small budgets, at the price of trainability
        * **svc**: the classical baseline, the better the less suitable the dataset is for QML

        Returns:
            list[ModelRecommendation]: Models sorted from the most suitable one
        """
        self._check_fitted()
        d, subs = self.details_, self.sub_scores_
        n_features = self.n_features_in_
        kernel_budget = min(self.n_qubits, MAX_QSVC_QUBITS)
        kernel_ok = n_features <= kernel_budget
        too_many = f"needs {n_features} qubits, budget is {kernel_budget}"
        large = self.n_samples_in_ > 2000
        large_reason = f"kernel matrix grows with {self.n_samples_in_}² sample pairs"
        reach_one = 100 - d["unreachable_zero_pct"]
        reach_two = 100 - d["unreachable_ternary_pct"]
        gap = d["nonlinearity_gap"]

        angle = 0.4 * self.score_ + 0.6 * (0.6 * reach_one + 0.4 * subs["fourier"]) - 10 * large
        angle_reasons = [f"{reach_one:.0f}% of the frequencies within reach of a single rotation"]
        if reach_one >= 80:
            angle += 10
            angle_reasons.append("smooth boundary, the simplest encoding should be enough")
        zz = 0.4 * self.score_ + 0.6 * (0.6 * reach_two + 0.4 * subs["fourier"]) + min(15.0, 50 * max(gap, 0))
        zz -= 10 * large
        zz_reasons = [f"{reach_two:.0f}% of the frequencies within reach of a repeated encoding"]
        if gap > 0.02:
            zz_reasons.append(f"decision tree beats logistic regression by {gap:+.3f}, feature products may help")
        if n_features > 8:
            # Kernel values of expressive feature maps concentrate as qubits grow (Thanasilp et al.)
            zz -= 2 * (n_features - 8)
            zz_reasons.append("kernel values of expressive feature maps concentrate with many qubits")
        if large:
            angle_reasons.append(large_reason)
            zz_reasons.append(large_reason)

        qnn_qubits = max(1, math.ceil(math.log2(n_features)))
        qnn = 0.4 * self.score_ + 0.6 * (0.5 * subs["trainability"] + 0.5 * subs["difficulty"])
        qnn_reasons = [f"amplitude embedding fits {n_features} features in {qnn_qubits} qubit{'s' * (qnn_qubits > 1)}"]
        if not kernel_ok:
            qnn += 15
            qnn_reasons.append("the only quantum model fitting the qubit budget")
        if self.samples_used_ < 200:
            qnn -= 10
            qnn_reasons.append("few samples to train the circuit parameters")

        # A classically easy task is better served by the classical model, whatever the other criteria say
        svc = max(100 - self.score_, 100 - subs["difficulty"])
        svc_reasons = [f"classical baseline, overall QML suitability is {self.verdict_.lower()}"]
        if d["lr_balanced_accuracy"] > 0.9:
            svc_reasons.append(f"logistic regression already reaches {d['lr_balanced_accuracy']:.3f} balanced accuracy")

        candidates = [
            ("qsvc", {"backend": "pennylane"}, angle, n_features, kernel_ok, angle_reasons),
            ("qsvc", {"backend": "qiskit"}, zz, n_features, kernel_ok, zz_reasons),
            ("qnn", {}, qnn, qnn_qubits, qnn_qubits <= self.n_qubits, qnn_reasons),
            ("svc", {}, svc, 0, True, svc_reasons),
        ]
        recommendations = [
            ModelRecommendation(
                model,
                options,
                round(float(np.clip(score, 0.0, 100.0)), 1) if feasible else 0.0,
                qubits,
                feasible,
                reasons if feasible else [too_many if model == "qsvc" else f"needs {qubits} qubits", *reasons],
            )
            for model, options, score, qubits, feasible, reasons in candidates
        ]
        return sorted(recommendations, key=lambda r: (r.feasible, r.score), reverse=True)

    @property
    def best_model(self) -> ModelRecommendation:
        """The most suitable model for the dataset."""
        self._check_fitted()
        return self.recommendations_[0]

    def advice(self) -> list[str]:
        """
        Practical hints derived from the assessment.

        Returns:
            list[str]: One hint per line
        """
        self._check_fitted()
        subs, d = self.sub_scores_, self.details_
        hints = []
        if subs["trainability"] < 50:
            hints.append("Many frequencies are out of reach, initialise encoding angles on a {-1, 0, 1} grid.")
        if subs["depth"] < 60:
            hints.append("Fixed frequency encodings get deep, prefer trainable frequency encodings or re-uploading.")
        if subs["fourier"] < 50:
            hints.append("The boundary is poorly captured by a few frequencies, angle encodings may not suit it.")
        if subs["difficulty"] < 40:
            hints.append("A linear model already solves the task, a quantum kernel is unlikely to add value.")
        if self.balance_[1] in ("moderate", "severe"):
            hints.append("Classes are imbalanced, use class weights or resampling before training.")
        if self.n_qubits < d["qubits_needed"]:
            hints.append(
                f"Raise the qubit budget to {d['qubits_needed']} or reduce the features with "
                "feature_selection() or preprocess(reduced_dimension=...)."
            )
        silent = self.features_used_ - d["informative_features"]
        if silent:
            hints.append(f"{silent} features show no signal along their axis, feature_selection() could drop them.")
        if self.n_features_in_ > min(self.n_qubits, MAX_QSVC_QUBITS):
            hints.append("Too many features for one qubit per feature, select features to try QSVC.")
        return hints

    def to_dict(self) -> dict[str, Any]:
        """
        The assessment as plain Python types, ready to be stored as JSON.

        Returns:
            dict: Score, verdict, sub-scores, details, diagnostics and recommendations
        """
        self._check_fitted()
        return {
            "score": self.score_,
            "verdict": self.verdict_,
            "sub_scores": dict(self.sub_scores_),
            "details": dict(self.details_),
            "balance": {"ratio": self.balance_[0], "label": self.balance_[1]},
            "baseline": self.baseline_,
            "kernel_spectrum": self.kernel_spectrum_,
            "feature_spectra": [vars(s) for s in self.feature_spectra_],
            "recommendations": [vars(r) for r in self.recommendations_],
            "advice": self.advice(),
        }

    # Rich rendering

    def __rich__(self) -> Group:
        self._check_fitted()
        colour = _colour(self.score_)
        d = self.details_

        header = Text.assemble(
            (f"{self.score_:.1f}", f"bold {colour}"),
            (" / 100   ", "dim"),
            (self.verdict_, f"bold {colour}"),
            "  ",
            VERDICTS[self.verdict_],
            "\n",
            (
                f"{self.samples_used_} of {self.n_samples_in_} samples, {self.features_used_} of "
                f"{self.n_features_in_} features, {self.n_classes_} classes, budget of {self.n_qubits} qubits",
                "dim",
            ),
        )
        notes = {
            "fourier": f"average R² {d['avg_fourier_r2']:.3f}",
            "trainability": f"{d['unreachable_ternary_pct']}% of frequencies unreachable (ternary init)",
            "depth": f"fixed frequency depth ≈ {d['depth_estimate']}",
            "difficulty": f"logistic bal. acc. {d['lr_balanced_accuracy']:.3f}, tree gap {d['nonlinearity_gap']:+.3f}"
            if self.baseline_
            else "baseline skipped",
            "imbalance": f"ratio {self.balance_[0]:.3f} ({self.balance_[1]})",
            "dimensionality": f"≥ {d['qubits_needed']} qubits needed",
        }
        scores = Table(box=None, show_header=False, pad_edge=False, expand=True)
        scores.add_column(style="bold", no_wrap=True)
        scores.add_column(width=22)
        scores.add_column(justify="right", width=5)
        scores.add_column(style="dim", ratio=1)
        for key, label in CRITERIA.items():
            value = self.sub_scores_[key]
            bar = ProgressBar(total=100, completed=value, width=22, complete_style=_colour(value))
            scores.add_row(label, bar, Text(f"{value:.0f}", style=_colour(value)), notes[key])

        parts = [
            Panel(header, title="[bold]QML suitability", title_align="left", border_style=colour),
            Panel(scores, title="Sub-scores", title_align="left", border_style="dim"),
        ]

        if self.baseline_:
            baseline = Table(title="Classical baseline (3 fold, out of fold)", title_justify="left", expand=True)
            baseline.add_column("Model", style="bold")
            metrics = {"balanced accuracy": "Bal. accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
            metrics["auc"] = "AUC"
            for header in metrics.values():
                baseline.add_column(header, justify="right")
            for key, label in BASELINES.items():
                baseline.add_row(label, *(f"{self.baseline_[key][m]:.3f}" for m in metrics))
            parts.append(baseline)

        spectral = Table(title="Spectrum", title_justify="left", show_header=False)
        spectral.add_column(style="bold")
        spectral.add_column(justify="right")
        spectral.add_row("Features showing signal", f"{d['informative_features']} of {self.features_used_}")
        spectral.add_row("Highest frequency ω", str(d["omega_max"]))
        spectral.add_row("Significant frequencies K", str(d["frequencies"]))
        spectral.add_row("Unreachable, zero init", f"{d['unreachable_zero_pct']}%")
        spectral.add_row("Unreachable, ternary init", f"{d['unreachable_ternary_pct']}%")
        if self.kernel_spectrum_:
            ks = self.kernel_spectrum_
            spectral.add_section()
            spectral.add_row("RBF kernel effective rank", f"{ks['effective_rank']:.2f}")
            spectral.add_row("RBF kernel spectral gap", f"{ks['spectral_gap']:.3f}")
            spectral.add_row("RBF kernel condition number", f"{ks['condition_number']:.3g}")

        features = Table(title="Most oscillating features", title_justify="left", expand=True)
        features.add_column("Feature", style="bold", overflow="ellipsis", max_width=28)
        features.add_column("ω", justify="right")
        features.add_column("K", justify="right")
        features.add_column("R²", justify="right")
        features.add_column("Unreachable", justify="right")
        for s in sorted(self.feature_spectra_, key=lambda s: (-s.omega_max, -s.frequencies))[:8]:
            features.add_row(
                s.name,
                str(s.omega_max),
                str(s.frequencies),
                f"{s.fourier_r2:.3f}",
                f"{s.unreachable_ternary}/{s.frequencies}",
            )
        parts.extend([spectral, features])

        models = Table(title="Recommended models", title_justify="left", expand=True)
        models.add_column("#", justify="right", style="dim")
        models.add_column("Model", style="bold cyan", no_wrap=True)
        models.add_column("Qubits", justify="right")
        models.add_column("Score", justify="right")
        models.add_column("Why", ratio=1)
        for rank, rec in enumerate(self.recommendations_, start=1):
            style = "" if rec.feasible else "dim strike"
            models.add_row(
                str(rank),
                Text(rec.name, style=style or "bold cyan"),
                str(rec.qubits) if rec.qubits else "-",
                Text(f"{rec.score:.0f}", style=_colour(rec.score) if rec.feasible else "dim"),
                Text("\n".join(rec.reasons), style="dim" if not rec.feasible else ""),
            )
        parts.append(models)

        best = self.best_model
        hints = Text.assemble(
            "Try ",
            (
                f'project.evaluate("{best.model}"' + "".join(f', {k}="{v}"' for k, v in best.options.items()) + ")",
                "bold cyan",
            ),
            ' first, or let project.evaluate("auto") pick it.',
        )
        for hint in self.advice():
            hints.append("\n• ", style="yellow")
            hints.append(hint)
        parts.append(Panel(hints, title="Next steps", title_align="left", border_style="cyan"))
        return Group(*parts)

    def report(self, console: Console | None = None):
        """
        Prints the assessment report, with colours in terminals and notebooks.

        Args:
            console (rich.console.Console): (Optional) Console to print to, a new one by default
        """
        (console or Console()).print(self)

    def _render(self, html: bool) -> str:
        console = Console(record=True, file=io.StringIO(), width=80, force_jupyter=False, color_system="truecolor")
        console.print(self)
        if html:
            return console.export_html(inline_styles=True, code_format=PRE_STYLE)
        return console.export_text()

    def _repr_mimebundle_(self, **kwargs) -> dict[str, str]:
        # Unfitted assessors keep scikit-learn's estimator display
        if not hasattr(self, "score_"):
            return super()._repr_mimebundle_(**kwargs)
        return {"text/plain": self._render(html=False), "text/html": self._render(html=True)}
