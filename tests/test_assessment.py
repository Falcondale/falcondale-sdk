import io
import json

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from sklearn import datasets

from falcondale import Model, Project, QMLAssessor
from falcondale.assessment import class_balance, feature_spectrum


@pytest.fixture(scope="module")
def circles() -> tuple[np.ndarray, np.ndarray]:
    """Hard for a linear model, smooth along each feature: a quantum kernel friendly task."""
    return datasets.make_circles(600, noise=0.08, factor=0.5, random_state=0)


def test_feature_spectrum_ignores_noise():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(600, 2))
    y = (np.cos(2 * np.pi * x[:, 0]) > 0).astype(int)

    signal = feature_spectrum(x[:, 0], y, "signal")
    noise = feature_spectrum(x[:, 1], y, "noise")

    assert signal.informative
    assert signal.omega_max <= 3
    assert signal.fourier_r2 > 0.8
    assert not noise.informative
    assert noise.omega_max == 0


def test_feature_spectrum_oscillations():
    x = np.linspace(0, 1, 800)
    slow = feature_spectrum(x, (np.sin(2 * np.pi * x) > 0).astype(int))
    fast = feature_spectrum(x, (np.sin(2 * np.pi * 6 * x) > 0).astype(int))
    assert slow.omega_max <= 3 < 6 <= fast.omega_max


def test_class_balance():
    assert class_balance(np.array([0, 1, 0, 1])) == (1.0, "balanced")
    assert class_balance(np.array(["a"] * 95 + ["b"] * 5))[1] == "severe"


def test_assessor_scores(circles):
    assessor = QMLAssessor().fit(*circles)

    assert 0 <= assessor.score_ <= 100
    assert assessor.verdict_ == "HIGH"
    assert set(assessor.sub_scores_) == {
        "fourier",
        "trainability",
        "depth",
        "difficulty",
        "imbalance",
        "dimensionality",
    }
    assert all(0 <= v <= 100 for v in assessor.sub_scores_.values())
    assert assessor.baseline_ is not None
    assert assessor.baseline_["lr"]["balanced accuracy"] < 0.75
    assert assessor.best_model.model == "qsvc"


def test_classically_easy_task_recommends_svc():
    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    assessor = QMLAssessor().fit(X, y)

    assert assessor.best_model.model == "svc"
    assert [s.name for s in assessor.feature_spectra_] == list(X.columns)
    scores = [r.score for r in assessor.recommendations_]
    assert scores == sorted(scores, reverse=True)


def test_qubit_budget_rules_out_kernels():
    X, y = datasets.make_classification(300, n_features=20, random_state=0)
    assessor = QMLAssessor(n_qubits=8).fit(X, y)

    infeasible = [r for r in assessor.recommendations_ if not r.feasible]
    assert {r.name for r in infeasible} == {"qsvc (backend=pennylane)", "qsvc (backend=qiskit)"}
    assert all(r.score == 0 for r in infeasible)
    assert assessor.recommendations_[-1].feasible is False
    assert any("select features" in hint for hint in assessor.advice())


def test_multiclass_and_subsampling():
    X, y = datasets.load_wine(return_X_y=True, as_frame=True)
    assessor = QMLAssessor(max_samples=120, max_features=5, kernel=False).fit(X, y.map({0: "a", 1: "b", 2: "c"}))

    assert assessor.n_classes_ == 3
    assert (assessor.samples_used_, assessor.features_used_) == (120, 5)
    assert assessor.n_features_in_ == 13
    assert assessor.kernel_spectrum_ is None


def test_single_class_fails():
    with pytest.raises(ValueError):
        QMLAssessor().fit(np.zeros((10, 2)), np.ones(10))


def test_unfitted_assessor():
    assessor = QMLAssessor()
    with pytest.raises(RuntimeError):
        assessor.report()
    # Unfitted assessors keep scikit-learn's display
    assert "QMLAssessor" in assessor._repr_mimebundle_()["text/plain"]


def test_report_and_exports(circles):
    assessor = QMLAssessor(baseline=False).fit(*circles)

    output = io.StringIO()
    assessor.report(Console(file=output, width=120))
    text = output.getvalue()
    assert "QML suitability" in text
    assert "Recommended models" in text
    assert "baseline skipped" in text

    bundle = assessor._repr_mimebundle_()
    assert "Next steps" in bundle["text/plain"]
    assert bundle["text/html"].startswith("<pre")

    payload = json.loads(json.dumps(assessor.to_dict()))
    assert payload["verdict"] == assessor.verdict_
    assert len(payload["feature_spectra"]) == 2


def test_project_assess_and_auto(project, capsys):
    assessment = project.assess(n_qubits=4)
    assert project.assessment is assessment
    assert assessment.n_qubits == 4

    model = project.evaluate("auto")
    assert isinstance(model, Model)
    assert model.type == assessment.best_model.model
    assert "Assessment recommends" in capsys.readouterr().out


def test_auto_reassesses_after_feature_selection(project):
    first = project.assess()
    project.feature_selection(max_cols=3, seed=0)
    project.evaluate("auto")
    assert project.assessment is not first
    assert project.assessment is not None
    assert project.assessment.n_features_in_ == 3


def test_assess_needs_target(breast_cancer):
    project = Project(breast_cancer.drop(columns="target"))
    project.preprocess()
    with pytest.raises(ValueError):
        project.assess()


def test_assess_keeps_names(project):
    names = [s.name for s in project.assess().feature_spectra_]
    assert names == project.dataset.columns()
    assert isinstance(project.show_features(), pd.DataFrame)
