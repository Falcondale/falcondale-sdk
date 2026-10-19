"""Missing optional dependencies must point to the extra that provides them."""

import sys

import numpy as np
import pytest

from falcondale import Project
from falcondale.classifiers import build_qnn, build_qsvc
from falcondale.clustering import QuantumClustering
from falcondale.feature_selection import build_qubo


def _hide(monkeypatch, *packages: str):
    """Makes the given packages (and their submodules) fail to import."""
    for name in list(sys.modules):
        # Falcondale helpers importing them must be loaded again
        if name.split(".")[0] in packages or name.startswith("falcondale.helpers."):
            monkeypatch.delitem(sys.modules, name)
    for package in packages:
        monkeypatch.setitem(sys.modules, package, None)


def test_feature_selection_hint(monkeypatch, project):
    _hide(monkeypatch, "dimod")
    with pytest.raises(ImportError, match=r"falcondale\[feature-selection\]"):
        build_qubo(project.dataset)


@pytest.mark.parametrize(("backend", "package"), [("qiskit", "qiskit_machine_learning"), ("pennylane", "pennylane")])
def test_qsvc_hint(monkeypatch, backend, package):
    _hide(monkeypatch, package)
    with pytest.raises(ImportError, match=r"falcondale\[qml\]"):
        build_qsvc(2, backend=backend)


def test_qnn_hint(monkeypatch):
    _hide(monkeypatch, "torch")
    with pytest.raises(ImportError, match=r"falcondale\[qml\]"):
        build_qnn(2)


def test_clustering_hint(monkeypatch):
    _hide(monkeypatch, "apqc")
    with pytest.raises(ImportError, match=r"falcondale\[clustering\]"):
        QuantumClustering().fit(np.zeros((10, 2)))


def test_list_options_hints_missing_extras(monkeypatch, capsys):
    monkeypatch.setattr("falcondale.project._installed", lambda extra: extra != "qml")
    Project.list_options()
    out = capsys.readouterr().out
    assert 'pip install "falcondale[qml]"' in out
    assert "falcondale[clustering]" not in out


def test_supervised_tasks_need_a_target(breast_cancer):
    project = Project(breast_cancer.drop(columns="target"))
    project.preprocess()
    assert len(project.show_features().columns) == len(breast_cancer.columns) - 1
    with pytest.raises(ValueError, match="target"):
        project.evaluate("svc")
    with pytest.raises(ValueError, match="target"):
        project.feature_selection(max_cols=2)
