"""
Checks an installation with only some extras: installed families must work and missing
ones must fail with a hint about the extra to install.

    uv run python scripts/check_extras.py [extra ...]
"""

import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn import datasets

from falcondale import Project
from falcondale.clustering import APQC_SUPPORTED, QuantumClustering, QuantumKMeans, SchrodingerClustering
from falcondale.helpers.interference import interference_circuit


def project() -> Project:
    features, target = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    data = features.iloc[:120, :4].assign(target=target.iloc[:120])
    project = Project(data, target="target")
    project.preprocess()
    return project


def feature_selection():
    assert len(project().feature_selection(max_cols=2, method="sa", seed=0)) == 2


def qml():
    p = project()
    p.evaluate("qsvc", backend="pennylane")
    p.evaluate("qnn", epochs=1)


def tabular():
    data = pd.DataFrame({"plan": ["basic", "plus", None, "plus"] * 10, "fee": [20.0, None, 45.0, 30.0] * 10})
    p = Project(data.assign(target=[0, 1] * 20), target="target")
    p.preprocess()
    assert not p.show_features().isna().any().any()


def clustering():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-2, 0.3, (20, 2)), rng.normal(2, 0.3, (20, 2))]
    assert interference_circuit(x[0], x[1]).num_qubits > 0
    if APQC_SUPPORTED:
        assert QuantumClustering(bandwidths=["scalar:0.2"], steps=100, random_state=0).fit(x).n_clusters_ == 2


CHECKS: dict[str, Callable[[], None]] = {
    "feature-selection": feature_selection,
    "qml": qml,
    "clustering": clustering,
    "tabular": tabular,
}


def main(installed: set[str]):
    # The classical baseline and the core clustering methods only need the core dependencies
    assert pd.notna(project().evaluate("svc").metric("auc"))
    assert 0 <= project().assess().score_ <= 100
    x = np.r_[np.random.default_rng(0).normal(-2, 0.3, (20, 2)), np.random.default_rng(1).normal(2, 0.3, (20, 2))]
    assert SchrodingerClustering().fit(x).n_clusters_ == 2
    assert QuantumKMeans(2, random_state=0).fit(x).n_clusters_ == 2
    print("core: ok")

    for extra, check in CHECKS.items():
        if extra in installed:
            check()
            print(f"{extra}: ok")
            continue
        try:
            check()
        except ImportError as err:
            assert f"falcondale[{extra}]" in str(err), f"Unhelpful error for {extra}: {err}"
            print(f"{extra}: not installed, hint ok")
        else:
            raise AssertionError(f"{extra} works although it was not expected to be installed")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
