# Installation

Falcondale is a Python library (3.11 to 3.14) that can be installed with the usual tools. To get everything:

```
pip install "falcondale[all]"
```

We encourage you to use [uv](https://docs.astral.sh/uv/) as it counts with a robust and fast dependency solving mechanism.

```
uv add "falcondale[all]"
```

Quantum computing frameworks (Qiskit, PennyLane, PyTorch, TensorFlow...) are heavy, so this can take a little while, we ask you for a little bit of patience at this stage.

## Choosing what to install

The core package only depends on numpy, pandas, scikit-learn, scipy and rich. Each family of quantum techniques, and the handling of non numeric tables, comes in its own extra, so you can install just what you need, which is handy when building container images:

| Extra | Enables | Brings in |
|-------|---------|-----------|
| `feature-selection` | `feature_selection()` with `sa`, `sb` and `qaoa` | dimod, dwave-samplers, simulated-bifurcation, PyTorch, Qiskit |
| `qml` | `evaluate()` with `qsvc` and `qnn` | Qiskit, Qiskit Machine Learning, PennyLane, PyTorch |
| `clustering` | `cluster("pqc")`, `QuantumClustering` and the circuits of quantum k-means | APQC and TensorFlow (Python 3.13 or lower), Qiskit |
| `tabular` | Tables with categories, text or dates in `preprocess()`, raw table diagnosis in `assess()` and `table_report()` | skrub |
| `all` | All of the above | |

```
pip install "falcondale[clustering]"
pip install "falcondale[feature-selection,qml]"
```

APQC and TensorFlow do not support Python 3.14 yet, so there the `clustering` extra only brings Qiskit and Probabilistic Quantum Clustering (`pqc`) is not available. The other clustering methods work on any version.

Using a technique whose extra is missing raises an error telling you what to install, and `Project.list_options()` shows which ones are available.

!!! tip "CPU only machines"
    PyTorch is installed from PyPI by default, which on Linux brings GPU libraries along. If you do not have a GPU you can save a few gigabytes by installing PyTorch from its CPU index first, check the [PyTorch documentation](https://pytorch.org/get-started/locally/) or the [uv guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## First steps

Once installed, simply invoke the objects that will help you build your QML models.

```py
from falcondale import Project, Model
```

More about the usage can be found in the following usage examples. But we would like to follow with some of the features that Falcondale counts with.
