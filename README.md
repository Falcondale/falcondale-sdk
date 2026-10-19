![falcondale](https://github.com/Falcondale/falcondale-sdk/blob/main/assets/falcondale_sdk_black.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/falcondale)](https://pypi.org/project/falcondale/)
[![Python](https://img.shields.io/pypi/pyversions/falcondale)](https://pypi.org/project/falcondale/)
[![CI](https://github.com/Falcondale/falcondale-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Falcondale/falcondale-sdk/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/falcondale/badge/?version=latest)](https://falcondale.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Falcondale SDK is an open source project that lowers the barrier for computer scientists and data scientists entering the field of Quantum Machine Learning. Bring a pandas DataFrame and try quantum feature selection, quantum classifiers and quantum clustering with a few lines of code, no quantum computing background or cloud account needed. Everything runs locally on simulators.

## Installation

```bash
pip install "falcondale[all]"
# or
uv add "falcondale[all]"
```

The core only depends on numpy, pandas, scikit-learn and scipy. Each family of techniques has its own extra (`feature-selection`, `qml` and `clustering`), so you can install just the one you need, for example `pip install "falcondale[clustering]"`.

## Quickstart

```python
from sklearn import datasets
from falcondale import Project

X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
data = X.assign(target=y)

project = Project(data, target="target")
project.profile_dataset()  # spot missing values, constant columns, class imbalance...
project.preprocess()  # fill gaps and scale features

# Pick the 3 most informative and least redundant features solving a QUBO
features = project.feature_selection(max_cols=3, method="sa")

# Train a Quantum Support Vector Classifier and compare it with a classical one
model = project.evaluate("qsvc")
baseline = project.evaluate("svc")
print(model.metric("auc"), baseline.metric("auc"))

model.print_report()
model.save("qsvc.pkl")
```

## What is inside

| Task | Methods | Built on |
|------|---------|----------|
| Feature selection | `sa` simulated annealing, `sb` simulated bifurcation, `qaoa`, `exact` | dimod, dwave-samplers, simulated-bifurcation, Qiskit |
| Classification | `qsvc` (Qiskit or PennyLane backend), `qnn`, `svc` classical baseline | Qiskit Machine Learning, PennyLane, PyTorch, scikit-learn |
| Clustering | `pqc` probabilistic quantum clustering, `qc` Horn and Gottlieb quantum clustering, `qkmeans` quantum k-means and k-medians | APQC (`pqc`), NumPy |

Every model follows the scikit-learn estimator API, so you can also use them in your own pipelines. Run `Project.list_options()` to see what is available.

Quantum clustering can be fitted on a train split and assign unseen samples with their membership probabilities:

```python
from falcondale.clustering import make_clustering_pipeline

pipeline = make_clustering_pipeline(components=6, random_state=101)  # imputation, scaling, PCA, APQC
pipeline.fit(train)
labels, probabilities = pipeline.predict(test), pipeline.predict_proba(test)
```

## Documentation

The official documentation can be found at https://falcondale.readthedocs.io/en/latest/, including explanations of each technique and example notebooks.

## Contributing

Contributions are welcome! Check the [contributing guide](CONTRIBUTING.md) to set up the development environment (uv, ruff, ty, pytest, pre-commit and commitizen) and the [open issues](https://github.com/Falcondale/falcondale-sdk/issues) for ideas. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/Falcondale/falcondale-sdk.git
cd falcondale-sdk
uv sync --all-extras
uv run pre-commit install --install-hooks
uv run pytest -m "not slow"
```

## Authors

Falcondale SDK is developed and maintained by Falcondale. Get in touch at [info@falcondale.pro](mailto:info@falcondale.pro).

## License

This project is licensed under the terms of the [MIT license](LICENSE).
