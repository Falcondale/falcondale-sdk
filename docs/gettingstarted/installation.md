# Installation

Falcondale is a Python library (3.11 to 3.13) that can be installed with the usual tools.

```
pip install falcondale
```

We encourage you to use [uv](https://docs.astral.sh/uv/) as it counts with a robust and fast dependency solving mechanism.

```
uv add falcondale
```

Falcondale installation will also install all additional dependencies needed, in particular those related to quantum computing frameworks (Qiskit, PennyLane) and their compatible versions. This can make it a little bit heavy during installation or initial loading of the classes indicated below (depending on your machine mostly), so we ask you for a little bit of patience at this stage.

!!! tip "CPU only machines"
    PyTorch is installed from PyPI by default, which on Linux brings GPU libraries along. If you do not have a GPU you can save a few gigabytes by installing PyTorch from its CPU index first, check the [PyTorch documentation](https://pytorch.org/get-started/locally/) or the [uv guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Optional dependencies

Quantum clustering relies on TensorFlow, which is why it is not installed by default.

```
pip install "falcondale[clustering]"
```

## First steps

Once installed, simply invoke the objects that will help you build your QML models.

```py
from falcondale import Project, Model
```

More about the usage can be found in the following usage examples. But we would like to follow with some of the features that Falcondale counts with.
