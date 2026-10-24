"""
This is Falcondale's core package.

Classes:
    Project: The main class that one can instantiate in order to
        start playing around with Falcondale's functionalities

        $ myproject = Project(dataset, target="target")
"""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .assessment import QMLAssessor
from .classifiers import MODELS, train_and_evaluate
from .clustering import CLUSTERING
from .data import Dataset
from .feature_selection import SOLVERS, select_features
from .model import Model
from .tabular import table_report

# A module from each extra, used to tell which ones are installed
EXTRA_PROBES = {
    "feature-selection": "dimod",
    "qml": "qiskit_machine_learning",
    "clustering": "apqc",
    "tabular": "skrub",
}


# Methods needing an extra, when only some of a task do (the rest run on the core dependencies)
NEEDS_EXTRA = {"clustering": ["pqc"]}


def _installed(extra: str) -> bool:
    return importlib.util.find_spec(EXTRA_PROBES[extra]) is not None


class Project:
    """
    Base class defining the set of steps to be performed for a given project.

    Wraps complex functionality calls and framework integrations so that it is
    simpler for the user to call specific training or data processing functions.
    """

    def __init__(self, input_data: pd.DataFrame, target: str | None = None):
        """
        Initializes the Project class to perform evaluations on different QML methods.

        Feature selection and classification are supervised tasks, so they need a target
        variable. Clustering does not, so the target can be left out for unsupervised projects.

        Args:
            input_data (pd.DataFrame): Pandas DataFrame
            target (str): (Optional) Target variable to be selected from the
                input_data dataframe
        """
        self._data = Dataset(input_data, target)
        self._assessment: QMLAssessor | None = None
        self._assessed_columns: list[str] = []

    def _require_target(self, task: str):
        if self._data.target is None:
            raise ValueError(f"{task} is a supervised task, create the project with Project(data, target=...).")

    @property
    def dataset(self) -> Dataset:
        """The Falcondale Dataset used by the project."""
        return self._data

    def preprocess(
        self, reduced_dimension: int | None = None, method: str = "pca", encode: bool | str = "auto", **kwargs: Any
    ):
        """
        Performs information preprocessing to clean the dataset. This is a mandatory
        step before feature selection, training or clustering.

        Mostly focused on data normalization and gap filling (missing values). Non numeric columns
        (categories, free text, dates) are encoded with skrub, which needs `falcondale[tabular]`. It also allows
        for simple dimensionality reduction by indicating the reduced dimension size. This feature
        is mainly used when specific models are selected as large feature spaces may
        require more computational resources than the ones available on a common laptop.

        Args:
            reduced_dimension (int): (Optional) Reduces initial feature set to a lower
                dimensional space.
            method (str): (Optional) Currently only supports PCA in combination with
                reduced_dimension parameter.
            encode (bool | str): (Optional) "auto" encodes with skrub only when there are non numeric
                columns, True always does (median imputation for numeric tables), False never does
            **kwargs: Encoding options, such as `cardinality_threshold` or `string_components`
                (see `falcondale.tabular.make_encoder`)

        Example:
            ```py
            myproject.preprocess(reduced_dimension=3)
            myproject.preprocess(cardinality_threshold=5)  # one hot encode up to 5 categories
            ```
        """
        self._data.preprocess(reduced_dimension=reduced_dimension, method=method, encode=encode, **kwargs)

    def profile_dataset(self) -> dict:
        """
        Profiles the dataset in order to identify potential issues with it
        (missing values, duplicates, non numeric or constant columns, class balance).

        Returns:
            dict: Summary of the dataset
        """
        return self._data.profile()

    def table_report(self, **kwargs: Any):
        """
        Interactive report of the raw dataset (distributions, missing values, associations between
        columns) built with skrub, which needs `falcondale[tabular]`. Renders in notebooks, call
        `.open()` on it to see it in the browser.

        Args:
            **kwargs: `skrub.TableReport` arguments, such as `title`

        Returns:
            skrub.TableReport: The report
        """
        return table_report(self._data.raw_features().assign(**self._target_column()), **kwargs)

    def _target_column(self) -> dict[str, pd.Series]:
        return {self._data.target: self._data.get_target()} if self._data.target else {}

    def show_features(self) -> pd.DataFrame:
        """
        Returns the features as they are internally used by the framework.

        Returns:
            pd.DataFrame: Pandas dataframe with the actual features being used within the Framework
        """
        return self._data.get_features()

    def feature_selection(self, max_cols: int, method: str = "sa", **kwargs: Any) -> list[str]:
        """
        Performs the quantum feature selection to reduce the columns to be used, selecting
        the obtained features as the ones to be used in following steps.

        Currently supports:

        * **sa**: For Simulated Annealing
        * **sb**: For Simulated Bifurcation
        * **qaoa**: For Quantum Approximate Optimization Algorithm (simulated)
        * **exact**: For the exact solution, enumerating every selection (small problems)

        Examples:
            Can be called from Project instance:

            ```py
            features = myproject.feature_selection(max_cols=5, method="sb")
            features = myproject.feature_selection(max_cols=3, method="qaoa", pool=12)
            ```

        Args:
            max_cols (int): Maximum number of features to be selected
            method (str): (default sa) Method to be used when trying to find the
                optimal set of features
            **kwargs: QUBO options (`formulation`, `bins`, `strategy`, `pool`, see
                `falcondale.feature_selection.select_features`) and solver specific ones such as `seed`

        Returns:
            list[str]: The list of selected features.

            These are also the features that will be included in the project as the current
            selection of features for following steps within the project.
        """
        self._require_target("Feature selection")
        if max_cols > len(self._data.columns()):
            raise ValueError(
                f"Cannot select {max_cols} features out of {len(self._data.columns())}. "
                "Preprocess the dataset again to start over from all features."
            )

        feature_cols = select_features(self._data, max_cols=max_cols, method=method, **kwargs)
        self._data.set_features(feature_cols)

        return feature_cols

    def assess(self, n_qubits: int | None = None, **kwargs: Any) -> QMLAssessor:
        """
        Assesses how well suited the current features are for quantum machine learning
        and ranks the models available in `evaluate` accordingly.

        The assessment looks at how fast the target oscillates along each feature (what angle
        encoded circuits have to express), how many of those frequencies are trainable, the
        circuit depth needed, how hard the task is for classical baselines, the class balance
        and the qubit budget. See `falcondale.assessment.QMLAssessor` for the details.

        With `falcondale[tabular]` installed the raw table is diagnosed too: how many features each
        column becomes, missing values, redundant columns and columns unrelated to the target.

        The result is kept by the project, so that `evaluate("auto")` trains the most suitable model.

        Examples:
            Can be called from Project instance:

            ```py
            assessment = myproject.assess(n_qubits=8)
            assessment.report()
            assessment.best_model
            ```

        Args:
            n_qubits (int): (Optional) Qubit budget, 16 by default (the statevector limit of QSVC)
            **kwargs: Additional `QMLAssessor` options, such as `max_samples` for large datasets

        Returns:
            QMLAssessor: The fitted assessor holding scores, diagnostics and model recommendations
        """
        self._require_target("Assessment")
        if n_qubits is not None:
            kwargs["n_qubits"] = n_qubits
        self._assessment = QMLAssessor(**kwargs).fit(
            self._data.get_features(),
            self._data.get_target(),
            table=self._data.raw_features(),
            encoder_options=self._data.encoder_options,
        )
        self._assessed_columns = self._data.columns()
        return self._assessment

    @property
    def assessment(self) -> QMLAssessor | None:
        """Latest assessment of the project, None until `assess` is called."""
        return self._assessment

    @staticmethod
    def list_options() -> dict[str, list[str]]:
        """
        Lists what can be done with Falcondale. A simple hint on the available options.

        Returns:
            dict[str, list[str]]: Available methods per task
        """
        options = {
            "feature_selection": sorted(SOLVERS),
            "evaluate": sorted(MODELS),
            "cluster": sorted(CLUSTERING),
        }
        tasks = [
            ("feature-selection", "Perform Quantum Feature Selection using", options["feature_selection"]),
            ("qml", "Train quantum-enhanced models such as", options["evaluate"]),
            ("clustering", "Perform Quantum Clustering using", options["cluster"]),
        ]
        print("Welcome to Falcondale SDK! you will be able to:")
        for extra, text, methods in tasks:
            needed = [m for m in methods if m in NEEDS_EXTRA.get(extra, methods)]
            hint = "" if _installed(extra) else f' ({", ".join(needed)} need: pip install "falcondale[{extra}]")'
            print(f" - {text}: {', '.join(methods)}{hint}")
        print(' - Assess how suited your data is for QML with assess(), then evaluate("auto") picks the best model')
        return options

    def evaluate(self, model: str, test_size: float = 0.3, **kwargs: Any) -> Model:
        """
        Given a model type, trains it and evaluates its performance
        for the initially given dataset.

        Currently supports:

        * **svc**: Classical Support Vector Classifier (baseline)
        * **qsvc**: For Quantum Support Vector Classifier (`backend="qiskit"` or `backend="pennylane"`)
        * **qnn**: For Quantum Neural Network (`layers`, `epochs`, `learning_rate`,...)
        * **auto**: The most suitable model according to `assess`, which is run first if the
          project has not been assessed yet or the features changed since

        Args:
            model (str): Model name to be trained
            test_size (float): (Optional) Test split size ratio. 30% default
            **kwargs: Model specific options, such as `verbose`

        Returns:
            Model: A Falcondale Model instance holding the trained model
                and some additional information

        Examples:
            Can be called from Project instance:

            ```py
            model = myproject.evaluate("qsvc", backend="pennylane")
            model = myproject.evaluate("auto")
            ```
        """
        self._require_target("Classification")
        model = model.lower()
        if model == "auto":
            if self._assessment is None or self._assessed_columns != self._data.columns():
                self.assess()
            assert self._assessment is not None
            best = self._assessment.best_model
            print(f"Assessment recommends {best.name}: {best.reasons[0]}")
            model, kwargs = best.model, {**best.options, **kwargs}
        estimator, report, metrics = train_and_evaluate(model, self._data, test_size=test_size, **kwargs)
        return Model(self._data, model, estimator, report, metrics)

    def cluster(self, ctype: str = "pqc", **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a type of clustering it is implemented using the
        information of the inner Dataset. Essentially takes the features
        to produce an estimate on the different segments it could be composed by.

        Currently supports:

        * **pqc**: For Probabilistic Quantum Clustering (needs `falcondale[clustering]`)
        * **qc**: For Horn and Gottlieb Quantum Clustering
        * **qkmeans**: For quantum k-means, distances estimated by interference (`n_clusters`,
          `center="median"` for k-medians, `shots`)

        The whole dataset is clustered. To fit on a train split and assign unseen samples,
        use the estimators in `falcondale.clustering` (`QuantumClustering`, `SchrodingerClustering`,
        `QuantumKMeans`) or `make_clustering_pipeline` directly.

        Examples:
            Can be called from Project instance:

            ```py
            label, label_prob = myproject.cluster(ctype="pqc", random_state=42)
            label, membership = myproject.cluster(ctype="qkmeans", n_clusters=3, random_state=42)
            ```

        Args:
            ctype (str): Type of clustering to be performed
            **kwargs: Method specific options, such as `sigma`, `bandwidths` or `random_state`
                (see `QuantumClustering`, `SchrodingerClustering` and `QuantumKMeans`)

        Returns:
            (np.ndarray, np.ndarray): A tuple containing the assigned cluster label
                and its likelihood or label probability.
        """
        ctype = ctype.lower()
        if ctype not in CLUSTERING:
            raise ValueError(f"Unknown clustering method '{ctype}'. Available ones: {sorted(CLUSTERING)}")

        return CLUSTERING[ctype](self._data, **kwargs)

    @staticmethod
    def save_model(model: Model, name: str | Path):
        """
        Saves the model under the filename provided. Same as `model.save(name)`.

        Args:
            model (Model): A Falcondale model type of object
            name (str | Path): File name
        """
        model.save(name)

    @staticmethod
    def load_model(name: str | Path) -> Model:
        """
        Loads a Falcondale type of model from the provided file name. Same as `Model.load(name)`.

        Args:
            name (str | Path): File name

        Returns:
            Model: Falcondale Model type
        """
        return Model.load(name)
