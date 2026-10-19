"""
Falcondale trained model wrapper.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset


class Model:
    """
    Model structure is the object class of models
    trained with Falcondale Project's evaluate function.

    Holds information regarding the model but also contains
    required preprocessing steps (normalization, dimensionality reduction,...)
    as well as the metrics obtained against testing data split while training.
    """

    def __init__(self, dataset: Dataset, m_type: str, model, classification_report: str, scores: dict[str, float]):
        """
        Inits the model structure with provided information.

        Args:
            dataset (Dataset): A Falcondale Dataset data structure.
            m_type (str): Model type name associated with the training phase.
                Should coincide with the options available in the evaluate method within Project.
            model (object): A scikit-learn compatible estimator, already trained.
            classification_report (str): Output of sklearn.metrics.classification_report
            scores (dict): A dictionary containing the metrics for the model
        """
        self._data = dataset
        self.type = m_type
        self.estimator = model
        self._creport = classification_report
        self._scores = scores

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict function applied to the provided dataframe. It
        also takes care of shaping it so that it will fit
        in the scheme that the model expects.

        Args:
            data (pd.DataFrame): Pandas DataFrame with the same raw columns used for training

        Returns:
            np.ndarray: Predicted class
        """
        return self.estimator.predict(self._data.transform(data).to_numpy())

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Probability of predicting the positive (last) class applied to the
        provided dataframe. It also takes care of shaping it so
        that it will fit in the scheme that the model expects.

        Args:
            data (pd.DataFrame): Pandas DataFrame with the same raw columns used for training

        Returns:
            np.ndarray: Probability of predicting the positive class
        """
        return self.estimator.predict_proba(self._data.transform(data).to_numpy())[:, -1]

    def list_metrics(self) -> list[str]:
        """
        Returns the names of the metrics computed against the test split during training.

        In a general case will contain:

        * Recall, sensitivity or True Positive Rate
        * Specificity or True Negative Rate (binary problems)
        * Precision or Positive Predictive Value
        * F1 score
        * Accuracy
        * Balanced accuracy
        * AUC

        Returns:
            list[str]: List of available metrics.

        Example:
            ```py
            model.list_metrics()
            ```
        """
        return list(self._scores)

    def metric(self, metric_name: str) -> float:
        """
        Returns the metric value against the test dataset during
        model training.

        Args:
            metric_name (str): Metric name to obtain the value from

        Returns:
            float: Value of the metric

        Example:
            ```py
            model.metric('auc')
            ```
        """
        if metric_name not in self._scores:
            raise KeyError(f"Metric '{metric_name}' not available. Available ones: {self.list_metrics()}")
        return self._scores[metric_name]

    def print_report(self):
        """
        Prints the classification report of the model taken from
        sklearns classification report similar to:

                          precision    recall  f1-score   support

                       0       0.83      0.75      0.79        60
                       1       0.87      0.92      0.89       111

                accuracy                           0.86       171
               macro avg       0.85      0.83      0.84       171
            weighted avg       0.86      0.86      0.86       171
        """
        print(self._creport)

    def save(self, path: str | Path):
        """
        Saves the model under the path provided.

        Args:
            path (str | Path): File name
        """
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Model":
        """
        Loads a Falcondale model from the provided path. Only load files you trust,
        as unpickling can execute arbitrary code.

        Args:
            path (str | Path): File name

        Returns:
            Model: Falcondale Model type
        """
        with open(path, "rb") as handle:
            model = pickle.load(handle)

        if not isinstance(model, Model):
            raise TypeError(f"{path} does not contain a Falcondale model.")
        return model
