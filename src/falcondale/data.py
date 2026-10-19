"""
This module implements the data structure that will be used
along any steps of a Falcondale project.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    """
    Dataset class gets the core information needed
    for classical data embedding.
    """

    def __init__(self, input_df: pd.DataFrame, target: str | None = None):
        """
        Initializes Falcondale's core dataset abstraction.

        Args:
            input_df (pd.DataFrame): Input pandas dataframe
            target (str): Target column variable name
        """
        if target is not None and target not in input_df.columns:
            raise KeyError(f"Target column '{target}' not found in the dataframe.")

        self._rawdataset = input_df.reset_index(drop=True)
        self._columns: list[str] = [col for col in input_df.columns if col != target]
        self.target = target

        self._scaler: MinMaxScaler | None = None
        self._dim_reduce: PCA | None = None
        self._features: pd.DataFrame | None = None

    @property
    def features(self) -> pd.DataFrame:
        """Preprocessed features, raising a helpful error if `preprocess` was not called yet."""
        if self._features is None:
            raise RuntimeError("No features available yet! Try calling preprocess() first.")
        return self._features

    def profile(self) -> dict:
        """
        Prints a short summary of the raw dataset highlighting common issues
        and returns it as a dictionary.

        Returns:
            dict: Summary values (samples, columns, missing ratio, ...)
        """
        df = self._rawdataset
        feats = df[self._columns]
        non_numeric = [col for col in feats.columns if not pd.api.types.is_numeric_dtype(feats[col])]
        summary: dict = {
            "samples": len(df),
            "columns": len(df.columns),
            "missing_ratio": float(df.isna().to_numpy().mean()),
            "duplicated_rows": int(df.duplicated().sum()),
            "non_numeric_columns": non_numeric,
            "constant_columns": [col for col in feats.columns if feats[col].nunique(dropna=False) <= 1],
        }
        if self.target:
            summary["target_balance"] = df[self.target].value_counts(normalize=True).to_dict()

        print(f"Number of samples on the dataset: {summary['samples']}")
        print(f"Number of columns on the dataset: {summary['columns']}")
        if self.target:
            print(f"{self.target} has been selected as target variable for supervised learning tasks.")
            balance = ", ".join(f"{k}: {v:.1%}" for k, v in summary["target_balance"].items())
            print(f"Target class balance: {balance}")
        print(f"Ratio of cells with missing values: {summary['missing_ratio']:.2%}")
        if summary["duplicated_rows"]:
            print(f"There are {summary['duplicated_rows']} duplicated rows.")
        if non_numeric:
            print(f"Some features are not numeric, encode them before moving on: {non_numeric}")
        else:
            print("All data is numeric therefore little preprocessing might need to be done.")
        if summary["constant_columns"]:
            print(f"Constant columns carry no information and could be dropped: {summary['constant_columns']}")

        return summary

    def preprocess(self, reduced_dimension: int | None = None, method: str = "pca"):
        """
        Preprocessing of the dataset: fills missing data with zeros and scales
        the features into the [0, 1] range. Optionally, reduces the dimension
        of the feature space using PCA.

        Args:
            reduced_dimension (int): (Optional) Number of dimensions to keep
            method (str): Dimensionality reduction method, only "pca" is supported
        """
        columns = [col for col in self._rawdataset.columns if col != self.target]
        features = self._rawdataset[columns].fillna(0)

        self._scaler = MinMaxScaler().fit(features)
        self._features = pd.DataFrame(self._scaler.transform(features), columns=columns)
        self._columns = columns
        self._dim_reduce = None

        if reduced_dimension:
            if method.lower() != "pca":
                raise ValueError(f"Unknown dimensionality reduction method '{method}', only 'pca' is supported.")
            self._dim_reduce = PCA(n_components=reduced_dimension).fit(self._features)
            self._columns = list(self._dim_reduce.get_feature_names_out())
            self._features = pd.concat([self._features, self._reduce(self._features)], axis=1)

    def _reduce(self, scaled: pd.DataFrame) -> pd.DataFrame:
        assert self._dim_reduce is not None
        return pd.DataFrame(self._dim_reduce.transform(scaled), columns=self._dim_reduce.get_feature_names_out())

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes an input dataframe and transforms it the same way the
        original dataset was treated.

        Args:
            input_df (pd.DataFrame): Dataframe with the same raw columns as the original one

        Returns:
            pd.DataFrame: Dataframe holding the selected features
        """
        if self._scaler is None:
            raise RuntimeError("The dataset has not been preprocessed yet! Try calling preprocess() first.")

        raw_columns = [col for col in self._rawdataset.columns if col != self.target]
        raw = input_df[raw_columns].fillna(0).reset_index(drop=True)
        output = pd.DataFrame(self._scaler.transform(raw), columns=raw_columns)

        if self._dim_reduce is not None:
            output = pd.concat([output, self._reduce(output)], axis=1)

        return output[self._columns]

    def get_target(self) -> pd.Series:
        """
        Obtain data for the target column.

        Returns:
            pd.Series: target values
        """
        if self.target is None:
            raise RuntimeError("No target column has been defined for this dataset.")
        return self._rawdataset[self.target]

    def get_features(self) -> pd.DataFrame:
        """
        Obtain data for the selected feature columns.

        Returns:
            pd.DataFrame: subset of the preprocessed features
        """
        return self.features[self._columns]

    def set_features(self, feat_list: list[str]):
        """
        Sets the subset of features to be used in following steps.

        Args:
            feat_list (list[str]): Feature names
        """
        unknown = set(feat_list) - set(self.features.columns)
        if unknown:
            raise KeyError(f"Unknown features: {sorted(unknown)}")
        self._columns = list(feat_list)

    def columns(self) -> list[str]:
        """
        Returns the selected feature column names as a list.

        Returns:
            list[str]: column names
        """
        return list(self.get_features().columns)

    def size(self) -> tuple[int, int]:
        """
        Returns the shape of the features to be encoded.

        Returns:
            tuple[int, int]: Number of samples and number of features
        """
        return self.get_features().shape

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Selected features and target as numpy arrays.

        Returns:
            tuple[np.ndarray, np.ndarray]: features (samples x features) and target (samples)
        """
        return self.get_features().to_numpy(), self.get_target().to_numpy()

    def train_test_split(
        self, test_size: float = 0.3, random_state: int = 12345
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Splits the dataset into train and test blocks, stratified by the target.

        Args:
            test_size (float): Ratio of samples kept for testing
            random_state (int): Seed for the splitting

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
        target = self.get_target().to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            self.get_features(),
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        return x_train, x_test, y_train, y_test
