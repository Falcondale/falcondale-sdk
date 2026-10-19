import pandas as pd
import pytest
from sklearn import datasets

from falcondale import Project


@pytest.fixture(scope="session")
def breast_cancer() -> pd.DataFrame:
    """Breast cancer dataset reduced to a size that keeps quantum simulations fast."""
    features, target = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    dataset = features.iloc[:, :8].copy()
    dataset["target"] = target
    return dataset.sample(160, random_state=0).reset_index(drop=True)


@pytest.fixture
def project(breast_cancer: pd.DataFrame) -> Project:
    project = Project(breast_cancer, target="target")
    project.preprocess()
    return project
