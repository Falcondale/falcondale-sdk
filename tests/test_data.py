import numpy as np
import pandas as pd
import pytest

from falcondale import Dataset


def test_unknown_target_raises(breast_cancer):
    with pytest.raises(KeyError):
        Dataset(breast_cancer, target="missing")


def test_features_need_preprocessing(breast_cancer):
    with pytest.raises(RuntimeError, match="preprocess"):
        Dataset(breast_cancer, target="target").get_features()


def test_preprocess_scales_and_fills_missing(breast_cancer):
    df = breast_cancer.copy()
    df.iloc[0, 0] = np.nan
    ds = Dataset(df, target="target")
    ds.preprocess()

    features = ds.get_features()
    assert "target" not in features.columns
    assert not features.isna().any().any()
    assert features.min().min() >= 0
    assert features.max().max() <= 1


def test_pca_reduction(breast_cancer):
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess(reduced_dimension=3)
    assert ds.columns() == ["pca0", "pca1", "pca2"]
    assert ds.size() == (len(breast_cancer), 3)


def test_unknown_reduction_method(breast_cancer):
    with pytest.raises(ValueError):
        Dataset(breast_cancer, target="target").preprocess(reduced_dimension=2, method="lda")


@pytest.mark.parametrize("reduced_dimension", [None, 3])
def test_transform_matches_training_features(breast_cancer, reduced_dimension):
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess(reduced_dimension=reduced_dimension)
    pd.testing.assert_frame_equal(ds.transform(breast_cancer), ds.get_features())


def test_set_features(breast_cancer):
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess()
    ds.set_features(["mean radius", "mean area"])
    assert ds.columns() == ["mean radius", "mean area"]
    with pytest.raises(KeyError):
        ds.set_features(["unknown"])


def test_train_test_split_is_stratified(breast_cancer):
    ds = Dataset(breast_cancer, target="target")
    ds.preprocess()
    _, x_test, _, y_test = ds.train_test_split(test_size=0.25)
    assert len(x_test) == len(y_test) == 40
    assert abs(y_test.mean() - breast_cancer["target"].mean()) < 0.05


def test_profile(breast_cancer, capsys):
    summary = Dataset(breast_cancer, target="target").profile()
    assert summary["samples"] == len(breast_cancer)
    assert summary["non_numeric_columns"] == []
    assert "target" in capsys.readouterr().out
