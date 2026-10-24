import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from falcondale import Dataset, Project, QMLAssessor
from falcondale.tabular import diagnose, make_encoder, non_numeric_columns, table_report


def customers(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Categories, free text, dates, a constant and a redundant column, and missing values."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "plus", "premium"], n)
    tenure = rng.integers(1, 72, n).astype(float)
    df = pd.DataFrame(
        {
            "plan": plan,
            "plan_code": pd.Series(plan).map({"basic": "B", "plus": "P", "premium": "X"}),
            "city": rng.choice(["Bilbao", "Donostia", "Gasteiz", None], n),
            "tenure": tenure,
            "fee": rng.normal(40, 10, n),
            "signup": (pd.Timestamp("2026-10-01") - pd.to_timedelta(tenure * 30, unit="D")).strftime("%Y-%m-%d"),
            "country": "ES",
            "job": rng.choice([f"role {i}" for i in range(30)], n),
            "churn": (np.cos(np.pi * tenure / 36) + rng.normal(0, 0.5, n) > 0).astype(int),
        }
    )
    df.loc[rng.choice(n, n // 4, replace=False), "fee"] = np.nan
    return df


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return customers()


def test_non_numeric_columns(table):
    assert non_numeric_columns(table) == ["plan", "plan_code", "city", "signup", "country", "job"]


def test_encoder_outputs_informed_numbers(table):
    encoded = make_encoder(string_components=3).fit_transform(table.drop(columns="churn"))

    assert not encoded.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)
    assert "country" not in "".join(encoded.columns)
    assert {"job_0", "job_1", "job_2"} <= set(encoded.columns)
    assert "plan_basic" in encoded.columns


def test_dataset_encodes_non_numeric_columns(table):
    ds = Dataset(table, target="churn")
    ds.preprocess()

    features = ds.get_features()
    assert not features.isna().any().any()
    assert features.min().min() >= 0
    assert features.max().max() <= 1
    pd.testing.assert_frame_equal(ds.transform(table), features)

    # Unseen categories and missing values in new samples are handled too
    new = customers(10, seed=1)
    new.loc[0, "city"] = "Iruña"
    new.loc[1, "tenure"] = np.nan
    assert ds.transform(new).shape == (10, features.shape[1])


def test_dataset_encoding_choices(table, breast_cancer):
    with pytest.raises(ValueError, match="Non numeric"):
        Dataset(table, target="churn").preprocess(encode=False)

    ds = Dataset(table, target="churn")
    ds.preprocess(cardinality_threshold=40)
    assert any(c.startswith("job_role") for c in ds.columns())

    # Numeric tables keep filling gaps with zeros unless encoding is forced
    numeric = breast_cancer.copy()
    numeric.iloc[0, 0] = np.nan
    plain, forced = Dataset(numeric, target="target"), Dataset(numeric, target="target")
    plain.preprocess()
    forced.preprocess(encode=True)
    assert plain.get_features().iloc[0, 0] == 0
    assert forced.get_features().iloc[0, 0] > 0


def test_diagnose(table):
    diagnosis = diagnose(table.drop(columns="churn"), table["churn"])
    columns = {c.name: c for c in diagnosis.columns}

    assert columns["country"].kind == "dropped"
    assert columns["country"].features == 0
    assert columns["signup"].kind == "datetime"
    assert columns["job"].kind == "high_cardinality"
    assert columns["fee"].missing == pytest.approx(0.25)
    assert (columns["tenure"].target_association or 0) > (columns["city"].target_association or 0)
    assert ("plan", "plan_code") in [pair[:2] for pair in diagnosis.redundant] or (
        "plan_code",
        "plan",
    ) in [pair[:2] for pair in diagnosis.redundant]
    encoded = make_encoder().fit_transform(table.drop(columns="churn"))
    assert diagnosis.features == encoded.shape[1]

    hints = "\n".join(diagnosis.hints())
    assert "country" in hints
    assert "'fee' misses 25%" in hints
    assert "redundant" in hints
    assert "date parts" in hints


def test_diagnose_without_target(table):
    diagnosis = diagnose(table.drop(columns="churn"))
    assert all(c.target_association is None for c in diagnosis.columns)
    assert diagnosis.to_dict()["features"] == diagnosis.features


def test_project_assessment_includes_raw_table(table):
    project = Project(table, target="churn")
    project.preprocess()
    assessment = project.assess()

    assert assessment.table_ is not None
    assert assessment.table_.features == len(project.dataset.columns())
    assert any("redundant" in hint for hint in assessment.advice())
    assert assessment.to_dict()["table"]["samples"] == len(table)

    output = io.StringIO()
    assessment.report(Console(file=output, width=120))
    assert "Raw table" in output.getvalue()


def test_project_predicts_raw_rows(table):
    project = Project(table, target="churn")
    project.preprocess()
    model = project.evaluate("svc")
    assert model.predict(customers(15, seed=2)).shape == (15,)


def test_assessor_encodes_dataframes(table):
    assessor = QMLAssessor(kernel=False).fit(table.drop(columns="churn"), table["churn"])
    assert assessor.table_ is not None
    assert assessor.n_features_in_ == assessor.table_.features

    skipped = QMLAssessor(kernel=False, diagnose=False).fit(table.drop(columns="churn"), table["churn"])
    assert skipped.table_ is None


def test_table_report(table):
    project = Project(table, target="churn")
    report = project.table_report(title="Customers")
    assert "Customers" in report.html_snippet()
    assert table_report(table).dataframe.shape == table.shape
