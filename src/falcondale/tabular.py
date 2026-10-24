"""
Module dealing with real world tables: non numeric columns, dates and missing values.

Quantum models need every feature to be a number, and each feature usually costs a qubit.
[skrub](https://skrub-data.org/) turns heterogeneous dataframes into numeric ones:

* low cardinality categories are one hot encoded
* high cardinality strings (free text, job titles,...) are compressed into a few components
* dates are split into numeric parts
* constant and empty columns are dropped

`make_encoder` chains that with the imputation of missing values, and `diagnose` looks at the
raw table before encoding, telling how many features (qubits) each column will become, which
columns are redundant and which ones are not related to the target.

Requires the optional `tabular` dependencies: `pip install "falcondale[tabular]"`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rich.console import Group
from rich.table import Table
from rich.text import Text
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from ._optional import optional_dependencies

# Columns above these thresholds are worth a hint
MISSING_WARNING = 0.2
REDUNDANT_ASSOCIATION = 0.9
UNRELATED_ASSOCIATION = 0.05
EXPANSION_WARNING = 6

KIND_LABELS = {"low_cardinality": "category", "high_cardinality": "text", "datetime": "date"}


def available() -> bool:
    """Whether the optional `tabular` dependencies are installed."""
    return importlib.util.find_spec("skrub") is not None


def non_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are neither numeric nor boolean."""
    return [str(col) for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]


def make_encoder(cardinality_threshold: int = 10, string_components: int = 4, **kwargs) -> Pipeline:
    """
    Pipeline turning a heterogeneous dataframe into a fully informed numeric one.

    A skrub `TableVectorizer` encodes every column according to its kind, then missing values
    left in numeric columns are filled with the median. The output keeps column names
    (`city_Bilbao`, `signup_year`, `job_0`,...).

    The defaults are more compact than skrub's, as every feature usually becomes a qubit.

    Args:
        cardinality_threshold (int): Columns with fewer unique values are one hot encoded,
            the others are compressed into `string_components` components
        string_components (int): Components kept for high cardinality string columns
        **kwargs: Additional `skrub.TableVectorizer` arguments, such as `datetime` or `specific_transformers`

    Returns:
        Pipeline: Unfitted scikit-learn pipeline
    """
    with optional_dependencies("tabular"):
        from skrub import StringEncoder, TableVectorizer

    kwargs.setdefault("drop_if_constant", True)
    kwargs.setdefault("high_cardinality", StringEncoder(n_components=string_components))
    vectorizer = TableVectorizer(cardinality_threshold=cardinality_threshold, **kwargs)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return make_pipeline(vectorizer, imputer).set_output(transform="pandas")


@dataclass(frozen=True)
class ColumnDiagnosis:
    """
    Summary of a raw column.

    Attributes:
        name (str): Column name
        kind (str): How it is encoded: numeric, datetime, low_cardinality (one hot), high_cardinality
            (compressed strings) or dropped
        missing (float): Ratio of missing values
        unique (int): Number of distinct values
        features (int): Number of numeric features it becomes after encoding
        target_association (float | None): Cramér's V with the target, None without target
    """

    name: str
    kind: str
    missing: float
    unique: int
    features: int
    target_association: float | None


@dataclass
class TableDiagnosis:
    """
    Diagnosis of a raw table before encoding.

    Attributes:
        columns (list[ColumnDiagnosis]): One entry per column, target excluded
        redundant (list[tuple[str, str, float]]): Pairs of strongly associated columns and their Cramér's V
        samples (int): Number of rows
    """

    columns: list[ColumnDiagnosis]
    redundant: list[tuple[str, str, float]] = field(default_factory=list)
    samples: int = 0

    @property
    def features(self) -> int:
        """Number of numeric features after encoding."""
        return sum(col.features for col in self.columns)

    def hints(self) -> list[str]:
        """
        Practical hints about the raw table.

        Returns:
            list[str]: One hint per line
        """
        hints = []
        dropped = [c.name for c in self.columns if c.kind == "dropped"]
        if dropped:
            hints.append(f"Constant or empty columns are dropped: {', '.join(dropped)}.")
        for col in self.columns:
            if col.missing >= MISSING_WARNING:
                hints.append(f"'{col.name}' misses {col.missing:.0%} of its values, imputation may blur its signal.")
        for col in self.columns:
            if col.kind == "datetime" and col.features > 1:
                hints.append(
                    f"'{col.name}' becomes {col.features} date parts, keeping the relevant ones "
                    "(or a single duration) would save qubits."
                )
            elif col.kind == "high_cardinality":
                hints.append(
                    f"'{col.name}' has {col.unique} distinct values compressed into {col.features} features, "
                    "grouping them into fewer categories keeps them interpretable."
                )
            elif col.kind == "low_cardinality" and col.features >= EXPANSION_WARNING:
                hints.append(
                    f"'{col.name}' is one hot encoded into {col.features} features, "
                    "grouping rare categories would save qubits."
                )
        for left, right, value in self.redundant:
            hints.append(f"'{left}' and '{right}' are redundant (Cramér's V {value:.2f}), keeping one is enough.")
        unrelated = [
            c.name
            for c in self.columns
            if c.kind != "dropped" and c.target_association is not None and c.target_association < UNRELATED_ASSOCIATION
        ]
        if unrelated:
            hints.append(f"Barely related to the target: {', '.join(unrelated)}.")
        return hints

    def to_dict(self) -> dict:
        """The diagnosis as plain Python types."""
        return {
            "samples": self.samples,
            "features": self.features,
            "columns": [vars(c) for c in self.columns],
            "redundant": [list(pair) for pair in self.redundant],
        }

    def __rich__(self) -> Group:
        table = Table(
            title=f"Raw table: {len(self.columns)} columns become {self.features} features",
            title_justify="left",
            expand=True,
        )
        table.add_column("Column", style="bold", overflow="ellipsis", max_width=24)
        table.add_column("Kind", no_wrap=True)
        table.add_column("Missing", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Features", justify="right")
        table.add_column("Target assoc.", justify="right")
        for col in self.columns:
            missing_style = "red" if col.missing >= MISSING_WARNING else "yellow" if col.missing else "dim"
            features_style = "yellow" if col.features >= EXPANSION_WARNING else ""
            assoc = col.target_association
            assoc_text = (
                Text("-", style="dim")
                if assoc is None
                else Text(
                    f"{assoc:.2f}", style="dim" if assoc < UNRELATED_ASSOCIATION else "green" if assoc >= 0.3 else ""
                )
            )
            table.add_row(
                col.name,
                Text(KIND_LABELS.get(col.kind, col.kind), style="dim strike" if col.kind == "dropped" else "cyan"),
                Text(f"{col.missing:.0%}", style=missing_style),
                str(col.unique),
                Text(str(col.features), style=features_style),
                assoc_text,
            )
        return Group(table)


def diagnose(df: pd.DataFrame, target: pd.Series | np.ndarray | None = None, **encoder_options) -> TableDiagnosis:
    """
    Looks at a raw table before encoding it.

    Args:
        df (pd.DataFrame): Raw features, target excluded
        target (pd.Series | np.ndarray): (Optional) Target values, to measure how related each column is to it
        **encoder_options: `make_encoder` options, so that feature counts match the actual encoding

    Returns:
        TableDiagnosis: Per column summary and redundant pairs
    """
    with optional_dependencies("tabular"):
        from skrub import column_associations

    df = df.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    vectorizer = make_encoder(**encoder_options).fit(df)[0]
    # Dropped columns may still be listed with their (discarded) outputs
    outputs = set(vectorizer.all_outputs_)

    with_target = df.copy()
    target_name = "__target__"
    if target is not None:
        with_target[target_name] = np.asarray(target)
    associations = column_associations(with_target) if with_target.shape[1] > 1 else pd.DataFrame()

    def association(left: str, right: str) -> float:
        rows = associations[
            ((associations.left_column_name == left) & (associations.right_column_name == right))
            | ((associations.left_column_name == right) & (associations.right_column_name == left))
        ]
        return float(rows.cramer_v.iloc[0]) if len(rows) else 0.0

    columns = [
        ColumnDiagnosis(
            name=col,
            kind=vectorizer.column_to_kind_.get(col, "dropped") if col in vectorizer.input_to_outputs_ else "dropped",
            missing=float(df[col].isna().mean()),
            unique=int(df[col].nunique()),
            features=sum(out in outputs for out in vectorizer.input_to_outputs_.get(col, [])),
            target_association=association(col, target_name) if target is not None else None,
        )
        for col in df.columns
    ]
    redundant = []
    if len(associations):
        strong = associations[
            (associations.cramer_v >= REDUNDANT_ASSOCIATION)
            & (associations.left_column_name != target_name)
            & (associations.right_column_name != target_name)
        ]
        redundant = [
            (str(row.left_column_name), str(row.right_column_name), float(row.cramer_v)) for row in strong.itertuples()
        ]
    return TableDiagnosis(columns=columns, redundant=redundant, samples=len(df))


def table_report(df: pd.DataFrame, **kwargs):
    """
    Interactive skrub report of a dataframe: distributions, missing values and associations.
    Renders in notebooks, `.open()` shows it in the browser.

    Args:
        df (pd.DataFrame): Any dataframe
        **kwargs: `skrub.TableReport` arguments, such as `title`

    Returns:
        skrub.TableReport: The report
    """
    with optional_dependencies("tabular"):
        from skrub import TableReport

    return TableReport(df, **kwargs)
