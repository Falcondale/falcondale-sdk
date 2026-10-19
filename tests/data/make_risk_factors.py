"""
Generates `risk_factors.csv`, a synthetic cohort to check feature selection.

Every row is simulated. Six causal factors of different kinds (binary, ordinal, skewed counts
and continuous measurements) drive a binary `diagnosis` and a continuous `progress` outcome.
The remaining candidates are noisy proxies of some causal factors, which carry information about
the outcomes only through them, and pure noise, including `region`, a categorical code with 17
levels that biased mutual information estimates tend to rank high. A `split` column holds a
70/30 train/test partition.

    uv run python tests/data/make_risk_factors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 101
SAMPLES = 1000

CAUSAL = ["age", "smoker", "severity", "hospital_days", "biomarker_a", "biomarker_b"]
PROXIES = ["age_reported", "severity_score", "biomarker_a_lab2", "visits"]
OUTCOMES = ["diagnosis", "progress"]


def make_risk_factors(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = SAMPLES

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std()

    causal = {
        "age": np.rint(rng.normal(55, 12, n)).clip(18, 95),
        "smoker": rng.binomial(1, 0.3, n),
        "severity": rng.choice(5, n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        # Mostly zeros with a long tail
        "hospital_days": rng.binomial(1, 0.25, n) * rng.geometric(0.15, n),
        "biomarker_a": rng.lognormal(0, 0.6, n).round(3),
        "biomarker_b": rng.normal(0, 1, n).round(3),
    }
    risk = sum(z(np.log1p(v) if k in {"hospital_days", "biomarker_a"} else v) for k, v in causal.items())

    proxies = {
        "age_reported": np.rint(causal["age"] + rng.normal(0, 10, n)),
        "severity_score": (causal["severity"] + rng.integers(-1, 2, n) * rng.binomial(1, 0.6, n)).clip(0, 4),
        "biomarker_a_lab2": (causal["biomarker_a"] * rng.lognormal(0, 0.5, n)).round(3),
        "visits": rng.poisson(1 + causal["hospital_days"] / 3),
    }

    noise = {
        "region": rng.integers(0, 17, n),
        "sex": rng.binomial(1, 0.5, n),
        "income_band": rng.integers(0, 5, n),
        "height": rng.normal(170, 9, n).round(1),
        "weight": rng.normal(75, 12, n).round(1),
        "coffee_cups": rng.poisson(2, n),
        "steps_per_day": rng.lognormal(8.5, 0.5, n).round(),
        "sleep_hours": rng.normal(7, 1, n).round(1),
        "allergies": rng.binomial(1, 0.15, n),
        "biomarker_c": rng.normal(0, 1, n).round(3),
        "biomarker_d": rng.exponential(1, n).round(3),
        "biomarker_e": rng.uniform(0, 1, n).round(3),
        "questionnaire": rng.integers(0, 11, n),
        "days_since_visit": rng.binomial(1, 0.2, n) * rng.geometric(0.05, n),
    }

    df = pd.DataFrame(causal | proxies | noise)
    df["diagnosis"] = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * risk - 0.5))))
    df["progress"] = (0.5 * risk + rng.normal(0, 1.5, n)).round(3)

    train, _ = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=df["diagnosis"])
    df["split"] = np.where(np.isin(np.arange(n), train), "train", "test")
    return df


if __name__ == "__main__":
    make_risk_factors().to_csv(Path(__file__).with_name("risk_factors.csv"), index=False)
