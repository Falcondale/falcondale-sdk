"""
Generates `four_profiles.csv`, a synthetic cohort with four well separated profiles.

Every row is simulated. Each profile raises a different block of the 30 ordinal
measurements (0 to 5 scales), so a clustering method should recover the four
groups. A `split` column holds a 70/30 train/test partition.

    uv run python tests/data/make_four_profiles.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 101
SAMPLES = 240
FEATURES = 30
PROFILES = 4


def make_four_profiles(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profile = rng.integers(0, PROFILES, SAMPLES)

    # Profile p raises the measurements of block p, the last profile stays mild everywhere
    means = np.full((PROFILES, FEATURES), 0.8)
    block = FEATURES // (PROFILES - 1)
    for p in range(PROFILES - 1):
        means[p, p * block : (p + 1) * block] = 3.8

    values = rng.normal(means[profile], 1.6)
    df = pd.DataFrame(np.clip(np.rint(values), 0, 5).astype(int), columns=[f"f{i:02d}" for i in range(FEATURES)])
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(SAMPLES)])
    df["profile"] = profile

    train, _ = train_test_split(np.arange(SAMPLES), test_size=0.3, random_state=seed, stratify=profile)
    df["split"] = np.where(np.isin(np.arange(SAMPLES), train), "train", "test")
    return df


if __name__ == "__main__":
    make_four_profiles().to_csv(Path(__file__).with_name("four_profiles.csv"), index=False)
