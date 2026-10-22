"""
Pytest configuration and shared fixtures.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@pytest.fixture
def machine_data():
    """Load manufacturing demo dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "machine_data.csv"))


@pytest.fixture
def retail_data():
    """Load retail demo dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "retail_demand.csv"))


@pytest.fixture
def churn_data():
    """Load customer churn demo dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "customer_churn.csv"))


@pytest.fixture
def small_classification_df():
    """A small 35-row imbalanced binary classification dataset."""
    rng = np.random.default_rng(99)
    n = 35
    # 92% class 0, 8% class 1 -> ~3 positive samples
    labels = np.zeros(n, dtype=int)
    labels[:3] = 1
    rng.shuffle(labels)
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(5, 2, n),
        "feat_c": rng.choice(["X", "Y", "Z"], n),
        "target": labels,
    })


@pytest.fixture
def dirty_data():
    """A DataFrame with extreme missingness, dead columns, duplicates."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "all_null": [np.nan] * n,
        "constant_col": [42.0] * n,
        "numeric_a": rng.normal(10, 3, n),
        "numeric_b": rng.normal(20, 5, n),
        "cat_col": rng.choice(["A", "B", "C"], n),
        "target": rng.integers(0, 2, n),
    })
    # Inject 30% random missing
    mask = rng.random((n, 6)) < 0.30
    for i in range(n):
        for j in range(6):
            if mask[i, j] and j not in (0, 1, 5):  # Keep target, all_null, constant intact
                df.iat[i, j] = np.nan
    # Inject 10% duplicate rows
    dup_rows = df.iloc[rng.integers(0, n, size=10)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


@pytest.fixture
def leakage_df():
    """DataFrame with a leaked feature perfectly correlated with target."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(5, 2, n),
        "leaked_col": target * 2.0 + 1.0,
        "target": target,
    })
