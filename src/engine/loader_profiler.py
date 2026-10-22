"""
Steps 1-2: Data Ingestion, dtype inference, ID detection, and profiling.
"""

from __future__ import annotations

import io
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.schemas.models import ColumnProfile, DataProfileResult


def load_dataframe(source: Union[str, io.BytesIO, io.StringIO, pd.DataFrame],
                   encoding: str = "utf-8") -> pd.DataFrame:
    """Load CSV from a filepath, bytes buffer, or return an existing DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    try:
        if isinstance(source, str):
            return pd.read_csv(source, encoding=encoding)
        else:
            return pd.read_csv(source, encoding=encoding)
    except UnicodeDecodeError:
        if isinstance(source, str):
            return pd.read_csv(source, encoding="latin-1")
        else:
            source.seek(0)
            return pd.read_csv(source, encoding="latin-1")


def _infer_semantic_type(series: pd.Series, col_name: str) -> str:
    """Infer the semantic type of a column."""
    # Drop NaN for analysis
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"

    # Check datetime first
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Try parsing as datetime
    if series.dtype == object:
        try:
            sample = non_null.head(20)
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except (ValueError, TypeError):
            pass

    # Boolean check
    unique_vals = set(non_null.unique())
    if unique_vals <= {0, 1, True, False, "True", "False", "true", "false", 0.0, 1.0}:
        if len(unique_vals) <= 3:
            return "boolean"

    # Numeric checks
    if pd.api.types.is_numeric_dtype(series):
        n_unique = non_null.nunique()
        total = len(non_null)
        # Identifier detection: sequential-like integers with very high uniqueness
        if _is_identifier(series, col_name, non_null, n_unique, total):
            return "identifier"
        if pd.api.types.is_float_dtype(series) or n_unique > 20:
            return "numeric_continuous"
        else:
            return "numeric_discrete"

    # Categorical / Identifier for string columns
    n_unique = non_null.nunique()
    total = len(non_null)
    if _is_identifier(series, col_name, non_null, n_unique, total):
        return "identifier"

    return "categorical"


def _is_identifier(series: pd.Series, col_name: str,
                   non_null: pd.Series, n_unique: int, total: int) -> bool:
    """Check if a column is a pure identifier column."""
    name_lower = col_name.lower()
    id_keywords = ["_id", "id_", "customer_id", "machine_id", "store_id",
                   "user_id", "order_id", "transaction_id", "index"]
    if any(kw in name_lower for kw in id_keywords):
        if n_unique / max(total, 1) > 0.5:
            return True

    # >98% unique string values or sequential integers
    if total > 0 and n_unique / total > 0.98:
        if series.dtype == object:
            return True
        if pd.api.types.is_integer_dtype(series):
            sorted_vals = non_null.sort_values().values
            if len(sorted_vals) > 10:
                diffs = np.diff(sorted_vals[:100])
                if np.all(diffs >= 0) and np.median(diffs) <= 2:
                    return True
    return False


def profile_dataframe(df: pd.DataFrame) -> DataProfileResult:
    """Profile a DataFrame and return structured metadata."""
    columns = []
    id_columns = []
    datetime_columns = []
    numeric_columns = []
    categorical_columns = []
    boolean_columns = []

    for col in df.columns:
        series = df[col]
        sem_type = _infer_semantic_type(series, col)

        n_unique = int(series.nunique())
        n_missing = int(series.isna().sum())
        n_total = len(series)

        sample_vals = series.dropna().head(5).tolist()

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            semantic_type=sem_type,
            unique_count=n_unique,
            missing_count=n_missing,
            missing_pct=round(n_missing / max(n_total, 1) * 100, 2),
            sample_values=sample_vals,
        )
        columns.append(profile)

        if sem_type == "identifier":
            id_columns.append(col)
        elif sem_type == "datetime":
            datetime_columns.append(col)
        elif sem_type in ("numeric_continuous", "numeric_discrete"):
            numeric_columns.append(col)
        elif sem_type == "boolean":
            boolean_columns.append(col)
        elif sem_type == "categorical":
            categorical_columns.append(col)

    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    return DataProfileResult(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_mb=memory_mb,
        columns=columns,
        id_columns=id_columns,
        datetime_columns=datetime_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
    )


def convert_datetime_columns(df: pd.DataFrame, datetime_cols: list[str]) -> pd.DataFrame:
    """Convert detected datetime columns to pandas datetime type."""
    df = df.copy()
    for col in datetime_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except (ValueError, TypeError):
                pass
    return df
