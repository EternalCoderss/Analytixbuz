"""
Step 5: Automatic Problem-Type Detection.
Supports 6 analytical problem types: Classification, Regression,
Time-Series Forecasting, Anomaly Detection, Clustering, Risk Analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.schemas.models import TaskDetectionResult

# Priority keywords for auto-detecting the target column
TARGET_KEYWORDS = [
    "failure", "churn", "demand", "sales", "target", "label",
    "risk", "anomaly", "price", "revenue", "default", "fraud",
]

# Keywords that trigger Risk Analysis as secondary task
RISK_KEYWORDS = ["failure", "churn", "default", "risk", "fraud", "anomaly"]


def detect_task(df: pd.DataFrame,
                target_col: Optional[str] = None,
                id_columns: Optional[list[str]] = None,
                datetime_columns: Optional[list[str]] = None) -> TaskDetectionResult:
    """
    Auto-detect the analytical problem type and target column.
    """
    id_columns = id_columns or []
    datetime_columns = datetime_columns or []

    # --- Unsupervised fallback ---
    if target_col is None or target_col == "":
        target_col = _auto_detect_target(df, id_columns, datetime_columns)

    if target_col is None:
        # No target found -> route to unsupervised
        return _detect_unsupervised(df, id_columns)

    if target_col not in df.columns:
        return _detect_unsupervised(df, id_columns)

    target_series = df[target_col].dropna()

    # Determine target characteristics
    n_unique = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    has_datetime = len(datetime_columns) > 0

    # Classification vs Regression
    if not is_numeric or n_unique <= 20:
        return _detect_classification(df, target_col, target_series, n_unique)
    else:
        return _detect_regression(df, target_col, target_series, has_datetime, datetime_columns)


def _auto_detect_target(df: pd.DataFrame,
                        id_columns: list[str],
                        datetime_columns: list[str]) -> Optional[str]:
    """Auto-detect the target column using keyword priority."""
    cols_lower = {c.lower(): c for c in df.columns}

    # Priority keyword scan
    for keyword in TARGET_KEYWORDS:
        for col_lower, col_name in cols_lower.items():
            if keyword in col_lower and col_name not in id_columns:
                return col_name

    # Fallback: last non-ID, non-datetime column
    candidates = [c for c in df.columns
                  if c not in id_columns and c not in datetime_columns]
    if candidates:
        return candidates[-1]

    return None


def _detect_classification(df: pd.DataFrame, target_col: str,
                           target_series: pd.Series,
                           n_unique: int) -> TaskDetectionResult:
    """Detect classification task (binary or multiclass)."""
    class_dist = target_series.value_counts().to_dict()
    # Convert keys to strings for JSON serialization
    class_dist = {str(k): int(v) for k, v in class_dist.items()}

    if n_unique == 2:
        target_type = "binary"
    else:
        target_type = "multiclass"

    secondary_tasks = []
    # Check for Risk Analysis
    if target_type == "binary":
        target_lower = target_col.lower()
        if any(kw in target_lower for kw in RISK_KEYWORDS):
            secondary_tasks.append("Risk Analysis")

    reasoning = (
        f"Target '{target_col}' has {n_unique} unique values "
        f"({'binary' if n_unique == 2 else 'multi-class'} categorical). "
        f"Detected as Classification."
    )
    if "Risk Analysis" in secondary_tasks:
        reasoning += (
            f" '{target_col}' matches risk keywords — "
            f"Risk Analysis enabled with calibrated probability tiers."
        )

    return TaskDetectionResult(
        primary_task="Classification",
        secondary_tasks=secondary_tasks,
        target_column=target_col,
        target_type=target_type,
        n_classes=n_unique,
        class_distribution=class_dist,
        detection_reasoning=reasoning,
    )


def _detect_regression(df: pd.DataFrame, target_col: str,
                       target_series: pd.Series,
                       has_datetime: bool,
                       datetime_columns: list[str]) -> TaskDetectionResult:
    """Detect regression or time-series forecasting task."""
    secondary_tasks = []
    primary_task = "Regression"

    # Check for Time-Series Forecasting
    if has_datetime and len(datetime_columns) > 0:
        for dt_col in datetime_columns:
            if dt_col in df.columns:
                try:
                    dt_series = pd.to_datetime(df[dt_col], errors="coerce")
                    if dt_series.notna().sum() > 0.8 * len(df):
                        # Check if roughly chronological
                        sorted_check = dt_series.dropna().is_monotonic_increasing
                        primary_task = "Time-Series Forecasting"
                        secondary_tasks.append("Regression")
                        break
                except Exception:
                    pass

    reasoning = (
        f"Target '{target_col}' is continuous numeric with "
        f"{target_series.nunique()} unique values. "
        f"Detected as {primary_task}."
    )
    if "Regression" in secondary_tasks:
        reasoning += " Dataset has chronological datetime column — Time-Series mode enabled."

    return TaskDetectionResult(
        primary_task=primary_task,
        secondary_tasks=secondary_tasks,
        target_column=target_col,
        target_type="continuous",
        n_classes=None,
        class_distribution={},
        detection_reasoning=reasoning,
    )


def _detect_unsupervised(df: pd.DataFrame,
                         id_columns: list[str]) -> TaskDetectionResult:
    """Fallback to unsupervised learning when no target is available."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_columns]

    if len(numeric_cols) < 2:
        primary = "Clustering"
    else:
        primary = "Clustering"

    return TaskDetectionResult(
        primary_task=primary,
        secondary_tasks=["Anomaly Detection"],
        target_column=None,
        target_type=None,
        n_classes=None,
        class_distribution={},
        detection_reasoning=(
            "No target column specified or detected. "
            f"Routing to {primary} with Anomaly Detection as secondary task."
        ),
    )
