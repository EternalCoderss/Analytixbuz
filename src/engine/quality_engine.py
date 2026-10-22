"""
Step 3: Data Quality Engine — Missing, duplicates, outliers, leakage, Health Score.
Implements the exact Health Score formula and auto-clean pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.schemas.models import (
    CleaningAction,
    QualityPenalties,
    QualityResult,
)


def compute_quality(df: pd.DataFrame,
                    target_col: Optional[str] = None,
                    id_columns: Optional[list[str]] = None) -> QualityResult:
    """
    Compute quality penalties and the Data Health Score.
    Returns QualityResult with penalties, score, and diagnostics.
    """
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)

    # --- P_null: total missing / total cells ---
    missing_total = int(df.isna().sum().sum())
    p_null = missing_total / total_cells

    # --- P_dup: duplicate rows / total rows ---
    dup_count = int(df.duplicated().sum())
    p_dup = dup_count / max(n_rows, 1)

    # --- P_out: average outlier fraction across numeric columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and ID columns from outlier computation
    exclude = set()
    if target_col:
        exclude.add(target_col)
    if id_columns:
        exclude.update(id_columns)
    numeric_for_outliers = [c for c in numeric_cols if c not in exclude]

    outlier_summary = {}
    outlier_fractions = []
    for col in numeric_for_outliers:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        outlier_summary[col] = n_outliers
        outlier_fractions.append(n_outliers / max(len(series), 1))

    p_out = float(np.mean(outlier_fractions)) if outlier_fractions else 0.0

    # --- P_leak: fraction of features with |r| > 0.95 with target ---
    leakage_columns = []
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            feature_cols = [c for c in numeric_cols if c != target_col and c not in (id_columns or [])]
            for col in feature_cols:
                try:
                    corr = df[col].corr(target_series)
                    if abs(corr) > 0.95:
                        leakage_columns.append(col)
                except Exception:
                    pass
    n_feature_cols = max(len([c for c in df.columns if c != target_col and c not in (id_columns or [])]), 1)
    p_leak = len(leakage_columns) / n_feature_cols

    # --- Health Score ---
    penalties = QualityPenalties(
        p_null=round(min(p_null, 1.0), 6),
        p_dup=round(min(p_dup, 1.0), 6),
        p_out=round(min(p_out, 1.0), 6),
        p_leak=round(min(p_leak, 1.0), 6),
    )

    raw = 1.0 - (0.35 * penalties.p_null + 0.15 * penalties.p_dup +
                  0.20 * penalties.p_out + 0.30 * penalties.p_leak)
    health_score = round(max(0.0, min(100.0, 100.0 * raw)), 2)

    return QualityResult(
        penalties=penalties,
        health_score=health_score,
        missing_cells_total=missing_total,
        duplicate_rows=dup_count,
        outlier_summary=outlier_summary,
        leakage_columns=leakage_columns,
        dropped_columns=[],
        cleaning_log=[],
        rows_after_clean=n_rows,
        cols_after_clean=n_cols,
    )


def auto_clean(df: pd.DataFrame,
               quality_result: QualityResult,
               target_col: Optional[str] = None,
               id_columns: Optional[list[str]] = None) -> tuple[pd.DataFrame, QualityResult]:
    """
    Auto-clean pipeline: drop dead columns, leakage, duplicates, impute missing.
    Returns cleaned DataFrame and updated QualityResult with cleaning log.
    """
    df = df.copy()
    log = []
    dropped_cols = []

    # 1. Drop 100% null columns
    null_pcts = df.isna().mean()
    full_null_cols = null_pcts[null_pcts >= 1.0].index.tolist()
    if full_null_cols:
        # Don't drop the target column
        cols_to_drop = [c for c in full_null_cols if c != target_col]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            dropped_cols.extend(cols_to_drop)
            log.append(CleaningAction(
                action="drop_full_null_columns",
                details=f"Dropped 100% null columns: {cols_to_drop}",
                affected_columns=len(cols_to_drop),
            ))

    # 2. Drop constant (zero-variance) columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1
                     and c != target_col]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        dropped_cols.extend(constant_cols)
        log.append(CleaningAction(
            action="drop_constant_columns",
            details=f"Dropped zero-variance columns: {constant_cols}",
            affected_columns=len(constant_cols),
        ))

    # 3. Drop target-leakage columns
    leak_cols = [c for c in quality_result.leakage_columns
                 if c in df.columns and c != target_col]
    if leak_cols:
        df = df.drop(columns=leak_cols)
        dropped_cols.extend(leak_cols)
        log.append(CleaningAction(
            action="drop_leakage_columns",
            details=f"Dropped target-leakage columns (|r|>0.95): {leak_cols}",
            affected_columns=len(leak_cols),
        ))

    # 4. Drop exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped_rows = n_before - len(df)
    if n_dropped_rows > 0:
        log.append(CleaningAction(
            action="drop_duplicates",
            details=f"Removed {n_dropped_rows} exact duplicate rows",
            affected_rows=n_dropped_rows,
        ))

    # 5. Impute remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    imputed_numeric = 0
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_numeric += n_missing

    imputed_cat = 0
    for col in cat_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"
            df[col] = df[col].fillna(fill_val)
            imputed_cat += n_missing

    if imputed_numeric > 0:
        log.append(CleaningAction(
            action="impute_numeric_median",
            details=f"Imputed {imputed_numeric} missing numeric cells with median",
            affected_rows=imputed_numeric,
        ))
    if imputed_cat > 0:
        log.append(CleaningAction(
            action="impute_categorical_mode",
            details=f"Imputed {imputed_cat} missing categorical cells with mode",
            affected_rows=imputed_cat,
        ))

    # Handle datetime column NaNs - fill with forward fill then backward fill
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in datetime_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    # Handle boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(False)

    # 6. Final dedup pass — imputation can create new duplicates
    n_before_final = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_final_dups = n_before_final - len(df)
    if n_final_dups > 0:
        log.append(CleaningAction(
            action="drop_post_imputation_duplicates",
            details=f"Removed {n_final_dups} duplicates created by imputation",
            affected_rows=n_final_dups,
        ))

    # Update quality result
    quality_result.dropped_columns = dropped_cols
    quality_result.cleaning_log = log
    quality_result.rows_after_clean = len(df)
    quality_result.cols_after_clean = len(df.columns)

    return df, quality_result

