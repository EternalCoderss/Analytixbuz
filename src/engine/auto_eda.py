"""
Step 4: Automated EDA — Statistical summaries, correlations, trend data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.schemas.models import ColumnStats, CorrelationPair, EDAResult


def run_eda(df: pd.DataFrame,
            target_col: Optional[str] = None,
            id_columns: Optional[list[str]] = None) -> EDAResult:
    """Generate automated EDA results."""
    id_columns = id_columns or []

    # --- Column statistics ---
    column_stats = []
    for col in df.columns:
        if col in id_columns:
            continue
        series = df[col]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_pct = round(n_missing / max(n_total, 1) * 100, 2)

        cs = ColumnStats(name=col, count=int(series.count()), missing_pct=missing_pct)

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            cs.mean = _safe_round(desc.get("mean"))
            cs.std = _safe_round(desc.get("std"))
            cs.min_val = _safe_round(desc.get("min"))
            cs.q25 = _safe_round(desc.get("25%"))
            cs.median = _safe_round(desc.get("50%"))
            cs.q75 = _safe_round(desc.get("75%"))
            cs.max_val = _safe_round(desc.get("max"))
            try:
                cs.skewness = _safe_round(series.dropna().skew())
            except Exception:
                cs.skewness = None

        column_stats.append(cs)

    # --- Top feature-to-feature correlations ---
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_columns]
    top_correlations = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method="pearson")
        spearman_matrix = df[numeric_cols].corr(method="spearman")

        pairs_seen = set()
        pairs = []
        for i, c1 in enumerate(numeric_cols):
            for j, c2 in enumerate(numeric_cols):
                if i >= j:
                    continue
                pair_key = tuple(sorted([c1, c2]))
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)
                p_val = corr_matrix.loc[c1, c2] if c1 in corr_matrix.index and c2 in corr_matrix.columns else None
                s_val = spearman_matrix.loc[c1, c2] if c1 in spearman_matrix.index and c2 in spearman_matrix.columns else None
                if p_val is not None and not np.isnan(p_val):
                    pairs.append((abs(p_val), c1, c2, p_val, s_val))

        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, c1, c2, p_val, s_val in pairs[:15]:
            top_correlations.append(CorrelationPair(
                feature_a=c1, feature_b=c2,
                pearson=_safe_round(p_val, 4),
                spearman=_safe_round(s_val, 4),
            ))

    # --- Feature-to-target correlations ---
    target_correlations = []
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        target_series = df[target_col]
        feature_corrs = []
        for col in numeric_cols:
            if col == target_col:
                continue
            try:
                p_corr = df[col].corr(target_series)
                s_corr = df[col].corr(target_series, method="spearman")
                if not np.isnan(p_corr):
                    feature_corrs.append((abs(p_corr), col, p_corr, s_corr))
            except Exception:
                pass
        feature_corrs.sort(key=lambda x: x[0], reverse=True)
        for _, col, p_val, s_val in feature_corrs[:10]:
            target_correlations.append(CorrelationPair(
                feature_a=col, feature_b=target_col,
                pearson=_safe_round(p_val, 4),
                spearman=_safe_round(s_val, 4),
            ))

    # --- Statistical takeaways ---
    takeaways = _generate_takeaways(df, column_stats, target_col, top_correlations, target_correlations)

    return EDAResult(
        column_stats=column_stats,
        top_correlations=top_correlations,
        target_correlations=target_correlations,
        takeaways=takeaways,
    )


def _safe_round(val, decimals=4):
    """Safely round a value, returning None for NaN/None."""
    if val is None:
        return None
    try:
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def _generate_takeaways(df, column_stats, target_col, top_corrs, target_corrs):
    """Generate plain-language statistical takeaways."""
    takeaways = []

    # Dataset overview
    takeaways.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")

    # Missing data summary
    total_missing = df.isna().sum().sum()
    if total_missing > 0:
        worst_col = df.isna().sum().idxmax()
        worst_pct = df[worst_col].isna().mean() * 100
        takeaways.append(f"Total missing values: {total_missing}. "
                         f"Column '{worst_col}' has the most missing data ({worst_pct:.1f}%).")

    # Highly skewed columns
    for cs in column_stats:
        if cs.skewness is not None and abs(cs.skewness) > 2.0:
            direction = "right" if cs.skewness > 0 else "left"
            takeaways.append(f"Column '{cs.name}' is highly {direction}-skewed "
                             f"(skewness={cs.skewness:.2f}).")

    # Strong correlations
    if top_corrs:
        strongest = top_corrs[0]
        if strongest.pearson is not None and abs(strongest.pearson) > 0.5:
            takeaways.append(f"Strongest feature correlation: '{strongest.feature_a}' and "
                             f"'{strongest.feature_b}' (r={strongest.pearson:.3f}).")

    # Target drivers
    if target_corrs and target_col:
        top_drivers = [tc for tc in target_corrs[:3]
                       if tc.pearson is not None and abs(tc.pearson) > 0.1]
        if top_drivers:
            driver_names = [f"'{d.feature_a}' (r={d.pearson:.3f})" for d in top_drivers]
            takeaways.append(f"Top drivers of '{target_col}': {', '.join(driver_names)}.")

    return takeaways
