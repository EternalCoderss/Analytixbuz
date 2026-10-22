"""
Live What-If Simulator — Perturbation engine for interactive scenario analysis.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.engine.explainability import compute_local_shap_for_row
from src.engine.feature_engineer import FeatureEngineer
from src.schemas.models import FeatureImportance, WhatIfResult


def _assign_risk_tier(prob: float) -> str:
    """Assign risk tier from probability."""
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def run_what_if(
    baseline_row: pd.DataFrame,
    overrides: dict[str, Any],
    feature_engineer: FeatureEngineer,
    model: Any,
    X_background: np.ndarray,
    feature_names: list[str],
    task_type: str = "Classification",
    is_risk_analysis: bool = False,
) -> WhatIfResult:
    """
    Run a What-If simulation.

    Args:
        baseline_row: Single-row DataFrame with original feature values.
        overrides: Dictionary of feature name -> new value overrides.
        feature_engineer: Fitted FeatureEngineer instance.
        model: Trained model.
        X_background: Background data for SHAP computation.
        feature_names: List of feature names after engineering.
        task_type: Classification, Regression, etc.
        is_risk_analysis: Whether risk tiers should be computed.

    Returns:
        WhatIfResult with baseline vs. simulated comparison.
    """
    # --- Baseline prediction ---
    baseline_df = baseline_row.copy()
    try:
        X_baseline = feature_engineer.transform(baseline_df)
    except Exception:
        # Fallback: try to construct input manually
        X_baseline = _fallback_transform(baseline_df, feature_engineer)

    baseline_pred = float(model.predict(X_baseline)[0])

    baseline_prob = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_baseline)
        if probs.shape[1] == 2:
            baseline_prob = float(probs[0, 1])
        else:
            baseline_prob = float(probs[0].max())

    # --- Simulated prediction ---
    simulated_df = baseline_row.copy()
    for col, val in overrides.items():
        if col in simulated_df.columns:
            simulated_df[col] = val

    try:
        X_simulated = feature_engineer.transform(simulated_df)
    except Exception:
        X_simulated = _fallback_transform(simulated_df, feature_engineer)

    simulated_pred = float(model.predict(X_simulated)[0])

    simulated_prob = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_simulated)
        if probs.shape[1] == 2:
            simulated_prob = float(probs[0, 1])
        else:
            simulated_prob = float(probs[0].max())

    # --- Compute deltas ---
    if task_type == "Classification" and is_risk_analysis:
        delta_value = round(simulated_prob - baseline_prob, 6)
        delta_pct = round(delta_value / max(abs(baseline_prob), 1e-6) * 100, 2)
        base_val = baseline_prob
        sim_val = simulated_prob
    else:
        delta_value = round(simulated_pred - baseline_pred, 6)
        delta_pct = round(delta_value / max(abs(baseline_pred), 1e-6) * 100, 2)
        base_val = baseline_pred
        sim_val = simulated_pred

    # Risk tiers
    baseline_tier = ""
    simulated_tier = ""
    if is_risk_analysis:
        baseline_tier = _assign_risk_tier(baseline_prob)
        simulated_tier = _assign_risk_tier(simulated_prob)

    # --- Local SHAP for simulated prediction ---
    updated_shap = []
    try:
        updated_shap = compute_local_shap_for_row(
            model, X_simulated[0] if X_simulated.ndim > 1 else X_simulated,
            X_background, feature_names, task_type,
        )
    except Exception:
        pass

    # --- Generate recommendation ---
    recommendation = _generate_recommendation(
        baseline_pred, simulated_pred, baseline_prob, simulated_prob,
        overrides, task_type, is_risk_analysis,
        baseline_tier, simulated_tier,
    )

    return WhatIfResult(
        baseline_prediction=round(base_val, 6),
        simulated_prediction=round(sim_val, 6),
        delta_value=delta_value,
        delta_pct=delta_pct,
        baseline_risk_tier=baseline_tier,
        simulated_risk_tier=simulated_tier,
        updated_local_shap=updated_shap,
        recommendation=recommendation,
    )


def _fallback_transform(df: pd.DataFrame, fe: FeatureEngineer) -> np.ndarray:
    """Fallback transformation when the main pipeline encounters issues."""
    # Build a numeric-only representation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        vals = df[numeric_cols].fillna(0).values
        # Pad or truncate to match expected shape
        expected = len(fe.feature_names_out) if fe.feature_names_out else vals.shape[1]
        if vals.shape[1] < expected:
            padding = np.zeros((vals.shape[0], expected - vals.shape[1]))
            vals = np.hstack([vals, padding])
        elif vals.shape[1] > expected:
            vals = vals[:, :expected]
        return vals
    return np.zeros((len(df), max(len(fe.feature_names_out), 1)))


def _generate_recommendation(baseline_pred, simulated_pred,
                              baseline_prob, simulated_prob,
                              overrides, task_type, is_risk_analysis,
                              baseline_tier, simulated_tier) -> str:
    """Generate dynamic recommendation based on the What-If result."""
    changes = ", ".join(f"{k}={v}" for k, v in overrides.items())

    if is_risk_analysis:
        if simulated_prob < baseline_prob:
            delta = baseline_prob - simulated_prob
            return (
                f"✅ RISK REDUCED: Adjusting {changes} reduces risk from "
                f"{baseline_tier} ({baseline_prob:.1%}) to "
                f"{simulated_tier} ({simulated_prob:.1%}) — "
                f"a {delta:.1%} reduction. This intervention is recommended."
            )
        elif simulated_prob > baseline_prob:
            delta = simulated_prob - baseline_prob
            return (
                f"⚠️ RISK INCREASED: Adjusting {changes} increases risk from "
                f"{baseline_tier} ({baseline_prob:.1%}) to "
                f"{simulated_tier} ({simulated_prob:.1%}) — "
                f"a {delta:.1%} increase. This change should be avoided."
            )
        else:
            return f"➡️ NO CHANGE: Adjusting {changes} does not significantly affect risk."
    else:
        if abs(simulated_pred - baseline_pred) < 1e-6:
            return f"➡️ NO CHANGE: Adjusting {changes} does not significantly affect the outcome."
        elif simulated_pred > baseline_pred:
            return (
                f"📈 POSITIVE: Adjusting {changes} increases the predicted value from "
                f"{baseline_pred:.2f} to {simulated_pred:.2f} (+{simulated_pred - baseline_pred:.2f})."
            )
        else:
            return (
                f"📉 DECREASE: Adjusting {changes} decreases the predicted value from "
                f"{baseline_pred:.2f} to {simulated_pred:.2f} ({simulated_pred - baseline_pred:.2f})."
            )
