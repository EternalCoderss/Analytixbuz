"""
Step 10: Explainable AI with SHAP — Global & local explanations + error analysis.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from src.schemas.models import (
    ExplainabilityResult,
    FeatureImportance,
    LocalExplanation,
)

warnings.filterwarnings("ignore")


def compute_shap_explanations(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    task_type: str = "Classification",
    max_samples: int = 200,
) -> ExplainabilityResult:
    """
    Compute SHAP explanations for the winning model.
    Uses TreeExplainer for tree models, LinearExplainer for linear, fallback otherwise.
    """
    import shap

    # Subsample for speed
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(len(X))

    # Determine explainer type
    model_type = type(model).__name__
    tree_models = {
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
    }
    linear_models = {"LogisticRegression", "Ridge", "LinearRegression", "Lasso"}

    shap_values = None
    base_value = 0.0

    try:
        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_sample)
            base_value = _extract_base_value(explainer.expected_value, task_type)
            shap_values = _normalize_shap_values(shap_values_raw, task_type)
        elif model_type in linear_models:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values_raw = explainer.shap_values(X_sample)
            base_value = _extract_base_value(explainer.expected_value, task_type)
            shap_values = _normalize_shap_values(shap_values_raw, task_type)
        else:
            # Permutation fallback
            explainer = shap.Explainer(model.predict, X_sample)
            explanation = explainer(X_sample)
            shap_values = explanation.values
            base_value = float(explanation.base_values.mean()) if hasattr(explanation, "base_values") else 0.0
    except Exception:
        try:
            # Ultimate fallback: Kernel SHAP on small sample
            background = X_sample[:min(50, len(X_sample))]
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                background,
            )
            shap_values_raw = explainer.shap_values(X_sample[:min(50, len(X_sample))])
            base_value = _extract_base_value(explainer.expected_value, task_type)
            shap_values = _normalize_shap_values(shap_values_raw, task_type)
            X_sample = X_sample[:len(shap_values)]
        except Exception:
            # Return empty but valid result
            return ExplainabilityResult(
                global_feature_importance=[],
                local_explanation=None,
                shap_values_sample=[],
            )

    if shap_values is None or len(shap_values) == 0:
        return ExplainabilityResult()

    # Ensure correct shapes
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    n_features = min(shap_values.shape[1], len(feature_names))
    feature_names = feature_names[:n_features]
    shap_values = shap_values[:, :n_features]

    # --- Global feature importance ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Determine direction using mean (not abs)
    mean_shap = shap_values.mean(axis=0)

    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    global_importance = []
    for idx in sorted_indices[:15]:
        direction = "positive" if mean_shap[idx] >= 0 else "negative"
        global_importance.append(FeatureImportance(
            feature=feature_names[idx],
            importance=round(float(mean_abs_shap[idx]), 6),
            direction=direction,
        ))

    # --- Local explanation for highest-impact instance ---
    if task_type == "Classification" and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_sample)
            if probs.shape[1] == 2:
                row_idx = int(np.argmax(probs[:, 1]))
            else:
                row_idx = int(np.argmax(probs.max(axis=1)))
        except Exception:
            row_idx = 0
    else:
        preds = model.predict(X_sample)
        row_idx = int(np.argmax(np.abs(preds - np.mean(preds))))

    local_shap_vals = shap_values[row_idx]
    pred_value = float(model.predict(X_sample[row_idx:row_idx + 1])[0])

    local_contributions = []
    local_sorted = np.argsort(np.abs(local_shap_vals))[::-1]
    for idx in local_sorted[:10]:
        direction = "positive" if local_shap_vals[idx] >= 0 else "negative"
        local_contributions.append(FeatureImportance(
            feature=feature_names[idx],
            importance=round(float(local_shap_vals[idx]), 6),
            direction=direction,
        ))

    narrative = _generate_local_narrative(local_contributions, pred_value, task_type)

    local_explanation = LocalExplanation(
        row_index=int(row_idx),
        base_value=round(float(base_value), 6),
        predicted_value=round(pred_value, 6),
        contributions=local_contributions,
        narrative=narrative,
    )

    # SHAP values sample for plotting (top 50 rows)
    plot_rows = min(50, len(shap_values))
    shap_sample = shap_values[:plot_rows].tolist()

    return ExplainabilityResult(
        global_feature_importance=global_importance,
        local_explanation=local_explanation,
        shap_values_sample=shap_sample,
    )


def compute_local_shap_for_row(
    model: Any,
    X_row: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    task_type: str = "Classification",
) -> list[FeatureImportance]:
    """Compute SHAP for a single row (used by What-If simulator)."""
    import shap

    model_type = type(model).__name__
    tree_models = {
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
    }

    try:
        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
        else:
            background = X_background[:min(50, len(X_background))]
            explainer = shap.Explainer(model.predict, background)

        row = X_row.reshape(1, -1) if X_row.ndim == 1 else X_row
        sv_raw = explainer.shap_values(row)
        sv = _normalize_shap_values(sv_raw, task_type)
        if sv.ndim > 1:
            sv = sv[0]
    except Exception:
        return []

    n_features = min(len(sv), len(feature_names))
    contributions = []
    sorted_idx = np.argsort(np.abs(sv[:n_features]))[::-1]
    for idx in sorted_idx[:10]:
        direction = "positive" if sv[idx] >= 0 else "negative"
        contributions.append(FeatureImportance(
            feature=feature_names[idx],
            importance=round(float(sv[idx]), 6),
            direction=direction,
        ))
    return contributions


def _extract_base_value(expected_value, task_type: str) -> float:
    """Extract scalar base value from SHAP expected_value."""
    if isinstance(expected_value, (list, np.ndarray)):
        if len(expected_value) == 2 and task_type == "Classification":
            return float(expected_value[1])
        return float(expected_value[0]) if len(expected_value) > 0 else 0.0
    return float(expected_value)


def _normalize_shap_values(shap_values, task_type: str) -> np.ndarray:
    """Normalize SHAP values to a 2D array."""
    if isinstance(shap_values, list):
        if task_type == "Classification" and len(shap_values) == 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0]) if len(shap_values) > 0 else np.array(shap_values)
    arr = np.array(shap_values)
    if arr.ndim == 3:
        if task_type == "Classification":
            return arr[:, :, 1] if arr.shape[2] == 2 else arr[:, :, 0]
        return arr[:, :, 0]
    return arr


def _generate_local_narrative(contributions: list[FeatureImportance],
                              predicted_value: float,
                              task_type: str) -> str:
    """Generate plain-language narrative for a local explanation."""
    if not contributions:
        return f"Predicted value: {predicted_value:.4f}"

    lines = [f"Predicted value: {predicted_value:.4f}"]
    lines.append("Top contributing factors:")

    for i, c in enumerate(contributions[:5]):
        effect = "increases" if c.direction == "positive" else "decreases"
        lines.append(f"  {i + 1}. '{c.feature}' {effect} the prediction "
                     f"(SHAP contribution: {c.importance:+.4f})")

    return "\n".join(lines)
