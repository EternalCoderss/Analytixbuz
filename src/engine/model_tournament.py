"""
Steps 7-9: Model Tournament — Multi-model training, Intelligent Score, Predict.
Implements the exact Intelligent Model Score formula.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from src.schemas.models import ModelResult, TournamentResult

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing optional boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def _get_explainability_score(model_name: str) -> float:
    """Compute explainability score based on model type."""
    linear_models = {"LogisticRegression", "Ridge"}
    tree_models = {"RandomForestClassifier", "RandomForestRegressor",
                   "GradientBoostingClassifier", "GradientBoostingRegressor",
                   "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}
    if model_name in linear_models:
        return 1.0
    elif model_name in tree_models:
        return 0.92
    return 0.70


def _compute_intelligent_score(accuracy_norm: float, speed_score: float,
                                robustness_score: float, explainability_score: float) -> float:
    """
    Compute the Intelligent Model Score (0 to 100).
    Formula: 100 * (0.55 * Accuracy_Norm + 0.15 * Speed + 0.20 * Robustness + 0.10 * Explainability)
    """
    raw = (0.55 * accuracy_norm + 0.15 * speed_score +
           0.20 * robustness_score + 0.10 * explainability_score)
    return round(100.0 * max(0.0, min(1.0, raw)), 2)


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


def run_tournament(X: np.ndarray, y: np.ndarray,
                   task_type: str,
                   feature_names: Optional[list[str]] = None,
                   is_risk_analysis: bool = False,
                   n_splits: int = 5) -> tuple[TournamentResult, Any, Any]:
    """
    Run model tournament. Returns (TournamentResult, best_model, X_test_for_shap).
    """
    if task_type in ("Classification",):
        return _run_classification_tournament(X, y, feature_names, is_risk_analysis, n_splits)
    elif task_type in ("Regression", "Time-Series Forecasting"):
        return _run_regression_tournament(X, y, task_type, feature_names, n_splits)
    elif task_type == "Clustering":
        return _run_clustering(X, feature_names)
    elif task_type == "Anomaly Detection":
        return _run_anomaly_detection(X, feature_names)
    else:
        return _run_classification_tournament(X, y, feature_names, is_risk_analysis, n_splits)


def _run_classification_tournament(X: np.ndarray, y: np.ndarray,
                                    feature_names: Optional[list[str]],
                                    is_risk_analysis: bool,
                                    n_splits: int) -> tuple[TournamentResult, Any, Any]:
    """Train and evaluate classification models."""
    # Determine unique classes
    unique_classes = np.unique(y[~np.isnan(y)] if np.issubdtype(y.dtype, np.floating) else y)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2

    # Adapt CV for small/imbalanced datasets
    min_class_count = min(np.bincount(y.astype(int)))
    actual_splits = min(n_splits, min_class_count)
    actual_splits = max(actual_splits, 2)

    # Define candidate models
    candidates = {}
    candidates["LogisticRegression"] = LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs",
        class_weight="balanced" if min_class_count < 20 else None,
    )
    candidates["RandomForestClassifier"] = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight="balanced" if min_class_count < 20 else None,
    )
    candidates["GradientBoostingClassifier"] = GradientBoostingClassifier(
        n_estimators=100, random_state=42, max_depth=4,
    )
    if HAS_XGB:
        scale = (len(y) - sum(y)) / max(sum(y), 1) if is_binary else 1
        candidates["XGBClassifier"] = XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
            scale_pos_weight=scale if min_class_count < 50 else 1,
        )
    if HAS_LGBM:
        candidates["LGBMClassifier"] = LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1,
            is_unbalance=True if min_class_count < 20 else False,
        )

    # Cross-validation
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    results = []
    trained_models = {}
    max_latency = 0.0

    for name, model in candidates.items():
        try:
            cv_aucs = []
            cv_f1s = []
            train_metrics = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model_clone = _clone_model(model)
                model_clone.fit(X_train, y_train)

                # Validation metrics
                y_pred = model_clone.predict(X_val)
                try:
                    y_prob = model_clone.predict_proba(X_val)
                    if is_binary:
                        auc = roc_auc_score(y_val, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")
                except Exception:
                    auc = 0.5
                f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                cv_aucs.append(auc)
                cv_f1s.append(f1)

                # Train metric for robustness
                y_train_pred = model_clone.predict(X_train)
                train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
                train_metrics.append(train_f1)

            # Final model trained on full data
            final_model = _clone_model(model)
            final_model.fit(X, y)
            trained_models[name] = final_model

            # Inference speed
            start = time.perf_counter()
            for _ in range(10):
                final_model.predict(X[:1])
            latency_ms = (time.perf_counter() - start) / 10 * 1000
            max_latency = max(max_latency, latency_ms)

            mean_auc = float(np.mean(cv_aucs))
            mean_f1 = float(np.mean(cv_f1s))
            mean_train = float(np.mean(train_metrics))
            mean_val = mean_f1

            accuracy_norm = 0.5 * min(max(mean_auc, 0), 1) + 0.5 * min(max(mean_f1, 0), 1)
            robustness = max(0.0, 1.0 - abs(mean_train - mean_val))
            explainability = _get_explainability_score(name)

            results.append({
                "name": name,
                "accuracy_norm": accuracy_norm,
                "robustness": robustness,
                "explainability": explainability,
                "latency_ms": latency_ms,
                "roc_auc": mean_auc,
                "f1_weighted": mean_f1,
                "train_metric": mean_train,
                "val_metric": mean_val,
                "cv_scores": cv_aucs,
            })

        except Exception as e:
            # Graceful skip
            continue

    if not results:
        raise RuntimeError("All classification models failed to train.")

    # Compute speed scores and intelligent scores
    model_results = []
    for r in results:
        speed = 1.0 - (r["latency_ms"] / (max_latency + 1e-6))
        speed = max(0.0, min(1.0, speed))

        intelligent = _compute_intelligent_score(
            r["accuracy_norm"], speed, r["robustness"], r["explainability"]
        )

        model_results.append(ModelResult(
            model_name=r["name"],
            accuracy_norm=round(r["accuracy_norm"], 4),
            speed_score=round(speed, 4),
            robustness_score=round(r["robustness"], 4),
            explainability_score=r["explainability"],
            intelligent_score=intelligent,
            roc_auc=round(r["roc_auc"], 4),
            f1_weighted=round(r["f1_weighted"], 4),
            train_metric=round(r["train_metric"], 4),
            val_metric=round(r["val_metric"], 4),
            inference_latency_ms=round(r["latency_ms"], 3),
            cv_scores=[round(s, 4) for s in r["cv_scores"]],
        ))

    # Sort by intelligent score
    model_results.sort(key=lambda m: m.intelligent_score, reverse=True)
    winner = model_results[0]
    best_model = trained_models[winner.model_name]

    # Generate predictions and probabilities
    predictions = best_model.predict(X).tolist()
    probabilities = []
    risk_tiers = []

    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X)
        if probs.shape[1] == 2:
            probabilities = probs[:, 1].tolist()
        else:
            probabilities = probs.max(axis=1).tolist()

        if is_risk_analysis:
            risk_tiers = [_assign_risk_tier(p) for p in probabilities]

    tournament = TournamentResult(
        task_type="Classification",
        models=model_results,
        winner_name=winner.model_name,
        winner_score=winner.intelligent_score,
        predictions=predictions,
        probabilities=probabilities,
        risk_tiers=risk_tiers,
    )

    return tournament, best_model, X


def _run_regression_tournament(X: np.ndarray, y: np.ndarray,
                                task_type: str,
                                feature_names: Optional[list[str]],
                                n_splits: int) -> tuple[TournamentResult, Any, Any]:
    """Train and evaluate regression models."""
    candidates = {}
    candidates["Ridge"] = Ridge(alpha=1.0)
    candidates["RandomForestRegressor"] = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1,
    )
    candidates["GradientBoostingRegressor"] = GradientBoostingRegressor(
        n_estimators=100, random_state=42, max_depth=4,
    )
    if HAS_XGB:
        candidates["XGBRegressor"] = XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0,
        )
    if HAS_LGBM:
        candidates["LGBMRegressor"] = LGBMRegressor(
            n_estimators=100, random_state=42, verbose=-1,
        )

    # CV strategy
    actual_splits = min(n_splits, max(len(y) // 5, 2))
    actual_splits = max(actual_splits, 2)

    if task_type == "Time-Series Forecasting":
        cv = TimeSeriesSplit(n_splits=actual_splits)
    else:
        cv = KFold(n_splits=actual_splits, shuffle=True, random_state=42)

    results = []
    trained_models = {}
    max_latency = 0.0

    for name, model in candidates.items():
        try:
            cv_r2s = []
            train_r2s = []

            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model_clone = _clone_model(model)
                model_clone.fit(X_train, y_train)

                y_pred = model_clone.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                cv_r2s.append(r2)

                y_train_pred = model_clone.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)
                train_r2s.append(train_r2)

            final_model = _clone_model(model)
            final_model.fit(X, y)
            trained_models[name] = final_model

            start = time.perf_counter()
            for _ in range(10):
                final_model.predict(X[:1])
            latency_ms = (time.perf_counter() - start) / 10 * 1000
            max_latency = max(max_latency, latency_ms)

            mean_r2 = float(np.mean(cv_r2s))
            mean_train_r2 = float(np.mean(train_r2s))

            accuracy_norm = max(0.0, min(1.0, mean_r2))
            robustness = max(0.0, 1.0 - abs(mean_train_r2 - mean_r2))
            explainability = _get_explainability_score(name)

            # Compute RMSE and MAE on full data
            y_pred_full = final_model.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred_full)))
            mae_val = float(mean_absolute_error(y, y_pred_full))

            results.append({
                "name": name,
                "accuracy_norm": accuracy_norm,
                "robustness": robustness,
                "explainability": explainability,
                "latency_ms": latency_ms,
                "r2_score": mean_r2,
                "rmse": rmse,
                "mae": mae_val,
                "train_metric": mean_train_r2,
                "val_metric": mean_r2,
                "cv_scores": cv_r2s,
            })

        except Exception:
            continue

    if not results:
        raise RuntimeError("All regression models failed to train.")

    model_results = []
    for r in results:
        speed = 1.0 - (r["latency_ms"] / (max_latency + 1e-6))
        speed = max(0.0, min(1.0, speed))
        intelligent = _compute_intelligent_score(
            r["accuracy_norm"], speed, r["robustness"], r["explainability"]
        )
        model_results.append(ModelResult(
            model_name=r["name"],
            accuracy_norm=round(r["accuracy_norm"], 4),
            speed_score=round(speed, 4),
            robustness_score=round(r["robustness"], 4),
            explainability_score=r["explainability"],
            intelligent_score=intelligent,
            r2_score=round(r["r2_score"], 4),
            rmse=round(r["rmse"], 4),
            mae=round(r["mae"], 4),
            train_metric=round(r["train_metric"], 4),
            val_metric=round(r["val_metric"], 4),
            inference_latency_ms=round(r["latency_ms"], 3),
            cv_scores=[round(s, 4) for s in r["cv_scores"]],
        ))

    model_results.sort(key=lambda m: m.intelligent_score, reverse=True)
    winner = model_results[0]
    best_model = trained_models[winner.model_name]
    predictions = best_model.predict(X).tolist()

    tournament = TournamentResult(
        task_type=task_type,
        models=model_results,
        winner_name=winner.model_name,
        winner_score=winner.intelligent_score,
        predictions=predictions,
        probabilities=[],
        risk_tiers=[],
    )
    return tournament, best_model, X


def _run_clustering(X: np.ndarray,
                    feature_names: Optional[list[str]]) -> tuple[TournamentResult, Any, Any]:
    """KMeans clustering with automatic k selection via Silhouette Score."""
    best_k = 2
    best_score = -1
    best_model = None
    n_samples = len(X)

    max_k = min(10, n_samples - 1)
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = km
        except Exception:
            continue

    if best_model is None:
        best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_model.fit(X)

    predictions = best_model.predict(X).tolist()

    model_result = ModelResult(
        model_name=f"KMeans(k={best_k})",
        accuracy_norm=round(max(best_score, 0), 4),
        speed_score=1.0,
        robustness_score=0.9,
        explainability_score=0.85,
        intelligent_score=round(100 * max(best_score, 0), 2),
    )

    tournament = TournamentResult(
        task_type="Clustering",
        models=[model_result],
        winner_name=model_result.model_name,
        winner_score=model_result.intelligent_score,
        predictions=predictions,
    )
    return tournament, best_model, X


def _run_anomaly_detection(X: np.ndarray,
                           feature_names: Optional[list[str]]) -> tuple[TournamentResult, Any, Any]:
    """IsolationForest anomaly detection."""
    model = IsolationForest(random_state=42, contamination=0.05, n_estimators=100)
    model.fit(X)
    predictions = model.predict(X).tolist()  # 1 = normal, -1 = anomaly
    scores = model.decision_function(X)
    anomaly_pct = sum(1 for p in predictions if p == -1) / max(len(predictions), 1)

    model_result = ModelResult(
        model_name="IsolationForest",
        accuracy_norm=round(1.0 - anomaly_pct, 4),
        speed_score=1.0,
        robustness_score=0.9,
        explainability_score=0.70,
        intelligent_score=round(100 * (1.0 - anomaly_pct), 2),
    )

    tournament = TournamentResult(
        task_type="Anomaly Detection",
        models=[model_result],
        winner_name="IsolationForest",
        winner_score=model_result.intelligent_score,
        predictions=predictions,
    )
    return tournament, model, X


def _clone_model(model):
    """Create a fresh clone of a model using sklearn's clone."""
    from sklearn.base import clone
    return clone(model)
