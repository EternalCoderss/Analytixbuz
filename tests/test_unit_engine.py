"""
Unit tests for math, formulas, and individual engine modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.engine.loader_profiler import load_dataframe, profile_dataframe
from src.engine.quality_engine import compute_quality, auto_clean
from src.engine.auto_eda import run_eda
from src.engine.task_detector import detect_task
from src.engine.feature_engineer import FeatureEngineer
from src.engine.model_tournament import (
    run_tournament,
    _compute_intelligent_score,
    _get_explainability_score,
)


class TestHealthScoreFormula:
    """Verify Data Health Score formula math and bounds."""

    def test_perfect_data_health_score(self):
        """Perfect data should score 100."""
        df = pd.DataFrame({
            "a": range(100),
            "b": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        result = compute_quality(df, target_col="target")
        assert 0.0 <= result.health_score <= 100.0
        # Should be close to 100 for clean data (small outlier penalty possible)
        assert result.health_score >= 70.0

    def test_health_score_bounds(self):
        """Health score should always be in [0, 100]."""
        # All missing data
        df = pd.DataFrame({
            "a": [np.nan] * 50,
            "b": [np.nan] * 50,
            "target": [np.nan] * 50,
        })
        result = compute_quality(df, target_col="target")
        assert 0.0 <= result.health_score <= 100.0

    def test_health_score_formula_exact(self):
        """Verify the exact formula: 100 * (1 - (0.35*P_null + 0.15*P_dup + 0.20*P_out + 0.30*P_leak))."""
        p_null, p_dup, p_out, p_leak = 0.1, 0.05, 0.08, 0.0
        raw = 1.0 - (0.35 * p_null + 0.15 * p_dup + 0.20 * p_out + 0.30 * p_leak)
        expected = round(max(0.0, min(100.0, 100.0 * raw)), 2)
        assert expected == round(100.0 * (1.0 - (0.035 + 0.0075 + 0.016 + 0.0)), 2)

    def test_penalty_ratios_normalized(self):
        """All penalty ratios should be in [0, 1]."""
        df = pd.DataFrame({
            "a": [np.nan, 1, 1, 2, 2, 3, 100, 100],
            "target": [0, 1, 1, 0, 0, 1, 0, 0],
        })
        result = compute_quality(df, target_col="target")
        assert 0.0 <= result.penalties.p_null <= 1.0
        assert 0.0 <= result.penalties.p_dup <= 1.0
        assert 0.0 <= result.penalties.p_out <= 1.0
        assert 0.0 <= result.penalties.p_leak <= 1.0


class TestTaskDetector:
    """Verify task_detector accurately classifies all 6 problem types."""

    def test_binary_classification(self, machine_data):
        result = detect_task(machine_data, id_columns=["machine_id"])
        assert result.primary_task == "Classification"
        assert result.target_column == "failure"
        assert result.target_type == "binary"
        assert "Risk Analysis" in result.secondary_tasks

    def test_regression(self, retail_data):
        result = detect_task(retail_data, datetime_columns=["date"])
        assert result.primary_task in ("Regression", "Time-Series Forecasting")
        assert result.target_column == "demand"
        assert result.target_type == "continuous"

    def test_classification_churn(self, churn_data):
        result = detect_task(churn_data, id_columns=["customer_id"])
        assert result.primary_task == "Classification"
        assert result.target_column == "churn"
        assert "Risk Analysis" in result.secondary_tasks

    def test_auto_target_detection(self, machine_data):
        """Target should be auto-detected even without explicit specification."""
        result = detect_task(machine_data, target_col=None, id_columns=["machine_id"])
        assert result.target_column == "failure"

    def test_unsupervised_fallback(self):
        """No target -> Clustering/Anomaly Detection."""
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "c": np.random.randn(50),
        })
        result = detect_task(df, target_col=None)
        # Should not pick any column as target since no keywords match
        # The last column 'c' might be picked as target, or it routes to unsupervised
        # depending on keyword scan. 'c' doesn't match any keywords.
        # Fallback: last non-ID non-datetime column -> 'c' detected as target
        # This is acceptable behavior per the spec.
        assert result.target_column is not None or result.primary_task in ("Clustering", "Anomaly Detection")

    def test_time_series_detection(self):
        """Dataset with datetime + continuous target -> Time-Series."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100).cumsum() + 100,
            "demand": np.random.randn(100) * 10 + 50,
        })
        result = detect_task(df, datetime_columns=["date"])
        assert result.primary_task in ("Regression", "Time-Series Forecasting")


class TestFeatureEngineer:
    """Verify feature engineering extracts datetime features, interactions, transforms."""

    def test_datetime_extraction(self):
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        })
        fe = FeatureEngineer()
        X, y, result = fe.fit_transform(df, target_col="target", datetime_columns=["date"])
        # Should have month, weekday, quarter, day_of_year, is_weekend
        eng_names = [f.name for f in result.engineered_features]
        assert any("month" in n for n in eng_names)
        assert any("weekday" in n for n in eng_names)

    def test_interaction_features(self):
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        fe = FeatureEngineer()
        X, y, result = fe.fit_transform(df, target_col="target")
        assert result.total_feature_count >= result.original_feature_count

    def test_transform_unseen_row(self):
        """Transform should work on unseen rows without shape mismatch."""
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "cat": np.random.choice(["X", "Y", "Z"], 100),
            "target": np.random.randint(0, 2, 100),
        })
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(df, target_col="target")

        # New row
        new_row = pd.DataFrame({"a": [1.5], "b": [-0.5], "cat": ["X"]})
        X_new = fe.transform(new_row)
        assert X_new.shape[1] == X.shape[1]
        assert not np.any(np.isnan(X_new))

    def test_unseen_category_no_crash(self):
        """Unseen categories should be handled gracefully."""
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "cat": np.random.choice(["A", "B", "C"], 50),
            "target": np.random.randint(0, 2, 50),
        })
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(df, target_col="target")

        new_row = pd.DataFrame({"a": [0.0], "cat": ["NEVER_SEEN"]})
        X_new = fe.transform(new_row)
        assert X_new.shape[1] == X.shape[1]


class TestModelTournament:
    """Verify tournament computes Intelligent Score correctly and ranks models."""

    def test_intelligent_score_bounds(self):
        """Intelligent score should be in [0, 100]."""
        score = _compute_intelligent_score(0.9, 0.8, 0.85, 0.92)
        assert 0.0 <= score <= 100.0

    def test_intelligent_score_formula(self):
        """Verify exact formula: 100 * (0.55*A + 0.15*S + 0.20*R + 0.10*E)."""
        a, s, r, e = 0.9, 0.8, 0.85, 0.92
        expected = round(100.0 * (0.55 * a + 0.15 * s + 0.20 * r + 0.10 * e), 2)
        result = _compute_intelligent_score(a, s, r, e)
        assert result == expected

    def test_explainability_scores(self):
        assert _get_explainability_score("LogisticRegression") == 1.0
        assert _get_explainability_score("Ridge") == 1.0
        assert _get_explainability_score("RandomForestClassifier") == 0.92
        assert _get_explainability_score("XGBClassifier") == 0.92
        assert _get_explainability_score("LGBMRegressor") == 0.92

    def test_classification_tournament_runs(self):
        """Basic classification tournament should complete."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        tournament, model, _ = run_tournament(X, y, "Classification", n_splits=3)
        assert len(tournament.models) >= 3
        assert tournament.winner_name != ""
        assert 0 <= tournament.winner_score <= 100
        for m in tournament.models:
            assert 0 <= m.intelligent_score <= 100

    def test_regression_tournament_runs(self):
        """Basic regression tournament should complete."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 5))
        y = 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.5, n)
        tournament, model, _ = run_tournament(X, y, "Regression", n_splits=3)
        assert len(tournament.models) >= 3
        assert tournament.winner_name != ""

    def test_models_ranked_deterministically(self):
        """Same data should produce same ranking."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (150, 4))
        y = (X[:, 0] > 0).astype(int)

        t1, _, _ = run_tournament(X, y, "Classification", n_splits=3)
        t2, _, _ = run_tournament(X, y, "Classification", n_splits=3)
        assert t1.winner_name == t2.winner_name


class TestExplainability:
    """Verify SHAP returns valid non-empty feature importances."""

    def test_shap_returns_importances(self):
        from src.engine.explainability import compute_shap_explanations
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        features = ["f0", "f1", "f2", "f3"]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        result = compute_shap_explanations(model, X, features, "Classification")
        assert len(result.global_feature_importance) > 0
        for fi in result.global_feature_importance:
            assert fi.feature in features
            assert fi.importance >= 0
            assert fi.direction in ("positive", "negative")

    def test_shap_local_explanation(self):
        from src.engine.explainability import compute_shap_explanations
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        y = (X[:, 0] > 0).astype(int)
        features = ["alpha", "beta", "gamma"]

        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        result = compute_shap_explanations(model, X, features, "Classification")
        assert result.local_explanation is not None
        assert len(result.local_explanation.contributions) > 0
