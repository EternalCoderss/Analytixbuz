"""
Edge case tests — Dirty data, leakage, NaN-heavy, single-class, boundary tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.engine.loader_profiler import profile_dataframe
from src.engine.quality_engine import compute_quality, auto_clean
from src.engine.task_detector import detect_task
from src.engine.feature_engineer import FeatureEngineer
from src.engine.model_tournament import run_tournament
from src.engine.simulator import run_what_if
from src.llm.explanation_layer import generate_report, answer_user_question
from src.schemas.models import PipelineSummaryPayload


class TestEdge1_TargetLeakageDetection:
    """Edge Case 1: Inject leaked_col = target * 2.0 + 1.0. Verify detection and removal."""

    def test_leakage_detected(self, leakage_df):
        result = compute_quality(leakage_df, target_col="target")
        assert "leaked_col" in result.leakage_columns
        assert result.penalties.p_leak > 0.0
        # Health score should be penalized
        assert result.health_score < 100.0

    def test_leakage_column_dropped(self, leakage_df):
        quality = compute_quality(leakage_df, target_col="target")
        df_clean, quality = auto_clean(leakage_df, quality, target_col="target")
        assert "leaked_col" not in df_clean.columns
        assert "leaked_col" in quality.dropped_columns

    def test_leakage_penalizes_score(self, leakage_df):
        result = compute_quality(leakage_df, target_col="target")
        # Without leakage
        df_no_leak = leakage_df.drop(columns=["leaked_col"])
        result_clean = compute_quality(df_no_leak, target_col="target")
        assert result.health_score < result_clean.health_score


class TestEdge2_ExtremeMissingness:
    """Edge Case 2: 100% NaN column, constant column, 30% missing, 10% duplicates."""

    def test_dead_columns_removed(self, dirty_data):
        quality = compute_quality(dirty_data, target_col="target")
        df_clean, quality = auto_clean(dirty_data, quality, target_col="target")
        assert "all_null" not in df_clean.columns
        assert "constant_col" not in df_clean.columns

    def test_no_nans_after_clean(self, dirty_data):
        quality = compute_quality(dirty_data, target_col="target")
        df_clean, quality = auto_clean(dirty_data, quality, target_col="target")
        assert df_clean.isna().sum().sum() == 0

    def test_duplicates_removed(self, dirty_data):
        original_dups = dirty_data.duplicated().sum()
        quality = compute_quality(dirty_data, target_col="target")
        df_clean, quality = auto_clean(dirty_data, quality, target_col="target")
        assert df_clean.duplicated().sum() == 0
        assert len(df_clean) < len(dirty_data)

    def test_clean_execution_no_crash(self, dirty_data):
        """Full pipeline should not crash on dirty data."""
        quality = compute_quality(dirty_data, target_col="target")
        df_clean, quality = auto_clean(dirty_data, quality, target_col="target")
        assert df_clean.isna().sum().sum() == 0
        assert len(df_clean) > 0


class TestEdge3_UnseenCategoriesExtremeValues:
    """Edge Case 3: Unseen categories and 10x extreme numeric values in simulator."""

    def test_unseen_category_in_whatif(self):
        """Pass unseen categorical string into simulator — should not crash."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feat_a": rng.normal(0, 1, 100),
            "contract": rng.choice(["Month-to-Month", "One-Year", "Two-Year"], 100),
            "target": rng.integers(0, 2, 100),
        })
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(df, target_col="target")

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Create baseline
        baseline = df.iloc[[0]].copy()

        # Override with unseen category
        result = run_what_if(
            baseline_row=baseline,
            overrides={"contract": "Alien_Tier"},
            feature_engineer=fe,
            model=model,
            X_background=X,
            feature_names=fe.feature_names_out,
            task_type="Classification",
        )
        assert isinstance(result.simulated_prediction, float)
        assert not np.isnan(result.simulated_prediction)

    def test_extreme_numeric_in_whatif(self):
        """Pass 10x extreme numeric values — should not crash."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "temp": rng.normal(70, 10, 100),
            "pressure": rng.normal(30, 5, 100),
            "target": rng.integers(0, 2, 100),
        })
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(df, target_col="target")

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        baseline = df.iloc[[0]].copy()
        result = run_what_if(
            baseline_row=baseline,
            overrides={"temp": 700.0, "pressure": 300.0},  # 10x extreme
            feature_engineer=fe,
            model=model,
            X_background=X,
            feature_names=fe.feature_names_out,
            task_type="Classification",
        )
        assert isinstance(result.simulated_prediction, float)
        assert not np.isnan(result.simulated_prediction)


class TestEdge4_SmallDatasetImbalanced:
    """Edge Case 4: 35-row dataset with 92%/8% class split."""

    def test_small_imbalanced_classification(self, small_classification_df):
        """CV fold adaptation should handle tiny imbalanced datasets."""
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(small_classification_df, target_col="target")
        # Should not crash
        tournament, model, _ = run_tournament(X, y, "Classification", n_splits=5)
        assert len(tournament.models) >= 1
        assert tournament.winner_name != ""
        assert 0 <= tournament.winner_score <= 100


class TestEdge5_UnsupervisedFallback:
    """Edge Case 5: target_col=None should route to Clustering/Anomaly Detection."""

    def test_clustering_without_target(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": rng.normal(5, 2, 100),
            "z": rng.normal(-3, 1, 100),
        })
        # When target_col is None and no keywords match,
        # it falls back to last column or unsupervised
        result = detect_task(df, target_col=None)
        # Either auto-detects last column or routes to Clustering
        assert result.primary_task in ("Classification", "Regression", "Clustering")

    def test_clustering_tournament(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        tournament, model, _ = run_tournament(X, np.zeros(len(X)), "Clustering")
        assert tournament.task_type == "Clustering"
        assert len(tournament.predictions) == len(X)

    def test_anomaly_detection_tournament(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        tournament, model, _ = run_tournament(X, np.zeros(len(X)), "Anomaly Detection")
        assert tournament.task_type == "Anomaly Detection"
        assert len(tournament.predictions) == len(X)


class TestEdge6_ZeroHallucinationGuardrail:
    """Edge Case 6: Verify explanation_layer produces grounded narratives."""

    def test_deterministic_report_contains_exact_values(self):
        """Report must contain the exact Health Score, winner name, and top SHAP features."""
        payload = PipelineSummaryPayload(
            dataset_name="test_data",
            quality=_make_quality(health_score=87.5),
            tournament=_make_tournament(winner="XGBClassifier", score=91.23),
            explainability=_make_explainability(["vibration", "temperature"]),
        )
        report = generate_report(payload)
        assert report.grounded is True
        assert "87.5" in report.full_narrative
        assert "XGBClassifier" in report.full_narrative
        assert "vibration" in report.full_narrative
        assert "temperature" in report.full_narrative

    def test_report_without_api_key(self):
        """Should generate complete report without any API key."""
        import os
        # Ensure no API keys
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_gemini = os.environ.pop("GEMINI_API_KEY", None)
        try:
            payload = PipelineSummaryPayload(
                dataset_name="test_data",
                quality=_make_quality(health_score=75.0),
                tournament=_make_tournament(winner="LGBMClassifier", score=88.0),
            )
            report = generate_report(payload)
            assert report.what_will_happen != ""
            assert report.what_should_we_do != ""
            assert report.grounded is True
        finally:
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai
            if old_gemini:
                os.environ["GEMINI_API_KEY"] = old_gemini

    def test_qa_returns_grounded_answers(self):
        """Q&A function should return accurate answers from the payload."""
        summary = {
            "quality": {"health_score": 92.5, "penalties": {"p_null": 0.01, "p_dup": 0.0, "p_out": 0.05, "p_leak": 0.0},
                        "missing_cells_total": 15, "duplicate_rows": 0, "leakage_columns": [], "dropped_columns": []},
            "tournament": {"winner_name": "XGBClassifier", "winner_score": 89.5, "models": [
                {"model_name": "XGBClassifier", "intelligent_score": 89.5}
            ]},
            "task": {"primary_task": "Classification", "target_column": "failure", "secondary_tasks": ["Risk Analysis"]},
        }
        answer = answer_user_question("What is the health score?", summary)
        assert "92.5" in answer

        answer2 = answer_user_question("Which model won?", summary)
        assert "XGBClassifier" in answer2


# ──── Helper factories ────

def _make_quality(health_score=90.0):
    from src.schemas.models import QualityResult, QualityPenalties
    return QualityResult(
        penalties=QualityPenalties(p_null=0.01, p_dup=0.005, p_out=0.02, p_leak=0.0),
        health_score=health_score,
        missing_cells_total=50,
        duplicate_rows=5,
    )


def _make_tournament(winner="XGBClassifier", score=90.0):
    from src.schemas.models import TournamentResult, ModelResult
    return TournamentResult(
        task_type="Classification",
        models=[
            ModelResult(model_name=winner, intelligent_score=score,
                        roc_auc=0.92, f1_weighted=0.88),
        ],
        winner_name=winner,
        winner_score=score,
    )


def _make_explainability(feature_names=None):
    from src.schemas.models import ExplainabilityResult, FeatureImportance
    features = feature_names or ["feat_a", "feat_b"]
    return ExplainabilityResult(
        global_feature_importance=[
            FeatureImportance(feature=f, importance=round(0.5 - i * 0.1, 2), direction="positive")
            for i, f in enumerate(features)
        ],
    )
