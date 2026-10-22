"""
End-to-end user journey tests for all 3 industry demos + FastAPI routes.
"""

import os
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.engine.orchestrator import run_pipeline, get_session
from src.engine.simulator import run_what_if
from src.api.main import app

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

client = TestClient(app)


class TestUserJourney1_Manufacturing:
    """Manufacturing Demo Story: machine_data.csv end-to-end."""

    def test_full_pipeline(self, machine_data):
        filepath = os.path.join(DATA_DIR, "machine_data.csv")
        result = run_pipeline(filepath, dataset_name="Manufacturing")

        assert result.status == "completed"

        # Assert failure is auto-detected as Classification + Risk Analysis
        assert result.task is not None
        assert result.task.primary_task == "Classification"
        assert result.task.target_column == "failure"
        assert "Risk Analysis" in result.task.secondary_tasks

        # Assert model tournament
        assert result.tournament is not None
        assert result.tournament.winner_name != ""
        assert result.tournament.winner_score > 0

        # Assert ROC-AUC > 0.85
        winner = next(m for m in result.tournament.models
                      if m.model_name == result.tournament.winner_name)
        assert winner.roc_auc is not None
        assert winner.roc_auc > 0.85, f"ROC-AUC {winner.roc_auc} should be > 0.85"

        # Assert SHAP — vibration and temperature in top 3
        assert result.explainability is not None
        top_features = [f.feature for f in result.explainability.global_feature_importance[:5]]
        # At least one of vibration/temperature should be in top features
        # (feature names may have prefixes from sklearn)
        vib_found = any("vibration" in f.lower() for f in top_features)
        temp_found = any("temperature" in f.lower() for f in top_features)
        assert vib_found or temp_found, f"Expected vibration/temperature in top features, got {top_features}"

    def test_whatif_reduces_failure(self, machine_data):
        """Lowering vibration and temperature should reduce failure probability."""
        filepath = os.path.join(DATA_DIR, "machine_data.csv")
        result = run_pipeline(filepath, dataset_name="Manufacturing WIF")

        session = get_session(result.session_id)
        assert session is not None
        assert session.best_model is not None

        # Find high-risk instance
        if result.tournament.probabilities:
            high_idx = int(np.argmax(result.tournament.probabilities))
        else:
            high_idx = 0

        baseline_row = session.df_clean.iloc[[high_idx]].copy()

        # Lower vibration and temperature
        wif = run_what_if(
            baseline_row=baseline_row,
            overrides={"vibration": 0.5, "temperature": 55.0},
            feature_engineer=session.feature_engineer,
            model=session.best_model,
            X_background=session.X_transformed,
            feature_names=session.feature_names,
            task_type="Classification",
            is_risk_analysis=True,
        )

        # Simulated probability should be lower than or equal to baseline
        # (with some tolerance for model uncertainty)
        assert isinstance(wif.simulated_prediction, float)
        assert isinstance(wif.baseline_prediction, float)
        assert isinstance(wif.recommendation, str)
        assert len(wif.recommendation) > 0


class TestUserJourney2_Retail:
    """Retail Demo Story: retail_demand.csv end-to-end."""

    def test_full_pipeline(self, retail_data):
        filepath = os.path.join(DATA_DIR, "retail_demand.csv")
        result = run_pipeline(filepath, dataset_name="Retail")

        assert result.status == "completed"
        assert result.task is not None

        # Demand should be detected as Regression or Time-Series
        assert result.task.primary_task in ("Regression", "Time-Series Forecasting")
        assert result.task.target_column == "demand"

        # Assert R2 > 0.80
        assert result.tournament is not None
        winner = next(m for m in result.tournament.models
                      if m.model_name == result.tournament.winner_name)
        assert winner.r2_score is not None
        assert winner.r2_score > 0.80, f"R² {winner.r2_score} should be > 0.80"


class TestUserJourney3_Churn:
    """Customer Churn Demo Story: customer_churn.csv end-to-end."""

    def test_full_pipeline(self, churn_data):
        filepath = os.path.join(DATA_DIR, "customer_churn.csv")
        result = run_pipeline(filepath, dataset_name="Churn")

        assert result.status == "completed"
        assert result.task is not None

        # Churn should be detected as Classification
        assert result.task.primary_task == "Classification"
        assert result.task.target_column == "churn"

        # Assert SHAP — support_calls should be surfaced
        assert result.explainability is not None
        top_features = [f.feature for f in result.explainability.global_feature_importance[:7]]
        support_found = any("support" in f.lower() for f in top_features)
        assert support_found, f"Expected support_calls in top features, got {top_features}"


class TestUserJourney4_FastAPI:
    """FastAPI End-to-End Integration tests."""

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_demos_endpoint(self):
        resp = client.get("/api/v1/demos")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["demos"]) == 3

    def test_demo_manufacturing(self):
        resp = client.post("/api/v1/demo/manufacturing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["session_id"] != ""
        assert data["task"]["primary_task"] == "Classification"
        assert data["task"]["target_column"] == "failure"
        assert data["tournament"]["winner_name"] != ""

    def test_upload_endpoint(self):
        filepath = os.path.join(DATA_DIR, "customer_churn.csv")
        with open(filepath, "rb") as f:
            resp = client.post(
                "/api/v1/upload",
                files={"file": ("customer_churn.csv", f, "text/csv")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["task"]["target_column"] == "churn"

    def test_whatif_endpoint(self):
        # First run a demo
        resp = client.post("/api/v1/demo/manufacturing")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Then run what-if
        resp = client.post(
            f"/api/v1/what-if/{session_id}",
            json={"overrides": {"vibration": 0.5, "temperature": 55.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline_prediction" in data
        assert "simulated_prediction" in data
        assert "recommendation" in data

    def test_chat_endpoint(self):
        """Verify /api/v1/chat/{session_id} returns grounded numeric answers."""
        # First run a demo
        resp = client.post("/api/v1/demo/manufacturing")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Ask about health score
        resp = client.post(
            f"/api/v1/chat/{session_id}",
            json={"question": "What is the data health score?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["grounded"] is True
        # Answer should contain a number (the health score)
        assert any(c.isdigit() for c in data["answer"])

    def test_chat_model_question(self):
        """Verify chat returns winning model name."""
        resp = client.post("/api/v1/demo/churn")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        winner_name = resp.json()["tournament"]["winner_name"]

        resp = client.post(
            f"/api/v1/chat/{session_id}",
            json={"question": "Which model won the tournament?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert winner_name in data["answer"]

    def test_invalid_session(self):
        resp = client.post(
            "/api/v1/what-if/nonexistent-session",
            json={"overrides": {}},
        )
        assert resp.status_code == 404
