"""
End-to-end 12-step pipeline runner & session manager.
Orchestrates all engine modules into a single autonomous pipeline.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.engine.auto_eda import run_eda
from src.engine.explainability import compute_shap_explanations
from src.engine.feature_engineer import FeatureEngineer
from src.engine.loader_profiler import (
    convert_datetime_columns,
    load_dataframe,
    profile_dataframe,
)
from src.engine.model_tournament import run_tournament
from src.engine.quality_engine import auto_clean, compute_quality
from src.engine.task_detector import detect_task
from src.llm.explanation_layer import generate_report
from src.schemas.models import (
    AIReport,
    PipelineExecutionResponse,
    PipelineSummaryPayload,
)


class PipelineSession:
    """Holds state for a single pipeline execution."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.best_model: Any = None
        self.X_transformed: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self.task_type: str = ""
        self.is_risk_analysis: bool = False
        self.target_col: Optional[str] = None
        self.pipeline_summary: Optional[PipelineSummaryPayload] = None
        self.response: Optional[PipelineExecutionResponse] = None


# Global session store
_sessions: dict[str, PipelineSession] = {}


def get_session(session_id: str) -> Optional[PipelineSession]:
    """Retrieve a pipeline session by ID."""
    return _sessions.get(session_id)


def run_pipeline(
    source,
    target_col: Optional[str] = None,
    dataset_name: str = "uploaded_dataset",
) -> PipelineExecutionResponse:
    """
    Run the full 12-step autonomous pipeline.
    Returns a complete PipelineExecutionResponse.
    """
    session = PipelineSession()
    errors = []

    try:
        # ═══ Step 1: Load ═══
        df = load_dataframe(source)
        session.df_raw = df.copy()

        # ═══ Step 2: Understand / Profile ═══
        profile = profile_dataframe(df)

        # Convert datetime columns
        df = convert_datetime_columns(df, profile.datetime_columns)

        # ═══ Step 5: Detect Problem Type (early, needed for quality) ═══
        task_result = detect_task(
            df,
            target_col=target_col,
            id_columns=profile.id_columns,
            datetime_columns=profile.datetime_columns,
        )
        session.target_col = task_result.target_column
        session.task_type = task_result.primary_task
        session.is_risk_analysis = "Risk Analysis" in task_result.secondary_tasks

        # ═══ Step 3: Quality Check ═══
        quality = compute_quality(
            df,
            target_col=task_result.target_column,
            id_columns=profile.id_columns,
        )

        # Auto-clean
        df, quality = auto_clean(
            df, quality,
            target_col=task_result.target_column,
            id_columns=profile.id_columns,
        )
        session.df_clean = df.copy()

        # ═══ Step 4: Automated EDA ═══
        eda = run_eda(
            df,
            target_col=task_result.target_column,
            id_columns=profile.id_columns,
        )

        # ═══ Step 6: Feature Engineering ═══
        fe = FeatureEngineer()
        X, y, fe_result = fe.fit_transform(
            df,
            target_col=task_result.target_column,
            id_columns=profile.id_columns,
            datetime_columns=profile.datetime_columns,
            task_type=task_result.primary_task,
        )
        session.feature_engineer = fe
        session.feature_names = fe.feature_names_out
        session.X_transformed = X
        session.y = y

        # ═══ Steps 7-9: Model Tournament ═══
        tournament = None
        best_model = None
        X_for_shap = X

        if task_result.target_column and y is not None and len(y) > 0:
            tournament, best_model, X_for_shap = run_tournament(
                X, y,
                task_type=task_result.primary_task,
                feature_names=fe.feature_names_out,
                is_risk_analysis=session.is_risk_analysis,
            )
            session.best_model = best_model
        else:
            # Unsupervised
            tournament, best_model, X_for_shap = run_tournament(
                X, np.zeros(len(X)),
                task_type=task_result.primary_task,
                feature_names=fe.feature_names_out,
            )
            session.best_model = best_model

        # ═══ Step 10: Explainability (SHAP) ═══
        explainability = None
        if best_model is not None and task_result.primary_task not in ("Clustering", "Anomaly Detection"):
            try:
                explainability = compute_shap_explanations(
                    best_model, X_for_shap,
                    fe.feature_names_out,
                    task_type=task_result.primary_task,
                )
            except Exception as e:
                errors.append(f"SHAP computation warning: {str(e)}")

        # ═══ Steps 11-12: Report & Recommendations ═══
        summary_payload = PipelineSummaryPayload(
            dataset_name=dataset_name,
            profile=profile,
            quality=quality,
            eda=eda,
            task=task_result,
            features=fe_result,
            tournament=tournament,
            explainability=explainability,
        )
        session.pipeline_summary = summary_payload

        report = generate_report(summary_payload)

        # Build response
        response = PipelineExecutionResponse(
            session_id=session.session_id,
            dataset_name=dataset_name,
            status="completed",
            profile=profile,
            quality=quality,
            eda=eda,
            task=task_result,
            features=fe_result,
            tournament=tournament,
            explainability=explainability,
            report=report,
            errors=errors,
        )
        session.response = response

    except Exception as e:
        response = PipelineExecutionResponse(
            session_id=session.session_id,
            dataset_name=dataset_name,
            status="error",
            errors=[str(e)],
        )
        session.response = response

    # Store session
    _sessions[session.session_id] = session
    return response
