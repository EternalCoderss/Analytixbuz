"""
Pydantic v2 schemas for all 12 pipeline stages of Analytixbuz.
Every data transfer between engine modules uses these strict typed models.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Step 1-2: Data Loading & Profiling
# ──────────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str  # numeric_continuous, numeric_discrete, categorical, boolean, datetime, identifier
    unique_count: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    sample_values: list[Any] = Field(default_factory=list)


class DataProfileResult(BaseModel):
    n_rows: int
    n_cols: int
    memory_mb: float
    columns: list[ColumnProfile]
    id_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Step 3: Data Quality Engine
# ──────────────────────────────────────────────────────────────

class QualityPenalties(BaseModel):
    p_null: float = Field(ge=0.0, le=1.0)
    p_dup: float = Field(ge=0.0, le=1.0)
    p_out: float = Field(ge=0.0, le=1.0)
    p_leak: float = Field(ge=0.0, le=1.0)


class CleaningAction(BaseModel):
    action: str
    details: str
    affected_rows: int = 0
    affected_columns: int = 0


class QualityResult(BaseModel):
    penalties: QualityPenalties
    health_score: float = Field(ge=0.0, le=100.0)
    missing_cells_total: int = 0
    duplicate_rows: int = 0
    outlier_summary: dict[str, int] = Field(default_factory=dict)
    leakage_columns: list[str] = Field(default_factory=list)
    dropped_columns: list[str] = Field(default_factory=list)
    cleaning_log: list[CleaningAction] = Field(default_factory=list)
    rows_after_clean: int = 0
    cols_after_clean: int = 0


# ──────────────────────────────────────────────────────────────
# Step 4: Automated EDA
# ──────────────────────────────────────────────────────────────

class ColumnStats(BaseModel):
    name: str
    count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    missing_pct: float = 0.0


class CorrelationPair(BaseModel):
    feature_a: str
    feature_b: str
    pearson: Optional[float] = None
    spearman: Optional[float] = None


class EDAResult(BaseModel):
    column_stats: list[ColumnStats] = Field(default_factory=list)
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    target_correlations: list[CorrelationPair] = Field(default_factory=list)
    takeaways: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Step 5: Task Detection
# ──────────────────────────────────────────────────────────────

class TaskDetectionResult(BaseModel):
    primary_task: str  # Classification, Regression, Time-Series Forecasting, Anomaly Detection, Clustering
    secondary_tasks: list[str] = Field(default_factory=list)  # e.g., Risk Analysis
    target_column: Optional[str] = None
    target_type: Optional[str] = None  # binary, multiclass, continuous
    n_classes: Optional[int] = None
    class_distribution: dict[str, int] = Field(default_factory=dict)
    detection_reasoning: str = ""


# ──────────────────────────────────────────────────────────────
# Step 6: Feature Engineering
# ──────────────────────────────────────────────────────────────

class EngineeredFeatureInfo(BaseModel):
    name: str
    source: str  # e.g., "datetime_extraction", "ratio", "interaction", "rolling"
    description: str = ""


class FeatureEngineeringResult(BaseModel):
    original_feature_count: int = 0
    engineered_feature_count: int = 0
    total_feature_count: int = 0
    engineered_features: list[EngineeredFeatureInfo] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Steps 7-9: Model Tournament
# ──────────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    model_name: str
    accuracy_norm: float = 0.0
    speed_score: float = 0.0
    robustness_score: float = 0.0
    explainability_score: float = 0.0
    intelligent_score: float = 0.0
    # Raw metrics
    roc_auc: Optional[float] = None
    f1_weighted: Optional[float] = None
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    train_metric: float = 0.0
    val_metric: float = 0.0
    inference_latency_ms: float = 0.0
    cv_scores: list[float] = Field(default_factory=list)


class TournamentResult(BaseModel):
    task_type: str
    models: list[ModelResult] = Field(default_factory=list)
    winner_name: str = ""
    winner_score: float = 0.0
    predictions: list[Any] = Field(default_factory=list)
    probabilities: list[float] = Field(default_factory=list)
    risk_tiers: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Step 10: Explainability (SHAP)
# ──────────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str = "positive"  # positive or negative


class LocalExplanation(BaseModel):
    row_index: int = 0
    base_value: float = 0.0
    predicted_value: float = 0.0
    contributions: list[FeatureImportance] = Field(default_factory=list)
    narrative: str = ""


class ExplainabilityResult(BaseModel):
    global_feature_importance: list[FeatureImportance] = Field(default_factory=list)
    local_explanation: Optional[LocalExplanation] = None
    shap_values_sample: list[list[float]] = Field(default_factory=list)  # For plotting


# ──────────────────────────────────────────────────────────────
# What-If Simulator
# ──────────────────────────────────────────────────────────────

class WhatIfResult(BaseModel):
    baseline_prediction: float = 0.0
    simulated_prediction: float = 0.0
    delta_value: float = 0.0
    delta_pct: float = 0.0
    baseline_risk_tier: str = ""
    simulated_risk_tier: str = ""
    updated_local_shap: list[FeatureImportance] = Field(default_factory=list)
    recommendation: str = ""


# ──────────────────────────────────────────────────────────────
# Steps 11-12: Report & Recommendations
# ──────────────────────────────────────────────────────────────

class AIReport(BaseModel):
    what_will_happen: str = ""
    why_will_it_happen: str = ""
    what_should_we_do: str = ""
    full_narrative: str = ""
    recommendations: list[str] = Field(default_factory=list)
    grounded: bool = True  # True if all numbers verified against payload


# ──────────────────────────────────────────────────────────────
# Full Pipeline Response
# ──────────────────────────────────────────────────────────────

class PipelineSummaryPayload(BaseModel):
    """Aggregated payload sent to the LLM explanation layer."""
    dataset_name: str = ""
    profile: Optional[DataProfileResult] = None
    quality: Optional[QualityResult] = None
    eda: Optional[EDAResult] = None
    task: Optional[TaskDetectionResult] = None
    features: Optional[FeatureEngineeringResult] = None
    tournament: Optional[TournamentResult] = None
    explainability: Optional[ExplainabilityResult] = None


class PipelineExecutionResponse(BaseModel):
    session_id: str
    dataset_name: str = ""
    status: str = "completed"
    profile: Optional[DataProfileResult] = None
    quality: Optional[QualityResult] = None
    eda: Optional[EDAResult] = None
    task: Optional[TaskDetectionResult] = None
    features: Optional[FeatureEngineeringResult] = None
    tournament: Optional[TournamentResult] = None
    explainability: Optional[ExplainabilityResult] = None
    report: Optional[AIReport] = None
    errors: list[str] = Field(default_factory=list)
