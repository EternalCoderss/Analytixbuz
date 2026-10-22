"""
FastAPI Production Backend — REST endpoints backed by the engine modules.
"""

from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.engine.orchestrator import get_session, run_pipeline
from src.engine.simulator import run_what_if
from src.llm.explanation_layer import answer_user_question
from src.schemas.models import PipelineExecutionResponse, WhatIfResult

import pandas as pd

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

app = FastAPI(
    title="Analytixbuz API",
    description="Autonomous Data Science Platform — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──── Request/Response Models ────

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = True


class WhatIfRequest(BaseModel):
    overrides: dict


# ──── Endpoints ────

@app.get("/health")
def health_check():
    """Health check and engine status."""
    return {
        "status": "healthy",
        "engine": "analytixbuz",
        "version": "1.0.0",
    }


@app.get("/api/v1/demos")
def list_demos():
    """List available pre-built industry datasets."""
    demos = [
        {
            "name": "manufacturing",
            "file": "machine_data.csv",
            "description": "Manufacturing: Predictive Maintenance — Binary classification of machine failure",
            "target": "failure",
            "industry": "Manufacturing",
        },
        {
            "name": "retail",
            "file": "retail_demand.csv",
            "description": "Retail: Demand Forecasting — Regression/Time-Series prediction of demand",
            "target": "demand",
            "industry": "Retail",
        },
        {
            "name": "churn",
            "file": "customer_churn.csv",
            "description": "Customer Analytics: Churn & Risk Analysis — Binary classification of customer churn",
            "target": "churn",
            "industry": "Customer Analytics",
        },
    ]
    return {"demos": demos}


@app.post("/api/v1/demo/{demo_name}", response_model=PipelineExecutionResponse)
def run_demo(demo_name: str):
    """Run full 12-step autonomous pipeline on a demo dataset."""
    demo_map = {
        "manufacturing": ("machine_data.csv", "Manufacturing: Predictive Maintenance"),
        "retail": ("retail_demand.csv", "Retail: Demand Forecasting"),
        "churn": ("customer_churn.csv", "Customer Analytics: Churn Analysis"),
    }

    if demo_name not in demo_map:
        raise HTTPException(status_code=404, detail=f"Demo '{demo_name}' not found. Available: {list(demo_map.keys())}")

    filename, display_name = demo_map[demo_name]
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {filepath}")

    result = run_pipeline(filepath, dataset_name=display_name)
    return result


@app.post("/api/v1/upload", response_model=PipelineExecutionResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(None, description="Optional target column name"),
):
    """Accept CSV upload, run full 12-step pipeline."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    buffer = io.BytesIO(contents)
    dataset_name = file.filename.rsplit(".", 1)[0] if file.filename else "uploaded"

    result = run_pipeline(
        buffer,
        target_col=target_column,
        dataset_name=dataset_name,
    )
    return result


@app.post("/api/v1/what-if/{session_id}", response_model=WhatIfResult)
def what_if_simulation(session_id: str, request: WhatIfRequest):
    """Run What-If simulation on a previous pipeline session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.best_model is None or session.feature_engineer is None:
        raise HTTPException(status_code=400, detail="Session does not have a trained model.")

    if session.df_clean is None:
        raise HTTPException(status_code=400, detail="Session has no clean data.")

    # Get baseline row (highest-risk or median)
    if session.is_risk_analysis and session.response and session.response.tournament:
        probs = session.response.tournament.probabilities
        if probs:
            import numpy as np
            high_risk_idx = int(np.argmax(probs))
            baseline_row = session.df_clean.iloc[[high_risk_idx]].copy()
        else:
            baseline_row = session.df_clean.iloc[[0]].copy()
    else:
        baseline_row = session.df_clean.iloc[[0]].copy()

    result = run_what_if(
        baseline_row=baseline_row,
        overrides=request.overrides,
        feature_engineer=session.feature_engineer,
        model=session.best_model,
        X_background=session.X_transformed,
        feature_names=session.feature_names,
        task_type=session.task_type,
        is_risk_analysis=session.is_risk_analysis,
    )
    return result


@app.post("/api/v1/chat/{session_id}", response_model=ChatResponse)
def chat_with_data_scientist(session_id: str, request: ChatRequest):
    """Answer natural-language questions about the analyzed dataset."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.pipeline_summary is None:
        raise HTTPException(status_code=400, detail="Pipeline has not completed for this session.")

    # Convert payload to dict for the Q&A function
    summary_dict = session.pipeline_summary.model_dump()

    # Also include report if available
    if session.response and session.response.report:
        summary_dict["report"] = session.response.report.model_dump()

    answer = answer_user_question(request.question, summary_dict)

    return ChatResponse(answer=answer, grounded=True)
