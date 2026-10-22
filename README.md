# 🧠 Analytixbuz — Autonomous AI Data Scientist Platform

> **"Not every company can afford a data scientist. Analytixbuz gives every company an AI one."**

Built for **SIH 2026 / BITS Pilani Internal Hackathon**

---

## 🌟 What is Analytixbuz?

Analytixbuz is an end-to-end **Autonomous Data Science Platform** that transforms raw CSV data into actionable business insights through a fully automated 12-step pipeline — **no coding required**.

### The 12-Step Autonomous Pipeline

```
[1. Upload] → [2. Understand] → [3. Quality Check] → [4. Explore (EDA)]
    → [5. Detect Problem] → [6. Engineer Features] → [7. Train Models]
    → [8. Compare] → [9. Predict] → [10. Explain] → [11. Recommend]
    → [12. Report] + [Live What-If Simulator]
```

### Hybrid Neuro-Symbolic Trust Architecture

- **Analytics Engine** (Deterministic Python): Computes 100% of real statistics, ML models, SHAP values
- **AI Explanation Layer** (Grounded LLM/Fallback): Translates computed metrics into plain-English — never invents numbers
- **Offline/Zero-Key Resilience**: Works fully without any API keys

---

## 🚀 Quick Start (Run in 2 Steps)

### Option 1: Direct Local Run (Recommended for Judges)

Datasets are already included in `data/`, so you can launch the interactive UI immediately:

```bash
# 1. Clone & install
git clone https://github.com/EternalCoderss/Analytixbuz.git
cd Analytixbuz
pip install -r requirements.txt

# 2. Launch Dashboard (Browser will automatically pop open!)
streamlit run src/ui/streamlit_app.py
```

> 🌐 **Browser UI will automatically open at:** `http://localhost:8501`  
> In the sidebar, select any demo dataset (🏭 Manufacturing, 🛒 Retail, or 👤 Churn) and click **"Run Autonomous Pipeline"**!

---

### Option 2: Run with Docker (1-Click)

```bash
docker-compose up --build
```
Open **`http://localhost:8501`** in your browser.

---

### Optional Commands

- **Run FastAPI REST Backend**: `uvicorn src.api.main:app --reload --port 8000` (Docs at `http://localhost:8000/docs`)
- **Run Full Pytest Suite**: `pytest tests/ -v`
- **Regenerate Demo Data**: `python data/generate_demo_datasets.py`

---

## 📊 Demo Datasets (3 Industries)

| Industry | Dataset | Target | Problem Type |
|----------|---------|--------|--------------|
| 🏭 Manufacturing | `machine_data.csv` | `failure` (0/1) | Classification + Risk Analysis |
| 🛒 Retail | `retail_demand.csv` | `demand` (continuous) | Regression / Time-Series |
| 👤 Customer Analytics | `customer_churn.csv` | `churn` (0/1) | Classification + Risk Analysis |

---

## 🏗️ Architecture

```
analytixbuz/
├── data/                          # Demo datasets & generator
├── src/
│   ├── schemas/models.py          # Pydantic v2 schemas for all 12 stages
│   ├── engine/
│   │   ├── loader_profiler.py     # Steps 1-2: Ingestion & profiling
│   │   ├── quality_engine.py      # Step 3: Data Health Score & auto-clean
│   │   ├── auto_eda.py            # Step 4: Automated EDA
│   │   ├── task_detector.py       # Step 5: Problem-type detection (6 types)
│   │   ├── feature_engineer.py    # Step 6: Auto feature engineering
│   │   ├── model_tournament.py    # Steps 7-9: Multi-model tournament
│   │   ├── explainability.py      # Step 10: SHAP global & local
│   │   ├── simulator.py           # Live What-If engine
│   │   └── orchestrator.py        # Pipeline runner & session manager
│   ├── llm/
│   │   └── explanation_layer.py   # Steps 11-12: Grounded AI reports + Q&A
│   ├── api/main.py                # FastAPI REST backend
│   └── ui/streamlit_app.py        # Streamlit Judge Demo Dashboard
├── tests/                         # Comprehensive pytest suite
├── Dockerfile & docker-compose.yml
└── requirements.txt
```

---

## 📐 Key Formulas

### Data Health Score (0-100)

```
Health_Score = 100 × (1 - (0.35×P_null + 0.15×P_dup + 0.20×P_out + 0.30×P_leak))
```

### Intelligent Model Score (0-100)

```
Intelligent_Score = 100 × (0.55×Accuracy_Norm + 0.15×Speed + 0.20×Robustness + 0.10×Explainability)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/demos` | List demo datasets |
| POST | `/api/v1/demo/{name}` | Run demo pipeline |
| POST | `/api/v1/upload` | Upload CSV & analyze |
| POST | `/api/v1/what-if/{session_id}` | What-If simulation |
| POST | `/api/v1/chat/{session_id}` | Grounded Q&A |

---

## 🧪 Test Coverage

- **Unit Tests**: Health Score math, task detection, feature engineering, model scoring, SHAP
- **Edge Cases**: Target leakage, 100% NaN columns, unseen categories, 35-row imbalanced data, unsupervised fallback, hallucination guard
- **User Journeys**: All 3 industry demos end-to-end with metric assertions
- **API Integration**: FastAPI TestClient for all endpoints

---

## 📄 License

MIT License — Built for SIH 2026 Hackathon
