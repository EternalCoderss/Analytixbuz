"""
Analytixbuz — Full Streamlit Judge Demo Dashboard
Interactive 12-step autonomous data science platform with 7 tabs.
"""

from __future__ import annotations

import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.engine.orchestrator import run_pipeline, get_session
from src.engine.simulator import run_what_if
from src.llm.explanation_layer import answer_user_question

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytixbuz — AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ──
    st.markdown('<div class="main-header">🧠 Analytixbuz</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Autonomous AI Data Scientist — '
        '"Not every company can afford a data scientist. Analytixbuz gives every company an AI one."</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("⚙️ Configuration")

        data_source = st.radio(
            "📊 Select Data Source",
            ["🏭 Manufacturing (Predictive Maintenance)",
             "🛒 Retail (Demand Forecasting)",
             "👤 Customer Analytics (Churn)",
             "📁 Upload Custom CSV"],
        )

        target_override = st.text_input(
            "🎯 Target Column (optional)",
            value="",
            placeholder="Auto-Detect",
            help="Leave empty for automatic detection",
        )

        run_button = st.button("🚀 Run Autonomous Data Scientist", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### 📋 12-Step Pipeline")
        steps = [
            "1️⃣ Upload", "2️⃣ Understand", "3️⃣ Quality Check",
            "4️⃣ Explore (EDA)", "5️⃣ Detect Problem", "6️⃣ Engineer Features",
            "7️⃣ Train Models", "8️⃣ Compare", "9️⃣ Predict",
            "🔟 Explain", "1️⃣1️⃣ Recommend", "1️⃣2️⃣ Report",
        ]
        for step in steps:
            st.markdown(f"  {step}")

    # ── Main Content ──
    if run_button:
        _run_analysis(data_source, target_override)
    elif "result" in st.session_state:
        _display_results()
    else:
        _show_welcome()


def _show_welcome():
    """Show welcome screen when no analysis has been run."""
    st.info("👈 Select a dataset from the sidebar and click **Run Autonomous Data Scientist** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏭 Manufacturing")
        st.markdown("Predict machine failures before they happen. Reduce downtime by up to 40%.")
    with col2:
        st.markdown("### 🛒 Retail")
        st.markdown("Forecast demand accurately. Optimize inventory and reduce waste.")
    with col3:
        st.markdown("### 👤 Customer Analytics")
        st.markdown("Identify at-risk customers. Prevent churn with targeted interventions.")


def _run_analysis(data_source: str, target_override: str):
    """Run the full pipeline and display results."""
    # Determine file path
    if "Manufacturing" in data_source:
        filepath = os.path.join(DATA_DIR, "machine_data.csv")
        name = "Manufacturing: Predictive Maintenance"
    elif "Retail" in data_source:
        filepath = os.path.join(DATA_DIR, "retail_demand.csv")
        name = "Retail: Demand Forecasting"
    elif "Customer" in data_source:
        filepath = os.path.join(DATA_DIR, "customer_churn.csv")
        name = "Customer Analytics: Churn Analysis"
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV file.")
            return
        filepath = uploaded
        name = uploaded.name.rsplit(".", 1)[0] if uploaded.name else "uploaded"

    target = target_override.strip() if target_override.strip() else None

    with st.spinner("🧠 Running 12-Step Autonomous Pipeline... This may take a moment."):
        result = run_pipeline(filepath, target_col=target, dataset_name=name)

    st.session_state["result"] = result
    st.session_state["session_id"] = result.session_id

    if result.status == "error":
        st.error(f"Pipeline failed: {result.errors}")
        return

    st.success(f"✅ Pipeline completed! Session: `{result.session_id}`")
    _display_results()


def _display_results():
    """Display all pipeline results in tabs."""
    result = st.session_state["result"]

    tabs = st.tabs([
        "📊 Executive Overview",
        "🔍 EDA & Problem Detection",
        "🏆 Model Tournament",
        "🎯 Predictions & SHAP",
        "🔮 What-If Simulator",
        "📝 AI Report",
        "💬 Ask AI Data Scientist",
    ])

    with tabs[0]:
        _tab_overview(result)
    with tabs[1]:
        _tab_eda(result)
    with tabs[2]:
        _tab_tournament(result)
    with tabs[3]:
        _tab_predictions_shap(result)
    with tabs[4]:
        _tab_whatif(result)
    with tabs[5]:
        _tab_report(result)
    with tabs[6]:
        _tab_chat(result)


def _tab_overview(result):
    """Tab 1: Executive Overview & Data Health."""
    st.header("📊 Executive Overview & Data Health")

    if result.profile:
        p = result.profile
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📏 Rows", f"{p.n_rows:,}")
        col2.metric("📐 Columns", p.n_cols)
        col3.metric("💾 Memory", f"{p.memory_mb:.2f} MB")
        col4.metric("🏷️ ID Columns", len(p.id_columns))

    if result.quality:
        q = result.quality
        st.subheader("🏥 Data Health Score")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=q.health_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Data Health Score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 40], "color": "#ff4444"},
                    {"range": [40, 70], "color": "#ffaa00"},
                    {"range": [70, 100], "color": "#00cc66"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": q.health_score,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Penalty breakdown
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🕳️ P_null", f"{q.penalties.p_null:.4f}", help="Missing data penalty")
        col2.metric("👯 P_dup", f"{q.penalties.p_dup:.4f}", help="Duplicate rows penalty")
        col3.metric("📊 P_out", f"{q.penalties.p_out:.4f}", help="Outliers penalty")
        col4.metric("🔗 P_leak", f"{q.penalties.p_leak:.4f}", help="Target leakage penalty")

        # Cleaning log
        if q.cleaning_log:
            st.subheader("🧹 Auto-Cleaning Audit Log")
            log_data = [{"Action": a.action, "Details": a.details,
                         "Affected Rows": a.affected_rows, "Affected Cols": a.affected_columns}
                        for a in q.cleaning_log]
            st.dataframe(pd.DataFrame(log_data), use_container_width=True)


def _tab_eda(result):
    """Tab 2: EDA & Problem Detection."""
    st.header("🔍 Automated EDA & Problem Detection")

    if result.task:
        t = result.task
        col1, col2 = st.columns(2)
        col1.metric("🎯 Problem Type", t.primary_task)
        col2.metric("🏷️ Target Column", t.target_column or "None")

        if t.secondary_tasks:
            st.info(f"Secondary tasks: {', '.join(t.secondary_tasks)}")
        st.caption(t.detection_reasoning)

        if t.class_distribution:
            st.subheader("Target Distribution")
            dist_df = pd.DataFrame([
                {"Class": k, "Count": v} for k, v in t.class_distribution.items()
            ])
            fig = px.bar(dist_df, x="Class", y="Count", color="Class",
                         title="Target Variable Distribution")
            st.plotly_chart(fig, use_container_width=True)

    if result.eda:
        # Correlation heatmap
        if result.eda.top_correlations:
            st.subheader("🔥 Feature Correlations")
            corr_data = []
            for c in result.eda.top_correlations[:10]:
                corr_data.append({
                    "Feature A": c.feature_a,
                    "Feature B": c.feature_b,
                    "Pearson": c.pearson,
                    "Spearman": c.spearman,
                })
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

        # Takeaways
        if result.eda.takeaways:
            st.subheader("📝 Statistical Takeaways")
            for t in result.eda.takeaways:
                st.markdown(f"- {t}")

    if result.features:
        st.subheader("🔧 Engineered Features")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Features", result.features.original_feature_count)
        col2.metric("Engineered", result.features.engineered_feature_count)
        col3.metric("Total Features", result.features.total_feature_count)

        if result.features.engineered_features:
            eng_data = [{"Name": f.name, "Source": f.source, "Description": f.description}
                        for f in result.features.engineered_features]
            st.dataframe(pd.DataFrame(eng_data), use_container_width=True)


def _tab_tournament(result):
    """Tab 3: Model Tournament & Intelligent Score."""
    st.header("🏆 Model Tournament & Intelligent Score")

    if not result.tournament:
        st.warning("No model tournament results available.")
        return

    tm = result.tournament
    st.success(f"🥇 Winner: **{tm.winner_name}** with Intelligent Score **{tm.winner_score}/100**")

    # Leaderboard table
    st.subheader("📊 Full Leaderboard")
    rows = []
    for m in tm.models:
        row = {
            "Model": m.model_name,
            "Intelligent Score": m.intelligent_score,
            "Accuracy Norm": round(m.accuracy_norm, 4),
            "Speed Score": round(m.speed_score, 4),
            "Robustness": round(m.robustness_score, 4),
            "Explainability": m.explainability_score,
        }
        if m.roc_auc is not None:
            row["ROC-AUC"] = m.roc_auc
        if m.f1_weighted is not None:
            row["F1-Weighted"] = m.f1_weighted
        if m.r2_score is not None:
            row["R²"] = m.r2_score
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Bar chart comparing models
    fig = go.Figure()
    model_names = [m.model_name for m in tm.models]
    fig.add_trace(go.Bar(name="Intelligent Score", x=model_names,
                         y=[m.intelligent_score for m in tm.models],
                         marker_color=["#667eea" if m.model_name == tm.winner_name else "#444"
                                       for m in tm.models]))
    fig.update_layout(
        title="Model Comparison — Intelligent Score",
        xaxis_title="Model",
        yaxis_title="Score (0-100)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_predictions_shap(result):
    """Tab 4: Predictions & SHAP."""
    st.header("🎯 Predictions & Explainable AI (SHAP)")

    if result.tournament:
        tm = result.tournament

        # Risk analysis alerts
        if tm.risk_tiers:
            st.subheader("⚠️ Risk Distribution")
            tier_counts = Counter(tm.risk_tiers)
            tier_df = pd.DataFrame([
                {"Risk Tier": k, "Count": v} for k, v in sorted(tier_counts.items())
            ])
            fig = px.bar(tier_df, x="Risk Tier", y="Count", color="Risk Tier",
                         color_discrete_map={"Low": "#00cc66", "Moderate": "#ffaa00",
                                             "High": "#ff6600", "Critical": "#ff0000"},
                         title="Risk Tier Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # High-risk alert table
            critical_count = tier_counts.get("Critical", 0) + tier_counts.get("High", 0)
            if critical_count > 0:
                st.error(f"🚨 {critical_count} instances at High/Critical risk level!")

    if result.explainability:
        ex = result.explainability

        # Global SHAP
        if ex.global_feature_importance:
            st.subheader("🌍 Global SHAP Feature Importance")
            gfi_data = [{"Feature": f.feature, "Importance": f.importance, "Direction": f.direction}
                        for f in ex.global_feature_importance[:10]]
            gfi_df = pd.DataFrame(gfi_data)
            fig = px.bar(gfi_df, y="Feature", x="Importance", orientation="h",
                         color="Direction",
                         color_discrete_map={"positive": "#ff6666", "negative": "#6699ff"},
                         title="SHAP Feature Importance (Top 10)")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Local explanation
        if ex.local_explanation:
            le = ex.local_explanation
            st.subheader("🔬 Local SHAP Explanation (Highest-Impact Instance)")
            st.markdown(f"**Row Index:** {le.row_index} | **Base Value:** {le.base_value:.4f} | "
                        f"**Predicted Value:** {le.predicted_value:.4f}")

            if le.contributions:
                contrib_data = [{"Feature": c.feature, "SHAP Value": c.importance, "Direction": c.direction}
                                for c in le.contributions[:8]]
                contrib_df = pd.DataFrame(contrib_data)
                fig = px.bar(contrib_df, y="Feature", x="SHAP Value", orientation="h",
                             color="Direction",
                             color_discrete_map={"positive": "#ff6666", "negative": "#6699ff"},
                             title=f"SHAP Waterfall — Row {le.row_index}")
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350)
                st.plotly_chart(fig, use_container_width=True)

            st.text(le.narrative)


def _tab_whatif(result):
    """Tab 5: What-If Simulator."""
    st.header("🔮 What-If Simulator")

    session_id = st.session_state.get("session_id")
    if not session_id:
        st.warning("No active session. Run the pipeline first.")
        return

    session = get_session(session_id)
    if session is None or session.df_clean is None:
        st.warning("Session data not available.")
        return

    st.markdown("Adjust feature values below and see how predictions change in real-time.")

    df = session.df_clean
    id_cols = session.feature_engineer._id_columns if session.feature_engineer else []
    target_col = session.target_col
    dt_cols = session.feature_engineer._datetime_columns if session.feature_engineer else []
    exclude = set(id_cols + dt_cols + ([target_col] if target_col else []))

    feature_cols = [c for c in df.columns if c not in exclude]
    overrides = {}

    # Create sliders for numeric and dropdowns for categorical
    cols = st.columns(min(3, len(feature_cols)))
    for i, col in enumerate(feature_cols[:9]):
        with cols[i % len(cols)]:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                median_val = float(df[col].median())
                step = (max_val - min_val) / 100 if max_val > min_val else 0.1
                val = st.slider(f"📊 {col}", min_val, max_val, median_val, step=step, key=f"wif_{col}")
                overrides[col] = val
            else:
                unique_vals = df[col].dropna().unique().tolist()
                val = st.selectbox(f"📋 {col}", unique_vals, key=f"wif_{col}")
                overrides[col] = val

    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Computing..."):
            # Get baseline row
            if session.is_risk_analysis and result.tournament and result.tournament.probabilities:
                high_risk_idx = int(np.argmax(result.tournament.probabilities))
            else:
                high_risk_idx = 0
            baseline_row = df.iloc[[high_risk_idx]].copy()

            wif_result = run_what_if(
                baseline_row=baseline_row,
                overrides=overrides,
                feature_engineer=session.feature_engineer,
                model=session.best_model,
                X_background=session.X_transformed,
                feature_names=session.feature_names,
                task_type=session.task_type,
                is_risk_analysis=session.is_risk_analysis,
            )

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline", f"{wif_result.baseline_prediction:.4f}",
                     help=f"Risk: {wif_result.baseline_risk_tier}" if wif_result.baseline_risk_tier else "")
        col2.metric("Simulated", f"{wif_result.simulated_prediction:.4f}",
                     delta=f"{wif_result.delta_value:+.4f}",
                     help=f"Risk: {wif_result.simulated_risk_tier}" if wif_result.simulated_risk_tier else "")
        col3.metric("Delta %", f"{wif_result.delta_pct:+.2f}%")

        if wif_result.baseline_risk_tier and wif_result.simulated_risk_tier:
            st.markdown(f"**Risk Tier Change:** {wif_result.baseline_risk_tier} → {wif_result.simulated_risk_tier}")

        st.markdown(f"**Recommendation:** {wif_result.recommendation}")


def _tab_report(result):
    """Tab 6: AI Report & Recommendations."""
    st.header("📝 AI Data Scientist Report & Recommendations")

    if not result.report:
        st.warning("Report not available.")
        return

    r = result.report

    st.subheader("1️⃣ What Will Happen?")
    st.markdown(r.what_will_happen)

    st.subheader("2️⃣ Why Will It Happen?")
    st.markdown(r.why_will_it_happen)

    st.subheader("3️⃣ What Should We Do Next?")
    st.markdown(r.what_should_we_do)

    st.divider()
    if r.grounded:
        st.success("✅ All numbers in this report are verified against computed metrics (zero hallucination).")
    else:
        st.warning("⚠️ Some numbers may not be fully verified.")

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Full Report (Markdown)",
            data=r.full_narrative,
            file_name="analytixbuz_report.md",
            mime="text/markdown",
        )
    with col2:
        report_json = result.model_dump_json(indent=2)
        st.download_button(
            "📥 Download JSON Report",
            data=report_json,
            file_name="analytixbuz_report.json",
            mime="application/json",
        )


def _tab_chat(result):
    """Tab 7: Ask AI Data Scientist (Grounded Q&A)."""
    st.header("💬 Ask AI Data Scientist")
    st.markdown("Ask questions about the dataset, model results, SHAP explanations, or risk analysis.")

    session_id = st.session_state.get("session_id")
    session = get_session(session_id) if session_id else None

    if not session or not session.pipeline_summary:
        st.warning("Run the pipeline first to enable Q&A.")
        return

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display chat history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your data...")
    if user_question:
        st.session_state["chat_messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get grounded answer
        summary_dict = session.pipeline_summary.model_dump()
        if session.response and session.response.report:
            summary_dict["report"] = session.response.report.model_dump()

        answer = answer_user_question(user_question, summary_dict)

        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
