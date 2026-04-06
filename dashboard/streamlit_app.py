"""
Audit Risk Analytics — Streamlit Dashboard
Multi-page interactive audit analytics dashboard.
"""
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Audit Risk Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared Data Loading ────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data...")
def load_full_pipeline():
    """Run the full pipeline and cache the result."""
    import pandas as pd
    from src.config import PROCESSED_PARQUET
    from src.feature_engineering import engineer_all_features
    from src.anomaly_model import run_anomaly_detection
    from src.risk_scorer import calculate_risk_scores

    df = pd.read_parquet(PROCESSED_PARQUET)
    df = engineer_all_features(df)
    df = run_anomaly_detection(df)
    df = calculate_risk_scores(df)
    return df


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("🔍 Audit Risk Analytics")
st.sidebar.markdown("---")

pages = {
    "📊 Overview": "overview",
    "🔎 Transaction Explorer": "explorer",
    "🧠 Anomaly Analysis": "anomaly",
    "⚖️ Risk Distribution": "risk",
    "📋 Findings Report": "report",
}
selected = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
page = pages[selected]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built for:**  \n"
    "Deloitte Future Leaders Academy  \n"
    "Audit & Assurance — Data Analytics"
)
st.sidebar.markdown(
    "**Tech Stack:**  \n"
    "Python · Scikit-learn · DuckDB · Streamlit"
)

# Load data
df = load_full_pipeline()

# ── Helper ─────────────────────────────────────────────────
CAT_COLORS = {"low": "#27ae60", "medium": "#f39c12", "high": "#e67e22", "critical": "#e74c3c"}
TARGET = "Class"


# ══════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════
if page == "overview":
    st.title("📊 Dashboard Overview")
    st.markdown("Real-time audit analytics for financial transaction monitoring.")

    total = len(df)
    fraud = int(df[TARGET].sum())
    high_risk = len(df[df["risk_category"].isin(["high", "critical"])])
    avg_risk = df["risk_score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Confirmed Fraud", f"{fraud:,}", f"{fraud/total*100:.3f}%")
    col3.metric("High/Critical Risk", f"{high_risk:,}", f"{high_risk/total*100:.2f}%")
    col4.metric("Avg Risk Score", f"{avg_risk:.1f}", "/ 100")

    st.markdown("---")

    import plotly.express as px
    import plotly.graph_objects as go

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Category Distribution")
        cat_counts = df["risk_category"].value_counts().reindex(["low", "medium", "high", "critical"]).fillna(0)
        fig = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            color=cat_counts.index,
            color_discrete_map=CAT_COLORS,
            labels={"x": "Risk Category", "y": "Count"},
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Fraud Rate by Risk Category")
        fraud_rates = df.groupby("risk_category", observed=True)[TARGET].mean().reindex(
            ["low", "medium", "high", "critical"]).fillna(0) * 100
        fig = px.bar(
            x=fraud_rates.index, y=fraud_rates.values,
            color=fraud_rates.index,
            color_discrete_map=CAT_COLORS,
            labels={"x": "Risk Category", "y": "Fraud Rate (%)"},
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Volume by Hour")
    hourly = df.groupby("time_hour_int").agg(
        count=("Amount", "count"), fraud=(TARGET, "sum")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly["time_hour_int"], y=hourly["count"],
                         name="Transactions", marker_color="steelblue"))
    fig.add_trace(go.Scatter(x=hourly["time_hour_int"], y=hourly["fraud"] * 100,
                             name="Fraud Count (×100)", mode="lines+markers",
                             marker_color="#e74c3c", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Transaction Count"),
        yaxis2=dict(title="Fraud (×100)", overlaying="y", side="right"),
        height=400, legend=dict(x=0.01, y=0.99),
    )
    # Shade off-hours
    fig.add_vrect(x0=0, x1=8, fillcolor="red", opacity=0.05, line_width=0)
    fig.add_vrect(x0=18, x1=24, fillcolor="red", opacity=0.05, line_width=0)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: Transaction Explorer
# ══════════════════════════════════════════════════════════
elif page == "explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("Filter and explore individual transactions by risk level.")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Risk Category", ["low", "medium", "high", "critical"],
            default=["high", "critical"]
        )
    with col2:
        amount_range = st.slider(
            "Amount Range (EUR)", 0.0, float(df["Amount"].max()),
            (0.0, float(df["Amount"].max()))
        )
    with col3:
        fraud_only = st.checkbox("Show confirmed fraud only", value=False)

    filtered = df[
        (df["risk_category"].isin(risk_filter)) &
        (df["Amount"].between(amount_range[0], amount_range[1]))
    ]
    if fraud_only:
        filtered = filtered[filtered[TARGET] == 1]

    st.markdown(f"**Showing {len(filtered):,} transactions**")

    display_cols = [
        "Amount", "risk_score", "risk_category", "anomaly_score",
        "amount_risk", "time_risk", "time_hour_int",
        "amount_bucket", "is_business_hours", TARGET,
    ]
    st.dataframe(
        filtered[display_cols].sort_values("risk_score", ascending=False).head(500),
        use_container_width=True, height=500,
    )

    st.markdown("---")
    st.subheader("Filtered Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", f"{len(filtered):,}")
    col2.metric("Fraud Cases", f"{int(filtered[TARGET].sum())}")
    col3.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.1f}")


# ══════════════════════════════════════════════════════════
# PAGE: Anomaly Analysis
# ══════════════════════════════════════════════════════════
elif page == "anomaly":
    st.title("🧠 Anomaly Analysis")
    st.markdown("Explore anomaly detection model outputs and score distributions.")

    import plotly.express as px

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Isolation Forest Score Distribution")
        fig = px.histogram(
            df, x="if_score", color=df[TARGET].map({0: "Legitimate", 1: "Fraud"}),
            nbins=80, barmode="overlay", opacity=0.6,
            color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
            labels={"color": "Class", "if_score": "Anomaly Score"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("LOF Score Distribution")
        fig = px.histogram(
            df, x="lof_score", color=df[TARGET].map({0: "Legitimate", 1: "Fraud"}),
            nbins=80, barmode="overlay", opacity=0.6,
            color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
            labels={"color": "Class", "lof_score": "Anomaly Score"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Score vs Amount")
    fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="Amount", y="anomaly_score", color=df.sample(min(5000, len(df)), random_state=42)[TARGET].map(
            {0: "Legitimate", 1: "Fraud"}),
        color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
        opacity=0.4, labels={"color": "Class"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("IF vs LOF Score Comparison")
    fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="if_score", y="lof_score",
        color=df.sample(min(5000, len(df)), random_state=42)[TARGET].map(
            {0: "Legitimate", 1: "Fraud"}),
        color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
        opacity=0.4, labels={"color": "Class", "if_score": "Isolation Forest", "lof_score": "LOF"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: Risk Distribution
# ══════════════════════════════════════════════════════════
elif page == "risk":
    st.title("⚖️ Risk Distribution")
    st.markdown("Deep-dive into risk score composition and patterns.")

    import plotly.express as px
    import plotly.graph_objects as go

    # Risk heatmap: hour x amount bucket
    st.subheader("Risk Heatmap: Hour × Materiality Bucket")
    heatmap_data = df.groupby(["time_hour_int", "amount_bucket"], observed=True).agg(
        avg_risk=("risk_score", "mean")).reset_index()
    pivot = heatmap_data.pivot_table(index="amount_bucket", columns="time_hour_int", values="avg_risk")
    pivot = pivot.reindex(["low", "medium", "high", "very_high"])

    fig = px.imshow(
        pivot, color_continuous_scale="YlOrRd", aspect="auto",
        labels=dict(x="Hour of Day", y="Materiality Bucket", color="Avg Risk"),
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Components")
        components = ["anomaly_score", "amount_risk", "time_risk", "pca_risk", "velocity_risk"]
        comp_means = df[components].mean()
        fig = px.bar(
            x=["Anomaly (35%)", "Amount (25%)", "Time (15%)", "PCA (15%)", "Velocity (10%)"],
            y=comp_means.values,
            color=comp_means.index,
            labels={"x": "Component", "y": "Mean Score"},
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Cumulative Fraud Detection")
        df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        df_sorted["cum_fraud"] = df_sorted[TARGET].cumsum()
        total_fraud = df_sorted[TARGET].sum()
        # Sample for performance
        step = max(1, len(df_sorted) // 500)
        plot_data = df_sorted.iloc[::step].copy()
        plot_data["pct_reviewed"] = (plot_data.index + 1) / len(df_sorted) * 100
        plot_data["pct_fraud_found"] = plot_data["cum_fraud"] / total_fraud * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data["pct_reviewed"], y=plot_data["pct_fraud_found"],
            mode="lines", name="Risk-Based", line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines", name="Random",
            line=dict(color="gray", dash="dash")))
        fig.update_layout(
            xaxis_title="% Reviewed", yaxis_title="% Fraud Found",
            height=400, xaxis=dict(range=[0, 50]))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: Findings Report
# ══════════════════════════════════════════════════════════
elif page == "report":
    st.title("📋 Audit Findings Report")

    from src.report_generator import generate_audit_report
    report = generate_audit_report(df)
    st.markdown(report)

    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report,
        file_name="audit_findings_report.md",
        mime="text/markdown",
    )
