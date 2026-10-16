"""
Streamlit Dashboard
Predictive Forecasting of Care Load & Placement Demand

How to run (from the project root folder):
    streamlit run app/streamlit_app.py

Requires: streamlit, pandas, plotly
(If packages are missing, open outputs/dashboard.html in a browser instead.)
"""

from pathlib import Path
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = BASE_DIR / "final_dataset.csv"

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="UAC Care Load & Placement Forecast",
    page_icon="📊",
    layout="wide"
)

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("UAC Program — short-term forecasts for HHS care load and discharge demand")

# -------------------------------------------------
# Load data (cached)
# -------------------------------------------------
@st.cache_data
def load_all():
    hist = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    forward = pd.read_csv(OUTPUT_DIR / "forward_forecast_14day.csv", parse_dates=["Date"])
    pressure = pd.read_csv(OUTPUT_DIR / "net_pressure_early_warning.csv", parse_dates=["Date"])
    test_pred = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", parse_dates=["Date"])
    kpi = pd.read_csv(OUTPUT_DIR / "kpi_summary.csv")
    return hist, metrics, forward, pressure, test_pred, kpi


try:
    hist, metrics, forward, pressure, test_pred, kpi = load_all()
except FileNotFoundError:
    st.error(
        "Output files not found. Please run `python complete_forecasting.py` "
        "from the project root first."
    )
    st.stop()

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")

target = st.sidebar.selectbox(
    "Target series",
    options=["HHS_Care", "Discharged_From_HHS"],
    format_func=lambda x: "Children in HHS Care" if x == "HHS_Care" else "Discharges (Placement Demand)"
)

model_options = ["Naive", "Moving_Average", "ARIMA", "ExpSmoothing", "RandomForest", "GradientBoosting"]
selected_models = st.sidebar.multiselect(
    "Models to display",
    options=model_options,
    default=["ExpSmoothing", "RandomForest", "GradientBoosting"]
)

horizon = st.sidebar.slider("Forward horizon (days)", min_value=3, max_value=14, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**Best model (by test MAE)**")
best_row = kpi[kpi["Target"] == target].iloc[0]
st.sidebar.metric("Best model", best_row["Best_Model"])
st.sidebar.metric("MAE", f"{best_row['Best_MAE']:.1f}")
st.sidebar.metric("Accuracy %", f"{best_row['Forecast_Accuracy_Pct']:.1f}%")

# -------------------------------------------------
# KPI row
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best model", best_row["Best_Model"])
with col2:
    st.metric("Test MAE", f"{best_row['Best_MAE']:.1f}")
with col3:
    st.metric("Test RMSE", f"{best_row['Best_RMSE']:.1f}")
with col4:
    st.metric("Forecast accuracy", f"{best_row['Forecast_Accuracy_Pct']:.1f}%")

# -------------------------------------------------
# Tab layout
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Future Care Load / Discharge Forecast",
    "Model Comparison (Test Period)",
    "Net Pressure Early Warning",
    "Metrics Table"
])

# ---- Tab 1: Forward forecast ----
with tab1:
    st.subheader("14-day forward forecast")
    fwd = forward[forward["Target"] == target].copy()
    fwd = fwd.head(horizon)

    if px is not None and selected_models:
        fig = go.Figure()
        # recent history for context
        recent = hist.tail(60)
        y_col = "HHS_Care" if target == "HHS_Care" else "Discharged_From_HHS"
        fig.add_trace(go.Scatter(
            x=recent["Date"], y=recent[y_col],
            name="Recent actual", line=dict(color="black", width=2)
        ))
        for m in selected_models:
            if m in fwd.columns:
                fig.add_trace(go.Scatter(
                    x=fwd["Date"], y=fwd[m],
                    name=m, mode="lines+markers"
                ))
        fig.update_layout(
            title=f"Forward forecast — {target}",
            xaxis_title="Date",
            yaxis_title=target,
            legend_title="Series",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(fwd)

    st.markdown(
        "The forward view uses the full history. "
        "LinearFeatures holds the last known feature row constant (simple short-horizon approach)."
    )

# ---- Tab 2: Test comparison ----
with tab2:
    st.subheader("Actual vs predicted on the held-out test period")
    tp = test_pred[test_pred["Target"] == target].copy()
    if selected_models:
        tp = tp[tp["Model"].isin(selected_models)]

    if px is not None and len(tp) > 0:
        fig2 = px.line(
            tp, x="Date", y="Predicted", color="Model",
            title=f"Test-period predictions — {target}"
        )
        # Add actual once
        actual_once = tp.drop_duplicates("Date")[["Date", "Actual"]]
        fig2.add_trace(go.Scatter(
            x=actual_once["Date"], y=actual_once["Actual"],
            name="Actual", line=dict(color="black", width=2, dash="dot")
        ))
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.dataframe(tp.head(100))

# ---- Tab 3: Net pressure ----
with tab3:
    st.subheader("Intake vs discharge balance (early warning)")
    st.markdown(
        "Net Pressure = Forecast Transfers into HHS − Forecast Discharges. "
        "Positive values signal rising care-load risk."
    )
    p = pressure.head(horizon)
    if px is not None:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=p["Date"], y=p["Forecast_Transfers_In"], name="Transfers in"))
        fig3.add_trace(go.Bar(x=p["Date"], y=p["Forecast_Discharges"], name="Discharges"))
        fig3.add_trace(go.Scatter(
            x=p["Date"], y=p["Net_Pressure"],
            name="Net pressure", mode="lines+markers", yaxis="y2"
        ))
        fig3.update_layout(
            barmode="group",
            yaxis=dict(title="Children"),
            yaxis2=dict(title="Net pressure", overlaying="y", side="right"),
            height=450
        )
        st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(p)

# ---- Tab 4: Metrics ----
with tab4:
    st.subheader("Model performance (test set)")
    st.dataframe(metrics[metrics["Target"] == target].reset_index(drop=True))
    st.subheader("KPI summary")
    st.dataframe(kpi)

st.markdown("---")
st.caption(
    "Project: Predictive Forecasting of Care Load & Placement Demand | "
    "Models: Naive, Moving Average, ARIMA, Exponential Smoothing, Random Forest, Gradient Boosting"
)
