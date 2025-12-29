import pandas as pd
import streamlit as st
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MON_DIR = BASE_DIR / "data" / "predictions" / "monitoring"

ALERTS_BY_DAY_PATH = MON_DIR / "alerts_by_day.parquet"
ALERTS_BY_DEVICE_PATH = MON_DIR / "alerts_by_device.parquet"
SCORE_BY_DAY_PATH = MON_DIR / "score_distribution_by_day.parquet"

st.set_page_config(
    page_title="Telemetry Anomaly Monitoring",
    layout="wide",
)

st.title("📡 Telemetry Anomaly Monitoring")

# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data
def load_data():
    alerts_by_day = pd.read_parquet(ALERTS_BY_DAY_PATH)
    alerts_by_device = pd.read_parquet(ALERTS_BY_DEVICE_PATH)
    score_by_day = pd.read_parquet(SCORE_BY_DAY_PATH)
    return alerts_by_day, alerts_by_device, score_by_day


alerts_by_day, alerts_by_device, score_by_day = load_data()

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Days",
        alerts_by_day["day"].nunique(),
    )

with col2:
    st.metric(
        "Avg Alert Rate",
        f"{alerts_by_day['alert_rate'].mean():.2%}",
    )

with col3:
    st.metric(
        "Max Daily Alert Rate",
        f"{alerts_by_day['alert_rate'].max():.2%}",
    )

st.divider()

# -----------------------------
# Alerts over time
# -----------------------------
st.subheader("Alert Rate Over Time")

alerts_by_day["day"] = pd.to_datetime(alerts_by_day["day"])

st.line_chart(
    alerts_by_day.set_index("day")[["alert_rate"]],
    height=300,
)

# -----------------------------
# Score distribution over time
# -----------------------------
st.subheader("Anomaly Score Distribution (Quantiles)")

pivot_scores = (
    score_by_day
    .pivot(index="day", columns="quantile", values="value")
    .sort_index()
)

st.line_chart(
    pivot_scores,
    height=300,
)

# -----------------------------
# Alerts by device
# -----------------------------
st.subheader("Devices with Highest Alert Rates")

top_n = st.slider("Show top N devices", 5, 50, 15)

st.dataframe(
    alerts_by_device.head(top_n),
    use_container_width=True,
)

# -----------------------------
# Device inspection
# -----------------------------
st.subheader("Inspect Single Device")

device_id = st.selectbox(
    "Select device",
    alerts_by_device["device_id"].unique(),
)

device_row = alerts_by_device[alerts_by_device["device_id"] == device_id]

st.write("**Device summary**")
st.dataframe(device_row, use_container_width=True)

st.caption("Dashboard reflects batch-scored anomaly outputs. No model logic runs in Streamlit.")
