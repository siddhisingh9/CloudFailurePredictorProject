import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEMO_DATA_PATH = Path(os.getenv(
    "DEMO_DATA_PATH",
    Path(__file__).resolve().parent.parent / "data" / "processed_gct.csv",
))
FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
INT_FEATURES = ["priority", "scheduling_class"]
LABEL = "failed"
WINDOW_SIZE = 50
REQUEST_TIMEOUT = 5

st.set_page_config(page_title="Cloud Failure Dashboard", layout="wide")
st.title("☁️📊 Real-Time Cloud Failure Prediction Dashboard")


@st.cache_resource
def http_session():
    return requests.Session()


@st.cache_data
def load_demo_rows():
    return pd.read_csv(DEMO_DATA_PATH, usecols=FEATURES + [LABEL])


def clean_rows(df):
    """Keep the model features (and the label if present), dropping rows the API would reject."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    cols = FEATURES + ([LABEL] if LABEL in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = (
        df[FEATURES].notna().all(axis=1)
        & (df[FEATURES] >= 0).all(axis=1)
        & (df[INT_FEATURES] % 1 == 0).all(axis=1)
        & (df["scheduling_class"] <= 3)
    )
    dropped = int((~valid).sum())
    if dropped:
        st.warning(f"Skipped {dropped} row(s) with missing or invalid values.")
    df = df[valid].reset_index(drop=True)
    if df.empty:
        st.error("No valid rows left to stream.")
        st.stop()
    return df


def reset_stream():
    st.session_state.streaming = False
    st.session_state.row_index = 0
    st.session_state.history = []
    st.session_state.current = None
    st.session_state.last_error = None


if "streaming" not in st.session_state:
    reset_stream()

choice = st.radio(
    "Choose data source:",
    ["Upload CSV", "Google Cluster Trace (demo)"],
    on_change=reset_stream,
)

if choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv", on_change=reset_stream)
    if uploaded_file is None:
        st.info(f"Upload a CSV with columns: {', '.join(FEATURES)}")
        st.stop()
    rows = clean_rows(pd.read_csv(uploaded_file))
else:
    rows = load_demo_rows()


def start_streaming():
    st.session_state.streaming = True


def stop_streaming():
    st.session_state.streaming = False
    st.session_state.row_index = 0


col1, col2 = st.columns(2)
with col1:
    st.button("▶️ Start Streaming", on_click=start_streaming, disabled=st.session_state.streaming)
with col2:
    st.button("⏹️ Stop Streaming", on_click=stop_streaming, disabled=not st.session_state.streaming)


def predict_next(rows):
    idx = st.session_state.row_index % len(rows)
    st.session_state.row_index = idx + 1
    row = rows.iloc[idx]

    data = {f: float(row[f]) for f in FEATURES}
    for f in INT_FEATURES:
        data[f] = int(data[f])

    try:
        resp = http_session().post(f"{API_URL}/predict", json=data, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        prob = resp.json()["failure_probability"]
    except (requests.RequestException, KeyError, ValueError) as e:
        st.session_state.last_error = f"API request failed: {e}"
        return

    st.session_state.last_error = None
    st.session_state.current = {
        "data": data,
        "prob": prob,
        "actual": int(row[LABEL]) if LABEL in rows.columns and pd.notna(row[LABEL]) else None,
    }
    st.session_state.history.append(prob)
    if len(st.session_state.history) > WINDOW_SIZE:
        st.session_state.history.pop(0)


@st.fragment(run_every="3s" if st.session_state.streaming else None)
def stream_panel(rows):
    if st.session_state.streaming:
        predict_next(rows)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    current = st.session_state.current
    if current is None:
        st.caption("Press Start Streaming to send rows to the prediction API.")
        return

    data, prob = current["data"], current["prob"]

    st.subheader("Latest Job Metrics")
    st.write(data)
    if prob > 0.8:
        st.warning(f"⚠️ High failure risk: {prob:.2f}")
    elif prob < 0.2:
        st.success(f"✅ Low risk: {prob:.2f}")
    else:
        st.info(f"Failure probability: {prob:.2f}")
    if current["actual"] is not None:
        st.caption(f"Actual outcome in trace: {'failed' if current['actual'] else 'finished'}")

    st.subheader("Historical Failure Probabilities")
    st.line_chart(st.session_state.history)

    st.subheader("CPU & Memory Usage")
    st.bar_chart({
        "CPU Request": [data["cpu_request"]],
        "Memory Request": [data["memory_request"]]
    })


stream_panel(rows)
