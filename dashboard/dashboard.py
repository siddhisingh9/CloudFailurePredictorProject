import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import redis
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEMO_DATA_PATH = Path(os.getenv(
    "DEMO_DATA_PATH",
    Path(__file__).resolve().parent.parent / "data" / "processed_gct.csv",
))
FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
INT_FEATURES = ["priority", "scheduling_class"]
LABEL = "failed"
WINDOW_SIZE = 50
REQUEST_TIMEOUT = 10
REDIS_URL = os.getenv("REDIS_URL")
PREDICTIONS_CHANNEL = "predictions"
FEED_SIZE = 15

st.set_page_config(page_title="Cloud Failure Dashboard", layout="wide")
st.title("☁️📊 Real-Time Cloud Failure Prediction Dashboard")


@st.cache_resource
def http_session():
    return requests.Session()


def wake_api():
    """Free-tier hosts sleep when idle; ping the API early so it is awake by the first prediction."""
    try:
        requests.get(f"{API_URL}/health", timeout=90)
    except requests.RequestException:
        pass


class LiveFeed:
    """Recent predictions from every client of the API, received over Redis Pub/Sub."""

    def __init__(self):
        self.lock = threading.Lock()
        self.messages = deque(maxlen=FEED_SIZE)
        self.status = "connecting…"

    def add(self, message):
        with self.lock:
            self.messages.appendleft(message)

    def items(self):
        with self.lock:
            return list(self.messages)

    def listen(self):
        while True:
            try:
                client = redis.from_url(
                    REDIS_URL, decode_responses=True, socket_connect_timeout=5,
                    socket_keepalive=True, health_check_interval=25,
                )
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(PREDICTIONS_CHANNEL)
                self.status = f"subscribed to '{PREDICTIONS_CHANNEL}'"
                while True:
                    message = pubsub.get_message(timeout=1.0)
                    if message is None:
                        continue
                    try:
                        self.add(json.loads(message["data"]))
                    except (ValueError, TypeError):
                        logger.warning("Ignoring malformed message: %r", message["data"])
            except (redis.RedisError, OSError) as e:
                self.status = f"disconnected ({e}); retrying…"
                logger.warning("Live feed connection failed: %s", e)
                time.sleep(5)


@st.cache_resource
def live_feed():
    # One subscriber per server process, shared by every browser session
    feed = LiveFeed()
    threading.Thread(target=feed.listen, daemon=True).start()
    return feed


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
    st.session_state.api_stats = None
    threading.Thread(target=wake_api, daemon=True).start()


def start_streaming():
    st.session_state.streaming = True


def stop_streaming():
    st.session_state.streaming = False
    st.session_state.row_index = 0


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
        result = resp.json()
        prob = result["failure_probability"]
    except (requests.ConnectionError, requests.Timeout):
        st.session_state.last_error = (
            "The API didn't respond. If it's hosted on a free tier it may be waking up "
            "(this can take up to a minute); retrying automatically…"
        )
        return
    except (requests.RequestException, KeyError, ValueError) as e:
        st.session_state.last_error = f"API request failed: {e}"
        return

    try:
        st.session_state.api_stats = http_session().get(f"{API_URL}/stats", timeout=REQUEST_TIMEOUT).json()
    except (requests.RequestException, ValueError):
        pass

    st.session_state.last_error = None
    st.session_state.current = {
        "data": data,
        "prob": prob,
        "cached": result.get("cached", False),
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
    st.caption("⚡ Served from Redis cache" if current["cached"] else "🧠 Computed by the model")

    api_stats = st.session_state.api_stats
    if api_stats and api_stats.get("predictions"):
        hit_rate = api_stats["hit_rate"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cache hit rate", f"{hit_rate:.0%}")
        c2.metric("Avg latency, cache hit", f"{api_stats['avg_ms_cache_hit']} ms" if api_stats["avg_ms_cache_hit"] is not None else "–")
        c3.metric("Avg latency, model", f"{api_stats['avg_ms_cache_miss']} ms" if api_stats["avg_ms_cache_miss"] is not None else "–")
        st.caption(f"Across all {api_stats['predictions']} predictions since the API last started.")

    st.subheader("Historical Failure Probabilities")
    st.line_chart(st.session_state.history)

    st.subheader("CPU & Memory Usage")
    st.bar_chart({
        "CPU Request": [data["cpu_request"]],
        "Memory Request": [data["memory_request"]]
    })


@st.fragment(run_every="2s")
def live_feed_panel():
    feed = live_feed()
    st.subheader("📡 Live Prediction Feed")
    st.caption(f"Every prediction from every user, via Redis Pub/Sub. Status: {feed.status}")
    messages = feed.items()
    if not messages:
        st.caption("Waiting for predictions…")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "time (UTC)": m.get("published_at", "")[11:19],
                **m.get("data", {}),
                "failure_probability": round(m.get("failure_probability", float("nan")), 3),
                "cached": m.get("cached"),
            }
            for m in messages
        ]),
        hide_index=True,
    )


main_col, feed_col = st.columns([3, 2], gap="large")

# The feed is drawn first so it shows even while the main column is waiting for an upload
with feed_col:
    if REDIS_URL:
        live_feed_panel()
    else:
        st.subheader("📡 Live Prediction Feed")
        st.caption("Set REDIS_URL to see every user's predictions streamed live over Redis Pub/Sub.")

with main_col:
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

    col1, col2 = st.columns(2)
    with col1:
        st.button("▶️ Start Streaming", on_click=start_streaming, disabled=st.session_state.streaming)
    with col2:
        st.button("⏹️ Stop Streaming", on_click=stop_streaming, disabled=not st.session_state.streaming)

    stream_panel(rows)
