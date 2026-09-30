import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import altair as alt
import pandas as pd
import redis
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
# The address a browser can reach, for links (API_URL may be an internal hostname)
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", API_URL).rstrip("/")
REDIS_URL = os.getenv("REDIS_URL")
DEMO_DATA_PATH = Path(os.getenv(
    "DEMO_DATA_PATH",
    Path(__file__).resolve().parent.parent / "data" / "demo_holdout.csv",
))
REPO_URL = "https://github.com/siddhisingh9/CloudFailurePredictorProject"

FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
INT_FEATURES = ["priority", "scheduling_class"]
FEATURE_LABELS = {
    "cpu_request": "CPU request",
    "memory_request": "Memory request",
    "priority": "Priority",
    "scheduling_class": "Scheduling class",
}
LABEL = "failed"
SEEN = "seen_in_training"
THRESHOLD = 0.5
CHART_WINDOW = 50
KPI_HEIGHT = 112
HISTORY_SIZE = 200
FEED_SIZE = 15
REQUEST_TIMEOUT = 10
PREDICTIONS_CHANNEL = "predictions"

RED, GREEN, GREY, BLUE = "#DC2626", "#16A34A", "#94A3B8", "#2563EB"

st.set_page_config(
    page_title="Cloud Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.html("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
  h1 { font-size: 2rem !important; letter-spacing: -0.01em; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
  [data-testid="stMetricLabel"] p { font-size: 0.85rem; color: #64748B; }
  .subtitle { color: #64748B; margin-top: -0.4rem; font-size: 0.95rem; }
  .big-prob { font-size: 2.4rem; font-weight: 650; line-height: 1.1; margin: 0.2rem 0 0.4rem; }
  .muted { color: #64748B; font-size: 0.85rem; }
</style>
""")


# ---------------------------------------------------------------- API access

@st.cache_resource
def http_session():
    return requests.Session()


def api_get(path, timeout):
    resp = http_session().get(f"{API_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=10, show_spinner=False)
def api_health():
    try:
        return api_get("/health", timeout=3)
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _model_info():
    return api_get("/model", timeout=REQUEST_TIMEOUT)  # errors aren't cached, so it retries


def model_info():
    try:
        return _model_info()
    except (requests.RequestException, ValueError):
        return None


def wake_api():
    """Free-tier hosts sleep when idle; ping the API early so it is awake by the first prediction."""
    try:
        requests.get(f"{API_URL}/health", timeout=90)
    except requests.RequestException:
        pass


# ---------------------------------------------------------------- live feed (Redis Pub/Sub)

class LiveFeed:
    """Recent predictions from every client of the API, received over Redis Pub/Sub."""

    def __init__(self):
        self.lock = threading.Lock()
        self.messages = deque(maxlen=FEED_SIZE)
        self.status = "connecting"

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
                self.status = "subscribed"
                while True:
                    message = pubsub.get_message(timeout=1.0)
                    if message is None:
                        continue
                    try:
                        self.add(json.loads(message["data"]))
                    except (ValueError, TypeError):
                        logger.warning("Ignoring malformed message: %r", message["data"])
            except (redis.RedisError, OSError) as e:
                self.status = "reconnecting"
                logger.warning("Live feed connection failed: %s", e)
                time.sleep(5)


@st.cache_resource
def live_feed():
    # One subscriber per server process, shared by every browser session
    feed = LiveFeed()
    threading.Thread(target=feed.listen, daemon=True).start()
    return feed


# ---------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load_holdout():
    # round_trip parsing keeps the exact float values the model was evaluated on
    return pd.read_csv(DEMO_DATA_PATH, float_precision="round_trip")


def clean_upload(df):
    """Keep valid rows with the model features (and the label if present). Returns (rows, error, note)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, f"The file is missing required columns: {', '.join(missing)}.", None

    cols = FEATURES + ([LABEL] if LABEL in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = (
        df[FEATURES].notna().all(axis=1)
        & (df[FEATURES] >= 0).all(axis=1)
        & (df[INT_FEATURES] % 1 == 0).all(axis=1)
        & (df["scheduling_class"] <= 3)
    )
    dropped = int((~valid).sum())
    df = df[valid].reset_index(drop=True)
    if df.empty:
        return None, "No valid rows to stream.", None
    return df, None, (f"Skipped {dropped} row(s) with missing or invalid values." if dropped else None)


# ---------------------------------------------------------------- session state

def reset_stream():
    st.session_state.streaming = False
    st.session_state.row_index = 0
    st.session_state.history = []
    st.session_state.scored = 0
    st.session_state.labelled = 0
    st.session_state.correct = 0
    st.session_state.last_error = None


if "streaming" not in st.session_state:
    reset_stream()
    st.session_state.api_stats = None
    threading.Thread(target=wake_api, daemon=True).start()


def start_streaming():
    st.session_state.streaming = True


def stop_streaming():
    st.session_state.streaming = False


def predict_next(rows):
    idx = st.session_state.row_index % len(rows)
    st.session_state.row_index = idx + 1
    row = rows.iloc[idx]

    data = {f: float(row[f]) for f in FEATURES}
    for f in INT_FEATURES:
        data[f] = int(data[f])

    sent = time.perf_counter()
    try:
        resp = http_session().post(f"{API_URL}/predict", json=data, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        prob = result["failure_probability"]
    except (requests.ConnectionError, requests.Timeout):
        st.session_state.last_error = (
            "The prediction API did not respond. If it is hosted on a free tier it may be waking up, "
            "which can take up to a minute. Retrying automatically."
        )
        return
    except (requests.RequestException, KeyError, ValueError) as e:
        st.session_state.last_error = f"Prediction request failed: {e}"
        return
    round_trip_ms = (time.perf_counter() - sent) * 1000

    try:
        st.session_state.api_stats = api_get("/stats", timeout=REQUEST_TIMEOUT)
    except (requests.RequestException, ValueError):
        pass

    actual = int(row[LABEL]) if LABEL in rows.columns and pd.notna(row[LABEL]) else None
    predicted = int(prob >= THRESHOLD)
    st.session_state.last_error = None
    st.session_state.scored += 1
    if actual is not None:
        st.session_state.labelled += 1
        st.session_state.correct += int(predicted == actual)

    st.session_state.history.append({
        "job": st.session_state.scored,
        **data,
        "probability": prob,
        "predicted": predicted,
        "actual": actual,
        "cached": result.get("cached", False),
        "server_ms": result.get("latency_ms"),
        "round_trip_ms": round_trip_ms,
        "seen": bool(row[SEEN]) if SEEN in rows.columns else None,
    })
    del st.session_state.history[:-HISTORY_SIZE]


# ---------------------------------------------------------------- presentation helpers

def risk_level(prob):
    if prob >= 0.7:
        return "High risk", "red"
    if prob >= 0.3:
        return "Moderate risk", "orange"
    return "Low risk", "green"


def outcome(value):
    return {1: "Failure", 0: "Success"}.get(value, "Unknown")


def fmt_ms(value):
    return "–" if value is None else f"{value:,.1f} ms"


def probability_chart(history):
    df = pd.DataFrame(history[-CHART_WINDOW:])
    df["Actual outcome"] = df["actual"].map(outcome)
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("job:O", title="Job", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("probability:Q", title="Failure probability", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("Actual outcome:N",
                        scale=alt.Scale(domain=["Failure", "Success", "Unknown"], range=[RED, GREEN, GREY]),
                        legend=alt.Legend(orient="top", title="Actual outcome")),
        tooltip=[
            alt.Tooltip("job:O", title="Job"),
            alt.Tooltip("probability:Q", title="Failure probability", format=".1%"),
            alt.Tooltip("Actual outcome:N"),
            alt.Tooltip("cpu_request:Q", title="CPU request", format=".4f"),
            alt.Tooltip("memory_request:Q", title="Memory request", format=".4f"),
            alt.Tooltip("priority:Q", title="Priority"),
            alt.Tooltip("scheduling_class:Q", title="Scheduling class"),
        ],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
        strokeDash=[5, 4], color="#334155", strokeWidth=1.5
    ).encode(y="y:Q")
    return (bars + threshold).properties(height=345)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Controls")
    source = st.radio("Data source", ["Held-out test set", "Upload CSV"], on_change=reset_stream, key="source")

    rows, upload_error, upload_note = None, None, None
    if source == "Held-out test set":
        unseen_only = st.toggle(
            "Only unseen resource profiles",
            on_change=reset_stream,
            help="Stream only the 987 test jobs whose inputs never appeared in training: "
                 "the honest test of how the model generalises to new kinds of jobs.",
        )
        holdout = load_holdout()
        rows = holdout[holdout[SEEN] == 0].reset_index(drop=True) if unseen_only else holdout
        st.caption(f"{len(rows):,} jobs from the 30% test split, never used for training.")
    else:
        uploaded = st.file_uploader(
            "CSV file", type="csv", on_change=reset_stream,
            help=f"Required columns: {', '.join(FEATURES)}. Optional: {LABEL} (0 or 1) to track accuracy.",
        )
        if uploaded is not None:
            rows, upload_error, upload_note = clean_upload(pd.read_csv(uploaded))

    interval = st.select_slider(
        "Interval between jobs", options=[1, 2, 3, 5], value=2, format_func=lambda s: f"{s} s",
    )

    c1, c2 = st.columns(2)
    c1.button("Start", type="primary", on_click=start_streaming, width="stretch",
              disabled=st.session_state.streaming or rows is None)
    c2.button("Stop", on_click=stop_streaming, width="stretch",
              disabled=not st.session_state.streaming)
    st.button("Reset session", on_click=reset_stream, width="stretch", type="tertiary")

    st.divider()
    st.caption(f"[API documentation]({PUBLIC_API_URL}/docs) · [Source code]({REPO_URL})")


# ---------------------------------------------------------------- header

head_left, head_right = st.columns([3, 2], vertical_alignment="bottom")
with head_left:
    st.title("Cloud Task Failure Prediction")
    st.html('<div class="subtitle">Real-time failure risk scoring for cluster jobs, '
            'using a random forest trained on the Google Borg 2019 cluster trace.</div>')


@st.fragment(run_every="10s")
def status_badges():
    health = api_health()
    with st.container(horizontal=True, horizontal_alignment="right"):
        if health is None:
            st.badge("API: waking up or offline", color="orange")
        else:
            st.badge("API: online", color="green")
            redis_status = health.get("redis")
            if redis_status == "ok":
                st.badge("Redis: connected", color="green")
            elif redis_status == "unreachable":
                st.badge("Redis: unreachable", color="red")
            else:
                st.badge("Redis: disabled", color="gray")
            st.badge(f"Model {health.get('model_version', '')}", color="gray")


with head_right:
    status_badges()


# ---------------------------------------------------------------- tabs

live_tab, model_tab, arch_tab = st.tabs(["Live monitor", "Model performance", "Architecture"])


@st.fragment(run_every=f"{interval}s" if st.session_state.streaming else None)
def live_monitor(rows):
    if st.session_state.streaming and rows is not None:
        predict_next(rows)

    if upload_error:
        st.error(upload_error)
    if upload_note:
        st.caption(upload_note)
    if st.session_state.last_error:
        st.warning(st.session_state.last_error)

    history = st.session_state.history
    if not history:
        if rows is None and not upload_error:
            st.info("Upload a CSV in the sidebar to begin.")
        elif st.session_state.streaming:
            st.info("Waiting for the first prediction.")
        elif rows is not None:
            st.info("Press Start in the sidebar to stream jobs to the prediction API.")
        return

    latest = history[-1]
    api_stats = st.session_state.api_stats or {}

    scored, labelled = st.session_state.scored, st.session_state.labelled
    predicted_failures = sum(h["predicted"] for h in history)
    hit_rate = api_stats.get("hit_rate")
    hit_ms, miss_ms = api_stats.get("avg_ms_cache_hit"), api_stats.get("avg_ms_cache_miss")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Jobs scored", f"{scored:,}", border=True, height=KPI_HEIGHT)
    k2.metric("Live accuracy",
              f"{st.session_state.correct / labelled:.1%}" if labelled else "–", border=True, height=KPI_HEIGHT,
              help="Share of jobs this session where the predicted outcome (threshold 50%) "
                   "matched the actual outcome recorded in the trace.")
    k3.metric("Predicted to fail", f"{predicted_failures / len(history):.0%}", border=True, height=KPI_HEIGHT,
              help=f"Share of the last {len(history)} jobs with a failure probability of at least 50%.")
    k4.metric("Cache hit rate", "–" if hit_rate is None else f"{hit_rate:.0%}", border=True, height=KPI_HEIGHT,
              help=f"Across all {api_stats.get('predictions', 0):,} predictions served by the API since it started.")
    latency = " / ".join("–" if v is None else f"{v:,.1f}" for v in (hit_ms, miss_ms))
    k5.metric("Cache vs. model", f"{latency} ms" if api_stats else "–", border=True, height=KPI_HEIGHT,
              help="Average server-side time to answer from the Redis cache vs. running the model (ms).")

    chart_col, latest_col = st.columns([2, 1])
    with chart_col, st.container(border=True):
        st.markdown("**Failure probability by job**")
        st.altair_chart(probability_chart(history), use_container_width=True, key="probability_chart")
        st.caption("Each bar is one job. Dashed line: 50% decision threshold. Colour: what actually happened.")

    with latest_col, st.container(border=True):
        st.markdown(f"**Latest job** &nbsp; <span class='muted'>#{latest['job']}</span>", unsafe_allow_html=True)
        label, color = risk_level(latest["probability"])
        st.badge(label, color=color)
        st.html(f"<div class='big-prob'>{latest['probability']:.1%}</div>"
                "<div class='muted'>probability of failure</div>")
        details = [
            ("Predicted outcome", outcome(latest["predicted"])),
            ("Actual outcome", outcome(latest["actual"])),
            ("Served by", "Redis cache" if latest["cached"] else "Model"),
            ("Server time", fmt_ms(latest["server_ms"])),
            ("Round trip", fmt_ms(latest["round_trip_ms"])),
        ]
        if latest["seen"] is not None:
            details.append(("Inputs seen in training", "Yes" if latest["seen"] else "No"))
        st.dataframe(pd.DataFrame(details, columns=["Field", "Value"]), hide_index=True)

    with st.container(border=True):
        st.markdown("**Recent predictions, this session**")
        recent = pd.DataFrame(history[::-1][:25])
        table = pd.DataFrame({
            "Job": recent["job"],
            "CPU": recent["cpu_request"],
            "Memory": recent["memory_request"],
            "Priority": recent["priority"],
            "Class": recent["scheduling_class"],
            "Failure probability": recent["probability"],
            "Predicted": recent["predicted"].map(outcome),
            "Actual": recent["actual"].map(outcome),
            "Correct": [None if a is None else ("Yes" if p == a else "No")
                        for p, a in zip(recent["predicted"], recent["actual"])],
            "Source": recent["cached"].map({True: "Cache", False: "Model"}),
            "Latency (ms)": recent["server_ms"],
        })
        st.dataframe(
            table, hide_index=True, height=320,
            column_config={
                "Failure probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "CPU": st.column_config.NumberColumn(format="%.4f"),
                "Memory": st.column_config.NumberColumn(format="%.4f"),
                "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
            },
        )


@st.fragment(run_every="2s")
def feed_panel():
    with st.container(border=True):
        st.markdown("**Live feed, all users**")
        if not REDIS_URL:
            st.caption("Set REDIS_URL to show every user's predictions, streamed over Redis Pub/Sub.")
            return
        feed = live_feed()
        st.caption(f"Streamed over Redis Pub/Sub, channel `{PREDICTIONS_CHANNEL}` · {feed.status}")
        messages = feed.items()
        if not messages:
            st.caption("Waiting for predictions.")
            return
        st.dataframe(
            pd.DataFrame([
                {
                    "Time (UTC)": m.get("published_at", "")[11:19],
                    "CPU": m.get("data", {}).get("cpu_request"),
                    "Memory": m.get("data", {}).get("memory_request"),
                    "Priority": m.get("data", {}).get("priority"),
                    "Class": m.get("data", {}).get("scheduling_class"),
                    "Failure probability": m.get("failure_probability"),
                    "Source": "Cache" if m.get("cached") else "Model",
                }
                for m in messages
            ]),
            hide_index=True,
            column_config={
                "Failure probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "CPU": st.column_config.NumberColumn(format="%.4f"),
                "Memory": st.column_config.NumberColumn(format="%.4f"),
            },
        )


with live_tab:
    live_monitor(rows)
    feed_panel()


# ---------------------------------------------------------------- model performance

def importance_chart(importances):
    df = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in importances],
                       "Importance": list(importances.values())})
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
        x=alt.X("Importance:Q", axis=alt.Axis(format="%")),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".1%")],
    ).properties(height=190)


def confusion_chart(matrix):
    labels = ["Success", "Failure"]
    df = pd.DataFrame([
        {"Actual": labels[i], "Predicted": labels[j], "Jobs": matrix[i][j]}
        for i in range(2) for j in range(2)
    ])
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=labels),
    )
    cells = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Jobs:Q", scale=alt.Scale(scheme="blues"), legend=None),
    )
    text = base.mark_text(fontSize=16, fontWeight=600).encode(
        text="Jobs:Q",
        color=alt.condition(alt.datum.Jobs > max(max(r) for r in matrix) / 2, alt.value("white"), alt.value("#0F172A")),
    )
    return (cells + text).properties(height=190)


with model_tab:
    info = model_info()
    evaluation = (info or {}).get("evaluation")
    if not info:
        st.info("Model details are loaded from the API, which appears to be offline or waking up. Refresh in a moment.")
    elif not evaluation:
        st.info("No evaluation results found. Run notebook/evaluate_unseen.py to generate them.")
    else:
        subsets = evaluation["subsets"]
        unseen, all_rows = subsets["unseen_inputs"], subsets["all_test_rows"]
        dataset = evaluation["dataset"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy (unseen inputs)", f"{unseen['accuracy']:.1%}", border=True,
                  delta=f"{(unseen['accuracy'] - unseen['majority_baseline']) * 100:+.1f} pts vs. baseline",
                  help="Test jobs whose resource profile never appeared in training. The best estimate "
                       "of performance on new kinds of jobs.")
        m2.metric("F1 (unseen inputs)", f"{unseen['f1']:.3f}", border=True)
        m3.metric("ROC-AUC (unseen inputs)", f"{unseen['roc_auc']:.3f}", border=True)
        m4.metric("Accuracy (random split)", f"{all_rows['accuracy']:.1%}", border=True,
                  help="The standard random 70/30 split. Optimistic, because most test inputs also appear in training.")

        st.markdown(
            f"The processed trace has **{dataset['rows']:,} jobs** but only "
            f"**{dataset['distinct_feature_combinations']:,} distinct resource profiles**, so a random "
            "train/test split places identical inputs on both sides and mostly measures recall of seen inputs. "
            "The model is therefore also evaluated on the test jobs whose inputs never occur in training."
        )

        names = {"all_test_rows": "All test jobs (random split)", "seen_inputs": "Inputs seen in training",
                 "unseen_inputs": "Inputs not seen in training"}
        st.dataframe(
            pd.DataFrame([
                {"Test subset": names[k], "Jobs": v["rows"], "Accuracy": v["accuracy"], "Precision": v["precision"],
                 "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"], "Majority baseline": v["majority_baseline"]}
                for k, v in subsets.items()
            ]),
            hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Majority baseline"]},
        )

        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown("**Feature importance**")
            st.altair_chart(importance_chart(info["feature_importances"]), use_container_width=True)
            st.caption("Mean decrease in impurity across the forest's trees.")
        with right, st.container(border=True):
            st.markdown("**Confusion matrix, unseen inputs**")
            st.altair_chart(confusion_chart(unseen["confusion_matrix"]), use_container_width=True)
            st.caption(f"{unseen['rows']:,} test jobs, decision threshold {evaluation['decision_threshold']:.0%}.")

        with st.container(border=True):
            st.markdown("**Model**")
            st.markdown(
                f"- **Algorithm:** {info['algorithm']} with {info['n_estimators']} trees, tuned by grid search "
                "(3-fold cross-validation, F1), with SMOTE oversampling applied to the training split only.\n"
                "- **Inputs:** CPU request, memory request, priority and scheduling class: information available "
                "when a job is submitted, before it runs. Event-type columns were excluded because they record the "
                "outcome itself.\n"
                "- **Label:** a job is a failure if it ended in FAIL, EVICT, LOST or KILL, and a success if it FINISHED.\n"
                f"- **Version:** `{info['model_version']}` (hash of the model file; also part of every cache key)."
            )


# ---------------------------------------------------------------- architecture

with arch_tab:
    with st.container(border=True):
        st.markdown("**System overview**")
        st.graphviz_chart("""
        digraph {
            rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.9; pad=0.2;
            node [shape=box, style="rounded,filled", fillcolor="#F5F7FA", color="#CBD5E1",
                  fontname="Helvetica", fontsize=13, margin="0.28,0.14", penwidth=1.2];
            edge [color="#64748B", fontname="Helvetica", fontsize=11, penwidth=1.2];
            dash  [label="Dashboard\nStreamlit"];
            api   [label="Prediction API\nFastAPI", fillcolor="#DBEAFE", color="#2563EB"];
            model [label="Random forest\nscikit-learn"];
            cache [label="Prediction cache\nRedis"];
            chan  [label="Pub/Sub channel\nRedis 'predictions'"];
            dash -> api   [label="POST /predict"];
            api  -> cache [label="1. lookup / store"];
            api  -> model [label="2. on cache miss"];
            api  -> chan  [label="3. publish"];
            chan -> dash  [label="subscribe (live feed)", style=dashed];
        }
        """, width="stretch")

    left, right = st.columns([2, 3])
    with left, st.container(border=True):
        st.markdown("**Request lifecycle**")
        st.markdown(
            "1. The dashboard sends a job's resource request to `POST /predict`.\n"
            "2. The API validates it, then looks up the prediction in Redis, keyed by model version and inputs.\n"
            "3. On a miss it runs the model and caches the result for 24 hours.\n"
            "4. Every prediction is published to a Redis Pub/Sub channel.\n"
            "5. A background subscriber in the dashboard shows all users' predictions in the live feed.\n\n"
            "If Redis is unavailable the API keeps serving uncached predictions."
        )
    with right, st.container(border=True):
        st.markdown("**Technology**")
        st.markdown(
            "| Layer | Technology | Why |\n"
            "|---|---|---|\n"
            "| Model | scikit-learn random forest | Strong on tabular data, fast single-row inference, interpretable importances |\n"
            "| API | FastAPI + Uvicorn | Request validation and interactive docs generated from type hints |\n"
            "| Cache and messaging | Redis | Sub-millisecond lookups; Pub/Sub broadcasts predictions to all listeners |\n"
            "| Dashboard | Streamlit | Interactive data application in pure Python |\n"
            "| Packaging and hosting | Docker, Render | Reproducible images deployed as independent web services |"
        )
