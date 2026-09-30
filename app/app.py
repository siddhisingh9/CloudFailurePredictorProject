import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import redis
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "failure_model.pkl"))
EVALUATION_PATH = MODEL_PATH.with_name("evaluation.json")
FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
PREDICTIONS_CHANNEL = "predictions"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 24 * 60 * 60))

model = joblib.load(MODEL_PATH)
# Part of every cache key, so retraining the model invalidates old cached predictions
MODEL_VERSION = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
# Written by notebook/evaluate_unseen.py; optional so a freshly trained model still serves
EVALUATION = json.loads(EVALUATION_PATH.read_text()) if EVALUATION_PATH.exists() else None


def connect_redis():
    """Redis is optional: without REDIS_URL the API still serves predictions, uncached."""
    url = os.getenv("REDIS_URL")
    if not url:
        logger.info("REDIS_URL not set; caching and publishing are disabled")
        return None
    return redis.from_url(url, decode_responses=True, socket_timeout=1, socket_connect_timeout=1)


r = connect_redis()


class Stats:
    """In-process counters for cache effectiveness, reset when the service restarts."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.counts = {"hit": 0, "miss": 0}
        self.total_ms = {"hit": 0.0, "miss": 0.0}

    def record(self, kind, ms):
        with self.lock:
            self.counts[kind] += 1
            self.total_ms[kind] += ms

    def snapshot(self):
        with self.lock:
            hits, misses = self.counts["hit"], self.counts["miss"]
            total = hits + misses
            return {
                "since": self.started_at,
                "predictions": total,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hits / total, 3) if total else None,
                "avg_ms_cache_hit": round(self.total_ms["hit"] / hits, 1) if hits else None,
                "avg_ms_cache_miss": round(self.total_ms["miss"] / misses, 1) if misses else None,
            }


stats = Stats()


class Metrics(BaseModel):
    """Resource request of a single job, as recorded in the Borg trace."""

    cpu_request: float = Field(ge=0, description="CPU requested, normalised to the largest machine (0-1)")
    memory_request: float = Field(ge=0, description="Memory requested, normalised to the largest machine (0-1)")
    priority: int = Field(ge=0, description="Job priority (0-450 in the trace)")
    scheduling_class: int = Field(ge=0, le=3, description="Latency sensitivity, 0 (batch) to 3 (most latency-sensitive)")

    model_config = {"json_schema_extra": {"examples": [
        {"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}
    ]}}


class Prediction(BaseModel):
    failure_probability: float = Field(description="Probability that the job fails, evicts, is lost or is killed")
    cached: bool = Field(description="True if served from the Redis cache instead of running the model")
    latency_ms: float = Field(description="Server-side time to produce the prediction")


def cache_key(metrics):
    values = ":".join(repr(getattr(metrics, f)) for f in FEATURES)
    return f"pred:{MODEL_VERSION}:{values}"


def cached_probability(key):
    try:
        value = r.get(key)
    except redis.RedisError as e:
        logger.warning("Redis cache read failed: %s", e)
        return None
    return float(value) if value is not None else None


def store_probability(key, prob):
    try:
        r.set(key, prob, ex=CACHE_TTL_SECONDS)
    except redis.RedisError as e:
        logger.warning("Redis cache write failed: %s", e)


def publish(metrics, prediction):
    payload = {
        "data": metrics.model_dump(),
        **prediction.model_dump(),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        r.publish(PREDICTIONS_CHANNEL, json.dumps(payload))
    except redis.RedisError as e:
        logger.warning("Redis publish failed: %s", e)


app = FastAPI(
    title="Cloud Failure Prediction API",
    description=(
        "Predicts whether a job in a compute cluster will fail, from the resources it requests. "
        "Random forest trained on the Google Borg cluster trace (2019). Predictions are cached "
        "in Redis and published to the Redis Pub/Sub channel `predictions`."
    ),
    version=MODEL_VERSION,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", summary="Service and dependency status")
def health():
    if r is None:
        redis_status = "disabled"
    else:
        try:
            r.ping()
            redis_status = "ok"
        except redis.RedisError:
            redis_status = "unreachable"
    return {"status": "ok", "redis": redis_status, "model_version": MODEL_VERSION}


@app.get("/stats", summary="Cache hit rate and latency since the service started")
def get_stats():
    return stats.snapshot()


@app.get("/model", summary="Model details, feature importances and evaluation results")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "algorithm": type(model).__name__,
        "n_estimators": model.n_estimators,
        "features": FEATURES,
        "feature_importances": dict(zip(FEATURES, (round(float(v), 4) for v in model.feature_importances_))),
        "evaluation": EVALUATION,
    }


@app.post("/predict", response_model=Prediction, summary="Predict the failure probability of a job")
def predict(metrics: Metrics):
    start = time.perf_counter()
    key = cache_key(metrics)
    prob = cached_probability(key) if r is not None else None
    cached = prob is not None

    if not cached:
        features = pd.DataFrame([metrics.model_dump()], columns=FEATURES)
        prob = float(model.predict_proba(features)[0][1])
        if r is not None:
            store_probability(key, prob)

    latency_ms = (time.perf_counter() - start) * 1000
    stats.record("hit" if cached else "miss", latency_ms)
    prediction = Prediction(failure_probability=prob, cached=cached, latency_ms=round(latency_ms, 2))
    if r is not None:
        publish(metrics, prediction)

    return prediction
