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
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "failure_model.pkl"))
FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
PREDICTIONS_CHANNEL = "predictions"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 24 * 60 * 60))

model = joblib.load(MODEL_PATH)
# Part of every cache key, so retraining the model invalidates old cached predictions
MODEL_VERSION = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]


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
                "predictions": total,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hits / total, 3) if total else None,
                "avg_ms_cache_hit": round(self.total_ms["hit"] / hits, 1) if hits else None,
                "avg_ms_cache_miss": round(self.total_ms["miss"] / misses, 1) if misses else None,
            }


stats = Stats()


class Metrics(BaseModel):
    # CPU and memory are normalised to the largest machine in the Borg trace
    cpu_request: float = Field(ge=0)
    memory_request: float = Field(ge=0)
    priority: int = Field(ge=0)
    scheduling_class: int = Field(ge=0, le=3)


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


def publish(metrics, prob, cached):
    payload = {
        "data": metrics.model_dump(),
        "failure_probability": prob,
        "cached": cached,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        r.publish(PREDICTIONS_CHANNEL, json.dumps(payload))
    except redis.RedisError as e:
        logger.warning("Redis publish failed: %s", e)


app = FastAPI(title="Cloud Failure Prediction API")


@app.get("/health")
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


@app.get("/stats")
def get_stats():
    return stats.snapshot()


@app.post("/predict")
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

    stats.record("hit" if cached else "miss", (time.perf_counter() - start) * 1000)
    if r is not None:
        publish(metrics, prob, cached)

    return {"failure_probability": prob, "cached": cached}
