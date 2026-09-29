import json
import logging
import os
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

model = joblib.load(MODEL_PATH)


def connect_redis():
    """Redis is optional: without REDIS_URL the API still serves predictions."""
    url = os.getenv("REDIS_URL")
    if not url:
        logger.info("REDIS_URL not set; predictions will not be published")
        return None
    return redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)


r = connect_redis()


class Metrics(BaseModel):
    # CPU and memory are normalised to the largest machine in the Borg trace
    cpu_request: float = Field(ge=0)
    memory_request: float = Field(ge=0)
    priority: int = Field(ge=0)
    scheduling_class: int = Field(ge=0, le=3)


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
    return {"status": "ok", "redis": redis_status}


@app.post("/predict")
def predict(metrics: Metrics):
    features = pd.DataFrame([metrics.model_dump()], columns=FEATURES)
    prob = float(model.predict_proba(features)[0][1])

    if r is not None:
        payload = {"data": metrics.model_dump(), "failure_probability": prob}
        try:
            r.publish(PREDICTIONS_CHANNEL, json.dumps(payload))
        except redis.RedisError as e:
            logger.warning("Redis publish failed: %s", e)

    return {"failure_probability": prob}
