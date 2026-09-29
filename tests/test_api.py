import json

import pytest
import redis
from fastapi.testclient import TestClient

from app import app as api

VALID = {"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}


class FakeRedis:
    def __init__(self, fail=False):
        self.fail = fail
        self.published = []

    def publish(self, channel, message):
        if self.fail:
            raise redis.ConnectionError("down")
        self.published.append((channel, json.loads(message)))

    def ping(self):
        if self.fail:
            raise redis.ConnectionError("down")
        return True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "r", None)
    return TestClient(api.app)


def test_health_without_redis(client):
    assert client.get("/health").json() == {"status": "ok", "redis": "disabled"}


def test_predict_returns_probability(client):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["failure_probability"] <= 1.0


@pytest.mark.parametrize("field,value", [
    ("cpu_request", -0.1),
    ("memory_request", -1),
    ("priority", -5),
    ("scheduling_class", 4),
    ("priority", 1.5),
])
def test_predict_rejects_invalid_input(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_predict_rejects_missing_field(client):
    body = {k: v for k, v in VALID.items() if k != "priority"}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_publishes_to_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(api, "r", fake)
    prob = TestClient(api.app).post("/predict", json=VALID).json()["failure_probability"]
    assert fake.published == [
        (api.PREDICTIONS_CHANNEL, {"data": VALID, "failure_probability": prob})
    ]


def test_predict_survives_redis_outage(monkeypatch):
    monkeypatch.setattr(api, "r", FakeRedis(fail=True))
    client = TestClient(api.app)
    assert client.post("/predict", json=VALID).status_code == 200
    assert client.get("/health").json()["redis"] == "unreachable"
