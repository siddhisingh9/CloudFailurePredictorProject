import json

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient

from app import app as api

VALID = {"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}


class BrokenRedis:
    """Every call fails, like a Redis server that is down."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise redis.ConnectionError("down")
        return fail


@pytest.fixture(autouse=True)
def fresh_stats(monkeypatch):
    monkeypatch.setattr(api, "stats", api.Stats())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "r", None)
    return TestClient(api.app)


@pytest.fixture
def fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(api, "r", fake)
    return fake


def test_health_without_redis(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["redis"] == "disabled"
    assert body["model_version"] == api.MODEL_VERSION


def test_predict_returns_probability(client):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["failure_probability"] <= 1.0
    assert resp.json()["cached"] is False


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


def test_second_identical_request_is_served_from_cache(fake_redis):
    client = TestClient(api.app)
    first = client.post("/predict", json=VALID).json()
    second = client.post("/predict", json=VALID).json()

    assert first["cached"] is False
    assert second == {"failure_probability": first["failure_probability"], "cached": True}
    assert client.get("/stats").json()["cache_hits"] == 1


def test_cache_entries_expire_and_include_model_version(fake_redis):
    TestClient(api.app).post("/predict", json=VALID)
    (key,) = fake_redis.keys("pred:*")
    assert key.startswith(f"pred:{api.MODEL_VERSION}:")
    assert 0 < fake_redis.ttl(key) <= api.CACHE_TTL_SECONDS


def test_cached_value_is_used(fake_redis):
    key = api.cache_key(api.Metrics(**VALID))
    fake_redis.set(key, 0.4242)
    resp = TestClient(api.app).post("/predict", json=VALID).json()
    assert resp == {"failure_probability": 0.4242, "cached": True}


def test_predictions_are_published(fake_redis):
    pubsub = fake_redis.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(api.PREDICTIONS_CHANNEL)
    client = TestClient(api.app)
    client.post("/predict", json=VALID)
    client.post("/predict", json=VALID)

    # get_message() returns None for the (ignored) subscribe confirmation, so poll a few times
    raw = [pubsub.get_message(timeout=0.1) for _ in range(5)]
    messages = [json.loads(m["data"]) for m in raw if m is not None]
    assert [m["cached"] for m in messages] == [False, True]
    assert all(m["data"] == VALID and "published_at" in m for m in messages)


def test_predict_survives_redis_outage(monkeypatch):
    monkeypatch.setattr(api, "r", BrokenRedis())
    client = TestClient(api.app)
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    assert resp.json()["cached"] is False
    assert client.get("/health").json()["redis"] == "unreachable"


def test_stats_reports_hit_rate(fake_redis):
    client = TestClient(api.app)
    assert client.get("/stats").json()["hit_rate"] is None
    for _ in range(4):
        client.post("/predict", json=VALID)
    stats = client.get("/stats").json()
    assert (stats["predictions"], stats["cache_hits"], stats["cache_misses"]) == (4, 3, 1)
    assert stats["hit_rate"] == 0.75
