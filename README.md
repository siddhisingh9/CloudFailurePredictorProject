# Cloud Task Failure Prediction

[![tests](https://github.com/siddhisingh9/CloudFailurePredictorProject/actions/workflows/tests.yml/badge.svg)](https://github.com/siddhisingh9/CloudFailurePredictorProject/actions/workflows/tests.yml)

Predicts whether a job in a compute cluster will fail, using only the resources it requests at submission time, so that high-risk jobs can be identified before they run. The project is an end-to-end system: a machine learning model trained on the Google Borg cluster trace, a prediction API with a Redis cache, real-time streaming over Redis Pub/Sub, and a live monitoring dashboard, deployed on free cloud tiers.

---

## Highlights

- **Honest evaluation.** A random forest reaches **83.4% accuracy (0.87 F1, 0.90 ROC-AUC) on jobs whose resource profile never appeared in training**, against a 63.2% majority-class baseline. The standard random split reports 92.9%, but most of its test inputs also occur in training, so the project measures generalisation separately (see [Evaluation](#evaluation)).
- **Leakage removal.** The trace's event-type columns predict the outcome with 100% accuracy on their own because they record how the job ended. The model uses only information available when a job is submitted.
- **Redis prediction cache.** The trace has 207,762 jobs but only 6,514 distinct resource profiles, so repeat requests are common: replaying the data gives a 97% hit rate, and a cache hit takes about 1 ms locally versus about 30 ms for the model. Cache keys include a hash of the model file, so retraining invalidates old entries automatically.
- **Real-time streaming.** Every prediction is published to a Redis Pub/Sub channel; the dashboard subscribes to it and shows a live feed of all users' predictions.
- **Live monitoring dashboard.** Streams held-out test jobs to the API and tracks live accuracy against the real outcomes, cache hit rate and latency. Includes model performance and architecture views.
- **Production practices.** Input validation, graceful degradation when Redis is down, health and statistics endpoints, pinned dependencies matched to the pickled model, 16 automated tests, Docker images and a docker-compose stack.

---

## Tech stack

| Layer | Technology |
|---|---|
| Machine learning | scikit-learn (random forest), imbalanced-learn (SMOTE), pandas |
| API | FastAPI, Uvicorn, Pydantic |
| Caching and messaging | Redis (key-value cache and Pub/Sub) |
| Dashboard | Streamlit, Altair |
| Packaging and hosting | Docker, docker-compose, Render |
| Testing | pytest, fakeredis |

---

## Architecture

```
                       POST /predict                          GET / SET (cache)
 dashboard (Streamlit) ─────────────▶ api (FastAPI + model) ───────────────────▶ Redis
        ▲                                                       PUBLISH             │
        └──────────────────── SUBSCRIBE "predictions" (live feed) ◀─────────────────┘
```

- **api** (`app/`) serves the model. With `REDIS_URL` set, it checks the Redis cache before running the model, stores new predictions with a 24-hour expiry, and publishes every prediction to the `predictions` channel.
- **dashboard** (`dashboard/`) replays jobs from the held-out test set (or a CSV you upload), sends them to the API, and charts the results. A background thread subscribes to `predictions` for the live feed.
- **notebook** (`notebook/`) cleans the raw trace into `data/processed_gct.csv`, trains `models/failure_model.pkl`, and evaluates it on unseen inputs (`evaluate_unseen.py`, which also writes `models/evaluation.json` and `data/demo_holdout.csv`).

Redis is optional. Without it the API serves uncached predictions, and a Redis outage never fails a request.

---

## Running with Docker

```sh
docker compose up --build
```

Then open http://localhost:8501. The API's interactive docs are at http://localhost:8000/docs.

## Running locally

Requires Python 3.10+.

```sh
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.app:app --port 8000            # terminal 1
streamlit run dashboard/dashboard.py       # terminal 2
```

Set `REDIS_URL` in both terminals to enable caching and the live feed. Run the tests with `pytest`; they use fakeredis, so they don't need a Redis server.

### Configuration

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `REDIS_URL` | api, dashboard | unset | Redis connection URL. If unset, caching, publishing and the live feed are disabled. |
| `CACHE_TTL_SECONDS` | api | `86400` | How long cached predictions are kept. |
| `MODEL_PATH` | api | `models/failure_model.pkl` | Path to the trained model. |
| `API_URL` | dashboard | `http://localhost:8000` | Base URL of the API, **without** `/predict`. |
| `PUBLIC_API_URL` | dashboard | `API_URL` | API address for the dashboard's documentation link, if different from `API_URL`. |
| `DEMO_DATA_PATH` | dashboard | `data/demo_holdout.csv` | Jobs replayed in demo mode. |
| `PORT` | both Docker images | `8000` / `8501` | Port to listen on; set automatically by Render. |

---

## Deploying on free tiers

Everything fits within free plans: the API uses about 210 MB of RAM (Render's free limit is 512 MB) and the cache uses well under 1 MB of Redis.

1. **Redis:** create a free database on [Redis Cloud](https://redis.io/cloud/) (30 MB) and copy its connection URL (`redis://default:<password>@<host>:<port>`). Pick the region closest to your Render services; every cache lookup is a network round trip, so a distant Redis can be slower than the model. The dashboard's latency figures show whether the cache is paying off.
2. **API on Render:** New, Web Service, this repo; runtime **Docker**, Dockerfile path `app/Dockerfile`, build context `.` (repo root), instance type **Free**. Set `REDIS_URL`. Health check path: `/health`.
3. **Dashboard**, either:
   - **on Render:** another Docker web service with Dockerfile path `dashboard/Dockerfile`, build context `.`, and environment variables `API_URL` (the API's public URL, e.g. `https://<api-name>.onrender.com`) and `REDIS_URL`; or
   - **on [Streamlit Community Cloud](https://streamlit.io/cloud):** New app, this repo, main file `dashboard/dashboard.py`. It installs `dashboard/requirements.txt` and applies the theme in `.streamlit/config.toml` automatically. Under Advanced settings, Secrets, add `API_URL = "https://<api-name>.onrender.com"` and `REDIS_URL = "redis://..."`; top-level secrets are exposed to the app as environment variables.

**Free-tier behaviour to expect:** Render's free services sleep after 15 minutes without traffic and take 30 to 60 seconds to wake. The dashboard pings the API as soon as it opens and shows a "waking up" status instead of failing while it waits. Cache statistics are kept in memory, so they reset whenever the API restarts or sleeps; cached predictions in Redis survive.

---

## API

Interactive documentation is served at `/docs`; the root URL redirects there.

`POST /predict`

```json
{"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}
```

returns `{"failure_probability": 0.97, "cached": false, "latency_ms": 27.4}`.

`cpu_request` and `memory_request` are normalised to the largest machine in the trace (0 to 1). `priority` is a non-negative integer (0 to 450 in the trace) and `scheduling_class` is 0 to 3. Invalid input gets a 422.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "redis": "ok" \| "disabled" \| "unreachable", "model_version": "058e4618e0ff"}` |
| `GET /stats` | Cache hit rate and average latency of cache hits and misses since the API started |
| `GET /model` | Model version, feature importances and the evaluation results from `models/evaluation.json` |

Each message on the `predictions` channel is JSON: `{"data": {...inputs}, "failure_probability": 0.97, "cached": false, "latency_ms": 27.4, "published_at": "2026-09-29T12:00:00+00:00"}`.

---

## Model

- **Data:** 405,894 task events from the Google Borg cluster trace (2019), reduced to 207,762 completed jobs after removing incomplete records, jobs without timestamps, non-terminal events and duplicates.
- **Features:** `cpu_request`, `memory_request`, `priority`, `scheduling_class`: what a job requests when it is submitted.
- **Label:** a job counts as *failed* if its final event is `FAIL`, `EVICT`, `LOST` or `KILL`, and as *not failed* if it is `FINISH` (55.9% failed).
- **Training:** random forest with 200 trees, hyperparameters chosen by grid search (108 combinations, 3-fold cross-validation, F1), with SMOTE oversampling applied to the training split only.
- **Versions:** the model was pickled with scikit-learn 1.7.1 and NumPy 2.x. Keep the versions pinned in `app/requirements.txt` or the pickle may not load.

### Evaluation

The 207,762 processed rows contain only 6,514 distinct feature combinations, so the notebook's random 70/30 split puts identical inputs in both the training and test sets, and the score on that split mostly measures inputs the model has already seen. `notebook/evaluate_unseen.py` rebuilds the exact split (it reproduces the notebook's confusion matrix) and scores the saved model separately on test rows whose inputs never appear in training:

| Test rows | Rows | Accuracy | F1 | ROC-AUC | Majority-class baseline |
|---|---|---|---|---|---|
| All (random split) | 62,329 | 0.929 | 0.935 | 0.977 | 0.559 |
| Inputs seen in training | 61,342 | 0.931 | 0.936 | 0.978 | 0.557 |
| **Inputs not seen in training** | **987** | **0.834** | **0.867** | **0.902** | **0.632** |

The last row is the better estimate of performance on new kinds of jobs. It is based on 987 rows, so it carries roughly ±2 percentage points of uncertainty. In 825 of the feature combinations the same inputs appear with both outcomes, which caps the achievable accuracy.

To retrain, download the raw trace to `data/borg_traces_data.csv` and run the notebook (`pip install -r notebook/requirements.txt`). To reproduce the evaluation, run `python notebook/evaluate_unseen.py` from the repo root.
