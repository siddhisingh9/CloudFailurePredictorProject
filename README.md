# 🚀 Task Failure Prediction using Google Cloud Trace Data  

This project predicts **task failures** using **Google Cloud Trace data**, helping optimize resource usage by identifying high-risk tasks before execution. It combines **machine learning**, **API deployment**, **caching**, and **real-time dashboards** into an end-to-end cloud-based solution.  

---

## 📌 Features  

- **Machine Learning Model**  
  - Preprocessed the [Google Borg cluster trace (2019)](https://github.com/google/cluster-data) with feature engineering and leakage removal  
  - Random Forest classifier: **83% accuracy (0.87 F1, 0.90 ROC-AUC) on jobs with resource requests never seen in training**, against a 63% majority-class baseline, and 93% on a standard random split (see [Model](#-model))  

- **Deployment**  
  - Model **dockerized** and served via **FastAPI**  
  - REST API with input validation, a health check and live cache statistics  

- **Redis Caching**  
  - Predictions are cached in **Redis**, keyed by the model version and the job's inputs  
  - The trace has only ~6.5k distinct inputs across 207k jobs, so repeats are common: replaying the dataset gives a 97% hit rate, and locally a cache hit takes ~1 ms versus ~30 ms for the model  

- **Real-time Streaming**  
  - Every prediction is published to a **Redis Pub/Sub** channel  
  - The dashboard subscribes to it and shows a live feed of every user's predictions  

- **Interactive Dashboard**  
  - Built with **Streamlit**  
  - Supports CSV uploads or demo mode with preloaded data  
  - Streams a row to the API every 3 seconds and shows the predicted vs. actual outcome  

- **Visualizations**  
  - Task failure probabilities  
  - CPU and memory usage  
  - Historical failure trends  
  - Cache hit rate and latency  

- **Cloud Hosting**  
  - End-to-end system deployed on **Render**, running entirely on free tiers  

---

## ⚙️ Tech Stack  

- **Machine Learning**: scikit-learn, pandas  
- **API**: FastAPI, Docker  
- **Dashboard**: Streamlit  
- **Caching & Messaging**: Redis (cache + Pub/Sub)  
- **Hosting**: Render  
- **Testing**: pytest, fakeredis  

---

## 🏗️ Architecture

```
                       POST /predict                          GET / SET (cache)
 dashboard (Streamlit) ─────────────▶ api (FastAPI + model) ───────────────────▶ Redis
        ▲                                                       PUBLISH             │
        └──────────────────── SUBSCRIBE "predictions" (live feed) ◀─────────────────┘
```

- **api** (`app/`) serves the model. With `REDIS_URL` set, it checks the Redis cache before running the model, stores new predictions with a 24-hour expiry, and publishes every prediction to the `predictions` channel. Cache keys include a hash of the model file, so retraining automatically invalidates old entries.
- **dashboard** (`dashboard/`) replays rows from the bundled trace sample or from a CSV you upload, sends them to the API, and charts the results. A background thread subscribes to `predictions` for the live feed.
- **notebook** (`notebook/`) cleans the raw trace into `data/processed_gct.csv`, trains `models/failure_model.pkl`, and re-evaluates it on unseen inputs.

Redis is optional. Without it the API serves uncached predictions, and a Redis outage never fails a request.

---

## 🐳 Running with Docker

```sh
docker compose up --build
```

Then open http://localhost:8501. The API's interactive docs are at http://localhost:8000/docs.

## 💻 Running locally

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
| `DEMO_DATA_PATH` | dashboard | `data/processed_gct.csv` | Rows replayed in demo mode. |
| `PORT` | both Docker images | `8000` / `8501` | Port to listen on; set automatically by Render. |

---

## ☁️ Deploying on free tiers

Everything fits within free plans: the API uses about 210 MB of RAM (Render's free limit is 512 MB) and the cache uses well under 1 MB of Redis.

1. **Redis:** create a free database on [Redis Cloud](https://redis.io/cloud/) (30 MB) and copy its connection URL (`redis://default:<password>@<host>:<port>`). Pick the region closest to your Render services; every cache lookup is a network round trip, so a far-away Redis can be slower than the model. The dashboard's latency figures show whether the cache is paying off.
2. **API on Render:** New → Web Service → this repo, runtime **Docker**, Dockerfile path `app/Dockerfile`, build context `.` (repo root), instance type **Free**. Set `REDIS_URL`. Health check path: `/health`.
3. **Dashboard**, either:
   - **on Render:** another Docker web service with Dockerfile path `dashboard/Dockerfile`, build context `.`, and environment variables `API_URL` (the API's public URL, e.g. `https://<api-name>.onrender.com`) and `REDIS_URL`; or
   - **on [Streamlit Community Cloud](https://streamlit.io/cloud):** New app → this repo, main file `dashboard/dashboard.py`. It installs `dashboard/requirements.txt` automatically. Under Advanced settings → Secrets, add `API_URL = "https://<api-name>.onrender.com"` and `REDIS_URL = "redis://..."`; top-level secrets are exposed to the app as environment variables.

**Free-tier behaviour to expect:** Render's free services sleep after 15 minutes without traffic and take 30–60 seconds to wake. The dashboard pings the API as soon as it opens and shows a "waking up" message instead of failing while it waits. The cache statistics are kept in memory, so they reset whenever the API restarts or sleeps; cached predictions in Redis survive.

---

## 🔌 API

`POST /predict`

```json
{"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}
```

returns `{"failure_probability": 0.97, "cached": false}`.

`cpu_request` and `memory_request` are normalised to the largest machine in the trace (0–1). `priority` is a non-negative integer (0–450 in the trace) and `scheduling_class` is 0–3. Invalid input gets a 422.

`GET /health` returns `{"status": "ok", "redis": "ok" | "disabled" | "unreachable", "model_version": "058e4618e0ff"}`.

`GET /stats` returns the cache hit rate and average latency for cache hits and misses since the API started.

Each message on the `predictions` channel is JSON: `{"data": {...inputs}, "failure_probability": 0.97, "cached": false, "published_at": "2026-09-29T12:00:00+00:00"}`.

---

## 🧠 Model

- **Features:** `cpu_request`, `memory_request`, `priority`, `scheduling_class`.
- **Label:** a job counts as *failed* if its final event is `FAIL`, `EVICT`, `LOST` or `KILL`, and as *not failed* if it is `FINISH`.
- **Training:** the random-forest hyperparameters were chosen by grid search, with SMOTE oversampling on the training split.
- **Versions:** the model was pickled with scikit-learn 1.7.1 and NumPy 2.x. Keep the versions pinned in `app/requirements.txt` or the pickle may not load.

### Evaluation

The 207,762 processed rows contain only about 6,500 distinct feature combinations, so the notebook's random 70/30 split puts identical inputs in both the training and test sets. The score on that split mostly measures inputs the model has already seen. `notebook/evaluate_unseen.py` rebuilds the exact split (it reproduces the notebook's confusion matrix) and scores the saved model separately on test rows whose inputs never appear in training:

| Test rows | Rows | Accuracy | F1 | ROC-AUC | Majority-class baseline |
|---|---|---|---|---|---|
| All (random split) | 62,329 | 0.929 | 0.935 | 0.977 | 0.559 |
| Inputs seen in training | 61,342 | 0.931 | 0.936 | 0.978 | 0.557 |
| **Inputs not seen in training** | **987** | **0.834** | **0.867** | **0.902** | **0.632** |

The last row is the better estimate of performance on new kinds of jobs. It is based on 987 rows, so it carries roughly ±2 percentage points of uncertainty. In 825 of the feature combinations the same inputs appear with both outcomes, which caps the achievable accuracy.

To retrain, download the raw trace to `data/borg_traces_data.csv` and run the notebook (`pip install -r notebook/requirements.txt`). To reproduce the evaluation, run `python notebook/evaluate_unseen.py` from the repo root.
