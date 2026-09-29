# CloudFailurePredictor

Predicts whether a job in a compute cluster will fail, from the resources it requests, and streams those predictions to a live dashboard. The model is a random forest trained on the [Google Borg cluster trace (2019)](https://github.com/google/cluster-data).

```
 dashboard (Streamlit)  ──POST /predict──▶  api (FastAPI + model)  ──PUBLISH predictions──▶  Redis (optional)
      :8501                                     :8000
```

- **api** (`app/`) serves the model. Every prediction is also published to the Redis channel `predictions` when `REDIS_URL` is set, so other services can subscribe to the stream.
- **dashboard** (`dashboard/`) replays rows, either from the bundled trace sample or from a CSV you upload, sends each one to the API every 3 seconds, and charts the results.
- **notebook** (`notebook/predict.ipynb`) cleans the raw trace into `data/processed_gct.csv` and trains `models/failure_model.pkl`.

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

Run the tests with `pytest`.

### Configuration

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `REDIS_URL` | api | unset | Redis to publish predictions to. If unset, publishing is skipped. |
| `MODEL_PATH` | api | `models/failure_model.pkl` | Path to the trained model. |
| `API_URL` | dashboard | `http://localhost:8000` | Base URL of the API. |
| `DEMO_DATA_PATH` | dashboard | `data/processed_gct.csv` | Rows replayed in demo mode. |

## API

`POST /predict`

```json
{"cpu_request": 0.0072, "memory_request": 0.0013, "priority": 360, "scheduling_class": 2}
```

returns `{"failure_probability": 0.97}`.

`cpu_request` and `memory_request` are normalised to the largest machine in the trace (0–1). `priority` is a non-negative integer (0–450 in the trace) and `scheduling_class` is 0–3. Invalid input gets a 422.

`GET /health` returns `{"status": "ok", "redis": "ok" | "disabled" | "unreachable"}`.

## Model

- **Features:** `cpu_request`, `memory_request`, `priority`, `scheduling_class`.
- **Label:** a job counts as *failed* if its final event is `FAIL`, `EVICT`, `LOST` or `KILL`, and as *not failed* if it is `FINISH`.
- **Training:** the random-forest hyperparameters were chosen by grid search, with SMOTE oversampling on the training split.
- **Reported score:** 93% accuracy and 0.93 F1 on a random 30% test split.
- **Caveat on that score:** the 207,762 processed rows contain only about 6,500 distinct feature combinations, so a random split puts identical inputs in both the training and test sets. The score is therefore optimistic, and performance on jobs with unseen resource requests is likely lower. In 825 of those combinations the same inputs have both outcomes, which puts a ceiling on achievable accuracy.
- **Versions:** the model was pickled with scikit-learn 1.7.1 and NumPy 2.x. Keep the versions pinned in `app/requirements.txt` or the pickle may not load.

To retrain, download the raw trace to `data/borg_traces_data.csv` and run the notebook (`pip install -r notebook/requirements.txt`).
