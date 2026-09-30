"""Re-evaluate the saved model on test rows whose inputs never appeared in training.

The processed data has ~207k rows but only ~6.5k distinct feature combinations, so the
notebook's random split tests the model mostly on inputs it has already seen. This script
rebuilds that exact split (same data, seed and stratification as predict.ipynb), checks it
reproduces the notebook's confusion matrix, and then reports metrics separately for test
rows whose feature combination is / is not present in the training split.

It also writes:
  models/evaluation.json  - the metrics, served by the API's /model endpoint
  data/demo_holdout.csv   - the held-out test rows, replayed by the dashboard's demo mode

The model itself is not retrained or modified.

Run from the repo root:  python notebook/evaluate_unseen.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
NOTEBOOK_CONFUSION = np.array([[26239, 1274], [3143, 31673]])

df = pd.read_csv(ROOT / "data" / "processed_gct.csv")
X, y = df[FEATURES], df["failed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

model = joblib.load(ROOT / "models" / "failure_model.pkl")
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, pred)
assert (cm == NOTEBOOK_CONFUSION).all(), f"split does not match the notebook:\n{cm}"

train_keys = set(map(tuple, X_train.to_numpy()))
seen = np.array([k in train_keys for k in map(tuple, X_test.to_numpy())])


def metrics(mask):
    yt, yp, pr = y_test[mask], pred[mask], prob[mask]
    return {
        "rows": int(mask.sum()),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp), 4),
        "recall": round(recall_score(yt, yp), 4),
        "f1": round(f1_score(yt, yp), 4),
        "roc_auc": round(roc_auc_score(yt, pr), 4),
        "majority_baseline": round(max(yt.mean(), 1 - yt.mean()), 4),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


subsets = {
    "all_test_rows": metrics(np.ones(len(y_test), bool)),
    "seen_inputs": metrics(seen),
    "unseen_inputs": metrics(~seen),
}

print("Split reproduces the notebook's confusion matrix.\n")
for name, m in subsets.items():
    print(f"{name:<14} rows={m['rows']:>6}  accuracy={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
          f"ROC-AUC={m['roc_auc']:.3f}  majority-class baseline={m['majority_baseline']:.3f}")

evaluation = {
    "dataset": {
        "rows": len(df),
        "distinct_feature_combinations": int(len(X.drop_duplicates())),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "failure_rate": round(float(y.mean()), 4),
    },
    "decision_threshold": 0.5,
    "subsets": subsets,
}
(ROOT / "models" / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")

holdout = X_test.assign(failed=y_test.to_numpy(), seen_in_training=seen.astype(int))
holdout.to_csv(ROOT / "data" / "demo_holdout.csv", index=False, float_format="%.17g")
print(f"\nWrote models/evaluation.json and data/demo_holdout.csv ({len(holdout)} rows)")
