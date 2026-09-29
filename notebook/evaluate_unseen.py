"""Re-evaluate the saved model on test rows whose inputs never appeared in training.

The processed data has ~207k rows but only ~6.5k distinct feature combinations, so the
notebook's random split tests the model mostly on inputs it has already seen. This script
rebuilds that exact split (same data, seed and stratification as predict.ipynb), checks it
reproduces the notebook's confusion matrix, and then reports metrics separately for test
rows whose feature combination is / is not present in the training split.

The model itself is not retrained or modified.

Run from the repo root:  python notebook/evaluate_unseen.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
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


def report(name, mask):
    yt, yp, pr = y_test[mask], pred[mask], prob[mask]
    majority = max(yt.mean(), 1 - yt.mean())
    auc = roc_auc_score(yt, pr) if yt.nunique() == 2 else float("nan")
    print(f"{name:<34} rows={mask.sum():>6}  accuracy={accuracy_score(yt, yp):.3f}  "
          f"F1={f1_score(yt, yp):.3f}  ROC-AUC={auc:.3f}  majority-class baseline={majority:.3f}")


print("Split reproduces the notebook's confusion matrix.\n")
report("All test rows (notebook's number)", np.ones(len(y_test), bool))
report("Inputs seen in training", seen)
report("Inputs NOT seen in training", ~seen)
