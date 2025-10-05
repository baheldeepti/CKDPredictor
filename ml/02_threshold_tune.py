import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from joblib import load

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set. Did you `source .env`?"

SEED = 42
MODEL_PATH = Path("models/xgb_ckd.joblib")
THRESH_PATH = Path("models/xgb_ckd_threshold.json")

FEATURE_COLS = [
    "age", "gender",
    "systolicbp", "diastolicbp",
    "serumcreatinine", "bunlevels",
    "gfr", "acr",
    "serumelectrolytessodium", "serumelectrolytespotassium",
    "hemoglobinlevels", "hba1c",
    "pulsepressure", "ureacreatinineratio",
    "ckdstage", "albuminuriacat",
    "bp_risk", "hyperkalemiaflag", "anemiaflag",
]
TARGET_COL = "diagnosis"  # original labels: 1 = CKD (majority), 0 = non-CKD (minority)

# ------------------------------------------------------------
# Load data (same split as training)
# ------------------------------------------------------------
print("🔌 Loading data from Postgres...")
engine = create_engine(DATABASE_URL)
with engine.begin() as conn:
    df = pd.read_sql(
        text(f"SELECT {', '.join(FEATURE_COLS + [TARGET_COL])} FROM health_metrics"),
        conn,
    )

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

# Same splits as training script
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
print("📦 Loading model:", MODEL_PATH)
model = load(MODEL_PATH)

# We trained with flipped labels (1 == original class 0).
# So proba for flipped positive ([:,1]) is the probability of original class 0.
proba_val_for_class0 = model.predict_proba(X_val)[:, 1]
y_val_is_class0 = (y_val == 0).astype(int)

# Reference metrics with default threshold 0.5
def metrics_at_threshold(proba, y_true_class0, thr):
    preds_class0 = (proba >= thr).astype(int)  # 1 predicts original class 0
    # back to original label space for F1 calculation?
    # We’ll compute precision/recall/F1 for class0 directly:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_class0, preds_class0, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)

ap_val = average_precision_score(y_val_is_class0, proba_val_for_class0)
auroc_val = roc_auc_score(y_val_is_class0, proba_val_for_class0)
p0, r0, f0 = metrics_at_threshold(proba_val_for_class0, y_val_is_class0, 0.5)

print(f"\nBaseline (thr=0.50) — class 0 detection")
print(f"  PR-AUC: {ap_val:.4f} | AUROC: {auroc_val:.4f}")
print(f"  Precision: {p0:.4f} | Recall: {r0:.4f} | F1: {f0:.4f}")

# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------
candidates = np.unique(np.percentile(proba_val_for_class0, np.linspace(1, 99, 99)))
rows = []
best = {"thr": 0.5, "f1": f0, "p": p0, "r": r0}

for thr in candidates:
    p, r, f1 = metrics_at_threshold(proba_val_for_class0, y_val_is_class0, thr)
    rows.append({"threshold": float(thr), "precision": p, "recall": r, "f1": f1})
    # You can bias for recall here if desired; we’ll pick max F1 for class 0 by default
    if f1 > best["f1"]:
        best = {"thr": float(thr), "f1": f1, "p": p, "r": r}

# Optionally, enforce a minimum precision for class 0 (to avoid silly thresholds)
MIN_PRECISION = 0.30
filtered = [r for r in rows if r["precision"] >= MIN_PRECISION]
if filtered:
    best_prec_floor = max(filtered, key=lambda r: r["f1"])
else:
    best_prec_floor = best

print("\nTop 10 thresholds by F1 (no precision floor):")
for r in sorted(rows, key=lambda r: r["f1"], reverse=True)[:10]:
    print(f"  thr={r['threshold']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

print("\nBest by F1 with precision ≥ 0.30:")
print(f"  thr={best_prec_floor['threshold']:.3f}  P={best_prec_floor['precision']:.3f} R={best_prec_floor['recall']:.3f} F1={best_prec_floor['f1']:.3f}")

# Save chosen threshold (use the precision-floored pick)
THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(THRESH_PATH, "w") as f:
    json.dump(
        {
            "threshold": best_prec_floor["threshold"],
            "precision_val_class0": best_prec_floor["precision"],
            "recall_val_class0": best_prec_floor["recall"],
            "f1_val_class0": best_prec_floor["f1"],
            "ap_val_class0": float(ap_val),
            "auroc_val_class0": float(auroc_val),
        },
        f,
        indent=2,
    )

print(f"\n✅ Saved threshold -> {THRESH_PATH}")
