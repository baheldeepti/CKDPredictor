import os, json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set. Did you `source .env`?"

SEED = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_ckd.joblib"
META_PATH  = MODEL_DIR / "rf_ckd_meta.json"

FEATURE_COLS = [
    "age","gender","systolicbp","diastolicbp",
    "serumcreatinine","bunlevels","gfr","acr",
    "serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c",
    "pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat","bp_risk","hyperkalemiaflag","anemiaflag",
]
TARGET_COL = "diagnosis"   # 1=CKD (majority), 0=non-CKD (minority)

print("🔌 Loading data…")
engine = create_engine(DATABASE_URL)
with engine.begin() as conn:
    df = pd.read_sql(
        text(f"SELECT {', '.join(FEATURE_COLS + [TARGET_COL])} FROM health_metrics"),
        conn,
    )

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Splits -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# Class imbalance: upweight minority (class 0) with class_weight balanced
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=SEED,
    class_weight="balanced",
)

print("🌲 Training RandomForest…")
model.fit(X_train, y_train)

def evaluate(split, Xs, ys):
    proba = model.predict_proba(Xs)[:, 1]      # prob of class 1 (CKD)
    # For minority (class 0) metrics, flip probability:
    proba_class0 = 1.0 - proba
    ys_is_class0 = (ys == 0).astype(int)

    auroc0 = roc_auc_score(ys_is_class0, proba_class0)
    ap0    = average_precision_score(ys_is_class0, proba_class0)
    preds  = (proba >= 0.5).astype(int)        # default threshold for report
    cm     = confusion_matrix(ys, preds, labels=[0,1])
    print(f"\n=== {split} ===")
    print(f"AUROC (class 0 detection): {auroc0:.4f}")
    print(f"PR-AUC (class 0 detection): {ap0:.4f}")
    print("Confusion matrix [rows=true 0,1; cols=pred 0,1]:")
    print(cm)
    print("Class report (original labels):")
    print(classification_report(ys, preds, digits=4))
    return {"auroc_class0": float(auroc0), "prauc_class0": float(ap0), "confusion_matrix": cm.tolist()}

metrics = {
    "train": evaluate("Train", X_train, y_train),
    "val":   evaluate("Validation", X_val, y_val),
    "test":  evaluate("Test", X_test, y_test),
}

dump(model, MODEL_PATH)
with open(META_PATH, "w") as f:
    json.dump({"features": FEATURE_COLS, "metrics": metrics, "random_state": SEED}, f, indent=2)

print(f"\n✅ Saved RF model -> {MODEL_PATH}")
print(f"🗂  Saved metadata -> {META_PATH}")
