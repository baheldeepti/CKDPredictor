import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from joblib import dump

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set. Did you `source .env` in this shell?"

SEED = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_ckd.joblib"
META_PATH = MODEL_DIR / "xgb_ckd_meta.json"

# Columns to pull from DB (all lowercase to match table)
FEATURE_COLS = [
    "age", "gender",
    "systolicbp", "diastolicbp",
    "serumcreatinine", "bunlevels",
    "gfr", "acr",
    "serumelectrolytessodium", "serumelectrolytespotassium",
    "hemoglobinlevels", "hba1c",
    # derived
    "pulsepressure", "ureacreatinineratio",
    "ckdstage", "albuminuriacat",
    "bp_risk", "hyperkalemiaflag", "anemiaflag",
]
TARGET_COL = "diagnosis"  # 1 = CKD (majority), 0 = non-CKD (minority)

# ---------------------------------------------------------------------
# Load data from Postgres
# ---------------------------------------------------------------------
print("🔌 Connecting to database and loading data...")
engine = create_engine(DATABASE_URL)
with engine.begin() as conn:
    df = pd.read_sql(
        text(f"""
            SELECT {", ".join(FEATURE_COLS + [TARGET_COL])}
            FROM health_metrics
        """),
        conn,
    )

print(f"Loaded {len(df)} rows with {len(FEATURE_COLS)} features.")

# Basic sanity: drop rows with any missing in selected columns (dataset looks clean)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

# Train/val/test split (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# ---------------------------------------------------------------------
# Class imbalance handling
# Majority class = 1 (CKD), Minority = 0 (non-CKD)
# scale_pos_weight in XGB applies to the "positive class" label=1.
# We want to give more weight to minority (label 0), so we flip labels for training
# and then interpret metrics accordingly.
# ---------------------------------------------------------------------
pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())  # minority / majority in flipped space
print(f"Computed scale_pos_weight for flipped training: {pos_weight:.3f}")

# Flip labels: new_y = 1 if original was 0, else 0
y_train_flip = (y_train == 0).astype(int)
y_val_flip   = (y_val   == 0).astype(int)
y_test_flip  = (y_test  == 0).astype(int)

# ---------------------------------------------------------------------
# Train XGBoost (light tuning; good defaults for tabular)
# ---------------------------------------------------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=pos_weight,  # emphasizes original class 0 via flipped labels
)

print("🚀 Training XGBoost...")
model.fit(
    X_train,
    y_train_flip,
    eval_set=[(X_val, y_val_flip)],
    verbose=False,
)

# ---------------------------------------------------------------------
# Evaluation (focus on original minority = class 0)
# We'll convert predicted probabilities for "flipped class 1" back to "original class 0"
# ---------------------------------------------------------------------
def evaluate(split_name, Xs, y_true_original):
    y_true_flip = (y_true_original == 0).astype(int)
    proba_flip = model.predict_proba(Xs)[:, 1]  # prob of flipped positive => original class 0
    preds_flip = (proba_flip >= 0.5).astype(int)

    # Metrics for detecting non-CKD (original class 0)
    # AUROC / PR-AUC computed against y_true_flip (1 = original 0)
    auroc = roc_auc_score(y_true_flip, proba_flip)
    ap = average_precision_score(y_true_flip, proba_flip)
    prec, rec, thr = precision_recall_curve(y_true_flip, proba_flip)

    # Confusion matrix back in original label space (0/1)
    preds_original = (preds_flip == 1).astype(int) * 0 + (preds_flip == 0).astype(int) * 1
    # ^ If flipped pred=1 => original class 0; else class 1
    cm = confusion_matrix(y_true_original, preds_original, labels=[0,1])

    print(f"\n=== {split_name} ===")
    print(f"AUROC (for class 0 detection): {auroc:.4f}")
    print(f"PR-AUC (for class 0 detection): {ap:.4f}")
    print("Confusion matrix [rows=true 0,1; cols=pred 0,1]:")
    print(cm)
    print("Class report (original labels):")
    print(classification_report(y_true_original, preds_original, digits=4))

    return {
        "auroc_class0": float(auroc),
        "prauc_class0": float(ap),
        "confusion_matrix": cm.tolist(),
    }

metrics_train = evaluate("Train", X_train, y_train)
metrics_val   = evaluate("Validation", X_val, y_val)
metrics_test  = evaluate("Test", X_test, y_test)

# ---------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------
dump(model, MODEL_PATH)
meta = {
    "features": FEATURE_COLS,
    "random_state": SEED,
    "scale_pos_weight_flipped": float(pos_weight),
    "metrics": {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
    },
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Saved model -> {MODEL_PATH}")
print(f"🗂  Saved metadata -> {META_PATH}")
