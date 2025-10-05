import os, json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from joblib import dump

# -------- Config --------
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set (expected in environment)"

SEED = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "age","gender","systolicbp","diastolicbp",
    "serumcreatinine","bunlevels","gfr","acr",
    "serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c",
    "pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat",
    "bp_risk","hyperkalemiaflag","anemiaflag",
]
TARGET = "diagnosis"

# Model registry definition
MODELS = {
    "xgb": {
        "path": MODEL_DIR / "xgb_ckd.joblib",
        "thr_path": MODEL_DIR / "xgb_ckd_threshold.json",
        "train": lambda: XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            # IMPORTANT: we will flip mapping at inference (keep standard here)
            # XGB proba[:,1] = P(class 1); we’ll compute class0 proba as (1-proba1)
        )
    },
    "rf": {
        "path": MODEL_DIR / "rf_ckd.joblib",
        "thr_path": MODEL_DIR / "rf_ckd_threshold.json",
        "train": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    },
    "logreg": {
        "path": MODEL_DIR / "logreg_ckd.joblib",
        "thr_path": MODEL_DIR / "logreg_ckd_threshold.json",
        "train": lambda: LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )
    },
}

def fetch_training_frame(engine):
    """
    Combine historical table (health_metrics) + new user inputs.
    health_metrics already contains TARGET (Diagnosis as 'diagnosis').
    user_inputs has only features -> no target; treat as unlabeled (exclude from supervised training).
    """
    with engine.begin() as conn:
        # Labeled historical data
        cols = ", ".join(FEATURE_COLS + [TARGET])
        hist = pd.read_sql(text(f"SELECT {cols} FROM health_metrics"), conn)

        # New unlabeled inputs (for now not used in supervised training unless you have labels)
        ui_cols = ", ".join(FEATURE_COLS)
        ui = pd.read_sql(text(f"SELECT {ui_cols} FROM user_inputs"), conn)

    # For supervised training we use only labeled rows (hist).
    # (If later you have labels for user_inputs, you can join and include.)
    df = hist.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    return df, ui

def tune_threshold_for_class0(proba_class0, y_is_class0, min_precision=0.30):
    # Try percentiles as candidate thresholds
    cands = np.unique(np.percentile(proba_class0, np.linspace(1, 99, 99)))
    def at_thr(t):
        pred0 = (proba_class0 >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_is_class0, pred0, average="binary", zero_division=0)
        return float(p), float(r), float(f1)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    filtered = []
    for t in cands:
        p, r, f1 = at_thr(t)
        row = {"threshold": float(t), "precision": p, "recall": r, "f1": f1}
        filtered.append(row)
        if p >= min_precision and f1 > best["f1"]:
            best = row
    # If none meet min precision, pick max F1 overall
    if best["f1"] == 0.0:
        best = max(filtered, key=lambda r: r["f1"]) if filtered else best
    return best

def train_and_tune(X_train, y_train, X_val, y_val, key, spec):
    model = spec["train"]()
    model.fit(X_train, y_train)

    # Compute probabilities of class 0 for threshold tuning
    proba1_val = model.predict_proba(X_val)[:, 1]  # P(class 1 = CKD)
    proba0_val = 1.0 - proba1_val                   # P(class 0 = non-CKD)
    y_val_is0 = (y_val == 0).astype(int)

    ap = average_precision_score(y_val_is0, proba0_val)
    au = roc_auc_score(y_val_is0, proba0_val)
    best = tune_threshold_for_class0(proba0_val, y_val_is0, min_precision=0.30)

    # Save model + threshold
    dump(model, spec["path"])
    with open(spec["thr_path"], "w") as f:
        json.dump({
            "threshold": best["threshold"],
            "precision_val_class0": best["precision"],
            "recall_val_class0": best["recall"],
            "f1_val_class0": best["f1"],
            "ap_val_class0": float(ap),
            "auroc_val_class0": float(au),
        }, f, indent=2)

    return {
        "threshold": best["threshold"],
        "ap_class0": float(ap),
        "auroc_class0": float(au),
    }

def main():
    print("🔌 Connecting to database…")
    engine = create_engine(DATABASE_URL)

    print("📦 Loading training data…")
    df, ui = fetch_training_frame(engine)
    print(f"  Labeled rows: {len(df)} | New unlabeled user inputs (not used in training): {len(ui)}")

    X = df[FEATURE_COLS]
    y = df[TARGET].astype(int)

    # Standard split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    report = {}
    for key, spec in MODELS.items():
        print(f"\n🛠 Training {key}…")
        summary = train_and_tune(X_train, y_train, X_val, y_val, key, spec)
        report[key] = summary
        print(f"   -> threshold={summary['threshold']:.4f}  AP0={summary['ap_class0']:.4f}  AUROC0={summary['auroc_class0']:.4f}")

    # Quick test-set sanity for XGB (one example)
    from joblib import load
    xgb = load(MODELS["xgb"]["path"])
    p1 = xgb.predict_proba(X_test)[:,1]
    preds = (p1 >= (report["xgb"]["threshold"])).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0,1]).tolist()
    print("\n✅ Retrain complete. XGB test-set confusion matrix [rows=true 0,1; cols=pred 0,1]:")
    print(cm)

    # Write an overall retrain report
    with open(MODEL_DIR / "retrain_report.json", "w") as f:
        json.dump({
            "models": report,
            "rows_labeled": int(len(df)),
            "rows_unlabeled_user_inputs": int(len(ui))
        }, f, indent=2)

    print("\n📄 Wrote models/* and retrain_report.json")

if __name__ == "__main__":
    main()
