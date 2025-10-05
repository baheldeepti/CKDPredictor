import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from joblib import load

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set."

SEED = 42
MODEL_PATH = Path("models/rf_ckd.joblib")
THRESH_PATH = Path("models/rf_ckd_threshold.json")

FEATURE_COLS = [
    "age","gender","systolicbp","diastolicbp",
    "serumcreatinine","bunlevels","gfr","acr",
    "serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c",
    "pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat","bp_risk","hyperkalemiaflag","anemiaflag",
]
TARGET_COL = "diagnosis"

engine = create_engine(DATABASE_URL)
with engine.begin() as conn:
    df = pd.read_sql(text(f"SELECT {', '.join(FEATURE_COLS + [TARGET_COL])} FROM health_metrics"), conn)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
X = df[FEATURE_COLS]; y = df[TARGET_COL].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

model = load(MODEL_PATH)

proba_val_ckd   = model.predict_proba(X_val)[:,1]
proba_val_cls0  = 1.0 - proba_val_ckd
y_val_is_cls0   = (y_val == 0).astype(int)

def metrics_at_thr(proba_cls0, y_true_cls0, thr):
    pred_cls0 = (proba_cls0 >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true_cls0, pred_cls0, average="binary", zero_division=0)
    return float(p), float(r), float(f1)

ap_val  = average_precision_score(y_val_is_cls0, proba_val_cls0)
auroc_v = roc_auc_score(y_val_is_cls0, proba_val_cls0)
p0, r0, f0 = metrics_at_thr(proba_val_cls0, y_val_is_cls0, 0.5)

print(f"\nBaseline (thr=0.50) — class 0 detection")
print(f"  PR-AUC: {ap_val:.4f} | AUROC: {auroc_v:.4f}")
print(f"  Precision: {p0:.4f} | Recall: {r0:.4f} | F1: {f0:.4f}")

cands = np.unique(np.percentile(proba_val_cls0, np.linspace(1, 99, 99)))
rows = []
best = {"threshold": 0.5, "f1": f0, "precision": p0, "recall": r0}
for thr in cands:
    p,r,f1 = metrics_at_thr(proba_val_cls0, y_val_is_cls0, thr)
    rows.append({"threshold": float(thr), "precision": p, "recall": r, "f1": f1})
    if f1 > best["f1"]:
        best = {"threshold": float(thr), "precision": p, "recall": r, "f1": f1}

MIN_P = 0.30
filtered = [r for r in rows if r["precision"] >= MIN_P]
pick = max(filtered, key=lambda r: r["f1"]) if filtered else best

print("\nBest by F1 with precision ≥ 0.30:")
print(f"  thr={pick['threshold']:.3f}  P={pick['precision']:.3f} R={pick['recall']:.3f} F1={pick['f1']:.3f}")

THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(THRESH_PATH, "w") as f:
    json.dump({
        "threshold": pick["threshold"],
        "precision_val_class0": pick["precision"],
        "recall_val_class0": pick["recall"],
        "f1_val_class0": pick["f1"],
        "ap_val_class0": float(ap_val),
        "auroc_val_class0": float(auroc_v),
    }, f, indent=2)

print(f"\n✅ Saved threshold -> {THRESH_PATH}")
