import os, json
from pathlib import Path
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from joblib import load

DATABASE_URL = os.getenv("DATABASE_URL"); assert DATABASE_URL
SEED = 42
MODEL_PATH = Path("models/logreg_ckd.joblib")
THRESH_PATH = Path("models/logreg_ckd_threshold.json")

FEATURE_COLS = [
    "age","gender","systolicbp","diastolicbp","serumcreatinine","bunlevels",
    "gfr","acr","serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c","pulsepressure","ureacreatinineratio",
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
proba1 = model.predict_proba(X_val)[:,1]
proba_class0 = 1.0 - proba1
y_val_is0 = (y_val == 0).astype(int)

def at_thr(proba0, y0, t):
    pred0 = (proba0 >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y0, pred0, average="binary", zero_division=0)
    return float(p), float(r), float(f1)

ap = average_precision_score(y_val_is0, proba_class0)
au = roc_auc_score(y_val_is0, proba_class0)
p0,r0,f0 = at_thr(proba_class0, y_val_is0, 0.5)
print(f"Baseline (thr=0.50) PR-AUC={ap:.4f} AUROC={au:.4f} P={p0:.3f} R={r0:.3f} F1={f0:.3f}")

cands = np.unique(np.percentile(proba_class0, np.linspace(1,99,99)))
rows = []
best = {"threshold": 0.5, "precision": p0, "recall": r0, "f1": f0}
for t in cands:
    p,r,f1 = at_thr(proba_class0, y_val_is0, t)
    rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f1})
    if f1 > best["f1"]:
        best = {"threshold": float(t), "precision": p, "recall": r, "f1": f1}

MIN_P = 0.30
filtered = [r for r in rows if r["precision"] >= MIN_P]
pick = max(filtered, key=lambda r: r["f1"]) if filtered else best

print("Best by F1 with precision ≥ 0.30:")
print(f"  thr={pick['threshold']:.3f}  P={pick['precision']:.3f} R={pick['recall']:.3f} F1={pick['f1']:.3f}")

THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(THRESH_PATH, "w") as f:
    json.dump({
        "threshold": pick["threshold"],
        "precision_val_class0": pick["precision"],
        "recall_val_class0": pick["recall"],
        "f1_val_class0": pick["f1"],
        "ap_val_class0": float(ap),
        "auroc_val_class0": float(au),
    }, f, indent=2)

print(f"✅ Saved threshold -> {THRESH_PATH}")
