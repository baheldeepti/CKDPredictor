import os, json
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from joblib import dump

DATABASE_URL = os.getenv("DATABASE_URL"); assert DATABASE_URL
SEED = 42
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "logreg_ckd.joblib"
META_PATH  = MODEL_DIR / "logreg_ckd_meta.json"

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

# class_weight balanced to help minority
model = LogisticRegression(
    solver="liblinear", max_iter=2000, class_weight="balanced", random_state=SEED
)
print("🧮 Training Logistic Regression…")
model.fit(X_train, y_train)

def eval_block(name, Xs, ys):
    proba1 = model.predict_proba(Xs)[:,1]
    ys_is0 = (ys == 0).astype(int)
    auroc0 = roc_auc_score(ys_is0, 1.0 - proba1)
    ap0    = average_precision_score(ys_is0, 1.0 - proba1)
    preds  = (proba1 >= 0.5).astype(int)
    cm     = confusion_matrix(ys, preds, labels=[0,1])
    print(f"\n=== {name} ===")
    print(f"AUROC (class 0 detection): {auroc0:.4f}")
    print(f"PR-AUC (class 0 detection): {ap0:.4f}")
    print(cm)
    print(classification_report(ys, preds, digits=4))
    return {"auroc_class0": float(auroc0), "prauc_class0": float(ap0), "confusion_matrix": cm.tolist()}

metrics = {
    "train": eval_block("Train", X_train, y_train),
    "val":   eval_block("Validation", X_val, y_val),
    "test":  eval_block("Test", X_test, y_test),
}

dump(model, MODEL_PATH)
with open(META_PATH, "w") as f:
    json.dump({"features": FEATURE_COLS, "metrics": metrics}, f, indent=2)

print(f"\n✅ Saved LogReg model -> {MODEL_PATH}")
print(f"🗂  Saved metadata -> {META_PATH}")
