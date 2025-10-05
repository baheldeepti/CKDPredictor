import os
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from joblib import load

# --------------------------
# Config
# --------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
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

def load_model_and_threshold():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    model = load(MODEL_PATH)
    thr = 0.5
    if THRESH_PATH.exists():
        with open(THRESH_PATH) as f:
            thr = json.load(f)["threshold"]
    return model, float(thr)

def proba_class0(model, X: pd.DataFrame) -> np.ndarray:
    """
    We trained with flipped labels, where model's proba[:,1] corresponds to original class 0 (non-CKD).
    """
    return model.predict_proba(X)[:, 1]

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    model, thr = load_model_and_threshold()
    X = df[FEATURE_COLS].copy()

    p0 = proba_class0(model, X)                  # prob of original class 0 (non-CKD)
    pred0 = (p0 >= thr).astype(int)              # 1 => predict original 0 (non-CKD)
    pred_orig = np.where(pred0 == 1, 0, 1)       # back to original labels (0/1)
    p1 = 1.0 - p0                                # prob of class 1 (CKD)

    out = df.copy()
    out["prob_non_ckd"] = p0
    out["prob_ckd"] = p1
    out["prediction"] = pred_orig
    out["threshold_used"] = thr
    return out

def predict_one_rowdict(d: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    df = pd.DataFrame([d])
    res = predict_df(df).iloc[0]
    return {
        "prediction": int(res["prediction"]),
        "prob_ckd": float(res["prob_ckd"]),
        "prob_non_ckd": float(res["prob_non_ckd"]),
        "threshold_used": float(res["threshold_used"]),
    }

def fetch_first_n_from_db(n=5) -> pd.DataFrame:
    assert DATABASE_URL, "DATABASE_URL not set (did you source .env?)"
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        df = pd.read_sql(
            text(f"""
                SELECT {", ".join(FEATURE_COLS + ["diagnosis", "patient_id"])}
                FROM health_metrics
                ORDER BY metric_id
                LIMIT :n
            """),
            conn,
            params={"n": n},
        )
    return df

if __name__ == "__main__":
    # Demo: pull 5 rows from DB and predict
    df = fetch_first_n_from_db(5)
    preds = predict_df(df)
    cols = ["patient_id", "diagnosis", "prob_non_ckd", "prob_ckd", "prediction", "threshold_used"]
    print(preds[cols].to_string(index=False))
