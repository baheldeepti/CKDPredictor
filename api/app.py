import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from joblib import load

# --------------------------
# Config / constants
# --------------------------
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

# --------------------------
# Models / schemas
# --------------------------
class PatientFeatures(BaseModel):
    age: float
    gender: int
    systolicbp: float
    diastolicbp: float
    serumcreatinine: float
    bunlevels: float
    gfr: float
    acr: float
    serumelectrolytessodium: float
    serumelectrolytespotassium: float
    hemoglobinlevels: float
    hba1c: float
    pulsepressure: float
    ureacreatinineratio: float
    ckdstage: int
    albuminuriacat: int
    bp_risk: int
    hyperkalemiaflag: int
    anemiaflag: int

class PredictResponse(BaseModel):
    prediction: int = Field(description="0 = non-CKD, 1 = CKD")
    prob_ckd: float
    prob_non_ckd: float
    threshold_used: float

class BatchPredictRequest(BaseModel):
    rows: List[PatientFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# --------------------------
# App + lazy-loaded artifacts
# --------------------------
app = FastAPI(title="CKD Predictor API", version="0.1.0")

_model = None
_threshold = None

def get_model_and_threshold():
    global _model, _threshold
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file not found. Train first.")
        _model = load(MODEL_PATH)
    if _threshold is None:
        _threshold = 0.5
        if THRESH_PATH.exists():
            with open(THRESH_PATH) as f:
                _threshold = float(json.load(f)["threshold"])
    return _model, _threshold

def predict_core(df: pd.DataFrame) -> pd.DataFrame:
    # ensure columns/order
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
    df = df[FEATURE_COLS].copy()

    model, thr = get_model_and_threshold()

    # model was trained with flipped labels; proba[:,1] is prob of original class 0 (non-CKD)
    p_non_ckd = model.predict_proba(df)[:, 1]
    p_ckd = 1.0 - p_non_ckd
    pred0 = (p_non_ckd >= thr).astype(int)       # 1 => predict original 0
    pred_orig = np.where(pred0 == 1, 0, 1)       # back to original labels

    out = pd.DataFrame({
        "prediction": pred_orig.astype(int),
        "prob_ckd": p_ckd.astype(float),
        "prob_non_ckd": p_non_ckd.astype(float),
        "threshold_used": thr
    })
    return out

# --------------------------
# Endpoints
# --------------------------
@app.get("/health")
def health():
    try:
        model, thr = get_model_and_threshold()
        return {"status": "ok", "threshold": thr}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(item: PatientFeatures):
    df = pd.DataFrame([item.dict()])
    res = predict_core(df).iloc[0].to_dict()
    return res

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    df = pd.DataFrame([r.dict() for r in req.rows])
    res = predict_core(df).to_dict(orient="records")
    return {"predictions": res}
from fastapi import status

@app.post("/admin/reload", status_code=status.HTTP_204_NO_CONTENT)
def admin_reload():
    """
    Clears the in-memory model cache so next prediction/load
    picks up the freshly written model files.
    """
    global _cache
    _cache.clear()
    return
