import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from joblib import load

# ----------------------------------
# Model registry (paths per model)
# ----------------------------------
REGISTRY: Dict[str, Dict[str, Path]] = {
    "xgb": {
        "model": Path("models/xgb_ckd.joblib"),
        "thr":   Path("models/xgb_ckd_threshold.json"),
    },
    "rf": {
        "model": Path("models/rf_ckd.joblib"),
        "thr":   Path("models/rf_ckd_threshold.json"),
    },
    "logreg": {
        "model": Path("models/logreg_ckd.joblib"),
        "thr":   Path("models/logreg_ckd_threshold.json"),
    },
}

FEATURE_COLS = [
    "age","gender",
    "systolicbp","diastolicbp",
    "serumcreatinine","bunlevels",
    "gfr","acr",
    "serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c",
    "pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat",
    "bp_risk","hyperkalemiaflag","anemiaflag",
]

class PatientFeatures(BaseModel):
    age: float; gender: int
    systolicbp: float; diastolicbp: float
    serumcreatinine: float; bunlevels: float
    gfr: float; acr: float
    serumelectrolytessodium: float; serumelectrolytespotassium: float
    hemoglobinlevels: float; hba1c: float
    pulsepressure: float; ureacreatinineratio: float
    ckdstage: int; albuminuriacat: int
    bp_risk: int; hyperkalemiaflag: int; anemiaflag: int

class PredictResponse(BaseModel):
    prediction: int = Field(description="0 = non-CKD, 1 = CKD")
    prob_ckd: float
    prob_non_ckd: float
    threshold_used: float
    model_used: str

class BatchPredictRequest(BaseModel):
    rows: List[PatientFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

app = FastAPI(title="CKD Predictor API", version="0.2.0")

_cache: Dict[str, Dict[str, object]] = {}  # {model: {"model": sklearn, "thr": float}}

def get_model_and_thr(model_key: str):
    model_key = model_key.lower()
    if model_key not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'. Use one of {list(REGISTRY)}")
    if model_key not in _cache:
        paths = REGISTRY[model_key]
        if not paths["model"].exists():
            raise HTTPException(status_code=400, detail=f"Model file missing for '{model_key}'. Train it first.")
        mdl = load(paths["model"])
        thr = 0.5
        if paths["thr"].exists():
            with open(paths["thr"]) as f:
                thr = float(json.load(f)["threshold"])
        _cache[model_key] = {"model": mdl, "thr": thr}
    return _cache[model_key]["model"], _cache[model_key]["thr"]

def predict_core(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
    df = df[FEATURE_COLS].copy()
    mdl, thr = get_model_and_thr(model_key)

    # Convention:
    # - XGB was trained with flipped labels => proba[:,1] = P(original class 0)
    # - RF / LogReg trained normal => proba[:,1] = P(class 1 = CKD)
    if model_key == "xgb":
        p_non_ckd = mdl.predict_proba(df)[:, 1]
        p_ckd = 1.0 - p_non_ckd
        pred0 = (p_non_ckd >= thr).astype(int)
        pred_orig = np.where(pred0 == 1, 0, 1)
    else:
        p_ckd = mdl.predict_proba(df)[:, 1]
        p_non_ckd = 1.0 - p_ckd
        pred_ckd = (p_ckd >= thr).astype(int)
        pred_orig = pred_ckd

    out = pd.DataFrame({
        "prediction": pred_orig.astype(int),
        "prob_ckd": p_ckd.astype(float),
        "prob_non_ckd": p_non_ckd.astype(float),
        "threshold_used": thr,
        "model_used": model_key,
    })
    return out

@app.get("/health")
def health(model: str = Query("xgb", description="xgb | rf | logreg")):
    try:
        _, thr = get_model_and_thr(model)
        return {"status": "ok", "model": model, "threshold": thr}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(item: PatientFeatures, model: str = Query("xgb")):
    df = pd.DataFrame([item.dict()])
    res = predict_core(df, model).iloc[0].to_dict()
    return res

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest, model: str = Query("xgb")):
    df = pd.DataFrame([r.dict() for r in req.rows])
    res = predict_core(df, model).to_dict(orient="records")
    return {"predictions": res}
