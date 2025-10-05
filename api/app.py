# api/app.py
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from joblib import load

# Optional DB logging
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# If you're still serving the *older* XGBoost model you trained before the
# unified retrain script, set this True (that model's proba[:,1] was P(non-CKD)).
# After switching to models produced by ml/99_retrain.py, set this to False
# (standard proba[:,1] = P(CKD)).
LEGACY_XGB_FLIPPED = True

REGISTRY: Dict[str, Dict[str, object]] = {
    "xgb": {
        "model": Path("models/xgb_ckd.joblib"),
        "thr":   Path("models/xgb_ckd_threshold.json"),
        "flip_probas": LEGACY_XGB_FLIPPED,  # compatibility switch for legacy XGB
    },
    "rf": {
        "model": Path("models/rf_ckd.joblib"),
        "thr":   Path("models/rf_ckd_threshold.json"),
        "flip_probas": False,
    },
    "logreg": {
        "model": Path("models/logreg_ckd.joblib"),
        "thr":   Path("models/logreg_ckd_threshold.json"),
        "flip_probas": False,
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

# Database (optional; used for logging)
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

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
    prediction: int = Field(description="0 = Non-CKD, 1 = CKD")
    prob_ckd: float
    prob_non_ckd: float
    threshold_used: float
    model_used: str

class BatchPredictRequest(BaseModel):
    rows: List[PatientFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# -----------------------------------------------------------------------------
# App + Model cache
# -----------------------------------------------------------------------------

app = FastAPI(title="CKD Predictor API", version="0.4.0")
_cache: Dict[str, Dict[str, object]] = {}  # {model: {"model": sklearn_obj, "thr": float, "flip": bool}}

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
        _cache[model_key] = {"model": mdl, "thr": thr, "flip": bool(paths.get("flip_probas", False))}
    return _cache[model_key]["model"], _cache[model_key]["thr"], _cache[model_key]["flip"]

def predict_core(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    # validate columns & order
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
    df = df[FEATURE_COLS].copy()

    mdl, thr, flip = get_model_and_thr(model_key)

    # Probabilities
    proba1 = mdl.predict_proba(df)[:, 1]  # standard: P(CKD)
    if flip:
        # Legacy XGB path: proba1 is actually P(Non-CKD)
        prob_non_ckd = proba1
        prob_ckd = 1.0 - proba1
        # Threshold tuned on P(non-CKD) -> class 0 if prob_non_ckd >= thr
        pred_class0 = (prob_non_ckd >= thr).astype(int)  # 1 means "Non-CKD"
        prediction = np.where(pred_class0 == 1, 0, 1)    # convert to original labels
    else:
        # Standard path: proba1 is P(CKD)
        prob_ckd = proba1
        prob_non_ckd = 1.0 - proba1
        prediction = (prob_ckd >= thr).astype(int)       # 1 means CKD

    out = pd.DataFrame({
        "prediction": prediction.astype(int),
        "prob_ckd": prob_ckd.astype(float),
        "prob_non_ckd": prob_non_ckd.astype(float),
        "threshold_used": float(thr),
        "model_used": model_key,
    })
    return out

# -----------------------------------------------------------------------------
# DB logging helpers (no-op if DATABASE_URL is not set)
# -----------------------------------------------------------------------------

def _save_user_input(conn, row: dict) -> int:
    q = text("""
        INSERT INTO user_inputs (
            age, gender, systolicbp, diastolicbp,
            serumcreatinine, bunlevels, gfr, acr,
            serumelectrolytessodium, serumelectrolytespotassium,
            hemoglobinlevels, hba1c, pulsepressure, ureacreatinineratio,
            ckdstage, albuminuriacat, bp_risk, hyperkalemiaflag, anemiaflag
        ) VALUES (
            :age, :gender, :systolicbp, :diastolicbp,
            :serumcreatinine, :bunlevels, :gfr, :acr,
            :serumelectrolytessodium, :serumelectrolytespotassium,
            :hemoglobinlevels, :hba1c, :pulsepressure, :ureacreatinineratio,
            :ckdstage, :albuminuriacat, :bp_risk, :hyperkalemiaflag, :anemiaflag
        ) RETURNING id;
    """)
    return conn.execute(q, row).scalar()

def _save_inference(conn, input_id: int, model_used: str, thr: float, pred_row: dict):
    q = text("""
        INSERT INTO inference_log (
            input_id, model_used, threshold_used, prediction, prob_ckd, prob_non_ckd
        ) VALUES (
            :input_id, :model_used, :threshold_used, :prediction, :prob_ckd, :prob_non_ckd
        );
    """)
    conn.execute(q, {
        "input_id": input_id,
        "model_used": model_used,
        "threshold_used": float(thr),
        "prediction": int(pred_row["prediction"]),
        "prob_ckd": float(pred_row["prob_ckd"]),
        "prob_non_ckd": float(pred_row["prob_non_ckd"]),
    })

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health(model: str = Query("xgb", description="xgb | rf | logreg")):
    try:
        _, thr, flip = get_model_and_thr(model)
        return {
            "status": "ok",
            "model": model,
            "threshold": thr,
            "legacy_flip_probas": flip,
            "db_logging": bool(engine is not None),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(
    item: PatientFeatures,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save input+prediction to DB"),
):
    df = pd.DataFrame([item.dict()])
    res = predict_core(df, model).iloc[0].to_dict()

    if save and engine is not None:
        try:
            with engine.begin() as conn:
                input_id = _save_user_input(conn, item.dict())
                _save_inference(conn, input_id, res["model_used"], res["threshold_used"], res)
        except Exception as e:
            # don't break inference if logging fails
            print(f"[warn] logging failed: {e}")

    return res

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save inputs+predictions to DB"),
):
    df = pd.DataFrame([r.dict() for r in req.rows])
    res_rows = predict_core(df, model).to_dict(orient="records")

    if save and engine is not None:
        try:
            with engine.begin() as conn:
                for row_in, row_out in zip(req.rows, res_rows):
                    input_id = _save_user_input(conn, row_in.dict())
                    _save_inference(conn, input_id, row_out["model_used"], row_out["threshold_used"], row_out)
        except Exception as e:
            print(f"[warn] batch logging failed: {e}")

    return {"predictions": res_rows}

# === Metrics ================================================================

@app.get("/metrics/retrain_report")
def retrain_report():
    """
    Returns last retraining summary if models/retrain_report.json exists.
    """
    path = Path("models/retrain_report.json")
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "retrain_report.json not found. Run ml/99_retrain.py or wait for CI retrain."},
        )
    try:
        data = json.loads(path.read_text())
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"failed to read report: {e}"})

@app.get("/metrics/last_inferences")
def last_inferences(limit: int = Query(10, ge=1, le=100)):
    """
    Returns recent inference rows (privacy-preserving), requires DATABASE_URL.
    """
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "DATABASE_URL not configured on server; cannot read inference_log."},
        )
    with engine.begin() as c:
        rows = c.execute(text("""
            SELECT il.created_at, il.model_used, il.prediction, il.prob_ckd, il.prob_non_ckd, il.threshold_used
            FROM inference_log il
            ORDER BY il.created_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"rows": [dict(r) for r in rows]}

# === Retrain / Reload =======================================================

def _run_retrain_pipeline():
    """
    Runs the unified retraining script, then clears in-memory cache so the
    next request serves the newly written artifacts.
    """
    try:
        # Adjust the script path if needed; this should write ./models/* and retrain_report.json
        subprocess.run([sys.executable, "ml/99_retrain.py"], check=True)
        # After artifacts are updated, clear cache to pick them up
        global _cache
        _cache.clear()
        print("[retrain] completed; cache cleared.")
    except subprocess.CalledProcessError as e:
        print(f"[retrain] failed: {e}")

@app.post("/admin/retrain")
def admin_retrain(background: BackgroundTasks, sync: bool = Query(False, description="Run synchronously (blocks request)")):
    """
    Kicks off retraining. By default runs in the background and returns immediately.
    Use sync=true to block until finished (not recommended for UI).
    """
    if sync:
        _run_retrain_pipeline()
        return {"status": "done", "mode": "sync"}
    else:
        background.add_task(_run_retrain_pipeline)
        return {"status": "started", "mode": "async"}

@app.post("/admin/reload", status_code=status.HTTP_204_NO_CONTENT)
def admin_reload():
    """
    Clears the in-memory model cache so the next call reloads fresh artifacts.
    Use this after CI retraining updates files in ./models.
    """
    global _cache
    _cache.clear()
    return
