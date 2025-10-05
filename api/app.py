# api/app.py
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, status, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional DB logging (safe if DATABASE_URL is unset)
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# App (instantiate first) + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="CKD Predictor API", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your UI origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# If your *older* XGBoost model (pre-ml/99_retrain.py) exposed proba[:,1] as P(non-CKD),
# set this True. If using models from ml/99_retrain.py (standard class1=CKD), keep False.
LEGACY_XGB_FLIPPED: bool = False

REGISTRY: Dict[str, Dict[str, object]] = {
    "xgb": {
        "model": Path("models/xgb_ckd.joblib"),
        "thr": Path("models/xgb_ckd_threshold.json"),
        "flip_probas": LEGACY_XGB_FLIPPED,
    },
    "rf": {
        "model": Path("models/rf_ckd.joblib"),
        "thr": Path("models/rf_ckd_threshold.json"),
        "flip_probas": False,
    },
    "logreg": {
        "model": Path("models/logreg_ckd.joblib"),
        "thr": Path("models/logreg_ckd_threshold.json"),
        "flip_probas": False,
    },
}

FEATURE_COLS: List[str] = [
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

# Database (optional; used for logging)
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# Optional lightweight admin protection (set ADMIN_TOKEN env var to enable)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


def _check_admin(token: str | None):
    if not ADMIN_TOKEN:
        return  # no protection configured
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

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
    prediction: int = Field(description="0 = Non-CKD, 1 = CKD")
    prob_ckd: float
    prob_non_ckd: float
    threshold_used: float
    model_used: str


class BatchPredictRequest(BaseModel):
    rows: List[PatientFeatures]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class ExplainResponse(BaseModel):
    base_value: float
    shap_values: Dict[str, float]   # feature -> signed impact
    top: List[Dict[str, float]]     # list of {feature, impact, signed}

# -----------------------------------------------------------------------------
# Model cache
# -----------------------------------------------------------------------------
_cache: Dict[str, Dict[str, object]] = {}  # {model: {"model": sklearn_obj, "thr": float, "flip": bool}}


def _load_artifacts(model_key: str) -> Tuple[object, float, bool]:
    mk = model_key.lower()
    if mk not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'. Use one of {list(REGISTRY)}")
    paths = REGISTRY[mk]
    if not paths["model"].exists():
        raise HTTPException(status_code=400, detail=f"Model file missing for '{mk}'. Train it first.")
    mdl = load(paths["model"])
    thr = 0.5
    if paths["thr"].exists():
        with open(paths["thr"]) as f:
            thr = float(json.load(f)["threshold"])
    flip = bool(paths.get("flip_probas", False))
    return mdl, thr, flip


def get_model_and_thr(model_key: str) -> Tuple[object, float, bool]:
    if model_key not in _cache:
        mdl, thr, flip = _load_artifacts(model_key)
        _cache[model_key] = {"model": mdl, "thr": thr, "flip": flip}
    d = _cache[model_key]
    return d["model"], d["thr"], d["flip"]

# -----------------------------------------------------------------------------
# Core prediction
# -----------------------------------------------------------------------------

def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
    return df[FEATURE_COLS].copy()


def predict_core(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = _validate_and_order(df)
    mdl, thr, flip = get_model_and_thr(model_key)

    proba1 = mdl.predict_proba(df)[:, 1]  # standard: P(CKD)
    if flip:
        # Legacy XGB path: proba1 is actually P(Non-CKD)
        prob_non_ckd = proba1
        prob_ckd = 1.0 - proba1
        pred_class0 = (prob_non_ckd >= thr).astype(int)  # 1 means "Non-CKD"
        prediction = np.where(pred_class0 == 1, 0, 1)    # convert to 0/1 in original labels
    else:
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

# === Explainability ==========================================================

@app.post("/explain", response_model=ExplainResponse)
def explain(
    item: PatientFeatures,
    model: str = Query("xgb"),
):
    """
    SHAP explanation for a single row.
    Returns base_value, per-feature signed shap values, and top features by |impact|.
    """
    try:
        import shap  # uses shap.Explainer which auto-picks model type (tree/linear/kernel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP not available: {e}")

    df = pd.DataFrame([item.dict()])
    df = _validate_and_order(df)
    mdl, _, _ = get_model_and_thr(model)

    try:
        explainer = shap.Explainer(mdl)     # works for tree/linear/sklearn pipelines
        sv = explainer(df)                  # shap.Explanation
        base = float(np.array(sv.base_values)[0])
        vals = np.array(sv.values)[0]       # per-feature shap values
    except Exception as e:
        # fallback: TreeExplainer for tree models
        try:
            explainer = shap.TreeExplainer(mdl)
            sv = explainer.shap_values(df)
            # For binary classification, shap_values may be list [class0, class1]; take class1
            if isinstance(sv, list) and len(sv) == 2:
                vals = np.array(sv[1][0])
            else:
                vals = np.array(sv[0])
            ev = getattr(explainer, "expected_value", 0.0)
            base = float(ev[1] if isinstance(ev, (list, tuple)) and len(ev) > 1 else ev)
        except Exception as ee:
            raise HTTPException(status_code=500, detail=f"Failed to compute SHAP values: {e}; fallback error: {ee}")

    shap_map = {feat: float(v) for feat, v in zip(FEATURE_COLS, vals)}
    top_sorted = sorted(
        [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_map.items()],
        key=lambda x: x["impact"],
        reverse=True
    )[:8]

    return {
        "base_value": base,
        "shap_values": shap_map,
        "top": top_sorted,
    }

# === Retrain / Reload =======================================================

def _run_retrain_pipeline():
    """
    Runs the unified retraining script, then clears in-memory cache so the
    next request serves the newly written artifacts.
    """
    try:
        # This should write ./models/* and models/retrain_report.json
        subprocess.run([sys.executable, "ml/99_retrain.py"], check=True)
        # After artifacts are updated, clear cache to pick them up
        global _cache
        _cache.clear()
        print("[retrain] completed; cache cleared.")
    except subprocess.CalledProcessError as e:
        print(f"[retrain] failed: {e}")


@app.post("/admin/retrain")
def admin_retrain(
    background: BackgroundTasks,
    sync: bool = Query(False, description="Run synchronously (blocks request)"),
    x_admin_token: str | None = Header(None),
):
    """
    Kicks off retraining. By default runs in the background and returns immediately.
    Use sync=true to block until finished (not recommended for UI).
    """
    _check_admin(x_admin_token)
    if sync:
        _run_retrain_pipeline()
        return {"status": "done", "mode": "sync"}
    else:
        background.add_task(_run_retrain_pipeline)
        return {"status": "started", "mode": "async"}


@app.post("/admin/reload", status_code=status.HTTP_204_NO_CONTENT)
def admin_reload(x_admin_token: str | None = Header(None)):
    """
    Clears the in-memory model cache so the next call reloads fresh artifacts.
    Use this after CI retraining updates files in ./models.
    """
    _check_admin(x_admin_token)
    global _cache
    _cache.clear()
    return  # 204 No Content
