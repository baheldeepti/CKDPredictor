# api/app.py
import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import shap
from joblib import load
from pydantic import BaseModel, Field, ConfigDict

from fastapi import FastAPI, HTTPException, Query, status, BackgroundTasks, Header
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional DB logging (safe if DATABASE_URL is unset)
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# App (instantiate FIRST) + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="CKD Predictor API", version="0.9.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your UI origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Friendly root: redirect "/" -> "/docs"
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# -----------------------------------------------------------------------------
# Configuration (paths & registry)
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (BASE_DIR.parent / "models"))

print(f"[boot] MODEL_DIR={MODEL_DIR.resolve()} exists={MODEL_DIR.exists()}")

# If your *older* XGB model (pre-ml/99_retrain.py) exposed proba[:,1] as P(non-CKD), set True.
# If using models from ml/99_retrain.py (standard class1=CKD), keep False.
LEGACY_XGB_FLIPPED: bool = False

REGISTRY: Dict[str, Dict[str, object]] = {
    "xgb": {
        "model": MODEL_DIR / "xgb_ckd.joblib",
        "thr": MODEL_DIR / "xgb_ckd_threshold.json",
        "flip_probas": LEGACY_XGB_FLIPPED,
    },
    "rf": {
        "model": MODEL_DIR / "rf_ckd.joblib",
        "thr": MODEL_DIR / "rf_ckd_threshold.json",
        "flip_probas": False,
    },
    "logreg": {
        "model": MODEL_DIR / "logreg_ckd.joblib",
        "thr": MODEL_DIR / "logreg_ckd_threshold.json",
        "flip_probas": False,
    },
}

FEATURE_COLS: List[str] = [
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

# Optional lightweight admin protection
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
def _check_admin(token: str | None):
    if not ADMIN_TOKEN:
        return
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
    model_config = ConfigDict(protected_namespaces=())
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
        prob_non_ckd = proba1
        prob_ckd = 1.0 - proba1
        pred_class0 = (prob_non_ckd >= thr).astype(int)  # 1 means "Non-CKD"
        prediction = np.where(pred_class0 == 1, 0, 1)
    else:
        prob_ckd = proba1
        prob_non_ckd = 1.0 - proba1
        prediction = (prob_ckd >= thr).astype(int)

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
# SHAP helper (hardened for RF + LogReg + XGB)
# -----------------------------------------------------------------------------

def _positive_class_index(model) -> int:
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        classes = list(classes)
        return classes.index(1) if 1 in classes else (np.argmax(classes))
    except Exception:
        return 1

def _extract_shap_values(sv, model) -> Tuple[np.ndarray, float]:
    """
    Robustly extract (values_for_positive_class, base_value) from a shap.Explanation
    across SHAP versions and model types.
    """
    # values may be array, list of arrays, or a 3D (classes, samples, features)
    v = getattr(sv, "values", None)
    if v is None:
        # Older API: sv is array-like
        v = sv[0].values
    vals = np.array(v)

    # base values
    base_raw = getattr(sv, "base_values", None)
    if base_raw is None:
        base_raw = 0.0
    base_arr = np.array(base_raw)

    if vals.ndim == 1:
        vals_out = vals
    elif vals.ndim == 2:
        # (samples, features)
        vals_out = vals[0]
    elif vals.ndim == 3:
        # (classes, samples, features)
        ci = _positive_class_index(model)
        vals_out = vals[ci, 0, :]
    else:
        raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

    # base handling: pick positive class if classwise, else first
    if base_arr.ndim == 0:
        base = float(base_arr)
    elif base_arr.ndim == 1:
        # could be (samples,) or (classes,)
        if len(base_arr) == 2 and hasattr(model, "classes_"):
            ci = _positive_class_index(model)
            base = float(base_arr[ci])
        else:
            base = float(base_arr[0])
    elif base_arr.ndim == 2:
        # (classes, samples)
        ci = _positive_class_index(model)
        base = float(base_arr[ci, 0])
    else:
        base = float(np.ravel(base_arr)[0])

    return vals_out, base

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    """
    Build a robust SHAP explainer, compute values for the first row.
    Works for tree (XGB/RF) and linear (logreg). Falls back to generic Explainer.
    Returns (base_value, dict{feature->value}, top_list[{feature, impact, signed}]).
    """
    X = features.copy()

    # Choose explainer
    explainer = None
    try:
        # Tree models: RandomForest*, XGB*, GradientBoost*
        name = model.__class__.__name__.lower()
        is_tree = ("forest" in name) or ("xgb" in name) or ("gradientboost" in name) or hasattr(model, "feature_importances_")
        if is_tree:
            # Use probability space for classifiers to avoid log-odds confusion
            explainer = shap.TreeExplainer(model, data=X, model_output="probability")
    except Exception:
        explainer = None

    if explainer is None:
        try:
            # Linear models (LogisticRegression/SGD/Linear*)
            name = model.__class__.__name__.lower()
            if name.startswith(("logisticregression", "sgdclassifier", "linearsvc", "linear")):
                mask = shap.maskers.Independent(X)
                explainer = shap.LinearExplainer(model, mask)
        except Exception:
            explainer = None

    # Generic fallback
    if explainer is None:
        try:
            explainer = shap.Explainer(model, X)
        except Exception:
            mask = shap.maskers.Independent(X)
            explainer = shap.Explainer(model, mask)

    sv = explainer(X.iloc[[0]])
    vals, base = _extract_shap_values(sv, model)

    feat_names = list(X.columns)
    shap_dict = {feat_names[i]: float(vals[i]) for i in range(len(feat_names))}
    top = sorted(
        [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_dict.items()],
        key=lambda d: d["impact"],
        reverse=True
    )[:k]
    return base, shap_dict, top

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

@app.post("/explain")
def explain(
    item: PatientFeatures,
    model: str = Query("xgb"),
):
    """
    Returns SHAP explanation for a single row, as JSON:
    {
      "base_value": float,
      "shap_values": {"feature": signed_value, ...},
      "top": [{"feature":"gfr","impact":0.34,"signed":-0.34}, ...]
    }
    """
    try:
        df_in = pd.DataFrame([item.dict()])
        for c in FEATURE_COLS:
            if c not in df_in.columns:
                raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
        X = df_in[FEATURE_COLS].copy()

        mdl, _, _ = get_model_and_thr(model)
        base, shap_values, top = _shap_top_k(X, mdl, k=6)
        return {
            "base_value": base,
            "shap_values": shap_values,
            "top": top,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Logs full traceback server-side; short message to client
        print("[/explain] error:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"explain_failed: {str(e)}", "model": model}
        )

# === Retrain / Reload =======================================================

def _run_retrain_pipeline():
    try:
        subprocess.run([sys.executable, "ml/99_retrain.py"], check=True)
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
    _check_admin(x_admin_token)
    if sync:
        _run_retrain_pipeline()
        return {"status": "done", "mode": "sync"}
    else:
        background.add_task(_run_retrain_pipeline)
        return {"status": "started", "mode": "async"}

@app.post("/admin/reload", status_code=status.HTTP_204_NO_CONTENT)
def admin_reload(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    global _cache
    _cache.clear()
    return  # 204 No Content

# === Metrics ================================================================

@app.get("/metrics/retrain_report")
def retrain_report():
    """
    Returns last retraining summary if models/retrain_report.json exists.
    """
    path = MODEL_DIR / "retrain_report.json"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "retrain_report.json not found"}
        )
    try:
        return json.loads(path.read_text())
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
