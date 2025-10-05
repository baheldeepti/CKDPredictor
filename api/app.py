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
app = FastAPI(title="CKD Predictor API", version="0.8.0")

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

# Base paths
BASE_DIR = Path(__file__).resolve().parent           # .../api
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (BASE_DIR.parent / "models"))  # .../<repo_root>/models

# Optional: print where we're loading models from
print(f"[boot] MODEL_DIR={MODEL_DIR.resolve()} exists={MODEL_DIR.exists()}")

# If your *older* XGBoost model (pre-ml/99_retrain.py) exposed proba[:,1] as P(non-CKD),
# set this True. If using models from ml/99_retrain.py (standard class1=CKD), keep False.
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
    # Silence: "Field 'model_used' conflicts with protected namespace 'model_'"
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
# SHAP helper (robust)
# -----------------------------------------------------------------------------

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    """
    Build a robust SHAP explainer, compute values for the first row.
    Works for tree (XGB/RF) and linear (logreg). Falls back to generic Explainer.
    Returns (base_value, dict{feature->value}, top_list[{feature, impact, signed}]).
    """
    X = features.copy()

    # Try specialized explainers first for stability & speed
    explainer = None
    try:
        # XGB / RF (tree-based)
        from xgboost import XGBClassifier  # noqa: F401
        is_tree = (
            hasattr(model, "feature_importances_")
            or model.__class__.__name__.lower().startswith(("xgb", "randomforest", "gradientboost"))
        )
        if is_tree:
            explainer = shap.TreeExplainer(model, feature_names=X.columns.tolist())
    except Exception:
        explainer = None

    if explainer is None:
        try:
            # Linear models (logreg, etc.)
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            if model.__class__.__name__.lower().startswith(("logisticregression", "sgdclassifier", "linearsvc")):
                mask = shap.maskers.Independent(X)
                explainer = shap.LinearExplainer(model, mask)
        except Exception:
            explainer = None

    # Generic fallback
    if explainer is None:
        try:
            explainer = shap.Explainer(model, X, feature_names=X.columns.tolist())
        except Exception:
            # Last resort: independent masker + generic
            mask = shap.maskers.Independent(X)
            explainer = shap.Explainer(model, mask)

    # Compute SHAP values for the first row only (what the UI needs)
    sv = explainer(X.iloc[[0]])
    # sv.values shape: (1, n_features); sv.base_values shape: (1,) or scalar
    vals = sv.values[0] if hasattr(sv, "values") else sv[0].values
    base = float(sv.base_values[0] if getattr(sv, "base_values", None) is not None else 0.0)

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
        # Build single-row DataFrame in correct column order
        df_in = pd.DataFrame([item.dict()])
        for c in FEATURE_COLS:
            if c not in df_in.columns:
                raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
        X = df_in[FEATURE_COLS].copy()

        mdl, _, _ = get_model_and_thr(model)

        # SHAP
        base, shap_values, top = _shap_top_k(X, mdl, k=6)
        return {
            "base_value": base,
            "shap_values": shap_values,
            "top": top,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to the API log; return brief error to client
        print("[/explain] error:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"explain_failed: {str(e)}"}
        )

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
    If ADMIN_TOKEN env is set, pass header: X-Admin-Token: <token>
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
    If ADMIN_TOKEN env is set, pass header: X-Admin-Token: <token>
    """
    _check_admin(x_admin_token)
    global _cache
    _cache.clear()
    return  # 204 No Content
