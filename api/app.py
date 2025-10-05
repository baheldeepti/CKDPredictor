# api/app.py
import os
import re
import sys
import json
import math
import random
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
app = FastAPI(title="CKD Predictor API", version="1.0.0")

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

# Environment overrides are allowed, e.g. RF_MODEL_PATH, LOGREG_MODEL_PATH, XGB_MODEL_PATH
ENV_MODEL_PATHS = {
    "rf": os.getenv("RF_MODEL_PATH"),
    "logreg": os.getenv("LOGREG_MODEL_PATH"),
    "xgb": os.getenv("XGB_MODEL_PATH"),
}
ENV_THR_PATHS = {
    "rf": os.getenv("RF_THR_PATH"),
    "logreg": os.getenv("LOGREG_THR_PATH"),
    "xgb": os.getenv("XGB_THR_PATH"),
}

# Default expected filenames (we'll also search for near-matches)
DEFAULT_FILES = {
    "xgb": ("xgb_ckd.joblib", "xgb_ckd_threshold.json"),
    "rf": ("rf_ckd.joblib", "rf_ckd_threshold.json"),
    "logreg": ("logreg_ckd.joblib", "logreg_ckd_threshold.json"),
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
# Utility: model key normalization & artifact discovery
# -----------------------------------------------------------------------------

ALIASES = {
    "rf": "rf",
    "random forest": "rf",
    "random_forest": "rf",
    "randomforest": "rf",
    "forest": "rf",
    "logreg": "logreg",
    "logistic regression": "logreg",
    "logistic": "logreg",
    "lr": "logreg",
    "xgb": "xgb",
    "xgboost": "xgb",
}

def _normalize_model_key(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"[\s\-_()]+", " ", t).strip()
    # try direct
    if t in ALIASES: return ALIASES[t]
    # token-based
    if "forest" in t or "rf" in t: return "rf"
    if "xgb" in t or "xgboost" in t: return "xgb"
    if "log" in t: return "logreg"
    return t  # maybe already exact

def _search_first(models_dir: Path, pat: str) -> Path | None:
    # find first file matching pattern (case-insensitive)
    pat_low = pat.lower()
    for p in models_dir.glob("*.joblib"):
        if pat_low in p.name.lower():
            return p
    return None

def _artifact_paths(kind: str) -> Tuple[Path, Path, bool]:
    """
    Return (model_path, threshold_path, flip_probas) for key 'rf'|'logreg'|'xgb'.
    Uses env overrides, else defaults, else best-effort search in MODEL_DIR.
    """
    if kind not in ("rf", "logreg", "xgb"):
        raise HTTPException(status_code=400, detail=f"Unknown model '{kind}'. Use one of ['rf','logreg','xgb'].")

    env_model = ENV_MODEL_PATHS.get(kind)
    env_thr = ENV_THR_PATHS.get(kind)

    if env_model:
        mpath = Path(env_model)
    else:
        default_name, default_thr = DEFAULT_FILES[kind]
        mpath = MODEL_DIR / default_name
        if not mpath.exists():
            # try fuzzy search: e.g., "*rf*ckd*.joblib"
            guess = _search_first(MODEL_DIR, kind+"_") or _search_first(MODEL_DIR, kind) or _search_first(MODEL_DIR, "ckd")
            if guess:
                mpath = guess

    if env_thr:
        tpath = Path(env_thr)
    else:
        default_name, default_thr = DEFAULT_FILES[kind]
        tpath = MODEL_DIR / default_thr

    flip = (LEGACY_XGB_FLIPPED if kind == "xgb" else False)
    return mpath, tpath, flip

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
    mk = _normalize_model_key(model_key)
    mpath, tpath, flip = _artifact_paths(mk)

    if not mpath.exists():
        raise HTTPException(status_code=400, detail=f"Model file not found for '{mk}'. Looked at: {mpath}")

    mdl = load(mpath)

    thr = 0.5
    if tpath.exists():
        try:
            with open(tpath) as f:
                thr = float(json.load(f).get("threshold", 0.5))
        except Exception:
            thr = 0.5

    return mdl, thr, flip

def get_model_and_thr(model_key: str) -> Tuple[object, float, bool]:
    mk = _normalize_model_key(model_key)
    if mk not in _cache:
        mdl, thr, flip = _load_artifacts(mk)
        _cache[mk] = {"model": mdl, "thr": thr, "flip": flip}
    d = _cache[mk]
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
        "model_used": _normalize_model_key(model_key),
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
# SHAP helpers (hardened for RF + LogReg + XGB)
# -----------------------------------------------------------------------------

def _positive_class_index(model) -> int:
    try:
        classes = getattr(model, "classes_", None)
        if classes is None: return 1
        classes = list(classes)
        return classes.index(1) if 1 in classes else int(np.argmax(classes))
    except Exception:
        return 1

def _build_background(row_df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """
    Create a small jittered background around the single row to avoid zero SHAP with
    degenerate background. This is for explanation only (hackathon-friendly).
    """
    rng = np.random.default_rng(42)
    row = row_df.iloc[0].copy()
    bg = []
    for _ in range(n):
        pert = {}
        for f in FEATURE_COLS:
            v = float(row[f])
            # 10% relative jitter (bounded), fallback to small absolute jitter
            scale = max(abs(v) * 0.10, 1.0)
            pert_val = v + rng.normal(0, scale)
            pert[f] = pert_val
        # clamp plausible ranges for ints / flags
        pert["gender"] = int(round(np.clip(pert["gender"], 0, 1)))
        pert["bp_risk"] = int(round(np.clip(pert["bp_risk"], 0, 3)))
        pert["hyperkalemiaflag"] = int(round(np.clip(pert["hyperkalemiaflag"], 0, 1)))
        pert["anemiaflag"] = int(round(np.clip(pert["anemiaflag"], 0, 1)))
        pert["ckdstage"] = int(round(np.clip(pert["ckdstage"], 0, 5)))
        pert["albuminuriacat"] = int(round(np.clip(pert["albuminuriacat"], 0, 3)))
        bg.append(pert)
    df_bg = pd.DataFrame(bg)[FEATURE_COLS]
    return df_bg

def _extract_shap_values(sv, model) -> Tuple[np.ndarray, float]:
    """
    Robustly extract (values_for_positive_class, base_value) from a shap.Explanation
    across SHAP versions and model types.
    """
    v = getattr(sv, "values", None)
    if v is None:
        v = sv[0].values
    vals = np.array(v)

    base_raw = getattr(sv, "base_values", None)
    if base_raw is None:
        base_raw = 0.0
    base_arr = np.array(base_raw)

    if vals.ndim == 1:
        vals_out = vals
    elif vals.ndim == 2:
        vals_out = vals[0]
    elif vals.ndim == 3:
        ci = _positive_class_index(model)
        vals_out = vals[ci, 0, :]
    else:
        raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

    # base handling
    if base_arr.ndim == 0:
        base = float(base_arr)
    elif base_arr.ndim == 1:
        if len(base_arr) == 2 and hasattr(model, "classes_"):
            ci = _positive_class_index(model)
            base = float(base_arr[ci])
        else:
            base = float(base_arr[0])
    elif base_arr.ndim == 2:
        ci = _positive_class_index(model)
        base = float(base_arr[ci, 0])
    else:
        base = float(np.ravel(base_arr)[0])

    return vals_out, base

def _proba_fn(model):
    ci = _positive_class_index(model)
    def f(Xnp: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(Xnp, columns=FEATURE_COLS)
        p = model.predict_proba(df)
        return p[:, ci]
    return f

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    """
    Build a robust SHAP explainer for a single row.
    1) Try Tree/Linear explainers
    2) Fallback to model-agnostic Permutation/Kernel with a jittered background
    Returns (base_value, dict{feature->value}, top_list[{feature, impact, signed}]).
    """
    X = features.copy()
    assert X.shape[0] == 1, "expect single-row dataframe"

    # Try specialized explainers
    explainer = None
    try:
        name = model.__class__.__name__.lower()
        is_tree = ("forest" in name) or ("xgb" in name) or ("gradientboost" in name) or hasattr(model, "feature_importances_")
        if is_tree:
            explainer = shap.TreeExplainer(model, model_output="probability")
    except Exception:
        explainer = None

    if explainer is None:
        try:
            name = model.__class__.__name__.lower()
            if name.startswith(("logisticregression", "sgdclassifier", "linearsvc", "linear")):
                mask = shap.maskers.Independent(_build_background(X, n=50))
                explainer = shap.LinearExplainer(model, mask)
        except Exception:
            explainer = None

    # Compute with specialized explainer if we have one
    if explainer is not None:
        try:
            sv = explainer(X)
            vals, base = _extract_shap_values(sv, model)
            if np.all(np.isfinite(vals)) and not np.allclose(vals, 0, atol=1e-10):
                feat_names = list(X.columns)
                shap_dict = {feat_names[i]: float(vals[i]) for i in range(len(feat_names))}
                top = sorted(
                    [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_dict.items()],
                    key=lambda d: d["impact"],
                    reverse=True
                )[:k]
                return base, shap_dict, top
        except Exception:
            pass  # fall through to model-agnostic path

    # Model-agnostic fallback (works for pipelines too)
    bg = _build_background(X, n=50)
    f = _proba_fn(model)

    # Prefer PermutationExplainer if available; else KernelExplainer
    try:
        expl = shap.explainers.Permutation(f, bg)
        sv = expl(X)
    except Exception:
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(X, nsamples=100)
        # KernelExplainer returns array, not Explanation
        vals = np.array(sv)[0] if isinstance(sv, list) else np.array(sv)[0]
        base = float(expl.expected_value) if np.isscalar(expl.expected_value) else float(np.array(expl.expected_value)[0])
        feat_names = list(X.columns)
        shap_dict = {feat_names[i]: float(vals[i]) for i in range(len(feat_names))}
        top = sorted(
            [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_dict.items()],
            key=lambda d: d["impact"],
            reverse=True
        )[:k]
        return base, shap_dict, top

    # PermutationExplainer path
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

@app.get("/models")
def models():
    """
    Report which model artifacts are detected.
    """
    out = {}
    for mk in ("rf", "logreg", "xgb"):
        mpath, tpath, flip = _artifact_paths(mk)
        out[mk] = {
            "model_path": str(mpath.resolve()),
            "model_exists": mpath.exists(),
            "threshold_path": str(tpath.resolve()),
            "threshold_exists": tpath.exists(),
            "legacy_flip_probas": flip,
        }
    return out

@app.get("/health")
def health(model: str = Query("xgb", description="xgb | rf | logreg (aliases accepted)")):
    try:
        mdl, thr, flip = get_model_and_thr(model)
        return {
            "status": "ok",
            "model": _normalize_model_key(model),
            "threshold": thr,
            "legacy_flip_probas": flip,
            "db_logging": bool(engine is not None),
        }
    except HTTPException as he:
        return JSONResponse(status_code=400, content={"status": "error", "detail": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

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
    SHAP explanation for a single row (robust across RF / LogReg / XGB).
    """
    try:
        df_in = pd.DataFrame([item.dict()])
        X = _validate_and_order(df_in)
        mdl, _, _ = get_model_and_thr(model)

        base, shap_values, top = _shap_top_k(X, mdl, k=6)
        return {
            "base_value": float(base),
            "shap_values": shap_values,
            "top": top,
        }
    except HTTPException:
        raise
    except Exception as e:
        print("[/explain] error:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"explain_failed: {str(e)}", "model": _normalize_model_key(model)}
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
