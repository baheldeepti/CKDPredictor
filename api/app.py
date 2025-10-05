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
app = FastAPI(title="CKD Predictor API", version="1.1.0")

# Allow Streamlit Cloud and your Render site by default; keep "*" for simplicity during hackathon
ALLOWED_ORIGINS = [
    "*",
    "https://ckdpredictor.streamlit.app",
    "https://ckdpredictor.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
BASE_DIR = Path(__file__).resolve().parent            # .../api
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (BASE_DIR.parent / "models"))

print(f"[boot] MODEL_DIR={MODEL_DIR.resolve()} exists={MODEL_DIR.exists()}")

# If your *older* XGB model exposed proba[:,1] as P(non-CKD), set True.
LEGACY_XGB_FLIPPED: bool = False

REGISTRY: Dict[str, Dict[str, object]] = {
    "xgb":    {"model": MODEL_DIR / "xgb_ckd.joblib",    "thr": MODEL_DIR / "xgb_ckd_threshold.json",    "flip_probas": LEGACY_XGB_FLIPPED},
    "rf":     {"model": MODEL_DIR / "rf_ckd.joblib",     "thr": MODEL_DIR / "rf_ckd_threshold.json",     "flip_probas": False},
    "logreg": {"model": MODEL_DIR / "logreg_ckd.joblib", "thr": MODEL_DIR / "logreg_ckd_threshold.json", "flip_probas": False},
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
print(f"[boot] DB logging enabled={bool(engine)}")

# Optional lightweight admin protection
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
def _check_admin(token: str | None):
    if not ADMIN_TOKEN:
        return
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_model_key(s: str) -> str:
    """Accept both 'rf' and labels like 'Random Forest (rf)' coming from UI."""
    if not s:
        return "xgb"
    t = s.strip().lower()
    if t in ("xgb","rf","logreg"):
        return t
    if "random" in t and "forest" in t:
        return "rf"
    if "xgboost" in t or "(xgb)" in t:
        return "xgb"
    if "logistic" in t or "logreg" in t:
        return "logreg"
    if "(" in t and ")" in t:
        inner = t[t.find("(")+1:t.find(")")]
        if inner in ("xgb","rf","logreg"):
            return inner
    return t

def _positive_class_index(model) -> int:
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        classes = list(classes)
        return classes.index(1) if 1 in classes else int(np.argmax(classes))
    except Exception:
        return 1

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
# Model cache & loading
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
            try:
                thr = float(json.load(f)["threshold"])
            except Exception:
                # fallback tolerant parser
                try:
                    thr = float(pd.read_json(paths["thr"]).get("threshold", 0.5))
                except Exception:
                    thr = 0.5
    flip = bool(paths.get("flip_probas", False))
    return mdl, thr, flip

def get_model_and_thr(model_key: str) -> Tuple[object, float, bool]:
    mk = normalize_model_key(model_key)
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

    # Use the actual positive class index (not always column 1)
    ci = _positive_class_index(mdl)
    proba_pos = mdl.predict_proba(df)[:, ci]

    if flip:
        # legacy XGB: proba is Non-CKD -> flip
        prob_non_ckd = proba_pos
        prob_ckd = 1.0 - proba_pos
        pred_class0 = (prob_non_ckd >= thr).astype(int)  # 1 == Non-CKD
        prediction = np.where(pred_class0 == 1, 0, 1)
    else:
        prob_ckd = proba_pos
        prob_non_ckd = 1.0 - proba_pos
        prediction = (prob_ckd >= thr).astype(int)

    out = pd.DataFrame({
        "prediction": prediction.astype(int),
        "prob_ckd": prob_ckd.astype(float),
        "prob_non_ckd": prob_non_ckd.astype(float),
        "threshold_used": float(thr),
        "model_used": normalize_model_key(model_key),
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
# SHAP helpers (lazy import + hardened across versions)
# -----------------------------------------------------------------------------
def _build_explainer(model, X: pd.DataFrame):
    try:
        import shap  # lazy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP not installed server-side: {e}")

    name = model.__class__.__name__.lower()
    # Prefer TreeExplainer for tree models; explain in probability space
    try:
        is_tree = ("forest" in name) or ("xgb" in name) or ("gradientboost" in name) or hasattr(model, "feature_importances_")
        if is_tree:
            return shap.TreeExplainer(model, data=X, model_output="probability")
    except Exception:
        pass
    # Linear models (Logistic Regression etc.)
    try:
        if name.startswith(("logisticregression", "sgdclassifier", "linearsvc", "linear")):
            mask = shap.maskers.Independent(X)
            return shap.LinearExplainer(model, mask)
    except Exception:
        pass
    # Generic fallback
    try:
        return shap.Explainer(model, X)
    except Exception:
        mask = shap.maskers.Independent(X)
        return shap.Explainer(model, mask)

def _extract_shap(sv, model) -> Tuple[np.ndarray, float]:
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
        vals_out = vals[0]            # (samples, features)
    elif vals.ndim == 3:
        ci = _positive_class_index(model)
        vals_out = vals[ci, 0, :]     # (classes, samples, features)
    else:
        vals_out = np.ravel(vals)

    if base_arr.ndim == 0:
        base = float(base_arr)
    elif base_arr.ndim in (1, 2):
        try:
            ci = _positive_class_index(model)
            base = float(base_arr[ci] if base_arr.ndim == 1 else base_arr[ci, 0])
        except Exception:
            base = float(np.ravel(base_arr)[0])
    else:
        base = float(np.ravel(base_arr)[0])

    return vals_out, base

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    X = features.copy()
    explainer = _build_explainer(model, X)
    sv = explainer(X.iloc[[0]])
    vals, base = _extract_shap(sv, model)

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
def health(model: str = Query("xgb", description="xgb | rf | logreg | labels ok")):
    try:
        mk = normalize_model_key(model)
        _, thr, flip = get_model_and_thr(mk)

        db_connected = False
        if engine is not None:
            try:
                with engine.begin() as c:
                    c.execute(text("SELECT 1"))
                db_connected = True
            except Exception:
                db_connected = False

        return {
            "status": "ok",
            "model": mk,
            "threshold": thr,
            "legacy_flip_probas": flip,
            "db_logging": bool(engine is not None),
            "db_connected": db_connected,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(
    item: PatientFeatures,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save input+prediction to DB"),
):
    mk = normalize_model_key(model)
    df = pd.DataFrame([item.dict()])
    res = predict_core(df, mk).iloc[0].to_dict()

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
    mk = normalize_model_key(model)
    df = pd.DataFrame([r.dict() for r in req.rows])
    res_rows = predict_core(df, mk).to_dict(orient="records")

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
    try:
        mk = normalize_model_key(model)
        df_in = pd.DataFrame([item.dict()])
        X = _validate_and_order(df_in)
        mdl, _, _ = get_model_and_thr(mk)

        base, shap_values, top = _shap_top_k(X, mdl, k=6)
        return {"base_value": base, "shap_values": shap_values, "top": top}
    except HTTPException:
        raise
    except Exception as e:
        print("[/explain] error:", e)
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"explain_failed: {str(e)}", "model": model})

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
    path = MODEL_DIR / "retrain_report.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": "retrain_report.json not found"})
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"failed to read report: {e}"})

@app.get("/metrics/last_inferences")
def last_inferences(limit: int = Query(10, ge=1, le=100)):
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
