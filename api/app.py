# api/app.py
import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import shap
from joblib import load

from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, Query, status, BackgroundTasks, Header, Body
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="CKD Predictor API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# -----------------------------------------------------------------------------
# Paths & Registry
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent           # .../api
ROOT_DIR = BASE_DIR.parent                           # repo root
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (ROOT_DIR / "models"))

print(f"[boot] MODEL_DIR={MODEL_DIR.resolve()} exists={MODEL_DIR.exists()}")

# If your *older* XGBoost model exposed proba[:,1] as P(non-CKD), flip it.
LEGACY_XGB_FLIPPED: bool = False

REGISTRY: Dict[str, Dict[str, object]] = {
    "xgb": {
        "model": MODEL_DIR / "xgb_ckd.joblib",
        "thr":   MODEL_DIR / "xgb_ckd_threshold.json",
        "flip_probas": LEGACY_XGB_FLIPPED,
    },
    "rf": {
        "model": MODEL_DIR / "rf_ckd.joblib",
        "thr":   MODEL_DIR / "rf_ckd_threshold.json",
        "flip_probas": False,
    },
    "logreg": {
        "model": MODEL_DIR / "logreg_ckd.joblib",
        "thr":   MODEL_DIR / "logreg_ckd_threshold.json",
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# -----------------------------------------------------------------------------
# Import optional modules (robust to different entrypoints)
# -----------------------------------------------------------------------------
def _import_or_error() -> Dict[str, Any]:
    from importlib import import_module
    mods: Dict[str, Any] = {"errors": {}}

    # Determine the package prefix safely (works whether run as api.app or as a script)
    pkg = __package__ or Path(__file__).resolve().parent.name  # usually "api"

    def _try(name: str, attr: str, key: str):
        try:
            m = import_module(f"{pkg}.{name}")
            mods[key] = getattr(m, attr, None)
        except Exception as e:
            mods[key] = None
            mods["errors"][name] = f"{e}\n{traceback.format_exc()}"

    _try("agents", "multi_agent_plan",   "multi_agent_plan")
    _try("digital_twin", "simulate_whatif", "simulate_whatif")
    _try("counterfactuals", "counterfactual", "counterfactual")
    _try("similarity", "knn_similar",     "knn_similar")
    return mods

_opt = _import_or_error()
multi_agent_plan = _opt.get("multi_agent_plan")
simulate_whatif   = _opt.get("simulate_whatif")
counterfactual    = _opt.get("counterfactual")
knn_similar       = _opt.get("knn_similar")
_import_errors: Dict[str, str] = _opt.get("errors", {})

# -----------------------------------------------------------------------------
# Database (default ON): Neon via DATABASE_URL, else SQLite fallback
# -----------------------------------------------------------------------------
def _db_url_with_fallback() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        return env_url
    # SQLite fallback inside repo (keeps Metrics tab alive everywhere)
    sqlite_path = (ROOT_DIR / "models" / "inference.db").resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path}"

DATABASE_URL = _db_url_with_fallback()
engine: Engine | None = create_engine(DATABASE_URL, future=True)

def _db_backend() -> str:
    try:
        name = engine.url.get_backend_name()  # type: ignore
        return name or "unknown"
    except Exception:
        return "unknown"

def _ensure_tables():
    """Create tables if missing (works for Postgres & SQLite)."""
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_inputs (
              id BIGSERIAL PRIMARY KEY,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              age DOUBLE PRECISION, gender INT,
              systolicbp DOUBLE PRECISION, diastolicbp DOUBLE PRECISION,
              serumcreatinine DOUBLE PRECISION, bunlevels DOUBLE PRECISION,
              gfr DOUBLE PRECISION, acr DOUBLE PRECISION,
              serumelectrolytessodium DOUBLE PRECISION, serumelectrolytespotassium DOUBLE PRECISION,
              hemoglobinlevels DOUBLE PRECISION, hba1c DOUBLE PRECISION,
              pulsepressure DOUBLE PRECISION, ureacreatinineratio DOUBLE PRECISION,
              ckdstage INT, albuminuriacat INT,
              bp_risk INT, hyperkalemiaflag INT, anemiaflag INT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS inference_log (
              id BIGSERIAL PRIMARY KEY,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              input_id BIGINT REFERENCES user_inputs(id) ON DELETE CASCADE,
              model_used TEXT NOT NULL,
              threshold_used DOUBLE PRECISION NOT NULL,
              prediction INT NOT NULL,
              prob_ckd DOUBLE PRECISION NOT NULL,
              prob_non_ckd DOUBLE PRECISION NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS inference_log_created_at_idx
            ON inference_log(created_at DESC);
        """))

# try to connect & ensure tables
_db_ok = False
try:
    with engine.begin() as _c:
        _c.exec_driver_sql("SELECT 1;")
    _db_ok = True
    _ensure_tables()
except Exception as e:
    print(f"[boot][db] connection failed: {e}")
    _db_ok = False

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
    shap_values: Dict[str, float]
    top: List[Dict[str, float]]

class WhatIfRequest(BaseModel):
    base: PatientFeatures
    deltas: Optional[Dict[str, float]] = None
    model: str = "xgb"
    grid: Optional[Dict[str, List[float]]] = None

class CFRequest(BaseModel):
    base: PatientFeatures
    target_prob: float = Field(0.2, ge=0.0, le=1.0)
    model: str = "xgb"
    method: str = "auto"  # "auto" | "greedy" | "dice" etc.

class SimilarRequest(BaseModel):
    base: PatientFeatures
    cohort: Optional[List[PatientFeatures]] = Field(default=None, description="Cohort to search in (optional)")

# -----------------------------------------------------------------------------
# Model cache
# -----------------------------------------------------------------------------
_cache: Dict[str, Dict[str, object]] = {}  # {key: {"model": obj, "thr": float, "flip": bool}}

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
        try:
            with open(paths["thr"]) as f:
                thr = float(json.load(f)["threshold"])
        except Exception:
            pass
    flip = bool(paths.get("flip_probas", False))
    return mdl, thr, flip

def get_model_and_thr(model_key: str) -> Tuple[object, float, bool]:
    if model_key not in _cache:
        mdl, thr, flip = _load_artifacts(model_key)
        _cache[model_key] = {"model": mdl, "thr": thr, "flip": flip}
    d = _cache[model_key]
    return d["model"], d["thr"], d["flip"]

# -----------------------------------------------------------------------------
# Prediction core
# -----------------------------------------------------------------------------
def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing feature: {c}")
    return df[FEATURE_COLS].copy()

def predict_core(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = _validate_and_order(df)
    mdl, thr, flip = get_model_and_thr(model_key)

    # sklearn predict_proba
    proba1 = mdl.predict_proba(df)[:, 1]  # P(class=1) in normal case
    if flip:
        # legacy xgb where [:,1] meant P(non-CKD)
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
# DB helpers
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
    try:
        return int(conn.execute(q, row).scalar())
    except Exception:
        # SQLite doesn't support RETURNING on older versions; fallback
        conn.execute(text(
            "INSERT INTO user_inputs ("
            "age, gender, systolicbp, diastolicbp, serumcreatinine, bunlevels, gfr, acr,"
            "serumelectrolytessodium, serumelectrolytespotassium, hemoglobinlevels, hba1c,"
            "pulsepressure, ureacreatinineratio, ckdstage, albuminuriacat, bp_risk, hyperkalemiaflag, anemiaflag"
            ") VALUES ("
            ":age, :gender, :systolicbp, :diastolicbp, :serumcreatinine, :bunlevels, :gfr, :acr,"
            ":serumelectrolytessodium, :serumelectrolytespotassium, :hemoglobinlevels, :hba1c,"
            ":pulsepressure, :ureacreatinineratio, :ckdstage, :albuminuriacat, :bp_risk, :hyperkalemiaflag, :anemiaflag"
            ");"
        ), row)
        rid = conn.execute(text("SELECT last_insert_rowid();")).scalar()
        return int(rid)

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
# SHAP utils (bulletproof for RF / XGB / LogReg)
# -----------------------------------------------------------------------------
def _pos_index(model) -> int:
    """Index of the positive class (prefer label 1)."""
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        arr = np.array(list(classes))
        if np.issubdtype(arr.dtype, np.number):
            if (arr == 1).any():
                return int(np.where(arr == 1)[0][0])
            return int(np.argmax(arr))
        labels = [str(x) for x in arr.tolist()]
        if "1" in labels:
            return labels.index("1")
        return len(labels) - 1
    except Exception:
        return 1

def _to_1d_float(vec) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(float, copy=False)

def _to_scalar_float(x, prefer_index: int | None = None) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 0:
        return float(arr)
    idx = 0 if prefer_index is None else min(max(prefer_index, 0), arr.shape[0] - 1)
    try:
        return float(arr.reshape(-1)[idx])
    except Exception:
        return float(np.ravel(arr)[0])

def _compute_shap_binary(model, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Compute SHAP values for the positive class for the first row."""
    pos = _pos_index(model)

    # Tree-based (RF/XGB)
    try:
        try:
            tree_expl = shap.TreeExplainer(model, model_output="probability")
        except TypeError:
            tree_expl = shap.TreeExplainer(model)
        if hasattr(tree_expl, "shap_values"):
            sv = tree_expl.shap_values(X)
            if isinstance(sv, list):
                idx = 0 if len(sv) == 1 else min(pos, len(sv) - 1)
                arr = np.asarray(sv[idx])
            else:
                arr = np.asarray(sv)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 2) and arr.shape[1] >= 1:
                    vals = arr[min(pos, arr.shape[0] - 1), 0, :]
                elif arr.shape[1] in (1, 2) and arr.shape[0] >= 1:
                    vals = arr[0, min(pos, arr.shape[1] - 1), :]
                else:
                    vals = arr.reshape(-1)[0: X.shape[1]]
            elif arr.ndim == 2:
                vals = arr[0]
            elif arr.ndim == 1:
                vals = arr
            else:
                vals = arr.reshape(-1)[0: X.shape[1]]

            base = getattr(tree_expl, "expected_value", 0.0)
            base_val = _to_scalar_float(base, prefer_index=None if np.asarray(base).size == 1 else pos)
            return _to_1d_float(vals), base_val
    except Exception:
        pass

    # Linear (LogReg)
    try:
        lin_expl = shap.LinearExplainer(model, X, feature_dependence="independent")
        if hasattr(lin_expl, "shap_values"):
            arr = np.asarray(lin_expl.shap_values(X))
            vals = arr[0] if arr.ndim == 2 else arr
            base = getattr(lin_expl, "expected_value", 0.0)
            base_val = _to_scalar_float(base, prefer_index=None if np.asarray(base).size == 1 else pos)
            return _to_1d_float(vals), base_val
    except Exception:
        pass

    # Unified API
    try:
        exp = shap.Explainer(model, X)
        res = exp(X.iloc[[0]])
        v = getattr(res, "values", None)
        b = getattr(res, "base_values", 0.0)
        if isinstance(v, list):
            idx = 0 if len(v) == 1 else min(pos, len(v) - 1)
            arr = np.asarray(v[idx])
            vals = arr[0] if arr.ndim >= 2 else arr
        else:
            arr = np.asarray(v)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 2) and arr.shape[1] >= 1:
                    vals = arr[min(pos, arr.shape[0] - 1), 0, :]
                elif arr.shape[1] in (1, 2) and arr.shape[0] >= 1:
                    vals = arr[0, min(pos, arr.shape[1] - 1), :]
                else:
                    vals = arr.reshape(-1)[0: X.shape[1]]
            elif arr.ndim == 2:
                vals = arr[0]
            elif arr.ndim == 1:
                vals = arr
            else:
                vals = arr.reshape(-1)[0: X.shape[1]]
        base_val = _to_scalar_float(b, prefer_index=None if np.asarray(b).size == 1 else pos)
        return _to_1d_float(vals), base_val
    except Exception:
        pass

    # Fallback
    try:
        if hasattr(model, "feature_importances_"):
            vals = _to_1d_float(getattr(model, "feature_importances_"))
            return vals, 0.0
        if hasattr(model, "coef_"):
            vals = _to_1d_float(getattr(model, "coef_"))
            return vals, 0.0
    except Exception:
        pass

    raise RuntimeError("Could not compute SHAP values for this model/version.")

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    X = features.copy()
    vals, base = _compute_shap_binary(model, X)
    feat_names = list(X.columns)
    n = min(len(feat_names), vals.size)
    vals = vals[:n]
    shap_dict = {feat_names[i]: float(vals[i]) for i in range(n)}
    top = sorted(
        [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_dict.items()],
        key=lambda d: d["impact"], reverse=True
    )[:k]
    return float(base), shap_dict, top

# -----------------------------------------------------------------------------
# Endpoints — Core
# -----------------------------------------------------------------------------
@app.get("/health")
def health(model: str = Query("xgb", description="xgb | rf | logreg")):
    try:
        _, thr, flip = get_model_and_thr(model)
        return {
            "status": "ok",
            "model": model,
            "threshold": float(thr),
            "legacy_flip_probas": bool(flip),
            "db_logging": True,
            "db_connected": bool(_db_ok),
            "db_backend": _db_backend(),
            "features": FEATURE_COLS,
            "modules": {
                "agents_loaded": multi_agent_plan is not None,
                "agents_error": _import_errors.get("agents"),
                "whatif_loaded": simulate_whatif is not None,
                "whatif_error": _import_errors.get("digital_twin"),
                "counterfactual_loaded": counterfactual is not None,
                "counterfactual_error": _import_errors.get("counterfactuals"),
                "similarity_loaded": knn_similar is not None,
                "similarity_error": _import_errors.get("similarity"),
            },
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "db_connected": bool(_db_ok)}

@app.get("/ready")
def ready():
    return {
        "ok": True,
        "agents": bool(multi_agent_plan),
        "whatif": bool(simulate_whatif),
        "counterfactual": bool(counterfactual),
        "similar": bool(knn_similar),
        "errors": {
            "agents": _import_errors.get("agents"),
            "digital_twin": _import_errors.get("digital_twin"),
            "counterfactuals": _import_errors.get("counterfactuals"),
            "similarity": _import_errors.get("similarity"),
        },
    }


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save inputs+predictions to DB"),
):
    df = pd.DataFrame([r.dict() for r in req.rows])
    res_rows = predict_core(df, model).to_dict(orient="records")

    if save and (engine is not None):
        try:
            with engine.begin() as conn:
                for row_in, row_out in zip(req.rows, res_rows):
                    input_id = _save_user_input(conn, row_in.dict())
                    _save_inference(conn, input_id, row_out["model_used"], row_out["threshold_used"], row_out)
        except Exception as e:
            print(f"[warn] batch logging failed: {e}")

    return {"predictions": res_rows}

# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
@app.post("/explain", response_model=ExplainResponse)
def explain(item: PatientFeatures, model: str = Query("xgb")):
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
        print("[/explain] error:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"explain_failed: {str(e)}", "model": model}
        )

# -----------------------------------------------------------------------------
# Digital Twin / What-If
# -----------------------------------------------------------------------------
@app.post("/whatif")
def whatif_api(req: WhatIfRequest):
    """
    Run a single what-if or a grid sweep:
    body: { base: PatientFeatures, deltas: {...}, model: "xgb", grid: { "systolicbp":[120,130,140], ... } }
    """
    try:
        if simulate_whatif is None:
            detail = _import_errors.get("digital_twin") or "simulate_whatif not available"
            return JSONResponse(status_code=503, content={"detail": detail})
        return simulate_whatif(req)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"whatif_failed: {e}"})

# -----------------------------------------------------------------------------
# Counterfactuals
# -----------------------------------------------------------------------------
@app.post("/counterfactual")
def counterfactual_api(req: CFRequest):
    """
    Find actionable changes to reduce risk below a target (greedy or DiCE).
    body: { base: PatientFeatures, target_prob: 0.2, model: "xgb", method: "auto" }
    """
    try:
        if counterfactual is None:
            detail = _import_errors.get("counterfactuals") or "counterfactual not available"
            return JSONResponse(status_code=503, content={"detail": detail})
        return counterfactual(req)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"counterfactual_failed: {e}"})

# -----------------------------------------------------------------------------
# Patient Similarity / Nearest Neighbors
# -----------------------------------------------------------------------------
@app.post("/similar")
def similar_api(
    req: SimilarRequest,
    k: int = Query(5, ge=1, le=50),
):
    """
    Returns top-k similar rows (euclidean). For production, back this with a DB/feature store query.
    """
    try:
        if knn_similar is None:
            detail = _import_errors.get("similarity") or "knn_similar not available"
            return JSONResponse(status_code=503, content={"detail": detail})
        base = req.base.dict()
        cohort_list = [r.dict() for r in (req.cohort or [])]
        return knn_similar(base, cohort_list, k=k)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"similar_failed: {e}"})

# -----------------------------------------------------------------------------
# Multi-Agent Care Plan
# -----------------------------------------------------------------------------
@app.post("/agents/plan")
def agents_plan_api(summary: dict = Body(...)):
    """
    LLM-powered care plan from a dict summary {metrics, flags, stage, etc}.
    """
    try:
        if multi_agent_plan is None:
            detail = _import_errors.get("agents") or "multi_agent_plan not available"
            return JSONResponse(status_code=503, content={"detail": detail})
        out = multi_agent_plan(summary)
        if isinstance(out, dict) and "error" in out:
            return JSONResponse(status_code=503, content=out)
        return out
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"agents_failed: {e}"})

# -----------------------------------------------------------------------------
# Retrain / Reload
# -----------------------------------------------------------------------------
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
    return  # 204

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
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
    if engine is None or not _db_ok:
        return JSONResponse(
            status_code=503,
            content={"detail": "DB not connected; cannot read inference_log."},
        )
    with engine.begin() as c:
        rows = c.execute(text("""
            SELECT il.created_at, il.model_used, il.prediction, il.prob_ckd, il.prob_non_ckd, il.threshold_used
            FROM inference_log il
            ORDER BY il.created_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"rows": [dict(r) for r in rows]}

# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
