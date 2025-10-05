# api/app.py
import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, Field, ConfigDict

from fastapi import FastAPI, HTTPException, Query, status, BackgroundTasks, Header
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional DB logging (Postgres via DATABASE_URL; otherwise SQLite fallback)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# App (instantiate FIRST) + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="CKD Predictor API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
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

BASE_DIR = Path(__file__).resolve().parent           # .../api
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (BASE_DIR.parent / "models"))

print(f"[boot] MODEL_DIR={MODEL_DIR.resolve()} exists={MODEL_DIR.exists()}")

# If an older XGB model had flipped proba semantics, set True; otherwise False.
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

# -----------------------------------------------------------------------------
# Database: Postgres if DATABASE_URL set; otherwise SQLite fallback (enabled by default)
# -----------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
USE_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "true").lower() in ("1", "true", "yes")

engine: Optional[Engine] = None
db_backend: str = "none"      # "postgres", "sqlite", "none"
db_connected: bool = False
DB_IS_SQLITE: bool = False

def _init_db():
    global engine, db_backend, db_connected, DB_IS_SQLITE

    if DATABASE_URL:
        try:
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            with engine.begin() as c:
                c.execute(text("SELECT 1"))
            db_backend = "postgres"
            db_connected = True
            DB_IS_SQLITE = False
            print("[db] Connected to Postgres.")
            _ensure_tables()
            return
        except Exception as e:
            print(f"[db] Postgres connection failed: {e}")

    if USE_SQLITE_FALLBACK:
        try:
            sqlite_path = MODEL_DIR / "inference.sqlite3"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
            db_backend = "sqlite"
            DB_IS_SQLITE = True
            with engine.begin() as c:
                c.execute(text("SELECT 1"))
            db_connected = True
            print(f"[db] Using SQLite fallback at {sqlite_path}.")
            _ensure_tables()
            return
        except Exception as e:
            print(f"[db] SQLite fallback failed: {e}")

    # If we reach here, DB unavailable
    engine = None
    db_backend = "none"
    db_connected = False
    DB_IS_SQLITE = False
    print("[db] No database connected. Metrics logging disabled.")

def _ensure_tables():
    """
    Create tables if they don't exist.
    Uses Postgres DDL if on Postgres; SQLite-friendly DDL if on SQLite.
    """
    if engine is None:
        return
    with engine.begin() as c:
        if DB_IS_SQLITE:
            # SQLite types & autoincrement semantics
            c.execute(text("""
                CREATE TABLE IF NOT EXISTS user_inputs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  age REAL, gender INTEGER,
                  systolicbp REAL, diastolicbp REAL,
                  serumcreatinine REAL, bunlevels REAL,
                  gfr REAL, acr REAL,
                  serumelectrolytessodium REAL, serumelectrolytespotassium REAL,
                  hemoglobinlevels REAL, hba1c REAL,
                  pulsepressure REAL, ureacreatinineratio REAL,
                  ckdstage INTEGER, albuminuriacat INTEGER,
                  bp_risk INTEGER, hyperkalemiaflag INTEGER, anemiaflag INTEGER
                );
            """))
            c.execute(text("""
                CREATE TABLE IF NOT EXISTS inference_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  input_id INTEGER REFERENCES user_inputs(id) ON DELETE CASCADE,
                  model_used TEXT NOT NULL,
                  threshold_used REAL NOT NULL,
                  prediction INTEGER NOT NULL,
                  prob_ckd REAL NOT NULL,
                  prob_non_ckd REAL NOT NULL
                );
            """))
            # SQLite doesn't use DESC index syntax; still fine without, data is tiny.
        else:
            # Postgres DDL (your original)
            c.execute(text("""
                CREATE TABLE IF NOT EXISTS user_inputs (
                  id BIGSERIAL PRIMARY KEY,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
            c.execute(text("""
                CREATE TABLE IF NOT EXISTS inference_log (
                  id BIGSERIAL PRIMARY KEY,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  input_id BIGINT REFERENCES user_inputs(id) ON DELETE CASCADE,
                  model_used TEXT NOT NULL,
                  threshold_used DOUBLE PRECISION NOT NULL,
                  prediction INT NOT NULL,
                  prob_ckd DOUBLE PRECISION NOT NULL,
                  prob_non_ckd DOUBLE PRECISION NOT NULL
                );
            """))
            c.execute(text("""
                CREATE INDEX IF NOT EXISTS inference_log_created_at_idx
                ON inference_log(created_at DESC);
            """))

_init_db()

# Optional lightweight admin protection
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
def _check_admin(token: Optional[str]):
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
# DB logging helpers
# -----------------------------------------------------------------------------

def _save_user_input(conn, row: dict) -> int:
    if DB_IS_SQLITE:
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
            );
        """)
        res = conn.execute(q, row)
        try:
            # SQLAlchemy 2.x + sqlite
            input_id = res.lastrowid  # type: ignore[attr-defined]
        except Exception:
            input_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        return int(input_id)
    else:
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
        return int(conn.execute(q, row).scalar())

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
# SHAP helper (super defensive)
# -----------------------------------------------------------------------------

def _is_tree_model(model) -> bool:
    name = model.__class__.__name__.lower()
    return ("forest" in name) or ("xgb" in name) or ("gradientboost" in name) or hasattr(model, "feature_importances_")

def _positive_class_index(model) -> int:
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        classes = list(classes)
        return classes.index(1) if 1 in classes else int(np.argmax(classes))
    except Exception:
        return 1

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    """
    Compute SHAP for one row robustly across SHAP versions & model types.
    Returns (base_value, shap_map, top_list).
    """
    import shap  # import here so the API can still run if shap missing at build time

    X = features.copy()
    # Prefer model-specific explainers; avoid forcing model_output="probability" due to shape quirks
    explainer = None

    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)  # class-wise sometimes list, sometimes array
    except Exception:
        explainer = None

    if explainer is None:
        try:
            name = model.__class__.__name__.lower()
            if name.startswith(("logisticregression", "sgdclassifier", "linear", "linearsvc")):
                mask = shap.maskers.Independent(X)
                explainer = shap.LinearExplainer(model, mask)
        except Exception:
            explainer = None

    if explainer is None:
        try:
            explainer = shap.Explainer(model, X)
        except Exception:
            mask = shap.maskers.Independent(X)
            explainer = shap.Explainer(model, mask)

    # Calculate for first row only
    sv = explainer(X.iloc[[0]])

    # --- normalize SHAP values shape ---
    v = getattr(sv, "values", None)
    base_raw = getattr(sv, "base_values", None)

    # If v is a list (per-class), choose positive class if present else last
    chosen_class = None
    if isinstance(v, list):
        ci = _positive_class_index(model)
        ci = ci if ci < len(v) else (len(v) - 1)
        chosen_class = ci
        v_arr = np.array(v[ci])
        vals = v_arr[0] if v_arr.ndim == 2 else v_arr
    else:
        vals_np = np.array(v) if v is not None else None
        if vals_np is None:
            # old API: sv is sequence
            vals_np = np.array(sv[0].values)
        if vals_np.ndim == 1:
            vals = vals_np
        elif vals_np.ndim == 2:
            # (samples, features)
            vals = vals_np[0]
        elif vals_np.ndim == 3:
            # could be (classes, samples, features) OR (1,1,features)
            first_dim = vals_np.shape[0]
            if hasattr(model, "classes_") and first_dim == len(model.classes_):
                ci = _positive_class_index(model)
                ci = ci if ci < first_dim else 0
                chosen_class = ci
                vals = vals_np[ci, 0, :]
            else:
                vals = vals_np[0, 0, :]
        else:
            vals = np.ravel(vals_np)

    # --- normalize base value shape ---
    base: float
    if isinstance(base_raw, list):
        if chosen_class is not None and chosen_class < len(base_raw):
            base = float(base_raw[chosen_class])
        else:
            base = float(base_raw[0])
    else:
        base_arr = np.array(base_raw) if base_raw is not None else np.array([0.0])
        if base_arr.ndim == 0:
            base = float(base_arr)
        elif base_arr.ndim == 1:
            if chosen_class is not None and len(base_arr) > chosen_class:
                base = float(base_arr[chosen_class])
            else:
                base = float(base_arr[0])
        elif base_arr.ndim == 2:
            # (classes, samples) or (1, samples)
            if chosen_class is not None and base_arr.shape[0] > chosen_class:
                base = float(base_arr[chosen_class, 0])
            else:
                base = float(base_arr[0, 0])
        else:
            base = float(np.ravel(base_arr)[0])

    feat_names = list(X.columns)
    shap_map = {feat_names[i]: float(vals[i]) for i in range(len(feat_names))}
    top = sorted(
        [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_map.items()],
        key=lambda d: d["impact"],
        reverse=True
    )[:k]
    return base, shap_map, top

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
            "db_connected": bool(db_connected),
            "db_backend": db_backend,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "db_connected": bool(db_connected), "db_backend": db_backend}

@app.post("/predict", response_model=PredictResponse)
def predict(
    item: PatientFeatures,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save input+prediction to DB"),
):
    df = pd.DataFrame([item.dict()])
    res = predict_core(df, model).iloc[0].to_dict()

    if save and engine is not None and db_connected:
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

    if save and engine is not None and db_connected:
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
    Returns SHAP explanation for a single row:
    {
      "base_value": float,
      "shap_values": {"feature": signed_value, ...},
      "top": [{"feature":"gfr","impact":0.34,"signed":-0.34}, ...]
    }
    """
    try:
        df_in = pd.DataFrame([item.dict()])
        X = _validate_and_order(df_in)

        mdl, _, _ = get_model_and_thr(model)
        base, shap_values, top = _shap_top_k(X, mdl, k=8)
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
    x_admin_token: Optional[str] = Header(None),
):
    _check_admin(x_admin_token)
    if sync:
        _run_retrain_pipeline()
        return {"status": "done", "mode": "sync"}
    else:
        background.add_task(_run_retrain_pipeline)
        return {"status": "started", "mode": "async"}

@app.post("/admin/reload", status_code=status.HTTP_204_NO_CONTENT)
def admin_reload(x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    global _cache
    _cache.clear()
    return  # 204 No Content

# === Metrics ================================================================

@app.get("/metrics/retrain_report")
def retrain_report():
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
    if engine is None or not db_connected:
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
# Entrypoint for `python -m api.app` (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Safe dev server — use uvicorn in prod (Render/Gunicorn)
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("api.app:app", host=host, port=port, reload=reload_flag)
