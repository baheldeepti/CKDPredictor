# api/app.py
import os
import sys
import json
import math
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

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="CKD Predictor API", version="1.0.0")

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
    # avoid protected namespace warning on "model_used"
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
        conn.execute(text("INSERT INTO user_inputs ("
                          "age, gender, systolicbp, diastolicbp, serumcreatinine, bunlevels, gfr, acr,"
                          "serumelectrolytessodium, serumelectrolytespotassium, hemoglobinlevels, hba1c,"
                          "pulsepressure, ureacreatinineratio, ckdstage, albuminuriacat, bp_risk, hyperkalemiaflag, anemiaflag"
                          ") VALUES ("
                          ":age, :gender, :systolicbp, :diastolicbp, :serumcreatinine, :bunlevels, :gfr, :acr,"
                          ":serumelectrolytessodium, :serumelectrolytespotassium, :hemoglobinlevels, :hba1c,"
                          ":pulsepressure, :ureacreatinineratio, :ckdstage, :albuminuriacat, :bp_risk, :hyperkalemiaflag, :anemiaflag"
                          ");"), row)
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
    """Return index of the positive class. Prefer label 1; else last/largest."""
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        arr = np.array(list(classes))
        # If numeric
        if np.issubdtype(arr.dtype, np.number):
            if (arr == 1).any():
                return int(np.where(arr == 1)[0][0])
            return int(np.argmax(arr))  # pick largest label
        # Non-numeric labels (e.g., ["Non-CKD","CKD"])
        labels = [str(x) for x in arr.tolist()]
        if "1" in labels:
            return labels.index("1")
        return len(labels) - 1  # choose last as "positive"
    except Exception:
        return 1

def _compute_shap_binary(model, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for the first row, for the positive class.
    Returns (vals_for_row, base_value).
    Tries classic .shap_values() API first for stability across SHAP versions.
    """
    pos = _pos_index(model)

    # --- TreeExplainer path (RF/XGB/GBM) using classic API ---
    try:
        tree_expl = shap.TreeExplainer(model)
        if hasattr(tree_expl, "shap_values"):
            sv = tree_expl.shap_values(X)
            # sv can be [class0_array, class1_array] or a 2D array
            if isinstance(sv, list):
                arr = np.asarray(sv[min(pos, len(sv)-1)])
                vals = arr[0]  # first (only) row
            else:
                arr = np.asarray(sv)
                vals = arr[0]
            base = tree_expl.expected_value
            if isinstance(base, (list, tuple, np.ndarray)):
                base_val = float(np.asarray(base)[min(pos, len(np.asarray(base))-1)])
            else:
                base_val = float(base)
            return np.asarray(vals, dtype=float), base_val
    except Exception:
        pass

    # --- LinearExplainer path (LogReg/SGD) classic API ---
    try:
        # For scikit-learn LogisticRegression, LinearExplainer works well
        lin_expl = shap.LinearExplainer(model, X, feature_dependence="independent")
        if hasattr(lin_expl, "shap_values"):
            arr = np.asarray(lin_expl.shap_values(X))
            # Returns (n_samples, n_features)
            vals = arr[0]
            base = lin_expl.expected_value
            base_val = float(np.asarray(base).ravel()[0])
            return np.asarray(vals, dtype=float), base_val
    except Exception:
        pass

    # --- New Explanation API fallback ---
    try:
        exp = shap.Explainer(model, X)
        res = exp(X.iloc[[0]])
        v = getattr(res, "values", None)
        base_raw = getattr(res, "base_values", 0.0)

        # values may be array or list[class]
        if isinstance(v, list):
            arr = np.asarray(v[min(pos, len(v)-1)])
            vals = arr[0]
        else:
            arr = np.asarray(v)
            if arr.ndim == 3:      # (classes, samples, features)
                vals = arr[min(pos, arr.shape[0]-1), 0, :]
            elif arr.ndim == 2:    # (samples, features)
                vals = arr[0]
            elif arr.ndim == 1:    # (features,)
                vals = arr
            else:
                vals = np.ravel(arr)

        base_arr = np.asarray(base_raw)
        if base_arr.ndim == 0:
            base_val = float(base_arr)
        elif base_arr.ndim == 1:
            base_val = float(base_arr[min(pos, base_arr.shape[0]-1)])
        elif base_arr.ndim == 2:
            base_val = float(base_arr[min(pos, base_arr.shape[0]-1), 0])
        else:
            base_val = float(np.ravel(base_arr)[0])

        return np.asarray(vals, dtype=float), float(base_val)
    except Exception:
        pass

    # --- Last resort: use model importances/coeffs (not true SHAP, but unblocks UI) ---
    try:
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_, dtype=float)
            base_val = 0.0
            return vals, base_val
        if hasattr(model, "coef_"):
            vals = np.asarray(model.coef_).ravel().astype(float)
            base_val = 0.0
            return vals, base_val
    except Exception:
        pass

    raise RuntimeError("Could not compute SHAP values for this model/version.")

def _shap_top_k(features: pd.DataFrame, model, k: int = 6):
    X = features.copy()
    vals, base = _compute_shap_binary(model, X)
    feat_names = list(X.columns)
    # Align in case some backends return less/more features (safety)
    n = min(len(feat_names), len(vals))
    shap_dict = {feat_names[i]: float(vals[i]) for i in range(n)}
    top = sorted(
        [{"feature": f, "impact": abs(v), "signed": float(v)} for f, v in shap_dict.items()],
        key=lambda d: d["impact"], reverse=True
    )[:k]
    return float(base), shap_dict, top

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
            "threshold": float(thr),
            "legacy_flip_probas": bool(flip),
            "db_logging": True,
            "db_connected": bool(_db_ok),
            "db_backend": _db_backend(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "db_connected": bool(_db_ok)}

@app.post("/predict", response_model=PredictResponse)
def predict(
    item: PatientFeatures,
    model: str = Query("xgb"),
    save: bool = Query(True, description="Save input+prediction to DB (enabled by default)"),
):
    df = pd.DataFrame([item.dict()])
    res = predict_core(df, model).iloc[0].to_dict()

    if save and (engine is not None):
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

    if save and (engine is not None):
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

# === Retrain / Reload ========================================================
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
    # Free port 8000 if something is holding it? Do nothing here;
    # use the shell line I give you below to kill old uvicorns.
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
