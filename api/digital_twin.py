# api/digital_twin.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable

import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Local registry & schema (avoid importing api.app to prevent circular import)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]     # repo root
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (ROOT_DIR / "models"))

REGISTRY: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "model": MODEL_DIR / "xgb_ckd.joblib",
        "thr":   MODEL_DIR / "xgb_ckd_threshold.json",
        "flip_probas": False,   # set True only if your old xgb had flipped outputs
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

def _load_artifacts(model_key: str) -> Tuple[Any, float, bool]:
    mk = model_key.lower()
    if mk not in REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Use one of {list(REGISTRY)}")
    p = REGISTRY[mk]
    if not p["model"].exists():
        raise FileNotFoundError(f"Model file missing for '{mk}': {p['model']}")
    mdl = load(p["model"])
    thr = 0.5
    if p["thr"].exists():
        try:
            with open(p["thr"]) as f:
                thr = float(json.load(f)["threshold"])
        except Exception:
            pass
    flip = bool(p.get("flip_probas", False))
    return mdl, float(thr), flip

def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing feature: {c}")
    return df[FEATURE_COLS].copy()

# ---------------------------------------------------------------------------
# Derived fields & flags (kept consistent with other modules)
# ---------------------------------------------------------------------------
def _derive_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    # coerce numerics where needed
    def f(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    sysbp = f(row.get("systolicbp"))
    diabp = f(row.get("diastolicbp"))
    bun   = f(row.get("bunlevels"))
    crea  = f(row.get("serumcreatinine"))
    gfr   = f(row.get("gfr"))
    acr   = f(row.get("acr"))
    na    = f(row.get("serumelectrolytessodium"))
    k     = f(row.get("serumelectrolytespotassium"))
    hb    = f(row.get("hemoglobinlevels"))

    # Derived
    row["pulsepressure"] = sysbp - diabp
    row["ureacreatinineratio"] = bun / (crea + 1e-6)

    # Stage (coarse)
    if gfr >= 90: stage = 1
    elif gfr >= 60: stage = 2
    elif gfr >= 45: stage = 3
    elif gfr >= 30: stage = 3
    elif gfr >= 15: stage = 4
    else: stage = 5
    row["ckdstage"] = int(stage)

    # Albuminuria category
    if acr < 30: alb = 1
    elif acr <= 300: alb = 2
    else: alb = 3
    row["albuminuriacat"] = int(alb)

    # Flags
    row["bp_risk"] = 1 if (sysbp >= 130 or diabp >= 80) else 0
    row["hyperkalemiaflag"] = 1 if k >= 5.5 else 0
    row["anemiaflag"] = 1 if hb < 12.0 else 0

    # Keep plausible ranges on electrolytes
    row["serumelectrolytessodium"] = float(np.clip(na, 110.0, 170.0))
    row["serumelectrolytespotassium"] = float(np.clip(k, 2.0, 7.5))
    return row

# ---------------------------------------------------------------------------
# Pydantic request (mirrors app schema but local to avoid circular import)
# ---------------------------------------------------------------------------
class WhatIfRequest(BaseModel):
    base: Dict[str, float] = Field(..., description="Baseline PatientFeatures dict")
    deltas: Optional[Dict[str, float]] = Field(default=None, description="Feature -> additive delta")
    model: str = "xgb"
    grid: Optional[Dict[str, List[float]]] = Field(default=None, description="Optional grid sweep")

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _simulate_once(base: Dict[str, float], deltas: Optional[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, Any] = dict(base)
    for k, dv in (deltas or {}).items():
        if k in out:
            out[k] = float(out[k]) + float(dv)
    # re-derive for consistency
    out = _derive_fields(out)
    return out

def _predict_probs(row: Dict[str, Any], model_key: str) -> Tuple[float, float]:
    mdl, _, flip = _load_artifacts(model_key)
    df = _validate_and_order(pd.DataFrame([row]))
    p1 = mdl.predict_proba(df)[:, 1]  # standard P(class=1)
    prob_ckd = float(1.0 - p1[0]) if flip else float(p1[0])
    return prob_ckd, float(1.0 - prob_ckd)

def _cartesian(arrays: List[List[float]]) -> Iterable[List[float]]:
    if not arrays:
        yield []
        return
    head, *tail = arrays
    for h in head:
        for rest in _cartesian(tail):
            yield [h] + rest

# ---------------------------------------------------------------------------
# Public API for FastAPI layer
# ---------------------------------------------------------------------------
def simulate_whatif(req: WhatIfRequest | Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single what-if (base + deltas) OR a grid sweep if req.grid is provided.
    Accepts either a Pydantic WhatIfRequest or a raw dict with matching keys.
    """
    payload: Dict[str, Any] = req if isinstance(req, dict) else req.dict()
    base: Dict[str, float] = payload.get("base", {}) or {}
    deltas: Optional[Dict[str, float]] = payload.get("deltas") or None
    model_key: str = str(payload.get("model", "xgb")).lower()
    grid: Optional[Dict[str, List[float]]] = payload.get("grid") or None

    # Ensure base has required keys — this function doesn’t invent missing features
    missing = [c for c in FEATURE_COLS if c not in base]
    if missing:
        raise ValueError(f"Missing features in base: {missing}")

    if not grid:
        row = _simulate_once(base, deltas)
        p_ckd, p_non = _predict_probs(row, model_key)
        return {
            "mode": "single",
            "rows": [row],
            "probs": [{"prob_ckd": p_ckd, "prob_non_ckd": p_non}],
            "model": model_key,
        }

    feats = list(grid.keys())
    grids = [grid[f] for f in feats]
    rows: List[Dict[str, float]] = []
    probs: List[Dict[str, float]] = []

    for values in _cartesian(grids):
        # interpret grid value as ABSOLUTE target for the feature, adjust delta accordingly
        deltas_abs = {f: (float(v) - float(base.get(f, 0.0))) for f, v in zip(feats, values)}
        row = _simulate_once(base, deltas_abs)
        p_ckd, p_non = _predict_probs(row, model_key)
        rows.append(row)
        probs.append({"prob_ckd": p_ckd, "prob_non_ckd": p_non})

    return {
        "mode": "grid",
        "features": feats,
        "rows": rows,
        "probs": probs,
        "model": model_key,
    }
