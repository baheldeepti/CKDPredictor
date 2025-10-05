# api/digital_twin.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable

import numpy as np
import pandas as pd
from joblib import load

# --------------------------------------------------------------------------------------
# Local registry & feature schema (avoids circular import with api.app)
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]     # repo root
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (ROOT_DIR / "models"))

REGISTRY: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "model": MODEL_DIR / "xgb_ckd.joblib",
        "thr":   MODEL_DIR / "xgb_ckd_threshold.json",
        "flip_probas": False,   # set True only if your old xgb has flipped outputs
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

# --------------------------------------------------------------------------------------
# Helpers (standalone)
# --------------------------------------------------------------------------------------
def _load_artifacts(model_key: str):
    mk = str(model_key).lower()
    if mk not in REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Use one of {list(REGISTRY)}")
    p = REGISTRY[mk]
    if not p["model"].exists():
        raise FileNotFoundError(f"Model file missing for '{mk}': {p['model']}")
    model = load(p["model"])
    thr = 0.5
    if p["thr"].exists():
        try:
            with open(p["thr"]) as f:
                thr = float(json.load(f)["threshold"])
        except Exception:
            pass
    flip = bool(p.get("flip_probas", False))
    return model, float(thr), flip

def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing feature: {c}")
    return df[FEATURE_COLS].copy()

def _rederive_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep derived values consistent when inputs change.
    Requires that the base record already contained all FEATURE_COLS once.
    """
    out = dict(row)
    # Derived
    if "systolicbp" in out and "diastolicbp" in out:
        out["pulsepressure"] = float(out["systolicbp"]) - float(out["diastolicbp"])
    if "bunlevels" in out and "serumcreatinine" in out:
        out["ureacreatinineratio"] = float(out["bunlevels"]) / (float(out["serumcreatinine"]) + 1e-6)

    # If stage/alb/flags were present, keep them plausible with simple rules
    try:
        gfr = float(out.get("gfr", 0.0))
        if gfr >= 90: stage = 1
        elif gfr >= 60: stage = 2
        elif gfr >= 45: stage = 3
        elif gfr >= 30: stage = 3
        elif gfr >= 15: stage = 4
        else: stage = 5
        out["ckdstage"] = int(stage)
    except Exception:
        pass

    try:
        acr = float(out.get("acr", 0.0))
        if acr < 30: alb = 1
        elif acr <= 300: alb = 2
        else: alb = 3
        out["albuminuriacat"] = int(alb)
    except Exception:
        pass

    try:
        sbp = float(out.get("systolicbp", 0.0))
        dbp = float(out.get("diastolicbp", 0.0))
        k   = float(out.get("serumelectrolytespotassium", 0.0))
        hb  = float(out.get("hemoglobinlevels", 0.0))
        out["bp_risk"] = 1 if (sbp >= 130 or dbp >= 80) else 0
        out["hyperkalemiaflag"] = 1 if k >= 5.5 else 0
        out["anemiaflag"] = 1 if hb < 12.0 else 0
    except Exception:
        pass

    return out

def _simulate_once(base: Dict[str, float], deltas: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in base.items()}
    for k, dv in (deltas or {}).items():
        if k in out:
            out[k] = float(out[k]) + float(dv)
    out = _rederive_fields(out)
    return out

def _predict_proba_dict(row: Dict[str, float], model_key: str) -> Tuple[float, float]:
    df = pd.DataFrame([row])
    df = _validate_and_order(df)
    mdl, _, flip = _load_artifacts(model_key)
    p1 = mdl.predict_proba(df)[:, 1]  # model's positive-class prob
    if flip:
        # legacy flip: [:,1] == P(non-CKD)
        prob_non_ckd = float(p1[0])
        prob_ckd = 1.0 - prob_non_ckd
    else:
        prob_ckd = float(p1[0])
    return prob_ckd, 1.0 - prob_ckd

def _cartesian(arrays: List[List[float]]) -> Iterable[List[float]]:
    if not arrays:
        yield []
        return
    head, *tail = arrays
    for h in head:
        for rest in _cartesian(tail):
            yield [h] + rest

# --------------------------------------------------------------------------------------
# Public entry (called from FastAPI layer). Accepts pydantic model or dict-like.
# --------------------------------------------------------------------------------------
def simulate_whatif(req) -> Dict[str, Any]:
    """
    Accepts either:
      - a Pydantic WhatIfRequest-like object (with .base, .deltas, .model, .grid)
      - a plain dict with the same keys.

    Returns:
      Single-run:
        { "mode": "single", "rows": [row], "probs": [{"prob_ckd": x, "prob_non_ckd": y}] }
      Grid:
        { "mode": "grid", "features": [...], "rows": [...], "probs": [{"prob_ckd":...}, ...] }
    """
    # Tolerate both Pydantic and dict inputs
    payload: Dict[str, Any]
    if hasattr(req, "dict"):
        payload = req.dict()
    else:
        payload = dict(req)

    base: Dict[str, float] = payload.get("base", {}) or {}
    deltas: Dict[str, float] = payload.get("deltas", {}) or {}
    model_key: str = str(payload.get("model", "xgb")).lower()
    grid: Optional[Dict[str, List[float]]] = payload.get("grid")

    # Single-shot case
    if not grid:
        row = _simulate_once(base, deltas)
        p_ckd, p_non = _predict_proba_dict(row
