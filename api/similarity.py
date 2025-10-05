# api/similarity.py
from __future__ import annotations
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Keep FEATURE_COLS local to avoid importing from api.app (prevents circular import)
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

def _derive_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute derived metrics and flags for consistency."""
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

    # Derived fields
    row["pulsepressure"] = sysbp - diabp
    row["ureacreatinineratio"] = bun / (crea + 1e-6)

    # CKD stage (coarse)
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

def _ensure_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS].astype(float)

def build_matrix(rows: List[Dict[str, float]]) -> pd.DataFrame:
    """Turn list of dicts into a numeric matrix with derived fields and correct column order."""
    fixed = [_derive_fields(dict(r)) for r in rows]
    df = pd.DataFrame(fixed)
    return _ensure_all_columns(df)

def knn_similar(base: Dict[str, float], cohort_rows: List[Dict[str, float]], k: int = 5):
    """
    Returns top-k nearest neighbors in cohort_rows to base using Euclidean distance
    over FEATURE_COLS. Output mirrors your UI expectations: {"neighbors": [row+_distance]}.
    """
    if not cohort_rows:
        return {"neighbors": []}

    X = build_matrix(cohort_rows)
    x = build_matrix([base])

    n_neighbors = int(min(max(k, 1), len(X)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X.values)
    dists, idxs = nbrs.kneighbors(x.values)

    out = []
    for d, i in zip(dists[0].tolist(), idxs[0].tolist()):
        row = dict(cohort_rows[i])  # return original row fields plus distance
        row["_distance"] = float(d)
        out.append(row)
    return {"neighbors": out}
