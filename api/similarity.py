# api/similarity.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
from .app import FEATURE_COLS

def build_matrix(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS].astype(float)

def knn_similar(base: Dict[str, float], cohort_rows: List[Dict[str, float]], k: int = 5):
    if not cohort_rows:
        return {"neighbors": []}
    X = build_matrix(cohort_rows)
    x = build_matrix([base])
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), metric="euclidean").fit(X.values)
    dists, idxs = nbrs.kneighbors(x.values)
    out = []
    for d, i in zip(dists[0].tolist(), idxs[0].tolist()):
        row = dict(cohort_rows[i])
        row["_distance"] = float(d)
        out.append(row)
    return {"neighbors": out}
