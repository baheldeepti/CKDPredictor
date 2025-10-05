# api/digital_twin.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pydantic import BaseModel
from .app import FEATURE_COLS, get_model_and_thr, _validate_and_order

# ---- What-if simulation -----------------------------------------------------

class WhatIfRequest(BaseModel):
    base: Dict[str, float]        # baseline PatientFeatures
    deltas: Dict[str, float]      # feature -> additive delta (e.g., {"systolicbp": -10})
    model: str = "xgb"
    grid: Dict[str, List[float]] | None = None  # optional: sweep specific features

def simulate_once(base: Dict[str, float], deltas: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, dv in deltas.items():
        if k in out:
            out[k] = float(out[k]) + float(dv)
    # recompute derived fields if present
    if "systolicbp" in out and "diastolicbp" in out:
        out["pulsepressure"] = float(out["systolicbp"]) - float(out["diastolicbp"])
    if "bunlevels" in out and "serumcreatinine" in out:
        out["ureacreatinineratio"] = float(out["bunlevels"]) / (float(out["serumcreatinine"]) + 1e-6)
    return out

def predict_proba_dict(row: Dict[str, float], model_key: str) -> Tuple[float, float]:
    df = pd.DataFrame([row])
    df = _validate_and_order(df)
    mdl, thr, flip = get_model_and_thr(model_key)
    p1 = mdl.predict_proba(df)[:, 1]  # standard P(CKD) unless flip
    if flip:
        prob_non = float(p1[0])
        prob_ckd = 1.0 - prob_non
    else:
        prob_ckd = float(p1[0])
    return prob_ckd, 1 - prob_ckd

def simulate_whatif(req: WhatIfRequest) -> Dict:
    """Run single-shot (base + deltas) OR grid sweep if req.grid provided."""
    base = req.base
    model = req.model

    if not req.grid:
        row = simulate_once(base, req.deltas)
        p_ckd, p_non = predict_proba_dict(row, model)
        return {"mode": "single", "rows": [row], "probs": [{"prob_ckd": p_ckd, "prob_non_ckd": p_non}]}

    # grid sweep over selected features
    feats = list(req.grid.keys())
    grids = [req.grid[f] for f in feats]
    rows, probs = [], []
    for values in _cartesian(grids):
        deltas = {f: (float(v) - float(base.get(f, 0.0))) for f, v in zip(feats, values)}
        row = simulate_once(base, deltas)
        p_ckd, p_non = predict_proba_dict(row, model)
        rows.append(row)
        probs.append({"prob_ckd": p_ckd, "prob_non_ckd": p_non})
    return {"mode": "grid", "features": feats, "rows": rows, "probs": probs}

def _cartesian(arrays: List[List[float]]):
    if not arrays:
        yield []
    else:
        head, *tail = arrays
        for h in head:
            for rest in _cartesian(tail):
                yield [h] + rest

# ---- Counterfactuals (greedy + DiCE) ---------------------------------------

class CFRequest(BaseModel):
    base: Dict[str, float]
    target_prob: float = 0.2         # e.g., reduce prob_ckd below 20%
    model: str = "xgb"
    method: str = "auto"             # "auto" | "greedy" | "dice"
    mutable: List[str] | None = None # which features can change
    per_feature_step: float = 0.1
    max_steps: int = 50

DEFAULT_MUTABLE = [
    "systolicbp","diastolicbp","acr","gfr","serumcreatinine","bunlevels",
    "serumelectrolytespotassium","hemoglobinlevels"
]

def cf_greedy(req: CFRequest) -> Dict:
    base = dict(req.base)
    mutable = req.mutable or DEFAULT_MUTABLE
    best = dict(base)
    p_best, _ = predict_proba_dict(best, req.model)

    trace = [{"step": 0, "prob_ckd": p_best, "changes": {}}]
    if p_best <= req.target_prob:
        return {"hit": True, "prob_ckd": p_best, "changes": {}, "trace": trace}

    step = 0
    while step < req.max_steps:
        step += 1
        improved = False
        best_local = None
        for f in mutable:
            # Try nudging positive or negative (domain-naive; works surprisingly well)
            for sign in (-1.0, +1.0):
                trial = dict(best)
                trial[f] = float(trial.get(f, 0.0)) + sign * req.per_feature_step
                # maintain derived
                if "systolicbp" in trial and "diastolicbp" in trial:
                    trial["pulsepressure"] = float(trial["systolicbp"]) - float(trial["diastolicbp"])
                if "bunlevels" in trial and "serumcreatinine" in trial:
                    trial["ureacreatinineratio"] = float(trial["bunlevels"]) / (float(trial["serumcreatinine"]) + 1e-6)
                p_ckd, _ = predict_proba_dict(trial, req.model)
                if p_ckd < p_best - 1e-6:
                    improved = True
                    p_best = p_ckd
                    best_local = (f, sign * req.per_feature_step, trial)
        if improved and best_local:
            f, dv, trial = best_local
            delta_map = {f: dv}
            best = trial
            trace.append({"step": step, "prob_ckd": p_best, "changes": delta_map})
            if p_best <= req.target_prob:
                return {"hit": True, "prob_ckd": p_best, "changes": _diff(base, best), "trace": trace}
        else:
            break
    return {"hit": False, "prob_ckd": p_best, "changes": _diff(base, best), "trace": trace}

def _diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in b:
        if k in a:
            dv = float(b[k]) - float(a[k])
            if abs(dv) > 1e-9:
                out[k] = dv
    return out

def cf_dice(req: CFRequest) -> Dict:
    try:
        import dice_ml
    except Exception as e:
        return {"error": f"DiCE unavailable: {e}"}
    # Minimal DiCE setup (works if your model is a scikit-learn classifier)
    import pandas as pd
    base_df = pd.DataFrame([req.base])
    base_df = base_df[[c for c in FEATURE_COLS if c in base_df.columns]]
    mdl, _, _ = get_model_and_thr(req.model)

    d = dice_ml.Data(dataframe=base_df, continuous_features=list(base_df.columns), outcome_name="prediction")
    m = dice_ml.Model(model=mdl, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    # Generate small number of CFs around target_prob
    cf = exp.generate_counterfactuals(base_df.drop(columns=["prediction"], errors="ignore"), total_CFs=3, desired_class="opposite")
    return {"dice": cf.cf_examples_list[0].final_cfs_df.to_dict(orient="records")}

def counterfactual(req: CFRequest) -> Dict:
    if req.method == "greedy":
        return cf_greedy(req)
    if req.method == "dice":
        return cf_dice(req)
    # auto: try DiCE, fall back to greedy
    r = cf_dice(req)
    if "error" in r or not r.get("dice"):
        return cf_greedy(req)
    return r
