# api/counterfactuals.py
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from joblib import load

# Try to import DiCE; it's optional
try:
    import dice_ml
    from dice_ml import Dice
    _DICE_AVAILABLE = True
except Exception:
    _DICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model registry (kept local to avoid circular import with api.app)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]     # repo root
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (ROOT_DIR / "models"))

REGISTRY = {
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

# ---------------------------------------------------------------------------
# Utilities: loading, validation, derived feature computation
# ---------------------------------------------------------------------------
def _load_artifacts(model_key: str):
    mk = model_key.lower()
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

def _derive_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute derived & flag fields to keep the record consistent."""
    # Required base fields (assume present)
    sysbp = float(row["systolicbp"])
    diabp = float(row["diastolicbp"])
    bun = float(row["bunlevels"])
    crea = float(row["serumcreatinine"])
    gfr = float(row["gfr"])
    acr = float(row["acr"])
    na = float(row["serumelectrolytessodium"])
    k = float(row["serumelectrolytespotassium"])
    hb = float(row["hemoglobinlevels"])

    # Derived
    row["pulsepressure"] = sysbp - diabp
    row["ureacreatinineratio"] = float(bun) / (float(crea) + 1e-6)

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

    # Keep Na within plausible range
    row["serumelectrolytessodium"] = float(np.clip(na, 110.0, 170.0))
    row["serumelectrolytespotassium"] = float(np.clip(k, 2.0, 7.5))
    return row

def _validate_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing feature: {c}")
    return df[FEATURE_COLS].copy()

def _predict_prob_ckd(model, df: pd.DataFrame, flip: bool) -> np.ndarray:
    proba1 = model.predict_proba(df)[:, 1]
    if flip:
        return 1.0 - proba1
    return proba1

# ---------------------------------------------------------------------------
# Greedy counterfactual search (deterministic, no external deps)
# ---------------------------------------------------------------------------
# Define which features are "actionable" and the direction toward lower risk.
# Each entry: (name, step_size, direction, min, max)
#   direction: +1 means increasing should reduce risk; -1 means decreasing helps; 0 means move toward target
_ACTIONABLES = [
    ("systolicbp",               2.0,  -1,  90.0, 200.0),
    ("diastolicbp",              2.0,  -1,  50.0, 120.0),
    ("hba1c",                    0.2,  -1,   4.5,  12.0),
    ("acr",                     10.0,  -1,   0.0, 3000.0),
    ("bunlevels",                1.0,  -1,   3.0,  90.0),
    ("serumcreatinine",          0.1,  -1,   0.4,  8.0),
    ("serumelectrolytespotassium", 0.1,  0,   3.0,  6.5),   # toward ~4.5
    ("serumelectrolytessodium",   0.5,  0, 130.0, 150.0),   # toward ~140
    ("hemoglobinlevels",         0.2,  +1,   8.0,  16.0),   # higher generally reduces anemia flag
    ("gfr",                      1.0,  +1,  10.0, 110.0),
]

def _toward(current: float, step: float, direction: int, target: float | None = None) -> float:
    if direction == -1:
        return current - step
    if direction == +1:
        return current + step
    # toward a target
    if target is None:
        return current
    return current + np.sign(target - current) * step

def _apply_clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))

def _greedy_counterfactual(
    model, flip: bool, base: Dict[str, Any],
    target_prob: float,
    max_iters: int = 60,
    min_improve: float = 1e-4,
) -> Dict[str, Any]:
    """
    Simple hill-climbing: at each iteration, try a small move on each actionable
    feature, pick the one that lowers prob_ckd the most, apply it, and repeat.
    """
    # Work on a local copy and ensure derived consistency
    cur = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in base.items()}
    cur = _derive_fields(cur)

    # Helper to evaluate prob
    def prob_of(d: Dict[str, Any]) -> float:
        df = _validate_and_order(pd.DataFrame([d]))
        return float(_predict_prob_ckd(model, df, flip)[0])

    p0 = prob_of(cur)
    steps = []
    last_p = p0
    converged = False

    # Targets for "toward" direction features
    target_k = 4.5
    target_na = 140.0

    for it in range(max_iters):
        best_feat = None
        best_delta = 0.0
        best_change = 0.0
        best_state = None
        base_p = last_p

        for (feat, step, direction, lo, hi) in _ACTIONABLES:
            if feat not in cur:
                continue
            old_val = float(cur[feat])
            tgt = None
            if direction == 0:
                tgt = target_k if feat == "serumelectrolytespotassium" else (target_na if feat == "serumelectrolytessodium" else old_val)
            trial_val = _apply_clip(_toward(old_val, step, direction, tgt), lo, hi)
            if abs(trial_val - old_val) < 1e-9:
                continue

            trial = cur.copy()
            trial[feat] = trial_val
            trial = _derive_fields(trial)
            p_trial = prob_of(trial)
            change = base_p - p_trial  # positive means improvement

            if change > best_change + 1e-9:
                best_change = change
                best_feat = feat
                best_delta = float(trial_val - old_val)
                best_state = trial

        if best_feat is None or best_state is None:
            break  # no improving move found

        # Apply best move
        cur = best_state
        last_p = prob_of(cur)  # recompute for stored state
        steps.append({
            "feature": best_feat,
            "delta": best_delta,
            "new_value": float(cur[best_feat]),
            "prob_ckd": last_p,
        })

        # stopping conditions
        if last_p <= target_prob:
            converged = True
            break
        if best_change < min_improve:
            break

    return {
        "method": "greedy",
        "target_prob": float(target_prob),
        "initial_prob": float(p0),
        "final_prob": float(last_p),
        "converged": bool(converged),
        "iterations": len(steps),
        "steps": steps,
        "result": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cur.items()},
    }

# ---------------------------------------------------------------------------
# DiCE-ML based counterfactual (optional)
# ---------------------------------------------------------------------------
def _dice_counterfactual(
    model, base: Dict[str, Any], target_prob: float, total_cfs: int = 3
) -> Dict[str, Any]:
    """
    Requires:
      - dice-ml installed
      - CF_DICE_DATA_CSV env var pointing to a CSV with FEATURE_COLS
    """
    if not _DICE_AVAILABLE:
        raise RuntimeError("DiCE not available")

    data_csv = os.getenv("CF_DICE_DATA_CSV")
    if not data_csv or not Path(data_csv).exists():
        raise RuntimeError("Set CF_DICE_DATA_CSV to a CSV with training-like rows for DiCE.")

    df = pd.read_csv(data_csv)
    # Minimal schema: all features numeric; outcome is "ckd" (we'll create on the fly using model)
    # Create a temporary outcome column by scoring the dataset
    def score_batch(df_in: pd.DataFrame) -> np.ndarray:
        X = _validate_and_order(df_in.copy())
        # Assume no flip here (DiCE uses a generic predictor wrapper); prob_ckd ~ [:,1]
        return model.predict_proba(X)[:, 1]

    df = df.copy()
    df["ckd"] = (score_batch(df) >= 0.5).astype(int)  # dummy label for DiCE’s data interface

    # Build DiCE data and model objects
    d = dice_ml.Data(dataframe=df, continuous_features=FEATURE_COLS, outcome_name="ckd")
    m = dice_ml.Model(model=model, backend="sklearn")
    dice = Dice(d, m, method="random")  # "random" or "kdtree" for continuous features

    query_instance = pd.DataFrame([base])[FEATURE_COLS]
    # Aim for target_prob by asking for the opposite class when current prob > target
    current_prob = float(model.predict_proba(query_instance)[:, 1][0])
    desired_class = 0 if current_prob > target_prob else 1

    cfs = dice.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_class=desired_class
    )
    # Parse DiCE result dataframe
    df_cfs = cfs.cf_examples_list[0].final_cfs_df
    cfs_list = df_cfs[FEATURE_COLS].to_dict(orient="records")

    # score them to include prob_ckd
    scored = []
    for r in cfs_list:
        r2 = _derive_fields(r.copy())
        p = float(model.predict_proba(pd.DataFrame([r2])[FEATURE_COLS])[:, 1][0])
        scored.append({"candidate": r2, "prob_ckd": p})

    scored = sorted(scored, key=lambda x: x["prob_ckd"])
    return {
        "method": "dice",
        "target_prob": float(target_prob),
        "initial_prob": current_prob,
        "final_prob": float(scored[0]["prob_ckd"]) if scored else current_prob,
        "candidates": scored,
        "best": scored[0] if scored else None,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def counterfactual(req) -> Dict[str, Any]:
    """
    Entry point called by FastAPI layer:
      body: { base: PatientFeatures, target_prob: 0.2, model: "xgb", method: "auto"|"greedy" }
    """
    payload = req if isinstance(req, dict) else req.dict()
    base = payload.get("base", {})
    model_key = str(payload.get("model", "xgb")).lower()
    target_prob = float(payload.get("target_prob", 0.2))
    method = str(payload.get("method", "auto")).lower()

    # Sanity: ensure required keys exist in base
    missing = [c for c in FEATURE_COLS if c not in base]
    if missing:
        raise ValueError(f"Missing features in base: {missing}")

    model, thr, flip = _load_artifacts(model_key)

    # Ensure derived fields are consistent in base
    base = _derive_fields({k: base[k] for k in base})

    if method == "greedy":
        return _greedy_counterfactual(model, flip, base, target_prob)

    # AUTO: DiCE if available + configured, else greedy
    if _DICE_AVAILABLE and os.getenv("CF_DICE_DATA_CSV"):
        try:
            return _dice_counterfactual(model, base, target_prob, total_cfs=3)
        except Exception as e:
            # fall back but include note
            greedy = _greedy_counterfactual(model, flip, base, target_prob)
            greedy["note"] = f"DiCE fallback: {e}"
            return greedy
    else:
        return _greedy_counterfactual(model, flip, base, target_prob)
