# api/metrics.py
from pathlib import Path
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/metrics", tags=["Metrics"])
MODEL_DIR = Path("models")

MODEL_META = {
    "logreg": MODEL_DIR / "logreg_ckd_meta.json",
    "rf": MODEL_DIR / "rf_ckd_meta.json",
    "xgb": MODEL_DIR / "xgb_ckd_meta.json",
}

@router.get("/confusion_matrix")
def get_confusion_matrices():
    output = {}

    for model, path in MODEL_META.items():
        if not path.exists():
            continue

        meta = json.loads(path.read_text())
        cm = meta.get("metrics", {}).get("test", {}).get("confusion_matrix")
        if not cm:
            continue

        tn, fp = cm[0]
        fn, tp = cm[1]

        output[model] = {
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "explanation": {
                "true_negative": "Correctly reassured",
                "false_positive": "Flagged but later found okay",
                "false_negative": "Missed during screening",
                "true_positive": "Correctly flagged for follow-up",
            }
        }

    if not output:
        raise HTTPException(status_code=404, detail="No metrics available")

    return output
