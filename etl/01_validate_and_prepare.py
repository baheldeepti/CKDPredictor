import os
from pathlib import Path
import numpy as np
import pandas as pd

RAW_CSV = os.getenv("CKD_RAW_CSV", "./data/raw/Chronic_Kidney_Dsease_data.csv")
OUT_PARQUET = os.getenv("CKD_PROCESSED_PARQUET", "./data/processed/ckd_prepared.parquet")

REQUIRED_COLS = [
    "PatientID","Age","Gender","SystolicBP","DiastolicBP","SerumCreatinine","BUNLevels",
    "GFR","ACR","SerumElectrolytesSodium","SerumElectrolytesPotassium","HemoglobinLevels",
    "HbA1c","Diagnosis",
]

RANGES = {
    "Age": (0,120), "SystolicBP": (70,260), "DiastolicBP": (40,160),
    "SerumCreatinine": (0.2,15), "BUNLevels": (1,200), "GFR": (1,200),
    "ACR": (0,5000), "SerumElectrolytesSodium": (110,170),
    "SerumElectrolytesPotassium": (2.0,7.5), "HemoglobinLevels": (5,20),
    "HbA1c": (3.5,15),
}

BINARY_LIKE = [
    "Gender","Diagnosis","Smoking","FamilyHistoryKidneyDisease","FamilyHistoryHypertension",
    "FamilyHistoryDiabetes","PreviousAcuteKidneyInjury","UrinaryTractInfections",
    "ACEInhibitors","Diuretics","Statins","AntidiabeticMedications","Edema",
    "HeavyMetalsExposure","OccupationalExposureChemicals","WaterQuality",
]

def check_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def clamp_outliers(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.DataFrame:
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df

def validate_ranges(df: pd.DataFrame):
    problems = []
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi) | df[col].isna()
        pct_bad = round(100 * mask.mean(), 3)
        if pct_bad > 0:
            problems.append(f"{col}: {pct_bad}% outside [{lo}, {hi}] or missing")
        df = clamp_outliers(df, col, lo, hi)
    return df, problems

def validate_binary(df: pd.DataFrame):
    notes = []
    for col in BINARY_LIKE:
        if col in df.columns:
            uniq = sorted(df[col].dropna().unique().tolist())
            if not set(uniq).issubset({0, 1}):
                notes.append(f"{col}: values not strictly 0/1 -> {uniq[:10]}")
    return notes

def ckd_stage_from_gfr(gfr: float):
    if pd.isna(gfr):
        return np.nan
    if gfr >= 90: return 1
    if 60 <= gfr < 90: return 2
    if 45 <= gfr < 60: return 3
    if 30 <= gfr < 45: return 3
    if 15 <= gfr < 30: return 4
    return 5

def albuminuria_cat_from_acr(acr: float):
    if pd.isna(acr): return np.nan
    if acr < 30: return 1
    if acr <= 300: return 2
    return 3

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["UreaCreatinineRatio"] = df["BUNLevels"] / (df["SerumCreatinine"] + 1e-6)
    df["CKDStage"] = df["GFR"].apply(ckd_stage_from_gfr).astype("Int64")
    df["AlbuminuriaCat"] = df["ACR"].apply(albuminuria_cat_from_acr).astype("Int64")
    df["BP_Risk"] = ((df["SystolicBP"] >= 130) | (df["DiastolicBP"] >= 80)).astype(int)
    df["HyperkalemiaFlag"] = (df["SerumElectrolytesPotassium"] >= 5.5).astype(int)
    df["AnemiaFlag"] = (df["HemoglobinLevels"] < 12.0).astype(int)
    return df

if __name__ == "__main__":
    raw_path = Path(RAW_CSV)
    out_path = Path(OUT_PARQUET)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"CSV not found at {raw_path.resolve()}")

    df = pd.read_csv(raw_path)
    print(f"Loaded: {raw_path}  -> rows={len(df)} cols={df.shape[1]}")

    check_required_columns(df)
    df, range_issues = validate_ranges(df)
    if range_issues:
        print("⚠ Range/missingness notes:")
        for n in range_issues:
            print("  -", n)

    bin_notes = validate_binary(df)
    if bin_notes:
        print("⚠ Binary value notes:")
        for n in bin_notes:
            print("  -", n)

    df = derive_features(df)

    if "Diagnosis" in df.columns:
        dist = df["Diagnosis"].value_counts(dropna=False).to_dict()
        print(f"Target distribution (Diagnosis): {dist}")

    df.to_parquet(out_path, index=False)
    print(f"✅ Wrote processed dataset -> {out_path}")
