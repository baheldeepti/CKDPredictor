import os
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")
PARQUET = os.getenv("CKD_PROCESSED_PARQUET", "./data/processed/ckd_prepared.parquet")

REQUIRED = [
    "PatientID","Age","Gender","SystolicBP","DiastolicBP","SerumCreatinine","BUNLevels",
    "GFR","ACR","SerumElectrolytesSodium","SerumElectrolytesPotassium","HemoglobinLevels",
    "HbA1c","PulsePressure","UreaCreatinineRatio","CKDStage","AlbuminuriaCat",
    "BP_Risk","HyperkalemiaFlag","AnemiaFlag","Diagnosis"
]

def main():
    assert DATABASE_URL, "DATABASE_URL not set (did you source .env?)"
    df = pd.read_parquet(PARQUET)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in parquet: {missing}")

    # Build outgoing frame with the columns we want in Postgres
    out = pd.DataFrame({
        "patient_id": df["PatientID"].astype(str),
        "Age": df["Age"],
        "Gender": df["Gender"],
        "SystolicBP": df["SystolicBP"],
        "DiastolicBP": df["DiastolicBP"],
        "SerumCreatinine": df["SerumCreatinine"],
        "BUNLevels": df["BUNLevels"],
        "GFR": df["GFR"],
        "ACR": df["ACR"],
        "SerumElectrolytesSodium": df["SerumElectrolytesSodium"],
        "SerumElectrolytesPotassium": df["SerumElectrolytesPotassium"],
        "HemoglobinLevels": df["HemoglobinLevels"],
        "HbA1c": df["HbA1c"],
        "PulsePressure": df["PulsePressure"],
        "UreaCreatinineRatio": df["UreaCreatinineRatio"],
        "CKDStage": df["CKDStage"].astype("Int64"),
        "AlbuminuriaCat": df["AlbuminuriaCat"].astype("Int64"),
        "BP_Risk": df["BP_Risk"],
        "HyperkalemiaFlag": df["HyperkalemiaFlag"],
        "AnemiaFlag": df["AnemiaFlag"],
        "Diagnosis": df["Diagnosis"],
    })

    # 🔧 Critical fix: lowercase all column names to match Postgres' unquoted identifiers
    out.columns = [c.lower() for c in out.columns]

    engine = create_engine(DATABASE_URL)
    out.to_sql("health_metrics", engine, if_exists="append", index=False)
    print(f"✅ Loaded {len(out)} rows into health_metrics.")

if __name__ == "__main__":
    main()
