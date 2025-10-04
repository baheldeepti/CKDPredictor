import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set (did you source .env in this shell?)"

DDL = """
CREATE TABLE IF NOT EXISTS health_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    patient_id TEXT,
    recorded_at TIMESTAMP DEFAULT NOW(),

    Age INT,
    Gender INT,
    SystolicBP DOUBLE PRECISION,
    DiastolicBP DOUBLE PRECISION,
    SerumCreatinine DOUBLE PRECISION,
    BUNLevels DOUBLE PRECISION,
    GFR DOUBLE PRECISION,
    ACR DOUBLE PRECISION,
    SerumElectrolytesSodium DOUBLE PRECISION,
    SerumElectrolytesPotassium DOUBLE PRECISION,
    HemoglobinLevels DOUBLE PRECISION,
    HbA1c DOUBLE PRECISION,

    PulsePressure DOUBLE PRECISION,
    UreaCreatinineRatio DOUBLE PRECISION,
    CKDStage INT,
    AlbuminuriaCat INT,
    BP_Risk INT,
    HyperkalemiaFlag INT,
    AnemiaFlag INT,

    Diagnosis INT
);
"""

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("✅ Schema initialized: health_metrics table ready.")
