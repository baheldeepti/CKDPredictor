import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL not set. Did you `set -a; source .env; set +a`?"

DDL = """
-- Table of raw inputs coming from UI/API
CREATE TABLE IF NOT EXISTS user_inputs (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMP DEFAULT NOW(),

    -- features (match API/FEATURE_COLUMNS)
    age                         DOUBLE PRECISION,
    gender                      INTEGER,
    systolicbp                  DOUBLE PRECISION,
    diastolicbp                 DOUBLE PRECISION,
    serumcreatinine             DOUBLE PRECISION,
    bunlevels                   DOUBLE PRECISION,
    gfr                         DOUBLE PRECISION,
    acr                         DOUBLE PRECISION,
    serumelectrolytessodium     DOUBLE PRECISION,
    serumelectrolytespotassium  DOUBLE PRECISION,
    hemoglobinlevels            DOUBLE PRECISION,
    hba1c                       DOUBLE PRECISION,
    pulsepressure               DOUBLE PRECISION,
    ureacreatinineratio         DOUBLE PRECISION,
    ckdstage                    INTEGER,
    albuminuriacat              INTEGER,
    bp_risk                     INTEGER,
    hyperkalemiaflag            INTEGER,
    anemiaflag                  INTEGER
);

-- Per-inference prediction result linked to a user_inputs row
CREATE TABLE IF NOT EXISTS inference_log (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT NOW(),
    input_id        BIGINT REFERENCES user_inputs(id) ON DELETE CASCADE,
    model_used      TEXT,
    threshold_used  DOUBLE PRECISION,
    prediction      INTEGER,               -- 0 = non-CKD, 1 = CKD
    prob_ckd        DOUBLE PRECISION,
    prob_non_ckd    DOUBLE PRECISION
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_user_inputs_created_at ON user_inputs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inference_log_input ON inference_log(input_id);
CREATE INDEX IF NOT EXISTS idx_inference_log_created_at ON inference_log(created_at DESC);
"""

def main():
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("✅ Tables ready: user_inputs, inference_log")

if __name__ == "__main__":
    main()
