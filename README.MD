# CKD AI (MVP)

## Quickstart
1) Put your dataset at `data/raw/Chronic_Kidney_Dsease_data.csv`.
2) `pip install -r requirements.txt`
3) `cp config/.env.example .env`
4) `make prep`  (runs ETL: validate + derive features + write parquet)
