# ui/app.py
import io, os, requests, streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
API_URL = st.secrets.get("API_URL", "http://0.0.0.0:8000")  # set via .streamlit/secrets.toml
FEATURE_COLUMNS = [
    "age","gender","systolicbp","diastolicbp","serumcreatinine","bunlevels",
    "gfr","acr","serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c","pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat","bp_risk","hyperkalemiaflag","anemiaflag",
]
MODEL_CHOICES = [("XGBoost (xgb)","xgb"), ("Random Forest (rf)","rf"), ("Logistic Regression (logreg)","logreg")]

st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="wide")

# ---------------------------
# Header
# ---------------------------
st.title("🩺 CKD Predictor")
st.caption("Real-time risk scoring + batch screening. Retrain on demand.")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    api_url = st.text_input("API URL", API_URL)
    model_label, model_key = st.selectbox("Model", MODEL_CHOICES, index=0, format_func=lambda x: x[0])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Health"):
            try:
                h = requests.get(f"{api_url}/health", params={"model": model_key}, timeout=10).json()
                st.success(h)
            except Exception as e:
                st.error(f"Health check failed: {e}")
    with c2:
        if st.button("Reload models"):
            try:
                r = requests.post(f"{api_url}/admin/reload", timeout=20)
                st.success("Reloaded model cache.")
            except Exception as e:
                st.error(f"Reload failed: {e}")

    st.divider()
    st.caption("Rebuild all models (server-side), then hot-reload.")
    if st.button("Rebuild models now"):
        try:
            r = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
            if r.get("status") == "started":
                st.success("Retraining started in background. Use Health to check when updated.")
            else:
                st.info(r)
        except Exception as e:
            st.error(f"Retrain call failed: {e}")

# ---------------------------
# Tabs: Single | Batch | Metrics
# ---------------------------
tab_single, tab_batch, tab_metrics = st.tabs(["Single prediction", "Batch predictions", "Metrics"])

# ---- Single prediction ----
with tab_single:
    with st.form("predict_form", border=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 60)
            gender = st.selectbox("Gender (0=female, 1=male)", [0,1], index=1)
            systolicbp = st.number_input("Systolic BP (mmHg)", 70, 260, 140)
            diastolicbp = st.number_input("Diastolic BP (mmHg)", 40, 160, 85)
            serumcreatinine = st.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, 2.0, step=0.1)
            bunlevels = st.number_input("BUN (mg/dL)", 1.0, 200.0, 28.0, step=0.5)
            gfr = st.number_input("GFR (mL/min/1.73m²)", 1.0, 200.0, 55.0, step=0.5)
            acr = st.number_input("ACR (mg/g)", 0.0, 5000.0, 120.0, step=1.0)
        with col2:
            serumelectrolytessodium = st.number_input("Sodium (mEq/L)", 110.0, 170.0, 138.0, step=0.5)
            serumelectrolytespotassium = st.number_input("Potassium (mEq/L)", 2.0, 7.5, 4.8, step=0.1)
            hemoglobinlevels = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.5, step=0.1)
            hba1c = st.number_input("HbA1c (%)", 3.5, 15.0, 6.8, step=0.1)

            bp_risk = 1 if (systolicbp >= 130 or diastolicbp >= 80) else 0
            hyperkalemiaflag = 1 if serumelectrolytespotassium >= 5.5 else 0
            anemiaflag = 1 if hemoglobinlevels < 12.0 else 0

            if gfr >= 90: ckdstage = 1
            elif gfr >= 60: ckdstage = 2
            elif gfr >= 45: ckdstage = 3
            elif gfr >= 30: ckdstage = 3
            elif gfr >= 15: ckdstage = 4
            else: ckdstage = 5

            if acr < 30: albuminuriacat = 1
            elif acr <= 300: albuminuriacat = 2
            else: albuminuriacat = 3

        pulsepressure = systolicbp - diastolicbp
        ureacreatinineratio = float(bunlevels) / (float(serumcreatinine) + 1e-6)

        st.caption(
            f"Derived → PulsePressure={pulsepressure} | Urea/Creatinine={ureacreatinineratio:.2f} "
            f"| Flags: BP={bp_risk} HK={hyperkalemiaflag} Anemia={anemiaflag} "
            f"| Stage={ckdstage} Albuminuria={albuminuriacat}"
        )
        submitted = st.form_submit_button("Predict")

        if submitted:
            payload = {
                "age": age, "gender": gender,
                "systolicbp": systolicbp, "diastolicbp": diastolicbp,
                "serumcreatinine": serumcreatinine, "bunlevels": bunlevels,
                "gfr": gfr, "acr": acr,
                "serumelectrolytessodium": serumelectrolytessodium,
                "serumelectrolytespotassium": serumelectrolytespotassium,
                "hemoglobinlevels": hemoglobinlevels, "hba1c": hba1c,
                "pulsepressure": pulsepressure, "ureacreatinineratio": ureacreatinineratio,
                "ckdstage": ckdstage, "albuminuriacat": albuminuriacat,
                "bp_risk": bp_risk, "hyperkalemiaflag": hyperkalemiaflag, "anemiaflag": anemiaflag
            }
            try:
                r = requests.post(f"{api_url}/predict", params={"model": model_key}, json=payload, timeout=20)
                r.raise_for_status()
                res = r.json()
                label = "CKD" if res.get("prediction", 1) == 1 else "Non-CKD"
                st.success(f"**{label}**  •  prob_ckd={res.get('prob_ckd', 0.0):.3f}  •  prob_non_ckd={res.get('prob_non_ckd', 0.0):.3f}")
                st.caption(f"Model: {res.get('model_used','unknown')}  •  Threshold: {res.get('threshold_used',0.5):.5f}")

                with st.expander("Maintenance"):
                    st.write("Rebuild models now (uses accumulated historical inputs).")
                    if st.button("Rebuild after this prediction"):
                        try:
                            rr = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
                            st.success("Retraining started. Use Health in the sidebar to check when updated.")
                        except Exception as e:
                            st.error(f"Retrain call failed: {e}")
            except Exception as e:
                st.error(f"API call failed: {e}")

# ---- Batch predictions ----
with tab_batch:
    st.caption("Upload a CSV with the exact columns (download template below).")
    template = pd.DataFrame(columns=FEATURE_COLUMNS)
    st.download_button("Download Template CSV", data=template.to_csv(index=False).encode("utf-8"),
                       file_name="ckd_template.csv", mime="text/csv")
    st.code(",".join(FEATURE_COLUMNS), language="text")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.dataframe(df.head())
                if st.button("Run Batch Predictions"):
                    rows = df.to_dict(orient="records")
                    r = requests.post(f"{api_url}/predict/batch", params={"model": model_key}, json={"rows": rows}, timeout=60)
                    r.raise_for_status()
                    preds = pd.DataFrame(r.json()["predictions"])
                    st.success("Batch complete.")
                    st.dataframe(preds.head())
                    merged = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                    st.download_button("Download Results CSV",
                        data=merged.to_csv(index=False).encode("utf-8"),
                        file_name="ckd_batch_predictions.csv", mime="text/csv")

                    with st.expander("Maintenance"):
                        st.write("Rebuild models now using the latest accumulated inputs.")
                        if st.button("Rebuild after this batch"):
                            try:
                                rr = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
                                st.success("Retraining started. Use Health in the sidebar to check when updated.")
                            except Exception as e:
                                st.error(f"Retrain call failed: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# ---- Metrics ----
with tab_metrics:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Current Health**")
        try:
            h = requests.get(f"{api_url}/health", params={"model": model_key}, timeout=10).json()
            st.json(h)
        except Exception as e:
            st.error(f"Health failed: {e}")
    with c2:
        st.write("**Recent Inferences**")
        try:
            li = requests.get(f"{api_url}/metrics/last_inferences?limit=10", timeout=10).json()
            rows = li.get("rows", [])
            if rows:
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No inferences found yet.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    st.write("**Last Retrain Report**")
    try:
        rr = requests.get(f"{api_url}/metrics/retrain_report", timeout=10)
        if rr.status_code == 200:
            st.json(rr.json())
        else:
            st.info("No retrain report found.")
    except Exception as e:
        st.error(f"Report fetch failed: {e}")
