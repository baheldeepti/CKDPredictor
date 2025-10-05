import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://0.0.0.0:8000")  # override via .streamlit/secrets.toml if you want

st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="centered")
st.title("🩺 CKD Predictor (MVP)")
st.caption("Model-backed API with tuned threshold. Enter metrics, get a risk prediction.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=60)
        gender = st.selectbox("Gender (0=female,1=male)", options=[0, 1], index=1)
        systolicbp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=260, value=140)
        diastolicbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=85)
        serumcreatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=15.0, value=2.0, step=0.1)
        bunlevels = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=28.0, step=0.5)
        gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=1.0, max_value=200.0, value=55.0, step=0.5)
        acr = st.number_input("ACR (mg/g)", min_value=0.0, max_value=5000.0, value=120.0, step=1.0)
    with col2:
        serumelectrolytessodium = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=170.0, value=138.0, step=0.5)
        serumelectrolytespotassium = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=7.5, value=4.8, step=0.1)
        hemoglobinlevels = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.5, step=0.1)
        hba1c = st.number_input("HbA1c (%)", min_value=3.5, max_value=15.0, value=6.8, step=0.1)
        # Derived or categorical
        bp_risk = 1 if (systolicbp >= 130 or diastolicbp >= 80) else 0
        hyperkalemiaflag = 1 if serumelectrolytespotassium >= 5.5 else 0
        anemiaflag = 1 if hemoglobinlevels < 12.0 else 0

        # CKD staging helpers (simple rules consistent with earlier code)
        if gfr >= 90: ckdstage = 1
        elif gfr >= 60: ckdstage = 2
        elif gfr >= 45: ckdstage = 3
        elif gfr >= 30: ckdstage = 3
        elif gfr >= 15: ckdstage = 4
        else: ckdstage = 5

        if acr < 30: albuminuriacat = 1
        elif acr <= 300: albuminuriacat = 2
        else: albuminuriacat = 3

    # Auto-derived features
    pulsepressure = systolicbp - diastolicbp
    ureacreatinineratio = float(bunlevels) / (float(serumcreatinine) + 1e-6)

    st.write("**Derived fields**")
    st.write(f"• Pulse Pressure: `{pulsepressure}`  |  • Urea/Creatinine Ratio: `{ureacreatinineratio:.2f}`")
    st.write(f"• BP Risk: `{bp_risk}`  • Hyperkalemia: `{hyperkalemiaflag}`  • Anemia: `{anemiaflag}`")
    st.write(f"• CKD Stage (rule-based): `{ckdstage}`  • Albuminuria Cat: `{albuminuriacat}`")

    submitted = st.form_submit_button("Predict")
    if submitted:
        payload = {
            "age": age,
            "gender": gender,
            "systolicbp": systolicbp,
            "diastolicbp": diastolicbp,
            "serumcreatinine": serumcreatinine,
            "bunlevels": bunlevels,
            "gfr": gfr,
            "acr": acr,
            "serumelectrolytessodium": serumelectrolytessodium,
            "serumelectrolytespotassium": serumelectrolytespotassium,
            "hemoglobinlevels": hemoglobinlevels,
            "hba1c": hba1c,
            "pulsepressure": pulsepressure,
            "ureacreatinineratio": ureacreatinineratio,
            "ckdstage": ckdstage,
            "albuminuriacat": albuminuriacat,
            "bp_risk": bp_risk,
            "hyperkalemiaflag": hyperkalemiaflag,
            "anemiaflag": anemiaflag
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            r.raise_for_status()
            res = r.json()
            label = "CKD" if res["prediction"] == 1 else "Non-CKD"
            st.success(f"**Prediction:** {label}")
            st.write(f"**prob_ckd:** {res['prob_ckd']:.3f}  |  **prob_non_ckd:** {res['prob_non_ckd']:.3f}")
            st.caption(f"Threshold used: {res['threshold_used']:.5f}")
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.code(payload, language="json")

st.divider()
st.caption("Tip: set `API_URL` via `.streamlit/secrets.toml` when deploying.")
