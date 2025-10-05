# ui/app.py
import os
import requests
import streamlit as st
import pandas as pd

# ---------------------------
# Config
# ---------------------------
API_URL_DEFAULT = "http://0.0.0.0:8000"
API_URL = st.secrets.get("API_URL", API_URL_DEFAULT)

FEATURE_COLUMNS = [
    "age","gender","systolicbp","diastolicbp","serumcreatinine","bunlevels",
    "gfr","acr","serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c","pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat","bp_risk","hyperkalemiaflag","anemiaflag",
]

MODEL_CHOICES = [
    ("XGBoost (xgb)","xgb"),
    ("Random Forest (rf)","rf"),
    ("Logistic Regression (logreg)","logreg"),
]

# Optional LLM config (OpenAI-compatible)
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openai")   # openai | together | openrouter | custom
LLM_API_KEY   = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL  = st.secrets.get("LLM_BASE_URL", "")        # e.g., "https://api.openai.com/v1" or provider gateway
LLM_MODEL     = st.secrets.get("LLM_MODEL", "gpt-4o-mini")  # or an OSS model name provided by your gateway

st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="wide")

# ===== Helpers ===============================================================

def _result_card(label: str, prob_ckd: float, thr: float, model_used: str):
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:12px;border:1px solid #e6e6e6;">
          <div style="font-size:20px;font-weight:700;margin-bottom:8px;">
            Prediction:
            <span style="padding:2px 8px;border-radius:999px;background:{'#fee2e2' if label=='CKD' else '#dcfce7'};color:{'#991b1b' if label=='CKD' else '#065f46'}">
              {label}
            </span>
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div><strong>Prob CKD</strong><br><span style="font-size:22px;">{prob_ckd:.3f}</span></div>
            <div><strong>Threshold</strong><br><span style="font-size:22px;">{thr:.3f}</span></div>
            <div><strong>Model</strong><br><span style="font-size:22px;">{model_used}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def call_llm(system_prompt: str, user_prompt: str):
    """Minimal OpenAI-compatible client; works with OpenAI/Together/OpenRouter/base_url gateways."""
    if not LLM_API_KEY:
        st.warning("No LLM API key configured. Add LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) in `.streamlit/secrets.toml`.")
        return None

    try:
        from openai import OpenAI
    except Exception:
        st.error("Package `openai` missing. Install with: `pip install openai`")
        return None

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None

def derive_flags_and_bins(systolicbp, diastolicbp, potassium, hb, gfr, acr):
    bp_risk = 1 if (systolicbp >= 130 or diastolicbp >= 80) else 0
    hyperk = 1 if potassium >= 5.5 else 0
    anemia = 1 if hb < 12.0 else 0

    # CKD stage from GFR (coarse)
    if gfr >= 90: stage = 1
    elif gfr >= 60: stage = 2
    elif gfr >= 45: stage = 3
    elif gfr >= 30: stage = 3
    elif gfr >= 15: stage = 4
    else: stage = 5

    if acr < 30: alb = 1
    elif acr <= 300: alb = 2
    else: alb = 3

    return bp_risk, hyperk, anemia, stage, alb

# Keep last prediction for the Recommendations tab
if "last_pred_payload" not in st.session_state:
    st.session_state["last_pred_payload"] = None
if "last_pred_result" not in st.session_state:
    st.session_state["last_pred_result"] = None

# ===== Header =================================================================
st.title("🩺 CKD Predictor")
st.caption("Single & batch predictions with thresholding, metrics, and AI recommendations.")

# ===== Sidebar Controls ========================================================
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
    st.caption("Rebuild all models on the server (uses accumulated historical inputs), then hot-reload.")
    if st.button("Rebuild models now"):
        try:
            r = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
            if r.get("status") == "started":
                st.success("Retraining started in background. Use Health to check when updated.")
            else:
                st.info(r)
        except Exception as e:
            st.error(f"Retrain call failed: {e}")

# ===== Tabs ===================================================================
tab_single, tab_batch, tab_metrics, tab_advice = st.tabs(
    ["Single prediction", "Batch predictions", "Metrics", "Recommendations (alpha)"]
)

# ---- Single prediction --------------------------------------------------------
with tab_single:
    st.markdown("**Use this tab to score a single case.** Fill the form and click **Predict**. "
                "Optionally tick **Retrain after submit** to kick off server-side retraining right away.")

    with st.form("predict_form", border=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 60)
            gender = st.selectbox("Gender (0=female, 1=male)", [0, 1], index=1)
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

            bp_risk, hyperkalemiaflag, anemiaflag, ckdstage, albuminuriacat = derive_flags_and_bins(
                systolicbp, diastolicbp, serumelectrolytespotassium, hemoglobinlevels, gfr, acr
            )

        pulsepressure = systolicbp - diastolicbp
        ureacreatinineratio = float(bunlevels) / (float(serumcreatinine) + 1e-6)

        st.caption(
            f"Derived → PulsePressure={pulsepressure} | Urea/Creatinine={ureacreatinineratio:.2f} "
            f"| Flags: BP={bp_risk} HK={hyperkalemiaflag} Anemia={anemiaflag} "
            f"| Stage={ckdstage} Albuminuria={albuminuriacat}"
        )

        retrain_after = st.checkbox("Retrain after submit", value=False)
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
            _result_card(label, float(res.get("prob_ckd", 0.0)), float(res.get("threshold_used", 0.5)), res.get("model_used", "unknown"))

            # Save to session for the Recommendations tab
            st.session_state["last_pred_payload"] = payload
            st.session_state["last_pred_result"] = res

            # Optional retrain (no form button here; we respect the checkbox)
            if retrain_after:
                try:
                    rr = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
                    st.success("Retraining started. Use Health in the sidebar to check when updated.")
                except Exception as e:
                    st.error(f"Retrain call failed: {e}")

        except Exception as e:
            st.error(f"API call failed: {e}")

# ---- Batch predictions -------------------------------------------------------
with tab_batch:
    st.markdown("**Use this tab to score a file of cases.** Download the template, fill it, upload, then click **Run Batch Predictions**.")

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
                    r = requests.post(
                        f"{api_url}/predict/batch",
                        params={"model": model_key},
                        json={"rows": rows},
                        timeout=60
                    )
                    r.raise_for_status()
                    preds = pd.DataFrame(r.json()["predictions"])
                    st.success("Batch complete.")
                    ckd_rate = preds["prediction"].mean() if "prediction" in preds else 0.0
                    st.markdown(f"**CKD positive rate:** {ckd_rate:.1%}  •  rows: {len(preds)}")
                    st.dataframe(preds.head())

                    merged = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                    st.download_button(
                        "Download Results CSV",
                        data=merged.to_csv(index=False).encode("utf-8"),
                        file_name="ckd_batch_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# ---- Metrics ----------------------------------------------------------------
with tab_metrics:
    st.markdown("**Use this tab to monitor the service and the last retrain.**")

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

# ---- Recommendations (alpha) ------------------------------------------------
with tab_advice:
    st.markdown(
        "**Use this tab for AI-assisted recommendations.** It interprets the latest prediction, CKD stage,"
        " albuminuria category, and flags (BP, hyperkalemia, anemia) and drafts patient-friendly next steps.\n\n"
        "_Safety rails:_ No medication dosing; not medical advice; PII removed."
    )

    if st.session_state["last_pred_payload"] is None or st.session_state["last_pred_result"] is None:
        st.info("Run a single prediction first to populate this tab.")
    else:
        payload = st.session_state["last_pred_payload"]
        res = st.session_state["last_pred_result"]

        label = "CKD" if res.get("prediction", 1) == 1 else "Non-CKD"
        prob_ckd = float(res.get("prob_ckd", 0.0))
        stage = int(payload["ckdstage"])
        alb = int(payload["albuminuriacat"])
        bp_risk = int(payload["bp_risk"])
        hyperk = int(payload["hyperkalemiaflag"])
        anemia = int(payload["anemiaflag"])

        system_prompt = (
            "You are a multi-agent coordinator for kidney risk recommendations. "
            "Agents: medical_analyst, diet_specialist, exercise_coach, care_coordinator. "
            "Use cautious tone. No medication dosing. Always include a 'Not medical advice' disclaimer."
        )

        user_prompt = f"""
Prediction summary:
- Label: {label}
- Prob_CKD: {prob_ckd:.3f}
- CKD Stage (from GFR): {stage}
- Albuminuria Category (from ACR): {alb}
- Flags: BP_Risk={bp_risk}, Hyperkalemia={hyperk}, Anemia={anemia}

Task:
1) medical_analyst: interpret stage/albuminuria and red flags (e.g., potassium ≥6.0).
2) diet_specialist: stage- and ACR-aware guidance (protein, sodium, potassium).
3) exercise_coach: moderate plan; contraindications if anemia/hyperkalemia present.
4) care_coordinator: suggest lab follow-ups (e.g., repeat ACR), check timing, non-directive med chats (ACEi/ARB note).

Constraints:
- No medication dosing.
- Include a short, readable 'next steps' list.
- Add: 'This is not medical advice; consult your clinician.'
"""

        if st.button("Generate recommendations"):
            text = call_llm(system_prompt, user_prompt)
            if text:
                st.markdown(text)
