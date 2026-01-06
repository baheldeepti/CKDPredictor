# ============================================================
# app.py — NephroCompass | Kidney Health Radar
# ============================================================
# Single & bulk CKD screening with explainers
# Digital Twin (What-If), Counterfactuals, Similar Patients
# Scoped Chat Assistant (answers ONLY from loaded contexts)
# Optional OpenRouter/OpenAI LLM integration
#
# Run:
#   streamlit run app.py
# ============================================================

import os
import json
import math
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Tuple

# ============================================================
# Branding & Config
# ============================================================
APP_PRODUCT_NAME = "NephroCompass"
APP_TAGLINE = "Kidney Health Radar"
APP_TITLE = "NephroCompass — Kidney Health Radar"
APP_REPO_URL = "https://github.com/baheldeepti/CKDPredictor"

APP_URL = st.secrets.get("APP_URL", APP_REPO_URL)

API_URL_DEFAULT = "https://ckdpredictor.onrender.com"
API_URL = (
    st.secrets.get("API_URL")
    or os.environ.get("CKD_API_URL")
    or API_URL_DEFAULT
)

st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

# ============================================================
# Feature Columns (model contract)
# ============================================================
FEATURE_COLUMNS = [
    "age","gender","systolicbp","diastolicbp","serumcreatinine","bunlevels",
    "gfr","acr","serumelectrolytessodium","serumelectrolytespotassium",
    "hemoglobinlevels","hba1c","pulsepressure","ureacreatinineratio",
    "ckdstage","albuminuriacat","bp_risk","hyperkalemiaflag","anemiaflag",
]

MODEL_CHOICES = [
    ("XGBoost (xgb)", "xgb"),
    ("Random Forest (rf)", "rf"),
    ("Logistic Regression (logreg)", "logreg"),
]
MODEL_LABELS = {label: key for (label, key) in MODEL_CHOICES}
MODEL_KEYS   = {key: label for (label, key) in MODEL_CHOICES}

# ============================================================
# LLM config
# ============================================================
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openrouter")
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = st.secrets.get("LLM_MODEL", "openrouter/auto")

# ============================================================
# UX helpers — tooltips, probability ranges, explanations
# ============================================================
INPUT_HELP = {
    "age": "Age affects baseline CKD risk (risk tends to rise with age).",
    "gender": "Encoded as 0=female, 1=male (numeric encoding required by the model).",
    "systolicbp": "Top BP number. Persistent ≥130 mmHg is commonly considered elevated.",
    "diastolicbp": "Bottom BP number. Persistent ≥80 mmHg is commonly considered elevated.",
    "serumcreatinine": "Waste marker (mg/dL). Higher values usually reflect lower filtration.",
    "bunlevels": "Blood Urea Nitrogen (mg/dL). Can rise with dehydration or reduced kidney clearance.",
    "gfr": "Estimated filtration rate (mL/min/1.73m²). Lower values indicate reduced kidney function.",
    "acr": "Albumin-to-creatinine ratio (mg/g). Higher values suggest kidney damage (protein leakage).",
    "serumelectrolytessodium": "Sodium (mEq/L). Extreme values can be clinically important.",
    "serumelectrolytespotassium": "Potassium (mEq/L). High potassium can be dangerous in CKD.",
    "hemoglobinlevels": "Hemoglobin (g/dL). Low levels may indicate anemia, common in CKD.",
    "hba1c": "HbA1c (%). Reflects long-term blood sugar control; diabetes strongly impacts CKD risk.",
}

def prob_bucket(prob: float) -> tuple[str, str]:
    p = float(prob)
    if p < 0.10:
        return "0–10% (Low)", "Low signal for CKD risk in this snapshot."
    if p < 0.33:
        return "10–33% (Mild)", "Some signal present; usually monitored over time."
    if p < 0.66:
        return "33–66% (Moderate)", "Meaningful signal; follow-up labs often appropriate."
    return "66–100% (High)", "Strong signal; follow-up is strongly recommended."

def plain_english_result(prob: float, thr: float) -> str:
    bucket, meaning = prob_bucket(prob)
    flagged = prob >= thr
    return (
        f"**Probability range:** {bucket}\n\n"
        f"**Decision:** {'Flagged for follow-up' if flagged else 'Not flagged'} "
        f"(probability {'≥' if flagged else '<'} model threshold {thr:.2f}).\n\n"
        f"**Plain English:** {meaning}\n\n"
        f"Probability is **not a diagnosis** — it reflects how similar this profile is to known CKD patterns."
    )

def shap_plain_english(top_rows: list[dict], max_items: int = 5) -> str:
    if not top_rows:
        return "No explanation available for this case."

    lines = []
    for r in top_rows[:max_items]:
        feat = r.get("feature", "unknown")
        signed = float(r.get("signed", 0))
        direction = "pushed risk **UP**" if signed > 0 else "pushed risk **DOWN**"
        lines.append(f"- **{feat}** {direction} (Δprob ≈ {signed:+.3f})")

    return (
        "For this prediction, the model relied most on:\n"
        + "\n".join(lines)
        + "\n\n**How to read this:**\n"
          "- This is not causation.\n"
          "- It shows which inputs mattered most *for this one case*.\n"
          "- Direction reflects influence relative to similar patients."
    )

# ============================================================
# Utilities
# ============================================================
def sanitize_payload(d: dict) -> dict:
    out = {}
    for k in FEATURE_COLUMNS:
        v = d.get(k, 0)
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except Exception:
            v = 0.0
        out[k] = v
    return out

def _normalize_api(url: str) -> str:
    return (url or API_URL_DEFAULT).rstrip("/")

def _api_get(url: str, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _api_post(url: str, json_body=None, params=None, timeout=60):
    r = requests.post(url, json=json_body, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ============================================================
# Session State
# ============================================================
for k in [
    "last_pred_payload","last_pred_results","batch_preds","batch_insights",
    "activity_log","chat_messages","ctx_digital_twin","ctx_counterfactual",
    "ctx_similar","ctx_explain"
]:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("api_url", _normalize_api(API_URL))
st.session_state.setdefault("activity_log", [])
st.session_state.setdefault("chat_messages", [])

def log(msg: str):
    st.session_state["activity_log"].append(msg)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🔌 Connection")
    api_in = st.text_input("Backend API URL", st.session_state["api_url"])
    api_in = _normalize_api(api_in)

    if st.button("Check API"):
        try:
            h = _api_get(f"{api_in}/health", params={"model":"rf"})
            st.success(f"OK • DB={h.get('db_backend')} • Connected={h.get('db_connected')}")
            st.session_state["api_url"] = api_in
        except Exception as e:
            st.error(f"API check failed: {e}")

    st.caption(f"Using `{st.session_state['api_url']}`")

    # Neon branding
    st.markdown("---")
    st.markdown("### ⚡ Data backend")
    st.markdown(
        """
<div style="display:flex;align-items:center;gap:10px;">
  <img src="https://neon.tech/favicon.ico" width="18"/>
  <b>Powered by Neon</b>
</div>
<div style="font-size:12px;color:#666;margin-top:4px;">
Serverless Postgres for modern clinical analytics.
</div>
""",
        unsafe_allow_html=True
    )

# ============================================================
# Header
# ============================================================
st.title(f"🧭 {APP_TITLE}")
st.caption(
    "Transparent CKD risk screening with explainability, simulations, and context-scoped AI assistance. "
    "Educational tool — not medical advice."
)

# ============================================================
# Tabs (ALL preserved)
# ============================================================
(
    tab_single,
    tab_batch,
    tab_metrics,
    tab_advice,
    tab_digital_twin,
    tab_counterfactuals,
    tab_similarity,
    tab_agents,
    tab_chat,
) = st.tabs([
    "Single check",
    "Bulk check (CSV)",
    "Service & metrics",
    "AI summary & next steps",
    "Digital Twin (What-If)",
    "Counterfactuals",
    "Similar Patients",
    "Care Plan",
    "Chat Assistant",
])

# ============================================================
# SINGLE CHECK (clinic-grade UX)
# ============================================================
with tab_single:
    st.subheader("Single patient screening")

    with st.form("single_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 0, 120, 60, help=INPUT_HELP["age"])
            gender = st.selectbox("Sex (0=female, 1=male)", [0,1], help=INPUT_HELP["gender"])
            systolicbp = st.number_input("Systolic BP (mmHg)", 70, 260, 140, help=INPUT_HELP["systolicbp"])
            diastolicbp = st.number_input("Diastolic BP (mmHg)", 40, 160, 85, help=INPUT_HELP["diastolicbp"])
            gfr = st.number_input("GFR", 1.0, 200.0, 55.0, help=INPUT_HELP["gfr"])
            acr = st.number_input("ACR (mg/g)", 0.0, 5000.0, 120.0, help=INPUT_HELP["acr"])

        with col2:
            serumcreatinine = st.number_input("Creatinine (mg/dL)", 0.2, 15.0, 2.0, help=INPUT_HELP["serumcreatinine"])
            bunlevels = st.number_input("BUN (mg/dL)", 1.0, 200.0, 28.0, help=INPUT_HELP["bunlevels"])
            serumelectrolytessodium = st.number_input("Sodium (mEq/L)", 110.0, 170.0, 138.0, help=INPUT_HELP["serumelectrolytessodium"])
            serumelectrolytespotassium = st.number_input("Potassium (mEq/L)", 2.0, 7.5, 4.8, help=INPUT_HELP["serumelectrolytespotassium"])
            hemoglobinlevels = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.5, help=INPUT_HELP["hemoglobinlevels"])
            hba1c = st.number_input("HbA1c (%)", 3.5, 15.0, 6.8, help=INPUT_HELP["hba1c"])

        submit = st.form_submit_button("Run CKD risk screening")

    if submit:
        pulsepressure = systolicbp - diastolicbp
        ureacreatinineratio = bunlevels / (serumcreatinine + 1e-6)

        bp_risk = int(systolicbp >= 130 or diastolicbp >= 80)
        hyperk = int(serumelectrolytespotassium >= 5.5)
        anemia = int(hemoglobinlevels < 12)
        ckdstage = 1 if gfr>=90 else 2 if gfr>=60 else 3 if gfr>=30 else 4 if gfr>=15 else 5
        albuminuriacat = 1 if acr<30 else 2 if acr<=300 else 3

        payload = sanitize_payload({
            "age":age,"gender":gender,"systolicbp":systolicbp,"diastolicbp":diastolicbp,
            "serumcreatinine":serumcreatinine,"bunlevels":bunlevels,"gfr":gfr,"acr":acr,
            "serumelectrolytessodium":serumelectrolytessodium,
            "serumelectrolytespotassium":serumelectrolytespotassium,
            "hemoglobinlevels":hemoglobinlevels,"hba1c":hba1c,
            "pulsepressure":pulsepressure,"ureacreatinineratio":ureacreatinineratio,
            "ckdstage":ckdstage,"albuminuriacat":albuminuriacat,
            "bp_risk":bp_risk,"hyperkalemiaflag":hyperk,"anemiaflag":anemia
        })

        st.session_state["last_pred_payload"] = payload
        st.session_state["last_pred_results"] = {}

        for label, model in MODEL_CHOICES:
            res = _api_post(
                f"{st.session_state['api_url']}/predict",
                json_body=payload,
                params={"model":model}
            )
            prob = res["prob_ckd"]
            thr = res["threshold_used"]

            st.markdown(f"### {label}")
            st.metric("CKD probability", f"{prob:.3f}")
            st.markdown(plain_english_result(prob, thr))

            st.session_state["last_pred_results"][model] = res

# =========================
# Bulk predictions
# =========================
with tab_batch:
    st.markdown("Upload a CSV, run through selected models, and compare results.")
    left, right = st.columns([2, 1])
    with left:
        st.caption("Required columns:")
        st.code(",".join(FEATURE_COLUMNS), language="text")
    with right:
        template_blank = pd.DataFrame(columns=FEATURE_COLUMNS)
        sample5 = sample_records_df()
        st.download_button("Blank template CSV",
                           data=template_blank.to_csv(index=False).encode("utf-8"),
                           file_name="ckd_template_blank.csv", mime="text/csv", key="dl_blank_template")
        st.download_button("Sample CSV (5 rows)",
                           data=sample5.to_csv(index=False).encode("utf-8"),
                           file_name="ckd_sample_5rows.csv", mime="text/csv", key="dl_sample5")

    file = st.file_uploader("Upload CSV", type=["csv"], key="batch_file")
    retrain_after_batch = st.checkbox("Start training after batch (optional)", value=False, key="batch_retrain")

    if file:
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.dataframe(df.head())

                if st.button("Run batch with selected models", key="btn_run_batch"):
                    st.session_state["batch_preds"] = {}
                    summary_rows = []
                    for m in selected_models:
                        try:
                            rows = df[FEATURE_COLUMNS].to_dict(orient="records")
                            rows = [sanitize_payload(r) for r in rows]
                            r = _api_post(
                                f"{st.session_state['api_url']}/predict/batch",
                                params={"model": m},
                                json_body={"rows": rows},
                                timeout=120
                            )
                            preds = pd.DataFrame(r["predictions"])
                            preds["model_used"] = m
                            st.session_state["batch_preds"][m] = preds

                            ckd_rate = preds["prediction"].mean() if "prediction" in preds else 0.0
                            avg_prob = preds["prob_ckd"].mean() if "prob_ckd" in preds else 0.0
                            thr = preds["threshold_used"].iloc[0] if "threshold_used" in preds and not preds.empty else None
                            summary_rows.append({
                                "Model": MODEL_KEYS.get(m, m),
                                "Positive rate": f"{ckd_rate:.1%}",
                                "Avg Prob_CKD": f"{avg_prob:.3f}",
                                "Threshold": f"{thr:.3f}" if thr is not None else "—",
                                "Rows": len(preds)
                            })
                            _log(f"Batch OK ({m}) rows={len(preds)}")
                        except requests.HTTPError as e:
                            st.error(f"{MODEL_KEYS.get(m,m)} failed")
                            st.code(getattr(e.response, "text", str(e)) or str(e), language="json")
                            _log(f"Batch HTTP ERROR ({m}): {e}")
                        except Exception as e:
                            st.error(f"{MODEL_KEYS.get(m,m)} error")
                            st.caption(str(e))
                            _log(f"Batch ERROR ({m}): {e}")

                    if summary_rows:
                        st.success("Batch complete.")
                        st.markdown("#### Model comparison (batch summary)")
                        st.table(pd.DataFrame(summary_rows))

                        merged_list = []
                        for m, preds in st.session_state["batch_preds"].items():
                            merged = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                            merged_list.append(merged.assign(model_used=m))
                            st.download_button(
                                f"Download results • {MODEL_KEYS.get(m,m)}",
                                data=merged.to_csv(index=False).encode("utf-8"),
                                file_name=f"ckd_batch_predictions_{m}.csv",
                                mime="text/csv",
                                key=f"dl_model_{m}"
                            )
                        if merged_list:
                            all_merged = pd.concat(merged_list, axis=0, ignore_index=True)
                            st.download_button(
                                "Download ALL results (combined)",
                                data=all_merged.to_csv(index=False).encode("utf-8"),
                                file_name="ckd_batch_predictions_all_models.csv",
                                mime="text/csv",
                                key="dl_all_models"
                            )

                        if retrain_after_batch:
                            try:
                                rr = requests.post(f"{st.session_state['api_url']}/admin/retrain", timeout=10)
                                if rr.status_code in (200, 202):
                                    st.success("Training started. Check API health above.")
                                    _log("Retrain triggered after batch.")
                                else:
                                    st.info(f"Training request sent ({rr.status_code}).")
                                    _log(f"Retrain response after batch: {getattr(rr,'text','')[:200]}")
                            except Exception as e:
                                st.error("Training call failed.")
                                st.caption(str(e))
                                _log(f"Retrain after batch ERROR: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # Cohort insights (AI)
    st.markdown("#### AI cohort summary")
    st.caption("Turns batch results into a short clinical overview. No medication dosing.")
    available_models = list(st.session_state.get("batch_preds", {}).keys())
    if not available_models:
        st.info("Run a batch first to enable insights.")
    else:
        insights_model = st.selectbox(
            "Summarize which model's output?",
            [MODEL_KEYS.get(m, m) for m in available_models],
            index=0,
            key="batch_insights_model"
        )
        chosen_m = next((k for k in available_models if MODEL_KEYS.get(k, k) == insights_model), available_models[0])

        preds = st.session_state["batch_preds"][chosen_m]
        pos_rate = preds["prediction"].mean() if "prediction" in preds else 0.0
        avg_prob = preds["prob_ckd"].mean() if "prob_ckd" in preds else 0.0

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Summarize cohort", key="btn_summarize_cohort"):
                system_prompt = (
                    "You are a cautious, clinical assistant creating cohort insights for CKD screening.\n"
                    "Keep it concise and structured. Avoid medication dosing. Provide red flags, high-level diet/exercise guidance,\n"
                    "and a follow-up checklist. End with 'This is not medical advice; consult your clinician.'"
                )
                user_prompt = f"""
We screened a batch of cases for CKD using model: {MODEL_KEYS.get(chosen_m, chosen_m)}.

Metrics:
- Positive rate: {pos_rate:.3f}
- Average Prob_CKD: {avg_prob:.3f}
- Rows: {len(preds)}

Task (format output as short sections with bullets):
1) Red Flag Overview (what the rates imply and who to focus on).
2) High-Level Diet & Exercise Guidance (cohort-safe, stage-agnostic).
3) Follow-up Checklist (repeat labs, possible referrals, next checks).
4) Close with: 'This is not medical advice; consult your clinician.'
"""
                text = call_llm(system_prompt, user_prompt)
                if text:
                    st.session_state["batch_insights"] = text
                    _log("Cohort insights generated.")
                else:
                    st.error("LLM did not return text. Check the key/credits.")
                    _log("Cohort insights FAILED.")
        with c2:
            if st.button("Clear cached summary", key="btn_clear_insights"):
                st.session_state["batch_insights"] = None
                _log("Cohort insights cleared.")

        if st.session_state["batch_insights"]:
            insights_html = nl2br(st.session_state["batch_insights"])
            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div style="font-weight:800;color:var(--brand-800)">Cohort Insights — {MODEL_KEYS.get(chosen_m, chosen_m)}</div>
                    <div class="small-muted">Positive rate {pos_rate:.1%} • Avg Prob_CKD {avg_prob:.3f} • Rows {len(preds)}</div>
                  </div>
                  <div style="line-height:1.55">{insights_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.download_button(
                "Download insights (.txt)",
                data=st.session_state["batch_insights"].encode("utf-8"),
                file_name="ckd_batch_insights.txt",
                mime="text/plain",
                key="dl_insights_txt"
            )

# =========================
# Metrics
# =========================
with tab_metrics:
    st.markdown("Keep an eye on service health, recent inferences, and training reports.")
    auto = st.checkbox("Auto-refresh every 10 seconds", value=False, key="mtx_autorefresh")
    if auto:
        st.markdown("<script>setTimeout(() => window.location.reload(), 10000);</script>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Service health")
        try:
            first = (selected_models[0] if selected_models else 'rf')
            h = _api_get(f"{st.session_state['api_url']}/health", params={"model": first}, timeout=10)
            ok = h.get("status") == "ok"
            if ok:
                st.success(h)
            else:
                st.warning(h)
        except Exception as e:
            st.error(f"Health failed: {e}")

    with c2:
        st.subheader("Recent inferences")
        try:
            li = _api_get(f"{st.session_state['api_url']}/metrics/last_inferences", params={"limit":10}, timeout=10)
            rows = li.get("rows", [])
            if rows:
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No inferences yet — run Single or Batch.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    st.subheader("Training status (last retrain)")
    try:
        rr = requests.get(f"{st.session_state['api_url']}/metrics/retrain_report", timeout=10)
        if rr.status_code == 200:
            st.json(rr.json())
        else:
            st.info("No retrain report found. Click **Train models**, then **Reload models**.")
    except Exception as e:
        st.error(f"Report fetch failed: {e}")

# =========================
# AI summary & next steps (frontend LLM)
# =========================
with tab_advice:
    st.markdown(
        "AI-assisted **next steps** for the last single prediction. "
        "Considers stage, albuminuria, and flags. No medication dosing. "
        "_This is not medical advice._"
    )

    payload = st.session_state.get("last_pred_payload")
    results = st.session_state.get("last_pred_results", {})

    if not payload or not results:
        st.info("Run a single prediction first (any model).")
    else:
        avail = list(results.keys())
        chosen = st.selectbox("Use result from model", [MODEL_KEYS.get(m, m) for m in avail], index=0, key="advice_model_pick")
        chosen_key = next((k for k in avail if MODEL_KEYS.get(k, k) == chosen), avail[0])

        res = results[chosen_key]
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
Prediction summary (model: {MODEL_KEYS.get(chosen_key, chosen_key)}):
- Prob_CKD: {prob_ckd:.3f}
- CKD Stage (from GFR): {stage}
- Albuminuria Category (from ACR): {alb}
- Flags: BP_Risk={bp_risk}, Hyperkalemia={hyperk}, Anemia={anemia}

Task:
1) medical_analyst: interpret stage/albuminuria and red flags (e.g., potassium ≥6.0).
2) diet_specialist: stage- and ACR-aware guidance (protein, sodium, potassium).
3) exercise_coach: moderate plan; contraindications if anemia/hyperkalemia present.
4) care_coordinator: suggest lab follow-ups (e.g., repeat ACR), timing, non-directive med chats (ACEi/ARB note).

Constraints:
- No medication dosing.
- Include a short, readable 'next steps' list.
- Add: 'This is not medical advice; consult your clinician.'
"""

        if st.button("Generate AI next steps", key="btn_ai_next_steps"):
            text = call_llm(system_prompt, user_prompt)
            if text:
                text_html = nl2br(text)
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <div style="font-weight:800;color:var(--brand-800)">AI next steps — {MODEL_KEYS.get(chosen_key, chosen_key)}</div>
                        <div class="small-muted">Prob_CKD {prob_ckd:.3f}</div>
                      </div>
                      <div style="line-height:1.55">{text_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.download_button(
                    "Download next steps (.txt)",
                    data=text.encode("utf-8"),
                    file_name="ckd_recommendations.txt",
                    mime="text/plain",
                    key="dl_next_steps"
                )
                _log("Recommendations generated.")
            else:
                st.error("LLM did not return text. Check API key/credits and model name in `.streamlit/secrets.toml`.")
                _log("Recommendations FAILED.")

# =========================
# Digital Twin (What-If) — uses /whatif
# =========================
with tab_digital_twin:
    st.markdown("### Digital Twin — What-If Scenarios")
    st.markdown(
        "Try **small nudges** (change one or two numbers) or run a **grid sweep** (try a list of values per knob). "
        "Designed for patients and lab assistants: clear flags, simple summaries, and charts."
    )
    with st.expander("What is a grid sweep?", expanded=False):
        st.markdown(GRID_SWEEP_HELP)

    # ---- Baseline selection: last single OR CSV upload
    base_from_single = st.session_state.get("last_pred_payload") or {}
    st.markdown("#### Baseline")
    cbase1, cbase2 = st.columns([2,1])

    with cbase1:
        if base_from_single:
            st.success("Using the last single-check as baseline. You can override below.")
        else:
            st.info("No single-check found. Upload a 1-row CSV baseline or paste JSON below.")

        csv_up = st.file_uploader("Optional: upload a 1-row CSV baseline", type=["csv"], key="dt_csv")
        base_csv = parse_csv_single_row(csv_up) if csv_up else None

        raw_base_default = json.dumps(base_csv or base_from_single, indent=2) if (base_csv or base_from_single) else ""
        base_json = st.text_area(
            "Baseline (JSON, optional)",
            value=raw_base_default,
            placeholder="Paste baseline JSON or leave empty if using last single / CSV…",
            key="dt_baseline_json"
        )
        use_base = sanitize_payload(safe_json_loads(base_json, fallback=(base_csv or base_from_single or {})))

    with cbase2:
        template_blank = pd.DataFrame(columns=FEATURE_COLUMNS)
        st.download_button(
            "Download baseline CSV template",
            data=template_blank.to_csv(index=False).encode("utf-8"),
            file_name="ckd_baseline_template.csv",
            mime="text/csv",
            use_container_width=True,
            key="dt_dl_baseline_template"
        )

    if not use_base:
        st.warning("Provide a baseline using the last single prediction, a 1-row CSV, or paste JSON above.")
        st.stop()

    # ---- Deltas vs Grid configuration
    st.divider()
    st.markdown("#### Choose a simulation mode")

    left, right = st.columns(2)
    with left:
        st.markdown("**Quick nudges** (applied to baseline)")
        d_sbp = st.number_input("Δ Systolic BP (mmHg)", -40, 40, 0, key="dt_d_sbp")
        d_dbp = st.number_input("Δ Diastolic BP (mmHg)", -30, 30, 0, key="dt_d_dbp")
        d_gfr = st.number_input("Δ GFR (mL/min/1.73m²)", -50, 50, 0, key="dt_d_gfr")
        d_acr = st.number_input("Δ ACR (mg/g)", -500, 500, 0, key="dt_d_acr")
        d_k   = st.number_input("Δ Potassium (mEq/L)", -2.0, 2.0, 0.0, step=0.1, key="dt_d_k")
        d_a1c = st.number_input("Δ HbA1c (%)", -3.0, 3.0, 0.0, step=0.1, key="dt_d_a1c")
        deltas = {
            "systolicbp": d_sbp, "diastolicbp": d_dbp, "gfr": d_gfr, "acr": d_acr,
            "serumelectrolytespotassium": d_k, "hba1c": d_a1c,
        }

    with right:
        st.markdown("**Grid sweep** (try lists of target values)")
        st.caption("We’ll simulate every combination you provide below.")
        grid_sbp = st.text_input("SBP list", "120,130,140", key="dt_grid_sbp")
        grid_gfr = st.text_input("GFR list", "30,45,60,75,90", key="dt_grid_gfr")
        grid_acr = st.text_input("ACR list", "10,30,300", key="dt_grid_acr")
        grid = {}
        try:
            if grid_sbp.strip(): grid["systolicbp"] = [float(x) for x in grid_sbp.split(",") if x.strip()]
            if grid_gfr.strip(): grid["gfr"] = [float(x) for x in grid_gfr.split(",") if x.strip()]
            if grid_acr.strip(): grid["acr"] = [float(x) for x in grid_acr.split(",") if x.strip()]
        except Exception as e:
            st.error(f"Grid parse error: {e}")

    model_for_sim = st.selectbox("Model for simulation", [MODEL_KEYS.get(m,m) for m in MODEL_KEYS], index=1, key="dt_model_select")
    model_key_sim = MODEL_LABELS.get(model_for_sim, "rf")

    cA, cB = st.columns(2)

    # ---- Single what-if action (robust to backend shape)
    with cA:
        if st.button("Run single what-if", key="btn_dt_single"):
            body = {"base": use_base, "deltas": deltas, "model": model_key_sim}
            try:
                out = _api_post(f"{st.session_state['api_url']}/whatif", json_body=body, timeout=60)

                # Backend may return separate rows & probs
                row = (out.get("rows") or [{}])[0]
                prob_from_block = None
                try:
                    prob_from_block = float((out.get("probs") or [{}])[0].get("prob_ckd"))
                except Exception:
                    pass
                prob = float(row.get("prob_ckd", prob_from_block or 0.0))

                thr = float(out.get("threshold_used", 0.5))
                flag = prob >= thr

                badge_cls = "is-bad" if flag else "is-ok"
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <div style="font-size:18px;font-weight:800;color:var(--brand-800);">{MODEL_KEYS.get(model_key_sim, model_key_sim)}</div>
                        <div class="badge {badge_cls}">{'Flagged' if flag else 'Not flagged'}</div>
                      </div>
                      <div class="kpi">
                        <div class="metric"><b>Probability</b><span>{prob:.3f}</span></div>
                        <div class="metric"><b>Decision threshold</b><span>{thr:.3f}</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                changed = []
                for k, dv in (deltas or {}).items():
                    if float(dv) != 0.0:
                        changed.append(f"- **{pretty_feature_name(k)}** {'+' if dv>=0 else ''}{dv}")
                if changed:
                    st.markdown("**Changes applied**")
                    st.markdown("\n".join(changed))
                else:
                    st.caption("No deltas applied; baseline only.")

                # Save Digital Twin context (single what-if)
                try:
                    st.session_state["ctx_digital_twin"] = {
                        "schema_version": 1,
                        "model": model_key_sim,
                        "threshold": thr,
                        "swept_features": list((deltas or {}).keys()),
                        "summary": {
                            "n": 1,
                            "n_flag": int(flag),
                            "min_prob": float(prob),
                            "max_prob": float(prob)
                        },
                        "rows_sample": [
                            {**{k: use_base.get(k) for k in FEATURE_COLUMNS if k in use_base},
                             **{"prob_ckd": float(prob), "flag": bool(flag)}}
                        ]
                    }
                except Exception:
                    pass

            except requests.HTTPError as e:
                msg = getattr(e.response, "text", str(e)) or str(e)
                if "simulate_whatif not available" in msg or getattr(e.response, "status_code", 0) == 503:
                    st.info("What-if is disabled on the server. Add `api/digital_twin.py` and restart the API.")
                else:
                    st.error("What-if failed")
                    st.code(msg, language="json")
            except Exception as e:
                st.error(f"What-if error: {e}")

    # ---- Grid sweep action (merge probs → rows, compute flags)
    with cB:
        if st.button("Run grid sweep", key="btn_dt_grid"):
            if not grid:
                st.warning("Enter at least one list (e.g., SBP or GFR) for a grid sweep.")
            else:
                body = {"base": use_base, "grid": grid, "model": model_key_sim}
                try:
                    out = _api_post(f"{st.session_state['api_url']}/whatif", json_body=body, timeout=120)
                    rows = out.get("rows", []) or []
                    probs = out.get("probs", []) or []
                    thr = float(out.get("threshold_used", 0.5))
                    swept_features = out.get("swept_features") or out.get("features") or list(grid.keys())

                    if not rows:
                        st.info("No rows returned.")
                    else:
                        df_rows = pd.DataFrame(rows)
                        df_probs = pd.DataFrame(probs) if probs else pd.DataFrame([{}] * len(df_rows))
                        if len(df_probs) != len(df_rows):
                            df_probs = df_probs.reindex(range(len(df_rows))).fillna(method="ffill").fillna(method="bfill")
                        df = pd.concat([df_rows.reset_index(drop=True), df_probs.reset_index(drop=True)], axis=1)

                        if "prob_ckd" not in df.columns:
                            st.warning("Backend did not return per-row probabilities; showing raw inputs only.")
                            df["prob_ckd"] = float("nan")
                        df["flag"] = df["prob_ckd"].apply(lambda p: bool(p >= thr) if pd.notna(p) else False)

                        cols_view = [*swept_features, "prob_ckd", "flag"]
                        cols_view = [c for c in cols_view if c in df.columns]
                        view = df[cols_view].copy()

                        n = len(view)
                        n_flag = int(view["flag"].sum()) if "flag" in view else 0
                        best = float(view["prob_ckd"].min()) if view["prob_ckd"].notna().any() else float("nan")
                        worst = float(view["prob_ckd"].max()) if view["prob_ckd"].notna().any() else float("nan")
                        st.markdown(
                            f"**{n} combinations** • **{n_flag} flagged ≥ {thr:.2f}** • "
                            f"min prob {best:.3f} • max prob {worst:.3f}"
                        )

                        st.dataframe(view, use_container_width=True)

                        if len(swept_features) == 1 and "prob_ckd" in view.columns:
                            f = swept_features[0]
                            try:
                                fig = px.line(view.sort_values(f), x=f, y="prob_ckd", markers=True,
                                              title=f"Risk vs {pretty_feature_name(f)} (thr {thr:.2f})")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                        elif len(swept_features) == 2 and "prob_ckd" in view.columns:
                            f1, f2 = swept_features
                            try:
                                pivot = view.pivot_table(index=f1, columns=f2, values="prob_ckd", aggfunc="mean")
                                fig = px.imshow(pivot, aspect="auto",
                                                labels=dict(color="Prob CKD"),
                                                title=f"Risk heatmap ({pretty_feature_name(f1)} × {pretty_feature_name(f2)})")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                        else:
                            st.caption("Tip: choose 1 feature for a line chart, 2 for a heatmap. More features still show in the table.")

                        st.download_button(
                            "Download grid results (CSV)",
                            data=view.to_csv(index=False).encode("utf-8"),
                            file_name="ckd_grid_sweep_results.csv",
                            mime="text/csv",
                            key="dl_grid_results"
                        )

                        # Save Digital Twin context (grid sweep)
                        try:
                            sample_rows = view.head(20).to_dict(orient="records")
                            st.session_state["ctx_digital_twin"] = {
                                "schema_version": 1,
                                "model": model_key_sim,
                                "threshold": thr,
                                "swept_features": swept_features,
                                "summary": {"n": int(len(view)),
                                            "n_flag": int(view["flag"].sum()) if "flag" in view else 0,
                                            "min_prob": float(view["prob_ckd"].min()) if view["prob_ckd"].notna().any() else None,
                                            "max_prob": float(view["prob_ckd"].max()) if view["prob_ckd"].notna().any() else None},
                                "rows_sample": sample_rows
                            }
                        except Exception:
                            pass

                except requests.HTTPError as e:
                    msg = getattr(e.response, "text", str(e)) or str(e)
                    if "simulate_whatif not available" in msg or getattr(e.response, "status_code", 0) == 503:
                        st.info("Grid what-if is disabled on the server. Add `api/digital_twin.py` and restart the API.")
                    else:
                        st.error("Grid what-if failed")
                        st.code(msg, language="json")
                except Exception as e:
                    st.error(f"Grid what-if error: {e}")

# =========================
# Counterfactuals — uses /counterfactual
# =========================
with tab_counterfactuals:
    st.markdown("### Counterfactuals — Actionable tweaks to get below a target")
    st.caption("We’ll propose small, realistic changes (BP, A1c, ACR, etc.) that move risk below your chosen target.")

    last_base = st.session_state.get("last_pred_payload") or {}
    c1, c2 = st.columns([2,1])
    with c1:
        if last_base:
            st.success("Using the last single-check as baseline. You can override below.")
        else:
            st.info("No single-check yet. Upload a 1-row CSV or paste JSON below.")

        cf_csv = st.file_uploader("Optional: upload a 1-row CSV baseline", type=["csv"], key="cf_csv")
        csv_row = None
        if cf_csv:
            try:
                dfx = pd.read_csv(cf_csv)
                if not dfx.empty:
                    for c in FEATURE_COLUMNS:
                        if c not in dfx.columns: dfx[c] = 0.0
                    csv_row = sanitize_payload(dfx[FEATURE_COLUMNS].iloc[0].to_dict())
            except Exception as e:
                st.error(f"CSV read failed: {e}")

        base_json = st.text_area(
            "Baseline JSON (optional)",
            value=json.dumps(csv_row or last_base, indent=2) if (csv_row or last_base) else "",
            placeholder="Paste baseline JSON or leave empty…",
            key="cf_baseline_json"
        )
        use_base = sanitize_payload(safe_json_loads(base_json, fallback=(csv_row or last_base or {})))

    with c2:
        tmpl = pd.DataFrame(columns=FEATURE_COLUMNS)
        st.download_button(
            "Download baseline CSV template",
            data=tmpl.to_csv(index=False).encode("utf-8"),
            file_name="ckd_cf_baseline_template.csv",
            mime="text/csv",
            use_container_width=True,
            key="cf_dl_tmpl"
        )

    if not use_base:
        st.warning("Provide a baseline using last single, CSV, or JSON above.")
        st.stop()

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        target_prob = st.number_input("Target probability (≤)", 0.0, 1.0, 0.20, step=0.01, key="cf_target_prob")
    with col2:
        method = st.selectbox("Method", ["auto", "greedy"], index=0, key="cf_method")
    with col3:
        model_for_cf = st.selectbox("Model", [MODEL_KEYS.get(m,m) for m in MODEL_KEYS], index=1, key="cf_model_select")
    model_key_cf = MODEL_LABELS.get(model_for_cf, "rf")

    if st.button("Compute counterfactual", key="btn_cf_compute"):
        body = {"base": use_base, "target_prob": target_prob, "model": model_key_cf, "method": method}
        try:
            out = _api_post(f"{st.session_state['api_url']}/counterfactual", json_body=body, timeout=120)

            thr = float(out.get("threshold_used", 0.5))
            p0  = float(out.get("initial_prob", 0.0))
            pf  = float(out.get("final_prob", 0.0))
            flag = bool(out.get("final_flag", pf >= thr))

            st.markdown(
                f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <div style="font-weight:800;color:var(--brand-800)">{MODEL_KEYS.get(model_key_cf, model_key_cf)} plan</div>
                        <div class="badge {'is-bad' if flag else 'is-ok'}">{'Flagged at end' if flag else 'Below threshold'}</div>
                    </div>
                    <div class="kpi">
                        <div class="metric"><b>Start prob</b><span>{p0:.3f}</span></div>
                        <div class="metric"><b>Target prob</b><span>{float(out.get('target_prob', target_prob)):.3f}</span></div>
                        <div class="metric"><b>Final prob</b><span>{pf:.3f}</span></div>
                        <div class="metric"><b>Threshold</b><span>{thr:.3f}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            steps = out.get("steps", []) or []
            if steps:
                df_steps = pd.DataFrame(steps)
                if "feature" in df_steps.columns:
                    df_steps["feature"] = df_steps["feature"].map(_cf_feature_nice)
                st.markdown("**Change path**")
                st.table(df_steps)

                grp = df_steps.groupby("feature", dropna=False)["delta"].sum().reset_index()
                grp["delta"] = grp["delta"].map(lambda x: f"{x:+.3f}")
                st.markdown("**Summary of adjustments**")
                st.table(grp)

            final_cand = out.get("final_candidate") or out.get("result") or out.get("best", {}).get("candidate")
            if isinstance(final_cand, dict):
                st.markdown("**Final candidate — key vitals (readable)**")
                rows = []

                def _add_row(label: str, key: str, unit: str, nd: int):
                    if key in final_cand:
                        try:
                            v = round(float(final_cand[key]), nd)
                            if nd == 0:
                                v = int(v)
                            rows.append({"Measure": label, "Value": f"{v}", "Unit": unit})
                        except Exception:
                            rows.append({"Measure": label, "Value": str(final_cand[key]), "Unit": unit})

                _add_row("Systolic BP", "systolicbp", "mmHg", 0)
                _add_row("Diastolic BP", "diastolicbp", "mmHg", 0)
                _add_row("GFR", "gfr", "mL/min/1.73m²", 0)
                _add_row("ACR", "acr", "mg/g", 0)
                _add_row("Potassium", "serumelectrolytespotassium", "mEq/L", 1)
                _add_row("HbA1c", "hba1c", "%", 1)
                _add_row("Hemoglobin", "hemoglobinlevels", "g/dL", 1)

                st.table(pd.DataFrame(rows))
            else:
                st.caption("No final candidate returned.")

            # Save Counterfactual context
            try:
                st.session_state["ctx_counterfactual"] = {
                    "schema_version": 1,
                    "model": model_key_cf,
                    "threshold": thr,
                    "target_prob": float(out.get("target_prob", target_prob)),
                    "initial_prob": float(out.get("initial_prob", 0.0)),
                    "final_prob": float(out.get("final_prob", 0.0)),
                    "converged": bool(out.get("converged", pf <= float(out.get("target_prob", target_prob)))),
                    "steps": out.get("steps", [])[:50],
                    "final_candidate": final_cand if isinstance(final_cand, dict) else None
                }
            except Exception:
                pass

        except requests.HTTPError as e:
            msg = getattr(e.response, "text", str(e)) or str(e)
            if "counterfactual not available" in msg or getattr(e.response, "status_code", 0) == 503:
                st.info("Counterfactuals are disabled on the server. Add `api/counterfactuals.py` and restart the API.")
            else:
                st.error("Counterfactual failed")
                st.code(msg, language="json")
        except Exception as e:
            st.error(f"Counterfactual error: {e}")

# =========================
# Similar Patients — uses /similar
# =========================
with tab_similarity:
    st.markdown("Find **nearest neighbors** in an uploaded cohort to compare similar profiles.")
    base = st.session_state.get("last_pred_payload") or {}
    if not base:
        st.info("Run a single prediction first to set a baseline, or paste JSON below.")
    base_json = st.text_area(
        "Baseline JSON (optional, overrides last single payload if provided)",
        value=json.dumps(base, indent=2),
        key="sim_baseline_json"
    )
    use_base = sanitize_payload(safe_json_loads(base_json, fallback=base or {}))

    cohort_file = st.file_uploader("Upload cohort CSV to search in", type=["csv"], key="sim_file")
    k = st.slider("Top-k similar", 1, 50, 5, key="sim_k")

    if st.button("Find similar", key="btn_find_similar"):
        try:
            cohort = []
            if cohort_file:
                dfc = pd.read_csv(cohort_file)
                for c in FEATURE_COLUMNS:
                    if c not in dfc.columns:
                        dfc[c] = 0.0
                dfc = dfc[FEATURE_COLUMNS]
                cohort = [sanitize_payload(r) for r in dfc.to_dict(orient="records")]
            body = {"base": use_base or base, "cohort": cohort}
            out = _api_post(f"{st.session_state['api_url']}/similar", params={"k": k}, json_body=body, timeout=60)
            st.success("Similar patients computed.")
            if isinstance(out, dict) and "neighbors" in out:
                nb = out.get("neighbors", [])
                st.table(pd.DataFrame(nb))

                # Save Similar Patients context
                try:
                    sample = nb[:20] if isinstance(nb, list) else []
                    df_nb = pd.DataFrame(nb) if nb else pd.DataFrame()
                    st.session_state["ctx_similar"] = {
                        "schema_version": 1,
                        "k": int(k),
                        "metric": out.get("metric", "euclidean"),
                        "neighbors_sample": sample,
                        "summary": {
                            "count": len(nb),
                            "median_prob": float(df_nb["prob_ckd"].median()) if ("prob_ckd" in df_nb.columns and not df_nb.empty) else None,
                            "min_prob": float(df_nb["prob_ckd"].min()) if ("prob_ckd" in df_nb.columns and not df_nb.empty) else None,
                            "max_prob": float(df_nb["prob_ckd"].max()) if ("prob_ckd" in df_nb.columns and not df_nb.empty) else None
                        }
                    }
                except Exception:
                    pass
            else:
                st.json(out)
        except requests.HTTPError as e:
            msg = getattr(e.response, "text", str(e)) or str(e)
            if "knn_similar not available" in msg or e.response.status_code == 503:
                st.info("Similarity search is disabled on the server. Add `api/similarity.py` and restart the API.")
            else:
                st.error("Similarity failed")
                st.code(msg, language="json")
        except Exception as e:
            st.error(f"Similarity error: {e}")

# =========================
# Care Plan (Agents) — uses /agents/plan
# =========================
with tab_agents:
    st.markdown("LLM multi-agent **care plan** generated on the server (via `/agents/plan`).")
    payload = st.session_state.get("last_pred_payload")
    results = st.session_state.get("last_pred_results", {})

    if not payload or not results:
        st.info("Run a single prediction first so we can build the summary for agents.")
    else:
        avail = list(results.keys())
        chosen = st.selectbox("Use result from model", [MODEL_KEYS.get(m, m) for m in avail], index=0, key="agents_model_pick")
        chosen_key = next((k for k in avail if MODEL_KEYS.get(k, k) == chosen), avail[0])
        res = results[chosen_key]

        summary = {
            "model": chosen_key,
            "prob_ckd": float(res.get("prob_ckd", 0.0)),
            "threshold": float(res.get("threshold_used", 0.5)),
            "flags": {
                "bp_risk": int(payload.get("bp_risk", 0)),
                "hyperkalemia": int(payload.get("hyperkalemiaflag", 0)),
                "anemia": int(payload.get("anemiaflag", 0)),
            },
            "stage": int(payload.get("ckdstage", 0)),
            "albuminuria_cat": int(payload.get("albuminuriacat", 0)),
            "metrics": {
                "gfr": float(payload.get("gfr", 0)),
                "acr": float(payload.get("acr", 0)),
                "hba1c": float(payload.get("hba1c", 0)),
                "systolicbp": float(payload.get("systolicbp", 0)),
                "diastolicbp": float(payload.get("diastolicbp", 0)),
                "potassium": float(payload.get("serumelectrolytespotassium", 0)),
                "hb": float(payload.get("hemoglobinlevels", 0)),
            }
        }

        if st.button("Generate care plan (server agents)", key="btn_agents_plan"):
            try:
                out = _api_post(f"{st.session_state['api_url']}/agents/plan", json_body=summary, timeout=120)
                st.success("Care plan generated.")
                if isinstance(out, dict) and out.get("sections"):
                    for sec in out["sections"]:
                        st.markdown(f"**{sec.get('title','Section')}**")
                        if "bullets" in sec and isinstance(sec["bullets"], list):
                            st.markdown("\n".join([f"- {b}" for b in sec["bullets"]]))
                        elif "text" in sec:
                            st.markdown(nl2br(str(sec["text"])), unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.json(out)
            except requests.HTTPError as e:
                msg = getattr(e.response, "text", str(e)) or str(e)
                if "multi_agent_plan not available" in msg or e.response.status_code == 503:
                    st.info("Server agents are disabled. Add `api/agents.py` and restart the API.")
                else:
                    st.error("Agents plan failed")
                    st.code(msg, language="json")
            except Exception as e:
                st.error(f"Agents error: {e}")
