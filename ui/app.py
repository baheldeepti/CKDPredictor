# ui/app.py
import os
import json
import requests
import streamlit as st
import pandas as pd

# =========================
# Config
# =========================
API_URL_DEFAULT = "http://0.0.0.0:8000"
API_URL = st.secrets.get("API_URL", API_URL_DEFAULT)

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
MODEL_LABELS = {k: v for k, v in MODEL_CHOICES}  # label->key
MODEL_KEYS = {v: k for k, v in MODEL_CHOICES}    # key->label

# Optional LLM config (OpenAI-compatible)
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openrouter")  # openai | together | openrouter | custom
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = st.secrets.get("LLM_MODEL", "openrouter/auto")

# Optional branding for OpenRouter headers
APP_URL   = st.secrets.get("APP_URL", "https://github.com/baheldeepti/CKDPredictor")
APP_TITLE = st.secrets.get("APP_TITLE", "CKD Predictor")

st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="wide")

# =========================
# Helpers
# =========================
def _result_card(label: str, prob_ckd: float, thr: float, model_used: str):
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:12px;border:1px solid #e6e6e6;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:18px;font-weight:700;">{MODEL_KEYS.get(model_used, model_used)}</div>
            <div style="font-size:14px;font-weight:700;">
              <span style="padding:2px 8px;border-radius:999px;background:{'#fee2e2' if label=='CKD' else '#dcfce7'};color:{'#991b1b' if label=='CKD' else '#065f46'}">
                {label}
              </span>
            </div>
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div><strong>Prob CKD</strong><br><span style="font-size:20px;">{prob_ckd:.3f}</span></div>
            <div><strong>Threshold</strong><br><span style="font-size:20px;">{thr:.3f}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _call_openrouter_chat(system_prompt: str, user_prompt: str) -> str | None:
    """Direct OpenRouter call with required headers."""
    if not LLM_API_KEY:
        st.warning("No LLM API key configured. Add LLM_API_KEY in `.streamlit/secrets.toml`.")
        return None
    url = (LLM_BASE_URL.rstrip("/") + "/chat/completions") if LLM_BASE_URL else "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"LLM call failed [{r.status_code}]: {r.text}")
            return None
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", None)
    except Exception as e:
        st.error(f"LLM network error: {e}")
        return None

def call_llm(system_prompt: str, user_prompt: str):
    """OpenRouter by default; fall back to OpenAI SDK style if not using OpenRouter."""
    if LLM_PROVIDER.lower() == "openrouter":
        with st.spinner("Generating cohort insights…"):
            return _call_openrouter_chat(system_prompt, user_prompt)

    if not LLM_API_KEY:
        st.warning("No LLM API key configured. Add LLM_API_KEY in `.streamlit/secrets.toml`.")
        return None

    try:
        from openai import OpenAI
    except Exception:
        st.error("Package `openai` missing. Install with: `pip install openai`")
        return None

    try:
        with st.spinner("Generating cohort insights…"):
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

def sample_records_df():
    rows = [
        [65, 1, 148, 86, 2.1, 32, 52, 120, 139, 4.8, 12.2, 6.5, 62, 15.2, 2, 2, 1, 0, 0],
        [54, 0, 132, 82, 1.6, 28, 65, 35, 141, 4.3, 13.5, 6.1, 50, 17.5, 2, 2, 1, 0, 0],
        [72, 1, 160, 95, 3.4, 48, 38, 420, 137, 5.7, 11.2, 7.8, 65, 12.5, 3, 3, 1, 1, 1],
        [43, 0, 118, 74, 0.9, 18, 92, 10, 142, 3.9, 13.9, 5.2, 44, 20.0, 1, 1, 0, 0, 0],
        [58, 1, 170, 110, 4.6, 60, 28, 750, 135, 6.2, 10.8, 8.3, 60, 13.0, 4, 3, 1, 1, 1],
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

# =========================
# Session state
# =========================
if "last_pred_payload" not in st.session_state:
    st.session_state["last_pred_payload"] = None
if "last_pred_results" not in st.session_state:
    st.session_state["last_pred_results"] = {}  # per-model
if "batch_preds" not in st.session_state:
    st.session_state["batch_preds"] = {}        # per-model DataFrame
if "batch_insights" not in st.session_state:
    st.session_state["batch_insights"] = None

# =========================
# Header + Primary Controls (no sidebar)
# =========================
st.title("🩺 CKD Predictor")
st.caption(
    "Built by a data engineer with 12+ years of ML systems experience — single & batch predictions, model comparison, "
    "explainability, and LLM-driven cohort insights."
)

with st.container():
    colA, colB, colC, colD = st.columns([2.3, 2.3, 1.6, 1.8])
    with colA:
        api_url = st.text_input("API URL", API_URL, help="FastAPI base. Example: http://0.0.0.0:8000")
    with colB:
        model_labels = [label for label, _ in MODEL_CHOICES]
        picked_labels = st.multiselect(
            "Select models to use",
            model_labels,
            default=[model_labels[0]],
            help="Pick one or more models. We'll compare outputs side-by-side."
        )
        selected_models = [MODEL_LABELS[l] for l in picked_labels] or ["xgb"]
    with colC:
        if st.button("Health"):
            try:
                h = requests.get(f"{api_url}/health", params={"model": selected_models[0]}, timeout=10).json()
                if h.get("status") == "ok":
                    st.success("OK")
                else:
                    st.warning("Check API")
                st.caption(f"Model: {h.get('model')} • Thr: {h.get('threshold')} • DB log: {h.get('db_logging')}")
            except Exception as e:
                st.error("Err")
                st.caption(str(e))
    with colD:
        if st.button("Rebuild models now"):
            st.caption("Rebuild retrains models on the server using accumulated historical inputs (if DB is configured). "
                       "After it finishes, click Reload to hot-swap artifacts.")
            try:
                r = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
                st.success("Started") if r.get("status") == "started" else st.info(r)
            except Exception as e:
                st.error("Failed to trigger retrain.")
                st.caption(str(e))

    # row 2 actions
    colE, colF = st.columns([1.2, 1.2])
    with colE:
        if st.button("Reload models"):
            try:
                r = requests.post(f"{api_url}/admin/reload", timeout=20)
                st.success("Reloaded model cache.")
            except Exception as e:
                st.error("Reload failed.")
                st.caption(str(e))
    with colF:
        st.info("**What do these buttons do?**  \n"
                "**Rebuild models now**: kicks off server-side retraining.  \n"
                "**Reload models**: clears in-memory cache so the API picks up the latest model files.", icon="ℹ️")

st.divider()

# =========================
# Tabs
# =========================
tab_single, tab_batch, tab_metrics, tab_advice = st.tabs(
    ["Single prediction (compare models)",
     "Batch predictions (compare models)",
     "Metrics",
     "Recommendations (alpha)"]
)

# =========================
# Single prediction (Compare)
# =========================
with tab_single:
    st.markdown("Fill the form, then **Predict with selected models**. Compare outputs and explanations side-by-side.")
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
            f"Derived → PulsePressure={pulsepressure} • Urea/Creatinine={ureacreatinineratio:.2f} • "
            f"Flags: BP={bp_risk}, HK={hyperkalemiaflag}, Anemia={anemiaflag} • "
            f"Stage={ckdstage}, Albuminuria={albuminuriacat}"
        )

        do_predict = st.form_submit_button("Predict with selected models")

    if do_predict:
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
        st.session_state["last_pred_payload"] = payload
        st.session_state["last_pred_results"] = {}

        cols = st.columns(len(selected_models))
        for idx, m in enumerate(selected_models):
            try:
                r = requests.post(f"{api_url}/predict", params={"model": m}, json=payload, timeout=20)
                r.raise_for_status()
                res = r.json()
                st.session_state["last_pred_results"][m] = res
                label = "CKD" if res.get("prediction", 1) == 1 else "Non-CKD"
                with cols[idx]:
                    _result_card(label, float(res.get("prob_ckd", 0.0)), float(res.get("threshold_used", 0.5)), res.get("model_used", m))
            except Exception as e:
                with cols[idx]:
                    st.error(f"{MODEL_KEYS.get(m,m)} failed")
                    st.caption(str(e))

        # Explainability row (per selected model)
        st.markdown("#### Explain prediction (top drivers)")
        ecols = st.columns(len(selected_models))
        for idx, m in enumerate(selected_models):
            if m not in st.session_state["last_pred_results"]:
                continue
            with ecols[idx]:
                with st.spinner(f"SHAP: {MODEL_KEYS.get(m,m)}…"):
                    try:
                        er = requests.post(f"{api_url}/explain", params={"model": m}, json=payload, timeout=30)
                        if er.status_code == 404:
                            st.info("API has no /explain yet.")
                            continue
                        er.raise_for_status()
                        exp = er.json()
                        top = exp.get("top", [])
                        if not top:
                            shap_map = exp.get("shap_values", {})
                            top = sorted(
                                [{"feature": k, "impact": abs(v), "signed": v} for k, v in shap_map.items()],
                                key=lambda x: x["impact"], reverse=True
                            )[:8]
                        if top:
                            df_top = pd.DataFrame(top)
                            if "impact" not in df_top.columns and "signed" in df_top.columns:
                                df_top["impact"] = df_top["signed"].abs()
                            st.bar_chart(df_top.set_index("feature")["impact"])
                            # compact bullets
                            st.markdown(
                                "\n".join([
                                    f"- **{row['feature']}** {'↑' if row.get('signed',0)>0 else '↓'} risk (SHAP={row.get('signed',0):+.3f})"
                                    for row in top[:5]
                                ])
                            )
                        else:
                            st.info("No features available.")
                    except requests.HTTPError as e:
                        st.error("Explain failed")
                        st.caption(getattr(e, 'response', None) and e.response.text)
                    except Exception as e:
                        st.error("Explain error")
                        st.caption(str(e))

# =========================
# Batch predictions (Compare)
# =========================
with tab_batch:
    st.markdown("Upload a CSV with the required columns, run batch predictions across **all selected models**, and compare metrics.")

    left, right = st.columns([2, 1])
    with left:
        st.caption("Required columns:")
        st.code(",".join(FEATURE_COLUMNS), language="text")
    with right:
        template_blank = pd.DataFrame(columns=FEATURE_COLUMNS)
        sample5 = sample_records_df()
        st.download_button("Blank Template CSV",
                           data=template_blank.to_csv(index=False).encode("utf-8"),
                           file_name="ckd_template_blank.csv", mime="text/csv")
        st.download_button("Sample CSV (5 rows)",
                           data=sample5.to_csv(index=False).encode("utf-8"),
                           file_name="ckd_sample_5rows.csv", mime="text/csv")

    file = st.file_uploader("Upload CSV", type=["csv"])
    retrain_after_batch = st.checkbox("Retrain after batch", value=False,
                                      help="Optionally kick off server-side retraining after this batch finishes.")

    if file:
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.dataframe(df.head())

                if st.button("Run Batch with selected models"):
                    st.session_state["batch_preds"] = {}
                    summary_rows = []
                    for m in selected_models:
                        try:
                            rows = df.to_dict(orient="records")
                            r = requests.post(
                                f"{api_url}/predict/batch",
                                params={"model": m},
                                json={"rows": rows},
                                timeout=90
                            )
                            r.raise_for_status()
                            preds = pd.DataFrame(r.json()["predictions"])
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
                        except requests.HTTPError as e:
                            st.error(f"{MODEL_KEYS.get(m,m)} failed")
                            st.caption(getattr(e, 'response', None) and e.response.text)
                        except Exception as e:
                            st.error(f"{MODEL_KEYS.get(m,m)} error")
                            st.caption(str(e))

                    if summary_rows:
                        st.success("Batch complete.")
                        st.markdown("#### Model comparison (batch summary)")
                        st.table(pd.DataFrame(summary_rows))

                        # Downloads per model and merged comparison
                        merged_list = []
                        for m, preds in st.session_state["batch_preds"].items():
                            merged = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                            merged_list.append(merged.assign(model_used=m))
                            st.download_button(
                                f"Download results • {MODEL_KEYS.get(m,m)}",
                                data=merged.to_csv(index=False).encode("utf-8"),
                                file_name=f"ckd_batch_predictions_{m}.csv",
                                mime="text/csv",
                                key=f"dl_{m}"
                            )
                        if merged_list:
                            all_merged = pd.concat(merged_list, axis=0, ignore_index=True)
                            st.download_button(
                                "Download ALL results (combined)",
                                data=all_merged.to_csv(index=False).encode("utf-8"),
                                file_name="ckd_batch_predictions_all_models.csv",
                                mime="text/csv",
                            )

                        if retrain_after_batch:
                            try:
                                rr = requests.post(f"{api_url}/admin/retrain", timeout=10).json()
                                st.success("Retraining started. Use Health at top to check when updated.")
                            except Exception as e:
                                st.error("Retrain call failed.")
                                st.caption(str(e))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # --- Cohort insights (AI) — compact card, cached ---
    st.markdown("#### Cohort insights (AI)")
    # Pick which model’s results feed the insights (default: first available)
    available_models = list(st.session_state.get("batch_preds", {}).keys())
    if not available_models:
        st.info("Run a batch first to enable insights.")
    else:
        insights_model = st.selectbox(
            "Choose model output to summarize",
            [MODEL_KEYS.get(m, m) for m in available_models],
            index=0
        )
        chosen_m = next((k for k in available_models if MODEL_KEYS.get(k, k) == insights_model), available_models[0])

        preds = st.session_state["batch_preds"][chosen_m]
        pos_rate = preds["prediction"].mean() if "prediction" in preds else 0.0
        avg_prob = preds["prob_ckd"].mean() if "prob_ckd" in preds else 0.0

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Summarize cohort"):
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
                else:
                    st.error("LLM did not return text. Check API key/credits and model name in `.streamlit/secrets.toml`.")
        with c2:
            if st.button("Clear cached summary"):
                st.session_state["batch_insights"] = None

        if st.session_state["batch_insights"]:
            st.markdown(
                f"""
                <div style="border:1px solid #e6e6e6;border-radius:12px;padding:16px;background:#fafafa">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div style="font-weight:700">Cohort Insights — {MODEL_KEYS.get(chosen_m, chosen_m)}</div>
                    <div style="font-size:12px;color:#666">Positive rate {pos_rate:.1%} • Avg Prob_CKD {avg_prob:.3f} • Rows {len(preds)}</div>
                  </div>
                  <div style="line-height:1.55">{st.session_state["batch_insights"].replace('\n','<br/>')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.download_button(
                "Download insights (.txt)",
                data=st.session_state["batch_insights"].encode("utf-8"),
                file_name="ckd_batch_insights.txt",
                mime="text/plain"
            )

# =========================
# Metrics
# =========================
with tab_metrics:
    st.markdown("Monitor service health, recent inferences (requires DB logging), and last retrain status.")
    auto = st.checkbox("Auto-refresh every 10s", value=False)
    if auto:
        st.markdown("<script>setTimeout(() => window.location.reload(), 10000);</script>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Service health")
        try:
            h = requests.get(f"{api_url}/health", params={"model": (selected_models[0] if selected_models else 'xgb')}, timeout=10).json()
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
            li = requests.get(f"{api_url}/metrics/last_inferences?limit=10", timeout=10).json()
            rows = li.get("rows", [])
            if rows:
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No inferences yet — run Single/Batch predictions.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    st.subheader("Model status (last retrain)")
    try:
        rr = requests.get(f"{api_url}/metrics/retrain_report", timeout=10)
        if rr.status_code == 200:
            st.json(rr.json())
        else:
            st.info("No retrain report found. Click **Rebuild models now** above, then **Reload models**.")
    except Exception as e:
        st.error(f"Report fetch failed: {e}")

# =========================
# Recommendations (alpha)
# =========================
with tab_advice:
    st.markdown(
        "AI-assisted recommendations for the **last single prediction**. "
        "Interprets stage, albuminuria, and flags. No medication dosing. "
        "_This is not medical advice._"
    )

    payload = st.session_state.get("last_pred_payload")
    results = st.session_state.get("last_pred_results", {})

    if not payload or not results:
        st.info("Run a single prediction first (any model).")
    else:
        # Let user choose which model’s single result to explain in recs
        avail = list(results.keys())
        chosen = st.selectbox("Use result from model", [MODEL_KEYS.get(m, m) for m in avail], index=0)
        chosen_key = next((k for k in avail if MODEL_KEYS.get(k, k) == chosen), avail[0])

        res = results[chosen_key]
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
Prediction summary (model: {MODEL_KEYS.get(chosen_key, chosen_key)}):
- Label: {label}
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
        if st.button("Generate recommendations"):
            text = call_llm(system_prompt, user_prompt)
            if text:
                st.markdown(
                    f"""
                    <div style="border:1px solid #e6e6e6;border-radius:12px;padding:16px;background:#fafafa">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <div style="font-weight:700">Recommendations — {MODEL_KEYS.get(chosen_key, chosen_key)}</div>
                        <div style="font-size:12px;color:#666">Label: {label} • Prob_CKD {prob_ckd:.3f}</div>
                      </div>
                      <div style="line-height:1.55">{text.replace('\n','<br/>')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("LLM did not return text. Check API key/credits and model name in `.streamlit/secrets.toml`.")
