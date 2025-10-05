# ui/app.py
import os
import json
import math
import requests
import streamlit as st
import pandas as pd

# =========================
# Branding & Config
# =========================
APP_TITLE_HUMAN = "Kidney Health Radar"
APP_REPO_URL    = "https://github.com/baheldeepti/CKDPredictor"

# Prefer hosted API on Render; allow override via Secrets or env.
API_URL_DEFAULT = "https://ckdpredictor.onrender.com"
API_URL = (
    st.secrets.get("API_URL")
    or os.environ.get("CKD_API_URL")
    or API_URL_DEFAULT
)

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

# Optional LLM config (OpenAI-compatible) — used only in the "AI summary & next steps" tab
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openrouter")  # openai | together | openrouter | custom
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = st.secrets.get("LLM_MODEL", "openrouter/auto")

# Optional branding for OpenRouter headers
APP_URL   = st.secrets.get("APP_URL", APP_REPO_URL)
APP_TITLE = st.secrets.get("APP_TITLE", APP_TITLE_HUMAN)

st.set_page_config(page_title=APP_TITLE_HUMAN, page_icon="🩺", layout="wide")

# -------------------------
# Light CSS lift
# -------------------------
STYLE = """
<style>
:root {
  --brand:#2563eb; --ok:#16a34a; --warn:#f59e0b; --bad:#ef4444; --muted:#6b7280; --bg:#f8fafc;
}
html, body, [class^="css"]  {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
section.main > div { padding-top: 0.5rem !important; }
h1, h2, h3 { letter-spacing: -0.01em; }
.small-muted { color: var(--muted); font-size: 12px; }
.badge { padding:2px 10px; border-radius:999px; font-weight:600; display:inline-block; }
.card  { padding:14px; border-radius:12px; border:1px solid #e6e6e6; background:#fff; }
hr     { border: none; border-top:1px solid #eee; margin: 0.5rem 0 1rem; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(0,0,0,0.08); }
.block-info { font-size:13px; color:#334155; }
.top-chip { font-size:12px; color:var(--muted); margin-top:-6px; }
.tooltip { font-size:12px; color:#475569; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def _normalize_api(url: str) -> str:
    if not url:
        return API_URL_DEFAULT
    return url.rstrip("/")

def nl2br(s: str) -> str:
    try:
        return s.replace("\n", "<br/>")
    except Exception:
        return s

def sanitize_payload(d: dict) -> dict:
    out = {}
    for k in FEATURE_COLUMNS:
        v = d.get(k, 0)
        if v is None:
            v = 0.0
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = 0.0
        out[k] = float(v)
    return out

def safe_json_loads(txt: str, fallback: dict = None) -> dict:
    """Parse JSON safely; return fallback (or {}) on error."""
    try:
        if not txt or not txt.strip():
            return fallback if fallback is not None else {}
        return json.loads(txt)
    except Exception:
        return fallback if fallback is not None else {}

def _risk_badge(prob_ckd: float, thr: float) -> tuple[str, str, str]:
    if prob_ckd >= thr:
        return "Above threshold", "#fee2e2", "#991b1b"
    if prob_ckd >= 0.33:
        return "Moderate risk", "#fef3c7", "#92400e"
    return "Lower risk", "#dcfce7", "#065f46"

def _result_card(prob_ckd: float, thr: float, model_used: str):
    label, bg, col = _risk_badge(prob_ckd, thr)
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:18px;font-weight:700;">{MODEL_KEYS.get(model_used, model_used)}</div>
            <div class="badge" style="background:{bg};color:{col}">{label}</div>
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div><strong>Probability of CKD</strong><br><span style="font-size:20px;">{prob_ckd:.3f}</span></div>
            <div><strong>Decision Threshold</strong><br><span style="font-size:20px;">{thr:.3f}</span></div>
          </div>
          <div class="small-muted" style="margin-top:8px;">
            A probability at or above the threshold is flagged for CKD follow-up by this model.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _api_get(url: str, params: dict | None = None, timeout: int = 30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _api_post(url: str, json_body: dict | None = None, params: dict | None = None, timeout: int = 60):
    r = requests.post(url, params=params, json=json_body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_openrouter_chat(system_prompt: str, user_prompt: str) -> str | None:
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
            st.error(f"LLM request failed: {r.status_code}")
            st.code(r.text, language="json")
            return None
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", None)
    except Exception as e:
        st.error(f"LLM network error: {e}")
        return None

def call_llm(system_prompt: str, user_prompt: str):
    if LLM_PROVIDER.lower() == "openrouter":
        with st.spinner("Generating insights…"):
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
        with st.spinner("Generating insights…"):
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
    hyperk  = 1 if potassium >= 5.5 else 0
    anemia  = 1 if hb < 12.0 else 0

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
    st.session_state["last_pred_results"] = {}
if "batch_preds" not in st.session_state:
    st.session_state["batch_preds"] = {}
if "batch_insights" not in st.session_state:
    st.session_state["batch_insights"] = None
if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []
if "api_url" not in st.session_state:
    st.session_state["api_url"] = _normalize_api(API_URL)

def _log(msg: str):
    st.session_state["activity_log"].append(msg)

# =========================
# Sidebar: Connection & About
# =========================
with st.sidebar:
    st.markdown("### 🔌 Connection")
    api_input = st.text_input("Backend API URL", value=st.session_state["api_url"], help="Override if running locally.", key="sidebar_api_url_input")
    api_input = _normalize_api(api_input)
    if st.button("Check API", key="sidebar_btn_check_api"):
        try:
            h = _api_get(f"{api_input}/health", params={"model":"rf"}, timeout=10)
            ok = h.get("status") == "ok"
            db = "connected" if h.get("db_connected") else "not connected"
            if ok:
                st.success(f"OK • DB {db} • {h.get('db_backend','?')}")
                st.session_state["api_url"] = api_input
            else:
                st.warning(h)
        except Exception as e:
            st.error(f"Failed: {e}")
    st.caption(f"Using: `{st.session_state['api_url']}`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "Transparent CKD risk screening with model comparison and explanations. "
        "This tool does **not** provide a medical diagnosis. Consult a clinician."
    )

# =========================
# Header + Controls
# =========================
st.title(f"🩺 {APP_TITLE_HUMAN}")
st.caption("Quick, transparent CKD risk checks—single or CSV—plus explainers, logs, and AI summaries.")

top_left, top_right = st.columns([3, 2])
with top_left:
    model_labels = [label for label, _ in MODEL_CHOICES]
    picked_labels = st.multiselect(
        "Models to compare",
        model_labels,
        default=[model_labels[1]],  # default RF
        help="Pick one or more models. We'll compare outputs side-by-side.",
        key="header_global_models_pick"
    )
    selected_models = [MODEL_LABELS[l] for l in picked_labels] or ["rf"]

    st.markdown(
        "<div class='top-chip'>"
        "<b>Train models</b> retrains on the server • "
        "<b>Reload models</b> refreshes cached artifacts"
        "</div>",
        unsafe_allow_html=True
    )

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button("API health", use_container_width=True, key="header_btn_health"):
            try:
                h = _api_get(f"{st.session_state['api_url']}/health", params={"model": selected_models[0]}, timeout=10)
                if h.get("status") == "ok":
                    st.success("Healthy", icon="✅")
                    _log(f"Health OK • model={h.get('model')} thr={h.get('threshold')} db={h.get('db_backend')}")
                else:
                    st.warning(h, icon="⚠️")
                    _log(f"Health WARN: {h}")
            except Exception as e:
                st.error("API not reachable", icon="🛑")
                _log(f"Health ERROR: {e}")

    with cB:
        if st.button("Train models", use_container_width=True, key="header_btn_train"):
            try:
                r = requests.post(f"{st.session_state['api_url']}/admin/retrain", timeout=10)
                if r.status_code in (200, 202):
                    st.success("Training started", icon="🛠️")
                else:
                    st.info(f"Request returned {r.status_code}", icon="ℹ️")
                _log(f"Retrain response: {getattr(r, 'text', '')[:200]}")
            except Exception as e:
                st.error("Failed to start training", icon="🛑")
                _log(f"Retrain ERROR: {e}")

    with cC:
        if st.button("Reload models", use_container_width=True, key="header_btn_reload"):
            try:
                _ = requests.post(f"{st.session_state['api_url']}/admin/reload", timeout=20)
                st.success("Reloaded", icon="🔄")
                _log("Model cache reloaded.")
            except Exception as e:
                st.error("Reload failed", icon="🛑")
                _log(f"Reload ERROR: {e}")

with top_right:
    with st.expander("Activity log", expanded=False):
        if st.session_state["activity_log"]:
            st.markdown("\n".join(f"- {msg}" for msg in st.session_state["activity_log"]))
        else:
            st.caption("No activity yet.")

st.divider()

# =========================
# Tabs
# =========================
(
    tab_single,
    tab_batch,
    tab_metrics,
    tab_advice,
    tab_digital_twin,
    tab_counterfactuals,
    tab_similarity,
    tab_agents
) = st.tabs(
    [
        "Single check (compare models)",
        "Bulk check (CSV)",
        "Service & logs",
        "AI summary & next steps",
        "Digital Twin (What-If)",
        "Counterfactuals",
        "Similar Patients",
        "Care Plan (Agents)"
    ]
)

# =========================
# Single prediction
# =========================
with tab_single:
    st.markdown("Enter values and click **Predict**. See model outputs and top drivers.")

    # Prefill / Reset controls
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        sample_idx = st.selectbox("Prefill sample", [1,2,3,4,5], index=1, key="single_sample_idx")
    with c2:
        do_prefill = st.button("Use sample", key="single_use_sample")
    with c3:
        do_reset = st.button("Reset form", key="single_reset_form")

    def _defaults(si=None):
        df = sample_records_df()
        if si is not None:
            row = df.iloc[si-1].to_dict()
            return (
                int(row["age"]), int(row["gender"]),
                int(row["systolicbp"]), int(row["diastolicbp"]),
                float(row["serumcreatinine"]), float(row["bunlevels"]),
                float(row["gfr"]), float(row["acr"]),
                float(row["serumelectrolytessodium"]), float(row["serumelectrolytespotassium"]),
                float(row["hemoglobinlevels"]), float(row["hba1c"])
            )
        return (60, 1, 140, 85, 2.0, 28.0, 55.0, 120.0, 138.0, 4.8, 12.5, 6.8)

    if do_reset:
        st.rerun()

    # Initialize form values
    if do_prefill:
        (age_d, gender_d, sbp_d, dbp_d, sc_d, bun_d, gfr_d, acr_d, na_d, k_d, hb_d, a1c_d) = _defaults(sample_idx)
    else:
        (age_d, gender_d, sbp_d, dbp_d, sc_d, bun_d, gfr_d, acr_d, na_d, k_d, hb_d, a1c_d) = _defaults()

    with st.form("single_predict_form", border=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", 0, 120, age_d, key="single_form_age")
            gender = st.selectbox("Sex (0=female, 1=male)", [0, 1], index=gender_d, key="single_form_gender")
            systolicbp = st.number_input("Systolic BP (mmHg)", 70, 260, sbp_d, key="single_form_sbp")
            diastolicbp = st.number_input("Diastolic BP (mmHg)", 40, 160, dbp_d, key="single_form_dbp")
            serumcreatinine = st.number_input("Serum creatinine (mg/dL)", 0.2, 15.0, sc_d, step=0.1, key="single_form_sc")
            bunlevels = st.number_input("BUN (mg/dL)", 1.0, 200.0, bun_d, step=0.5, key="single_form_bun")
            gfr = st.number_input("GFR (mL/min/1.73m²)", 1.0, 200.0, gfr_d, step=0.5, key="single_form_gfr")
            acr = st.number_input("Albumin/creatinine ratio, ACR (mg/g)", 0.0, 5000.0, acr_d, step=1.0, key="single_form_acr")
        with col2:
            serumelectrolytessodium = st.number_input("Sodium (mEq/L)", 110.0, 170.0, na_d, step=0.5, key="single_form_na")
            serumelectrolytespotassium = st.number_input("Potassium (mEq/L)", 2.0, 7.5, k_d, step=0.1, key="single_form_k")
            hemoglobinlevels = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, hb_d, step=0.1, key="single_form_hb")
            hba1c = st.number_input("HbA1c (%)", 3.5, 15.0, a1c_d, step=0.1, key="single_form_a1c")

            bp_risk, hyperkalemiaflag, anemiaflag, ckdstage, albuminuriacat = derive_flags_and_bins(
                systolicbp, diastolicbp, serumelectrolytespotassium, hemoglobinlevels, gfr, acr
            )

        pulsepressure = systolicbp - diastolicbp
        ureacreatinineratio = float(bunlevels) / (float(serumcreatinine) + 1e-6)

        st.caption(
            f"Derived → Pulse pressure={pulsepressure} • Urea/Creatinine={ureacreatinineratio:.2f} • "
            f"Flags: High BP={bp_risk}, Hyperkalemia={hyperkalemiaflag}, Anemia={anemiaflag} • "
            f"CKD stage={ckdstage}, Albuminuria category={albuminuriacat}"
        )

        # Submit button inside the form
        do_predict = st.form_submit_button("Predict with selected models", use_container_width=True)

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
        payload = sanitize_payload(payload)
        st.session_state["last_pred_payload"] = payload
        st.session_state["last_pred_results"] = {}

        cols = st.columns(len(selected_models))
        for idx, m in enumerate(selected_models):
            try:
                res = _api_post(f"{st.session_state['api_url']}/predict", json_body=payload, params={"model": m}, timeout=20)
                st.session_state["last_pred_results"][m] = res
                with cols[idx]:
                    _result_card(float(res.get("prob_ckd", 0.0)),
                                 float(res.get("threshold_used", 0.5)),
                                 res.get("model_used", m))
                with cols[idx]:
                    prob = float(res.get("prob_ckd", 0.0))
                    thr  = float(res.get("threshold_used", 0.5))
                    st.markdown("**Plain-English result**")
                    st.markdown("This model would **flag** for CKD **follow-up**." if prob >= thr else "This model would **not** flag for CKD at this time.")
            except requests.HTTPError as e:
                with cols[idx]:
                    st.error(f"{MODEL_KEYS.get(m,m)} failed")
                    error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                    st.code(error_text or str(e), language="json")
                _log(f"Single predict HTTP ERROR ({m}): {e}")
            except Exception as e:
                with cols[idx]:
                    st.error(f"{MODEL_KEYS.get(m,m)} error")
                    st.caption(str(e))
                _log(f"Single predict ERROR ({m}): {e}")

        # Explainability per selected model
        st.markdown("#### Why did the model think that? (Top drivers)")
        st.caption("Higher bars = stronger influence on the decision for this case.")
        ecols = st.columns(len(selected_models))
        for idx, m in enumerate(selected_models):
            if m not in st.session_state["last_pred_results"]:
                continue
            with ecols[idx]:
                with st.spinner(f"{MODEL_KEYS.get(m,m)} — computing SHAP"):
                    try:
                        exp = _api_post(
                            f"{st.session_state['api_url']}/explain",
                            json_body=st.session_state["last_pred_payload"],
                            params={"model": m},
                            timeout=60
                        )
                        top = exp.get("top") or []
                        if not top:
                            shap_map = exp.get("shap_values", {}) or {}
                            top = sorted(
                                [{"feature": k, "impact": abs(v), "signed": v} for k, v in shap_map.items()],
                                key=lambda x: x["impact"], reverse=True
                            )[:8]
                        if top:
                            df_top = pd.DataFrame(top)
                            if "impact" not in df_top.columns and "signed" in df_top.columns:
                                df_top["impact"] = df_top["signed"].abs()
                            st.bar_chart(df_top.set_index("feature")["impact"])
                            bullets = "\n".join(
                                f"- **{row['feature']}** {'↑' if float(row.get('signed',0))>0 else '↓'} risk (SHAP={float(row.get('signed',0)):+.3f})"
                                for row in top[:5]
                            )
                            st.markdown(bullets)
                            with st.expander("Raw explanation (debug)", expanded=False):
                                st.json(exp)
                        else:
                            st.info("No features available.")
                        _log(f"Explain OK ({m}).")
                    except requests.HTTPError as e:
                        st.warning(f"⚠️ Explain feature not available for {MODEL_KEYS.get(m,m)}")
                        error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                        with st.expander("Error details", expanded=False):
                            st.code(error_text or str(e), language="json")
                        _log(f"Explain HTTP ERROR ({m}): {e}")
                    except Exception as e:
                        st.warning(f"⚠️ Explain error for {MODEL_KEYS.get(m,m)}")
                        with st.expander("Error details", expanded=False):
                            st.caption(str(e))
                        _log(f"Explain ERROR ({m}): {e}")

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
                           file_name="ckd_template_blank.csv", mime="text/csv", key="batch_dl_blank_template")
        st.download_button("Sample CSV (5 rows)",
                           data=sample5.to_csv(index=False).encode("utf-8"),
                           file_name="ckd_sample_5rows.csv", mime="text/csv", key="batch_dl_sample5")

    file = st.file_uploader("Upload CSV", type=["csv"], key="batch_file_uploader")
    retrain_after_batch = st.checkbox("Start training after batch (optional)", value=False, key="batch_retrain_checkbox")

    if file:
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.dataframe(df.head())

                if st.button("Run batch with selected models", key="batch_btn_run_batch"):
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
                            error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                            st.code(error_text or str(e), language="json")
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
                                key=f"batch_dl_model_{m}"
                            )
                        if merged_list:
                            all_merged = pd.concat(merged_list, axis=0, ignore_index=True)
                            st.download_button(
                                "Download ALL results (combined)",
                                data=all_merged.to_csv(index=False).encode("utf-8"),
                                file_name="ckd_batch_predictions_all_models.csv",
                                mime="text/csv",
                                key="batch_dl_all_models"
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

    # Cohort insights (AI) – frontend LLM helper
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
            key="batch_insights_model_select"
        )
        chosen_m = next((k for k in available_models if MODEL_KEYS.get(k, k) == insights_model), available_models[0])

        preds = st.session_state["batch_preds"][chosen_m]
        pos_rate = preds["prediction"].mean() if "prediction" in preds else 0.0
        avg_prob = preds["prob_ckd"].mean() if "prob_ckd" in preds else 0.0

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Summarize cohort", key="batch_btn_summarize_cohort"):
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
            if st.button("Clear cached summary", key="batch_btn_clear_insights"):
                st.session_state["batch_insights"] = None
                _log("Cohort insights cleared.")

        if st.session_state["batch_insights"]:
            insights_html = nl2br(st.session_state["batch_insights"])
            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div style="font-weight:700">Cohort Insights — {MODEL_KEYS.get(chosen_m, chosen_m)}</div>
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
                key="batch_dl_insights_txt"
            )

# =========================
# Metrics
# =========================
with tab_metrics:
    st.markdown("Keep an eye on service health, recent inferences, and training reports.")
    auto = st.checkbox("Auto-refresh every 10 seconds", value=False, key="metrics_autorefresh_checkbox")
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
        chosen = st.selectbox("Use result from model", [MODEL_KEYS.get(m, m) for m in avail], index=0, key="advice_model_select")
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

        if st.button("Generate AI next steps", key="advice_btn_generate_nextsteps"):
            text = call_llm(system_prompt, user_prompt)
            if text:
                text_html = nl2br(text)
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <div style="font-weight:700">AI next steps — {MODEL_KEYS.get(chosen_key, chosen_key)}</div>
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
                    key="advice_dl_next_steps"
                )
                _log("Recommendations generated.")
            else:
                st.error("LLM did not return text. Check API key/credits and model name in `.streamlit/secrets.toml`.")
                _log("Recommendations FAILED.")

# =========================
# Digital Twin (What-If) — uses /whatif
# =========================
with tab_digital_twin:
    st.markdown("Run **what-if** scenarios on the last single case or custom values. Either apply small deltas or a grid sweep.")
    st.info("⚠️ This feature requires backend support for `/whatif` endpoint. If not available, you'll see an error.")
    
    base = st.session_state.get("last_pred_payload") or {}
    if not base:
        st.info("Run a single prediction first to set a baseline, or paste JSON below.")
    base_json = st.text_area(
        "Baseline JSON (optional, overrides last single payload if provided)",
        value=json.dumps(base, indent=2),
        key="twin_baseline_json"
    )
    use_base = sanitize_payload(safe_json_loads(base_json, fallback=base or {}))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Simple deltas** (applied to baseline)")
        d_sbp = st.number_input("Δ Systolic BP (mmHg)", -40, 40, 0, key="twin_d_sbp")
        d_dbp = st.number_input("Δ Diastolic BP (mmHg)", -30, 30, 0, key="twin_d_dbp")
        d_gfr = st.number_input("Δ GFR (mL/min/1.73m²)", -50, 50, 0, key="twin_d_gfr")
        d_acr = st.number_input("Δ ACR (mg/g)", -500, 500, 0, key="twin_d_acr")
        d_k   = st.number_input("Δ Potassium (mEq/L)", -2.0, 2.0, 0.0, step=0.1, key="twin_d_k")
        d_a1c = st.number_input("Δ HbA1c (%)", -3.0, 3.0, 0.0, step=0.1, key="twin_d_a1c")
        deltas = {
            "systolicbp": d_sbp,
            "diastolicbp": d_dbp,
            "gfr": d_gfr,
            "acr": d_acr,
            "serumelectrolytespotassium": d_k,
            "hba1c": d_a1c,
        }
    with c2:
        st.markdown("**Grid sweep** (comma-separated values)")
        grid_sbp = st.text_input("SBP list", "120,130,140", key="twin_grid_sbp")
        grid_gfr = st.text_input("GFR list", "30,45,60,75,90", key="twin_grid_gfr")
        grid_acr = st.text_input("ACR list", "10,30,300", key="twin_grid_acr")
        grid = {}
        try:
            if grid_sbp.strip():
                grid["systolicbp"] = [float(x) for x in grid_sbp.split(",") if x.strip()]
            if grid_gfr.strip():
                grid["gfr"] = [float(x) for x in grid_gfr.split(",") if x.strip()]
            if grid_acr.strip():
                grid["acr"] = [float(x) for x in grid_acr.split(",") if x.strip()]
        except Exception as e:
            st.error(f"Grid parse error: {e}")

    model_for_sim = st.selectbox("Model for simulation", [MODEL_KEYS.get(m,m) for m in MODEL_KEYS], index=1, key="twin_model_select")
    model_key_sim = MODEL_LABELS.get(model_for_sim, "rf")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Run single what-if", key="twin_btn_single_whatif"):
            body = {"base": use_base or base, "deltas": deltas, "model": model_key_sim}
            try:
                out = _api_post(f"{st.session_state['api_url']}/whatif", json_body=body, timeout=60)
                st.success("Done.")
                st.json(out)
                _log("What-if single completed.")
            except requests.HTTPError as e:
                st.warning("⚠️ What-if feature not available on backend")
                error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                with st.expander("Error details", expanded=False):
                    st.code(error_text or str(e), language="json")
                _log(f"What-if HTTP ERROR: {e}")
            except Exception as e:
                st.error(f"What-if error: {e}")
                _log(f"What-if ERROR: {e}")
    with cB:
        if st.button("Run grid sweep", key="twin_btn_grid_whatif"):
            body = {"base": use_base or base, "grid": grid, "model": model_key_sim}
            try:
                out = _api_post(f"{st.session_state['api_url']}/whatif", json_body=body, timeout=120)
                st.success("Done.")
                if isinstance(out, dict) and "rows" in out:
                    st.dataframe(pd.DataFrame(out["rows"]))
                else:
                    st.json(out)
                _log("What-if grid completed.")
            except requests.HTTPError as e:
                st.warning("⚠️ What-if grid feature not available on backend")
                error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                with st.expander("Error details", expanded=False):
                    st.code(error_text or str(e), language="json")
                _log(f"What-if grid HTTP ERROR: {e}")
            except Exception as e:
                st.error(f"Grid what-if error: {e}")
                _log(f"What-if grid ERROR: {e}")

# =========================
# Counterfactuals — uses /counterfactual
# =========================
with tab_counterfactuals:
    st.markdown("Find **actionable changes** to reduce risk below a target using greedy search or DiCE (if backend enabled).")
    st.info("⚠️ This feature requires backend support for `/counterfactual` endpoint. If not available, you'll see an error.")
    
    base = st.session_state.get("last_pred_payload") or {}
    if not base:
        st.info("Run a single prediction first to set a baseline, or paste JSON below.")
    base_json = st.text_area(
        "Baseline JSON (optional, overrides last single payload if provided)",
        value=json.dumps(base, indent=2),
        key="cf_baseline_json_input"
    )
    use_base = sanitize_payload(safe_json_loads(base_json, fallback=base or {}))

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        target_prob = st.number_input("Target prob (≤ this)", 0.0, 1.0, 0.2, step=0.01, key="cf_target_prob_input")
    with col2:
        method = st.selectbox("Method", ["auto", "greedy"], index=0, key="cf_method_select")
    with col3:
        model_for_cf = st.selectbox("Model", [MODEL_KEYS.get(m,m) for m in MODEL_KEYS], index=1, key="cf_model_select_dropdown")
    model_key_cf = MODEL_LABELS.get(model_for_cf, "rf")

    if st.button("Compute counterfactual", key="cf_btn_compute"):
        body = {"base": use_base or base, "target_prob": target_prob, "model": model_key_cf, "method": method}
        try:
            out = _api_post(f"{st.session_state['api_url']}/counterfactual", json_body=body, timeout=120)
            st.success("Counterfactual ready.")
            if isinstance(out, dict) and "steps" in out:
                st.markdown("**Search path**")
                st.table(pd.DataFrame(out["steps"]))
            st.markdown("**Final candidate**")
            res = out.get("result") or out.get("best", {}).get("candidate")
            if res:
                st.json(res)
            else:
                st.json(out)
            _log("Counterfactual computed.")
        except requests.HTTPError as e:
            st.warning("⚠️ Counterfactual feature not available on backend")
            error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
            with st.expander("Error details", expanded=False):
                st.code(error_text or str(e), language="json")
            _log(f"Counterfactual HTTP ERROR: {e}")
        except Exception as e:
            st.error(f"Counterfactual error: {e}")
            _log(f"Counterfactual ERROR: {e}")

# =========================
# Similar Patients — uses /similar
# =========================
with tab_similarity:
    st.markdown("Find **nearest neighbors** in an uploaded cohort to compare similar profiles.")
    st.info("⚠️ This feature requires backend support for `/similar` endpoint. If not available, you'll see an error.")
    
    base = st.session_state.get("last_pred_payload") or {}
    if not base:
        st.info("Run a single prediction first to set a baseline, or paste JSON below.")
    base_json = st.text_area(
        "Baseline JSON (optional, overrides last single payload if provided)",
        value=json.dumps(base, indent=2),
        key="sim_baseline_json_input"
    )
    use_base = sanitize_payload(safe_json_loads(base_json, fallback=base or {}))

    cohort_file = st.file_uploader("Upload cohort CSV to search in", type=["csv"], key="sim_file_uploader")
    k = st.slider("Top-k similar", 1, 50, 5, key="sim_k_slider")

    if st.button("Find similar", key="sim_btn_find_similar"):
        try:
            cohort = []
            if cohort_file:
                dfc = pd.read_csv(cohort_file)
                # keep only expected columns; fill missing
                for c in FEATURE_COLUMNS:
                    if c not in dfc.columns:
                        dfc[c] = 0.0
                dfc = dfc[FEATURE_COLUMNS]
                cohort = [sanitize_payload(r) for r in dfc.to_dict(orient="records")]
            body = {"base": use_base or base, "cohort": cohort}
            out = _api_post(f"{st.session_state['api_url']}/similar", params={"k": k}, json_body=body, timeout=60)
            st.success("Similar patients computed.")
            if isinstance(out, dict) and "neighbors" in out:
                st.table(pd.DataFrame(out["neighbors"]))
            else:
                st.json(out)
            _log("Similarity search completed.")
        except requests.HTTPError as e:
            st.warning("⚠️ Similarity feature not available on backend")
            error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
            with st.expander("Error details", expanded=False):
                st.code(error_text or str(e), language="json")
            _log(f"Similarity HTTP ERROR: {e}")
        except Exception as e:
            st.error(f"Similarity error: {e}")
            _log(f"Similarity ERROR: {e}")

# =========================
# Care Plan (Agents) — uses /agents/plan
# =========================
with tab_agents:
    st.markdown("LLM multi-agent **care plan** generated on the server (via `/agents/plan`).")
    st.info("⚠️ This feature requires backend support for `/agents/plan` endpoint with LLM integration. If not available, you'll see an error.")
    
    payload = st.session_state.get("last_pred_payload")
    results = st.session_state.get("last_pred_results", {})

    if not payload or not results:
        st.info("Run a single prediction first so we can build the summary for agents.")
    else:
        avail = list(results.keys())
        chosen = st.selectbox("Use result from model", [MODEL_KEYS.get(m, m) for m in avail], index=0, key="agents_model_select_dropdown")
        chosen_key = next((k for k in avail if MODEL_KEYS.get(k, k) == chosen), avail[0])
        res = results[chosen_key]

        # Build summary for API agents — keep it compact
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

        if st.button("Generate care plan (server agents)", key="agents_btn_generate_plan"):
            try:
                out = _api_post(f"{st.session_state['api_url']}/agents/plan", json_body=summary, timeout=120)
                st.success("Care plan generated.")
                # Try to render common structure: {sections: [{title, bullets/text}...]}
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
                _log("Care plan generated.")
            except requests.HTTPError as e:
                st.warning("⚠️ Care plan (agents) feature not available on backend")
                error_text = getattr(e.response, "text", str(e)) if hasattr(e, 'response') else str(e)
                with st.expander("Error details", expanded=False):
                    st.code(error_text or str(e), language="json")
                _log(f"Agents plan HTTP ERROR: {e}")
            except Exception as e:
                st.error(f"Agents error: {e}")
                _log(f"Agents ERROR: {e}")
