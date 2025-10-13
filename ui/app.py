# app.py — Kidney Health Radar (with scoped Chat Assistant)
# -----------------------------------------------
# - Single & bulk CKD screening with explainers
# - Digital Twin (What-If), Counterfactuals, Similar Patients
# - NEW: Chat (Assistant) tab that ONLY answers from loaded contexts
# - Optional OpenRouter/OpenAI LLM integration for summaries
#
# Quick start:
#   streamlit run app.py
#
# Secrets (optional):
#   API_URL="https://ckdpredictor.onrender.com"
#   LLM_PROVIDER="openrouter"  # openai | openrouter | custom
#   LLM_API_KEY="..."
#   LLM_BASE_URL="https://openrouter.ai/api/v1"
#   LLM_MODEL="openrouter/auto"
#   APP_URL="https://github.com/baheldeepti/CKDPredictor"
#   APP_TITLE="Kidney Health Radar"

import os
import json
import math
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Tuple

# =========================
# Branding & Config
# =========================
# .streamlit/secrets.toml
APP_PRODUCT_NAME = "NephroCompass"
APP_TAGLINE      = "Kidney Health Radar"
APP_TITLE        = "NephroCompass — Kidney Health Radar"
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
MODEL_LABELS = {label: key for (label, key) in MODEL_CHOICES}  # UI label -> key
MODEL_KEYS   = {key: label for (label, key) in MODEL_CHOICES}  # key -> UI label

# Optional LLM config for the “AI summary & next steps” tab + Chat
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openrouter")  # openai | together | openrouter | custom
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = st.secrets.get("LLM_MODEL", "openrouter/auto")



APP_REPO_URL = "https://github.com/baheldeepti/CKDPredictor"

st.set_page_config(page_title=APP_TITLE_HUMAN, page_icon="🧭", layout="wide")


# -------------------------
# PREMIUM UI THEME (CSS) — Professionalized
# -------------------------
STYLE = """
<style>
/* =========================
   Design Tokens — Healthcare UI
   ========================= */
:root {
  /* Brand palette (calm clinical) */
  --brand-900:#0B3B5A;
  --brand-800:#114C74;
  --brand-700:#165D8F;
  --brand-600:#1B6EA9;
  --brand-500:#1F7FC4; /* Primary */
  --brand-400:#4798CF;
  --brand-300:#74B3DB;
  --brand-200:#A4CDE7;
  --brand-100:#D9ECF8;

  --teal-700:#0F766E;  --teal-100:#D1FAF5; /* Secondary accent */

  /* Functional/status */
  --ok-700:#126B3F;   --ok-100:#E7F6EE;
  --warn-700:#8A5B0A; --warn-100:#FFF3D6;
  --bad-700:#8E1E28;  --bad-100:#FBE6E8;
  --info-700:#1E4E79; --info-100:#E5F1FB;

  /* Neutrals */
  --fg-900:#0F172A;
  --fg-800:#1F2937;
  --fg-700:#334155;
  --fg-600:#475569;
  --fg-500:#64748B;
  --fg-400:#94A3B8;

  /* Surfaces & borders */
  --bg-0:#F7FAFC;  /* app */
  --bg-1:#FFFFFF;  /* cards */
  --bg-2:#F1F5F9;  /* tabs, headers */
  --brd-300:#E2E8F0;
  --brd-200:#EDF2F7;

  /* Elevation */
  --elev-1:0 1px 2px rgba(16,24,40,.06), 0 1px 1px rgba(16,24,40,.04);
  --elev-2:0 6px 16px rgba(2,32,71,.08), 0 2px 4px rgba(2,32,71,.06);
  --elev-3:0 12px 24px rgba(2,32,71,.10), 0 4px 8px rgba(2,32,71,.06);

  /* Radius & spacing */
  --r-sm:10px; --r-md:14px; --r-lg:18px;
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px; --space-5:20px; --space-6:24px;

  /* Typography */
  --font-sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --fs-h1: clamp(24px, 2.2vw, 30px);
  --fs-h2: clamp(18px, 1.8vw, 22px);
  --fs-h3: clamp(15px, 1.6vw, 18px);
  --fs-body: 15px;
  --fs-caption: 12.5px;

  /* Focus ring */
  --focus: 0 0 0 3px rgba(31,127,196,.28);
}

/* =========================
   Base
   ========================= */
html, body, [class^="css"] {
  font-family: var(--font-sans);
  color: var(--fg-800);
  background: var(--bg-0);
}
section.main > div { padding-top: .6rem; }

h1, h2, h3 { letter-spacing:-.01em; color: var(--fg-900); }
h1 { font: 800 var(--fs-h1)/1.2 var(--font-sans); margin: 4px 0; }
h2 { font: 700 var(--fs-h2)/1.35 var(--font-sans); margin: 8px 0 6px; }
h3 { font: 700 var(--fs-h3)/1.35 var(--font-sans); margin: 8px 0 6px; }

p, .block-info { font-size: var(--fs-body); line-height: 1.6; color: var(--fg-700); }
.small-muted, .caption, .tooltip { color: var(--fg-500); font-size: var(--fs-caption); }
.top-chip { color: var(--fg-500); font-size: var(--fs-caption); margin-top: -6px; }

a { color: var(--brand-600); text-decoration: none; }
a:hover { text-decoration: underline; }

hr { border: none; border-top:1px solid var(--brd-300); margin: .75rem 0 1.25rem; }

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}

/* =========================
   Hero (brand header)
   ========================= */
.hero {
  margin: 6px 0 10px;
  padding: 14px 16px;
  border-radius: var(--r-md);
  background: linear-gradient(180deg, var(--bg-1) 0%, #FAFDFF 100%);
  border: 1px solid var(--brd-300);
  box-shadow: var(--elev-1);
}
.hero-title { display:flex; align-items:center; gap:10px; line-height:1.2; }
.hero-logo { font-size: 24px; transform: translateY(-1px); }
.hero .brand { font-weight: 900; letter-spacing:-.02em; font-size: var(--fs-h1); color: var(--fg-900); }
.hero .tag {
  margin-left:6px; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:700;
  color: var(--info-700); background: var(--info-100); border: 1px solid #CFE3FA;
}
.hero-subtitle { margin-top: 2px; font-size: clamp(14px,1.5vw,16px); font-weight: 650; color: var(--brand-700); }
.hero-meta { margin-top: 6px; color: var(--fg-600); font-size: 13.5px; }

/* =========================
   Cards & Surfaces
   ========================= */
.card {
  padding: 16px 18px;
  border-radius: var(--r-md);
  border: 1px solid var(--brd-300);
  background: var(--bg-1);
  box-shadow: var(--elev-1);
  transition: box-shadow .18s ease;
}
.card:hover { box-shadow: var(--elev-2); }
.card--elevated { box-shadow: var(--elev-3); }
.card--header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
.card--meta { color: var(--fg-500); font-size: var(--fs-caption); }

.surface { background: var(--bg-1); border:1px solid var(--brd-300); border-radius: var(--r-md); box-shadow: var(--elev-1); }

/* =========================
   Buttons
   ========================= */
.stButton>button, .stDownloadButton>button {
  appearance: none;
  border-radius: var(--r-sm);
  border: 1px solid rgba(31,127,196,.22);
  background: linear-gradient(180deg, var(--brand-500), var(--brand-600));
  color: #fff;
  font-weight: 700;
  padding: .55rem .9rem;
  box-shadow: var(--elev-1);
  transition: transform .12s ease, box-shadow .18s ease, filter .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  filter: brightness(.98);
  transform: translateY(-1px);
  box-shadow: var(--elev-2);
}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline: none; box-shadow: var(--focus); }
.stButton>button:disabled { opacity: .55; cursor: not-allowed; }

/* Secondary (outlined) */
button[kind="secondary"] {
  background: var(--bg-1);
  border-color: var(--brd-300);
  color: var(--brand-700);
}

/* Icon-only button normalization */
button:has(svg) { display:inline-flex; align-items:center; gap:8px; }

/* =========================
   Inputs & Forms
   ========================= */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input,
.stSelectbox div[data-baseweb="select"]>div {
  border-radius: var(--r-sm);
  border: 1px solid var(--brd-300);
  background: #fff;
  box-shadow: none;
  font-size: var(--fs-body);
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus,
.stDateInput input:focus {
  outline: none; box-shadow: var(--focus); border-color: var(--brand-500);
}

.stSlider [role="slider"] { border-color: var(--brand-500); }
.stCheckbox input[type="checkbox"] { accent-color: var(--brand-600); }

/* Labels a touch bolder for readability */
label, .st-emotion-cache-1cypcdb { font-weight: 600; color: var(--fg-700); }

/* =========================
   Tabs
   ========================= */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--brd-300);
  margin-bottom: .5rem;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--fg-600);
  border-radius: var(--r-sm) var(--r-sm) 0 0;
  padding: .5rem .9rem;
  border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--brand-700);
  font-weight: 800;
  background: var(--bg-1);
  border-color: var(--brd-300);
  border-bottom-color: var(--bg-1);
  box-shadow: var(--elev-1);
}

/* =========================
   Expanders
   ========================= */
.streamlit-expanderHeader { font-weight: 800; color: var(--fg-800); }
.st-expander { border: 1px solid var(--brd-300); border-radius: var(--r-md); background: var(--bg-1); box-shadow: var(--elev-1); }

/* =========================
   DataFrames & Tables
   ========================= */
.stDataFrame, .stTable {
  border-radius: var(--r-sm);
  overflow: hidden;
  box-shadow: var(--elev-1);
  border: 1px solid var(--brd-300);
}
.stTable table thead th, .stDataFrame table thead th {
  background: var(--bg-2) !important;
  color: var(--fg-700) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  border-bottom: 1px solid var(--brd-300) !important;
}
.stTable table tbody tr:nth-child(even) { background:#FAFCFF; }
.table-hover tbody tr:hover { background: #F6FAFF; }

/* =========================
   Code blocks
   ========================= */
pre, code {
  white-space: pre-wrap;
  font-family: var(--font-mono);
  font-size: .86rem;
  background: #0b2942;
  color: #e6edf3;
  border-radius: var(--r-sm);
  padding: .5rem .6rem;
  border: 1px solid rgba(255,255,255,.06);
}

/* =========================
   Badges & KPI
   ========================= */
.badge {
  display:inline-flex; align-items:center; gap:6px;
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: .8rem;
  border: 1px solid var(--brd-300);
  background: var(--bg-2);
  color: var(--fg-700);
}
.badge.is-ok   { background: var(--ok-100);   color: var(--ok-700);   border-color:#CDE9DB; }
.badge.is-warn { background: var(--warn-100); color: var(--warn-700); border-color:#FFE7B0; }
.badge.is-bad  { background: var(--bad-100);  color: var(--bad-700);  border-color:#F6B9BF; }
.badge.is-info { background: var(--info-100); color: var(--info-700); border-color:#CFE3FA; }
.badge.is-brand{ background: var(--brand-100);color: var(--brand-700);border-color: var(--brand-200); }

.kpi { display:flex; gap:18px; flex-wrap:wrap; align-items:flex-end; }
.kpi .metric { min-width: 120px; }
.kpi .metric b { display:block; font-size:.8rem; color: var(--fg-600); font-weight:700; }
.kpi .metric span { font-size:1.25rem; font-weight:900; color: var(--fg-900); }

/* =========================
   Plotly / Charts
   ========================= */
.element-container:has(.plotly) { padding: 6px; }

/* =========================
   Sidebar
   ========================= */
.stSidebar { background: linear-gradient(180deg, #F8FBFE, var(--bg-0)); }
.stSidebar [data-testid="stSidebarNav"] { padding-top: .5rem; }

/* =========================
   Accessibility
   ========================= */
button:focus-visible, a:focus-visible, [role="tab"]:focus-visible { outline: none; box-shadow: var(--focus); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def _strip_context_echo(reply: str, ctx_text: str) -> str:
    # Remove any line from reply that appears verbatim in the context digest
    ctx_lines_set = set([ln.strip() for ln in ctx_text.splitlines() if ln.strip()])
    cleaned = []
    for ln in (reply or "").splitlines():
        if ln.strip() in ctx_lines_set:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def _enforce_min_structure(reply: str) -> str:
    txt = (reply or "").strip()
    if not txt:
        return txt
    # If the assistant forgot structure, reshape lightly
    has_headers = any(h in txt.lower() for h in ["takeaway", "drivers", "next step"])
    if not has_headers:
        parts = txt.split("\n")
        head = parts[0].strip()
        rest = [p.strip() for p in parts[1:] if p.strip()]
        bullets = "\n".join([f"- {p}" for p in rest[:4]])
        txt = f"**Takeaway**\n{head}\n\n**Key drivers**\n{bullets}\n\n**Next step**\n- Consider follow-up labs / monitoring."
    return txt

def _normalize_api(url: str) -> str:
    if not url:
        return API_URL_DEFAULT
    return url.rstrip("/")

def nl2br(s: str) -> str:
    try:
        return s.replace("\n", "<br/>")
    except Exception:
        return s

def _cf_feature_nice(f: str) -> str:
    names = {
        "systolicbp":"Systolic BP", "diastolicbp":"Diastolic BP", "gfr":"GFR", "acr":"ACR",
        "serumelectrolytespotassium":"Potassium", "serumelectrolytessodium":"Sodium",
        "hba1c":"HbA1c", "hemoglobinlevels":"Hemoglobin"
    }
    return names.get(f, f)

def sanitize_payload(d: dict) -> dict:
    out = {}
    for k in FEATURE_COLUMNS:
        v = d.get(k, 0)
        if v is None:
            v = 0.0
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = 0.0
        try:
            out[k] = float(v)
        except Exception:
            try:
                out[k] = float(int(v))
            except Exception:
                out[k] = 0.0
    return out

def safe_json_loads(txt: str, fallback: dict = None) -> dict:
    """Parse JSON safely; return fallback (or {}) on error."""
    try:
        if not txt or not txt.strip():
            return fallback if fallback is not None else {}
        return json.loads(txt)
    except Exception:
        return fallback if fallback is not None else {}

def _api_get(url: str, params: dict | None = None, timeout: int = 30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _api_post(url: str, json_body: dict | None = None, params: dict | None = None, timeout: int = 60):
    r = requests.post(url, params=params, json=json_body, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        e = requests.HTTPError(f"{r.status_code}: {detail}")
        e.response = r
        raise e
    return r.json()

# ---- LLM helpers
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
        "temperature": 0.35,
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
                temperature=0.35,
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

    # Stage by GFR (G1=≥90, G2=60-89, G3a=45-59, G3b=30-44, G4=15-29, G5<15)
    if gfr >= 90: stage = 1
    elif gfr >= 60: stage = 2
    elif gfr >= 45: stage = 3
    elif gfr >= 30: stage = 3
    elif gfr >= 15: stage = 4
    else: stage = 5

    # Albuminuria by ACR (A1 <30, A2 30–300, A3 >300)
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

GRID_SWEEP_HELP = """
**What is a grid sweep?** Pick one or more knobs (e.g., SBP, GFR, ACR), give each a list of target values,
and we simulate **every combination** to see how the risk changes. It’s like trying out all “what-if”
settings at once to spot safe ranges and red flags.
"""

def parse_csv_single_row(upload) -> dict | None:
    try:
        df = pd.read_csv(upload)
        if df.empty:
            return None
        for c in FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
        row = df[FEATURE_COLUMNS].iloc[0].to_dict()
        return sanitize_payload(row)
    except Exception:
        return None

def pretty_feature_name(f: str) -> str:
    mapping = {
        "systolicbp":"Systolic BP", "diastolicbp":"Diastolic BP",
        "gfr":"GFR", "acr":"ACR", "serumelectrolytespotassium":"Potassium",
        "hba1c":"HbA1c"
    }
    return mapping.get(f, f)

# =========================
# Local fallback explainer
# =========================
def _safe_predict_prob(api_base: str, payload: Dict[str, Any], model_key: str) -> float:
    try:
        res = _api_post(f"{api_base}/predict", json_body=payload, params={"model": model_key}, timeout=20)
        return float(res.get("prob_ckd", 0.0))
    except Exception:
        return 0.0

def _sensitivity_top_drivers(
    api_base: str, base_payload: Dict[str, Any], model_key: str, probe_features: List[str], k: int = 8
) -> List[Dict[str, float]]:
    base_prob = _safe_predict_prob(api_base, base_payload, model_key)
    rows = []
    for f in probe_features:
        if f not in base_payload:
            continue
        v = float(base_payload[f])
        if f in ("systolicbp", "diastolicbp"): delta = 5.0
        elif f in ("gfr",): delta = 5.0
        elif f in ("acr",): delta = max(5.0, abs(v) * 0.05)
        elif f in ("serumelectrolytessodium",): delta = 2.0
        elif f in ("serumelectrolytespotassium",): delta = 0.2
        elif f in ("hba1c",): delta = 0.2
        elif f in ("hemoglobinlevels",): delta = 0.3
        elif f in ("serumcreatinine",): delta = 0.2
        elif f in ("bunlevels",): delta = 2.0
        elif f in ("pulsepressure",): delta = 3.0
        elif f in ("ureacreatinineratio",): delta = 1.0
        else:
            delta = max(0.05 * abs(v), 1.0)

        up = dict(base_payload); up[f] = v + delta
        dn = dict(base_payload); dn[f] = max(0.0, v - delta)
        p_up = _safe_predict_prob(api_base, sanitize_payload(up), model_key)
        p_dn = _safe_predict_prob(api_base, sanitize_payload(dn), model_key)

        d_up = p_up - base_prob
        d_dn = p_dn - base_prob
        signed = float(d_up) if abs(d_up) >= abs(d_dn) else float(d_dn)
        rows.append({"feature": f, "impact": abs(signed), "signed": signed})

    rows.sort(key=lambda r: r["impact"], reverse=True)
    return rows[:k]

def explain_with_fallback(api_base: str, payload: Dict[str, Any], model_key: str) -> Tuple[List[Dict[str, float]], Dict[str, Any] | None, str]:
    try:
        exp = _api_post(f"{api_base}/explain", json_body=payload, params={"model": model_key}, timeout=60)
        top = exp.get("top") or []
        if not top:
            shap_map = exp.get("shap_values", {}) or {}
            top = sorted(
                [{"feature": k, "impact": abs(v), "signed": float(v)} for k, v in shap_map.items()],
                key=lambda x: x["impact"], reverse=True
            )[:8]
        if top:
            return top, exp, "server_shap"
        top_local = _sensitivity_top_drivers(api_base, payload, model_key, FEATURE_COLUMNS, k=8)
        return top_local, None, "local_sensitivity"
    except requests.HTTPError as e:
        try:
            status = e.response.status_code
            _ = e.response.json() if e.response.headers.get("content-type","").startswith("application/json") else e.response.text
        except Exception:
            status = None
        if status in (500, 503):
            top_local = _sensitivity_top_drivers(api_base, payload, model_key, FEATURE_COLUMNS, k=8)
            return top_local, None, "local_sensitivity"
        raise
    except Exception:
        top_local = _sensitivity_top_drivers(api_base, payload, model_key, FEATURE_COLUMNS, k=8)
        return top_local, None, "local_sensitivity"

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

# ---- Chat + cross-tab contexts (data contracts)
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []  # [{"role":"user"/"assistant","content":str}]
if "ctx_digital_twin" not in st.session_state:
    st.session_state["ctx_digital_twin"] = None   # {schema_version, model, threshold, swept_features, summary, rows_sample}
if "ctx_counterfactual" not in st.session_state:
    st.session_state["ctx_counterfactual"] = None # {schema_version, model, threshold, target_prob, initial_prob, final_prob, converged, steps, final_candidate}
if "ctx_similar" not in st.session_state:
    st.session_state["ctx_similar"] = None        # {schema_version, k, metric, neighbors_sample, summary}
if "ctx_explain" not in st.session_state:
    st.session_state["ctx_explain"] = None        # {schema_version, model, top, mode}

# =========================
# Sidebar: Connection & About
# =========================
with st.sidebar:
    st.markdown("### 🔌 Connection")
    api_input = st.text_input("Backend API URL", value=st.session_state["api_url"], help="Override if running locally.", key="sb_api_url")
    api_input = _normalize_api(api_input)
    if st.button("Check API", key="btn_check_api"):
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
st.caption("Quick, transparent CKD risk checks—single or CSV—plus explainers, logs, AI summaries, and a scoped Chat Assistant.")

top_left, top_right = st.columns([3, 2])
with top_left:
    model_labels = [label for label, _ in MODEL_CHOICES]
    picked_labels = st.multiselect(
        "Models to compare",
        model_labels,
        default=[model_labels[1]],  # default RF
        help="Pick one or more models. We'll compare outputs side-by-side.",
        key="global_models_pick"
    )
    selected_models = [MODEL_LABELS[l] for l in picked_labels] or ["rf"]

    st.markdown(
        "<div class='top-chip'><b>Train models</b> retrains on the server • "
        "<b>Reload models</b> refreshes cached artifacts</div>",
        unsafe_allow_html=True
    )

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button("API health", use_container_width=True, key="btn_health"):
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
        if st.button("Train models", use_container_width=True, key="btn_train"):
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
        if st.button("Reload models", use_container_width=True, key="btn_reload"):
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
# Tabs (Chat added first)
# =========================
(
    tab_chat,
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
        "Chat (Assistant)",
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
# Chat (Assistant) — consumes contexts from other tabs
# =========================
def _build_chat_prompts(user_msg: str) -> tuple[str, str]:
    """
    System & user prompts for the scoped Chat. The assistant can ONLY answer using
    the three context objects (and explainability if present). This is our PRD-ready
    guardrail block and the data interfaces.
    """
    ctx_dt  = st.session_state.get("ctx_digital_twin")
    ctx_cf  = st.session_state.get("ctx_counterfactual")
    ctx_sim = st.session_state.get("ctx_similar")
    ctx_exp = st.session_state.get("ctx_explain")

    system_prompt = """
You are a CKD risk assistant LIMITED to four context digests (digital_twin, counterfactual, similar, explain).
You must synthesize insights; never quote or paraphrase raw context lines verbatim.

Hard rules:
- Do NOT copy the context digest back to the user. Use it only to reason.
- Answer with short sections and bullets. Keep total output under ~180 words unless asked for more.
- No medication dosing/prescribing. Use plain units (mmHg, mg/g, mEq/L, %).
- If data is missing for the user’s question, say: "Out of scope for this chat. Load Digital Twin, Counterfactuals, or Similar Patients first."
- When referencing thresholds, briefly distinguish threshold vs probability.
- End with: "Educational tool; not medical advice."

Output format (enforce):
1) Brief takeaway
2) Key drivers or trade-offs (2–4 bullets)
3) Practical next step(s)
""".strip()


    def _short(x, n=6):
        """Return up to n key:value pairs from a dict or list of dicts, as a human summary."""
        try:
            if isinstance(x, dict):
                items = list(x.items())
                return "; ".join([f"{k}={str(v)[:18]}" for k, v in items[:n]])
            if isinstance(x, list) and x and isinstance(x[0], dict):
                keys = list(x[0].keys())[:n]
                return "rows=" + str(len(x)) + "; keys=" + ",".join(keys)
            return str(x)[:120]
        except Exception:
            return str(x)[:120]

    # Compact, human-readable context (no raw JSON)
    ctx_lines = ["CONTEXT DIGEST"]
    if ctx_dt:
        ctx_lines.append(
            "digital_twin: "
            f"model={ctx_dt.get('model')}; thr={ctx_dt.get('threshold')}; "
            f"swept={','.join(ctx_dt.get('swept_features', [])[:4])}; "
            f"summary={_short(ctx_dt.get('summary', {}))}; "
            f"sample={_short(ctx_dt.get('rows_sample', []))}"
        )
    if ctx_cf:
        ctx_lines.append(
            "counterfactual: "
            f"model={ctx_cf.get('model')}; thr={ctx_cf.get('threshold')}; "
            f"target={ctx_cf.get('target_prob')}; p0={ctx_cf.get('initial_prob')}; pf={ctx_cf.get('final_prob')}; "
            f"converged={ctx_cf.get('converged')}; steps={_short(ctx_cf.get('steps', []))}"
        )
    if ctx_sim:
        ctx_lines.append(
            "similar: "
            f"k={ctx_sim.get('k')}; metric={ctx_sim.get('metric')}; "
            f"summary={_short(ctx_sim.get('summary', {}))}; "
            f"neighbors={_short(ctx_sim.get('neighbors_sample', []))}"
        )
    if ctx_exp:
        top_feats = [t.get('feature','?') for t in ctx_exp.get('top', [])[:6]]
        ctx_lines.append(
            "explain: "
            f"model={ctx_exp.get('model')}; top={','.join(top_feats)}; mode={ctx_exp.get('mode')}"
        )
    if not any([ctx_dt, ctx_cf, ctx_sim, ctx_exp]):
        ctx_lines.append("NO_CONTEXT_AVAILABLE")
    ctx_text = "\n".join(ctx_lines)

    user_prompt = f"{ctx_text}\n\nQUESTION:\n{user_msg}".strip()

   
    return system_prompt, user_prompt

with tab_chat:
    st.markdown("### Chat (Assistant)")
    st.caption("Scoped to Digital Twin, Counterfactuals, Similar Patients, and Explainability. Answers only from loaded context.")

    if st.button("Clear chat", key="btn_clear_chat"):
        st.session_state["chat_messages"] = []
        st.success("Chat history cleared.")

    # Render history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(nl2br(msg["content"]), unsafe_allow_html=True)

    user_msg = st.chat_input("Ask about the simulations, counterfactual plan, or similar patients…")
    if user_msg:
        st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        sys_p, usr_p = _build_chat_prompts(user_msg)
        raw_reply = call_llm(sys_p, usr_p)

        # Post-process: remove any echoed context lines and enforce structure
        ctx_text = usr_p.split("\n\nQUESTION:\n", 1)[0] if "\n\nQUESTION:\n" in usr_p else ""
        reply = _strip_context_echo(raw_reply or "", ctx_text)
        reply = _enforce_min_structure(reply)
        if not reply:
            reply = "Out of scope or no context loaded. Load Digital Twin, Counterfactuals, or Similar Patients."

        if "Educational tool; not medical advice." not in reply:
            reply = f"{reply}\n\n*Educational tool; not medical advice.*"


        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(nl2br(reply), unsafe_allow_html=True)

    # Context chips
    chips = []
    if st.session_state.get("ctx_digital_twin"):
        c = st.session_state["ctx_digital_twin"]
        chips.append(f"`digital-twin: {c.get('summary',{}).get('n','?')} rows`")
    if st.session_state.get("ctx_counterfactual"):
        c = st.session_state["ctx_counterfactual"]
        chips.append(f"`cf: {'converged' if c.get('converged') else 'not converged'}`")
    if st.session_state.get("ctx_similar"):
        c = st.session_state["ctx_similar"]
        chips.append(f"`similar: k={c.get('k','?')}`")
    if st.session_state.get("ctx_explain"):
        c = st.session_state["ctx_explain"]
        chips.append(f"`explain: top={len(c.get('top',[]))}`")
    if chips:
        st.caption("Context loaded → " + " • ".join(chips))
    else:
        st.caption("No context loaded yet.")

# =========================
# Single prediction
# =========================
with tab_single:
    st.markdown("Enter values and click **Predict**. See model outputs and top drivers.")

    # Prefill / Reset controls
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        sample_idx = st.selectbox("Prefill sample", [1,2,3,4,5], index=1, key="sp_sample_idx")
    with c2:
        do_prefill = st.button("Use sample", key="sp_use_sample")
    with c3:
        do_reset = st.button("Reset form", key="sp_reset")

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

    with st.form("predict_form", border=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", 0, 120, age_d, key="sp_age")
            gender_index = 1 if int(gender_d) == 1 else 0
            gender = st.selectbox("Sex (0=female, 1=male)", [0, 1], index=gender_index, key="sp_gender")
            systolicbp = st.number_input("Systolic BP (mmHg)", 70, 260, sbp_d, key="sp_sbp")
            diastolicbp = st.number_input("Diastolic BP (mmHg)", 40, 160, dbp_d, key="sp_dbp")
            serumcreatinine = st.number_input("Serum creatinine (mg/dL)", 0.2, 15.0, sc_d, step=0.1, key="sp_sc")
            bunlevels = st.number_input("BUN (mg/dL)", 1.0, 200.0, bun_d, step=0.5, key="sp_bun")
            gfr = st.number_input("GFR (mL/min/1.73m²)", 1.0, 200.0, gfr_d, step=0.5, key="sp_gfr")
            acr = st.number_input("Albumin/creatinine ratio, ACR (mg/g)", 0.0, 5000.0, acr_d, step=1.0, key="sp_acr")
        with col2:
            serumelectrolytessodium = st.number_input("Sodium (mEq/L)", 110.0, 170.0, na_d, step=0.5, key="sp_na")
            serumelectrolytespotassium = st.number_input("Potassium (mEq/L)", 2.0, 7.5, k_d, step=0.1, key="sp_k")
            hemoglobinlevels = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, hb_d, step=0.1, key="sp_hb")
            hba1c = st.number_input("HbA1c (%)", 3.5, 15.0, a1c_d, step=0.1, key="sp_a1c")

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

        do_predict = st.form_submit_button("Predict with selected models", use_container_width=True, disabled=False, help="Runs the selected models")

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
                    # Result card
                    prob = float(res.get("prob_ckd", 0.0))
                    thr  = float(res.get("threshold_used", 0.5))
                    model_used = res.get("model_used", m)
                    label = "Above threshold" if prob >= thr else ("Moderate risk" if prob >= 0.33 else "Lower risk")
                    # Use tokenized badge variants for consistent polish
                    badge_cls = "is-bad" if prob >= thr else ("is-warn" if prob >= 0.33 else "is-ok")
                    st.markdown(
                        f"""
                        <div class="card">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                            <div style="font-size:18px;font-weight:800;color:var(--brand-800);">{MODEL_KEYS.get(model_used, model_used)}</div>
                            <div class="badge {badge_cls}">{label}</div>
                          </div>
                          <div class="kpi">
                            <div class="metric"><b>Probability of CKD</b><span>{prob:.3f}</span></div>
                            <div class="metric"><b>Decision Threshold</b><span>{thr:.3f}</span></div>
                          </div>
                          <div class="small-muted" style="margin-top:8px;">
                            A probability at or above the threshold is flagged for CKD follow-up by this model.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**What it means**")
                    st.markdown("This model would **flag** for CKD **follow-up**." if prob >= thr else "This model would **not** flag for CKD at this time.")
            except requests.HTTPError as e:
                with cols[idx]:
                    st.error(f"{MODEL_KEYS.get(m,m)} failed")
                    msg = getattr(e.response, "text", str(e)) or str(e)
                    st.code(msg, language="json")
                _log(f"Single predict HTTP ERROR ({m}): {e}")
            except Exception as e:
                with cols[idx]:
                    st.error(f"{MODEL_KEYS.get(m,m)} error")
                    st.caption(str(e))
                _log(f"Single predict ERROR ({m}): {e}")

        # Explainability
        st.markdown("#### Why did the model think that? (Top drivers)")
        st.caption("Higher bars = stronger influence on the decision for this case.")
        ecols = st.columns(len(selected_models))
        for idx, m in enumerate(selected_models):
            if m not in st.session_state["last_pred_results"]:
                continue
            with ecols[idx]:
                try:
                    top, raw_exp, mode = explain_with_fallback(st.session_state["api_url"], st.session_state["last_pred_payload"], m)
                    if top:
                        df_top = pd.DataFrame(top)
                        if "impact" not in df_top.columns and "signed" in df_top.columns:
                            df_top["impact"] = df_top["signed"].abs()
                        st.bar_chart(df_top.set_index("feature")["impact"])
                        bullets = "\n".join(
                            f"- **{row['feature']}** {'↑' if float(row.get('signed',0))>0 else '↓'} risk (Δprob={float(row.get('signed',0)):+.3f})"
                            for row in top[:5]
                        )
                        st.markdown(bullets)
                        note = "Server SHAP" if mode == "server_shap" else "Local sensitivity fallback"
                        st.caption(f"*Explainer: {note}*")

                        # Store explainability context for Chat
                        try:
                            st.session_state["ctx_explain"] = {
                                "schema_version": 1,
                                "model": m,
                                "top": top[:8],
                                "mode": mode
                            }
                        except Exception:
                            pass

                        if raw_exp:
                            with st.expander("Raw explanation (debug)", expanded=False):
                                st.json(raw_exp)
                    else:
                        st.info("No features available for explanation.")
                    _log(f"Explain OK ({m}) via {mode}.")
                except requests.HTTPError as e:
                    try:
                        det = e.response.json()
                    except Exception:
                        det = getattr(e.response, "text", str(e))
                    st.error("Explain failed")
                    st.code(det, language="json")
                    _log(f"Explain HTTP ERROR ({m}): {e}")
                except Exception as e:
                    st.error("Explain error")
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
