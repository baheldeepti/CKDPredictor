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
MODEL_LABELS = {label: key for (label, key) in MODEL_CHOICES}  # UI label -> key
MODEL_KEYS   = {key: label for (label, key) in MODEL_CHOICES}  # key -> UI label

# Optional LLM config for the "AI summary & next steps" tab + Chat
LLM_PROVIDER = st.secrets.get("LLM_PROVIDER", "openrouter")  # openai | together | openrouter | custom
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL    = st.secrets.get("LLM_MODEL", "openrouter/auto")

# Optional branding for OpenRouter headers
APP_URL   = st.secrets.get("APP_URL", APP_REPO_URL)
APP_TITLE = st.secrets.get("APP_TITLE", APP_TITLE_HUMAN)

st.set_page_config(page_title=APP_TITLE_HUMAN, page_icon="🩺", layout="wide")

# -------------------------
# Professional Medical UI Styling
# -------------------------
STYLE = """
<style>
/* Import professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  /* Professional Medical Color Palette */
  --primary-blue: #0066CC;
  --primary-blue-dark: #004C99;
  --primary-blue-light: #E6F2FF;
  --primary-blue-hover: #0052A3;
  
  --secondary-teal: #008B8B;
  --secondary-teal-dark: #006666;
  --secondary-teal-light: #E6F7F7;
  
  --accent-green: #00A878;
  --accent-green-dark: #008060;
  --accent-green-light: #E6F9F4;
  
  --warning-amber: #F59E0B;
  --warning-amber-dark: #D97706;
  --warning-amber-light: #FEF3C7;
  
  --danger-red: #DC2626;
  --danger-red-dark: #B91C1C;
  --danger-red-light: #FEE2E2;
  
  --success-green: #059669;
  --success-green-dark: #047857;
  --success-green-light: #D1FAE5;
  
  /* Neutral Grays */
  --gray-50: #F9FAFB;
  --gray-100: #F3F4F6;
  --gray-200: #E5E7EB;
  --gray-300: #D1D5DB;
  --gray-400: #9CA3AF;
  --gray-500: #6B7280;
  --gray-600: #4B5563;
  --gray-700: #374151;
  --gray-800: #1F2937;
  --gray-900: #111827;
  
  /* Backgrounds */
  --bg-primary: #FFFFFF;
  --bg-secondary: #F9FAFB;
  --bg-tertiary: #F3F4F6;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  
  /* Border Radius */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --radius-xl: 18px;
  
  /* Transitions */
  --transition-fast: 150ms ease;
  --transition-base: 250ms ease;
  --transition-slow: 350ms ease;
}

/* Global Styles */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
  color: var(--gray-800);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Main Container */
.main {
  background-color: var(--bg-secondary);
}

section.main > div {
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
}

/* Typography Hierarchy */
h1 {
  font-size: 2.5rem !important;
  font-weight: 700 !important;
  color: var(--gray-900) !important;
  letter-spacing: -0.025em !important;
  line-height: 1.2 !important;
  margin-bottom: 0.5rem !important;
}

h2 {
  font-size: 1.875rem !important;
  font-weight: 600 !important;
  color: var(--gray-800) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.3 !important;
  margin-top: 2rem !important;
  margin-bottom: 1rem !important;
}

h3 {
  font-size: 1.5rem !important;
  font-weight: 600 !important;
  color: var(--primary-blue) !important;
  letter-spacing: -0.015em !important;
  line-height: 1.4 !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}

h4 {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
  color: var(--gray-700) !important;
  letter-spacing: -0.01em !important;
  margin-top: 1.25rem !important;
  margin-bottom: 0.5rem !important;
}

p, .stMarkdown, .stText {
  font-size: 1rem !important;
  line-height: 1.6 !important;
  color: var(--gray-700) !important;
}

/* Caption and Small Text */
.small-muted, .stCaption, caption {
  color: var(--gray-500) !important;
  font-size: 0.875rem !important;
  line-height: 1.5 !important;
  font-weight: 400 !important;
}

.top-chip {
  font-size: 0.8125rem !important;
  color: var(--gray-500) !important;
  margin-top: -4px !important;
  margin-bottom: 12px !important;
  font-weight: 500 !important;
}

.tooltip {
  font-size: 0.8125rem !important;
  color: var(--gray-600) !important;
  line-height: 1.5 !important;
}

.block-info {
  font-size: 0.9375rem !important;
  color: var(--gray-600) !important;
  line-height: 1.6 !important;
}

/* Professional Card Component */
.card {
  background: var(--bg-primary);
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  margin: 1rem 0;
  box-shadow: var(--shadow-sm);
  transition: all var(--transition-base);
}

.card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--gray-300);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--gray-100);
}

.card-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--gray-900);
  letter-spacing: -0.01em;
}

/* Badge Component */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.375rem 0.875rem;
  border-radius: 9999px;
  font-size: 0.8125rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  text-transform: uppercase;
  line-height: 1;
  transition: all var(--transition-fast);
}

.badge-success {
  background: var(--success-green-light);
  color: var(--success-green-dark);
}

.badge-warning {
  background: var(--warning-amber-light);
  color: var(--warning-amber-dark);
}

.badge-danger {
  background: var(--danger-red-light);
  color: var(--danger-red-dark);
}

.badge-info {
  background: var(--primary-blue-light);
  color: var(--primary-blue-dark);
}

/* Button Styling */
.stButton > button, .stDownloadButton > button {
  background: var(--primary-blue) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 0.625rem 1.25rem !important;
  font-weight: 600 !important;
  font-size: 0.9375rem !important;
  letter-spacing: 0.01em !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all var(--transition-base) !important;
  cursor: pointer !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--primary-blue-hover) !important;
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px);
}

.stButton > button:active, .stDownloadButton > button:active {
  transform: translateY(0);
  box-shadow: var(--shadow-sm) !important;
}

.stButton > button:focus, .stDownloadButton > button:focus {
  outline: 2px solid var(--primary-blue-light) !important;
  outline-offset: 2px !important;
}

/* Secondary Button Variant */
.stButton > button[kind="secondary"] {
  background: var(--bg-primary) !important;
  color: var(--primary-blue) !important;
  border: 1px solid var(--gray-300) !important;
}

.stButton > button[kind="secondary"]:hover {
  background: var(--gray-50) !important;
  border-color: var(--gray-400) !important;
}

/* Form Elements */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div,
.stMultiselect > div > div {
  border: 1px solid var(--gray-300) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.625rem 0.875rem !important;
  font-size: 0.9375rem !important;
  background: var(--bg-primary) !important;
  color: var(--gray-900) !important;
  transition: all var(--transition-fast) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea textarea:focus,
.stSelectbox > div > div:focus-within,
.stMultiselect > div > div:focus-within {
  border-color: var(--primary-blue) !important;
  box-shadow: 0 0 0 3px var(--primary-blue-light) !important;
  outline: none !important;
}

/* Form Labels */
.stTextInput > label,
.stNumberInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiselect > label,
.stSlider > label,
.stCheckbox > label {
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  color: var(--gray-700) !important;
  margin-bottom: 0.375rem !important;
  letter-spacing: 0.01em !important;
}

/* Slider Styling */
.stSlider > div > div > div {
  background: var(--primary-blue) !important;
}

.stSlider > div > div > div > div {
  background: var(--primary-blue) !important;
  border: 2px solid white !important;
  box-shadow: var(--shadow-md) !important;
}

/* Checkbox Styling */
.stCheckbox > label > div {
  background: var(--bg-primary) !important;
  border: 2px solid var(--gray-300) !important;
  border-radius: var(--radius-sm) !important;
}

.stCheckbox > label > div[data-checked="true"] {
  background: var(--primary-blue) !important;
  border-color: var(--primary-blue) !important;
}

/* File Uploader */
.stFileUploader > div {
  border: 2px dashed var(--gray-300) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--gray-50) !important;
  padding: 2rem !important;
  transition: all var(--transition-base) !important;
}

.stFileUploader > div:hover {
  border-color: var(--primary-blue) !important;
  background: var(--primary-blue-light) !important;
}

/* Dataframe Styling */
.stDataFrame {
  border: 1px solid var(--gray-200) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

.stDataFrame thead tr th {
  background: var(--gray-50) !important;
  color: var(--gray-700) !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.75rem !important;
  border-bottom: 2px solid var(--gray-200) !important;
}

.stDataFrame tbody tr td {
  padding: 0.625rem 0.75rem !important;
  font-size: 0.875rem !important;
  border-bottom: 1px solid var(--gray-100) !important;
}

.stDataFrame tbody tr:hover {
  background: var(--gray-50) !important;
}

/* Table Styling */
table {
  border-collapse: separate !important;
  border-spacing: 0 !important;
  width: 100% !important;
  border: 1px solid var(--gray-200) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

thead tr th {
  background: var(--gray-50) !important;
  color: var(--gray-700) !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.875rem !important;
  text-align: left !important;
  border-bottom: 2px solid var(--gray-200) !important;
}

tbody tr td {
  padding: 0.75rem 0.875rem !important;
  font-size: 0.875rem !important;
  color: var(--gray-700) !important;
  border-bottom: 1px solid var(--gray-100) !important;
}

tbody tr:last-child td {
  border-bottom: none !important;
}

tbody tr:hover {
  background: var(--gray-50) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: var(--bg-primary);
  padding: 0.5rem;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
}

.stTabs [data-baseweb="tab"] {
  height: auto;
  padding: 0.625rem 1.25rem !important;
  background: transparent;
  border-radius: var(--radius-md);
  font-weight: 500;
  font-size: 0.9375rem;
  color: var(--gray-600);
  border: none;
  transition: all var(--transition-fast);
}

.stTabs [data-baseweb="tab"]:hover {
  background: var(--gray-50);
  color: var(--gray-800);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: var(--primary-blue) !important;
  color: white !important;
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}

.stTabs [data-baseweb="tab-highlight"] {
  display: none;
}

/* Expander Styling */
.streamlit-expanderHeader {
  background: var(--gray-50) !important;
  border: 1px solid var(--gray-200) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.875rem 1rem !important;
  font-weight: 600 !important;
  font-size: 0.9375rem !important;
  color: var(--gray-800) !important;
  transition: all var(--transition-fast) !important;
}

.streamlit-expanderHeader:hover {
  background: var(--gray-100) !important;
  border-color: var(--gray-300) !important;
}

.streamlit-expanderContent {
  border: 1px solid var(--gray-200) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
  padding: 1rem !important;
  background: var(--bg-primary) !important;
}

/* Alert/Message Boxes */
.stAlert {
  border-radius: var(--radius-md) !important;
  border: 1px solid !important;
  padding: 1rem 1.25rem !important;
  font-size: 0.9375rem !important;
  line-height: 1.5 !important;
}

.stSuccess {
  background: var(--success-green-light) !important;
  border-color: var(--success-green) !important;
  color: var(--success-green-dark) !important;
}

.stWarning {
  background: var(--warning-amber-light) !important;
  border-color: var(--warning-amber) !important;
  color: var(--warning-amber-dark) !important;
}

.stError {
  background: var(--danger-red-light) !important;
  border-color: var(--danger-red) !important;
  color: var(--danger-red-dark) !important;
}

.stInfo {
  background: var(--primary-blue-light) !important;
  border-color: var(--primary-blue) !important;
  color: var(--primary-blue-dark) !important;
}

/* Divider */
hr {
  border: none !important;
  border-top: 1px solid var(--gray-200) !important;
  margin: 2rem 0 !important;
}

/* Code Blocks */
pre, code {
  background: var(--gray-900) !important;
  color: var(--gray-100) !important;
  border-radius: var(--radius-md) !important;
  padding: 1rem !important;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
  font-size: 0.875rem !important;
  line-height: 1.5 !important;
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
  background: var(--bg-primary) !important;
  border-right: 1px solid var(--gray-200) !important;
  padding: 2rem 1.5rem !important;
}

section[data-testid="stSidebar"] h3 {
  font-size: 1.125rem !important;
  font-weight: 700 !important;
  color: var(--gray-900) !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}

section[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  justify-content: center !important;
}

/* Chat Messages */
.stChatMessage {
  background: var(--bg-primary) !important;
  border: 1px solid var(--gray-200) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem 1.25rem !important;
  margin: 0.75rem 0 !important;
  box-shadow: var(--shadow-sm) !important;
}

.stChatMessage[data-testid="user-message"] {
  background: var(--primary-blue-light) !important;
  border-color: var(--primary-blue) !important;
}

.stChatMessage[data-testid="assistant-message"] {
  background: var(--bg-primary) !important;
  border-color: var(--gray-200) !important;
}

/* Spinner */
.stSpinner > div {
  border-top-color: var(--primary-blue) !important;
  border-right-color: var(--primary-blue) !important;
}

/* Progress Bar */
.stProgress > div > div > div {
  background: var(--primary-blue) !important;
}

/* Metric Component */
.stMetric {
  background: var(--bg-primary);
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-lg);
  padding: 1.25rem;
  box-shadow: var(--shadow-sm);
}

.stMetric label {
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  color: var(--gray-600) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

.stMetric [data-testid="stMetricValue"] {
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--gray-900) !important;
}

/* Tooltips */
[data-testid="stTooltipIcon"] {
  color: var(--gray-400) !important;
}

/* JSON/Dict Display */
.stJson {
  background: var(--gray-50) !important;
  border: 1px solid var(--gray-200) !important;
  border-radius: var(--radius-md) !important;
  padding: 1rem !important;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
  font-size: 0.875rem !important;
}

/* Plotly Charts */
.js-plotly-plot {
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Loading State */
.stApp [data-testid="stAppViewContainer"] > section:first-child {
  background: var(--bg-secondary);
}

/* Scrollbar Styling */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: var(--gray-100);
  border-radius: 5px;
}

::-webkit-scrollbar-thumb {
  background: var(--gray-400);
  border-radius: 5px;
  transition: background var(--transition-fast);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--gray-500);
}

/* Responsive Improvements */
@media (max-width: 768px) {
  h1 {
    font-size: 2rem !important;
  }
  
  h2 {
    font-size: 1.5rem !important;
  }
  
  h3 {
    font-size: 1.25rem !important;
  }
  
  .card {
    padding: 1rem;
  }
}

/* Focus Visible for Accessibility */
*:focus-visible {
  outline: 2px solid var(--primary-blue) !important;
  outline-offset: 2px !important;
}

/* Print Styles */
@media print {
  .stButton, .stDownloadButton {
    display: none !important;
  }
  
  .card {
    break-inside: avoid;
    box-shadow: none !important;
    border: 1px solid var(--gray-300) !important;
  }
}
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
and we simulate **every combination** to see how the risk changes. It's like trying out all "what-if"
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
        "<div class='top-chip'>"
        "<b>Train models</b> retrains on the server • "
        "<b>Reload models</b> refreshes cached artifacts"
        "</div>",
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
- If data is missing for the user's question, say: "Out of scope for this chat. Load Digital Twin, Counterfactuals, or Similar Patients first."
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
                    
                    # Use CSS variables for consistent styling
                    if prob >= thr:
                        badge_class = "badge-danger"
                    elif prob >= 0.33:
                        badge_class = "badge-warning"
                    else:
                        badge_class = "badge-success"
                    
                    st.markdown(
                        f"""
                        <div class="card">
                          <div class="card-header">
                            <div class="card-title">{MODEL_KEYS.get(model_used, model_used)}</div>
                            <div class="badge {badge_class}">{label}</div>
                          </div>
                          <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem;">
                            <div>
                              <div style="font-size:0.875rem;font-weight:600;color:var(--gray-600);margin-bottom:0.25rem;">Probability of CKD</div>
                              <div style="font-size:1.875rem;font-weight:700;color:var(--gray-900);">{prob:.3f}</div>
                            </div>
                            <div>
                              <div style="font-size:0.875rem;font-weight:600;color:var(--gray-600);margin-bottom:0.25rem;">Decision Threshold</div>
                              <div style="font-size:1.875rem;font-weight:700;color:var(--gray-900);">{thr:.3f}</div>
                            </div>
                          </div>
                          <div class="small-muted" style="margin-top:1rem;padding-top:1rem;border-top:1px solid var(--gray-100);">
                            A probability at or above the threshold is flagged for CKD follow-up by this model.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Clinical Interpretation**")
                    if prob >= thr:
                        st.markdown("🔴 This model would **flag for CKD follow-up**. Consider further evaluation and specialist consultation.")
                    else:
                        st.markdown("🟢 This model would **not flag for CKD** at this time. Continue routine monitoring as appropriate.")
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
                                f"Downloa
