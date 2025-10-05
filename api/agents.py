# api/agents.py
"""
LLM agent utilities for NephroCompass.

- Uses OpenRouter via the OpenAI SDK-compatible client.
- Default model: meta-llama/llama-3.1-8b-instruct
- Exposes:
    * multi_agent_plan(summary: dict) -> dict
    * cohort_insights_card(metrics: dict) -> dict
    * ask_llm(system: str, user: str, temperature: float = 0.2) -> dict

Environment variables:
- LLM_API_KEY     : required for OpenRouter
- LLM_BASE_URL    : defaults to https://openrouter.ai/api/v1
- LLM_MODEL       : defaults to meta-llama/llama-3.1-8b-instruct
- APP_URL         : optional (OpenRouter best-practice header via raw fallback)
- APP_TITLE       : optional (OpenRouter best-practice header via raw fallback)
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any

# Preferred: OpenAI SDK with custom base_url (works with OpenRouter)
from openai import OpenAI

# Optional raw HTTP fallback (used only if SDK path fails)
import requests


# ---------------------------
# Config
# ---------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")  # REQUIRED in production
LLM_MODEL    = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")

# Optional OpenRouter branding headers (used by HTTP fallback only)
APP_URL   = os.getenv("APP_URL", "https://github.com/baheldeepti/CKDPredictor")
APP_TITLE = os.getenv("APP_TITLE", "NephroCompass")


# ---------------------------
# Internal helpers
# ---------------------------
def _client() -> Optional[OpenAI]:
    """Return an OpenAI client configured for OpenRouter, or None if key is missing."""
    if not LLM_API_KEY:
        return None
    try:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    except Exception:
        return None


def _http_fallback_chat(system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Raw HTTP call to OpenRouter (fallback if SDK path fails).
    Returns {"ok": bool, "text": str|None, "error": str|None}
    """
    if not LLM_API_KEY:
        return {"ok": False, "text": None, "error": "LLM_API_KEY not configured."}

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter best practice (optional)
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
    }
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return {"ok": False, "text": None, "error": f"HTTP {r.status_code}: {r.text}"}
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not text:
            return {"ok": False, "text": None, "error": "No content in LLM response."}
        return {"ok": True, "text": text, "error": None}
    except Exception as e:
        return {"ok": False, "text": None, "error": f"HTTP error: {e}"}


def ask_llm(system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    High-level LLM call. Tries OpenAI SDK first; falls back to raw HTTP.
    Returns dict: { "ok": bool, "text": str|None, "error": str|None }
    """
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content if resp.choices else None
            if not text:
                return {"ok": False, "text": None, "error": "Empty completion."}
            return {"ok": True, "text": text, "error": None}
        except Exception as e:
            # fall back to HTTP call
            return _http_fallback_chat(system, user, temperature)
    else:
        return _http_fallback_chat(system, user, temperature)


# ---------------------------
# Public: Multi-agent care plan
# ---------------------------
_MULTI_AGENT_SYSTEM = (
    "You are a cautious multi-agent coordinator for kidney risk recommendations.\n"
    "Agents: medical_analyst, diet_specialist, exercise_coach, care_coordinator.\n"
    "Rules:\n"
    "- No medication dosing.\n"
    "- Be specific but concise for a clinical audience.\n"
    "- Output short sections with clear bullets.\n"
    "- Always end with: 'This is not medical advice; consult your clinician.'"
)

def multi_agent_plan(summary: dict) -> dict:
    """
    Create a structured care plan from model outputs + flags/stage summary.

    Input example:
    {
      "label": "CKD",
      "prob_ckd": 0.81,
      "stage": 3,
      "albuminuria": 2,
      "flags": {"bp_risk":1, "hyperkalemia":0, "anemia":1},
      "notes": "Any short free text…"
    }

    Return:
    { "plan": "<markdown text>" } or { "error": "<msg>" }
    """
    user = (
        "Create a care plan for the following case summary (JSON):\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Structure exactly as:\n"
        "### Clinical Interpretation\n"
        "- ...\n\n"
        "### Diet & Lifestyle\n"
        "- ...\n\n"
        "### Follow-up & Referrals\n"
        "- ...\n\n"
        "### Patient Communication (Plain Language)\n"
        "- ...\n\n"
        "End with: 'This is not medical advice; consult your clinician.'"
    )

    res = ask_llm(_MULTI_AGENT_SYSTEM, user, temperature=0.2)
    if not res["ok"]:
        return {"error": res["error"]}
    return {"plan": res["text"]}


# ---------------------------
# Public: Cohort insights card
# ---------------------------
_COHORT_SYSTEM = (
    "You are a clinical data summarizer. Keep output concise and useful for an analytics dashboard card.\n"
    "Audience: data scientists + clinicians. Do not give medication dosing.\n"
)

def cohort_insights_card(metrics: dict) -> dict:
    """
    Produce a compact, sectioned cohort summary suitable for a UI card.
    Input example:
      {
        "model": "XGBoost",
        "positive_rate": 0.24,
        "avg_prob_ckd": 0.41,
        "rows": 512
      }

    Returns:
      { "summary": "<markdown text>" } or { "error": "<msg>" }
    """
    user = (
        "Summarize the cohort using these metrics (JSON):\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        "Output format (brief, bullet-heavy; no preamble):\n"
        "#### Red Flag Overview\n"
        "- ...\n"
        "#### High-Level Diet & Exercise\n"
        "- ...\n"
        "#### Follow-up Checklist\n"
        "- ...\n"
        "_This is not medical advice; consult your clinician._"
    )
    res = ask_llm(_COHORT_SYSTEM, user, temperature=0.2)
    if not res["ok"]:
        return {"error": res["error"]}
    return {"summary": res["text"]}


# ---------------------------
# Module exports
# ---------------------------
__all__ = [
    "ask_llm",
    "multi_agent_plan",
    "cohort_insights_card",
]
