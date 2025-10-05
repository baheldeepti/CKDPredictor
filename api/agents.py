# api/agents.py
"""
LLM agent utilities for NephroCompass / CKD Predictor.

- Uses an OpenAI-compatible client (OpenRouter, Together, OpenAI, etc.) via either:
    * the OpenAI Python SDK (preferred when installed), or
    * a raw HTTP fallback.

Exports:
    * multi_agent_plan(summary: dict) -> dict
        Returns a structured object with 'sections' for easy UI rendering.
    * cohort_insights_card(metrics: dict) -> dict
    * ask_llm(system: str, user: str, temperature: float = 0.2) -> dict
        => {"ok": bool, "text": str|None, "error": str|None}

Environment variables:
- LLM_API_KEY   : API key for your provider (required for production).
- LLM_BASE_URL  : default https://openrouter.ai/api/v1 (works for OpenRouter).
- LLM_MODEL     : default meta-llama/llama-3.1-8b-instruct (set to your provider’s model).
- APP_URL       : optional; sent as HTTP-Referer for OpenRouter best practices.
- APP_TITLE     : optional; sent as X-Title for OpenRouter best practices.
"""

from __future__ import annotations

import os
import re
import json
from typing import Optional, Dict, Any, List

# Try to import the OpenAI SDK; if unavailable, we fall back to raw HTTP.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

import requests

# ---------------------------
# Config
# ---------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")

# Optional OpenRouter branding headers (used by HTTP fallback only)
APP_URL   = os.getenv("APP_URL", "https://github.com/baheldeepti/CKDPredictor")
APP_TITLE = os.getenv("APP_TITLE", "NephroCompass")

# ---------------------------
# Internal helpers
# ---------------------------
def _client() -> Optional["OpenAI"]:
    """Return an OpenAI client configured for an OpenAI-compatible base_url, or None."""
    if not LLM_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    except Exception:
        return None

def _http_fallback_chat(system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Raw HTTP call to {LLM_BASE_URL}/chat/completions (OpenAI-compatible).
    Returns {"ok": bool, "text": str|None, "error": str|None}
    """
    if not LLM_API_KEY:
        return {"ok": False, "text": None, "error": "LLM_API_KEY not configured."}

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter best practice (harmless for other providers)
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
    }
    payload = {
        "model": LLM_MODEL,
        "temperature": float(temperature),
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
    High-level LLM call. Tries OpenAI SDK first (if installed), then raw HTTP.
    Returns: { "ok": bool, "text": str|None, "error": str|None }
    """
    client = _client()
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=float(temperature),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content if getattr(resp, "choices", None) else None
            if not text:
                return {"ok": False, "text": None, "error": "Empty completion."}
            return {"ok": True, "text": text, "error": None}
        except Exception:
            # Silent fall-through to HTTP fallback; provider differences are common
            pass
    return _http_fallback_chat(system, user, temperature)

# ---------------------------
# Parsing helpers for structured UI output
# ---------------------------
_SECTION_TITLES = [
    "Clinical Interpretation",
    "Diet & Lifestyle",
    "Follow-up & Referrals",
    "Patient Communication (Plain Language)",
]

def _split_to_sections(md_text: str) -> List[Dict[str, Any]]:
    """
    Parse a simple markdown plan into sections the UI can render:
      [{"title":..., "bullets":[...]}]
    We look for H3 headings matching the expected titles; otherwise we create a
    single section with the whole text.
    """
    if not md_text or not isinstance(md_text, str):
        return []

    # Normalize line endings
    text = md_text.replace("\r\n", "\n").strip()

    # Try to split on ### headers first
    parts = re.split(r"\n\s*###\s+", "\n### " + text)  # ensure it starts with a header for uniformity
    sections = []
    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Title = first line up to newline
        first_nl = chunk.find("\n")
        if first_nl == -1:
            title = chunk.strip("# ").strip()
            body = ""
        else:
            title = chunk[:first_nl].strip("# ").strip()
            body = chunk[first_nl + 1 :].strip()

        # Collect bullets: lines that start with "-" or "•"
        bullets = []
        for line in body.splitlines():
            l = line.strip()
            if l.startswith("- "):
                bullets.append(l[2:].strip())
            elif l.startswith("• "):
                bullets.append(l[2:].strip())
        # Fallback: if no bullets, keep body as text
        sec: Dict[str, Any] = {"title": title}
        if bullets:
            sec["bullets"] = bullets
        elif body:
            sec["text"] = body
        sections.append(sec)

    if sections:
        return sections

    # Fallback: one big section
    return [{"title": "Care Plan", "text": text}]

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

    Return format (preferred by UI):
      { "sections": [ { "title": "...", "bullets": ["..."] }, ... ] }
    On failure:
      { "error": "<message>" }
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

    sections = _split_to_sections(res["text"])
    if not sections:
        # Keep a plain fallback to display *something* in the UI
        return {"plan": res["text"]}
    return {"sections": sections}

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
    Returns either {"summary": "...markdown..."} or {"error": "..."}.
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
