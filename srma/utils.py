"""
srma/utils.py — Standalone utilities (no external framework dependencies)

Covers:
  - Google Gemini API call (reads GOOGLE_AI_API_KEY from env or credentials file)
  - Text normalization helpers
  - JSON parsing helpers
  - Project directory helpers
"""

from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Optional

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Optional .env fallback. Env var GOOGLE_AI_API_KEY always wins.
# Override the file location with GEMINI_ENV_FILE; otherwise ./gemini.env or ./.env.

# Default model mapping. Cheap flash for everything by design.
ROLE_TO_MODEL: dict[str, str] = {
    "screening":   "gemini-3-flash-preview",
    "extraction":  "gemini-3-flash-preview",
    "drafting":    "gemini-3-flash-preview",
    "polishing":   "gemini-3-flash-preview",
}


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Returns {} if absent."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def get_api_key() -> str:
    """Resolve Google Gemini API key: env var first, then optional .env file."""
    key = os.environ.get("GOOGLE_AI_API_KEY", "").strip()
    if not key:
        candidates = []
        override = os.environ.get("GEMINI_ENV_FILE", "").strip()
        if override:
            candidates.append(Path(override))
        candidates += [Path.cwd() / "gemini.env", Path.cwd() / ".env"]
        for p in candidates:
            key = _load_env_file(p).get("GOOGLE_AI_API_KEY", "").strip()
            if key:
                break
    if not key:
        raise RuntimeError(
            "GOOGLE_AI_API_KEY not set. Export it, or put it in a "
            "gemini.env / .env file, or set GEMINI_ENV_FILE=/path/to/file"
        )
    return key


def call_llm(
    prompt:        str,
    role:          str   = "screening",
    system_prompt: str   = "",
    temperature:   float = 0.0,
    max_tokens:    int   = 2000,
    timeout:       int   = 60,
    model:         Optional[str] = None,
) -> str:
    """
    Call the Google Gemini API and return the text response.

    Parameters
    ----------
    prompt        : User message
    role          : Determines model via ROLE_TO_MODEL (default flash)
    system_prompt : Optional system-level instruction
    temperature   : 0.0 for deterministic screening, higher for drafting
    max_tokens    : Max response tokens
    timeout       : Request timeout in seconds
    model         : Override model string (e.g. "gemini-3-flash-preview")

    Returns
    -------
    str : Model response text

    Raises
    ------
    RuntimeError : If API key missing or request fails
    """
    _model  = model or ROLE_TO_MODEL.get(role, "gemini-3-flash-preview")
    api_key = get_api_key()

    url = f"{GEMINI_API_BASE}/{_model}:generateContent?key={api_key}"
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
            # Gemini 3 is a thinking model; "low" keeps screening cheap and
            # prevents thinking tokens from starving the output budget.
            "thinkingConfig":  {"thinkingLevel": "low"},
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    resp = requests.post(
        url,
        headers={"content-type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    try:
        parts = data["candidates"][0]["content"]["parts"]
        text  = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

    usage = data.get("usageMetadata", {})
    if usage:
        print(
            f"  [API] {role} → {_model} "
            f"| in:{usage.get('promptTokenCount',0)} "
            f"out:{usage.get('candidatesTokenCount',0)}"
        )
    return text


# ─── Text normalization ───────────────────────────────────────────────────────

def normalize_doi(doi: str) -> str:
    """
    Strip URL prefix and lowercase a DOI string.

    >>> normalize_doi("https://doi.org/10.1234/example")
    '10.1234/example'
    >>> normalize_doi("  10.1234/EXAMPLE  ")
    '10.1234/example'
    """
    if not doi:
        return ""
    doi = str(doi).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def normalize_title(title: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.

    >>> normalize_title("  Effect of Surgery on Outcomes: A Meta-Analysis  ")
    'effect of surgery on outcomes a metaanalysis'
    """
    if not title:
        return ""
    t = str(title).lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def safe_parse_json(response: str) -> Optional[list]:
    """
    Extract a JSON array from an LLM response, tolerant to surrounding text.

    Returns None if no valid JSON array found.

    >>> safe_parse_json('[{"id": "1", "decision": "include"}]')
    [{'id': '1', 'decision': 'include'}]
    >>> safe_parse_json('Sure, here is the result: [{"id":"1"}]')
    [{'id': '1'}]
    >>> safe_parse_json('No JSON here') is None
    True
    """
    try:
        start = response.find("[")
        end   = response.rfind("]") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        return None


def safe_parse_json_object(response: str) -> Optional[dict]:
    """
    Extract a JSON object from an LLM response.

    >>> safe_parse_json_object('{"key": "value"}')
    {'key': 'value'}
    >>> safe_parse_json_object('Result: {"a": 1} done')
    {'a': 1}
    """
    try:
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        return None


# ─── Project directory helpers ────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_artifacts_dir(project_path: str | Path) -> Path:
    """Return (and create) the artifacts/ subdirectory of a project."""
    return ensure_dir(Path(project_path) / "artifacts")


def project_raw_dir(project_path: str | Path) -> Path:
    """
    Return the raw/ directory for a project.
    Falls back to artifacts/ for backward compatibility.
    """
    raw = Path(project_path) / "raw"
    if raw.is_dir():
        return raw
    return project_artifacts_dir(project_path)
