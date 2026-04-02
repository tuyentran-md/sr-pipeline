"""
srma/utils.py — Standalone utilities (no external framework dependencies)

Covers:
  - Anthropic API call (reads ANTHROPIC_API_KEY from env)
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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Default model mapping
ROLE_TO_MODEL: dict[str, str] = {
    "screening":   "claude-haiku-4-5-20251001",
    "extraction":  "claude-sonnet-4-6-20250514",
    "drafting":    "claude-sonnet-4-6-20250514",
    "polishing":   "claude-sonnet-4-6-20250514",
}


def get_api_key() -> str:
    """Resolve Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
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
    Call Anthropic API and return the text response.

    Parameters
    ----------
    prompt        : User message
    role          : Determines model (screening→haiku, extraction→sonnet, etc.)
    system_prompt : Optional system-level instruction
    temperature   : 0.0 for deterministic screening, higher for drafting
    max_tokens    : Max response tokens
    timeout       : Request timeout in seconds
    model         : Override model string (e.g. "claude-sonnet-4-6-20250514")

    Returns
    -------
    str : Model response text

    Raises
    ------
    RuntimeError : If API key missing or request fails
    """
    _model   = model or ROLE_TO_MODEL.get(role, "claude-haiku-4-5-20251001")
    api_key  = get_api_key()

    headers = {
        "x-api-key":         api_key,
        "content-type":      "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload: dict = {
        "model":       _model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if system_prompt:
        payload["system"] = system_prompt

    resp = requests.post(
        ANTHROPIC_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data  = resp.json()
    text  = data["content"][0]["text"]
    usage = data.get("usage", {})
    if usage:
        print(
            f"  [API] {role} → {_model} "
            f"| in:{usage.get('input_tokens',0)} "
            f"out:{usage.get('output_tokens',0)}"
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
