"""
srma/extraction.py — Full-text Structured Data Extraction

The screening module (srma.screening) decides include/exclude from title+abstract.
This module does the next step: extract structured, prespecified codes from the
full text of each included study against a coding instrument you define.

Design principles (why this exists)
-----------------------------------
1. **Prespecified instrument.** You pass a coding schema (dimensions + anchor
   definitions). The model codes against YOUR rubric, not its own opinion.
2. **Human-in-the-loop.** Every record gets a per-dimension `confidence` and a
   source-checked `evidence` quote. Every model code remains provisional with
   `needs_review=True` until a human supplies the confirmed code.
3. **Ground-truth isolation.** This module must NEVER receive your gold-standard
   / human-consensus labels in its context. It codes blind. `assert_no_ground_truth`
   guards the schema against accidentally embedding an answer key.
   See academic-research-skills/shared/ground_truth_isolation_pattern.md.
4. **No inference.** The system prompt forbids guessing from absent text: if the
   report does not state something, the code is the "absent" anchor (usually 0),
   not the model's prior about what surgeons "probably" did.

Usage
-----
    from srma.extraction import load_schema_from_yaml, extract_records
    schema = load_schema_from_yaml("coding_schema.yaml")   # or build a dict
    coded = extract_records(df, schema, text_col="full_text")
    coded.to_csv("extraction_results.csv", index=False)

A coding schema is a plain dict:
    {
      "name": "MSR1 operator-dependence handling",
      "dimensions": [
        {
          "key": "D4_skill",
          "label": "Validated quantitative skill measurement",
          "anchors": {
            "0": "None, or credentialing/volume/outcome-benchmark only",
            "1": "Competency verification without a validated quantitative score",
            "2": "Validated instrument w/ published psychometrics, blinded, "
                 "independent of patient outcome",
          },
          "rule": "Volume and 'experienced surgeon' do NOT count. A named "
                  "instrument with published validity+reliability is required for 2.",
        },
        ...
      ],
    }
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from srma.utils import call_llm, safe_parse_json_object


# Tokens that, if present as a dimension key/field, suggest a leaked answer key.
_GROUND_TRUTH_MARKERS = {
    "gold", "gold_label", "answer", "answer_key", "expected",
    "consensus", "truth", "ground_truth", "correct_code",
}


# ─── SCHEMA HANDLING ──────────────────────────────────────────────────────────

def assert_no_ground_truth(schema: dict) -> None:
    """
    Guard: refuse a schema that appears to carry gold-standard labels.

    Ground-truth isolation is architectural, not advisory: if the extractor can
    see the answer key it will pattern-match to it. We block the obvious leak
    (an answer field embedded in the schema). This is a tripwire, not a sandbox.

    Raises
    ------
    ValueError : if any dimension carries a ground-truth-looking field.

    >>> assert_no_ground_truth({"dimensions": [{"key": "D1", "anchors": {}}]})
    >>> assert_no_ground_truth({"dimensions": [{"key": "D1", "gold": 2}]})
    Traceback (most recent call last):
        ...
    ValueError: schema dimension 'D1' carries ground-truth-like field 'gold' ...
    """
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary")
    dimensions = schema.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("schema must contain a non-empty dimensions list")

    seen: set[str] = set()
    for dim in dimensions:
        if not isinstance(dim, dict):
            raise TypeError("each schema dimension must be a dictionary")
        key = str(dim.get("key", "")).strip()
        if not key or key in seen:
            raise ValueError(f"schema dimension key is missing or duplicated: '{key}'")
        seen.add(key)
        anchors = dim.get("anchors")
        if not isinstance(anchors, dict) or not anchors:
            raise ValueError(f"schema dimension '{key}' must define anchors")

        stack = [dim]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for field, child in value.items():
                    if str(field).lower() in _GROUND_TRUTH_MARKERS:
                        raise ValueError(
                            f"schema dimension '{key}' carries ground-truth-like "
                            f"field '{field}' — extraction must run blind to gold labels."
                        )
                    stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)


def load_schema_from_yaml(path: str | Path) -> dict:
    """Load a coding schema from a YAML file (requires PyYAML)."""
    import yaml  # local import: optional dependency
    with open(path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    assert_no_ground_truth(schema)
    return schema


# ─── PROMPT CONSTRUCTION ──────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are a meticulous data-extraction assistant for a methodological study.
You apply a PRESPECIFIED coding instrument to the full text of a study report.

Hard rules:
- Code ONLY from the text provided. Do NOT use outside knowledge about the trial,
  the authors, or what surgeons "usually" do.
- If the report does not state something, code the ABSENT anchor (usually 0).
  Absence of evidence is coded as absence, never inferred upward.
- For every code above the absent anchor, you MUST supply a verbatim quote from
  the text as evidence. No quote -> code the absent anchor.
- confidence in [0,1] reflects how unambiguously the text supports the code.
- If the text is truncated, ambiguous, or you are unsure, set needs_review=true.
- Return ONLY one valid JSON object. No prose, no markdown."""


def build_extraction_prompt(text: str, schema: dict) -> str:
    """Construct the per-record extraction prompt from a coding schema."""
    assert_no_ground_truth(schema)
    dims_block = ""
    for dim in schema["dimensions"]:
        anchors = "\n".join(
            f"      {score} = {desc}" for score, desc in dim["anchors"].items()
        )
        rule = f"\n    RULE: {dim['rule']}" if dim.get("rule") else ""
        dims_block += (
            f"\n  - {dim['key']} ({dim.get('label', '')}):\n"
            f"{anchors}{rule}\n"
        )

    keys = [d["key"] for d in schema["dimensions"]]
    example_obj = ",\n".join(
        f'    "{k}": {{"code": <anchor>, "confidence": <0..1>, '
        f'"evidence": "<verbatim quote or empty>", "needs_review": <true|false>}}'
        for k in keys
    )

    return f"""\
CODING INSTRUMENT: {schema.get('name', 'unnamed')}

Dimensions and anchor definitions:
{dims_block}

STUDY FULL TEXT (code only from this):
\"\"\"
{text.strip()}
\"\"\"

Return a JSON object exactly in this shape:
{{
{example_obj}
}}
Return ONLY the JSON object."""


# ─── EXTRACTION ───────────────────────────────────────────────────────────────

def _coerce_code(value) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def _quote_is_supported(text: str, quote: str) -> bool:
    """Allow whitespace variation, but require the quote to occur in the source."""
    source = re.sub(r"\s+", " ", text).strip().casefold()
    evidence = re.sub(r"\s+", " ", quote).strip().casefold()
    return bool(evidence) and evidence in source


def extract_record(text: str, schema: dict, model: str = "extraction") -> dict:
    """
    Extract structured codes for ONE study's text against the schema.

    Returns a flat dict:
      {"D1_*_code", "D1_*_confidence", "D1_*_evidence", "D1_*_needs_review", ...,
       "_extraction_ok": bool}

    On parse failure, every dimension is left blank with needs_review=True and
    _extraction_ok=False (so a human picks it up — never silently dropped).
    """
    prompt = build_extraction_prompt(text, schema)
    dimensions = schema["dimensions"]
    keys = [d["key"] for d in dimensions]
    out: dict = {}

    try:
        raw = call_llm(prompt, role=model, system_prompt=_EXTRACTION_SYSTEM,
                       temperature=0.0, max_tokens=2000)
        parsed = safe_parse_json_object(raw)
    except Exception as exc:  # noqa: BLE001 — never let one record crash the batch
        print(f"  Error extracting record: {exc}")
        parsed = None

    if not parsed:
        for k in keys:
            out[f"{k}_code"] = ""
            out[f"{k}_confidence"] = 0.0
            out[f"{k}_evidence"] = ""
            out[f"{k}_needs_review"] = True
            out[f"{k}_human_code"] = ""
            out[f"{k}_review_status"] = "pending"
        out["_extraction_ok"] = False
        return out

    record_valid = True
    for dim in dimensions:
        k = dim["key"]
        item = parsed.get(k, {}) if isinstance(parsed.get(k), dict) else {}
        code = _coerce_code(item.get("code"))
        evidence = str(item.get("evidence", "")).strip()
        try:
            conf = float(item.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        allowed_codes = {str(value).strip() for value in dim["anchors"]}
        absent_code = str(dim.get("absent_code", next(iter(dim["anchors"])))).strip()
        code_valid = code in allowed_codes
        quote_valid = not evidence or _quote_is_supported(text, evidence)
        support_valid = code == absent_code or (bool(evidence) and quote_valid)
        if not code_valid or not support_valid:
            record_valid = False
        if not code_valid:
            code = ""

        out[f"{k}_code"] = code or ""
        out[f"{k}_confidence"] = conf
        out[f"{k}_evidence"] = evidence
        # Every model code remains a candidate until a human enters human_code.
        out[f"{k}_needs_review"] = True
        out[f"{k}_human_code"] = ""
        out[f"{k}_review_status"] = "pending"

    out["_extraction_ok"] = record_valid
    return out


def extract_records(
    df: pd.DataFrame,
    schema: dict,
    text_col: str = "full_text",
    model: str = "extraction",
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Extract codes for every row of *df* against *schema*.

    Parameters
    ----------
    df       : DataFrame; must contain *text_col* with the study text.
    schema   : coding schema dict (see module docstring).
    text_col : column holding the full text (or abstract) to code.
    model    : LLM role key (default 'extraction' -> Gemini Pro).
    out_path : if given, write CSV after every record (resume-safe checkpoint).

    Returns
    -------
    DataFrame = input columns + per-dimension code/confidence/evidence/needs_review
    + `_extraction_ok` + `_n_needs_review` (count of flagged dimensions per row).

    Notes
    -----
    One API call per record (full text is long; no batching). Records whose
    extraction fails default to all-blank + needs_review (never dropped).
    """
    assert_no_ground_truth(schema)
    if text_col not in df.columns:
        raise KeyError(f"text_col '{text_col}' not in DataFrame columns")

    df = df.reset_index(drop=True).copy()
    keys = [d["key"] for d in schema["dimensions"]]
    rows: list[dict] = []
    total = len(df)
    print(f"  Extracting {total} records against '{schema.get('name','schema')}'...")

    for i, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()
        if not text:
            rec = {}
            for k in keys:
                rec.update({
                    f"{k}_code": "",
                    f"{k}_confidence": 0.0,
                    f"{k}_evidence": "",
                    f"{k}_needs_review": True,
                    f"{k}_human_code": "",
                    f"{k}_review_status": "pending",
                })
            rec["_extraction_ok"] = False
        else:
            rec = extract_record(text, schema, model=model)

        rec["_n_needs_review"] = sum(
            bool(rec.get(f"{k}_needs_review")) for k in keys
        )
        rows.append(rec)
        print(f"  [{i+1}/{total}] ok={rec['_extraction_ok']} "
              f"flagged={rec['_n_needs_review']}/{len(keys)}")

        if out_path:  # checkpoint each record so a crash loses nothing
            pd.concat([df.iloc[: i + 1].reset_index(drop=True),
                       pd.DataFrame(rows)], axis=1).to_csv(out_path, index=False)

    result = pd.concat([df, pd.DataFrame(rows)], axis=1)
    n_flag = int(result["_n_needs_review"].gt(0).sum())
    print(f"\n  Done. {n_flag}/{total} records have >=1 dimension needing human review.")
    return result


def apply_human_extraction(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Validate and apply human codes; blank values remain pending review."""
    assert_no_ground_truth(schema)
    out = df.copy()
    for dim in schema["dimensions"]:
        k = dim["key"]
        human_col = f"{k}_human_code"
        if human_col not in out.columns:
            out[human_col] = ""
        human = out[human_col].fillna("").astype(str).str.strip()
        allowed = {str(value).strip() for value in dim["anchors"]}
        invalid = sorted(set(human[human.ne("")]) - allowed)
        if invalid:
            raise ValueError(f"invalid human code(s) for {k}: {', '.join(invalid)}")
        confirmed = human.isin(allowed)
        out[human_col] = human
        out[f"{k}_review_status"] = confirmed.map({True: "confirmed", False: "pending"})
        out[f"{k}_needs_review"] = ~confirmed
    return out


def extraction_summary(df: pd.DataFrame, schema: dict) -> str:
    """Summarize human-confirmed codes; never treat AI candidates as final."""
    df = apply_human_extraction(df, schema)
    lines = [f"# Human-confirmed extraction summary — {schema.get('name','schema')}", ""]
    for dim in schema["dimensions"]:
        k = dim["key"]
        col = f"{k}_human_code"
        if col not in df.columns:
            lines.append(f"## {k} — {dim.get('label','')}")
            lines.append("  pending human review: all rows")
            lines.append("")
            continue
        confirmed = df[col].fillna("").astype(str).str.strip()
        counts = confirmed[confirmed.ne("")].value_counts().to_dict()
        pending = int(confirmed.eq("").sum())
        lines.append(f"## {k} — {dim.get('label','')}")
        for code, n in sorted(counts.items(), key=lambda x: str(x[0])):
            lines.append(f"  confirmed code {code}: {n}")
        lines.append(f"  pending human review: {pending}")
        lines.append("")
    return "\n".join(lines)
