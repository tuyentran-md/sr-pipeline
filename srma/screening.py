"""
srma/screening.py — Title/Abstract Screening Pipeline

Steps
-----
1. merge_csvs       — Combine multiple database exports (PubMed, Scopus, Embase)
2. deduplicate      — Remove duplicates by DOI (exact) then title (fuzzy ≥ 0.90)
3. screen_records   — LLM batch screening against inclusion/exclusion criteria
4. generate_prisma  — Produce PRISMA 2020 flow report

Supported input formats
-----------------------
- Zotero CSV export (recommended)
- PubMed CSV export
- Scopus CSV export (partial column mapping)

Usage (CLI)
-----------
    python -m srma.screening \\
        --project-dir ./my_review \\
        --model screening

Usage (Python API)
------------------
    from srma.screening import run_pipeline

    results = run_pipeline(
        project_dir="./my_review",
        inclusion=["RCT or cohort study", "pediatric patients (0-18 years)"],
        exclusion=["case reports", "non-English"],
    )
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

from srma.utils import (
    call_llm,
    normalize_doi,
    normalize_title,
    safe_parse_json,
    project_artifacts_dir,
    project_raw_dir,
)

# Columns we try to read from any input CSV
_USE_COLS = [
    "Key", "Item Type", "Publication Year", "Author",
    "Title", "Abstract Note", "DOI", "Journal Abbreviation",
]

BATCH_SIZE = 10


# ─── STEP 1 — MERGE ───────────────────────────────────────────────────────────

def merge_csvs(raw_dir: str | Path) -> pd.DataFrame:
    """
    Merge all CSV files from *raw_dir* into a single DataFrame.

    Parameters
    ----------
    raw_dir : path to directory containing database-export CSVs

    Returns
    -------
    pd.DataFrame with standardized columns

    Raises
    ------
    FileNotFoundError : if no CSV files exist in raw_dir
    """
    raw_dir = Path(raw_dir)
    SKIP = {
        "merged.csv", "dedup.csv", "screening_results.csv",
        "r_dataset.csv", "extraction_template.csv",
    }
    files = [
        f for f in os.listdir(raw_dir)
        if f.endswith(".csv") and f not in SKIP
    ]
    if not files:
        raise FileNotFoundError(
            f"No source CSV files found in {raw_dir}\n"
            "Place PubMed/Zotero/Scopus exported CSVs there."
        )

    frames = []
    for fname in files:
        fpath = raw_dir / fname
        try:
            df   = pd.read_csv(fpath, low_memory=False)
            cols = [c for c in _USE_COLS if c in df.columns]
            df   = df[cols].copy()
            df["source_file"] = fname
            frames.append(df)
            print(f"  Loaded {fname}: {len(df)} records")
        except Exception as exc:
            print(f"  Warning: could not load {fname}: {exc}")

    if not frames:
        raise RuntimeError("All CSV files failed to load.")

    merged = pd.concat(frames, ignore_index=True)

    for col in ["Key", "Title", "Abstract Note", "DOI"]:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged.fillna("")
    print(f"\n  Merged total: {len(merged)} records from {len(frames)} files")
    return merged


# ─── STEP 2 — DEDUPLICATE ─────────────────────────────────────────────────────

def deduplicate(
    df: pd.DataFrame,
    title_threshold: float = 0.90,
) -> tuple[pd.DataFrame, int, int]:
    """
    Remove duplicate records by DOI (exact) then title similarity (fuzzy).

    Strategy
    --------
    1. Normalize DOI: strip URL prefix, lowercase
    2. If two records share a non-empty DOI → keep first
    3. If two titles share SequenceMatcher ratio ≥ threshold → keep first

    Parameters
    ----------
    df              : Input DataFrame (must have 'DOI' and 'Title' columns)
    title_threshold : Fuzzy match threshold, 0.0–1.0 (default 0.90)

    Returns
    -------
    (deduplicated_df, n_before, n_after)

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "Title": ["Effect of A on B", "Effect of A on B", "Different study"],
    ...     "DOI":   ["10.1/a",           "10.1/a",           "10.2/b"],
    ... })
    >>> clean, before, after = deduplicate(df)
    >>> after
    2
    """
    df = df.copy()
    df["_doi_norm"]   = df["DOI"].apply(normalize_doi)
    df["_title_norm"] = df["Title"].apply(normalize_title)
    df["_duplicate"]  = False

    seen_dois:   dict[str, int] = {}
    seen_titles: list[tuple[str, int]] = []

    for i, row in df.iterrows():
        doi   = row["_doi_norm"]
        title = row["_title_norm"]

        if doi:
            if doi in seen_dois:
                df.at[i, "_duplicate"] = True
                continue
            seen_dois[doi] = i

        is_dup = False
        for prev_title, _ in seen_titles:
            if SequenceMatcher(None, title, prev_title).ratio() >= title_threshold:
                df.at[i, "_duplicate"] = True
                is_dup = True
                break

        if not is_dup:
            seen_titles.append((title, i))

    n_before = len(df)
    df_clean = df[~df["_duplicate"]].drop(
        columns=["_doi_norm", "_title_norm", "_duplicate"]
    )
    n_after = len(df_clean)
    print(f"  Dedup: {n_before} → {n_after} (removed {n_before - n_after})")
    return df_clean, n_before, n_after


# ─── STEP 3 — SCREENING ───────────────────────────────────────────────────────

_SCREENING_SYSTEM = """\
You are a systematic review screener with expertise in medical literature.
Apply eligibility criteria precisely and consistently to screen titles and abstracts.

Rules:
- Apply criteria STRICTLY — do not speculate about missing information
- If abstract is missing or too short → "uncertain"
- confidence: 0.0–1.0, reflects how clearly criteria are met
- reason: 1 short sentence citing the specific criterion driving the decision
- Return ONLY a valid JSON array — no explanation, no markdown"""


def _build_screening_prompt(
    batch:     pd.DataFrame,
    inclusion: list[str],
    exclusion: list[str],
) -> str:
    incl_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(inclusion))
    excl_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(exclusion))

    records = ""
    for _, row in batch.iterrows():
        records += (
            f"\n---\n"
            f"ID: {row.get('_row_id', '')}\n"
            f"KEY: {row.get('Key', '')}\n"
            f"TITLE: {str(row.get('Title', '')).strip() or '(no title)'}\n"
            f"ABSTRACT: {str(row.get('Abstract Note', '')).strip() or '(no abstract)'}\n"
        )

    return f"""\
Screen the following records against the eligibility criteria.

INCLUSION CRITERIA (ALL must be met):
{incl_block}

EXCLUSION CRITERIA (ANY triggers exclusion):
{excl_block}

RECORDS:
{records}

Return a JSON array, one object per record:
[
  {{
    "id": "<ID>",
    "key": "<KEY or empty>",
    "decision": "include" | "exclude" | "uncertain",
    "confidence": <0.0–1.0>,
    "reason": "<one sentence>"
  }}
]
Return ONLY the JSON array."""


def _norm_decision(value: str) -> str:
    d = str(value).strip().lower()
    return d if d in {"include", "exclude", "uncertain"} else "uncertain"


def _norm_confidence(value) -> float:
    try:
        c = float(value)
        return max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        return 0.0


def screen_records(
    df:        pd.DataFrame,
    inclusion: list[str],
    exclusion: list[str],
    model:     str = "screening",
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Screen a DataFrame of records against inclusion/exclusion criteria via LLM.

    Parameters
    ----------
    df        : DataFrame with 'Title' and 'Abstract Note' columns
    inclusion : List of inclusion criteria strings
    exclusion : List of exclusion criteria strings
    model     : LLM role key (maps to model in utils.ROLE_TO_MODEL)
    batch_size: Records per API call (default 10)

    Returns
    -------
    DataFrame with added columns: decision, confidence, reason

    Notes
    -----
    Records that fail processing default to decision='uncertain'.
    Re-run with screen_records(...) on the uncertain subset to retry.
    """
    df = df.copy()
    if "_row_id" not in df.columns:
        df["_row_id"] = df.index.astype(str)

    df["decision"]   = "uncertain"
    df["confidence"] = 0.0
    df["reason"]     = ""

    id_to_idx = {str(rid): idx for idx, rid in df["_row_id"].items()}
    total     = len(df)
    failed    = []

    print(f"  Screening {total} records in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        batch   = df.iloc[start:start + batch_size]
        end     = min(start + batch_size, total)
        prompt  = _build_screening_prompt(batch, inclusion, exclusion)

        try:
            raw    = call_llm(prompt, role=model, system_prompt=_SCREENING_SYSTEM,
                              temperature=0.0, max_tokens=1500)
            parsed = safe_parse_json(raw)

            if not parsed:
                print(f"  Warning: batch {start+1}-{end}: JSON parse failed")
                failed.append((start, end))
                continue

            matched = 0
            for item in parsed:
                row_id = str(item.get("id", "")).strip()
                idx    = id_to_idx.get(row_id)
                if idx is None:
                    continue
                df.at[idx, "decision"]   = _norm_decision(item.get("decision", "uncertain"))
                df.at[idx, "confidence"] = _norm_confidence(item.get("confidence", 0.0))
                df.at[idx, "reason"]     = str(item.get("reason", "")).strip()
                matched += 1

            print(f"  Batch {start+1}-{end}: {matched}/{len(batch)} matched")

        except Exception as exc:
            print(f"  Error batch {start+1}-{end}: {exc}")
            failed.append((start, end))

    if failed:
        n_failed = sum(e - s for s, e in failed)
        print(f"\n  {len(failed)} batches failed ({n_failed} records → 'uncertain')")

    return df


# ─── STEP 4 — PRISMA REPORT ───────────────────────────────────────────────────

def generate_prisma_report(
    project_name: str,
    n_raw:        int,
    n_after_dedup: int,
    screening_df: pd.DataFrame,
) -> tuple[str, int, int, int]:
    """
    Generate a PRISMA 2020-style flow summary.

    Returns
    -------
    (report_text, n_included, n_excluded, n_uncertain)
    """
    n_included  = int((screening_df["decision"] == "include").sum())
    n_excluded  = int((screening_df["decision"] == "exclude").sum())
    n_uncertain = int((screening_df["decision"] == "uncertain").sum())

    report = f"""\
# PRISMA Flow — {project_name}

## Identification
- Records from database searches: {n_raw}

## Screening
- After deduplication: {n_after_dedup}
- Screened (title + abstract): {n_after_dedup}
  - Included:  {n_included}
  - Excluded:  {n_excluded}
  - Uncertain: {n_uncertain} (manual review needed)

## Next Step
- Manual review of {n_uncertain} uncertain record(s)
- Full-text assessment of {n_included} included record(s)

---
*Generated by sr-pipeline (https://github.com/tuyentran-md/sr-pipeline)*
"""
    return report, n_included, n_excluded, n_uncertain


# ─── HIGH-LEVEL API ───────────────────────────────────────────────────────────

def run_pipeline(
    project_dir:    str | Path,
    inclusion:      list[str],
    exclusion:      list[str],
    model:          str = "screening",
    skip_dedup:     bool = False,
    skip_screen:    bool = False,
    retry_uncertain: bool = False,
) -> dict:
    """
    End-to-end screening pipeline for a project directory.

    Directory layout expected
    -------------------------
    project_dir/
      raw/            ← Put your exported CSVs here
      artifacts/      ← Pipeline outputs written here (auto-created)

    Parameters
    ----------
    project_dir     : Path to project folder
    inclusion       : List of inclusion criterion strings
    exclusion       : List of exclusion criterion strings
    model           : LLM role for screening (default: 'screening' → Haiku)
    skip_dedup      : Skip deduplication (useful if data already deduped)
    skip_screen     : Only merge + dedup, no LLM call
    retry_uncertain : Re-screen only existing 'uncertain' records

    Returns
    -------
    dict with keys: included, excluded, uncertain, output_dir, files
    """
    project_dir   = Path(project_dir)
    artifacts_dir = project_artifacts_dir(project_dir)
    raw_dir       = project_raw_dir(project_dir)
    results_path  = artifacts_dir / "screening_results.csv"
    project_name  = project_dir.name

    # ── Retry mode ────────────────────────────────────────────────────────────
    if retry_uncertain:
        if not results_path.exists():
            raise FileNotFoundError("screening_results.csv not found. Run normally first.")
        df = pd.read_csv(results_path).fillna("")
        mask = df["decision"] == "uncertain"
        n    = mask.sum()
        if n == 0:
            print("No uncertain records. Nothing to retry.")
            return {}
        print(f"Retrying {n} uncertain records...")
        df_u     = df[mask].copy()
        df_u     = screen_records(df_u, inclusion, exclusion, model)
        df.update(df_u)
        df.to_csv(results_path, index=False)
        print("Updated screening_results.csv")
        return {}

    # ── Normal mode ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SR-PIPELINE — {project_name}")
    print(f"{'='*55}")

    print(f"\n[1/3] Merging CSVs from {raw_dir.name}/...")
    df      = merge_csvs(raw_dir)
    n_raw   = len(df)
    df.to_csv(artifacts_dir / "merged.csv", index=False)

    if not skip_dedup:
        print("\n[2/3] Deduplicating...")
        df, _, n_after_dedup = deduplicate(df)
        df.to_csv(artifacts_dir / "dedup.csv", index=False)
    else:
        n_after_dedup = n_raw
        print("\n[2/3] Dedup skipped.")

    if not skip_screen:
        print(f"\n[3/3] Screening {len(df)} records...")
        df = screen_records(df, inclusion, exclusion, model)
    else:
        print("\n[3/3] Screening skipped.")
        for col in ["decision", "confidence", "reason"]:
            df[col] = ""

    if "_row_id" in df.columns:
        df = df.drop(columns=["_row_id"])
    df.to_csv(results_path, index=False)

    report, n_inc, n_exc, n_unc = generate_prisma_report(
        project_name, n_raw, n_after_dedup, df
    )
    prisma_path = artifacts_dir / "prisma_report.md"
    prisma_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*55}")
    print("DONE")
    print(f"  Include  : {n_inc}")
    print(f"  Exclude  : {n_exc}")
    print(f"  Uncertain: {n_unc}")
    if n_unc:
        print(f"\n  Tip: re-run with retry_uncertain=True to reprocess uncertain records")
    print(f"{'='*55}\n")

    return {
        "included":  n_inc,
        "excluded":  n_exc,
        "uncertain": n_unc,
        "output_dir": str(artifacts_dir),
        "files": [str(results_path), str(prisma_path)],
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_criteria_file(path: str) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip().lstrip("-*•").strip()
            if cleaned and not cleaned.startswith("#"):
                lines.append(cleaned)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="SR-Pipeline: title/abstract screening via Claude"
    )
    parser.add_argument("--project-dir",      required=True,
                        help="Path to project directory (must contain raw/ folder with CSVs)")
    parser.add_argument("--inclusion",         required=False,
                        help="Path to inclusion criteria text file (one criterion per line)")
    parser.add_argument("--exclusion",         required=False,
                        help="Path to exclusion criteria text file (one criterion per line)")
    parser.add_argument("--model",             default="screening",
                        choices=list(__import__("srma.utils", fromlist=["ROLE_TO_MODEL"]).ROLE_TO_MODEL),
                        help="LLM role (default: screening → Haiku)")
    parser.add_argument("--skip-dedup",        action="store_true")
    parser.add_argument("--skip-screen",       action="store_true")
    parser.add_argument("--retry-uncertain",   action="store_true")
    args = parser.parse_args()

    inclusion = _parse_criteria_file(args.inclusion) if args.inclusion else []
    exclusion = _parse_criteria_file(args.exclusion) if args.exclusion else []

    if not args.skip_screen and (not inclusion or not exclusion):
        parser.error(
            "--inclusion and --exclusion are required unless --skip-screen is set"
        )

    run_pipeline(
        project_dir     = args.project_dir,
        inclusion       = inclusion,
        exclusion       = exclusion,
        model           = args.model,
        skip_dedup      = args.skip_dedup,
        skip_screen     = args.skip_screen,
        retry_uncertain = args.retry_uncertain,
    )


if __name__ == "__main__":
    main()
