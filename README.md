# sr-pipeline

> Safe, hybrid AI pipeline for systematic reviews and meta-analyses. Automation with human-in-the-loop.

[![Tests](https://github.com/tuyentran-md/sr-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/tuyentran-md/sr-pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

Doing a systematic review means spending days on deduplication and title/abstract screening before you even touch the science. This toolkit automates those steps using Google Gemini while keeping you in control of the decisions that matter.

Built from real SR/MA work in pediatric surgery. Tested on 500+ records across multiple projects. This is not "end-to-end automation." This is **acceleration with human-in-the-loop**, explicitly designed to prevent AI hallucinations from corrupting your data extraction and meta-analysis.

## What it does

```
Database exports (PubMed / Scopus / Embase)
        │
        ▼
  1. merge_csvs()          → combine multiple exports into one DataFrame
  2. deduplicate()         → DOI-exact + title-fuzzy match (SequenceMatcher ≥ 0.90)
  3. screen_records()      → provisional LLM screening against your PICO criteria
  4. human adjudication    → explicit include/exclude confirmation
  5. extract_records()     → provisional full-text codes with evidence quotes
        │
        ▼
   artifacts/
     screening_results.csv    → AI suggestion + human/final decision columns
     prisma_report.md         → confirmed counts; pending rows stay pending
     dedup.csv                → post-deduplication records
```

The screener uses Google Gemini Flash by default (fast, cheap). Uncertain records can be re-run for a second opinion.

## Quickstart

```bash
pip install "git+https://github.com/tuyentran-md/sr-pipeline.git"
export GEMINI_API_KEY=...  # GOOGLE_AI_API_KEY also works; gemini.env/.env supported
```

```python
from srma.screening import run_pipeline

results = run_pipeline(
    project_dir = "./my_review",    # must contain raw/ folder with exported CSVs
    inclusion = [
        "Original clinical study (RCT, cohort, or case series)",
        "Pediatric patients aged 0–18 years",
        "Diagnosis of anorectal malformation confirmed",
        "Reports at least one functional outcome",
    ],
    exclusion = [
        "Animal or in vitro studies",
        "Case reports (n < 5)",
        "Review articles, editorials, or conference abstracts",
        "Non-English publications",
    ],
)

# Before human adjudication, AI suggestions remain pending in final_decision.
```

Or via CLI:

```bash
srma --project-dir ./my_review \
     --inclusion inclusion_criteria.txt \
     --exclusion exclusion_criteria.txt
```

## Project layout

```
my_review/
  raw/
    pubmed_export.csv        ← PubMed CSV export
    scopus_export.csv        ← Scopus CSV export
    embase_zotero.csv        ← Embase via Zotero CSV
  artifacts/                 ← auto-created by sr-pipeline
    merged.csv
    dedup.csv
    screening_results.csv
    prisma_report.md
```

**Export format**: Zotero CSV export is recommended (works for PubMed, Scopus, Embase). Direct PubMed CSV also works.

## Fetching legal open-access PDFs

The downloader checks PMC Open Access first, then Unpaywall, validates PDF bytes,
and writes unresolved papers to `missing.tsv` for manual retrieval:

```bash
python -m srma.download \
  --input papers.tsv \
  --outdir pdfs/ \
  --email researcher@example.com
```

`papers.tsv` uses the columns `pmid`, `doi`, `title`, and `pmc`. A DOI or PMC ID
is sufficient when a PMID is unavailable.

## Required human adjudication

`decision`, `confidence`, and `reason` are AI suggestions. Every row starts with
`final_decision="pending"` and `needs_review=True`. Enter `include`, `exclude`,
or `uncertain` in `human_decision`, then apply the explicit human decisions and
refresh the PRISMA report without an API call:

```bash
srma --project-dir ./my_review --finalize-human
```

AI-uncertain records can be re-run with the stronger extraction role before the
human decision:

```python
# Retry uncertain records
run_pipeline(
    project_dir     = "./my_review",
    inclusion       = INCLUSION_CRITERIA,
    exclusion       = EXCLUSION_CRITERIA,
    model           = "extraction",   # → Gemini Flash
    retry_uncertain = True,
)
```

## API reference

### `deduplicate(df, title_threshold=0.90)`

Remove duplicates from a DataFrame of citations.

```python
from srma.screening import deduplicate
clean_df, n_before, n_after = deduplicate(df)
```

| Parameter | Default | Description |
|---|---|---|
| `df` | — | DataFrame with `Title` and `DOI` columns |
| `title_threshold` | `0.90` | Fuzzy match threshold for title deduplication |

Returns `(cleaned_df, n_before, n_after)`.

### `screen_records(df, inclusion, exclusion, model="screening")`

Screen a DataFrame against eligibility criteria via LLM.

```python
from srma.screening import screen_records
df = screen_records(df, inclusion=["..."], exclusion=["..."])
# AI: decision, confidence, reason
# Human gate: human_decision, final_decision, needs_review
```

Decision values: `"include"` | `"exclude"` | `"uncertain"`

### `extract_records(df, schema, text_col="full_text")`

Generate provisional structured codes from full text. Codes outside the schema,
missing evidence, and evidence quotes absent from the source are rejected or
flagged. Every candidate still has `review_status="pending"` and an empty
`human_code` until a person verifies it.

```python
from srma.extraction import (
    load_schema_from_yaml,
    extract_records,
    apply_human_extraction,
)

schema = load_schema_from_yaml("coding_schema.yaml")
coded = extract_records(df, schema, out_path="extraction_results.csv")
# After reviewers fill each <dimension>_human_code column:
coded = apply_human_extraction(coded, schema)
```

From a checkout, install YAML support with `pip install -e ".[yaml]"`, or use a
Python dict and the core install.

### `generate_prisma_report(project_name, n_raw, n_after_dedup, df)`

Generate a PRISMA 2020 flow report string.

```python
from srma.screening import generate_prisma_report
report, n_inc, n_exc, n_unc = generate_prisma_report("MY_PROJECT", 500, 420, df)
```

### `normalize_doi(doi)` / `normalize_title(title)`

Text normalization helpers used internally — useful for custom deduplication logic.

```python
from srma.utils import normalize_doi, normalize_title
normalize_doi("https://doi.org/10.1234/abc")  # → "10.1234/abc"
normalize_title("Effect of Surgery: A Review")  # → "effect of surgery a review"
```

## Model selection

| Role key | Default model | Best for |
|---|---|---|
| `"screening"` | Gemini 3.5 Flash (stable) | High-volume title/abstract screening |
| `"extraction"` | Gemini 3.1 Pro (preview) | Data extraction, uncertain records |
| `"drafting"` | Gemini 3.1 Pro (preview) | Results section drafting |
| `"polishing"` | Gemini 3.1 Pro (preview) | Manuscript polish |

Override: `screen_records(df, ..., model="extraction")`

## Running tests

```bash
git clone https://github.com/tuyentran-md/sr-pipeline
cd sr-pipeline
pip install -e ".[dev]"
pytest
```

110 tests, no API calls required. Tests use mocked LLM and download responses.

## Roadmap

- [x] Structured full-text extraction candidates with mandatory human confirmation (`srma.extraction`)
- [ ] R analysis script generator (`srma.r_analysis`)
- [ ] Reference verification via CrossRef API (`srma.references`)
- [ ] PROSPERO protocol outline generator (`srma.outline`)
- [ ] Network meta-analysis support (`srma.nma`)

## Background

This repo grew out of a real systematic review on outcomes after anorectal malformation repair ([E1_ARM project](https://aiforacademic.world)). The deduplication and screening logic has been validated against manual screening on ~500 records. Our core belief: AI should map and screen, but humans must extract and interpret.

Read the full methodology on how to use AI for systematic reviews safely: [How to Use AI for Systematic Reviews Without Compromising Rigor](https://aiforacademic.world/blog/how-to-use-ai-for-systematic-reviews-without-compromising-rigor)

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Tuyen Tran](https://github.com/tuyentran-md) — pediatric surgeon & clinical researcher.*
