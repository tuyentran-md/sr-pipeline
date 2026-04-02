# sr-pipeline

> AI-powered systematic review and meta-analysis automation toolkit for clinical researchers.

[![Tests](https://github.com/tuyentran-md/sr-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/tuyentran-md/sr-pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

Doing a systematic review means spending days on deduplication and title/abstract screening before you even touch the science. This toolkit automates those steps using Claude while keeping you in control of the decisions that matter.

Built from real SR/MA work in pediatric surgery. Tested on 500+ records across multiple projects.

## What it does

```
Database exports (PubMed / Scopus / Embase)
        │
        ▼
  1. merge_csvs()          → combine multiple exports into one DataFrame
  2. deduplicate()         → DOI-exact + title-fuzzy match (SequenceMatcher ≥ 0.90)
  3. screen_records()      → batch LLM screening against your PICO criteria
  4. generate_prisma()     → PRISMA 2020-compliant flow report
        │
        ▼
   artifacts/
     screening_results.csv    → all records with decision / confidence / reason
     prisma_report.md         → flow numbers ready to paste into your paper
     dedup.csv                → post-deduplication records
```

The screener uses Claude Haiku by default (fast, cheap — ~$0.03 per 100 records). Uncertain records can be re-run with Sonnet for a second opinion.

## Quickstart

```bash
pip install sr-pipeline
export ANTHROPIC_API_KEY=sk-ant-...
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

# results = {"included": 42, "excluded": 187, "uncertain": 8, "output_dir": "..."}
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

## Handling uncertain records

The screener marks records as `uncertain` when the abstract is too short to judge or the criteria fit is ambiguous. Review these manually or retry with a stronger model:

```python
# Retry uncertain records with Sonnet
run_pipeline(
    project_dir     = "./my_review",
    inclusion       = INCLUSION_CRITERIA,
    exclusion       = EXCLUSION_CRITERIA,
    model           = "extraction",   # → Claude Sonnet
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
# df now has: decision, confidence, reason columns
```

Decision values: `"include"` | `"exclude"` | `"uncertain"`

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
| `"screening"` | Claude Haiku | High-volume title/abstract screening |
| `"extraction"` | Claude Sonnet | Data extraction, uncertain records |
| `"drafting"` | Claude Sonnet | Results section drafting |
| `"polishing"` | Claude Sonnet | Manuscript polish |

Override: `screen_records(df, ..., model="extraction")`

## Running tests

```bash
git clone https://github.com/tuyentran-md/sr-pipeline
cd sr-pipeline
pip install -e ".[dev]"
pytest
```

68 tests, no API calls required. Tests use mocked LLM responses.

## Roadmap

- [ ] Full-text PDF extraction (`srma.extraction`)
- [ ] R analysis script generator (`srma.r_analysis`)
- [ ] Reference verification via CrossRef API (`srma.references`)
- [ ] PROSPERO protocol outline generator (`srma.outline`)
- [ ] Network meta-analysis support (`srma.nma`)

## Background

This repo grew out of a real systematic review on outcomes after anorectal malformation repair ([E1_ARM project](https://aiforacademic.world)). The deduplication and screening logic has been validated against manual screening on ~500 records.

More on the methodology and how AI fits into evidence synthesis: [aiforacademic.world](https://aiforacademic.world)

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Tuyen Tran](https://github.com/tuyentran-md) — pediatric surgeon & clinical researcher.*
