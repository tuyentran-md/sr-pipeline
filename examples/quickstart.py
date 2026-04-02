"""
examples/quickstart.py — sr-pipeline quickstart

Prerequisites
-------------
1. Set your API key:
   export ANTHROPIC_API_KEY=sk-ant-...

2. Create a project directory with exported CSVs:
   my_review/
     raw/
       pubmed_export.csv      ← Export from PubMed (CSV format)
       scopus_export.csv      ← Export from Scopus (CSV format)
       embase_export.csv      ← Export from Embase via Zotero (CSV)

3. Run this script:
   python examples/quickstart.py

What it does
------------
- Merges all CSVs
- Removes duplicates (DOI-exact + title-fuzzy)
- Screens title/abstract via Claude against your criteria
- Outputs:
    artifacts/screening_results.csv  ← All records with decision/confidence/reason
    artifacts/prisma_report.md       ← PRISMA 2020 flow summary
    artifacts/dedup.csv              ← Post-dedup records
    artifacts/merged.csv             ← Raw merged records
"""

import os
from srma.screening import run_pipeline

# ── Configure your review ────────────────────────────────────────────────────

PROJECT_DIR = "./my_review"   # Change to your project path

INCLUSION_CRITERIA = [
    "Original clinical study (RCT, cohort, case series, or cross-sectional)",
    "Pediatric patients aged 0–18 years",
    "Diagnosis of anorectal malformation (ARM) confirmed",
    "Reports at least one functional outcome (fecal continence, urinary function, or QoL)",
]

EXCLUSION_CRITERIA = [
    "Animal or in vitro studies",
    "Case reports (n < 5)",
    "Review articles, editorials, letters, or conference abstracts without full data",
    "Non-English language publications",
    "Insufficient outcome data to extract",
]

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)

    results = run_pipeline(
        project_dir = PROJECT_DIR,
        inclusion   = INCLUSION_CRITERIA,
        exclusion   = EXCLUSION_CRITERIA,
        model       = "screening",     # Uses Claude Haiku (fast + cheap)
    )

    print("\n--- Summary ---")
    print(f"Included  : {results['included']}")
    print(f"Excluded  : {results['excluded']}")
    print(f"Uncertain : {results['uncertain']}")
    print(f"\nOutput: {results['output_dir']}")

    # If there are uncertain records, retry with a stronger model
    if results.get("uncertain", 0) > 0:
        print("\nRetrying uncertain records with Sonnet...")
        run_pipeline(
            project_dir     = PROJECT_DIR,
            inclusion       = INCLUSION_CRITERIA,
            exclusion       = EXCLUSION_CRITERIA,
            model           = "extraction",    # Uses Claude Sonnet
            retry_uncertain = True,
        )
