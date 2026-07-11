"""
sr-pipeline — Systematic Review & Meta-Analysis Automation Toolkit
==================================================================
A Python toolkit for automating the systematic review and meta-analysis
workflow: deduplication, AI-powered screening, full-text structured
extraction, R-based statistical analysis, and reference management.

Author : Tuyen Tran (tuyentran-md)
License: MIT
"""

__version__ = "0.2.0"
__author__  = "Tuyen Tran"
__email__   = "tuyen.tran97@gmail.com"

from srma.screening import (
    deduplicate,
    screen_records,
    apply_human_decisions,
    finalize_human_screening,
    generate_prisma_report,
)
from srma.extraction import (
    extract_record,
    extract_records,
    apply_human_extraction,
    build_extraction_prompt,
    assert_no_ground_truth,
    load_schema_from_yaml,
    extraction_summary,
)
from srma.utils import call_llm, normalize_doi, normalize_title

__all__ = [
    "deduplicate",
    "screen_records",
    "apply_human_decisions",
    "finalize_human_screening",
    "generate_prisma_report",
    "extract_record",
    "extract_records",
    "apply_human_extraction",
    "build_extraction_prompt",
    "assert_no_ground_truth",
    "load_schema_from_yaml",
    "extraction_summary",
    "call_llm",
    "normalize_doi",
    "normalize_title",
]
