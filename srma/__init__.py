"""
sr-pipeline — Systematic Review & Meta-Analysis Automation Toolkit
==================================================================
A Python toolkit for automating the systematic review and meta-analysis
workflow: deduplication, AI-powered screening, data extraction,
R-based statistical analysis, and reference management.

Author : Tuyen Tran (tuyentran-md)
License: MIT
"""

__version__ = "0.1.0"
__author__  = "Tuyen Tran"
__email__   = "tuyen.tran97@gmail.com"

from srma.screening import deduplicate, screen_records, generate_prisma_report
from srma.utils import call_llm, normalize_doi, normalize_title

__all__ = [
    "deduplicate",
    "screen_records",
    "generate_prisma_report",
    "call_llm",
    "normalize_doi",
    "normalize_title",
]
