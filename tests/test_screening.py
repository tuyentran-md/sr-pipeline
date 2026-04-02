"""
tests/test_screening.py — Unit tests for srma.screening

Covers all non-LLM logic. No API calls made.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from srma.screening import (
    merge_csvs,
    deduplicate,
    generate_prisma_report,
    screen_records,
    _build_screening_prompt,
    _norm_decision,
    _norm_confidence,
)

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# ─── merge_csvs ───────────────────────────────────────────────────────────────

class TestMergeCsvs:
    def test_loads_sample_fixture(self):
        df = merge_csvs(FIXTURES)
        assert len(df) == 10
        assert "Title" in df.columns
        assert "DOI" in df.columns

    def test_adds_source_file_column(self):
        df = merge_csvs(FIXTURES)
        assert "source_file" in df.columns
        assert df["source_file"].notna().all()

    def test_fills_na_with_empty_string(self):
        df = merge_csvs(FIXTURES)
        # No NaN values
        assert not df["Title"].isna().any()
        assert not df["DOI"].isna().any()

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_csvs(tmp_path)

    def test_skips_output_files(self, tmp_path):
        # Create a dummy source CSV and a screening_results.csv (should be skipped)
        pd.DataFrame({"Title": ["Test"], "DOI": ["10.1/a"]}).to_csv(
            tmp_path / "source.csv", index=False
        )
        pd.DataFrame({"Title": ["Result"], "DOI": ["10.2/b"]}).to_csv(
            tmp_path / "screening_results.csv", index=False
        )
        df = merge_csvs(tmp_path)
        assert len(df) == 1  # Only source.csv, not screening_results.csv


# ─── deduplicate ─────────────────────────────────────────────────────────────

class TestDeduplicate:
    def test_removes_exact_doi_duplicate(self):
        df = pd.DataFrame({
            "Title": ["Study A", "Study A copy"],
            "DOI":   ["10.1/a",  "10.1/a"],
        })
        clean, before, after = deduplicate(df)
        assert before == 2
        assert after == 1

    def test_removes_fuzzy_title_duplicate(self):
        df = pd.DataFrame({
            "Title": [
                "Laparoscopic repair of anorectal malformations in neonates",
                "Laparoscopic repair of anorectal malformations in neonates.",
            ],
            "DOI": ["", ""],  # No DOI → must match by title
        })
        clean, before, after = deduplicate(df)
        assert after == 1

    def test_keeps_distinct_records(self):
        df = pd.DataFrame({
            "Title": ["Study A", "Study B", "Study C"],
            "DOI":   ["10.1/a", "10.1/b", "10.1/c"],
        })
        clean, before, after = deduplicate(df)
        assert after == 3

    def test_handles_empty_doi(self):
        df = pd.DataFrame({
            "Title": ["Unique title one", "Unique title two"],
            "DOI":   ["", ""],
        })
        clean, _, after = deduplicate(df)
        assert after == 2

    def test_returns_correct_types(self):
        df = pd.DataFrame({"Title": ["A"], "DOI": ["10.1/a"]})
        clean, before, after = deduplicate(df)
        assert isinstance(clean, pd.DataFrame)
        assert isinstance(before, int)
        assert isinstance(after, int)

    def test_drops_internal_columns(self):
        df = pd.DataFrame({"Title": ["A", "A"], "DOI": ["10.1/a", "10.1/a"]})
        clean, _, _ = deduplicate(df)
        for col in ["_doi_norm", "_title_norm", "_duplicate"]:
            assert col not in clean.columns

    def test_fixture_has_one_duplicate(self):
        df = merge_csvs(FIXTURES)
        clean, before, after = deduplicate(df)
        assert before == 10
        assert after == 9  # ZK003 is duplicate of ZK001

    def test_custom_threshold_strict(self):
        # At threshold=0.99, near-identical titles should NOT be deduped
        df = pd.DataFrame({
            "Title": ["A study of ARM", "A study of ARM outcomes"],
            "DOI":   ["", ""],
        })
        clean, _, after = deduplicate(df, title_threshold=0.99)
        assert after == 2

    def test_custom_threshold_loose(self):
        # At threshold=0.5, loosely similar titles would dedup
        df = pd.DataFrame({
            "Title": ["ARM surgery", "ARM surgery outcomes data"],
            "DOI":   ["", ""],
        })
        clean, _, after = deduplicate(df, title_threshold=0.50)
        assert after == 1


# ─── _norm_decision / _norm_confidence ──────────────────────────────────────

class TestNormHelpers:
    def test_norm_decision_valid(self):
        assert _norm_decision("include")   == "include"
        assert _norm_decision("exclude")   == "exclude"
        assert _norm_decision("uncertain") == "uncertain"

    def test_norm_decision_case_insensitive(self):
        assert _norm_decision("INCLUDE") == "include"
        assert _norm_decision("Exclude") == "exclude"

    def test_norm_decision_invalid_fallback(self):
        assert _norm_decision("maybe")  == "uncertain"
        assert _norm_decision("")       == "uncertain"
        assert _norm_decision("yes")    == "uncertain"

    def test_norm_confidence_valid(self):
        assert _norm_confidence(0.8) == pytest.approx(0.8)
        assert _norm_confidence(0.0) == pytest.approx(0.0)
        assert _norm_confidence(1.0) == pytest.approx(1.0)

    def test_norm_confidence_clamps_above_one(self):
        assert _norm_confidence(1.5) == pytest.approx(1.0)

    def test_norm_confidence_clamps_below_zero(self):
        assert _norm_confidence(-0.5) == pytest.approx(0.0)

    def test_norm_confidence_invalid_string(self):
        assert _norm_confidence("high") == pytest.approx(0.0)

    def test_norm_confidence_none(self):
        assert _norm_confidence(None) == pytest.approx(0.0)


# ─── _build_screening_prompt ─────────────────────────────────────────────────

class TestBuildScreeningPrompt:
    def setup_method(self):
        self.batch = pd.DataFrame({
            "_row_id":       ["0", "1"],
            "Key":           ["ZK001", "ZK002"],
            "Title":         ["Study on ARM repair", "Outcomes after PSARP"],
            "Abstract Note": ["Methods: ...", "Results: ..."],
        })
        self.inclusion = ["Pediatric patients", "ARM diagnosis confirmed"]
        self.exclusion = ["Animal studies", "Case reports"]

    def test_prompt_contains_inclusion(self):
        prompt = _build_screening_prompt(self.batch, self.inclusion, self.exclusion)
        assert "Pediatric patients" in prompt
        assert "ARM diagnosis confirmed" in prompt

    def test_prompt_contains_exclusion(self):
        prompt = _build_screening_prompt(self.batch, self.inclusion, self.exclusion)
        assert "Animal studies" in prompt
        assert "Case reports" in prompt

    def test_prompt_contains_record_ids(self):
        prompt = _build_screening_prompt(self.batch, self.inclusion, self.exclusion)
        assert "ID: 0" in prompt
        assert "ID: 1" in prompt

    def test_prompt_contains_titles(self):
        prompt = _build_screening_prompt(self.batch, self.inclusion, self.exclusion)
        assert "Study on ARM repair" in prompt

    def test_prompt_requests_json(self):
        prompt = _build_screening_prompt(self.batch, self.inclusion, self.exclusion)
        assert "JSON" in prompt
        assert '"decision"' in prompt


# ─── generate_prisma_report ──────────────────────────────────────────────────

class TestGeneratePrismaReport:
    def setup_method(self):
        self.df = pd.DataFrame({
            "decision": ["include", "include", "exclude", "exclude",
                         "exclude", "uncertain", "include", "exclude",
                         "uncertain"],
        })

    def test_correct_counts(self):
        _, n_inc, n_exc, n_unc = generate_prisma_report(
            "TEST", 100, 90, self.df
        )
        assert n_inc == 3
        assert n_exc == 4
        assert n_unc == 2

    def test_report_contains_project_name(self):
        report, *_ = generate_prisma_report("MY_PROJECT", 50, 45, self.df)
        assert "MY_PROJECT" in report

    def test_report_contains_raw_count(self):
        report, *_ = generate_prisma_report("X", 50, 45, self.df)
        assert "50" in report

    def test_report_is_string(self):
        report, *_ = generate_prisma_report("X", 10, 9, self.df)
        assert isinstance(report, str)

    def test_report_mentions_next_steps(self):
        report, *_ = generate_prisma_report("X", 10, 9, self.df)
        assert "Next Step" in report or "Full-text" in report


# ─── screen_records (mocked LLM) ─────────────────────────────────────────────

class TestScreenRecordsMocked:
    """Test screen_records logic without real API calls."""

    def _make_df(self):
        return pd.DataFrame({
            "Title":         ["Study of ARM repair", "Rat gut study"],
            "Abstract Note": ["Methods: RCT in 50 children...", "Sprague-Dawley rats..."],
            "DOI":           ["10.1/a", "10.2/b"],
            "Key":           ["ZK001", "ZK002"],
        })

    def _mock_response(self):
        return '[{"id":"0","key":"ZK001","decision":"include","confidence":0.9,"reason":"Meets all criteria"},{"id":"1","key":"ZK002","decision":"exclude","confidence":0.95,"reason":"Animal study"}]'

    def test_adds_decision_column(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value=self._mock_response()):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert "decision" in result.columns

    def test_adds_confidence_column(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value=self._mock_response()):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert "confidence" in result.columns

    def test_adds_reason_column(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value=self._mock_response()):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert "reason" in result.columns

    def test_correct_decisions(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value=self._mock_response()):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert result.iloc[0]["decision"] == "include"
        assert result.iloc[1]["decision"] == "exclude"

    def test_api_failure_defaults_to_uncertain(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", side_effect=Exception("API error")):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        # All records should default to 'uncertain' on failure
        assert (result["decision"] == "uncertain").all()

    def test_bad_json_defaults_to_uncertain(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value="This is not JSON"):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert (result["decision"] == "uncertain").all()

    def test_preserves_original_columns(self):
        df = self._make_df()
        with patch("srma.screening.call_llm", return_value=self._mock_response()):
            result = screen_records(df, ["Pediatric"], ["Animals"])
        assert "Title" in result.columns
        assert "DOI" in result.columns
