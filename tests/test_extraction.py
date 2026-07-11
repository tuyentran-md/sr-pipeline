"""
tests/test_extraction.py — Unit tests for srma.extraction

Covers all non-LLM logic + the human-in-the-loop discipline guards.
LLM calls are monkeypatched; no API calls are made.
"""

import pandas as pd
import pytest

import srma.extraction as ex
from srma.extraction import (
    assert_no_ground_truth,
    build_extraction_prompt,
    extract_record,
    extract_records,
    apply_human_extraction,
    extraction_summary,
)

SCHEMA = {
    "name": "test schema",
    "dimensions": [
        {"key": "D1", "label": "first",
         "anchors": {"0": "absent", "1": "partial", "2": "full"},
         "rule": "RULE TEXT D1"},
        {"key": "D2", "label": "second",
         "anchors": {"0": "no", "2": "yes"}},
    ],
}


# ─── ground-truth isolation guard ─────────────────────────────────────────────

class TestGroundTruthGuard:
    def test_clean_schema_passes(self):
        assert_no_ground_truth(SCHEMA)  # no raise

    @pytest.mark.parametrize("leaked", ["gold", "answer", "expected",
                                        "consensus", "ground_truth", "correct_code"])
    def test_leaked_field_blocked(self, leaked):
        bad = {"dimensions": [{"key": "D1", leaked: 2, "anchors": {}}]}
        with pytest.raises(ValueError):
            assert_no_ground_truth(bad)

    def test_extract_records_blocks_leaked_schema(self):
        bad = {"name": "x", "dimensions": [{"key": "D1", "gold": 1, "anchors": {}}]}
        with pytest.raises(ValueError):
            extract_records(pd.DataFrame({"full_text": ["t"]}), bad)

    def test_nested_leak_and_duplicate_keys_are_blocked(self):
        nested = {"dimensions": [{"key": "D1", "anchors": {"0": "no"},
                                   "meta": {"answer_key": 0}}]}
        with pytest.raises(ValueError):
            assert_no_ground_truth(nested)
        duplicate = {"dimensions": [
            {"key": "D1", "anchors": {"0": "no"}},
            {"key": "D1", "anchors": {"0": "no"}},
        ]}
        with pytest.raises(ValueError):
            assert_no_ground_truth(duplicate)


# ─── prompt construction ──────────────────────────────────────────────────────

class TestPromptBuild:
    def test_contains_keys_anchors_rule_and_text(self):
        p = build_extraction_prompt("THE STUDY TEXT", SCHEMA)
        assert "D1" in p and "D2" in p
        assert "RULE TEXT D1" in p
        assert "THE STUDY TEXT" in p
        assert "JSON object" in p

    def test_text_is_stripped(self):
        p = build_extraction_prompt("   padded   ", SCHEMA)
        assert "padded" in p


# ─── extraction discipline ────────────────────────────────────────────────────

def _patch(monkeypatch, response):
    monkeypatch.setattr(ex, "call_llm", lambda prompt, **kw: response)


class TestExtractRecord:
    def test_clean_parse(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":1,"confidence":0.9,"evidence":"q","needs_review":false},'
               '"D2":{"code":0,"confidence":0.9,"evidence":"","needs_review":false}}')
        rec = extract_record("text containing q", SCHEMA)
        assert rec["D1_code"] == "1" and rec["D2_code"] == "0"
        assert rec["_extraction_ok"] is True
        assert rec["D1_needs_review"] is True
        assert rec["D1_review_status"] == "pending"
        assert rec["D1_human_code"] == ""

    def test_positive_code_without_quote_flagged(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":2,"confidence":0.99,"evidence":"","needs_review":false},'
               '"D2":{"code":0,"confidence":0.9,"evidence":"","needs_review":false}}')
        rec = extract_record("text", SCHEMA)
        assert rec["D1_needs_review"] is True   # no quote -> review
        assert rec["D2_needs_review"] is True   # all AI codes remain provisional
        assert rec["_extraction_ok"] is False

    def test_invalid_code_and_fabricated_quote_are_rejected(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":9,"confidence":0.99,"evidence":"fabricated",'
               '"needs_review":false},"D2":{"code":0,"confidence":0.9,'
               '"evidence":"","needs_review":false}}')
        rec = extract_record("actual source text", SCHEMA)
        assert rec["D1_code"] == ""
        assert rec["_extraction_ok"] is False
        assert rec["D1_needs_review"] is True

    def test_low_confidence_flagged(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":2,"confidence":0.3,"evidence":"q","needs_review":false},'
               '"D2":{"code":0,"confidence":0.9,"evidence":"","needs_review":false}}')
        rec = extract_record("text", SCHEMA)
        assert rec["D1_needs_review"] is True

    def test_parse_failure_is_safe(self, monkeypatch):
        _patch(monkeypatch, "I cannot produce JSON")
        rec = extract_record("text", SCHEMA)
        assert rec["_extraction_ok"] is False
        assert rec["D1_needs_review"] is True and rec["D2_needs_review"] is True

    def test_llm_exception_is_safe(self, monkeypatch):
        def boom(prompt, **kw):
            raise RuntimeError("api down")
        monkeypatch.setattr(ex, "call_llm", boom)
        rec = extract_record("text", SCHEMA)
        assert rec["_extraction_ok"] is False

    def test_confidence_clamped(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":1,"confidence":5,"evidence":"q","needs_review":false},'
               '"D2":{"code":0,"confidence":-1,"evidence":"","needs_review":false}}')
        rec = extract_record("text", SCHEMA)
        assert rec["D1_confidence"] == 1.0
        assert rec["D2_confidence"] == 0.0

    def test_non_finite_confidence_becomes_zero(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":0,"confidence":"NaN","evidence":""},'
               '"D2":{"code":0,"confidence":"Infinity","evidence":""}}')
        rec = extract_record("text", SCHEMA)
        assert rec["D1_confidence"] == 0.0
        assert rec["D2_confidence"] == 0.0


# ─── batch extraction ─────────────────────────────────────────────────────────

class TestExtractRecords:
    def test_missing_text_col_raises(self):
        with pytest.raises(KeyError):
            extract_records(pd.DataFrame({"x": [1]}), SCHEMA, text_col="full_text")

    def test_empty_text_flagged_not_dropped(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":0,"confidence":0.9,"evidence":"","needs_review":false},'
               '"D2":{"code":0,"confidence":0.9,"evidence":"","needs_review":false}}')
        df = pd.DataFrame({"full_text": ["real text", ""]})
        out = extract_records(df, SCHEMA)
        assert len(out) == 2                       # nothing dropped
        assert not out.iloc[1]["_extraction_ok"]   # empty row flagged (numpy bool)
        assert "_n_needs_review" in out.columns

    def test_columns_added_for_each_dimension(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":1,"confidence":0.9,"evidence":"q","needs_review":false},'
               '"D2":{"code":2,"confidence":0.9,"evidence":"q","needs_review":false}}')
        out = extract_records(pd.DataFrame({"full_text": ["t"]}), SCHEMA)
        for k in ("D1", "D2"):
            for suffix in ("_code", "_confidence", "_evidence", "_needs_review"):
                assert f"{k}{suffix}" in out.columns


class TestExtractionSummary:
    def test_summary_counts(self, monkeypatch):
        _patch(monkeypatch,
               '{"D1":{"code":2,"confidence":0.9,"evidence":"q","needs_review":false},'
               '"D2":{"code":0,"confidence":0.9,"evidence":"","needs_review":false}}')
        out = extract_records(pd.DataFrame({"full_text": ["q a", "q b"]}), SCHEMA)
        out["D1_human_code"] = ["2", "2"]
        out["D1_review_status"] = ["confirmed", "confirmed"]
        summ = extraction_summary(out, SCHEMA)
        assert "D1" in summ and "confirmed code 2: 2" in summ
        assert "pending human review: 0" in summ


class TestHumanExtraction:
    def test_only_valid_human_codes_clear_review(self):
        df = pd.DataFrame({"D1_human_code": ["2", ""], "D2_human_code": ["0", "2"]})
        out = apply_human_extraction(df, SCHEMA)
        assert out["D1_review_status"].tolist() == ["confirmed", "pending"]
        assert out["D1_needs_review"].tolist() == [False, True]

    def test_invalid_human_code_is_rejected(self):
        with pytest.raises(ValueError):
            apply_human_extraction(pd.DataFrame({"D1_human_code": ["9"]}), SCHEMA)
