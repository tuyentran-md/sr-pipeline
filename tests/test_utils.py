"""
tests/test_utils.py — Unit tests for srma.utils

No API calls — all tests are pure logic.
"""

import pytest
from srma.utils import (
    normalize_doi,
    normalize_title,
    safe_parse_json,
    safe_parse_json_object,
    ensure_dir,
)


# ─── normalize_doi ────────────────────────────────────────────────────────────

class TestNormalizeDoi:
    def test_strips_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_dx_prefix(self):
        assert normalize_doi("https://dx.doi.org/10.1234/abc") == "10.1234/abc"

    def test_lowercases(self):
        assert normalize_doi("10.1234/ABC") == "10.1234/abc"

    def test_strips_whitespace(self):
        assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"

    def test_empty_string(self):
        assert normalize_doi("") == ""

    def test_none_equivalent(self):
        # Passing None-like empty values
        assert normalize_doi("   ") == ""

    def test_plain_doi_unchanged(self):
        assert normalize_doi("10.1002/jps.12345") == "10.1002/jps.12345"

    def test_doi_with_slash_preserved(self):
        doi = "10.1016/j.jpedsurg.2021.01.001"
        assert normalize_doi(doi) == doi


# ─── normalize_title ─────────────────────────────────────────────────────────

class TestNormalizeTitle:
    def test_lowercases(self):
        assert "effect" in normalize_title("Effect of Surgery")

    def test_strips_punctuation(self):
        result = normalize_title("Meta-Analysis: A Review")
        assert ":" not in result
        assert "-" not in result

    def test_collapses_whitespace(self):
        result = normalize_title("A  study   of  outcomes")
        assert "  " not in result

    def test_strips_leading_trailing(self):
        assert normalize_title("  title  ") == "title"

    def test_empty_string(self):
        assert normalize_title("") == ""

    def test_identical_titles_match(self):
        t1 = normalize_title("Laparoscopic repair of anorectal malformations")
        t2 = normalize_title("Laparoscopic repair of anorectal malformations")
        assert t1 == t2

    def test_case_insensitive_match(self):
        t1 = normalize_title("Laparoscopic Repair of ARM")
        t2 = normalize_title("laparoscopic repair of arm")
        assert t1 == t2


# ─── safe_parse_json ─────────────────────────────────────────────────────────

class TestSafeParseJson:
    def test_clean_array(self):
        result = safe_parse_json('[{"id": "1", "decision": "include"}]')
        assert result == [{"id": "1", "decision": "include"}]

    def test_array_with_surrounding_text(self):
        result = safe_parse_json('Sure, here is the result:\n[{"id":"1"}]\nDone.')
        assert result == [{"id": "1"}]

    def test_empty_array(self):
        result = safe_parse_json("[]")
        assert result == []

    def test_no_json_returns_none(self):
        assert safe_parse_json("No JSON here") is None

    def test_malformed_json_returns_none(self):
        assert safe_parse_json('[{"id": "1",}]') is None

    def test_multiple_records(self):
        raw = '[{"id":"1","decision":"include"},{"id":"2","decision":"exclude"}]'
        result = safe_parse_json(raw)
        assert len(result) == 2
        assert result[0]["decision"] == "include"
        assert result[1]["decision"] == "exclude"

    def test_nested_objects(self):
        raw = '[{"id":"1","meta":{"conf":0.9}}]'
        result = safe_parse_json(raw)
        assert result[0]["meta"]["conf"] == 0.9


# ─── safe_parse_json_object ──────────────────────────────────────────────────

class TestSafeParseJsonObject:
    def test_clean_object(self):
        result = safe_parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_with_surrounding_text(self):
        result = safe_parse_json_object('Result: {"a": 1} done')
        assert result == {"a": 1}

    def test_no_object_returns_none(self):
        assert safe_parse_json_object("no json") is None


# ─── ensure_dir ──────────────────────────────────────────────────────────────

class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "new_dir"
        result = ensure_dir(target)
        assert result.is_dir()

    def test_creates_nested_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        ensure_dir(target)
        assert target.is_dir()

    def test_existing_directory_ok(self, tmp_path):
        # Should not raise if directory already exists
        ensure_dir(tmp_path)
        assert tmp_path.is_dir()
