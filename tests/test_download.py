"""Regression tests for legal OA PDF retrieval and cache validation."""

import io
import tarfile
from types import SimpleNamespace

import srma.download as dl


PDF = b"%PDF-1.7\n" + b"0" * 200


def test_valid_pdf_cache_is_reused(tmp_path):
    dest = tmp_path / "paper.pdf"
    dest.write_bytes(PDF)
    assert dl.fetch_paper("1", "", "", dest, "a@example.com") == "cached"


def test_html_cache_is_deleted_not_treated_as_pdf(tmp_path, monkeypatch):
    dest = tmp_path / "paper.pdf"
    dest.write_bytes(b"<html>" + b"x" * 2000)
    monkeypatch.setattr(dl, "_pmc_id_from_pmid", lambda pmid: None)
    monkeypatch.setattr(dl, "_download_unpaywall", lambda *args: False)
    assert dl.fetch_paper("1", "", "", dest, "a@example.com") == "failed"
    assert not dest.exists()


def test_pmc_tgz_package_extracts_pdf(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        info = tarfile.TarInfo("article/main.pdf")
        info.size = len(PDF)
        tf.addfile(info, io.BytesIO(PDF))

    monkeypatch.setattr(
        dl, "_get",
        lambda url, **kwargs: SimpleNamespace(
            text='<link format="tgz" href="ftp://example/article.tar.gz"/>'
        ),
    )
    monkeypatch.setattr(dl, "_download_bytes", lambda url: archive.getvalue())
    dest = tmp_path / "paper.pdf"
    assert dl._download_pmc_oa("PMC123", dest) is True
    assert dest.read_bytes() == PDF


def test_unpaywall_url_encodes_doi(tmp_path, monkeypatch):
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if "api.unpaywall.org" in url:
            return SimpleNamespace(json=lambda: {
                "best_oa_location": {"url_for_pdf": "https://example/p.pdf"}
            })
        return SimpleNamespace(content=PDF)

    monkeypatch.setattr(dl, "_get", fake_get)
    assert dl._download_unpaywall("10.1/a b", tmp_path / "paper.pdf", "a@example.com")
    assert "10.1%2Fa%20b" in seen[0]


def test_pmc_only_rows_get_distinct_stems():
    assert dl._paper_stem("", "", "PMC123") == "PMC123"
    assert dl._paper_stem("", "", "PMC456") == "PMC456"
