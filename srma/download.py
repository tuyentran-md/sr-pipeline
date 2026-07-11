"""
srma/download.py — PDF fetcher for pilot and main extraction runs.

Priority chain per paper:
  1. PMC OA API → FTP direct PDF (free, reliable for OA papers)
  2. Unpaywall (free legal OA PDF, needs email)
  3. FAIL → log to missing.tsv for manual download

Usage
-----
    python -m srma.download --input papers.tsv --outdir pdfs/ --email your@email.com
    # papers.tsv: tab-separated, columns: pmid  doi  title  pmc  (header row required)

Output
------
    pdfs/<pmid>.pdf          — downloaded PDFs
    pdfs/missing.tsv         — PMID, DOI, title for papers that failed all sources
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import tarfile
import time
import urllib.request
from urllib.parse import quote
from pathlib import Path

import requests

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
HEADERS = {"User-Agent": "sr-pipeline/0.2 (research; contact via github)"}


def _is_pdf(content: bytes) -> bool:
    return len(content) > 100 and b"%PDF-" in content[:1024]


def _download_bytes(url: str) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except Exception:
        return None


def _paper_stem(pmid: str, doi: str, pmc: str) -> str:
    if pmid:
        return pmid
    if doi:
        return f"doi_{hashlib.sha256(doi.encode()).hexdigest()[:12]}"
    return re.sub(r"[^A-Za-z0-9_-]", "", pmc) or "pmc_unknown"


def _get(url: str, **kwargs) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


def _pmc_id_from_pmid(pmid: str) -> str | None:
    """Look up PMC ID for a given PMID via E-link."""
    url = f"{NCBI_EUTILS}/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
    r = _get(url)
    if not r:
        return None
    try:
        data = r.json()
        for lb in data["linksets"][0]["linksetdbs"]:
            if lb.get("linkname") == "pubmed_pmc":
                ids = lb.get("links", [])
                if ids:
                    return str(ids[0])
    except Exception:
        pass
    return None


def _download_pmc_oa(pmcid: str, dest: Path) -> bool:
    """Use PMC OA API to get FTP path, then download via urllib (FTP supported)."""
    pmc_str = f"PMC{pmcid}" if not pmcid.startswith("PMC") else pmcid
    url = f"{PMC_OA_API}?id={pmc_str}"
    r = _get(url)
    if not r:
        return False
    # Parse XML for pdf link
    pdf_match = re.search(r'format="pdf"[^>]*href="([^"]+)"', r.text)
    if pdf_match:
        content = _download_bytes(pdf_match.group(1))
        if content and _is_pdf(content):
            dest.write_bytes(content)
            return True

    # PMC commonly provides a .tar.gz package rather than a direct PDF.
    tgz_match = re.search(r'format="tgz"[^>]*href="([^"]+)"', r.text)
    if tgz_match:
        archive = _download_bytes(tgz_match.group(1))
        if archive:
            try:
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
                    for member in tf.getmembers():
                        if member.isfile() and member.name.lower().endswith(".pdf"):
                            extracted = tf.extractfile(member)
                            content = extracted.read() if extracted else b""
                            if _is_pdf(content):
                                dest.write_bytes(content)
                                return True
            except (tarfile.TarError, OSError):
                pass
    dest.unlink(missing_ok=True)
    return False


def _download_unpaywall(doi: str, dest: Path, email: str) -> bool:
    if not doi:
        return False
    r = _get(f"{UNPAYWALL_BASE}/{quote(doi, safe='')}?email={quote(email, safe='@')}")
    if not r:
        return False
    try:
        data = r.json()
        pdf_url = None
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    break
        if pdf_url:
            r2 = _get(pdf_url)
            if r2 and _is_pdf(r2.content):
                dest.write_bytes(r2.content)
                return True
    except Exception:
        pass
    return False


def fetch_paper(pmid: str, doi: str, pmc: str, dest: Path, email: str) -> str:
    """Return source string or 'failed'."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with dest.open("rb") as cached:
            header = cached.read(1024)
        if dest.stat().st_size > 100 and _is_pdf(header):
            return "cached"
        dest.unlink(missing_ok=True)

    # 1. PMC OA (use provided PMC ID or look it up)
    pmcid = pmc.replace("PMC", "") if pmc else None
    if not pmcid and pmid:
        pmcid = _pmc_id_from_pmid(pmid)
        time.sleep(0.35)
    if pmcid and _download_pmc_oa(pmcid, dest):
        return "pmc_oa"

    # 2. Unpaywall
    if _download_unpaywall(doi, dest, email):
        return "unpaywall"

    return "failed"


def main():
    parser = argparse.ArgumentParser(description="Download PDFs for sr-pipeline")
    parser.add_argument("--input", required=True, help="TSV with header: pmid\\tdoi\\ttitle\\tpmc")
    parser.add_argument("--outdir", default="pdfs")
    parser.add_argument("--email", required=True, help="Email for Unpaywall API")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    missing = []
    for row in rows:
        pmid = row.get("pmid", "").strip()
        doi = row.get("doi", "").strip()
        title = row.get("title", "").strip()
        pmc = row.get("pmc", "").strip()
        if not (pmid or doi or pmc):
            missing.append({"pmid": pmid, "doi": doi, "title": title})
            print(f"✗ (no identifier)  [failed      ]  {title[:60]}")
            continue
        stem = _paper_stem(pmid, doi, pmc)
        dest = outdir / f"{stem}.pdf"
        result = fetch_paper(pmid, doi, pmc, dest, args.email)
        icon = "✓" if result not in ("failed",) else "✗"
        print(f"{icon} {pmid}  [{result:<12}]  {title[:60]}")
        if result == "failed":
            missing.append({"pmid": pmid, "doi": doi, "title": title})

    missing_path = outdir / "missing.tsv"
    if missing:
        with open(missing_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pmid","doi","title"], delimiter="\t")
            w.writeheader()
            w.writerows(missing)
        print(f"\n{len(missing)} failed → {missing_path}")
    else:
        missing_path.unlink(missing_ok=True)
        print("\nAll downloaded.")


if __name__ == "__main__":
    main()
