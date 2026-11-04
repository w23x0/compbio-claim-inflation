"""Build a real M3 context snapshot from OpenAlex for the valid claims.

Writes data/raw/context/context.json.  Missing DOIs are left without a paper;
they become missing context fields, not fabricated values.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.m3.openalex import fetch_openalex_citations, fetch_openalex_work, normalize_openalex_work
from cbi.util import read_json, write_json


def main() -> int:
    claims = read_json(ROOT / "reports" / "P1_public_pmc_claims_valid.json")
    papers, citations, authors = [], [], {}
    seen_works = set()
    for claim in claims:
        if claim.get("endpoint") != "OS":
            continue
        doi = (claim.get("source") or {}).get("doi")
        if not doi:
            continue
        try:
            work = fetch_openalex_work(doi)
        except Exception as exc:
            print(f"skip {claim['claim_id']} {doi}: {type(exc).__name__}: {exc}")
            continue
        norm = normalize_openalex_work(work, claim_id=claim["claim_id"])
        if norm["paper_id"] in seen_works:
            continue
        seen_works.add(norm["paper_id"])
        papers.append(norm)
        for aid in norm.get("author_ids", []):
            authors.setdefault(aid, {"id": aid})
        try:
            cites = fetch_openalex_citations(norm["paper_id"], per_page=200)
        except Exception as exc:
            print(f"citations skip {norm['paper_id']}: {type(exc).__name__}")
            cites = []
        for cit in cites:
            cid = str(cit.get("id", "")).rsplit("/", 1)[-1]
            pub = cit.get("publication_date") or cit.get("publication_year")
            if cid and pub:
                citations.append({"citing_paper_id": cid, "cited_paper_id": norm["paper_id"],
                                  "citing_date": str(pub)})
        time.sleep(0.2)
    snapshot = {"papers": papers, "citations": citations, "authors": authors, "support_links": []}
    out = ROOT / "data" / "raw" / "context" / "context.json"
    write_json(out, snapshot)
    print(json.dumps({"papers": len(papers), "citations": len(citations),
                      "authors": len(authors), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
