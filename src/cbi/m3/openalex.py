"""OpenAlex / Crossref adapters for a real M3 snapshot build.

These are intentionally small and cache-friendly.  The real-data pipeline reads a
local normalized snapshot instead, so no network call is made during tests or
offline runs.  Set CBI_OPENALEX_MAILTO to identify your client as OpenAlex asks.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests

OPENALEX = "https://api.openalex.org"
CROSSREF = "https://api.crossref.org/works"


def _mailto() -> dict[str, str]:
    mail = os.environ.get("CBI_OPENALEX_MAILTO")
    return {"mailto": mail} if mail else {}


def fetch_openalex_work(doi: str, *, timeout: int = 60) -> dict[str, Any]:
    doi = doi.removeprefix("https://doi.org/")
    resp = requests.get(f"{OPENALEX}/works/https://doi.org/{doi}", params=_mailto(), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_openalex_citations(work_id: str, *, per_page: int = 200, timeout: int = 60) -> list[dict[str, Any]]:
    """Citations are fetched by publication date so as_of filtering is possible."""
    params = {"filter": f"cites:{work_id}", "per-page": per_page, "sort": "publication_date:asc"}
    params.update(_mailto())
    out: list[dict[str, Any]] = []
    cursor = "*"
    while cursor:
        params["cursor"] = cursor
        resp = requests.get(f"{OPENALEX}/works", params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        out.extend(payload.get("results", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return out


def normalize_openalex_work(work: dict[str, Any], *, claim_id: str | None = None) -> dict[str, Any]:
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        aid = (author.get("id") or "").rsplit("/", 1)[-1]
        if aid:
            authors.append(f"A{aid}")
    pub = work.get("publication_date") or work.get("publication_year")
    return {
        "paper_id": work.get("id", "").rsplit("/", 1)[-1],
        "claim_id": claim_id,
        "title": work.get("title"),
        "journal": (work.get("primary_location") or {}).get("source", {}).get("display_name"),
        "pub_date": str(pub) if pub else None,
        "author_ids": authors,
        "doi": work.get("doi"),
        "oa": bool((work.get("open_access") or {}).get("is_oa", False)),
    }


def fetch_crossref(doi: str, *, timeout: int = 60) -> dict[str, Any]:
    doi = doi.removeprefix("https://doi.org/")
    resp = requests.get(f"{CROSSREF}/{doi}", timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {})
