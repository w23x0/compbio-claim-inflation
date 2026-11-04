"""PMC Open Access batch fetch via NCBI E-utilities.

This is the automated input path for the M1 public-PMC audit.  It stores raw
XML under `data/raw/pmc_real/<pmcid>.xml`, so a rerun uses the cache and does not hit
NCBI again.  Requests are rate-limited to stay under the 3/s anonymous limit.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
from pathlib import Path

from cbi.util import write_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = (
    'open access[filter] AND prognostic[Title/Abstract] AND '
    '"overall survival"[Title/Abstract] AND (gene[Title/Abstract] OR expression[Title/Abstract])'
)
DEFAULT_TERM = "cancer"


class FetchError(RuntimeError):
    pass


def _get(url: str, *, retries: int = 5, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cbi-m1/1.2 (public PMC OA audit)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            # 429 is a rate-limit signal, not a transient one-second hiccup.
            delay = 8.0 * (attempt + 1) if exc.code == 429 else 2.0 * (attempt + 1)
            time.sleep(delay)
        except Exception as exc:  # network / throttling
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise FetchError(f"failed after {retries} attempts: {url}: {last}")


def search_pmc(query: str, *, retmax: int = 100, email: str | None = None,
               api_key: str | None = None) -> list[str]:
    params = {
        "db": "pmc", "term": query, "retmax": str(retmax),
        "retmode": "json", "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    payload = json.loads(_get(url).decode("utf-8", "replace"))
    return list(payload.get("esearchresult", {}).get("idlist", []))


def fetch_pmc_xml(pmcid: str, *, email: str | None = None, api_key: str | None = None) -> bytes:
    params = {"db": "pmc", "id": str(pmcid), "retmode": "xml"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return _get(f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode(params)}")


class _RateLimiter:
    """Space request starts so concurrent workers stay under the NCBI limit."""

    def __init__(self, rate: float):
        self.interval = 1.0 / max(float(rate), 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


def fetch_batch(query: str, out_dir: str | Path, *, limit: int = 100,
                email: str | None = None, api_key: str | None = None,
                sleep: float = 0.34, workers: int = 1,
                rate: float | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = search_pmc(query, retmax=limit, email=email, api_key=api_key)
    by_id: dict[str, Path] = {}
    missing: list[str] = []
    for pmcid in ids:
        if not str(pmcid).isdigit():
            raise FetchError(f"unexpected PMC id from E-utilities: {pmcid!r}")
        path = out_dir / f"PMC{pmcid}.xml"
        if path.is_file() and path.stat().st_size > 0:
            by_id[str(pmcid)] = path
        else:
            missing.append(str(pmcid))

    workers = max(1, min(int(workers), 8))
    if missing:
        if workers == 1:
            for i, pmcid in enumerate(missing):
                path = out_dir / f"PMC{pmcid}.xml"
                path.write_bytes(fetch_pmc_xml(pmcid, email=email, api_key=api_key))
                by_id[pmcid] = path
                if i + 1 < len(missing):
                    time.sleep(sleep)
        else:
            # Anonymous E-utilities allow ~3 requests/s; an API key raises this.
            limiter = _RateLimiter(float(rate if rate is not None else (8.0 if api_key else 2.0)))

            def download(pmcid: str) -> tuple[str, Path]:
                limiter.wait()
                path = out_dir / f"PMC{pmcid}.xml"
                path.write_bytes(fetch_pmc_xml(pmcid, email=email, api_key=api_key))
                return pmcid, path

            failures: dict[str, str] = {}

            def safe_download(pmcid: str) -> tuple[str, Path | None, str | None]:
                try:
                    _, path = download(pmcid)
                    return pmcid, path, None
                except Exception as exc:
                    return pmcid, None, f"{type(exc).__name__}: {exc}"

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for pmcid, path, error in pool.map(safe_download, missing):
                    if path is not None:
                        by_id[pmcid] = path
                    else:
                        failures[pmcid] = error or "unknown error"
    else:
        failures = {}

    paths = [by_id[str(pmcid)] for pmcid in ids if str(pmcid) in by_id]
    write_json(out_dir / "manifest.json", {
        "query": query, "requested": limit, "returned": len(ids),
        "workers": workers, "rate": rate,
        "ids": ids, "files": [p.name for p in paths],
        "failures": failures,
    })
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-pmc-fetch")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--term", default=DEFAULT_TERM,
                        help="extra term appended to the default query")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="data/raw/pmc_real")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    parser.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY")
                        or os.environ.get("CBI_NCBI_API_KEY"))
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel efetch workers (1 = sequential)")
    parser.add_argument("--rate", type=float, default=None,
                        help="request starts per second (default: 2 anonymous, 8 with key)")
    args = parser.parse_args(argv)
    query = args.query
    if args.term and args.term not in query:
        query = f"({query}) AND {args.term}"
    paths = fetch_batch(query, args.out, limit=args.limit, email=args.email,
                        api_key=args.api_key, workers=args.workers, rate=args.rate)
    print(f"downloaded {len(paths)} PMC XML files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
