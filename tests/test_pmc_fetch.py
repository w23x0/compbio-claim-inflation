"""Concurrent PMC fetch: cache skip, worker fan-out and manifest."""
from __future__ import annotations

import json
import threading
import time

from cbi.m1.pmc_fetch import fetch_batch


def test_fetch_batch_workers_and_cache(monkeypatch, tmp_path):
    calls = []
    lock = threading.Lock()

    def fake_search(query, *, retmax=100, email=None, api_key=None):
        return ["1", "2", "3"]

    def fake_fetch(pmcid, *, email=None, api_key=None):
        with lock:
            calls.append(pmcid)
        time.sleep(0.05)
        return f"<article><id>{pmcid}</id></article>".encode()

    monkeypatch.setattr("cbi.m1.pmc_fetch.search_pmc", fake_search)
    monkeypatch.setattr("cbi.m1.pmc_fetch.fetch_pmc_xml", fake_fetch)
    paths = fetch_batch("q", tmp_path, limit=3, workers=3, rate=1000)
    assert [p.name for p in paths] == ["PMC1.xml", "PMC2.xml", "PMC3.xml"]
    assert sorted(calls) == ["1", "2", "3"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["workers"] == 3

    # Existing files are served from cache, not re-fetched.
    calls.clear()
    fetch_batch("q", tmp_path, limit=3, workers=3, rate=1000)
    assert calls == []
