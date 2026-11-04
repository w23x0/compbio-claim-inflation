"""P4 score cache persists every finished request (short API windows)."""
from __future__ import annotations

import json

import pytest

from cbi.m4.train_llm import _fill_cache


def _read(path):
    return json.loads(path.read_text())["claims"]


def test_every_finished_chunk_is_flushed(tmp_path):
    cache_path = tmp_path / "scores.json"
    cache: dict[str, dict] = {}
    ids = [f"c{i}" for i in range(6)]

    def score_chunk(chunk):
        return {cid: {"score": 0.5} for cid in chunk}

    _fill_cache(ids, cache, cache_path, score_chunk, meta={"model": "m", "version": "v"},
                workers=1, batch_size=1)
    assert set(_read(cache_path)) == set(ids)


def test_interrupt_keeps_completed_scores(tmp_path):
    cache_path = tmp_path / "scores.json"
    cache: dict[str, dict] = {}
    ids = [f"c{i}" for i in range(5)]
    calls = {"n": 0}

    def score_chunk(chunk):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt
        return {cid: {"score": 0.5} for cid in chunk}

    with pytest.raises(KeyboardInterrupt):
        _fill_cache(ids, cache, cache_path, score_chunk, meta={"model": "m", "version": "v"},
                    workers=1, batch_size=1)
    assert set(_read(cache_path)) == {"c0", "c1", "c2"}


def test_deadline_stops_with_partial_cache(tmp_path):
    import time as _time

    cache_path = tmp_path / "scores.json"
    cache: dict[str, dict] = {}
    ids = [f"c{i}" for i in range(10)]

    def score_chunk(chunk):
        _time.sleep(0.02)
        return {cid: {"score": 0.5} for cid in chunk}

    _fill_cache(ids, cache, cache_path, score_chunk, meta={"model": "m", "version": "v"},
                workers=1, batch_size=1, deadline_seconds=0.05)
    saved = _read(cache_path)
    assert 0 < len(saved) < len(ids)


def test_outage_stops_before_submitting_rest_of_corpus(tmp_path):
    calls = []

    def score_chunk(chunk):
        calls.extend(chunk)
        return {}

    _fill_cache([f"c{i}" for i in range(100)], {}, tmp_path / "scores.json", score_chunk,
                meta={}, workers=3, batch_size=2)
    assert len(calls) == 6


def test_parallel_results_are_persisted_before_other_requests_finish(tmp_path):
    from threading import Event

    cache_path = tmp_path / "scores.json"
    flushed = Event()

    def score_chunk(chunk):
        if chunk == ["slow"]:
            assert flushed.wait(timeout=5)
            assert "fast" in _read(cache_path)
        return {chunk[0]: {"score": 0.5}}

    from cbi.m4 import train_llm
    original_write = train_llm.write_json

    def observed_write(path, obj):
        original_write(path, obj)
        if "fast" in obj["claims"]:
            flushed.set()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_llm, "write_json", observed_write)
        _fill_cache(["fast", "slow"], {}, cache_path, score_chunk, meta={}, workers=2)
    assert set(_read(cache_path)) == {"fast", "slow"}
