"""Preflight probe: a closed API window must abort before any scoring run."""
from __future__ import annotations

from cbi.m4.llm_judge import llm_ping


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_ping_true_on_200(monkeypatch):
    monkeypatch.setenv("CBI_LLM_ENDPOINT", "https://example.test/v1")
    monkeypatch.setenv("CBI_LLM_MODEL", "test-model")
    monkeypatch.setattr("cbi.m4.llm_judge.llm_post", lambda *a, **k: _Resp(200))
    assert llm_ping() is True


def test_ping_false_on_503(monkeypatch):
    monkeypatch.setenv("CBI_LLM_ENDPOINT", "https://example.test/v1")
    monkeypatch.setenv("CBI_LLM_MODEL", "test-model")
    monkeypatch.setattr("cbi.m4.llm_judge.llm_post", lambda *a, **k: _Resp(503))
    assert llm_ping() is False
