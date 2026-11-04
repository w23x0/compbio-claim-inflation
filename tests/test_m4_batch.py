"""Batched M4 content scoring: alignment and explicit-missing semantics."""
from __future__ import annotations

import json
from datetime import date

from cbi.m4.llm_judge import llm_checklist_score_many
from cbi.schemas import Claim, DataAvailability, Entities, ReportedChoices, ReportedCohort, ReportedEffect, SourceRef


def _claim(cid: str) -> Claim:
    return Claim(
        claim_id=cid, source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene="TP53", cancer_type="LIHC"), endpoint="OS",
        direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8),
        reported_cohort=ReportedCohort(accessions=["TCGA-LIHC"]),
        reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
        data_availability=DataAvailability(),
    )


class _Resp:
    def __init__(self, payload: dict):
        self.status_code = 200
        self.text = ""
        self.content = json.dumps(payload).encode()
        self._payload = payload

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}

    def raise_for_status(self):
        return None


def test_many_scores_are_aligned_by_index_and_missing_is_none(monkeypatch):
    calls = []

    def fake_post(url, *, headers=None, timeout=120, **kwargs):
        calls.append(kwargs["json"])
        return _Resp({"items": [
            {"index": 2, "score": 0.8, "rationale": ["c"], "checklist": {}},
            {"index": 0, "score": 0.2, "rationale": ["a"], "checklist": {}},
        ]})

    monkeypatch.setenv("CBI_LLM_ENDPOINT", "https://example.test/v1")
    monkeypatch.setenv("CBI_LLM_MODEL", "test-model")
    monkeypatch.setattr("cbi.m4.llm_judge.llm_post", fake_post)
    items = [(_claim("c0"), "text a"), (_claim("c1"), "text b"), (_claim("c2"), "text c")]
    out = llm_checklist_score_many(items)
    assert len(calls) == 1
    prompt = calls[0]["messages"][0]["content"]
    assert "index=0" in prompt and "index=2" in prompt
    assert out[0]["score"] == 0.2
    assert out[1] is None
    assert out[2]["score"] == 0.8
