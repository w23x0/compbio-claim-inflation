"""Batched M1 LLM extraction (request shape + per-article fallback)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cbi.config import load_config
from cbi.m1.extractor import ClaimExtractor
from cbi.m1.llm import extract_batch_with_llm
from cbi.m1.pmc import PMCArticle
from cbi.schemas import Claim, DataAvailability, Entities, ReportedChoices, ReportedCohort, ReportedEffect, SourceRef


def _article(pmcid: str, gene: str = "TP53") -> PMCArticle:
    return PMCArticle(
        pmcid=pmcid, pmid=None, doi=None, title=f"{gene} in liver cancer",
        journal="Test Journal", pub_date=date(2020, 1, 1),
        abstract=f"High {gene} expression was associated with worse overall survival.",
        body="Methods and results.", full_text="", tables_text="",
    )


def _claim_obj(gene: str = "TP53", cancer: str = "LIHC") -> dict:
    return {
        "claim_type": "prognostic_gene",
        "entities": {"gene": gene, "cancer_type": cancer, "tissue": "tumor"},
        "endpoint": "OS",
        "direction": "high_expression_worse",
        "reported_effect": {"metric": "HR", "value": 2.1, "ci_low": 1.4, "ci_high": 3.2, "p": 0.001},
        "reported_cohort": {"accessions": ["TCGA-LIHC"], "n_reported": 200},
        "reported_choices": {"contrast_type": "dichotomous", "cutpoint": "median"},
        "data_availability": {"raw_deposited": True, "code_available": False},
        "extraction_confidence": 0.9,
    }


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


def test_batch_prompt_contains_all_articles_and_maps_pmcid(monkeypatch):
    calls = []

    def fake_post(url, *, headers=None, timeout=120, **kwargs):
        calls.append(kwargs["json"])
        return _Resp({"claims": [
            {"pmcid": "PMC1", "claim": _claim_obj("TP53")},
            {"pmcid": "PMC2", "claim": None},
        ]})

    monkeypatch.setenv("CBI_LLM_ENDPOINT", "https://example.test/v1")
    monkeypatch.setenv("CBI_LLM_MODEL", "test-model")
    monkeypatch.setattr("cbi.m1.llm.llm_post", fake_post)
    out = extract_batch_with_llm([_article("PMC1"), _article("PMC2")])
    assert len(calls) == 1
    prompt = calls[0]["messages"][0]["content"]
    assert "PMC1" in prompt and "PMC2" in prompt
    assert set(out) == {"PMC1", "PMC2"}
    assert out["PMC1"] is not None and out["PMC1"].entities.gene == "TP53"
    assert out["PMC2"] is None


def test_missing_batch_entry_falls_back_to_single_article(monkeypatch):
    # Use the real project config for the HGNC table and cancer scope.
    cfg = load_config(Path(__file__).resolve().parents[1])
    extractor = ClaimExtractor(cfg)
    articles = [_article("PMC1"), _article("PMC2", gene="ERBB2")]
    single_calls = []

    batch_claim = Claim(
        claim_id="clm_PMC1",
        source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene="TP53", cancer_type="LIHC"),
        endpoint="OS", direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=2.1, ci_low=1.4, ci_high=3.2),
        reported_cohort=ReportedCohort(accessions=["TCGA-LIHC"]),
        reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
        data_availability=DataAvailability(),
    )

    def fake_batch(chunk, **kwargs):
        return {"PMC1": batch_claim}

    def fake_single(article):
        single_calls.append(article.pmcid)
        return Claim(
            claim_id=f"clm_{article.pmcid}",
            source=SourceRef(pub_date=date(2020, 1, 1)),
            entities=Entities(gene="ERBB2", cancer_type="LIHC"),
            endpoint="OS", direction="high_expression_worse",
            reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8),
            reported_cohort=ReportedCohort(accessions=["TCGA-LIHC"]),
            reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
            data_availability=DataAvailability(),
        )

    monkeypatch.setattr("cbi.m1.extractor.llm_available", lambda: True)
    monkeypatch.setattr("cbi.m1.extractor.extract_batch_with_llm", fake_batch)
    monkeypatch.setattr("cbi.m1.extractor.extract_with_llm", fake_single)
    results = extractor.extract_articles(articles, batch_size=2)
    assert [r[1].entities.gene for r in results] == ["TP53", "ERBB2"]
    assert single_calls == ["PMC2"]


PMC_XML = '''<?xml version="1.0"?>
<article><front><article-meta>
<article-id pub-id-type="pmc">PMC999001</article-id>
<title-group><article-title>TP53 expression in liver cancer</article-title></title-group>
<pub-date><year>2020</year><month>1</month><day>1</day></pub-date>
<abstract><p>High TP53 expression was associated with worse overall survival in hepatocellular carcinoma.</p></abstract>
</article-meta></front><body><p>Results.</p></body></article>'''


def test_public_audit_uses_one_batch_for_two_files(monkeypatch, tmp_path):
    from cbi.config import load_config as load_cfg
    from cbi.m1.public_audit import run_public_audit
    from cbi.schemas import Claim

    (tmp_path / "PMC1.xml").write_text(PMC_XML.replace("PMC999001", "PMC999001"), encoding="utf-8")
    (tmp_path / "PMC2.xml").write_text(PMC_XML.replace("PMC999001", "PMC999002"), encoding="utf-8")
    calls = []

    class FakeExtractor:
        def __init__(self, cfg):
            self.cfg = cfg

        def extract_article(self, article):
            raise AssertionError("batch path should not call the single path")

        def extract_articles(self, articles, *, batch_size=None):
            calls.append((len(articles), batch_size))
            out = []
            for article in articles:
                claim = Claim(
                    claim_id=f"clm_{article.pmcid}",
                    source=SourceRef(pub_date=date(2020, 1, 1)),
                    entities=Entities(gene="TP53", cancer_type="LIHC"),
                    endpoint="OS", direction="high_expression_worse",
                    reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8),
                    reported_cohort=ReportedCohort(accessions=["TCGA-LIHC"]),
                    reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
                    data_availability=DataAvailability(),
                )
                out.append((article, claim, None))
            return out

    monkeypatch.setattr("cbi.m1.public_audit.ClaimExtractor", FakeExtractor)
    cfg = load_cfg(Path(__file__).resolve().parents[1])
    result = run_public_audit(cfg, tmp_path, batch_size=2, workers=1)
    assert calls == [(2, 2)]
    assert result["batch_size"] == 2
    assert result["n_claims"] == 2 and result["audit_passed"] == 2
    assert all(row["passed"] for row in result["rows"])
