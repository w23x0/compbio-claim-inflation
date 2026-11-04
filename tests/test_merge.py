"""Unit tests for the MD-aligned real-data paths."""
from __future__ import annotations

import gzip
import zipfile
from datetime import date
from pathlib import Path

import pytest

from cbi.config import load_config
from cbi.m1.audit import audit_claim, audit_claims
from cbi.m1.geo import geo_exists, set_geo_cache
from cbi.m1.hgnc import HGNCResolver
from cbi.m1.pmc import parse_pmc_xml
from cbi.schemas import Claim, DataAvailability, Entities, ReportedChoices, ReportedCohort, ReportedEffect, SourceRef

PMC_XML = '''<?xml version="1.0"?>
<article><front><article-meta>
<article-id pub-id-type="pmc">PMC999001</article-id>
<title-group><article-title>TP53 expression in liver cancer</article-title></title-group>
<pub-date><year>2020</year><month>1</month><day>1</day></pub-date>
<abstract><p>High TP53 expression was associated with worse overall survival in hepatocellular carcinoma.</p></abstract>
</article-meta></front><body>
<table-wrap><label>Table 1</label><caption>Univariate Cox model</caption><table>
<tr><th>Gene</th><th>HR</th><th>95% CI</th><th>P</th></tr>
<tr><td>TP53</td><td>2.1</td><td>1.4-3.2</td><td>0.001</td></tr>
</table></table-wrap>
</body></article>'''


@pytest.fixture
def root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pmc_zip_and_nxml_round_trip(tmp_path):
    xml_bytes = PMC_XML.encode("utf-8")
    direct = parse_pmc_xml(xml_bytes)
    assert direct.pmcid == "PMC999001"
    assert direct.path is None

    archive = tmp_path / "bulk.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("article.nxml", xml_bytes)
    from_archive = parse_pmc_xml(archive)
    assert from_archive.pmcid == direct.pmcid
    assert from_archive.full_text == direct.full_text

    from_string = parse_pmc_xml(PMC_XML)
    assert from_string.pmcid == direct.pmcid


def test_jats_table_rows_are_rendered():
    article = parse_pmc_xml(PMC_XML.encode("utf-8"))
    assert "Gene: TP53; HR: 2.1" in article.tables_text
    assert "95% CI: 1.4-3.2" in article.tables_text


def test_geo_cache_distinguishes_unknown_from_missing():
    set_geo_cache({"GSE14520"})
    assert geo_exists("GSE14520") is True
    assert geo_exists("GSE999999") is False
    set_geo_cache(None)
    assert geo_exists("GSE14520") is None


def test_hgnc_alias_returns_canonical_symbol(tmp_path):
    path = tmp_path / "hgnc.tsv"
    path.write_text(
        "hgnc_id\tsymbol\talias_symbol\n"
        "HGNC:3430\tERBB2\tHER2|NEU\n"
        "HGNC:11998\tTP53\tP53\n",
        encoding="utf-8",
    )
    resolver = HGNCResolver(path)
    assert resolver.normalize("HER2") == ("ERBB2", "HGNC:3430")
    assert resolver.normalize("p53") == ("TP53", "HGNC:11998")


def _claim(**overrides):
    data = dict(
        claim_id="c1", source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene="TP53", hgnc_id="HGNC:11998", cancer_type="LIHC"),
        endpoint="OS", direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8, p=0.001),
        reported_cohort=ReportedCohort(accessions=["GSE14520"]),
        reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
        data_availability=DataAvailability(),
    )
    data.update(overrides)
    return Claim(**data)


def test_consistency_audit_flags_bad_accession():
    good = _claim()
    assert audit_claim(good) == []
    bad = good.model_copy(deep=True)
    bad.reported_cohort.accessions = ["not-a-geo-id"]
    assert any(issue.startswith("bad_accession") for issue in audit_claim(bad))
    assert audit_claims([good, bad])["pass_rate"] == 0.5


def test_llm_output_coercion_handles_common_model_drift():
    from cbi.m1.llm import _coerce_claim_obj

    article = parse_pmc_xml(PMC_XML.encode("utf-8"))
    raw = {
        "type": "json_object",
        "entities": {"gene": "PTTG1 (SECURIN)", "cancer_type": "liver cancer", "tissue": "breast"},
        "endpoint": "os",
        "direction": "high expression worse",
        "reported_effect": {"metric": "hr", "value": "2.1", "ci_low": "1.4", "ci_high": "3.2", "p": "0.001"},
        "reported_choices": {"contrast_type": "continuous expression", "cutpoint": None,
                             "normalization": None, "covariates": None},
        "reported_cohort": {"accessions": "TCGA-LIHC", "n_reported": "200"},
        "data_availability": {"code_available": None, "raw_deposited": "true"},
        "evidence_span": {"start": None, "end": None},
        "extraction_confidence": "0.9",
    }
    out = _coerce_claim_obj(raw, article)
    assert out["entities"]["gene"] == "PTTG1"
    assert out["entities"]["cancer_type"] == "LIHC"
    assert out["entities"]["tissue"] == "tumor"
    assert out["direction"] == "high_expression_worse"
    assert out["reported_effect"]["value"] == 2.1
    assert out["reported_choices"]["contrast_type"] == "continuous"
    assert out["reported_choices"]["cutpoint"] == "unspecified"
    assert out["reported_choices"]["covariates"] == []
    assert out["reported_cohort"]["accessions"] == ["TCGA-LIHC"]
    assert out["data_availability"]["code_available"] is False
    assert out["evidence_span"] is None


def test_worker_cannot_request_non_dev_roles(root):
    from cbi.cohort_pool.loader import AccessDenied, CohortStore

    cfg = load_config(root)
    with pytest.raises(AccessDenied):
        CohortStore(cfg, allowed_roles={"dev", "heldout"})


def test_audit_rejects_scope_and_assay_drift(root):
    from cbi.m1.hgnc import HGNCResolver

    resolver = HGNCResolver(root / "data" / "raw" / "hgnc" / "hgnc_complete.tsv")
    base = _claim()
    ihc = base.model_copy(deep=True)
    ihc.reported_choices.normalization = "IHC"
    assert "non_expression_assay" in audit_claim(ihc, resolver=resolver)

    review = base.model_copy(deep=True)
    review.reported_cohort.subset_desc = "meta-analysis across 17 cancer types"
    assert "review_or_pancancer" in audit_claim(review, resolver=resolver)

    out_of_scope = base.model_copy(deep=True)
    out_of_scope.entities.cancer_type = "LUAD"
    assert "cancer_out_of_scope" in audit_claim(out_of_scope, allowed_cancers={"LIHC"}, resolver=resolver)

    bad_hgnc = base.model_copy(deep=True)
    bad_hgnc.entities.hgnc_id = "HGNC:3430"  # ERBB2, not TP53
    assert "hgnc_symbol_mismatch" in audit_claim(bad_hgnc, resolver=resolver)
