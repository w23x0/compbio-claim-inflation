"""M1 fixture generator round-trips through the PMC parser."""
from __future__ import annotations

from cbi.m1.fixtures import generate_claims, render_claim
from cbi.m1.pmc import parse_pmc_xml
from scripts.p1_fixture_audit import _score


def test_fixtures_render_to_parseable_articles_with_methods():
    claims = generate_claims(10, seed=7)
    for claim in claims:
        article = parse_pmc_xml(render_claim(claim))
        assert claim.entities.gene in article.abstract
        assert "Materials and methods" in article.body_sections
        assert claim.reported_cohort.accessions[0] in article.body_sections["Materials and methods"]


def test_perfect_extractions_score_one():
    claims = generate_claims(5, seed=3)
    score = _score([(claim, claim, None) for claim in claims])
    assert score["min_field_precision"] == 1.0
    assert score["effect_error_rate"] == 0.0
