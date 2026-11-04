"""`as_of_date` unit tests for M3 (spec P3 acceptance)."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from cbi.m3.asof import add_years
from cbi.m3.features import ContextFeatureBuilder
from cbi.m3.graph import author_clusters
from cbi.m3.sources import ContextSnapshot


def _paper(pid: str, authors: list[str], pub: date, **extra) -> dict:
    return {
        "paper_id": pid,
        "pub_date": pub.isoformat(),
        "author_ids": authors,
        **extra,
    }


def _claim(cid: str, pub: date) -> SimpleNamespace:
    return SimpleNamespace(claim_id=cid, source=SimpleNamespace(pub_date=pub))


def _gt(score: float) -> SimpleNamespace:
    return SimpleNamespace(robustness_score=score, label_eligible=True)


def _builder(papers, claims, gts, citations=None) -> ContextFeatureBuilder:
    snapshot = ContextSnapshot({
        "papers": papers,
        "citations": citations or [],
        "authors": {},
        "support_links": [],
    })
    return ContextFeatureBuilder(None, claims, gts, snapshot)


def test_author_clusters_exclude_future_papers():
    papers = [
        _paper("p_old", ["A1"], date(2020, 1, 1)),
        _paper("p_new", ["A1", "A2"], date(2022, 1, 1)),
    ]
    early = author_clusters(papers, date(2021, 1, 1))
    late = author_clusters(papers, date(2023, 1, 1))
    assert "A1" in early and "A2" not in early
    assert late["A1"] == late["A2"]


def test_future_citation_is_invisible():
    papers = [
        _paper("p1", ["A1"], date(2020, 1, 1), claim_id="c1"),
        _paper("p2", ["A1"], date(2022, 1, 1)),
    ]
    citations = [{
        "cited_paper_id": "p1", "citing_paper_id": "p2",
        "citing_date": "2022-06-01",
    }]
    builder = _builder(papers, [_claim("c1", date(2020, 1, 1))],
                       {"c1": _gt(0.5)}, citations)
    early = builder.build(_claim("c1", date(2020, 1, 1)), date(2021, 6, 1))
    late = builder.build(_claim("c1", date(2020, 1, 1)), date(2023, 6, 1))
    assert early.citation_cascade_score is None
    assert late.citation_cascade_score == 1.0


def test_future_retraction_is_invisible():
    papers = [
        _paper("p1", ["A1"], date(2020, 1, 1), claim_id="c1",
               retraction_date="2023-01-01"),
    ]
    builder = _builder(papers, [_claim("c1", date(2020, 1, 1))],
                       {"c1": _gt(0.5)})
    before = builder.build(_claim("c1", date(2020, 1, 1)), date(2022, 1, 1))
    after = builder.build(_claim("c1", date(2020, 1, 1)), date(2024, 1, 1))
    assert before.post_pub_signals["retraction"] is False
    assert after.post_pub_signals["retraction"] is True


def test_lab_history_is_leave_one_out_and_truncated():
    papers = [_paper("p0", ["A1"], date(2020, 1, 1), claim_id="c0")]
    claims = [_claim("c0", date(2020, 1, 1))]
    gts = {"c0": _gt(0.9)}
    for i in range(1, 6):
        papers.append(_paper(f"p{i}", ["A1"], date(2019, 1, 1), claim_id=f"c{i}"))
        claims.append(_claim(f"c{i}", date(2019, 1, 1)))
        gts[f"c{i}"] = _gt(0.4 + i * 0.1)
    # A future paper in the same cluster must not enter the history.
    papers.append(_paper("p6", ["A1"], date(2022, 1, 1), claim_id="c6"))
    claims.append(_claim("c6", date(2022, 1, 1)))
    gts["c6"] = _gt(0.0)

    builder = _builder(papers, claims, gts)
    features = builder.build(claims[0], date(2021, 1, 1))
    assert features.lab_historical_n == 5
    assert features.lab_historical_replication_rate == (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5

    # Fewer than five eligible prior claims: the field must stay null.
    builder_short = _builder(papers[:5], claims[:5], {k: gts[k] for k in gts if k in {f"c{i}" for i in range(5)}})
    short = builder_short.build(claims[0], date(2021, 1, 1))
    assert short.lab_historical_n == 4
    assert short.lab_historical_replication_rate is None


def test_add_years_handles_leap_day():
    assert add_years(date(2020, 2, 29), 1) == date(2021, 2, 28)
