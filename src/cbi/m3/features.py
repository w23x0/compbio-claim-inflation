"""M3 context features, all computed with an explicit as_of_date."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from cbi.config import Config
from cbi.m3.asof import add_years, is_visible, parse_date
from cbi.m3.graph import author_clusters
from cbi.m3.sources import ContextSnapshot
from cbi.schemas import Claim, ContextFeatures, GroundTruthResult


@dataclass
class ContextFeatureBuilder:
    cfg: Config
    claims: list[Claim]
    gt_results: dict[str, GroundTruthResult]
    snapshot: ContextSnapshot

    def __post_init__(self) -> None:
        self.by_id = {c.claim_id: c for c in self.claims}
        self.paper_by_claim = {p.get("claim_id"): p for p in self.snapshot.papers if p.get("claim_id")}
        self.papers_by_id = {p["paper_id"]: p for p in self.snapshot.papers}
        self.citations_to: dict[str, list[dict]] = {}
        for cit in self.snapshot.citations:
            self.citations_to.setdefault(cit["cited_paper_id"], []).append(cit)
        self._cluster_cache: dict[date, dict[str, str]] = {}

    def clusters(self, as_of: date) -> dict[str, str]:
        if as_of not in self._cluster_cache:
            self._cluster_cache[as_of] = author_clusters(self.snapshot.papers, as_of)
        return self._cluster_cache[as_of]

    def _cluster_of_paper(self, paper: dict | None, as_of: date) -> str | None:
        if not paper:
            return None
        authors = paper.get("author_ids", [])
        cmap = self.clusters(as_of)
        for a in authors:
            if a in cmap:
                return cmap[a]
        return None

    def _lab_history(self, claim: Claim, paper: dict | None, as_of: date) -> tuple[float | None, int]:
        cluster = self._cluster_of_paper(paper, as_of)
        if cluster is None:
            return None, 0
        values: list[float] = []
        for other in self.claims:
            if other.claim_id == claim.claim_id:
                continue
            if other.source.pub_date >= as_of:
                continue
            op = self.paper_by_claim.get(other.claim_id)
            if op is None or op.get("paper_id") == (paper or {}).get("paper_id"):
                continue
            if self._cluster_of_paper(op, as_of) != cluster:
                continue
            result = self.gt_results.get(other.claim_id)
            if result is None or not result.label_eligible:
                continue
            values.append(result.robustness_score)
        if len(values) < 5:
            return None, len(values)
        return float(sum(values) / len(values)), len(values)

    def build(self, claim: Claim, as_of: date) -> ContextFeatures:
        paper = self.paper_by_claim.get(claim.claim_id)
        cmap = self.clusters(as_of)
        cluster = self._cluster_of_paper(paper, as_of)
        citations = [
            c for c in self.citations_to.get((paper or {}).get("paper_id", ""), [])
            if is_visible(c.get("citing_date"), as_of)
        ]
        support_links = [
            s for s in self.snapshot.support_links
            if s.get("claim_id") == claim.claim_id and is_visible(s.get("pub_date"), as_of)
        ]
        supporting_papers = len(support_links)
        citing_clusters = []
        for link in support_links:
            p = self.papers_by_id.get(link.get("paper_id", ""))
            c = self._cluster_of_paper(p, as_of)
            citing_clusters.append(c or link.get("paper_id"))
        effective = float(len(set(citing_clusters))) if citing_clusters else None
        same_cluster = 0
        for link in citations:
            p = self.papers_by_id.get(link.get("citing_paper_id", ""))
            if self._cluster_of_paper(p, as_of) == cluster:
                same_cluster += 1
        cascade = float(same_cluster / len(citations)) if citations else None
        self_ratio = cascade
        lab_rate, lab_n = self._lab_history(claim, paper, as_of)
        coi = list((paper or {}).get("coi_flags", []))
        review_days = None
        if paper:
            received, accepted = parse_date(paper.get("received_date")), parse_date(paper.get("accepted_date"))
            if received and accepted and accepted <= as_of and accepted >= received:
                review_days = int((accepted - received).days)
        post: dict[str, Any] = {"retraction": False, "correction": False, "pubpeer_threads": 0}
        if paper:
            post["retraction"] = is_visible(paper.get("retraction_date"), as_of)
            post["correction"] = is_visible(paper.get("correction_date"), as_of)
            post["pubpeer_threads"] = sum(
                1 for t in paper.get("pubpeer_threads", [])
                if is_visible(t.get("date"), as_of)
            )
        venue = {
            "journal_id": (paper or {}).get("journal"),
            "oa": bool((paper or {}).get("oa", True)),
        }
        features = ContextFeatures(
            claim_id=claim.claim_id,
            as_of_date=as_of,
            author_cluster_id=cluster,
            supporting_papers=supporting_papers if paper else None,
            effective_independent_support=effective,
            citation_cascade_score=cascade,
            lab_historical_replication_rate=lab_rate,
            lab_historical_n=lab_n,
            self_citation_ratio=self_ratio,
            coi_flags=coi,
            review_time_days=review_days,
            venue_features=venue,
            post_pub_signals=post,
        )
        features.missing_mask = {
            "author_cluster_id": features.author_cluster_id is None,
            "supporting_papers": features.supporting_papers is None,
            "effective_independent_support": features.effective_independent_support is None,
            "citation_cascade_score": features.citation_cascade_score is None,
            "lab_historical_replication_rate": features.lab_historical_replication_rate is None,
            "self_citation_ratio": features.self_citation_ratio is None,
            "review_time_days": features.review_time_days is None,
        }
        return features

    def build_all(self, *, variant: str = "pub_plus_3y") -> dict[str, ContextFeatures]:
        out = {}
        for claim in self.claims:
            if variant == "publication":
                as_of = claim.source.pub_date
            elif variant == "pub_plus_3y":
                as_of = add_years(claim.source.pub_date, 3)
            else:
                raise ValueError(f"unknown as_of variant: {variant}")
            out[claim.claim_id] = self.build(claim, as_of)
        return out
