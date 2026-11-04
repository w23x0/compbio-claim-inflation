"""Cached per-cohort effect estimation shared by M2, M5 and M6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from cbi.cohort_pool.loader import CohortStore
from cbi.config import Config
from cbi.m2.specs import AnalysisSpec, CohortEffect, estimate_effect
from cbi.schemas import Claim, Direction

_SIGN = {
    "high_expression_worse": 1.0,
    "low_expression_better": 1.0,
    "high_expression_better": -1.0,
    "low_expression_worse": -1.0,
}


def align_sign(direction: Direction | str) -> float:
    return _SIGN[str(direction)]


def aligned(effect: CohortEffect, direction: Direction | str) -> CohortEffect:
    sign = align_sign(direction)
    if sign > 0:
        return effect
    return CohortEffect(
        cohort_id=effect.cohort_id,
        gene=effect.gene,
        endpoint=effect.endpoint,
        log_effect=-effect.log_effect,
        se=effect.se,
        p=effect.p,
        n=effect.n,
        events=effect.events,
        spec_id=effect.spec_id,
        role=effect.role,
    )


@dataclass
class CohortAnalyzer:
    cfg: Config
    store: CohortStore

    def __post_init__(self) -> None:
        self._cache: dict[tuple[str, str, str, str], CohortEffect | None] = {}
        self._gene_cache: dict[tuple[str, str], pd.Series | None] = {}
        self.min_samples = int(self.cfg.scoring["gt"]["min_samples_per_cohort"])
        self.min_events = int(self.cfg.scoring["gt"]["min_events_per_cohort"])

    def effect(self, cohort_id: str, gene: str, spec: AnalysisSpec, endpoint: str) -> CohortEffect | None:
        key = (cohort_id, gene, spec.signature(), endpoint)
        if key in self._cache:
            return self._cache[key]
        clinical = self.store.clinical(cohort_id)
        gkey = (cohort_id, gene)
        if gkey not in self._gene_cache:
            self._gene_cache[gkey] = self.store.gene_expression(cohort_id, gene)
        series = self._gene_cache[gkey]
        if series is None:
            out = None
        else:
            out = estimate_effect(clinical, series, gene, spec, endpoint,
                                  min_samples=self.min_samples, min_events=self.min_events)
        if out is not None:
            out = CohortEffect(**{**out.__dict__, "role": self.cfg.cohort_split["roles"].get(cohort_id, "dev")})
        self._cache[key] = out
        return out

    def effects_for_cohorts(self, cohort_ids: Iterable[str], gene: str, spec: AnalysisSpec,
                            endpoint: str) -> list[CohortEffect]:
        out = []
        for cid in cohort_ids:
            effect = self.effect(cid, gene, spec, endpoint)
            if effect is not None and abs(effect.log_effect) <= 6.0 and effect.se <= 3.0:
                out.append(effect)
        return out

    def accession_map(self) -> dict[str, str]:
        return {
            c["accession"]: c["cohort_id"]
            for c in self.cfg.cohort_pool.get("cohorts", [])
            if c.get("accession")
        }

    def cohort_for_accessions(self, accessions: Iterable[str]) -> str | None:
        mapping = self.accession_map()
        for acc in accessions:
            if acc in mapping:
                return mapping[acc]
        return None


def claim_gene(claim: Claim) -> str:
    return claim.entities.gene
