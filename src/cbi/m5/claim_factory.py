"""Build a Claim from an analysis result (used by the red-team worker).

This is not a data generator: the effect comes from `estimate_effect` on a real
cohort.  It only formats that result as a paper-like claim.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

from cbi.m2.specs import AnalysisSpec, CohortEffect
from cbi.schemas import (Claim, DataAvailability, Entities, EvidenceSpan, ReportedChoices,
                         ReportedCohort, ReportedEffect, SourceRef)
from cbi.textgen import locate_evidence_span, render_manuscript


def _choices_from_spec(spec: AnalysisSpec, *, policy: str, n_comparisons: int | None) -> ReportedChoices:
    cutpoint = {
        "median": "median", "tertile": "tertile", "per_sd": "per_sd",
        "optimal": "optimal", "percentile_custom": "percentile_custom",
    }[spec.grouping]
    covariates = []
    if spec.covariates == "age_stage":
        covariates = ["age", "stage"]
    elif spec.covariates == "age_stage_grade":
        covariates = ["age", "stage", "grade"]
    subset = None
    if spec.subgroup == "stage_I_II":
        subset = "stage I-II only"
    elif spec.subgroup == "stage_III_IV":
        subset = "stage III-IV only"
    note = None
    if spec.grouping == "optimal":
        note = "cutoff selected by maximising the log-rank statistic"
    elif policy == "covariate_fishing":
        note = "covariates selected by stepwise regression"
    return ReportedChoices(
        contrast_type="continuous" if spec.grouping == "per_sd" else "dichotomous",
        cutpoint=cutpoint,
        normalization="platform_native",
        covariates=covariates,
        time_cap_years=spec.followup_cap_years,
        subset=subset,
        probe_selection="sig" if policy == "multiverse_fishing" else "max_mean",
        n_comparisons=n_comparisons,
        multiple_testing_correction=(n_comparisons is None or n_comparisons <= 1),
        selection_note=note,
    )


def make_claim(*, claim_id: str, gene: str, cancer: str, paper_cid: str, spec: AnalysisSpec,
               endpoint: str, effect: CohortEffect, meta: dict[str, Any], rng: np.random.Generator,
               policy: str, style: str, pub_date: date, journal: str, n_comparisons: int | None,
               code_available: bool) -> Claim:
    raw = float(effect.log_effect)
    direction = "high_expression_worse" if raw >= 0 else "high_expression_better"
    hr = math.exp(raw)
    se = float(effect.se)
    if math.isfinite(se) and 0 < se < 5:
        ci_low = math.exp(max(-50.0, raw - 1.959963984540054 * se))
        ci_high = math.exp(min(50.0, raw + 1.959963984540054 * se))
    else:
        # A huge/non-finite SE cannot produce a meaningful CI; omit it.
        ci_low = ci_high = None
    choices = _choices_from_spec(spec, policy=policy, n_comparisons=n_comparisons)
    claim = Claim(
        claim_id=claim_id,
        source=SourceRef(pub_date=pub_date, journal=journal,
                         title=f"{gene} expression and {endpoint} in {cancer}"),
        claim_type="prognostic_gene",
        entities=Entities(gene=gene, cancer_type=cancer, tissue="tumor"),
        endpoint=endpoint,
        direction=direction,
        reported_effect=ReportedEffect(metric="HR", value=float(hr), ci_low=float(ci_low),
                                       ci_high=float(ci_high), p=float(effect.p)),
        reported_cohort=ReportedCohort(
            accessions=[meta["accession"]], in_standard_pool=True,
            n_reported=int(effect.n), subset_desc=choices.subset,
        ),
        reported_choices=choices,
        data_availability=DataAvailability(raw_deposited=True, code_available=code_available,
                                           deposited_n=int(meta.get("n_deposited") or meta.get("n") or effect.n)),
        extraction_confidence=float(rng.uniform(0.86, 0.99)),
        extraction_method="redteam_analysis",
    )
    manuscript = render_manuscript(claim, style=style, watermark=False)
    span = locate_evidence_span(claim, manuscript)
    if span is not None:
        claim = claim.model_copy(update={"evidence_span": EvidenceSpan(start=span[0], end=span[1])})
    return claim
