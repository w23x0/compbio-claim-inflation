"""Ground-truth engine (M2): pure function over (claim, cohorts, snapshot).

Scores follow spec v1.1:
  independent_support = Phi(mu / se_mu)
  multiverse_support  = fraction of reference specs significant in claim direction
  shrinkage           = clip(m / r, 0, 1)
  robustness          = weighted sum
  inflation           = tanh(max(0, r - mu) / s), s = ln 2
  inflation_z         = (r - mu) / sqrt(se_r^2 + se_mu^2 + tau^2)
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats

from cbi.config import Config
from cbi.m2.analysis import CohortAnalyzer, align_sign, aligned
from cbi.m2.meta import MetaAnalysis, meta_random_effects
from cbi.m2.specs import (AnalysisSpec, CohortEffect, aligned_reference_specs,
                          claim_spec, default_spec, reference_specs)
from cbi.schemas import (Claim, GroundTruthResult, IndependentCohortEffect,
                         MetaResult, MultiverseResult, OriginalCohortRepro,
                         SpecCurveRow)
from cbi.util import clip01, percentile_rank

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _min_conf(a: str, b: str) -> str:
    return a if _CONF_RANK[a] <= _CONF_RANK[b] else b


def _meta_result(m: MetaAnalysis) -> MetaResult:
    return MetaResult(
        method=m.method,
        pooled_log_effect=m.pooled,
        se=m.se,
        tau2=m.tau2,
        pred_interval=(m.pred_low, m.pred_high),
        k=m.k,
        q=m.q,
        i2=m.i2,
        p=m.p,
    )


def _align_effect(effect: CohortEffect, direction: str) -> CohortEffect:
    return aligned(effect, direction)


class GroundTruthEngine:
    def __init__(self, cfg: Config, analyzer: CohortAnalyzer):
        self.cfg = cfg
        self.analyzer = analyzer
        self.scale = float(cfg.scoring["inflation_scale"])
        self.weights = cfg.weights
        self.gt_cfg = cfg.scoring["gt"]

    def _independent(self, cancer: str, role: str, exclude: set[str]) -> list[str]:
        ids = self.analyzer.store.cohort_ids(cancer=cancer, role=role)
        return [cid for cid in ids if cid not in exclude]

    def _original_repro(self, claim: Claim, paper_cid: str | None) -> tuple[OriginalCohortRepro, str | None]:
        if paper_cid is None:
            return OriginalCohortRepro(status="out_of_pool", reason="accession not in standard cohort pool"), None
        try:
            meta = self.analyzer.store.meta(paper_cid)
        except Exception:  # role-inaccessible cohort is not in the worker's pool
            return OriginalCohortRepro(status="out_of_pool", reason="cohort not accessible in this role"), paper_cid
        spec = claim_spec(claim)
        effect = self.analyzer.effect(paper_cid, claim.entities.gene, spec, claim.endpoint)
        n_dep = int(meta.get("n_deposited") or 0)
        if effect is None:
            return OriginalCohortRepro(status="no_data", n_deposited=n_dep,
                                       reason="claim specification produced no analyzable fit"), paper_cid
        aligned_effect = align_sign(claim.direction) * effect.log_effect
        r = claim.reported_log_effect
        rel = abs(aligned_effect - r) / max(abs(r), 0.15)
        same_sign = np.sign(aligned_effect) == np.sign(r)
        n_reported = claim.reported_cohort.n_reported
        n_gap = abs((n_reported or effect.n) - effect.n) / max(effect.n, 1)
        if not same_sign:
            status, reason = "failed", "reproduced effect has the opposite direction"
        elif rel <= 0.25 and n_gap <= 0.20:
            status, reason = "reproduced", None
        elif rel <= 0.75:
            status, reason = "partial", f"effect differs by {rel:.0%}"
        else:
            status, reason = "failed", f"effect differs by {rel:.0%}"
        return (
            OriginalCohortRepro(
                status=status,
                log_effect=float(aligned_effect),
                se=float(effect.se),
                p=float(effect.p),
                n_used=int(effect.n),
                n_deposited=n_dep,
                reason=reason,
            ),
            paper_cid,
        )

    def _multiverse(self, claim: Claim, cohort_ids: list[str]) -> MultiverseResult:
        specs = reference_specs()
        rows: list[SpecCurveRow] = []
        pooled_by_spec: dict[str, tuple[float, float, float, int]] = {}
        for spec in specs:
            effects = self.analyzer.effects_for_cohorts(cohort_ids, claim.entities.gene, spec, claim.endpoint)
            if not effects:
                rows.append(SpecCurveRow(
                    spec_id=spec.spec_id, grouping=spec.grouping,
                    contrast_type=spec.contrast_type, covariates=spec.covariates,
                    sample_scope=spec.sample_scope, followup_cap_years=spec.followup_cap_years,
                    k=0, supports_claim=False,
                ))
                continue
            aligned_y = [align_sign(claim.direction) * e.log_effect for e in effects]
            try:
                m = meta_random_effects(np.asarray(aligned_y), np.asarray([e.se for e in effects]))
                pooled_mu, pooled_se, pooled_p = m.pooled, m.se, m.p
            except ValueError:
                pooled_mu = float(np.median(aligned_y))
                pooled_se = float(np.median([e.se for e in effects]))
                pooled_p = float(2 * stats.norm.sf(abs(pooled_mu / pooled_se)))
            pooled_by_spec[spec.spec_id] = (pooled_mu, pooled_se, pooled_p, len(effects))
            supports = bool(pooled_mu > 0 and pooled_p < 0.05)
            rows.append(SpecCurveRow(
                spec_id=spec.spec_id, grouping=spec.grouping,
                contrast_type=spec.contrast_type, covariates=spec.covariates,
                sample_scope=spec.sample_scope, followup_cap_years=spec.followup_cap_years,
                pooled_log_effect=float(pooled_mu), se=float(pooled_se),
                p=float(pooled_p), k=len(effects), supports_claim=supports,
            ))
        used = [r for r in rows if r.pooled_log_effect is not None]
        frac = float(np.mean([r.supports_claim for r in used])) if used else 0.0
        scale_group = "median" if claim.reported_choices.contrast_type == "dichotomous" else "per_sd"
        scaled = [r.pooled_log_effect for r in used
                  if r.grouping == scale_group and r.pooled_log_effect is not None]
        if not scaled:
            scaled = [r.pooled_log_effect for r in used if r.pooled_log_effect is not None]
        median_m = float(np.median(scaled)) if scaled else None
        r = claim.reported_log_effect
        pct = percentile_rank(scaled, r) if scaled else None
        return MultiverseResult(
            n_specs_reference=len(specs),
            n_specs_used=len(used),
            frac_same_direction_sig=frac,
            median_log_effect=median_m,
            reported_spec_percentile=pct,
            n_specs_aligned=len(scaled),
            specs=rows,
        )

    def _confidence(self, claim: Claim, repro: OriginalCohortRepro, k: int,
                    flags: list[str]) -> str:
        conf = "low"
        if k >= int(self.gt_cfg["high_min_k"]) and repro.status in {"reproduced", "partial"}:
            conf = "high"
        elif k >= int(self.gt_cfg["medium_min_k"]):
            conf = "medium"
        if "out_of_standard_pool" in flags:
            conf = _min_conf(conf, str(self.gt_cfg["out_of_pool_confidence_cap"]))
        if repro.status in {"no_data", "out_of_pool"}:
            conf = _min_conf(conf, "medium" if k >= 2 else "low")
        if k == 1:
            conf = _min_conf(conf, "low")
        if "endpoint_not_in_data" in flags:
            conf = "low"
        return conf

    def evaluate_claim(self, claim: Claim, *, independent_role: str = "dev",
                       paper_cohort_id: str | None = None) -> GroundTruthResult:
        gene, cancer = claim.entities.gene, claim.entities.cancer_type
        flags: list[str] = []
        accessions = list(claim.reported_cohort.accessions)
        paper_cid = paper_cohort_id or self.analyzer.cohort_for_accessions(accessions)
        repro, paper_cid = self._original_repro(claim, paper_cid)
        if repro.status == "out_of_pool":
            flags.append("out_of_standard_pool")
        exclude = {paper_cid} if paper_cid else set()
        # Exclude every cohort named in the claim, not just the first accession.
        for accession in claim.reported_cohort.accessions:
            cid = self.analyzer.cohort_for_accessions([accession])
            if cid:
                exclude.add(cid)
        cohort_ids = self._independent(cancer, independent_role, exclude)
        if not cohort_ids:
            flags.append("no_independent_cohorts")
        default = default_spec(claim.reported_choices.contrast_type)
        effects: list[CohortEffect] = []
        for cid in cohort_ids:
            e = self.analyzer.effect(cid, gene, default, claim.endpoint)
            if e is not None and abs(e.log_effect) <= 6.0 and e.se <= 3.0:
                effects.append(_align_effect(e, claim.direction))
        independent = [
            IndependentCohortEffect(
                cohort_id=e.cohort_id, role=e.role, log_effect=float(e.log_effect),
                se=float(e.se), n=int(e.n), events=int(e.events), p=float(e.p),
                spec=default.spec_id,
            )
            for e in effects
        ]
        meta = None
        if effects:
            meta = meta_random_effects(
                np.asarray([e.log_effect for e in effects]),
                np.asarray([e.se for e in effects]),
            )
        multiverse = self._multiverse(claim, cohort_ids)
        r = claim.reported_log_effect
        se_r = claim.reported_se
        if meta is not None:
            mu, se_mu, tau2 = meta.pooled, meta.se, meta.tau2
        elif multiverse.median_log_effect is not None:
            mu = multiverse.median_log_effect
            aligned_rows = [x for x in multiverse.specs
                            if x.grouping == ("median" if claim.reported_choices.contrast_type == "dichotomous" else "per_sd")
                            and x.se is not None]
            se_mu = float(np.median([x.se for x in aligned_rows])) if aligned_rows else se_r
            tau2 = 0.0
            flags.append("no_independent_cohorts")
        else:
            mu, se_mu, tau2 = float("nan"), float("nan"), 0.0
        if math.isfinite(mu) and math.isfinite(se_mu) and se_mu > 0:
            independent_support = float(stats.norm.cdf(mu / se_mu))
            inflation = float(math.tanh(max(0.0, r - mu) / self.scale))
            denom = math.sqrt(se_r ** 2 + se_mu ** 2 + max(tau2, 0.0))
            inflation_z = float((r - mu) / denom) if denom > 0 else 0.0
        else:
            independent_support = 0.0
            inflation, inflation_z = 0.0, 0.0
        m = multiverse.median_log_effect
        shrinkage = clip01(m / r) if (m is not None and r > 0) else 0.0
        robustness = clip01(
            self.weights["independent_support"] * independent_support
            + self.weights["multiverse_support"] * multiverse.frac_same_direction_sig
            + self.weights["shrinkage"] * shrinkage
        )

        # Reconciliation flags (only pool-internal conclusions get this section).
        if not claim.data_availability.raw_deposited:
            flags.append("no_raw_data")
        if not claim.data_availability.code_available:
            flags.append("no_code")
        n_rep = claim.reported_cohort.n_reported
        n_dep = repro.n_deposited
        if n_rep and n_dep:
            if n_rep > n_dep:
                flags.append("n_reported_exceeds_deposited")
            elif n_rep < 0.9 * n_dep:
                flags.append("sample_count_gap")
        if claim.reported_cohort.subset_desc or claim.reported_choices.subset:
            flags.append("subset_not_prespecified")
        if claim.reported_choices.cutpoint == "optimal":
            flags.append("data_driven_cutpoint")
        if claim.reported_choices.cutpoint == "percentile_custom":
            flags.append("custom_percentile_cutpoint")
        if claim.reported_choices.time_cap_years not in (None, 5.0):
            flags.append("followup_cap_not_prespecified")
        if claim.reported_choices.n_comparisons and claim.reported_choices.n_comparisons > 10 \
                and claim.reported_choices.multiple_testing_correction is not True:
            flags.append("multiplicity_uncorrected")
        if claim.extraction_confidence < 0.7:
            flags.append("low_extraction_confidence")
        if claim.reported_effect.ci_low is None or claim.reported_effect.ci_high is None:
            flags.append("reported_ci_missing")
        if claim.reported_effect.se_derived_from_p:
            flags.append("se_derived_from_p")
        if (claim.direction in {"high_expression_worse", "low_expression_better"}
                and claim.reported_effect.value < 1.0):
            flags.append("direction_effect_inconsistent")
        if (claim.direction in {"high_expression_better", "low_expression_worse"}
                and claim.reported_effect.value > 1.0):
            flags.append("direction_effect_inconsistent")
        check_cid = cohort_ids[0] if cohort_ids else paper_cid
        if check_cid is None:
            flags.append("endpoint_not_in_data")
        else:
            try:
                clinical_check = self.analyzer.store.clinical(check_cid)
                tcol = f"{claim.endpoint.lower()}_time_days"
                ecol = f"{claim.endpoint.lower()}_event"
                if tcol not in clinical_check.columns or ecol not in clinical_check.columns:
                    flags.append("endpoint_not_in_data")
            except Exception:
                flags.append("endpoint_not_in_data")
        if len(effects) == 1:
            flags.append("single_independent_cohort")

        conf = self._confidence(claim, repro, len(effects), flags)
        notes: list[str] = []
        if meta is None:
            notes.append("No independent cohort meta-analysis; inflation uses multiverse median where available.")
        if repro.status != "reproduced":
            notes.append(f"Original-cohort reproduction: {repro.status} ({repro.reason or 'no reason recorded'}).")
        # A label is eligible only when there is an independent meta-analysis;
        # low confidence is handled by the training weight, not by turning
        # missing evidence into a negative class.
        label_eligible = meta is not None
        return GroundTruthResult(
            claim_id=claim.claim_id,
            engine_version=str(self.cfg.scoring.get("engine_version", "gt-1.1.0")),
            data_snapshot=self.cfg.snapshot,
            contrast_type=claim.reported_choices.contrast_type,
            gene=gene,
            cancer_type=cancer,
            endpoint=claim.endpoint,
            reported_log_effect=float(r),
            reported_se=float(se_r),
            alpha=claim.reported_effect.p,
            original_cohort_repro=repro,
            independent_cohorts=independent,
            meta=_meta_result(meta) if meta is not None else None,
            multiverse=multiverse,
            reconciliation_flags=sorted(set(flags)),
            robustness_score=robustness,
            inflation=clip01(inflation),
            inflation_z=inflation_z,
            gt_confidence=conf,
            weights_used=dict(self.weights),
            label_eligible=label_eligible,
            notes=notes,
        )
