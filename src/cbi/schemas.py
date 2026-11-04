"""Typed schemas (spec §4).  `extra="forbid"` is deliberate: a red-team
submission carrying a context/metadata field must be rejected at parse time
(C7), not silently ignored."""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

ClaimType = Literal["prognostic_gene", "prognostic_signature"]
ContrastType = Literal["dichotomous", "continuous"]
Endpoint = Literal["OS", "DSS", "PFI", "DFS"]
Direction = Literal[
    "high_expression_worse",
    "high_expression_better",
    "low_expression_worse",
    "low_expression_better",
]
Cutpoint = Literal[
    "median",
    "tertile",
    "optimal",
    "percentile_custom",
    "per_sd",
    "per_unit",
    "unspecified",
    "other",
]
ReproStatus = Literal["reproduced", "partial", "failed", "no_data", "out_of_pool"]
GTConfidence = Literal["high", "medium", "low"]
Role = Literal["dev", "visible_eval", "heldout"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), str_strip_whitespace=True)


class SourceRef(StrictModel):
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    pub_date: date
    journal: str | None = None
    title: str | None = None


class Entities(StrictModel):
    gene: str
    hgnc_id: str | None = None
    cancer_type: str
    tissue: Literal["tumor", "normal", "both"] = "tumor"


class ReportedEffect(StrictModel):
    metric: Literal["HR", "OR", "RR"] = "HR"
    value: float = Field(gt=0)
    ci_low: float | None = Field(default=None, gt=0)
    ci_high: float | None = Field(default=None, gt=0)
    p: float | None = Field(default=None, ge=0, le=1)
    se: float | None = Field(default=None, gt=0)
    se_derived_from_p: bool = False

    @model_validator(mode="after")
    def _derive_se(self) -> "ReportedEffect":
        if self.ci_low is not None and self.ci_high is not None:
            if not (self.ci_low < self.value < self.ci_high):
                raise ValueError("effect value must lie inside its confidence interval")
            if self.se is None:
                self.se = (math.log(self.ci_high) - math.log(self.ci_low)) / (2 * 1.959963984540054)
        elif self.p is not None and self.se is None:
            if self.p <= 0:
                # Report of p=0 is common in papers; use a conservative floor.
                z = 6.0
            else:
                z = abs(float(stats.norm.ppf(self.p / 2.0)))
                z = max(z, 0.05)
            self.se = abs(math.log(self.value)) / z
            self.se_derived_from_p = True
        if self.se is not None and self.se <= 0:
            raise ValueError("derived SE must be positive")
        return self


class ReportedCohort(StrictModel):
    accessions: list[str] = Field(default_factory=list)
    in_standard_pool: bool = False
    n_reported: int | None = Field(default=None, gt=0)
    subset_desc: str | None = None


class ReportedChoices(StrictModel):
    contrast_type: ContrastType
    cutpoint: Cutpoint = "unspecified"
    normalization: str = "platform_native"
    covariates: list[str] = Field(default_factory=list)
    time_cap_years: float | None = Field(default=None, gt=0)
    subset: str | None = None
    probe_selection: str | None = None
    n_comparisons: int | None = Field(default=None, ge=1)
    multiple_testing_correction: bool | None = None
    selection_note: str | None = None

    @model_validator(mode="after")
    def _contrast_matches_cutpoint(self) -> "ReportedChoices":
        if self.cutpoint == "per_sd" and self.contrast_type != "continuous":
            raise ValueError("per_sd cutpoint requires continuous contrast")
        if self.cutpoint in {"median", "tertile", "optimal", "percentile_custom"} and self.contrast_type != "dichotomous":
            raise ValueError("grouped cutpoint requires dichotomous contrast")
        return self


class DataAvailability(StrictModel):
    raw_deposited: bool = False
    code_available: bool = False
    deposited_n: int | None = Field(default=None, gt=0)
    notes: str | None = None


class EvidenceSpan(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    sentence_hash: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "EvidenceSpan":
        if self.end < self.start:
            raise ValueError("evidence span end precedes start")
        return self


class Claim(StrictModel):
    claim_id: str
    source: SourceRef
    claim_type: ClaimType = "prognostic_gene"
    entities: Entities
    endpoint: Endpoint
    direction: Direction
    reported_effect: ReportedEffect
    reported_cohort: ReportedCohort = Field(default_factory=ReportedCohort)
    reported_choices: ReportedChoices
    data_availability: DataAvailability = Field(default_factory=DataAvailability)
    extraction_confidence: float = Field(default=0.0, ge=0, le=1)
    evidence_span: EvidenceSpan | None = None
    extraction_method: str | None = None

    @property
    def as_of_date(self) -> date:
        return self.source.pub_date

    @property
    def reported_log_effect(self) -> float:
        """Direction-aligned log effect: positive means 'as claimed'."""
        raw = math.log(self.reported_effect.value)
        if self.direction in {"high_expression_worse", "low_expression_better"}:
            return raw
        return -raw

    @property
    def reported_se(self) -> float:
        assert self.reported_effect.se is not None
        return float(self.reported_effect.se)


class OriginalCohortRepro(StrictModel):
    status: ReproStatus
    log_effect: float | None = None
    se: float | None = None
    p: float | None = None
    n_used: int | None = None
    n_deposited: int | None = None
    reason: str | None = None


class IndependentCohortEffect(StrictModel):
    cohort_id: str
    role: str = "dev"
    log_effect: float
    se: float
    n: int
    events: int
    p: float | None = None
    spec: str = "reference_dichotomous"


class MetaResult(StrictModel):
    method: str
    pooled_log_effect: float
    se: float
    tau2: float
    pred_interval: tuple[float, float]
    k: int
    q: float | None = None
    i2: float | None = None
    p: float | None = None


class SpecCurveRow(StrictModel):
    spec_id: str
    grouping: str
    contrast_type: ContrastType
    covariates: str
    sample_scope: str
    followup_cap_years: float | None
    pooled_log_effect: float | None = None
    se: float | None = None
    p: float | None = None
    k: int
    supports_claim: bool


class MultiverseResult(StrictModel):
    n_specs_reference: int
    n_specs_used: int
    frac_same_direction_sig: float
    median_log_effect: float | None
    reported_spec_percentile: float | None = None
    n_specs_aligned: int = 0
    spec_curve_uri: str | None = None
    specs: list[SpecCurveRow] = Field(default_factory=list)


class GroundTruthResult(StrictModel):
    claim_id: str
    engine_version: str = "gt-1.1.0"
    data_snapshot: str
    contrast_type: ContrastType
    gene: str
    cancer_type: str
    endpoint: Endpoint
    reported_log_effect: float
    reported_se: float
    alpha: float | None = None
    original_cohort_repro: OriginalCohortRepro
    independent_cohorts: list[IndependentCohortEffect]
    meta: MetaResult | None
    multiverse: MultiverseResult
    reconciliation_flags: list[str]
    robustness_score: float = Field(ge=0, le=1)
    inflation: float = Field(ge=0, le=1)
    inflation_z: float
    gt_confidence: GTConfidence
    weights_used: dict[str, float]
    label_eligible: bool = True
    notes: list[str] = Field(default_factory=list)


class ContextFeatures(StrictModel):
    claim_id: str
    as_of_date: date
    author_cluster_id: str | None = None
    supporting_papers: int | None = None
    effective_independent_support: float | None = None
    citation_cascade_score: float | None = None
    lab_historical_replication_rate: float | None = None
    lab_historical_n: int | None = None
    lab_historical_method: Literal["leave_one_out"] = "leave_one_out"
    self_citation_ratio: float | None = None
    coi_flags: list[str] = Field(default_factory=list)
    review_time_days: int | None = None
    venue_features: dict[str, Any] = Field(default_factory=dict)
    post_pub_signals: dict[str, Any] = Field(default_factory=dict)
    missing_mask: dict[str, bool] = Field(default_factory=dict)

    @property
    def context_coverage(self) -> float:
        keys = [
            "author_cluster_id",
            "supporting_papers",
            "effective_independent_support",
            "citation_cascade_score",
            "lab_historical_replication_rate",
            "self_citation_ratio",
            "review_time_days",
        ]
        if not keys:
            return 0.0
        present = 0
        for key in keys:
            value = getattr(self, key)
            if value is not None and not self.missing_mask.get(key, False):
                present += 1
        return present / len(keys)


class DetectorOutput(StrictModel):
    claim_id: str
    detector_version: str
    evidence_score: float = Field(ge=0, le=1)
    context_prior: float | None = Field(default=None, ge=0, le=1)
    context_coverage: float = Field(default=0.0, ge=0, le=1)
    combined_inflation_pred: float = Field(ge=0, le=1)
    flag: bool
    threshold_used: float
    rationale: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)


class Report(StrictModel):
    report_id: str
    claim: Claim
    manuscript_md_uri: str
    analysis_code_uri: str
    code_sha256: str
    declared_data: list[str]
    declared_choices: ReportedChoices
    reported_numbers: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _choices_match(self) -> "Report":
        if self.declared_choices.contrast_type != self.claim.reported_choices.contrast_type:
            raise ValueError("declared contrast_type does not match claim")
        return self


class Submission(StrictModel):
    submission_id: str
    round: int = Field(ge=0)
    worker: int = Field(ge=0)
    strategy_family: str
    parent_ids: list[str] = Field(default_factory=list)
    preregistration_text: str
    preregistration_hash: str
    reports: list[Report] = Field(min_length=1)
    worker_trace_uri: str | None = None
    created_at: str
    declared_choices: dict[str, Any] = Field(default_factory=dict)


class ExperimentLogEntry(StrictModel):
    submission_id: str
    round: int
    worker: int
    strategy_family: str
    summary: str
    score: float
    flag_rate: float
    visible_inflation: float | None
    rejected_by: str | None = None
    realism_pass: bool | None = None
    cheat_check: str = "pass"
    detector_version: str
    timestamp: str


class StrategyPrereg(StrictModel):
    submission_id: str
    round: int
    worker: int
    strategy_family: str
    choices: dict[str, Any]
    target_cancers: list[str]
    target_cohorts: list[str]
    motivation: str
    reports_planned: int
    created_at: str


class CohortMeta(StrictModel):
    cohort_id: str
    cancer_type: str
    accession: str | None
    role: Role
    n: int
    n_deposited: int
    platform: str
    expr_unit: str
    treatment_era: str | None = None
    source: str = "public"
    license: str = "unknown"
    subset_desc: str | None = None
    notes: str | None = None


def claim_from_dict(obj: dict[str, Any]) -> Claim:
    return Claim.model_validate(obj)


def result_from_dict(obj: dict[str, Any]) -> GroundTruthResult:
    return GroundTruthResult.model_validate(obj)
