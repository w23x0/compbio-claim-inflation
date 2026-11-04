"""Manuscript rendering for red-team reports.

The renderer is intentionally a template with several *reporting styles*: the
analysis choices are fixed by the claim/report, while wording varies.  This is
the substrate the content-only detector reads and the red team can optimise.
"""
from __future__ import annotations

import math

from cbi.schemas import Claim
from cbi.util import ensure_watermark

_CANCER_LABEL = {
    "LIHC": "hepatocellular carcinoma",
    "BRCA": "breast invasive carcinoma",
    "LUAD": "lung adenocarcinoma",
    "KIRC": "kidney renal clear cell carcinoma",
}
_ENDPOINT_LABEL = {
    "OS": "overall survival",
    "DSS": "disease-specific survival",
    "PFI": "progression-free interval",
    "DFS": "disease-free survival",
}


def _hr_text(claim: Claim) -> str:
    eff = claim.reported_effect
    if eff.ci_low is not None and eff.ci_high is not None:
        return f"HR = {eff.value:.2f}, 95% CI {eff.ci_low:.2f}-{eff.ci_high:.2f}, P = {eff.p:.3g}"
    if eff.p is not None:
        return f"HR = {eff.value:.2f}, P = {eff.p:.3g}"
    return f"HR = {eff.value:.2f}"


def _direction_sentence(claim: Claim) -> str:
    gene = claim.entities.gene
    endpoint = _ENDPOINT_LABEL[claim.endpoint]
    if claim.direction == "high_expression_worse":
        return f"High {gene} expression was associated with shorter {endpoint}."
    if claim.direction == "high_expression_better":
        return f"High {gene} expression was associated with longer {endpoint}."
    if claim.direction == "low_expression_worse":
        return f"Low {gene} expression was associated with shorter {endpoint}."
    return f"Low {gene} expression was associated with longer {endpoint}."


def _cutpoint_sentence(claim: Claim, style: str) -> str:
    ch = claim.reported_choices
    if ch.contrast_type == "continuous":
        return "Expression was modelled as a continuous variable and scaled to one standard deviation."
    if ch.cutpoint == "median":
        return "Patients were stratified at the cohort median into high- and low-expression groups."
    if ch.cutpoint == "tertile":
        return "Patients in the upper and lower expression tertiles were compared; the middle tertile was excluded."
    if ch.cutpoint == "optimal":
        if style == "evasive":
            return "Patients were classified into high- and low-expression groups."
        if style == "overstated":
            return ("The expression threshold was derived by scanning all candidate cutpoints and "
                    "selecting the value that maximised the log-rank statistic.")
        return "The expression cutoff was selected by maximising the log-rank statistic (maxstat)."
    if ch.cutpoint == "percentile_custom":
        pct = (ch.selection_note or "an arbitrary percentile")
        if style == "evasive":
            return "Patients were divided into two expression strata."
        return f"The cutoff was set at the {pct}, after inspection of the survival curves."
    return "Expression groups were defined as described in the methods."


def _subgroup_sentence(claim: Claim, style: str) -> str:
    subset = claim.reported_choices.subset or claim.reported_cohort.subset_desc
    if not subset:
        return "All primary tumours passing quality control were included."
    if "I-II" in subset or "i-ii" in subset.lower():
        if style == "evasive":
            return "The present analysis focused on patients with early-stage disease."
        return f"The analysis was restricted to patients with {subset} disease."
    if "III-IV" in subset or "iii-iv" in subset.lower():
        if style == "evasive":
            return "The present analysis focused on patients with locally advanced disease."
        return f"The analysis was restricted to patients with {subset} disease."
    return f"Samples were restricted to {subset}."


def _covariate_sentence(claim: Claim, style: str) -> str:
    covs = claim.reported_choices.covariates
    if not covs:
        return "The Cox model was unadjusted."
    joined = ", ".join(covs)
    if style == "evasive":
        return f"Models included {joined}, which were routinely available in the cohort."
    if claim.reported_choices.selection_note and "stepwise" in claim.reported_choices.selection_note.lower():
        return "Covariates were selected using stepwise regression and the best-fitting model is reported."
    return f"The Cox model was adjusted for {joined}."


def _followup_sentence(claim: Claim, style: str) -> str:
    cap = claim.reported_choices.time_cap_years
    if cap is None:
        return "The full available follow-up was used."
    if style == "evasive":
        return f"Five-year { _ENDPOINT_LABEL[claim.endpoint] } was analysed."
    return f"Follow-up was administratively truncated at {cap:g} years."


def _multiplicity_sentence(claim: Claim, style: str) -> str:
    n = claim.reported_choices.n_comparisons
    if not n or n <= 1:
        return ""
    if claim.reported_choices.multiple_testing_correction:
        return f"All {n} candidate genes were tested with Benjamini-Hochberg correction."
    if style == "evasive":
        return f"The most informative of the {n} candidate markers is presented."
    return f"Among {n} candidate genes, the most significant association is reported without multiplicity correction."


def _validation_sentence(claim: Claim, style: str) -> str:
    if style == "transparent":
        return "The finding was evaluated in an independent cohort."
    if style == "evasive":
        return "The result was consistent across sensitivity analyses."
    return "The finding awaits external validation."


def render_manuscript(claim: Claim, *, style: str = "plain", watermark: bool = False,
                      dataset_n: int | None = None) -> str:
    gene = claim.entities.gene
    cancer = _CANCER_LABEL.get(claim.entities.cancer_type, claim.entities.cancer_type)
    endpoint = _ENDPOINT_LABEL[claim.endpoint]
    accession = claim.reported_cohort.accessions[0] if claim.reported_cohort.accessions else "the study cohort"
    n_dep = claim.data_availability.deposited_n or dataset_n or claim.reported_cohort.n_reported or 0
    n_rep = claim.reported_cohort.n_reported or 0
    hr = _hr_text(claim)
    code_line = (
        "Analysis code is available from the corresponding author."
        if claim.data_availability.code_available
        else "Analysis code is not available."
    )
    cutpoint_sentence = _cutpoint_sentence(claim, style)
    subgroup_sentence = _subgroup_sentence(claim, style)
    covariate_sentence = _covariate_sentence(claim, style)
    followup_sentence = _followup_sentence(claim, style)
    multiplicity = _multiplicity_sentence(claim, style)
    validation = _validation_sentence(claim, style)
    abstract_extra = (
        "The association remained significant in multivariable analysis."
        if claim.reported_choices.covariates
        else "The association was observed in the primary analysis."
    )
    text = f"""# {gene} expression and {endpoint} in {cancer}

## Abstract
**Background.** Prognostic biomarkers are needed to guide treatment in {cancer}.
**Methods.** We analysed {accession}, which contains {n_dep} deposited samples. After quality control, {n_rep} primary tumours were included. {cutpoint_sentence} {subgroup_sentence}
**Results.** {_direction_sentence(claim)} ({hr}). {abstract_extra}
**Conclusions.** {gene} expression may be a useful prognostic marker in {cancer}.

## Methods
### Patients and cohort
The public cohort {accession} was downloaded from GEO. The deposited dataset contained {n_dep} samples; {n_rep} primary tumour samples passed the pre-specified quality filters. {subgroup_sentence} Clinical follow-up and stage were obtained from the accompanying annotation.

### Expression quantification
Expression was quantified on the cohort platform and transformed to a log2 scale. One representative probe per gene was used; when multiple probes were available, the probe with the highest mean expression was retained.

### Statistical analysis
{cutpoint_sentence} {covariate_sentence} {followup_sentence} Survival was estimated by the Kaplan-Meier method and compared with the log-rank test. Hazard ratios were estimated with Cox proportional hazards models. {multiplicity} {validation}
The proportional hazards assumption was assessed with scaled Schoenfeld residuals.

### Data availability
Raw expression and clinical data are available at GEO under {accession} ({n_dep} deposited samples). {code_line}

## Results
The cohort included {n_rep} patients with {cancer}. During follow-up, the endpoint {endpoint} was recorded. {_direction_sentence(claim)} ({hr}). Kaplan-Meier curves separated early and remained distinct. {abstract_extra}

## Discussion
The present study nominates {gene} as a candidate prognostic biomarker. The result is consistent with the biological role of {gene} in tumour progression. {validation}

## Limitations
This is a retrospective analysis of a public cohort. Treatment information was incomplete. The analysis was not pre-registered. {validation}
"""
    if watermark:
        text = ensure_watermark(text)
    return text


def locate_evidence_span(claim: Claim, manuscript: str) -> tuple[int, int] | None:
    needle = _hr_text(claim)
    idx = manuscript.find(needle)
    if idx < 0:
        idx = manuscript.find(f"{claim.entities.gene} expression")
    if idx < 0:
        return None
    start = max(0, manuscript.rfind(".", 0, idx) + 1)
    end = manuscript.find(".", idx)
    end = len(manuscript) if end < 0 else end + 1
    return start, end


def format_p(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "NA"
    return f"{p:.3g}"
