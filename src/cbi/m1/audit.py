"""Automatic consistency audit for extracted claims (spec M1 v1.2).

This replaces the human spot-check.  It cannot measure semantic precision
without labels, but it catches the failure modes that make a claim unusable
downstream: unresolved genes, impossible CIs, CI/p disagreement, bad
accessions, and malformed evidence spans.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

from scipy import stats

from cbi.m1.geo import geo_exists
from cbi.schemas import Claim

GEO_RE = re.compile(r"^GSE\d{3,}$")
TCGA_RE = re.compile(r"^TCGA-[A-Z0-9]{2,8}$")
ARRAYEXPRESS_RE = re.compile(r"^E-[A-Z]+-\d+$")
TCGA_CODES = {
    "LIHC", "BRCA", "LUAD", "KIRC", "COAD", "READ", "STAD", "PRAD", "PAAD",
    "OV", "SKCM", "BLCA", "THCA", "GBM", "LGG", "HNSC", "ESCA", "UCEC",
}


def audit_claim(claim: Claim, *, known_accessions: set[str] | None = None,
                allowed_cancers: set[str] | None = None, resolver: Any = None) -> list[str]:
    issues: list[str] = []
    if not claim.entities.gene:
        issues.append("missing_gene")
    if allowed_cancers is not None and claim.entities.cancer_type not in allowed_cancers:
        issues.append("cancer_out_of_scope")
    elif claim.entities.cancer_type not in TCGA_CODES:
        issues.append("unmapped_cancer_type")
    if claim.entities.hgnc_id is not None:
        if not claim.entities.hgnc_id.startswith("HGNC:"):
            issues.append("bad_hgnc_id")
        elif resolver is not None:
            symbol = resolver.id_to_symbol.get(claim.entities.hgnc_id)
            if symbol is None:
                issues.append("unknown_hgnc_id")
            elif symbol != claim.entities.gene.upper():
                issues.append("hgnc_symbol_mismatch")
    if claim.reported_effect.metric != "HR":
        issues.append("non_hr_effect")
    choice_text = " ".join([
        claim.reported_choices.normalization or "",
        claim.reported_choices.probe_selection or "",
        claim.reported_choices.selection_note or "",
        claim.reported_choices.subset or "",
        claim.reported_cohort.subset_desc or "",
    ]).lower()
    if any(term in choice_text for term in [
        "ihc", "immunohist", "mutation", "mutat", "cnv", "copy number",
        "methylation", "proteom", "elisa", "serum", "plasma",
    ]):
        issues.append("non_expression_assay")
    if any(term in choice_text for term in [
        "meta-analysis", "meta analysis", "systematic review", "scoping review",
        "pan-cancer", "pan cancer", "multiple cancer", "multiple cancers",
        "across 17 cancer", "studies across",
    ]):
        issues.append("review_or_pancancer")
    eff = claim.reported_effect
    if not math.isfinite(eff.value) or eff.value <= 0:
        issues.append("nonpositive_effect")
    if eff.ci_low is not None and eff.ci_high is not None:
        if not (eff.ci_low < eff.value < eff.ci_high):
            issues.append("effect_outside_ci")
        if eff.p is not None:
            excludes_null = eff.ci_low > 1.0 or eff.ci_high < 1.0
            if excludes_null != (eff.p < 0.05):
                issues.append("ci_p_inconsistent")
    if eff.p is not None and not (0.0 <= eff.p <= 1.0):
        issues.append("p_out_of_range")
    high_worse = claim.direction in {"high_expression_worse", "low_expression_better"}
    if abs(math.log(eff.value)) > 1e-9:
        agrees = (eff.value > 1.0) == high_worse
        if not agrees:
            issues.append("direction_effect_disagreement")
    for accession in claim.reported_cohort.accessions:
        if TCGA_RE.match(accession) or ARRAYEXPRESS_RE.match(accession):
            continue
        if not GEO_RE.match(accession):
            issues.append(f"bad_accession:{accession}")
            continue
        if known_accessions is not None and accession not in known_accessions:
            issues.append(f"accession_not_in_pool:{accession}")
        cached = geo_exists(accession)
        if cached is False:
            issues.append(f"accession_not_in_cache:{accession}")
    if claim.evidence_span is not None and claim.evidence_span.end < claim.evidence_span.start:
        issues.append("bad_evidence_span")
    if not (0.0 <= claim.extraction_confidence <= 1.0):
        issues.append("bad_confidence")
    return issues


def audit_claims(claims: Iterable[Claim], *, known_accessions: set[str] | None = None,
                 allowed_cancers: set[str] | None = None, resolver: Any = None) -> dict[str, Any]:
    rows = [audit_claim(c, known_accessions=known_accessions, allowed_cancers=allowed_cancers,
                        resolver=resolver) for c in claims]
    passed = sum(1 for issues in rows if not issues)
    counts: dict[str, int] = {}
    for issues in rows:
        for issue in issues:
            key = issue.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return {
        "n": len(rows),
        "passed": passed,
        "pass_rate": float(passed / len(rows)) if rows else 0.0,
        "issue_counts": dict(sorted(counts.items())),
    }
