"""Section-aware monitor document for the M4 content judge (spec §M4).

The spec defines the detector input as the paper's conclusion, methods and data
availability (plus the claim itself), not the first N characters of the file.
PMC body text is a bag of paragraphs in ``PMCArticle.body``; taking its prefix
shows only abstract + introduction, which hides exactly the checklist items the
judge has to decide (cutpoint selection, subset selection, multiplicity,
validation cohort, data/code availability, CI/p consistency).

``build_monitor_document`` therefore assembles a bounded document from the
titled body sections and the table rows that mention the claim's gene.
"""
from __future__ import annotations

import json
import os
import re

from cbi.m1.pmc import PMCArticle
from cbi.schemas import Claim

DEFAULT_MAX_CHARS = int(os.environ.get("CBI_MONITOR_MAX_CHARS", "14000"))

_METHOD_TITLES = ("method", "material", "statistical", "study design", "data source",
                  "patients", "cohort", "sample")
_RESULTS_TITLES = ("result",)
_DATA_TITLES = ("data availability", "availability of data", "data and code",
                "availability of data and material")
_CONCLUSION_TITLES = ("conclusion", "discussion")

_STAT_TERMS = ("hazard ratio", "hr=", "hr =", "95% ci", "confidence interval",
               "multivariate", "univariate", "cutoff", "cut-off", "cut point",
               "dichotom", "median", "tertile", "log-rank", "cox", "p<", "p <")


def _gene_pattern(gene: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])", re.IGNORECASE)


def _claim_block(claim: Claim) -> str:
    eff = claim.reported_effect
    ci = ""
    if eff.ci_low is not None and eff.ci_high is not None:
        ci = f" (95% CI {eff.ci_low}-{eff.ci_high})"
    p = f", p={eff.p}" if eff.p is not None else ""
    choices = claim.reported_choices
    availability = claim.data_availability
    lines = [
        f"claim_id: {claim.claim_id}",
        f"gene: {claim.entities.gene}",
        f"cancer: {claim.entities.cancer_type}",
        f"endpoint: {claim.endpoint}",
        f"direction: {claim.direction}",
        f"reported_effect: {eff.metric} {eff.value}{ci}{p}",
        f"reported_cohort: accessions={claim.reported_cohort.accessions}, "
        f"n_reported={claim.reported_cohort.n_reported}, "
        f"subset={claim.reported_cohort.subset_desc}",
        f"reported_choices: contrast={choices.contrast_type}, cutpoint={choices.cutpoint}, "
        f"covariates={choices.covariates}, subset={choices.subset}, "
        f"n_comparisons={choices.n_comparisons}, "
        f"multiple_testing_correction={choices.multiple_testing_correction}",
        f"claimed_data_availability: raw_deposited={availability.raw_deposited}, "
        f"code_available={availability.code_available}, "
        f"deposited_n={availability.deposited_n}",
    ]
    return "## CLAIM (structured extraction)\n" + "\n".join(lines)


def _matching_titles(sections: dict[str, str], needles: tuple[str, ...]) -> list[str]:
    return [title for title in sections
            if any(needle in title.lower() for needle in needles)]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _select_results(results: str, gene: str, *, limit: int = 40,
                    max_chars: int = 6000) -> str:
    pattern = _gene_pattern(gene)
    picked: list[str] = []
    seen: set[str] = set()
    for sentence in _sentences(results):
        low = sentence.lower()
        if pattern.search(sentence) or any(term in low for term in _STAT_TERMS):
            if sentence not in seen:
                seen.add(sentence)
                picked.append(sentence)
        if len(picked) >= limit or sum(len(s) for s in picked) >= max_chars:
            break
    return " ".join(picked)


def _select_table_rows(tables: str, gene: str, *, limit: int = 20) -> str:
    pattern = _gene_pattern(gene)
    rows = [row for row in tables.splitlines() if pattern.search(row)]
    return "\n".join(rows[:limit])


def _trim(text: str, limit: int) -> str:
    """Cut to ``limit`` characters at a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    head, sep, _ = cut.rpartition(". ")
    return (head + ".") if sep and len(head) > limit // 2 else cut


def build_monitor_document(article: PMCArticle, claim: Claim,
                           *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return the bounded monitor input for one claim.

    Priority: claim, abstract, methods, data availability, claim-relevant
    results, conclusions, gene rows.  The data-availability statement is
    reserved before methods are fitted, so a long methods section cannot push
    out the availability evidence the checklist asks about.
    """
    sections = article.body_sections or {}
    method_text = " ".join(sections[t] for t in _matching_titles(sections, _METHOD_TITLES))
    results_text = " ".join(sections[t] for t in _matching_titles(sections, _RESULTS_TITLES))
    data_text = " ".join(sections[t] for t in _matching_titles(sections, _DATA_TITLES))
    conclusion_text = " ".join(sections[t] for t in _matching_titles(sections, _CONCLUSION_TITLES))

    head = "\n\n".join(part for part in (
        _claim_block(claim),
        "## ABSTRACT\n" + article.abstract if article.abstract.strip() else "",
    ) if part)
    data_block = "## DATA AVAILABILITY STATEMENT\n" + data_text if data_text.strip() else ""
    results_block = "## RESULTS (claim-relevant paragraphs)\n" + _select_results(
        results_text, claim.entities.gene) if results_text.strip() else ""
    conclusion_block = ""
    if conclusion_text.strip():
        conclusion_block = "## CONCLUSIONS\n" + " ".join(_sentences(conclusion_text)[:8])
    table_block = ""
    if article.tables_text.strip():
        rows = _select_table_rows(article.tables_text, claim.entities.gene)
        table_block = "## TABLE ROWS MENTIONING THE GENE\n" + rows if rows else ""

    reserved = len(data_block) if data_block else 0
    methods_room = max(0, max_chars - len(head) - reserved - 4)
    methods_block = ""
    if method_text.strip() and methods_room > 200:
        methods_block = "## MATERIALS AND METHODS\n" + _trim(method_text, methods_room - 40)

    blocks = [head, methods_block, data_block]
    for block in (results_block, conclusion_block, table_block):
        if block:
            blocks.append(block)
    doc = ""
    for block in blocks:
        candidate = (doc + "\n\n" if doc else "") + block
        if len(candidate) <= max_chars:
            doc = candidate
    return doc
