"""Monitor-document assembly and red-flag rubric (spec §M4)."""
from __future__ import annotations

from datetime import date

from cbi.m1.pmc import PMCArticle
from cbi.m4.llm_judge import FLAG_KEYS, rubric_score
from cbi.m4.monitor_doc import build_monitor_document
from cbi.schemas import (Claim, DataAvailability, Entities, ReportedChoices,
                         ReportedCohort, ReportedEffect, SourceRef)


def _claim() -> Claim:
    return Claim(
        claim_id="c1", source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene="TP53", cancer_type="LIHC"), endpoint="OS",
        direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8),
        reported_cohort=ReportedCohort(accessions=["TCGA-LIHC"], n_reported=300),
        reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="unspecified"),
        data_availability=DataAvailability(),
    )


def _article(filler: str = "") -> PMCArticle:
    return PMCArticle(
        pmcid="PMC1", pmid=None, doi=None, title="t", journal=None,
        pub_date=date(2020, 1, 1),
        abstract="Purpose: to test. Methods: cohort study. Results: HR 2.0.",
        body="",
        full_text="",
        tables_text=(
            "Table 1 Univariate Cox analysis\n"
            "Gene: TP53; HR: 2.0; 95% CI: 1.4-2.8; p: 0.001\n"
            "Gene: EGFR; HR: 1.1; 95% CI: 0.9-1.3; p: 0.4"
        ),
        body_sections={
            "Materials and methods": "Patients were split at the median TP53 expression."
                                       + filler,
            "Results": ("Unrelated paragraph about cell culture handling. "
                        "TP53 high expression was associated with worse OS (HR 2.0, 95% CI 1.4-2.8)."),
            "Data availability statement": "No code or raw data were deposited.",
        },
    )


def test_methods_survive_even_when_the_body_prefix_would_cut_them():
    article = _article(filler=" " + "Padding sentence. " * 2000)
    doc = build_monitor_document(article, _claim(), max_chars=14000)
    assert "split at the median" in doc
    assert "No code or raw data" in doc
    assert len(doc) <= 14000


def test_results_keep_claim_relevant_rows_and_drop_noise():
    doc = build_monitor_document(_article(), _claim())
    assert "TP53 high expression was associated" in doc
    assert "cell culture handling" not in doc
    assert "TP53; HR: 2.0" in doc
    assert "EGFR" not in doc


def test_rubric_is_weighted_flag_fraction_and_needs_enough_answers():
    flags = {key: True for key in FLAG_KEYS}
    assert rubric_score(flags) == 1.0
    assert rubric_score({key: False for key in FLAG_KEYS}) == 0.0
    assert rubric_score({key: False for key in FLAG_KEYS[:5]}) is None
    partial = {key: False for key in FLAG_KEYS}
    partial["cutpoint_not_prespecified"] = True
    assert 0.0 < rubric_score(partial) < 1.0
