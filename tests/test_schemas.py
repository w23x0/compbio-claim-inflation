from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from cbi.schemas import (Claim, DataAvailability, Entities, ReportedChoices,
                         ReportedCohort, ReportedEffect, SourceRef, Submission)


def test_se_derived_from_ci_and_p():
    e = ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8, p=0.001)
    assert e.se is not None and e.se > 0
    e2 = ReportedEffect(metric="HR", value=2.0, p=0.01)
    assert e2.se_derived_from_p is True
    assert e2.se is not None


def test_claim_rejects_bad_ci():
    with pytest.raises(ValidationError):
        ReportedEffect(metric="HR", value=2.0, ci_low=2.2, ci_high=3.0)


def test_submission_rejects_context_metadata():
    claim = Claim(
        claim_id="c1", source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene="GENE001", cancer_type="LIHC"), endpoint="OS",
        direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=2.0, ci_low=1.4, ci_high=2.8),
        reported_cohort=ReportedCohort(accessions=["GSE1"]),
        reported_choices=ReportedChoices(contrast_type="dichotomous", cutpoint="median"),
        data_availability=DataAvailability(),
    )
    with pytest.raises(ValidationError):
        Submission(submission_id="s", round=1, worker=1, strategy_family="x",
                   preregistration_text="{}", preregistration_hash="sha256:x",
                   reports=[], created_at="now", context_metadata={})
