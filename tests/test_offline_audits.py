"""Offline audits should detect leakage and preserve statistical definitions."""
import numpy as np
import pytest
from scipy import stats

from scripts.audit_corpus import audit_corpus
from scripts.analyze_m2_null import compare_row, support_variants
from cbi.util import read_json
from test_p4_offline import local_corpus


def test_paper_aliases_are_transitive_and_doi_prefixes_normalized(local_corpus):
    cfg, _ = local_corpus
    base = cfg.paths.processed / "corpus"
    claims = read_json(base / "claims.json")
    claims[0]["source"].update(doi="https://doi.org/10.123/ABC", pmcid="PMC1")
    claims[2]["source"].update(doi="doi:10.123/abc", pmid="123")
    claims[4]["source"].update(pmid="123")
    report = audit_corpus(claims, read_json(base / "natural_split.json")["by_claim"],
                          read_json(cfg.paths.processed / "ground_truth.json"), cfg.cohort_pool,
                          cfg.scoring["detector"])
    assert report["paper_overlap"] == [{"claim_ids": ["test0", "train0", "val0"], "splits": ["test", "train", "val"]}]


def test_audit_uses_engine_pool_status_and_checks_cohort_overlap(local_corpus):
    cfg, _ = local_corpus
    base = cfg.paths.processed / "corpus"
    claims = read_json(base / "claims.json")
    gt = read_json(cfg.paths.processed / "ground_truth.json")
    gt["train0"]["original_cohort_repro"]["status"] = "out_of_pool"
    claims[0]["reported_cohort"]["in_standard_pool"] = True
    pool = {"cohorts": [{"accession": "TCGA-LIHC", "cohort_id": "test-only"}]}
    report = audit_corpus(claims, read_json(base / "natural_split.json")["by_claim"], gt,
                          pool, cfg.scoring["detector"], {"train0": "same", "test0": "same"})
    assert report["stratified"]["pool"]["out_of_pool"]["n"] == 1
    assert len(report["original_independent_cohort_overlap"]) == 6
    assert len(report["document_overlap"]) == 1


def test_null_baseline_is_not_a_floor_and_centering_does_not_zero_null_mean():
    assert .5 * support_variants(-1)["phi"] < .2
    z = stats.norm.ppf((np.arange(2000) + .5) / 2000)
    for name, expectation in (("phi", .5), ("positive_centered", .25), ("excess_above_95pct", .025)):
        assert np.mean([support_variants(x)[name] for x in z]) == pytest.approx(expectation, abs=1e-6)
    assert support_variants(0)["positive_centered"] == 0


def test_alternatives_preserve_other_components_and_reject_mismatched_controls():
    row = compare_row(z=0, robustness=.35, weight=.5)
    assert row["unchanged_other_components"] == pytest.approx(.1)
    assert row["robustness"]["phi"] == pytest.approx(.35)
    assert row["robustness"]["positive_centered"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="inconsistent"):
        compare_row(z=3, robustness=.1, weight=.5)
