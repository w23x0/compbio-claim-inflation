"""Synthetic JATS fixture generator for the M1 regression (spec §M1 acceptance 1).

The generator renders a ``Claim`` into a small PMC-style XML article so the
extractor can be scored on cases whose fields are known exactly.  It is for
regression only and is never used as evidence about real-corpus precision; the
fixtures live under a temp directory and are not written into ``data/raw``.
"""
from __future__ import annotations

import random
from datetime import date
from xml.sax.saxutils import escape

from cbi.schemas import (Claim, DataAvailability, Entities, ReportedChoices,
                         ReportedCohort, ReportedEffect, SourceRef)

GENES = ["TP53", "MKI67", "EGFR", "VHL", "CTNNB1", "BIRC5", "TOP2A", "AURKA",
         "CDK1", "CCNB1", "CA9", "ERBB2", "ESR1", "KRAS", "MYC", "PTEN",
         "RB1", "SMAD4", "STAT3", "BCL2", "CASP3", "VEGFA", "HIF1A", "MET"]
CANCERS = ["BRCA", "LIHC", "LUAD", "KIRC"]
CUTPOINTS = ["median", "tertile", "optimal", "per_sd"]
ACCESSIONS = ["TCGA-BRCA", "TCGA-LIHC", "TCGA-LUAD", "TCGA-KIRC",
              "GSE14520", "GSE37745", "GSE68465", "GSE96058", "METABRIC"]


def generate_claims(n: int = 100, seed: int = 20261101) -> list[Claim]:
    rng = random.Random(seed)
    claims: list[Claim] = []
    for i in range(n):
        gene = rng.choice(GENES)
        cancer = rng.choice(CANCERS)
        cutpoint = rng.choice(CUTPOINTS)
        continuous = cutpoint == "per_sd"
        if continuous:
            hr = round(rng.uniform(1.2, 2.6), 3)
        else:
            hr = round(rng.uniform(1.3, 3.0), 3)
        if rng.random() < 0.25:
            hr = round(1.0 / hr, 3)
            direction = "high_expression_better"
        else:
            direction = "high_expression_worse"
        claims.append(Claim(
            claim_id=f"fx_{i:03d}_{gene}",
            source=SourceRef(pub_date=date(2021, 1, 1)),
            entities=Entities(gene=gene, cancer_type=cancer),
            endpoint="OS",
            direction=direction,
            reported_effect=ReportedEffect(metric="HR", value=hr,
                                           ci_low=round(hr * 0.75, 3),
                                           ci_high=round(hr * 1.3, 3)),
            reported_cohort=ReportedCohort(accessions=[rng.choice(ACCESSIONS)], n_reported=400),
            reported_choices=ReportedChoices(
                contrast_type="continuous" if continuous else "dichotomous",
                cutpoint=cutpoint,
                multiple_testing_correction=False,
            ),
            data_availability=DataAvailability(raw_deposited=False, code_available=False),
        ))
    return claims


def _comparison(claim: Claim) -> str:
    if claim.reported_choices.contrast_type == "continuous":
        return ("expression was analysed as a continuous variable and the hazard ratio "
                "is reported per 1 standard deviation")
    if claim.reported_choices.cutpoint == "median":
        return "patients were dichotomised at the median expression level"
    if claim.reported_choices.cutpoint == "tertile":
        return "patients were split into top versus bottom tertile of expression"
    return "the optimal cutpoint was selected by maximally selected rank statistics"


def _direction_text(claim: Claim) -> str:
    if claim.direction.startswith("high_expression_better"):
        return "High expression was associated with better overall survival."
    return "High expression was associated with worse overall survival."


def render_claim(claim: Claim) -> str:
    gene = escape(claim.entities.gene)
    cancer = escape(claim.entities.cancer_type)
    accession = escape(claim.reported_cohort.accessions[0] if claim.reported_cohort.accessions else "")
    effect = claim.reported_effect
    ci = f"95% CI {effect.ci_low} to {effect.ci_high}"
    sentence = (f"In {cancer}, {gene} expression was prognostic for overall survival "
                f"(HR {effect.value}, {ci}, p < 0.001). {_direction_text(claim)}")
    methods = (f"{_comparison(claim)}. Cox proportional hazards models were fitted in "
               f"{accession}. Overall survival was the endpoint.")
    sentence_xml = escape(sentence)
    methods_xml = escape(methods)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">FX{claim.claim_id}</article-id>
      <title-group><article-title>{gene} expression and survival in {cancer}</article-title></title-group>
      <pub-date><year>2021</year><month>1</month><day>1</day></pub-date>
      <abstract><p>{sentence_xml}</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title><p>Prognostic markers in {cancer} remain incompletely characterised.</p></sec>
    <sec><title>Materials and methods</title><p>{methods_xml}</p></sec>
    <sec><title>Results</title><p>{sentence_xml} The analysis included {claim.reported_cohort.n_reported} patients.</p></sec>
    <sec><title>Data availability statement</title><p>No data or code were deposited.</p></sec>
  </body>
</article>
"""
