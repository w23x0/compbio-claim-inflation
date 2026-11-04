"""M2 acceptance controls (spec §M2).

Three checks that were never implemented:

1. Positive controls: canonical prognostic markers per cancer, claimed at a
   fixed HR=2.0 per SD (chosen before looking at these cohorts), must reach a
   median ``robustness_score`` above 0.6.
2. Negative controls: random genes measured in every dev cohort of the cancer,
   claimed the same way, must stay below 0.2.
3. Inflation sensitivity: for the same random genes, a report built from the
   best data-driven cutpoint (``optimal``) must score higher ``inflation`` than
   the same gene reported under the reference default specification (median
   split), whose inflation is 0 by construction.

Only dev cohorts are used (spec §6 isolation).  The report records the raw
per-gene numbers so the thresholds can be re-derived without rerunning.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.cohort_pool.loader import CohortStore
from cbi.config import load_config
from cbi.m2.analysis import CohortAnalyzer
from cbi.m2.engine import GroundTruthEngine
from cbi.m2.meta import meta_random_effects
from cbi.m2.specs import AnalysisSpec, default_spec
from cbi.schemas import (Claim, DataAvailability, Entities, ReportedChoices,
                         ReportedCohort, ReportedEffect, SourceRef)
from cbi.util import read_json, write_json

POSITIVE = {
    "BRCA": ["MKI67", "TOP2A", "AURKA", "BIRC5", "CDK1"],
    "LIHC": ["BIRC5", "TOP2A", "CDK1", "CCNB1", "MKI67"],
    "LUAD": ["MKI67", "TOP2A", "BIRC5", "CDK1", "CCNB1"],
    "KIRC": ["CA9", "BIRC5", "TOP2A", "MKI67", "CDK1"],
}
BASIS = ("canonical proliferation/hypoxia markers with a published prognostic "
         "association in the cancer; list fixed before any dev-cohort effect was "
         "computed for them")

_CONTINUOUS_SPEC = default_spec("continuous")
_OPTIMAL_SPEC = AnalysisSpec(spec_id="optimal|none|all_primary|full", grouping="optimal")


def _claim(cancer: str, gene: str, *, continuous: bool, hr: float = 2.0) -> Claim:
    return Claim(
        claim_id=f"ctl_{cancer}_{gene}",
        source=SourceRef(pub_date=date(2020, 1, 1)),
        entities=Entities(gene=gene, cancer_type=cancer),
        endpoint="OS",
        direction="high_expression_worse",
        reported_effect=ReportedEffect(metric="HR", value=hr, ci_low=hr * 0.75,
                                       ci_high=hr * 1.35),
        reported_cohort=ReportedCohort(accessions=[]),
        reported_choices=ReportedChoices(
            contrast_type="continuous" if continuous else "dichotomous",
            cutpoint="per_sd" if continuous else "median"),
        data_availability=DataAvailability(),
    )


def _pooled(analyzer: CohortAnalyzer, cohorts: list[str], gene: str,
            spec: AnalysisSpec) -> tuple[float, float] | None:
    effects = [e for e in (analyzer.effect(cid, gene, spec, "OS") for cid in cohorts) if e]
    if not effects:
        return None
    try:
        m = meta_random_effects(np.asarray([e.log_effect for e in effects]),
                                np.asarray([e.se for e in effects]))
        return float(m.pooled), float(m.se)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-per-cancer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    cfg = load_config(ROOT)
    store = CohortStore(cfg, allowed_roles={"dev"})
    analyzer = CohortAnalyzer(cfg, store)
    engine = GroundTruthEngine(cfg, analyzer)
    scale = float(cfg.scoring["inflation_scale"])
    rng = random.Random(args.seed)

    cohorts_by_cancer: dict[str, list[str]] = {}
    genes_by_cohort: dict[str, set[str]] = {}
    for cid in store.cohort_ids(role="dev"):
        cancer = str(store.meta(cid).get("cancer_type"))
        cohorts_by_cancer.setdefault(cancer, []).append(cid)
        genes_by_cohort[cid] = set(store.genes(cid))
    corpus_genes = {c["entities"]["gene"].upper()
                    for c in read_json(cfg.paths.processed / "corpus" / "claims.json")}

    positives: list[dict] = []
    negatives: list[dict] = []
    sensitivity: list[dict] = []
    for cancer, cohorts in sorted(cohorts_by_cancer.items()):
        measured = set.intersection(*(genes_by_cohort[c] for c in cohorts))
        pool = sorted(measured - corpus_genes - set(POSITIVE.get(cancer, [])))
        picked = rng.sample(pool, min(args.negative_per_cancer, len(pool)))

        for gene in POSITIVE.get(cancer, []):
            if gene not in measured:
                positives.append({"cancer": cancer, "gene": gene, "error": "not measured in every dev cohort"})
                continue
            result = engine.evaluate_claim(_claim(cancer, gene, continuous=True),
                                           independent_role="dev")
            positives.append({"cancer": cancer, "gene": gene,
                              "robustness": result.robustness_score,
                              "inflation": result.inflation,
                              "confidence": result.gt_confidence})

        for gene in picked:
            result = engine.evaluate_claim(_claim(cancer, gene, continuous=True),
                                           independent_role="dev")
            negatives.append({"cancer": cancer, "gene": gene,
                              "robustness": result.robustness_score,
                              "inflation": result.inflation})
            # Optimal-cutpoint claim vs the median-split reference on the same gene.
            ref = _pooled(analyzer, cohorts, gene, default_spec("dichotomous"))
            opt = _pooled(analyzer, cohorts, gene, _OPTIMAL_SPEC)
            if ref and opt:
                mu_ref = ref[0]
                r_opt = opt[0]
                sensitivity.append({
                    "cancer": cancer, "gene": gene,
                    "mu_median": mu_ref, "r_optimal": r_opt,
                    "inflation_optimal": float(math.tanh(max(0.0, r_opt - mu_ref) / scale)),
                    "inflation_reference": 0.0,
                })
        print(f"{cancer}: {len(cohorts)} dev cohorts, {len(picked)} negative genes", flush=True)

    def _median(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        return float(statistics.median(vals)) if vals else None

    pos_med = _median(positives, "robustness")
    neg_med = _median(negatives, "robustness")
    sens_med = _median(sensitivity, "inflation_optimal")
    report = {
        "spec": "M2 acceptance controls",
        "basis": BASIS,
        "cohorts": cohorts_by_cancer,
        "positive": {"n": len(positives), "median_robustness": pos_med,
                     "threshold": 0.6, "pass": bool(pos_med is not None and pos_med > 0.6),
                     "rows": positives},
        "negative": {"n": len(negatives), "median_robustness": neg_med,
                     "threshold": 0.2, "pass": bool(neg_med is not None and neg_med < 0.2),
                     "rows": negatives},
        "inflation_sensitivity": {
            "n": len(sensitivity),
            "median_inflation_optimal": sens_med,
            "median_inflation_reference": 0.0,
            "pass": bool(sens_med is not None and sens_med > 0.05),
            "rows": sensitivity,
        },
        "cross_snapshot_stability": {
            "status": "not evaluable: only one processed snapshot exists",
        },
    }
    write_json(cfg.paths.reports / "M2_controls.json", report)
    lines = [
        "# M2 — acceptance controls",
        "",
        f"- Positive controls (canonical markers): n={len(positives)}, "
        f"median robustness **{pos_med}** (threshold > 0.6: {'PASS' if report['positive']['pass'] else 'FAIL'})",
        f"- Negative controls (random genes): n={len(negatives)}, "
        f"median robustness **{neg_med}** (threshold < 0.2: {'PASS' if report['negative']['pass'] else 'FAIL'})",
        "- At zero effect, 0.5*Phi(0)=0.25 is a baseline, not a hard lower bound. "
        "Negative z values can score below 0.2. See M2_null_analysis.md for the ideal-null expectation and exploratory alternatives.",
        f"- Inflation sensitivity: n={len(sensitivity)}, median optimal-cutpoint inflation "
        f"**{sens_med}** vs reference 0.0 (threshold > 0.05: {'PASS' if report['inflation_sensitivity']['pass'] else 'FAIL'})",
        "- Cross-snapshot stability: not evaluable (one snapshot).",
        "",
        f"Positive basis: {BASIS}.",
    ]
    (cfg.paths.reports / "M2_controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in
                      {"positive", "negative", "inflation_sensitivity"}}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
