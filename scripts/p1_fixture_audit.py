"""M1 synthetic-fixture regression (spec §M1 acceptance 1).

Renders 100 known claims into PMC-style XML, extracts them with the production
LLM extractor, and reports field-level precision plus the effect / direction /
contrast error rates.  Writes ``reports/P1_metrics.json`` and ``P1_metrics.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m1.extractor import ClaimExtractor, LLMRequiredError
from cbi.m1.fixtures import generate_claims, render_claim
from cbi.m1.pmc import parse_pmc_xml
from cbi.util import write_json


def _effect_close(pred, truth, tol: float = 0.05) -> bool:
    try:
        return abs(float(pred) - float(truth)) <= tol * abs(float(truth))
    except (TypeError, ValueError):
        return False


def _score(pairs: list[tuple]) -> dict:
    fields = {"gene": 0, "cancer_type": 0, "endpoint": 0, "direction": 0,
              "reported_effect": 0, "contrast_type": 0, "cutpoint": 0, "accession": 0}
    usable = 0
    for truth, pred, error in pairs:
        if error is not None or pred is None:
            continue
        usable += 1
        fields["gene"] += pred.entities.gene.upper() == truth.entities.gene.upper()
        fields["cancer_type"] += pred.entities.cancer_type == truth.entities.cancer_type
        fields["endpoint"] += pred.endpoint == truth.endpoint
        fields["direction"] += pred.direction == truth.direction
        fields["reported_effect"] += _effect_close(pred.reported_effect.value,
                                                   truth.reported_effect.value)
        fields["contrast_type"] += pred.reported_choices.contrast_type == truth.reported_choices.contrast_type
        fields["cutpoint"] += pred.reported_choices.cutpoint == truth.reported_choices.cutpoint
        fields["accession"] += bool(set(map(str.upper, pred.reported_cohort.accessions))
                                    & set(map(str.upper, truth.reported_cohort.accessions)))
    n = len(pairs)
    precision = {k: (v / n if n else 0.0) for k, v in fields.items()}
    return {
        "n": n, "usable": usable,
        "field_precision": precision,
        "min_field_precision": min(precision.values()) if precision else 0.0,
        "effect_error_rate": 1.0 - precision["reported_effect"],
        "direction_error_rate": 1.0 - precision["direction"],
        "contrast_error_rate": 1.0 - precision["contrast_type"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    cfg = load_config(ROOT)
    truth = generate_claims(args.n, seed=args.seed)
    extractor = ClaimExtractor(cfg)
    with tempfile.TemporaryDirectory(prefix="cbi_fixtures_") as tmp:
        articles = []
        for claim in truth:
            path = Path(tmp) / f"{claim.claim_id}.xml"
            path.write_text(render_claim(claim), encoding="utf-8")
            articles.append(parse_pmc_xml(path))
        try:
            results = extractor.extract_articles(articles, batch_size=args.batch_size)
        except LLMRequiredError as exc:
            print(f"BLOCKED: {exc}")
            return 2
    pairs = []
    for claim, (article, pred, error) in zip(truth, results):
        pairs.append((claim, pred, error))
    score = _score(pairs)
    errors = [str(e) for _, _, e in pairs if e is not None]
    score["errors"] = len(errors)
    score["error_examples"] = errors[:5]
    acceptance = {
        "field_precision_at_least_90pct": score["min_field_precision"] >= 0.90,
        "effect_error_at_most_3pct": score["effect_error_rate"] <= 0.03,
        "direction_error_at_most_3pct": score["direction_error_rate"] <= 0.03,
        "contrast_error_at_most_3pct": score["contrast_error_rate"] <= 0.03,
    }
    score["acceptance"] = acceptance
    score["pass"] = all(acceptance.values())
    score["note"] = ("Synthetic fixtures only; not evidence about real-corpus precision "
                     "(spec §M1 acceptance 1).")
    write_json(cfg.paths.reports / "P1_metrics.json", score)
    lines = ["# P1 — synthetic fixture regression (M1 acceptance 1)", "",
             f"- Fixtures: {score['n']}; usable extractions: {score['usable']}; errors: {score['errors']}",
             f"- Minimum field precision: **{score['min_field_precision']:.3f}** (threshold 0.90)",
             f"- Effect error rate: {score['effect_error_rate']:.3f} (threshold 0.03)",
             f"- Direction error rate: {score['direction_error_rate']:.3f} (threshold 0.03)",
             f"- Contrast error rate: {score['contrast_error_rate']:.3f} (threshold 0.03)",
             f"- Overall: **{'PASS' if score['pass'] else 'FAIL'}**",
             "", "| Field | Precision |", "|---|---:|"]
    for key, value in score["field_precision"].items():
        lines.append(f"| {key} | {value:.3f} |")
    lines += ["", score["note"]]
    (cfg.paths.reports / "P1_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2, sort_keys=True))
    return 0 if score["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
