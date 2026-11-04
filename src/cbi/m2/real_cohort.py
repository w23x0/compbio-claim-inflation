"""Evaluate a standard-schema cohort directly (real-data path).

The real-data pipeline uses `CohortStore` + frozen split roles.  A freshly ingested
GEO cohort is not part of that split yet, so this module evaluates the standard
parquet files directly with the same `estimate_effect` used by M2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from cbi.m2.specs import default_spec, estimate_effect
from cbi.util import read_json


def load_standard_cohort(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = Path(directory)
    clinical = pd.read_parquet(directory / "clinical.parquet")
    expression = pd.read_parquet(directory / "expression.parquet")
    meta_path = directory / "meta.json"
    meta = read_json(meta_path) if meta_path.is_file() else {}
    return clinical, expression, meta


def evaluate_genes(directory: str | Path, genes: list[str], *, endpoint: str = "OS",
                   contrast_type: str = "dichotomous") -> list[dict]:
    clinical, expression, _ = load_standard_cohort(directory)
    spec = default_spec(contrast_type)
    out = []
    for gene in genes:
        gene = gene.strip().upper()
        if gene not in expression.columns:
            out.append({"gene": gene, "error": "gene not in expression matrix"})
            continue
        effect = estimate_effect(clinical, expression[gene], gene, spec, endpoint)
        if effect is None:
            out.append({"gene": gene, "error": "no analyzable fit"})
            continue
        out.append({
            "gene": gene, "hr": float(math.exp(effect.log_effect)),
            "log_effect": effect.log_effect, "se": effect.se, "p": effect.p,
            "n": effect.n, "events": effect.events, "spec": spec.spec_id,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-real-eval")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--genes", default="TP53,MKI67,BIRC5,AFP,CTNNB1")
    parser.add_argument("--endpoint", default="OS")
    parser.add_argument("--contrast", default="dichotomous", choices=["dichotomous", "continuous"])
    args = parser.parse_args(argv)
    results = evaluate_genes(args.cohort, args.genes.split(","), endpoint=args.endpoint,
                             contrast_type=args.contrast)
    for row in results:
        if "error" in row:
            print(f"{row['gene']}\t{row['error']}")
        else:
            print(f"{row['gene']}\tHR={row['hr']:.3f}\tp={row['p']:.4g}\tn={row['n']}\tevents={row['events']}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
