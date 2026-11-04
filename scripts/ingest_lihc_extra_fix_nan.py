"""Remove NaN cells from an existing standard cohort's expression matrix.

`cbi.cohort_pool.sanity.sanity_check_cohort` treats non-finite expression as a
critical failure, so a cohort whose matrix keeps unmeasured cells (genes absent
from one of two platforms, a handful of samples without a profile) cannot pass
the P0 gate.  This fixer removes the smallest amount of data that makes the
matrix finite:

  1. if dropping the affected genes keeps >= 80% of the genes, drop those genes
     (the usual case when one platform lacks some probes);
  2. otherwise drop the samples that carry missing cells (small number), then
     drop any gene still incomplete.

Clinical rows follow the surviving samples, meta is rewritten, and the cohort's
own parquet files are the only things touched.

Usage:
    python scripts/ingest_lihc_extra_fix_nan.py            # LIRI-JP, TCGA-LIHC, GSE14520
    python scripts/ingest_lihc_extra_fix_nan.py GSE14520
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbi.util import read_json, write_json  # noqa: E402

DEFAULT_COHORTS = ["LIRI-JP", "TCGA-LIHC", "GSE14520"]


def clean(cohort: str, root: Path) -> dict:
    directory = root / cohort
    clinical = pd.read_parquet(directory / "clinical.parquet")
    expr = pd.read_parquet(directory / "expression.parquet")
    before = (int(expr.shape[0]), int(expr.shape[1]))
    na = expr.isna()
    if not na.to_numpy().any():
        return {"cohort": cohort, "changed": False, **dict(zip(("n", "n_genes"), before))}

    n_genes = expr.shape[1]
    n_samples = expr.shape[0]
    genes_to_drop = int(na.any(axis=0).sum())
    dropped_samples = 0
    if genes_to_drop > 0.4 * n_genes:
        # too many genes carry blanks: instead drop the few affected samples
        bad_samples = na.any(axis=1)
        if int(bad_samples.sum()) > 0.2 * n_samples:
            raise SystemExit(f"{cohort}: missingness spread over genes and samples")
        dropped_samples = int(bad_samples.sum())
        expr = expr.loc[~bad_samples]
        clinical = clinical[clinical["sample_id"].isin(expr.index)]
        na = expr.isna()

    kept_genes = ~na.any(axis=0)
    dropped_genes = int((~kept_genes).sum())
    expr = expr.loc[:, kept_genes]
    clinical = clinical[clinical["sample_id"].isin(expr.index)].reset_index(drop=True)
    clinical, expr = clinical.set_index("sample_id").loc[expr.index].reset_index(), expr

    if expr.isna().to_numpy().any():
        raise SystemExit(f"{cohort}: NaNs remain after cleaning")
    if len(clinical) < 80:
        raise SystemExit(f"{cohort}: cleaning leaves only {len(clinical)} samples")
    if set(expr.index) != set(clinical["sample_id"]):
        raise SystemExit(f"{cohort}: clinical/expression samples diverged")

    expr.to_parquet(directory / "expression.parquet")
    clinical.to_parquet(directory / "clinical.parquet", index=False)

    meta = read_json(directory / "meta.json")
    meta.update({
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "nan_cleanup": {
            "dropped_genes": dropped_genes,
            "dropped_samples": dropped_samples,
            "shape_before": list(before),
            "shape_after": [int(expr.shape[0]), int(expr.shape[1])],
        },
    })
    meta["notes"] = str(meta.get("notes", "")).rstrip() + (
        f" Expression NaN cleanup (scripts/ingest_lihc_extra_fix_nan.py): "
        f"dropped {dropped_genes} genes and {dropped_samples} samples that had "
        f"missing cells; the matrix is finite."
    )
    write_json(directory / "meta.json", meta)
    return {"cohort": cohort, "changed": True, "n": int(len(clinical)),
            "n_genes": int(expr.shape[1]), "dropped_genes": dropped_genes,
            "dropped_samples": dropped_samples,
            "events": int(clinical["os_event"].sum())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lihc-fix-nan")
    parser.add_argument("cohorts", nargs="*", default=DEFAULT_COHORTS)
    parser.add_argument("--root", default=str(ROOT / "data" / "raw" / "geo"))
    args = parser.parse_args(argv)
    for cohort in args.cohorts:
        print(clean(cohort, Path(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
