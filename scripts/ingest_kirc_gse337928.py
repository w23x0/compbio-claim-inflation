"""Build a standard-schema KIRC cohort from GEO GSE337928 (ccRCC RNA-seq).

GSE337928 provides no values in the series matrix: expression lives in the
supplementary ``DCC_gene_tpm.csv.gz`` and ``DCC_gene_counts.csv.gz`` files,
while survival and stage are in ``!Sample_characteristics_ch1``.  Tumour and
adjacent-normal samples come in pairs and share the patient's OS, so just the
tumour samples are kept.

    python3 scripts/ingest_kirc_gse337928.py --dir data/raw/geo/GSE337928
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_ID = "GSE337928"
PLATFORM = "GPL24676 (Illumina NovaSeq 6000, RNA-seq)"
EXPR_UNIT = "TPM"

STAGE = {"1": "I", "2": "II", "3": "III", "4": "IV"}


def parse_series(path: Path) -> tuple[list[str], list[str], dict[str, dict[int, str]]]:
    samples: list[str] = []
    titles: list[str] = []
    chars: dict[str, dict[int, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            parts = line.rstrip("\n").split("\t")
            key = parts[0]
            values = [p.strip().strip('"') for p in parts[1:]]
            if key == "!Sample_geo_accession":
                samples = values
            elif key == "!Sample_title":
                titles = values
            elif key == "!Sample_characteristics_ch1":
                for i, value in enumerate(values):
                    if ":" in value:
                        k, v = value.split(":", 1)
                        chars.setdefault(k.strip().lower(), {})[i] = v.strip()
    return samples, titles, chars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=f"data/raw/geo/{COHORT_ID}")
    args = parser.parse_args()
    out = Path(args.dir)

    samples, titles, chars = parse_series(out / f"{COHORT_ID}_series_matrix.txt.gz")
    tpm = pd.read_csv(out / "GSE337928_DCC_gene_tpm.csv.gz")
    tpm = tpm.dropna(subset=["gene_name"])
    tpm["gene_name"] = tpm["gene_name"].astype(str).str.upper().str.strip()
    tpm = tpm[tpm["gene_name"].str.len() > 0]
    tpm = tpm.groupby("gene_name").mean(numeric_only=True)

    tokens = [re.search(r"(DCC\w+)", t).group(1) for t in titles]
    if set(tokens) != set(tpm.columns):
        raise SystemExit("DCC sample tokens in the series matrix do not match the TPM columns")
    value_col = {tok: col for tok, col in zip(tokens, tpm.columns)}

    rows = []
    for i, sample_id in enumerate(samples):
        if chars["condition"].get(i) != "tumor":
            continue
        os_days = pd.to_numeric(chars["os days"].get(i), errors="coerce")
        vital = chars["vital status"].get(i, "")
        if not np.isfinite(os_days) or os_days <= 0 or vital not in {"Event", "Censored"}:
            continue
        stage = STAGE.get(chars["stage ajcc"].get(i, ""), "NA")
        rows.append({
            "sample_id": sample_id,
            "patient_id": tokens[i].rstrip("N"),
            "os_time_days": float(os_days),
            "os_event": 1 if vital == "Event" else 0,
            "stage_ajcc": stage,
            "age_years": pd.to_numeric(chars["age"].get(i), errors="coerce"),
            "sex": chars["sex"].get(i, "NA").strip().lower() or "NA",
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": PLATFORM,
            "expr_unit": EXPR_UNIT,
            "cohort_id": COHORT_ID,
            "_col": value_col[tokens[i]],
        })
    clinical = pd.DataFrame(rows)
    if clinical.empty:
        raise SystemExit("no GSE337928 tumours with OS")
    clinical["age_years"] = clinical["age_years"].round().astype("Int64")
    clinical["age_years"] = clinical["age_years"].fillna(
        clinical["age_years"].median()).astype("int64")

    expr = tpm[clinical["_col"].tolist()].T
    expr.index = clinical["sample_id"].tolist()
    expr.index.name = "sample_id"
    expr = expr.loc[:, expr.notna().any(axis=0)].fillna(0.0)

    clinical = clinical.drop(columns=["_col"])
    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": COHORT_ID,
        "cancer_type": "KIRC",
        "accession": COHORT_ID,
        "platform": PLATFORM,
        "expr_unit": EXPR_UNIT,
        "source": "GEO:GSE337928 (DCC gene TPM supplementary)",
        "license": "public GEO record; check the series citation and GEO terms",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": "Tumour samples only; paired adjacent normals share the patient's OS and are dropped.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
