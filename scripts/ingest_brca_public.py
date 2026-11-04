"""Ingest the two large open breast-cancer resources into the P0 cohort contract.

  * TCGA-BRCA  -- UCSC Xena public hub (HiSeqV2 log2(norm_count+1) matrix +
                  BRCA clinical matrix).  Open-access tier, no controlled data.
  * METABRIC   -- cBioPortal public study ``brca_metabric`` (Curtis et al. 2012),
                  clinical patient/sample tables + Illumina HT-12 v3 expression.

Both sources ship gene symbols directly, so no probe collapsing is needed.
Output files follow `cbi.cohort_pool.geo` (clinical.parquet, expression.parquet,
meta.json) with samples as expression rows.

Usage:
    python scripts/ingest_brca_public.py            # both cohorts
    python scripts/ingest_brca_public.py metabric
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbi.cohort_pool.geo import _split_stage, download  # noqa: E402
from cbi.util import write_json  # noqa: E402

XENA_BRCA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2F"
CBIO_METABRIC = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric"

MONTH_DAYS = 30.4375


def _write(out: Path, clinical: pd.DataFrame, expr: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    clinical = clinical.reset_index(drop=True).set_index("sample_id")
    n_genes_raw = int(expr.shape[1])
    expr = expr.dropna(axis=0, how="all")          # samples without any call
    expr = expr.dropna(axis=1, how="any")          # genes with missing values
    meta["genes_dropped_partial"] = n_genes_raw - int(expr.shape[1])
    common = clinical.index.intersection(expr.index)
    meta["dropped_no_expression"] = int(len(clinical) - len(common))
    clinical, expr = clinical.loc[common], expr.loc[common]
    if len(clinical) < 80:
        raise ValueError(f"{meta['cohort_id']}: only {len(clinical)} samples with both clinical and expression")
    clinical = clinical.reset_index()
    if expr.isna().any().any():
        raise ValueError(f"{meta['cohort_id']}: expression has NaN values")
    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta.update({
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
    })
    write_json(out / "meta.json", meta)
    return meta


def _stage_roman(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "not reported", "[not available]"}:
        return "NA"
    if text.replace(".0", "").isdigit():
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(text.replace(".0", ""), "NA")
    return _split_stage(text)


def ingest_tcga(out_root: Path) -> dict[str, Any]:
    out = out_root / "TCGA-BRCA"
    out.mkdir(parents=True, exist_ok=True)
    clin_path = download(XENA_BRCA + "BRCA_clinicalMatrix",
                         out / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix.tsv")
    expr_path = download(XENA_BRCA + "HiSeqV2.gz", out / "TCGA.BRCA.sampleMap_HiSeqV2.gz")

    raw = pd.read_csv(clin_path, sep="\t", low_memory=False).set_index("sampleID")
    raw = raw[raw["sample_type"].astype(str).str.lower() == "primary tumor"]

    vital = raw["vital_status"].astype(str).str.strip().str.lower()
    event = np.where(vital.isin(["dead", "deceased"]), 1.0,
                     np.where(vital.isin(["alive", "living"]), 0.0, np.nan))
    days = np.where(np.asarray(event) == 1.0, raw["days_to_death"], raw["days_to_last_followup"])
    days = pd.to_numeric(pd.Series(days, index=raw.index), errors="coerce")
    clinical = pd.DataFrame({
        "sample_id": raw.index.astype(str),
        "patient_id": raw["_PATIENT"].astype(str).to_numpy(),
        "os_time_days": days.to_numpy(dtype=float),
        "os_event": event,
        "stage_ajcc": [_stage_roman(v) for v in raw["pathologic_stage"]],
        "age_years": pd.to_numeric(raw["age_at_initial_pathologic_diagnosis"], errors="coerce").to_numpy(),
        "sex": raw["gender"].astype(str).str.lower().replace({"nan": "NA"}).to_numpy(),
        "neoadjuvant": False,
        "sample_type": "primary",
        "platform": "IlluminaHiSeq_RNASeqV2",
        "expr_unit": "log2_norm_count",
        "cohort_id": "TCGA-BRCA",
    })
    clinical = clinical[
        np.isfinite(clinical["os_time_days"]) & (clinical["os_time_days"] > 0)
        & clinical["os_event"].isin([0.0, 1.0]) & np.isfinite(clinical["age_years"])
    ]
    clinical["os_event"] = clinical["os_event"].astype(int)
    if len(clinical) < 80:
        raise ValueError(f"TCGA-BRCA: only {len(clinical)} usable samples")

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(level=0).mean().T  # duplicate symbols -> mean
    meta = {
        "cohort_id": "TCGA-BRCA",
        "cancer_type": "BRCA",
        "accession": "TCGA-BRCA",
        "platform": "IlluminaHiSeq_RNASeqV2",
        "expr_unit": "log2_norm_count",
        "source": "UCSC Xena public hub: TCGA.BRCA.sampleMap/HiSeqV2.gz + BRCA_clinicalMatrix",
        "license": "TCGA open-access tier data redistributed by UCSC Xena; see TCGA publication guidelines",
        "n_deposited": int(len(raw)),
        "sample_type": "primary tumour only (TCGA sample type 01)",
        "sex_source": "gender field",
        "neoadjuvant": "set False: TCGA-BRCA primary tumours are treatment-naive resections",
        "notes": (
            "OS = days_to_death when vital_status=DECEASED else days_to_last_followup; "
            "stage from pathologic_stage (AJCC). Expression is Xena HiSeqV2 "
            "log2(norm_count+1), gene symbols already collapsed (duplicates averaged)."
        ),
    }
    return _write(out, clinical, expr, meta)


def ingest_metabric(out_root: Path) -> dict[str, Any]:
    out = out_root / "METABRIC"
    out.mkdir(parents=True, exist_ok=True)
    patient_path = download(CBIO_METABRIC + "/data_clinical_patient.txt", out / "data_clinical_patient.txt")
    sample_path = download(CBIO_METABRIC + "/data_clinical_sample.txt", out / "data_clinical_sample.txt")
    expr_path = download(CBIO_METABRIC + "/data_mrna_illumina_microarray.txt",
                         out / "data_mrna_illumina_microarray.txt")

    patient = pd.read_csv(patient_path, sep="\t", comment="#", low_memory=False)
    sample = pd.read_csv(sample_path, sep="\t", comment="#", low_memory=False)
    merged = sample.merge(patient, on="PATIENT_ID", how="inner", suffixes=("", "_pat"))
    merged = merged[merged["SAMPLE_TYPE"].astype(str).str.lower() == "primary"]

    status = merged["OS_STATUS"].astype(str).str.strip()
    event = np.where(status.str.startswith("1"), 1.0, np.where(status.str.startswith("0"), 0.0, np.nan))
    clinical = pd.DataFrame({
        "sample_id": merged["SAMPLE_ID"].astype(str).to_numpy(),
        "patient_id": merged["PATIENT_ID"].astype(str).to_numpy(),
        "os_time_days": pd.to_numeric(merged["OS_MONTHS"], errors="coerce").to_numpy() * MONTH_DAYS,
        "os_event": event,
        "stage_ajcc": [_stage_roman(v) for v in merged["TUMOR_STAGE"]],
        "age_years": pd.to_numeric(merged["AGE_AT_DIAGNOSIS"], errors="coerce").to_numpy(),
        "sex": merged["SEX"].astype(str).str.lower().replace({"nan": "NA"}).to_numpy(),
        "neoadjuvant": False,
        "sample_type": "primary",
        "platform": "Illumina_HumanHT-12_v3",
        "expr_unit": "log2_intensity",
        "cohort_id": "METABRIC",
    })
    clinical = clinical[
        np.isfinite(clinical["os_time_days"]) & (clinical["os_time_days"] > 0)
        & clinical["os_event"].isin([0.0, 1.0]) & np.isfinite(clinical["age_years"])
    ].drop_duplicates("sample_id")
    clinical["os_event"] = clinical["os_event"].astype(int)
    clinical = clinical.set_index("sample_id")
    if len(clinical) < 80:
        raise ValueError(f"METABRIC: only {len(clinical)} usable samples")

    keep = ["Hugo_Symbol"] + [c for c in clinical.index if c in set(pd.read_csv(
        expr_path, sep="\t", nrows=0).columns)]
    expr = pd.read_csv(expr_path, sep="\t", usecols=keep, low_memory=False)
    expr = expr.dropna(subset=["Hugo_Symbol"])
    expr = expr.groupby("Hugo_Symbol").mean(numeric_only=True).T
    expr = expr.loc[expr.index.intersection(clinical.index)]
    clinical = clinical.loc[expr.index.intersection(clinical.index)].reset_index()

    meta = {
        "cohort_id": "METABRIC",
        "cancer_type": "BRCA",
        "accession": "brca_metabric",
        "platform": "Illumina_HumanHT-12_v3",
        "expr_unit": "log2_intensity",
        "source": "cBioPortal public study brca_metabric (Curtis et al., Nature 2012)",
        "license": "cBioPortal open-access study; ODbL-style cBioPortal data terms, see study page",
        "n_deposited": int(len(sample)),
        "sample_type": "primary tumour only",
        "sex_source": "SEX field",
        "neoadjuvant": "set False: METABRIC expression is from primary surgery specimens",
        "notes": (
            "OS = OS_MONTHS x 30.4375 days, OS_STATUS 1:DECEASED -> event. Stage from "
            "TUMOR_STAGE (coded 1-4 -> AJCC I-IV). Expression file is the cBioPortal "
            "Illumina HT-12 v3 matrix (Hugo_Symbol rows, log2 intensities)."
        ),
    }
    return _write(out, clinical, expr, meta)


GSE96058_SERIES = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/"
                   "matrix/GSE96058-GPL11154_series_matrix.txt.gz")
GSE96058_EXPR = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/"
                 "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz")


def ingest_gse96058(out_root: Path) -> dict[str, Any]:
    """SCAN-B cohort: the series matrix carries clinical data only (the table is
    empty), expression is the deposited gene-level supplement."""
    from scripts.ingest_brca_geo import _event, _num, _split_stage, read_header

    out = out_root / "GSE96058"
    out.mkdir(parents=True, exist_ok=True)
    series = download(GSE96058_SERIES, out / "GSE96058-GPL11154_series_matrix.txt.gz")
    csv_path = download(GSE96058_EXPR, out / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz")

    sample_ids, records, info = read_header(series)
    rows = []
    for gsm, rec in zip(sample_ids, records):
        days = _num(rec.get("overall survival days"))
        event = _event(rec.get("overall survival event"))
        age = _num(rec.get("age at diagnosis"))
        if not np.isfinite(days) or days <= 0 or event is None or not np.isfinite(age):
            continue
        grade = rec.get("nhg", "")
        stage = _split_stage(rec["tumor stage"]) if "tumor stage" in rec else "NA"
        rows.append({
            "sample_id": gsm,
            "patient_id": gsm,
            "os_time_days": float(days),
            "os_event": int(event),
            "stage_ajcc": stage,
            "age_years": float(age),
            "sex": "female",
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": str(info["platform"]),
            "expr_unit": "log2_fpkm",
            "cohort_id": "GSE96058",
            "grade": grade,
        })
    clinical = pd.DataFrame(rows).drop(columns=["grade"])

    usable = set(clinical["sample_id"])
    title_to_gsm = {
        rec["_title"]: gsm for gsm, rec in zip(sample_ids, records)
        if "_title" in rec and gsm in usable
    }
    header = pd.read_csv(csv_path, nrows=0)
    first_col = header.columns[0]
    cols = [first_col] + [c for c in header.columns if c in title_to_gsm]
    expr = pd.read_csv(csv_path, usecols=cols, low_memory=False).set_index(first_col)
    expr.index = expr.index.astype(str)
    expr = expr.groupby(level=0).mean().T          # rows = sample titles, cols = genes
    expr.index = [title_to_gsm[t] for t in expr.index]
    expr = expr.astype("float32")

    meta = {
        "cohort_id": "GSE96058",
        "cancer_type": "BRCA",
        "accession": "GSE96058",
        "platform": str(info["platform"]),
        "expr_unit": "log2_fpkm",
        "source": "GEO:GSE96058 (SCAN-B) series matrix + deposited gene-level supplement",
        "license": "public GEO record; see series citation and GEO terms of use",
        "n_deposited": len(sample_ids),
        "sample_type": "primary tumour (population-based SCAN-B series)",
        "sex_source": "cohort is an all-female breast-cancer series",
        "neoadjuvant": "set False: treatment-naive primary tumours (SCAN-B)",
        "notes": (
            "Series matrix contains clinical data only; expression is "
            "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz "
            "(log2(FPKM+0.1), gene symbols). Samples are joined on the sample "
            "title (F-number) shared by the matrix and the supplement; the 136 "
            "replicate columns are not used."
        ),
    }
    return _write(out, clinical, expr, meta)


INGESTORS = {"tcga": ingest_tcga, "metabric": ingest_metabric, "gse96058": ingest_gse96058}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-brca-public")
    parser.add_argument("cohorts", nargs="*", default=None, choices=[*INGESTORS, []])
    parser.add_argument("--out-root", default=str(ROOT / "data" / "raw" / "geo"))
    args = parser.parse_args(argv)
    out_root = Path(args.out_root)
    for name in (args.cohorts or list(INGESTORS)):
        meta = INGESTORS[name](out_root)
        print(f"{meta['cohort_id']}: n={meta['n']} events={meta['events_os']} genes={meta['n_genes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
