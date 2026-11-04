"""Build a standard-schema KIRC cohort from the open UCSC Xena TCGA-KIRC hub.

Sources (all public, no token):
  * HiSeqV2.gz          genes x samples, log2(norm_count+1) RNA-seq
  * KIRC_survival.txt   sample, _PATIENT, OS, OS.time (days)
  * KIRC_clinicalMatrix stage / age / sex / neoadjuvant / sample type

The Xena sample IDs are tumour barcodes truncated to ``TCGA-XX-XXXX-01``.
Only primary tumour samples with a finite OS time are kept.

    python3 scripts/ingest_kirc_tcga_xena.py --dir data/raw/geo/TCGA-KIRC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_ID = "TCGA-KIRC"
PLATFORM = "Illumina HiSeq 2000 RNA-seq (Xena HiSeqV2)"
EXPR_UNIT = "log2(norm_count+1)"


def stage_code(value: object) -> str:
    text = str(value).upper().replace("STAGE", "").strip()
    for stage in ("IV", "III", "II", "I"):
        if text.startswith(stage):
            return stage
    return "NA"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=f"data/raw/geo/{COHORT_ID}")
    args = parser.parse_args()
    out = Path(args.dir)

    survival = pd.read_csv(out / "KIRC_survival.txt", sep="\t")
    clinical_src = pd.read_csv(out / "KIRC_clinicalMatrix", sep="\t", low_memory=False)
    expression = pd.read_csv(out / "HiSeqV2.gz", sep="\t", index_col=0)
    expression.columns = [str(c).strip() for c in expression.columns]
    expression.index = [str(g).strip() for g in expression.index]

    clinical_src = clinical_src.set_index("sampleID")
    rows = []
    for _, r in survival.iterrows():
        sample_id = str(r["sample"])
        if sample_id not in clinical_src.index:
            continue
        src = clinical_src.loc[sample_id]
        os_time = pd.to_numeric(r["OS.time"], errors="coerce")
        os_event = pd.to_numeric(r["OS"], errors="coerce")
        if not np.isfinite(os_time) or os_time <= 0 or not np.isfinite(os_event):
            continue
        if str(src.get("sample_type", "")).lower() != "primary tumor":
            continue
        rows.append({
            "sample_id": sample_id,
            "patient_id": sample_id[:12],
            "os_time_days": float(os_time),
            "os_event": int(os_event),
            "stage_ajcc": stage_code(src.get("pathologic_stage")),
            "age_years": pd.to_numeric(src.get("age_at_initial_pathologic_diagnosis"), errors="coerce"),
            "sex": str(src.get("gender", "NA")).strip().lower() or "NA",
            "neoadjuvant": str(src.get("history_of_neoadjuvant_treatment", "No")).strip().lower() == "yes",
            "sample_type": "primary",
            "platform": PLATFORM,
            "expr_unit": EXPR_UNIT,
            "cohort_id": COHORT_ID,
        })
    clinical = pd.DataFrame(rows).drop_duplicates("sample_id")
    if clinical.empty:
        raise SystemExit("no TCGA-KIRC samples with OS")
    clinical["age_years"] = clinical["age_years"].astype("float64").round().astype("Int64")
    if clinical["age_years"].isna().any():
        clinical["age_years"] = clinical["age_years"].fillna(
            clinical["age_years"].median()).astype("int64")

    keep = [s for s in clinical["sample_id"] if s in expression.columns]
    clinical = clinical.set_index("sample_id").loc[keep].reset_index()
    expr = expression[keep].T
    expr.index.name = "sample_id"
    expr = expr.loc[:, expr.notna().any(axis=0)]
    expr = expr.dropna(axis=1, how="all")
    expr = expr.fillna(expr.median(numeric_only=True))
    expr.columns = [str(c).upper() for c in expr.columns]
    expr = expr.T.groupby(level=0).max().T

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": COHORT_ID,
        "cancer_type": "KIRC",
        "accession": "TCGA-KIRC",
        "platform": PLATFORM,
        "expr_unit": EXPR_UNIT,
        "source": "UCSC Xena TCGA hub: TCGA.KIRC.sampleMap/HiSeqV2 + survival/KIRC_survival.txt",
        "license": "TCGA open-access data (dbGaP-controlled raw data not used)",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": "Open TCGA-KIRC RNA-seq (Xena HiSeqV2) joined to the Xena survival table; primary tumours only.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
