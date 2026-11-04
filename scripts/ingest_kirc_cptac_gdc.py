"""Build a standard-schema KIRC cohort from the open GDC CPTAC-3 kidney arm.

Inputs (downloaded from the GDC API, no token required):
  * gdc_clinical.json  cases?filters=project CPTAC-3, primary_site Kidney
  * gdc/manifest.tsv   case, file_id, file_name for one primary-tumour RNA-seq
  * gdc/*.tsv          STAR - Counts gene quantification (tpm_unstranded)

OS time uses days_to_death when the patient died and days_to_last_follow_up
otherwise, so censored patients keep their follow-up time.

    python3 scripts/ingest_kirc_cptac_gdc.py --dir data/raw/geo/CPTAC-KIRC
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_ID = "CPTAC-KIRC"
PLATFORM = "GDC CPTAC-3 RNA-seq (Illumina NovaSeq, STAR - Counts)"
EXPR_UNIT = "TPM"


def age_years(demo: dict) -> float:
    age = demo.get("age_at_index")
    if age is not None:
        return float(age)
    days = demo.get("days_to_birth")
    if days is not None:
        return -float(days) / 365.25
    return np.nan


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

    cases = {c["submitter_id"]: c for c in json.loads((out / "gdc_clinical.json").read_text())}
    manifest = list(csv.reader((out / "gdc" / "manifest.tsv").open(), delimiter="\t"))

    rows = []
    columns: dict[str, pd.Series] = {}
    for case_id, _file_id, name in manifest:
        case = cases.get(case_id)
        if case is None:
            continue
        demo = case.get("demographic", {}) or {}
        diag = (case.get("diagnoses") or [{}])[0]
        if str(diag.get("morphology", "")) != "8312/3":
            continue  # 8312/3 = renal cell carcinoma, clear cell type
        vital = demo.get("vital_status")
        death = demo.get("days_to_death")
        followup = diag.get("days_to_last_follow_up")
        os_days = death if death is not None else followup
        if vital not in {"Dead", "Alive"} or os_days is None:
            continue
        os_days = float(os_days)
        if not np.isfinite(os_days) or os_days <= 0:
            continue
        table = pd.read_csv(out / "gdc" / name, sep="\t", comment="#", low_memory=False)
        series = table.dropna(subset=["gene_name"]).set_index("gene_name")["tpm_unstranded"]
        series.index = series.index.astype(str).str.upper().str.strip()
        series = series.groupby(level=0).mean()
        columns[case_id] = series
        rows.append({
            "sample_id": case_id,
            "patient_id": case_id,
            "os_time_days": os_days,
            "os_event": 1 if vital == "Dead" else 0,
            "stage_ajcc": stage_code(diag.get("ajcc_pathologic_stage")),
            "age_years": age_years(demo),
            "sex": str(demo.get("sex_at_birth", "NA")).strip().lower() or "NA",
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": PLATFORM,
            "expr_unit": EXPR_UNIT,
            "cohort_id": COHORT_ID,
        })
    clinical = pd.DataFrame(rows)
    if clinical.empty:
        raise SystemExit("no CPTAC kidney cases with OS")
    clinical["age_years"] = clinical["age_years"].round().astype("Int64").fillna(
        int(clinical["age_years"].median())).astype("int64")

    expr = pd.DataFrame(columns).T
    expr.index.name = "sample_id"
    expr = expr.reindex(clinical["sample_id"].tolist())
    expr = expr.loc[:, expr.notna().any(axis=0)].fillna(0.0)

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": COHORT_ID,
        "cancer_type": "KIRC",
        "accession": "GDC:CPTAC-3 kidney",
        "platform": PLATFORM,
        "expr_unit": EXPR_UNIT,
        "source": "NCI GDC open API, project CPTAC-3, primary_site Kidney",
        "license": "GDC open-access data (no controlled-access files used)",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": "One primary-tumour STAR gene-count file per case; censored follow-up from diagnoses.days_to_last_follow_up.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
