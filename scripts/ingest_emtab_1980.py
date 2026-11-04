"""Build the E-MTAB-1980 (University of Tokyo, Sato et al. 2013) ccRCC cohort.

Public sources only:
  * ArrayExpress E-MTAB-1980 processed matrix (Agilent 4x44K, log2 quantile
    normalized): 101 primary ccRCC tumours, ``ccRCC_exp_log_quantile_normalized.txt.gz``.
  * cBioPortal ``ccrcc_utokyo_2013`` (same cohort, 106 patients) for the
    per-patient clinical data: FOLLOW_UP_TIME_MONTHS (months to death or last
    follow-up) and VITAL_STATUS, plus AGE/SEX and sample-level
    STAGE_AT_DIAGNOSIS/GRADE.  The cBioPortal study carries no mRNA profile,
    which is why the expression comes from ArrayExpress.
  * GPL6480 annotation (Agilent-014850 4x44K) for probe -> gene symbol.

The two identifiers differ only by '-' vs '_' (ccRCC-1 <-> ccRCC_1); all 101
matrix samples match a cBioPortal patient.

    python3 scripts/ingest_emtab_1980.py --dir data/raw/geo/E-MTAB-1980
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_ID = "E-MTAB-1980"
PLATFORM = "Agilent-014850 Whole Human Genome 4x44K (A-MEXP-2183)"
EXPR_UNIT = "log2(quantile_normalized)"
MONTH_DAYS = 30.4375


def load_annotation(path: Path) -> dict[str, str]:
    """probe ID -> gene symbol, from a GEO .annot file (skips '#' preamble)."""
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = None
        for line in fh:
            if reader is None:
                if line.startswith("ID\t"):
                    header = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
                    reader = True
                    id_col = header.index("ID")
                    try:
                        sym_col = header.index("Gene symbol")
                    except ValueError:
                        sym_col = header.index("GENE_SYMBOL")
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_col, sym_col):
                continue
            probe, symbol = parts[id_col].strip(), parts[sym_col].strip()
            if probe and symbol and symbol not in {"---", "", "null"}:
                mapping[probe] = symbol.upper()
    if not mapping:
        raise SystemExit(f"no probe->symbol mapping in {path}")
    return mapping


def parse_stage(value: object) -> str:
    """pT1bN0M0 / pT3aN1M0 -> AJCC I..IV (RCC 7th edition T/N/M rules)."""
    text = str(value).upper().replace(" ", "")
    if "M1" in text:
        return "IV"
    if text.startswith("PT4") or text.startswith("T4"):
        return "IV"
    if "N1" in text or "N2" in text:
        return "III"
    if text.startswith("PT3") or text.startswith("T3"):
        return "III"
    if text.startswith("PT2") or text.startswith("T2"):
        return "II"
    if text.startswith("PT1") or text.startswith("T1"):
        return "I"
    return "NA"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=f"data/raw/geo/{COHORT_ID}")
    args = parser.parse_args()
    out = Path(args.dir)

    # ---- clinical (cBioPortal, patient- and sample-level) -------------------
    patient_rows = json.loads((out / "cbioportal_clinical_patient.json").read_text())
    sample_rows = json.loads((out / "cbioportal_clinical_sample.json").read_text())
    patient: dict[str, dict[str, str]] = {}
    for row in patient_rows:
        patient.setdefault(row["patientId"], {})[row["clinicalAttributeId"]] = row["value"]
    sample: dict[str, dict[str, str]] = {}
    for row in sample_rows:
        sample.setdefault(row["sampleId"], {})[row["clinicalAttributeId"]] = row["value"]

    # ---- expression ---------------------------------------------------------
    mat = pd.read_csv(out / "ccRCC_exp_log_quantile_normalized.txt.gz", sep="\t")
    mat = mat.rename(columns={mat.columns[0]: "probe"})
    mat["probe"] = mat["probe"].astype(str)
    sample_cols = [c for c in mat.columns if c not in {"probe", "Feature Numbers", "SystematicName"}]
    expr = mat.set_index("probe")[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr.columns = [str(c).replace("-", "_") for c in expr.columns]
    expr.index.name = "probe"

    annotation = load_annotation(out / "GPL6480.annot.gz")
    expr = expr.loc[expr.index.isin(annotation)]
    expr.index = [annotation[p] for p in expr.index]
    # One probe per gene: the probe with the highest mean signal (max_mean).
    order = expr.mean(axis=1).sort_values(ascending=False).index
    expr = expr.loc[order]
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.T
    expr.index.name = "sample_id"
    expr = expr.fillna(expr.median(axis=0))

    # ---- assemble clinical --------------------------------------------------
    rows = []
    for sid in expr.index:
        pat = patient.get(sid, {})
        smp = sample.get(sid, {})
        months = pd.to_numeric(pat.get("FOLLOW_UP_TIME_MONTHS"), errors="coerce")
        vital = str(pat.get("VITAL_STATUS", "")).strip().upper()
        if not np.isfinite(months) or months <= 0 or vital not in {"DECEASED", "LIVING"}:
            continue
        age = pd.to_numeric(pat.get("AGE"), errors="coerce")
        rows.append({
            "sample_id": sid,
            "patient_id": sid,
            "os_time_days": float(months) * MONTH_DAYS,
            "os_event": int(vital == "DECEASED"),
            "stage_ajcc": parse_stage(smp.get("STAGE_AT_DIAGNOSIS")),
            "age_years": age,
            "sex": {"MALE": "male", "FEMALE": "female"}.get(
                str(pat.get("SEX", "")).strip().upper(), "NA"),
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": PLATFORM,
            "expr_unit": EXPR_UNIT,
            "cohort_id": COHORT_ID,
        })
    clinical = pd.DataFrame(rows)
    if clinical.empty:
        raise SystemExit("no E-MTAB-1980 samples with OS time and vital status")
    if clinical["age_years"].isna().any():
        raise SystemExit("missing age for some samples")
    clinical["age_years"] = clinical["age_years"].astype("float64").round().astype("int64")

    keep = [s for s in clinical["sample_id"] if s in expr.index]
    clinical = clinical.set_index("sample_id").loc[keep].reset_index()
    expr = expr.loc[keep]
    expr.columns = [str(c).upper() for c in expr.columns]
    expr = expr.T.groupby(level=0).max().T

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")

    meta = {
        "cohort_id": COHORT_ID,
        "cancer_type": "KIRC",
        "accession": "E-MTAB-1980",
        "platform": PLATFORM,
        "expr_unit": EXPR_UNIT,
        "source": (
            "ArrayExpress E-MTAB-1980 processed matrix (log2 quantile normalized) + "
            "cBioPortal ccrcc_utokyo_2013 clinical (FOLLOW_UP_TIME_MONTHS, VITAL_STATUS, "
            "AGE, SEX, STAGE_AT_DIAGNOSIS); probe->symbol via GEO GPL6480"
        ),
        "license": "public ArrayExpress record; cBioPortal public study ccrcc_utokyo_2013",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": (
            "Sato et al. 2013 (University of Tokyo) ccRCC; 101 primary tumours with an "
            "Agilent expression profile out of 106 cBioPortal patients. OS time is the "
            "cBioPortal FOLLOW_UP_TIME_MONTHS (time to death for deceased, last follow-up "
            "for living) converted at 30.4375 days/month. Neoadjuvant therapy is not "
            "reported and coded False (primary nephrectomy cohort). Stage derived from "
            "pTNM strings. Same patients as cBioPortal ccrcc_utokyo_2013."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
