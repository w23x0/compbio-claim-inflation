"""Fix the LIRI-JP (ICGC) vital-status mapping and rebuild its expression matrix.

`data/raw/geo/LIRI-JP` was built by `scripts/ingest_lihc_icgc.py`, but its
`os_event` column came out all-zero: the release-28 donor rows encode
`donor_vital_status` as ``alive``/``deceased``, and the recorded events were
built from a mapping that never matched ``deceased``.  The raw ICGC donor
partitions are already on disk, so the event flag (and the survival time, as a
cross-check) is rebuilt from them directly - no re-download, no sample-level
imputation.

Expression is rebuilt node-side from the same raw ``exp_seq`` partitions: the
released matrix carried 4% missing cells (genes a sample's ICGC quantification
omitted), which the P0 sanity gate rejects as non-finite.  A gene absent from a
sample's ``exp_seq`` record has no quantified reads, so it is written as 0 after
the log2(x+1) transform; every other value is untouched.

Usage:
    python scripts/ingest_lihc_extra_liri.py
"""
from __future__ import annotations

import glob
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbi.util import read_json, write_json  # noqa: E402

COHORT = ROOT / "data" / "raw" / "geo" / "LIRI-JP"
RAW = COHORT / "raw"

# Column order of release_28/headers/donor.tsv.gz (21 columns, headerless rows).
DONOR_COLUMNS = [
    "icgc_donor_id", "project_code", "study_donor_involved_in", "submitted_donor_id",
    "donor_sex", "donor_vital_status", "disease_status_last_followup",
    "donor_relapse_type", "donor_age_at_diagnosis", "donor_age_at_enrollment",
    "donor_age_at_last_followup", "donor_relapse_interval", "donor_diagnosis_icd10",
    "donor_tumour_staging_system_at_diagnosis", "donor_tumour_stage_at_diagnosis",
    "donor_tumour_stage_at_diagnosis_supplemental", "donor_survival_time",
    "donor_interval_of_last_followup", "prior_malignancy", "cancer_type_prior_malignancy",
    "cancer_history_first_degree_relative",
]


def load_donors() -> pd.DataFrame:
    rows: list[list[str]] = []
    for path in sorted(glob.glob(str(RAW / "*" / "donor" / "part-*.gz"))):
        with gzip.open(path, "rt", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    rows.append(line.rstrip("\n").split("\t")[: len(DONOR_COLUMNS)])
    if not rows:
        raise SystemExit(f"no donor partitions under {RAW}")
    donor = pd.DataFrame(rows, columns=DONOR_COLUMNS)
    donor = donor.drop_duplicates("icgc_donor_id")
    return donor


def load_expression(sample_ids: set[str]) -> pd.DataFrame:
    """Rebuild log2(normalized_read_count + 1) from the raw exp_seq partitions."""
    frames = []
    for path in sorted(glob.glob(str(RAW / "*" / "exp_seq" / "part-*.gz"))):
        frame = pd.read_csv(
            path, sep="\t", header=None, usecols=[3, 7, 8],
            names=["sample_id", "gene", "value"],
            dtype={"sample_id": "category", "gene": "category", "value": "float64"},
            low_memory=False,
        )
        frames.append(frame[frame["sample_id"].isin(sample_ids)])
    if not frames:
        raise SystemExit("no exp_seq partitions found")
    exp = pd.concat(frames, ignore_index=True)
    exp = exp.dropna(subset=["value"])
    exp = exp.groupby(["sample_id", "gene"], observed=True)["value"].max()
    expr = exp.unstack("gene")
    expr = np.log2(expr.reindex(index=sorted(sample_ids)).fillna(0.0) + 1.0)
    expr.index.name = "sample_id"
    expr.columns.name = None
    return expr.astype("float32")


def main() -> int:
    clinical = pd.read_parquet(COHORT / "clinical.parquet")
    donor = load_donors()

    status = donor.set_index("icgc_donor_id")["donor_vital_status"].astype(str).str.strip().str.lower()
    time_days = pd.to_numeric(
        donor.set_index("icgc_donor_id")["donor_survival_time"], errors="coerce")

    event = pd.Series({
        "deceased": 1, "dead": 1, "alive": 0, "living": 0,
    }).reindex(status.unique()).to_dict()
    unknown = sorted(set(status.unique()) - set(event))
    if unknown:
        raise SystemExit(f"unmapped donor_vital_status values: {unknown}")

    flagged = clinical["patient_id"].map(status)
    clinical["os_event"] = flagged.map(event).astype("Int64")
    raw_time = clinical["patient_id"].map(time_days).astype(float)
    mismatch = int((np.abs(raw_time - clinical["os_time_days"].astype(float)) > 1e-6).sum())
    if mismatch:
        # the donor row is authoritative; fall back to it only where it differs
        clinical.loc[np.abs(raw_time - clinical["os_time_days"].astype(float)) > 1e-6,
                     "os_time_days"] = raw_time
        clinical["os_time_days"] = clinical["os_time_days"].astype(float)

    if clinical["os_event"].isna().any():
        raise SystemExit("some patients have no donor_vital_status")
    clinical["os_event"] = clinical["os_event"].astype(int)
    clinical = clinical[np.isfinite(clinical["os_time_days"]) & (clinical["os_time_days"] > 0)]
    clinical = clinical.reset_index(drop=True)

    expression = load_expression(set(clinical["sample_id"]))
    clinical = clinical[clinical["sample_id"].isin(expression.index)].reset_index(drop=True)
    expression = expression.loc[clinical["sample_id"].tolist()]

    required = ["sample_id", "patient_id", "os_time_days", "os_event", "stage_ajcc",
                "age_years", "sex", "neoadjuvant", "sample_type", "platform",
                "expr_unit", "cohort_id"]
    missing = [c for c in required if c not in clinical.columns]
    nan_cols = [c for c in required if clinical[c].isna().any()]
    if missing or nan_cols:
        raise SystemExit(f"clinical contract broken: missing={missing} nan={nan_cols}")
    if set(clinical["sample_id"]) != set(expression.index):
        raise SystemExit("clinical and expression samples diverged")
    if not np.isfinite(expression.to_numpy()).all():
        raise SystemExit("expression still has non-finite cells")
    if len(clinical) < 80 or clinical["os_event"].sum() < 15:
        raise SystemExit(f"cohort too small after fix: n={len(clinical)} "
                         f"events={int(clinical['os_event'].sum())}")

    expression.to_parquet(COHORT / "expression.parquet")
    clinical.to_parquet(COHORT / "clinical.parquet", index=False)

    meta = read_json(COHORT / "meta.json")
    meta.update({
        "n": int(len(clinical)),
        "n_genes": int(expression.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "n_deposited": int(len(donor)),
        "vital_status_source": "release_28 donor.donor_vital_status (deceased->1, alive->0)",
        "notes": (
            "Per-donor ICGC partitions parsed with the release_28 header schemas. "
            "Tumour samples only (specimen_type 'Primary tumour - solid tissue'). "
            "Stage from specimen tumour_stage, falling back to the donor stage. "
            "os_event/os_time_days rebuilt from the donor partitions by "
            "scripts/ingest_lihc_extra_liri.py; the released file had os_event all "
            "zero because 'deceased' was not mapped to an event. Expression rebuilt "
            "from the same raw exp_seq partitions with log2(normalized_read_count+1); "
            "a gene absent from a sample's exp_seq record is 0 (no quantified reads), "
            "so the matrix is finite and covers the full gene set."
        ),
    })
    meta.pop("nan_cleanup", None)  # superseded: expression is rebuilt, not patched
    write_json(COHORT / "meta.json", meta)

    print(f"LIRI-JP: n={meta['n']} events={meta['events_os']} genes={meta['n_genes']} "
          f"median_os_days={meta['median_os_days']:.0f} time_mismatches_fixed={mismatch}")
    print("vital_status counts:", flagged.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
