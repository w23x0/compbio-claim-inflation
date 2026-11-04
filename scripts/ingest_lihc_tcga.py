"""Build the standard cohort schema for TCGA-LIHC from open TCGA data.

Expression and clinical tables are the open-access TCGA-LIHC distributions
mirrored by UCSC Xena (GDC harmonised data).  OS is taken from the
``LIHC_survival.txt`` table (days, ``OS`` 1=dead) and the clinical matrix
supplies age, sex, stage and sample type.  Only primary tumour samples with a
usable OS record are kept.

Run::

    python scripts/ingest_lihc_tcga.py --out data/raw/geo/TCGA-LIHC
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XENA = "https://tcga-xena-hub.s3.amazonaws.com/download"
EXPRESSION_URL = f"{XENA}/TCGA.LIHC.sampleMap%2FHiSeqV2.gz"
SURVIVAL_URL = f"{XENA}/survival%2FLIHC_survival.txt"
CLINICAL_URL = f"{XENA}/TCGA.LIHC.sampleMap%2FLIHC_clinicalMatrix"
COHORT = "TCGA-LIHC"


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cbi-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def split_stage(value: object) -> str:
    text = str(value).upper()
    if not text or text in {"NAN", "NA", "[NOT AVAILABLE]", "NOT REPORTED"}:
        return "NA"
    for stage in ("IV", "III", "II", "I"):
        if f"STAGE {stage}" in text or text.strip() == stage:
            return stage
    return "NA"


def to_sex(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return "NA"


def build(out: Path) -> dict:
    expr_path = download(EXPRESSION_URL, out / "TCGA.LIHC.HiSeqV2.gz")
    surv_path = download(SURVIVAL_URL, out / "LIHC_survival.txt")
    clin_path = download(CLINICAL_URL, out / "LIHC_clinicalMatrix")

    survival = pd.read_csv(surv_path, sep="\t")
    clin = pd.read_csv(clin_path, sep="\t", low_memory=False)

    clin = clin.set_index("sampleID")
    frames = []
    for _, row in survival.iterrows():
        sid = str(row["sample"])
        if sid not in clin.index:
            continue
        rec = clin.loc[sid]
        os_days = pd.to_numeric(row.get("OS.time"), errors="coerce")
        os_event = pd.to_numeric(row.get("OS"), errors="coerce")
        if not np.isfinite(os_days) or not np.isfinite(os_event) or os_days <= 0:
            continue
        if str(rec.get("sample_type", "")).strip().lower() != "primary tumor":
            continue
        frames.append({
            "sample_id": sid,
            "patient_id": str(row.get("_PATIENT") or sid[:12]),
            "os_time_days": float(os_days),
            "os_event": int(os_event),
            "stage_ajcc": split_stage(rec.get("pathologic_stage")),
            "age_years": pd.to_numeric(rec.get("age_at_initial_pathologic_diagnosis"), errors="coerce"),
            "sex": to_sex(rec.get("gender")),
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": "Illumina HiSeq (RNA-seq)",
            "expr_unit": "log2(norm_count+1)",
            "cohort_id": COHORT,
        })
    clinical = pd.DataFrame(frames).dropna(subset=["age_years"]).reset_index(drop=True)
    if clinical.empty:
        raise ValueError("no TCGA-LIHC samples with complete OS and age")

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr = expr.drop(columns=[c for c in expr.columns if c not in set(clinical["sample_id"])])
    expr = expr.T
    expr = expr.reindex(clinical["sample_id"].tolist())
    expr.index.name = None

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": COHORT,
        "cancer_type": "LIHC",
        "accession": COHORT,
        "platform": "Illumina HiSeq (RNA-seq)",
        "expr_unit": "log2(norm_count+1)",
        "source": "TCGA-LIHC open access (GDC harmonised, UCSC Xena mirror)",
        "license": "TCGA open-access data; see the TCGA data-use policy",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": (
            "Primary tumour samples only; OS from LIHC_survival.txt (days, 1=dead); "
            "age/sex/stage from the Xena LIHC clinical matrix; samples without a "
            "recorded age were dropped so the required clinical columns are complete."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ingest-lihc-tcga")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or ROOT / "data" / "raw" / "geo" / COHORT)
    info = build(out)
    print(f"{info['cohort_id']}: {info['n']} samples, {info['events_os']} OS events, "
          f"{info['n_genes']} genes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
