"""Ingest the ICGC LIRI-JP (Japanese HCC, PCAWG) open cohort as a standard
M2 cohort.

Source: ICGC release 28 open data, mirrored on the public S3 bucket
``https://object.genomeinformatics.org/icgc25k-open`` under
``release_28/data/LIRI-JP/<DONOR>/{donor,sample,specimen,exp_seq}/part-*.gz``.
The data are per-donor partitions with headerless rows; the column order is
documented in ``release_28/headers/*.tsv.gz`` (read at run time, not hard-coded).

Output (``data/raw/geo/LIRI-JP``):
  clinical.parquet    standard clinical schema, one row per tumour sample
  expression.parquet  log2(normalized_read_count + 1), rows = sample_id
  meta.json           provenance

Run::

    python scripts/ingest_lihc_icgc.py
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://object.genomeinformatics.org/icgc25k-open"
PROJECT = "LIRI-JP"
PROJECT_PREFIX = f"release_28/data/{PROJECT}/"
OUT_DIR = ROOT / "data" / "raw" / "geo" / PROJECT
RAW_DIR = OUT_DIR / "raw"
COHORT_ID = PROJECT

HEADERS: dict[str, list[str]] = {}


def configure(project: str) -> None:
    global PROJECT, PROJECT_PREFIX, OUT_DIR, RAW_DIR, COHORT_ID
    PROJECT = project
    PROJECT_PREFIX = f"release_28/data/{project}/"
    OUT_DIR = ROOT / "data" / "raw" / "geo" / project
    RAW_DIR = OUT_DIR / "raw"
    COHORT_ID = project


# --------------------------------------------------------------------------- #
# S3 helpers
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cbi-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={urllib.parse.quote(prefix, safe='/')}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        xml = _get(url).decode("utf-8")
        keys.extend(part.split("</Key>")[0] for part in xml.split("<Key>")[1:])
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        token = xml.split("<NextContinuationToken>")[1].split("</NextContinuationToken>")[0]
    return keys


def read_header(name: str) -> list[str]:
    raw = _get(f"{BUCKET}/release_28/headers/{name}.tsv.gz")
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")


def fetch_part(key: str) -> pd.DataFrame | None:
    rel = key[len(PROJECT_PREFIX):]
    local = RAW_DIR / rel
    if not local.is_file() or local.stat().st_size == 0:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(_get(f"{BUCKET}/{key}"))
    kind = rel.split("/")[1]
    columns = HEADERS[kind]
    try:
        with gzip.open(local, "rt") as fh:
            rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    except OSError:
        return None
    return pd.DataFrame(
        [(r + [""] * len(columns))[: len(columns)] for r in rows], columns=columns)


# --------------------------------------------------------------------------- #
# clinical helpers
# --------------------------------------------------------------------------- #
STAGE_MAP = {
    "1": "I", "2": "II", "3": "III", "4": "IV",
    "1a": "I", "1b": "I", "2a": "II", "2b": "II",
    "3a": "III", "3b": "III", "3c": "III",
    "4a": "IV", "4b": "IV",
    "i": "I", "ii": "II", "iii": "III", "iv": "IV",
}


def norm_stage(raw: str) -> str:
    s = str(raw).strip().lower().replace("stage", "").replace(" ", "")
    if s.startswith("t"):
        # TNM string such as t1n0m0 / t3bn0m0 (AJCC 7th HCC grouping).
        import re as _re
        m = _re.match(r"t(\d)([ab]?)n(\d)m(\d)", s)
        if m:
            t, n, mm = m.group(1), m.group(3), m.group(4)
            if mm == "1" or n == "1":
                return "IV"
            return {"1": "I", "2": "II", "3": "III", "4": "III"}.get(t, "NA")
        return "NA"
    s = (s.replace("iiia", "3a").replace("iiib", "3b").replace("iiic", "3c")
          .replace("iva", "4a").replace("ivb", "4b"))
    return STAGE_MAP.get(s, "NA")


def is_primary(specimen_type: str) -> bool:
    s = str(specimen_type).lower()
    return "primary" in s and "tumour" in s


def norm_sex(raw: str) -> str:
    s = str(raw).strip().lower()
    return {"male": "male", "female": "female", "m": "male", "f": "female"}.get(s, "NA")


HGNC_TSV = ROOT / "data" / "raw" / "hgnc" / "hgnc_complete.tsv"


def ensembl_to_symbol() -> dict[str, str]:
    """ensembl_gene_id -> HGNC symbol, from the repo's local HGNC table."""
    table = pd.read_csv(HGNC_TSV, sep="\t", usecols=["symbol", "ensembl_gene_id"],
                        dtype=str, low_memory=False)
    mapping: dict[str, str] = {}
    for sym, eid in zip(table["symbol"], table["ensembl_gene_id"]):
        if not isinstance(sym, str) or not isinstance(eid, str):
            continue
        for one in eid.split("|"):
            one = one.strip()
            if one:
                mapping.setdefault(one, sym)
    return mapping


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #
def build(workers: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for name in ("donor", "sample", "specimen", "exp_seq"):
        HEADERS[name] = read_header(name)

    keys = [k for k in list_keys(PROJECT_PREFIX)
            if k.split("/")[-2] in {"donor", "sample", "specimen", "exp_seq"}]
    print(f"{len(keys)} partition files")
    with ThreadPoolExecutor(workers) as pool:
        frames = [f for f in pool.map(fetch_part, sorted(keys)) if f is not None and len(f)]

    def concat(kind: str) -> pd.DataFrame:
        sub = [f for f in frames if f.columns.equals(pd.Index(HEADERS[kind]))]
        return pd.concat(sub, ignore_index=True).drop_duplicates() if sub else pd.DataFrame()

    donor = concat("donor")
    sample = concat("sample")
    specimen = concat("specimen")
    exp = concat("exp_seq")
    print(f"donor={len(donor)} sample={len(sample)} specimen={len(specimen)} exp_rows={len(exp)}")

    # ---- clinical -------------------------------------------------------- #
    clin = sample[["icgc_sample_id", "icgc_donor_id", "icgc_specimen_id"]].drop_duplicates(
        "icgc_sample_id")
    clin = clin.merge(specimen[["icgc_specimen_id", "specimen_type", "tumour_stage"]],
                      on="icgc_specimen_id", how="left")
    clin = clin.merge(
        donor[["icgc_donor_id", "donor_sex", "donor_vital_status", "donor_survival_time",
               "donor_age_at_diagnosis", "donor_tumour_stage_at_diagnosis"]],
        on="icgc_donor_id", how="left")
    clin = clin[clin["specimen_type"].map(is_primary)].copy()

    stage = clin["tumour_stage"].where(clin["tumour_stage"].astype(str).str.strip() != "",
                                       clin["donor_tumour_stage_at_diagnosis"])
    clinical = pd.DataFrame({
        "sample_id": clin["icgc_sample_id"].to_numpy(),
        "patient_id": clin["icgc_donor_id"].to_numpy(),
        "os_time_days": pd.to_numeric(clin["donor_survival_time"], errors="coerce").to_numpy(),
        "os_event": clin["donor_vital_status"].astype(str).str.lower()
                    .isin({"dead", "deceased"}).astype(int).to_numpy(),
        "stage_ajcc": [norm_stage(v) for v in stage],
        "age_years": pd.to_numeric(clin["donor_age_at_diagnosis"], errors="coerce").to_numpy(),
        "sex": [norm_sex(v) for v in clin["donor_sex"]],
        "neoadjuvant": False,
        "sample_type": "primary",
        "platform": "ICGC_RNA-Seq",
        "expr_unit": "log2_normalized_read_count",
        "cohort_id": COHORT_ID,
    })
    clinical = clinical.dropna(subset=["os_time_days", "age_years"])
    clinical = clinical[(clinical["os_time_days"] > 0)
                        & (clinical["sex"] != "NA")
                        & (clinical["stage_ajcc"] != "NA")]
    clinical["age_years"] = clinical["age_years"].astype(int)
    clinical["os_event"] = clinical["os_event"].astype(int)

    # ---- expression ------------------------------------------------------ #
    exp = exp[exp["icgc_sample_id"].isin(set(clinical["sample_id"]))]
    exp = exp[exp["gene_id"].astype(str).str.strip().ne("")]
    exp["value"] = pd.to_numeric(exp["normalized_read_count"], errors="coerce")
    exp = exp.dropna(subset=["value"])
    exp["value"] = np.log2(exp["value"] + 1.0)
    gene_ids = exp["gene_id"].astype(str).str.strip()
    if gene_ids.str.startswith("ENSG").any():
        mapping = ensembl_to_symbol()
        exp = exp.assign(gene_symbol=gene_ids.str.split(".").str[0].map(mapping))
        exp = exp[exp["gene_symbol"].notna()]
        gene_col = "gene_symbol"
    else:
        gene_col = "gene_id"
    expr = exp.pivot_table(index="icgc_sample_id", columns=gene_col, values="value",
                           aggfunc="max")
    expr.columns.name = None
    expr.index.name = "sample_id"

    keep = clinical["sample_id"].isin(expr.index)
    clinical = clinical.loc[keep].reset_index(drop=True)
    expr = expr.loc[clinical["sample_id"]]
    # cbi.cohort_pool.sanity requires a finite matrix: keep samples with a full
    # gene profile and genes measured in every retained sample.
    expr = expr.loc[expr.notna().all(axis=1)]
    expr = expr.dropna(axis=1)
    clinical = clinical.loc[clinical["sample_id"].isin(expr.index)].reset_index(drop=True)
    expr = expr.loc[clinical["sample_id"]]
    if len(expr) < 80:
        print(f"WARNING: only {len(expr)} samples survive the finite-matrix filter")
    print(f"clinical={clinical.shape} expression={expr.shape} "
          f"events={int(clinical['os_event'].sum())}")

    meta = {
        "cohort_id": COHORT_ID,
        "cancer_type": "LIHC",
        "accession": "ICGC:LIRI-JP (release 28, open)",
        "platform": "Illumina HiSeq RNA-Seq",
        "expr_unit": "log2_normalized_read_count",
        "source": f"https://object.genomeinformatics.org/icgc25k-open/{PROJECT_PREFIX}",
        "license": "ICGC open data (release 28); cite ICGC/TCGA PCAWG Pan-Cancer consortium",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": (
            "Per-donor ICGC partitions parsed with the release_28 header schemas. "
            "Primary tumour samples only; samples without OS time, age, sex or an "
            "AJCC stage are dropped. Stage from specimen tumour_stage, falling back "
            "to the donor stage (numeric or TNM string)."
        ),
    }
    return clinical, expr, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-lihc-icgc")
    parser.add_argument("--project", default=PROJECT,
                        help="ICGC project code under release_28/data/ (e.g. LIRI-JP, LICA-FR)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    configure(args.project)
    out = Path(args.out) if args.out else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    clinical, expr, meta = build(args.workers)
    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
