"""Per-cohort adapters for HCC GEO series whose OS fields the generic parser
in :mod:`cbi.cohort_pool.geo` does not recognise.

Two cohorts are handled:

* ``GSE10141`` -- Hoshida et al. training set (fixed-tissue HCC).  OS is in
  ``!Sample_characteristics_ch1`` as ``survival time (days)`` and
  ``survival status (0: alive or censored, 1: dead)``; probes are custom
  ``DAP1_*`` ids annotated on GPL5474 (column ``Symbol``).
* ``GSE14520`` -- Roessler et al. HCC cohort.  Expression ships as two
  platform-suffixed series matrices (GPL3921, GPL571); survival, stage, age and
  sex come from the series ``GSE14520_Extra_Supplement.txt.gz`` table, keyed by
  the ``Affy_GSM`` column.

Run::

    python scripts/ingest_lihc_geo_survival.py --gse GSE10141
    python scripts/ingest_lihc_geo_survival.py --gse GSE14520
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
PLATFORM_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"


def geo_dir(acc: str, prefix: str) -> str:
    num = acc[len(prefix):]
    return f"{prefix}{num[:-3]}nnn"


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cbi-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# series matrix helpers
# --------------------------------------------------------------------------- #
def read_header(path: Path) -> dict[str, list[str]]:
    """Return characteristic key -> per-sample raw values (first block wins)."""
    fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            values = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if not values or ":" not in values[0]:
                continue
            key = values[0].split(":", 1)[0].strip().lower()
            # Annotation-only keys ("survival status (0: alive, 1: dead)") keep
            # colons in the key text, so take the value after the *last* colon.
            fields.setdefault(key, [v.rsplit(":", 1)[1].strip() if ":" in v else "" for v in values])
    return fields


def read_matrix(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", comment="!", index_col=0)
    table.index = table.index.astype(str)
    table.columns = [str(c).strip().strip('"') for c in table.columns]
    return table.apply(pd.to_numeric, errors="coerce")


def read_annotation(path: Path) -> pd.DataFrame:
    """Parse a GEO annotation table (``.annot.gz`` or platform SOFT)."""
    opener = gzip.open if path.suffix == ".gz" else open
    skip = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[operator]
        for i, line in enumerate(fh):
            if line.startswith("!platform_table_begin"):
                skip = i + 1
                break
    if skip is None:
        raise ValueError(f"{path}: !platform_table_begin not found")
    df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, comment="#", compression="infer")
    id_col = next((c for c in df.columns if c.strip().lower() in {"id", "id_ref"}), None)
    sym_col = next((c for c in df.columns if c.strip().lower() in {"gene symbol", "symbol", "genesymbol"}), None)
    if id_col is None or sym_col is None:
        raise ValueError(f"{path}: ID / symbol columns not found: {list(df.columns)[:8]}")
    out = df[[id_col, sym_col]].rename(columns={id_col: "probe_id", sym_col: "gene"})
    out = out.dropna()
    out["gene"] = out["gene"].str.split("///", regex=False).str[0].str.strip()
    out = out[out["gene"].str.len() > 0]
    return out.drop_duplicates("probe_id")


def collapse(expr: pd.DataFrame, annot: pd.DataFrame) -> pd.DataFrame:
    annot = annot.set_index("probe_id")
    common = expr.index.intersection(annot.index)
    if len(common) == 0:
        raise ValueError("no probes shared by expression matrix and annotation")
    sub = expr.loc[common]
    order = sub.mean(axis=1).sort_values(ascending=False).index
    ann = annot.loc[order]
    keep = ~ann["gene"].duplicated()
    sub = sub.loc[order][keep.to_numpy()]
    sub.index = ann.loc[order, "gene"][keep].to_numpy()
    return sub


def split_stage(text: str) -> str:
    t = str(text).upper().replace("STAGE", "").strip()
    if t.startswith(("IV", "III", "II", "I")):
        for s in ("IV", "III", "II", "I"):
            if t.startswith(s):
                return s
    return "NA"


def to_sex(text: str) -> str:
    t = str(text).strip().lower()
    if t in {"m", "male", "1", "1.0"}:
        return "male"
    if t in {"f", "female", "2", "2.0"}:
        return "female"
    return "NA"


# --------------------------------------------------------------------------- #
# GSE10141
# --------------------------------------------------------------------------- #
def ingest_gse10141(out: Path) -> dict:
    gse = "GSE10141"
    gpl = "GPL5474"
    matrix = download(
        f"{SERIES_BASE}/{geo_dir(gse, 'GSE')}/{gse}/matrix/{gse}_series_matrix.txt.gz",
        out / f"{gse}_series_matrix.txt.gz",
    )
    annot_path = download(
        f"{PLATFORM_BASE}/{geo_dir(gpl, 'GPL')}/{gpl}/soft/{gpl}_family.soft.gz",
        out / f"{gpl}_family.soft.gz",
    )
    fields = read_header(matrix)
    time_key = next(k for k in fields if "survival time" in k)
    event_key = next(k for k in fields if "survival status" in k)
    title_key = None

    times = fields[time_key]
    events = fields[event_key]
    sample_ids = read_matrix(matrix).columns.tolist()
    titles: list[str] = []
    with gzip.open(matrix, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                break

    rows = []
    for i, sid in enumerate(sample_ids):
        ev = re.sub(r"[^0-9]", "", str(events[i])) if i < len(events) else ""
        t = pd.to_numeric(str(times[i]).strip(), errors="coerce") if i < len(times) else np.nan
        if ev not in {"0", "1"} or not np.isfinite(t) or t <= 0:
            continue
        m = re.search(r"\(([^)]+)\)", titles[i]) if i < len(titles) else None
        patient = m.group(1) if m else sid
        rows.append({
            "sample_id": sid,
            "patient_id": patient,
            "os_time_days": float(t),
            "os_event": int(ev),
            "stage_ajcc": "NA",
            "age_years": np.nan,
            "sex": "NA",
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": gpl,
            "expr_unit": "log2_intensity",
            "cohort_id": gse,
        })
    clinical = pd.DataFrame(rows)
    expr = collapse(read_matrix(matrix), read_annotation(annot_path))
    expr = np.log2(expr.clip(lower=0) + 1.0)
    expr = expr.T.reindex(clinical["sample_id"].tolist())

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": gse,
        "cancer_type": "LIHC",
        "accession": gse,
        "platform": gpl,
        "expr_unit": "log2_intensity",
        "source": f"GEO:{gse}",
        "license": "public GEO record; see the series citation and GEO terms",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": (
            "Hoshida et al. fixed-tissue HCC training set. OS parsed from "
            "'survival time (days)' / 'survival status' characteristics; probes mapped "
            "through GPL5474 'Symbol'. Age and sex are not part of the GEO record "
            "(recorded as NaN / 'NA')."
        ),
    }
    write_json(out / "meta.json", meta)
    return meta


# --------------------------------------------------------------------------- #
# GSE14520
# --------------------------------------------------------------------------- #
def ingest_gse14520(out: Path) -> dict:
    gse = "GSE14520"
    platforms = ["GPL3921", "GPL571"]
    suppl = download(
        f"{SERIES_BASE}/{geo_dir(gse, 'GSE')}/{gse}/suppl/{gse}_Extra_Supplement.txt.gz",
        out / f"{gse}_Extra_Supplement.txt.gz",
    )
    clin = pd.read_csv(suppl, sep="\t", dtype=str)
    clin["Affy_GSM"] = clin["Affy_GSM"].str.strip()

    frames = []
    used_platforms = []
    for gpl in platforms:
        matrix_path = download(
            f"{SERIES_BASE}/{geo_dir(gse, 'GSE')}/{gse}/matrix/{gse}-{gpl}_series_matrix.txt.gz",
            out / f"{gse}-{gpl}_series_matrix.txt.gz",
        )
        annot_path = download(
            f"{PLATFORM_BASE}/{geo_dir(gpl, 'GPL')}/{gpl}/annot/{gpl}.annot.gz",
            out / f"{gpl}.annot.gz",
        )
        expr = read_matrix(matrix_path)
        expr = collapse(expr, read_annotation(annot_path))
        frames.append(expr)
        used_platforms.append((gpl, expr.shape[1]))

    genes = frames[0].index
    for frame in frames[1:]:
        genes = genes.union(frame.index)
    combined = pd.concat([f.reindex(genes) for f in frames], axis=1)

    tumor = clin[clin["Tissue Type"].astype(str).str.strip().str.lower() == "tumor"].copy()
    rows = []
    for _, r in tumor.iterrows():
        sid = r["Affy_GSM"]
        if sid not in combined.columns:
            continue
        ev = pd.to_numeric(r.get("Survival status"), errors="coerce")
        months = pd.to_numeric(r.get("Survival months"), errors="coerce")
        if not np.isfinite(ev) or not np.isfinite(months) or months <= 0:
            continue
        rows.append({
            "sample_id": sid,
            "patient_id": str(r.get("LCS ID") or sid),
            "os_time_days": float(months) * 30.4375,
            "os_event": int(ev),
            "stage_ajcc": split_stage(r.get("TNM staging")),
            "age_years": pd.to_numeric(r.get("Age"), errors="coerce"),
            "sex": to_sex(r.get("Gender")),
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": "GPL3921" if sid in set(frames[0].columns) else "GPL571",
            "expr_unit": "log2_intensity",
            "cohort_id": gse,
        })
    clinical = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    expr = combined[clinical["sample_id"].tolist()].T

    clinical.to_parquet(out / "clinical.parquet", index=False)
    expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": gse,
        "cancer_type": "LIHC",
        "accession": gse,
        "platform": "GPL3921+GPL571",
        "expr_unit": "log2_intensity",
        "source": f"GEO:{gse}",
        "license": "public GEO record; see the series citation and GEO terms",
        "n": int(len(clinical)),
        "n_genes": int(expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": (
            "Roessler et al. HCC cohort. Tumour samples only; OS from "
            "GSE14520_Extra_Supplement ('Survival status', 'Survival months' -> days) "
            "joined on the 'Affy_GSM' column. Expression combines the GPL3921 and "
            "GPL571 series matrices (already log2-normalised by GEO); genes missing on "
            "one platform are NaN for those samples."
        ),
    }
    write_json(out / "meta.json", meta)
    return meta


INGESTERS = {"GSE10141": ingest_gse10141, "GSE14520": ingest_gse14520}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ingest-lihc-geo-survival")
    ap.add_argument("--gse", required=True, choices=sorted(INGESTERS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or ROOT / "data" / "raw" / "geo" / args.gse)
    info = INGESTERS[args.gse](out)
    print(f"{info['cohort_id']}: {info['n']} samples, {info['events_os']} OS events, "
          f"{info['n_genes']} genes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
