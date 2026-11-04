"""Download and convert a public GEO series matrix into the standard schema.

This is the automated path for real public cohorts; it does not require manual
sample-level editing.  The parser understands the common `!Sample_characteristics_ch1`
layout (key: value per sample) and the platform annotation table.  If a required
field is absent the function fails loudly with the available keys instead of
silently fabricating values.

Example:

    python -m cbi.cohort_pool.geo --gse GSE76427 --out data/raw/geo/GSE76427
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbi.util import write_json

SERIES_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
PLATFORM_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"


def _geo_dir(acc: str, prefix: str) -> str:
    """GSE76427 -> GSE76nnn; GPL10558 -> GPL10nnn."""
    num = acc[len(prefix):]
    return f"{prefix}{num[:-3]}nnn"


def series_matrix_url(gse: str) -> str:
    gse = gse.upper()
    return f"{SERIES_BASE}/{_geo_dir(gse, 'GSE')}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def platform_annotation_url(gpl: str) -> str:
    gpl = gpl.upper()
    return f"{PLATFORM_BASE}/{_geo_dir(gpl, 'GPL')}/{gpl}/annot/{gpl}.annot.gz"


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def _field(fields: dict[str, list[str]], *names: str) -> list[str] | None:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _split_stage(value: str) -> str:
    text = str(value).upper().replace("STAGE", "").strip()
    for stage in ("IV", "III", "II", "I"):
        if text.startswith(stage):
            return stage
    return "NA"


def _parse_event(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "dead", "deceased", "event", "yes", "true"}:
        return 1
    if text in {"0", "0.0", "alive", "censored", "no", "false"}:
        return 0
    return None


def _parse_sex(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "m", "male"}:
        return "male"
    if text in {"2", "2.0", "f", "female"}:
        return "female"
    return text or "NA"


def parse_series_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (clinical, expression, series_meta).

    clinical is one row per GSM sample, expression is samples x gene symbols.
    """
    path = Path(path)
    fields: dict[str, list[str]] = {}
    sample_ids: list[str] = []
    platform: str | None = None
    title = ""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            parts = line.rstrip("\n").split("\t")
            key = parts[0]
            values = [p.strip().strip('"') for p in parts[1:]]
            if key == "!Sample_geo_accession":
                sample_ids = values
            elif key == "!Sample_characteristics_ch1":
                first = values[0] if values else ""
                name = first.split(":", 1)[0].strip().lower() if ":" in first else ""
                if name:
                    fields[name] = [v.split(":", 1)[1].strip() if ":" in v else v.strip() for v in values]
            elif key == "!Series_platform_id":
                platform = values[0] if values else None
            elif key == "!Series_title":
                title = values[0] if values else ""
    if not sample_ids:
        raise ValueError(f"{path}: no !Sample_geo_accession row")
    if platform is None:
        raise ValueError(f"{path}: no !Series_platform_id row")

    event_values = _field(fields, "event_os", "os_event", "vital_status", "status")
    time_values = _field(fields, "duryears_os", "os_time_years", "survival_time_years", "duryears")
    if event_values is None or time_values is None:
        raise ValueError(f"{path}: OS fields not found; available keys: {sorted(fields)}")

    time_multiplier = 365.25 if any(k in fields for k in ("duryears_os", "os_time_years", "survival_time_years")) else 1.0
    ages = _field(fields, "age (years)", "age", "age_years")
    sexes = _field(fields, "gender (1=m, 2=f)", "sex", "gender")
    stages = _field(fields, "tnm_staging_clinical", "stage", "ajcc_stage")
    patients = _field(fields, "patient id", "patient_id")
    tissues = _field(fields, "tissue", "sample_type")

    rows = []
    for i, sample_id in enumerate(sample_ids):
        event = _parse_event(event_values[i]) if i < len(event_values) else None
        try:
            time_days = float(time_values[i]) * time_multiplier
        except (ValueError, TypeError):
            time_days = np.nan
        if event is None or not np.isfinite(time_days) or time_days <= 0:
            continue
        rows.append({
            "sample_id": sample_id,
            "patient_id": patients[i] if patients and i < len(patients) else sample_id,
            "os_time_days": time_days,
            "os_event": event,
            "stage_ajcc": _split_stage(stages[i]) if stages and i < len(stages) else "NA",
            "age_years": pd.to_numeric(ages[i], errors="coerce") if ages and i < len(ages) else np.nan,
            "sex": _parse_sex(sexes[i]) if sexes and i < len(sexes) else "NA",
            "neoadjuvant": False,
            "sample_type": (tissues[i].lower() if tissues and i < len(tissues) else "primary"),
            "platform": platform,
            "expr_unit": "log2_intensity",
        })
    clinical = pd.DataFrame(rows)
    if clinical.empty:
        raise ValueError(f"{path}: no samples with valid OS time/event")
    tissue = clinical["sample_type"].astype(str).str.lower()
    clinical["sample_type"] = np.where(
        tissue.str.contains("tumour|tumor|primary", regex=True), "primary",
        np.where(tissue.str.contains("normal|adjacent"), "normal", tissue),
    )

    table = pd.read_csv(path, sep="\t", comment="!", index_col=0)
    table.index = table.index.astype(str)
    table.columns = [str(c).strip().strip('"') for c in table.columns]
    table = table.apply(pd.to_numeric, errors="coerce")

    return clinical, table, {"platform": platform, "sample_ids": sample_ids, "title": title}


def parse_platform_annotation(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    skip = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("!platform_table_begin"):
                skip = i + 1
                break
    if skip is None:
        raise ValueError(f"{path}: !platform_table_begin not found")
    df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, comment="!")
    id_col = next((c for c in df.columns if c.strip().lower() in {"id", "id_ref"}), None)
    symbol_col = next((c for c in df.columns if c.strip().lower() == "gene symbol"), None)
    if id_col is None or symbol_col is None:
        raise ValueError(f"{path}: ID / Gene symbol columns not found: {list(df.columns)[:6]}")
    out = df[[id_col, symbol_col]].rename(columns={id_col: "probe_id", symbol_col: "gene"})
    out = out.dropna()
    out["gene"] = out["gene"].str.split("///", regex=False).str[0].str.strip()
    out = out[out["gene"].str.len() > 0]
    return out


def collapse_probes(expression: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Map probes to genes, keeping the max-mean probe per gene."""
    annotation = annotation.drop_duplicates("probe_id").set_index("probe_id")
    common = expression.index.intersection(annotation.index)
    if len(common) == 0:
        raise ValueError("no probes shared by expression matrix and annotation")
    expr = expression.loc[common]
    mean = expr.mean(axis=1)
    ann = annotation.loc[common].copy()
    ann["mean"] = mean
    chosen = ann.sort_values("mean", ascending=False).drop_duplicates("gene")
    expr = expr.loc[chosen.index]
    expr.index = chosen["gene"].to_numpy()
    return expr.T


def _infer_cancer(title: str) -> str:
    low = title.lower()
    for phrase, code in [
        ("hepatocellular", "LIHC"), ("liver", "LIHC"),
        ("breast", "BRCA"), ("lung", "LUAD"),
        ("kidney", "KIRC"), ("renal", "KIRC"),
    ]:
        if phrase in low:
            return code
    return "NA"


def ingest_geo(gse: str, out_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    gse = gse.upper()
    out_dir = Path(out_dir)
    series_path = download(series_matrix_url(gse), out_dir / f"{gse}_series_matrix.txt.gz", force=force)
    clinical, probe_expr, meta = parse_series_matrix(series_path)
    gpl = str(meta["platform"]).upper()
    annot_path = download(platform_annotation_url(gpl), out_dir / f"{gpl}.annot.gz", force=force)
    annotation = parse_platform_annotation(annot_path)
    gene_expr = collapse_probes(probe_expr, annotation)
    gene_expr = np.log2(gene_expr.clip(lower=0) + 1.0)
    gene_expr = gene_expr.reindex(clinical["sample_id"].tolist())
    clinical.to_parquet(out_dir / "clinical.parquet", index=False)
    gene_expr.to_parquet(out_dir / "expression.parquet")
    info = {
        "cohort_id": gse,
        "cancer_type": _infer_cancer(meta.get("title", "")),
        "accession": gse,
        "platform": gpl,
        "expr_unit": "log2_intensity",
        "source": f"GEO:{gse}",
        "license": "public GEO record; check the series citation and GEO terms",
        "n": int(len(clinical)),
        "n_deposited": int(len(clinical)),
        "n_genes": int(gene_expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "notes": "Downloaded and normalized by cbi.cohort_pool.geo; no manual sample editing.",
    }
    write_json(out_dir / "meta.json", info)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-geo-ingest")
    parser.add_argument("--gse", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out or f"data/raw/geo/{args.gse.upper()}")
    info = ingest_geo(args.gse, out, force=args.force)
    print(f"{info['cohort_id']}: {info['n']} samples, {info['events_os']} OS events, {info['n_genes']} genes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
