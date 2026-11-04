#!/usr/bin/env python3
"""Ingest public lung-adenocarcinoma GEO cohorts into the P0 standard contract.

The generic ``cbi.cohort_pool.geo`` parser assumes that every
``!Sample_characteristics_ch1`` row uses one key across all samples.  Several
LUAD series (GSE31210, GSE26939) emit characteristics in a per-sample order,
and the survival keys differ per series, so this adapter reads the header
sample by sample and applies an explicit per-cohort field map.

Output per cohort (data/raw/geo/<GSE>/):
    clinical.parquet, expression.parquet, meta.json, plus the raw series
    matrix and platform annotation that were downloaded.

Only open GEO records are used.
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbi.util import write_json  # noqa: E402

SERIES_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
PLATFORM_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"
MONTH_DAYS = 30.4375
YEAR_DAYS = 365.25
NA_TOKENS = {"", "na", "n/a", "nan", "none", "--", "-", "unknown", "not known", "missing"}

# ---------------------------------------------------------------------------
# Per-cohort field maps.  ``include``/``exclude`` filter the series down to
# primary lung adenocarcinomas; ``time`` is (characteristic key, days factor).
# ---------------------------------------------------------------------------
COHORTS: dict[str, dict] = {
    "GSE68465": {
        "platform": "GPL96",
        "cancer_type": "LUAD",
        "include": {"disease_state": {"lung adenocarcinoma"}},
        "time": ("months_to_last_contact_or_death", MONTH_DAYS),
        "event": ("vital_status", {"dead": 1, "alive": 0}),
        "age": "age",
        "sex": "Sex",
        "stage": "disease_stage",
        "stage_mode": "tnm",
        "expr_unit": "log2_intensity",
        "log_transform": True,
        "notes": "Director's Challenge multi-site LUAD validation set; OS from months_to_last_contact_or_death.",
    },
    "GSE31210": {
        "platform": "GPL570",
        "cancer_type": "LUAD",
        "include": {"tissue": {"primary lung tumor"}},
        "exclude": {"exclude for prognosis analysis due to incomplete resection or adjuvant therapy": {"exclude"}},
        "time": ("days before death/censor", 1.0),
        "event": ("death", {"dead": 1, "alive": 0}),
        "age": "age (years)",
        "sex": "gender",
        "stage": "pathological stage",
        "stage_mode": "roman",
        "expr_unit": "log2_intensity",
        "log_transform": True,
        "notes": "Japanese stage I-II LUAD; samples flagged 'exclude' for incomplete resection/adjuvant therapy removed.",
    },
    "GSE13213": {
        "platform": "GPL6480",
        "cancer_type": "LUAD",
        "time": ("Survival (days)", 1.0),
        "event": ("Status", {"dead": 1, "alive": 0}),
        "age": "Age",
        "sex": "Sex",
        "stage": "Stage (Pathological )",
        "stage_mode": "roman",
        "expr_unit": "log2_ratio",
        "log_transform": False,
        "notes": "Japanese LUAD (two datasets); all samples are adenocarcinoma.",
    },
    "GSE50081": {
        "platform": "GPL570",
        "cancer_type": "LUAD",
        "include": {"histology": {"adenocarcinoma", "nsclc-favor adenocarcinoma"}},
        "time": ("survival time", YEAR_DAYS),
        "event": ("status", {"dead": 1, "alive": 0}),
        "age": "age",
        "sex": "Sex",
        "stage": "Stage",
        "stage_mode": "roman",
        "expr_unit": "log2_intensity",
        "log_transform": False,
        "notes": "Early-stage NSCLC series; restricted to adenocarcinoma histology. Survival time reported in years.",
    },
    "GSE37745": {
        "platform": "GPL570",
        "cancer_type": "LUAD",
        "include": {"histology": {"adeno"}},
        "time": ("days to determined death status", 1.0),
        "event": ("dead", {"yes": 1, "no": 0}),
        "age": "age",
        "sex": "gender",
        "stage": "tumor stage",
        "stage_mode": "roman",
        "expr_unit": "log2_intensity",
        "log_transform": False,
        "notes": "NSCLC series (Karolinska); restricted to adenocarcinoma histology.",
    },
    "GSE11969": {
        "platform": "GPL7015",
        "cancer_type": "LUAD",
        "include": {"Histology": {"AD"}},
        "time": ("Survival (days)", 1.0),
        "event": ("Status", {"dead": 1, "alive": 0}),
        "age": "Age",
        "sex": "Sex",
        "stage": "pStage",
        "stage_mode": "roman",
        "expr_unit": "log2_ratio",
        "log_transform": False,
        "notes": "Japanese NSCLC series; restricted to adenocarcinoma (AD) histology.",
    },
    "GSE26939": {
        "platform": "GPL9053",
        "cancer_type": "LUAD",
        "time": ("survival_months", MONTH_DAYS),
        "event": ("survival_status(0='alive',1='dead')", {"1": 1, "0": 0}),
        "age": "age (90=greater than or equal to 90)",
        "sex": "Sex",
        "stage": "Stage",
        "stage_mode": "roman",
        "expr_unit": "log2_ratio",
        "log_transform": False,
        "notes": "LUAD cohort with mRNA subtypes; survival_months converted to days.",
    },
}


def geo_subdir(acc: str) -> str:
    num = acc[3:]
    return f"GSE{num[:-3]}nnn" if len(num) > 3 else "GSEnnn"


def platform_subdir(gpl: str) -> str:
    num = gpl[3:]
    return f"GPL{num[:-3]}nnn" if len(num) > 3 else "GPLnnn"


def series_matrix_url(gse: str) -> str:
    return f"{SERIES_BASE}/{geo_subdir(gse)}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def platform_annot_url(gpl: str) -> str:
    return f"{PLATFORM_BASE}/{platform_subdir(gpl)}/{gpl}/annot/{gpl}.annot.gz"


def platform_soft_url(gpl: str) -> str:
    return f"{PLATFORM_BASE}/{platform_subdir(gpl)}/{gpl}/soft/{gpl}_family.soft.gz"


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.is_file() and dest.stat().st_size > 5000 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cbi-luad-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=900) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if tmp.stat().st_size < 5000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small (likely 404): {url}")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Series matrix parsing (per-sample characteristics)
# ---------------------------------------------------------------------------
def parse_series_header(path: Path) -> tuple[list[str], list[dict[str, str]], str, str]:
    rows: list[list[str]] = []
    sample_ids: list[str] = []
    platform = ""
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
                rows.append(values)
            elif key == "!Series_platform_id":
                platform = values[0] if values else ""
            elif key == "!Series_title":
                title = values[0] if values else ""
    if not sample_ids:
        raise ValueError(f"{path}: no !Sample_geo_accession row")
    samples: list[dict[str, str]] = []
    for i in range(len(sample_ids)):
        rec: dict[str, str] = {}
        for row in rows:
            if i >= len(row):
                continue
            value = row[i]
            if ":" not in value:
                continue
            name, val = value.split(":", 1)
            rec[name.strip()] = val.strip()
        samples.append(rec)
    return sample_ids, samples, platform, title


# ---------------------------------------------------------------------------
# Platform annotation
# ---------------------------------------------------------------------------
def _read_platform_table(path: Path) -> pd.DataFrame:
    lines: list[str] = []
    started = False
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                started = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if started:
                lines.append(line)
    if not lines:
        raise ValueError(f"{path}: no platform table")
    from io import StringIO

    return pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str)


def load_annotation(gpl: str, out_dir: Path) -> pd.DataFrame:
    """Return a probe_id / gene frame, falling back to the SOFT family file."""
    annot = out_dir / f"{gpl}.annot.gz"
    soft = out_dir / f"{gpl}_family.soft.gz"
    frame: pd.DataFrame | None = None
    if annot.is_file() or _try_download(platform_annot_url(gpl), annot):
        try:
            frame = _read_platform_table(annot)
        except Exception:
            frame = None
    if frame is None:
        download(platform_soft_url(gpl), soft)
        frame = _read_platform_table(soft)
    id_col = next((c for c in frame.columns if c.strip().upper() in {"ID", "ID_REF"}), None)
    gene_col = next(
        (c for c in frame.columns
         if c.strip().lower() in {"gene symbol", "gene_symbol", "genesymbol", "symbol", "orf"}),
        None,
    )
    if id_col is None or gene_col is None:
        raise ValueError(f"{gpl}: no ID / gene-symbol column in {list(frame.columns)}")
    out = frame[[id_col, gene_col]].rename(columns={id_col: "probe_id", gene_col: "gene"})
    out = out.dropna()
    out["probe_id"] = out["probe_id"].astype(str).str.strip().str.strip('"')
    out["gene"] = (
        out["gene"].astype(str).str.split("///", regex=False).str[0].str.strip().str.strip('"')
    )
    out = out[~out["gene"].str.upper().isin({"", "NA", "---", "N/A", "NULL", "NODATA"})]
    out = out[~out["gene"].str.upper().str.startswith(("NODATA", "UNQ"))]
    return out.drop_duplicates("probe_id")


def _try_download(url: str, dest: Path) -> bool:
    try:
        download(url, dest)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError):
        return False


def collapse_probes(expression: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    annotation = annotation.set_index("probe_id")
    common = expression.index.intersection(annotation.index)
    if len(common) == 0:
        raise ValueError("no probes shared by expression matrix and annotation")
    expr = expression.loc[common]
    ann = annotation.loc[common].copy()
    ann["_mean"] = expr.mean(axis=1)
    chosen = ann.sort_values("_mean", ascending=False).drop_duplicates("gene")
    expr = expr.loc[chosen.index]
    expr.index = chosen["gene"].to_numpy()
    return expr.T


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------
def _lookup(rec: dict[str, str], key: str) -> str:
    """Case-insensitive characteristic lookup (GSE26939 mixes ``Sex``/``sex``)."""
    if key in rec:
        return rec[key]
    low = key.lower()
    for name, value in rec.items():
        if name.lower() == low:
            return value
    return ""


def _clean(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _is_na(value: str) -> bool:
    return value.strip().lower() in NA_TOKENS


def parse_time_days(value: str, factor: float) -> float:
    text = _clean(value).replace(",", "")
    if _is_na(text):
        return np.nan
    try:
        return float(text) * factor
    except ValueError:
        return np.nan


def parse_event(value: str, mapping: dict[str, int]) -> int | None:
    key = _clean(value).lower()
    if key in mapping:
        return mapping[key]
    return None


def parse_age(value: str) -> float:
    text = _clean(value)
    if _is_na(text):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_sex(value: str) -> str:
    text = _clean(value).lower()
    if text in {"m", "male", "1"}:
        return "male"
    if text in {"f", "female", "2"}:
        return "female"
    return "NA"


def stage_group(value: str, mode: str = "roman") -> str:
    text = _clean(value).upper().replace('"', "")
    text = re.sub(r"^STAGE\s*", "", text)
    if _is_na(text):
        return "NA"
    if mode == "tnm":
        match = re.search(r"N(\d)\s*.*?T(\d)", text)
        if match:
            n, t = int(match.group(1)), int(match.group(2))
            if n >= 2:
                return "III"
            if n == 1 or t >= 3:
                return "II"
            return "I"
    match = re.match(r"^(IV|III|II|I)([ABC])?\b", text)
    if match:
        return match.group(1)
    match = re.match(r"^([1-4])", text)
    if match:
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}[match.group(1)]
    return "NA"


# ---------------------------------------------------------------------------
# Build one cohort
# ---------------------------------------------------------------------------
def build_cohort(gse: str, out_dir: Path, *, force: bool = False) -> dict:
    cfg = COHORTS[gse]
    gse = gse.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    series = download(series_matrix_url(gse), out_dir / f"{gse}_series_matrix.txt.gz", force=force)
    sample_ids, samples, series_platform, title = parse_series_header(series)
    platform = cfg.get("platform") or series_platform

    annotation = load_annotation(platform, out_dir)

    time_key, factor = cfg["time"]
    event_key, event_map = cfg["event"]
    include = cfg.get("include", {})
    exclude = cfg.get("exclude", {})

    rows: list[dict] = []
    dropped = {"filtered": 0, "no_survival": 0}
    for sample_id, rec in zip(sample_ids, samples):
        if any(_clean(_lookup(rec, key)).lower() not in {v.lower() for v in vals}
               for key, vals in include.items()):
            dropped["filtered"] += 1
            continue
        if any(_clean(_lookup(rec, key)).lower() in {v.lower() for v in vals}
               for key, vals in exclude.items()):
            dropped["filtered"] += 1
            continue
        event = parse_event(_lookup(rec, event_key), event_map)
        time_days = parse_time_days(_lookup(rec, time_key), factor)
        if event is None or not np.isfinite(time_days) or time_days <= 0:
            dropped["no_survival"] += 1
            continue
        rows.append({
            "sample_id": sample_id,
            "patient_id": sample_id,
            "os_time_days": float(time_days),
            "os_event": int(event),
            "stage_ajcc": stage_group(_lookup(rec, cfg["stage"]), cfg.get("stage_mode", "roman")),
            "age_years": parse_age(_lookup(rec, cfg["age"])),
            "sex": parse_sex(_lookup(rec, cfg["sex"])),
            "neoadjuvant": False,
            "sample_type": "primary",
            "platform": platform,
            "expr_unit": cfg["expr_unit"],
            "cohort_id": gse,
        })
    clinical = pd.DataFrame(rows)
    if clinical.empty:
        raise ValueError(f"{gse}: no samples with valid OS time/event (dropped={dropped})")
    age_na = int(clinical["age_years"].isna().sum())
    if age_na:
        clinical["age_years"] = clinical["age_years"].fillna(clinical["age_years"].median())

    table = pd.read_csv(series, sep="\t", comment="!", index_col=0)
    table.index = table.index.astype(str).str.strip().str.strip('"')
    table.columns = [str(c).strip().strip('"') for c in table.columns]
    table = table.apply(pd.to_numeric, errors="coerce")
    table = table.dropna(axis=0, how="all")
    if cfg["log_transform"]:
        table = np.log2(table.clip(lower=0) + 1.0)
    gene_expr = collapse_probes(table, annotation)
    gene_expr.index = gene_expr.index.astype(str)
    keep = clinical["sample_id"].tolist()
    gene_expr = gene_expr.reindex(keep)
    gene_expr.columns = [str(c) for c in gene_expr.columns]
    gene_expr = gene_expr.loc[:, ~gene_expr.columns.duplicated()]

    clinical.to_parquet(out_dir / "clinical.parquet", index=False)
    gene_expr.to_parquet(out_dir / "expression.parquet")

    notes = cfg.get("notes", "")
    if age_na:
        notes += f" age_years imputed with cohort median for {age_na} samples."
    notes += f" Series title: {title.rstrip(chr(46))}."
    info = {
        "cohort_id": gse,
        "cancer_type": cfg["cancer_type"],
        "accession": gse,
        "platform": platform,
        "expr_unit": cfg["expr_unit"],
        "source": f"GEO:{gse}",
        "license": "open-access GEO record; see series citation and GEO terms",
        "n": int(len(clinical)),
        "n_genes": int(gene_expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "n_stage_known": int((clinical["stage_ajcc"] != "NA").sum()),
        "notes": notes.strip(),
    }
    write_json(out_dir / "meta.json", info)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-luad-geo")
    parser.add_argument("--only", default=None,
                        help="comma-separated GSE ids (default: every configured cohort)")
    parser.add_argument("--out-root", default=str(ROOT / "data" / "raw" / "geo"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    targets = ([g.strip().upper() for g in args.only.split(",")]
               if args.only else list(COHORTS))
    out_root = Path(args.out_root)
    failures = []
    for gse in targets:
        try:
            info = build_cohort(gse, out_root / gse, force=args.force)
            print(f"{gse}: n={info['n']} events={info['events_os']} "
                  f"genes={info['n_genes']} platform={info['platform']}")
        except Exception as exc:  # keep going, report at the end
            failures.append((gse, repr(exc)))
            print(f"{gse}: FAILED {exc!r}", file=sys.stderr)
    if failures:
        print("failures:", failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
