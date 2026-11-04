"""Ingest breast-cancer GEO series into the P0 standard cohort contract.

The generic `cbi.cohort_pool.geo` parser only understands a few fixed
`!Sample_characteristics_ch1` names (event_os / duryears_os / ...).  The BRCA
series below encode the same information under different keys ("e.os", "t.os",
"death", "os (days)", "event_death", ...), and several of them mix different
keys inside one characteristics row (empty cells for samples that do not carry
that field).  This adapter therefore parses each characteristics *cell* as its
own ``key: value`` pair, then maps the per-series keys onto the standard
columns.

Probe -> gene collapsing, platform annotation download and the output schema
are reused from `cbi.cohort_pool.geo`, so the resulting files are identical in
shape to cohorts produced by ``python -m cbi.cohort_pool.geo``.

Usage:
    python scripts/ingest_brca_geo.py                 # all configured series
    python scripts/ingest_brca_geo.py GSE20685 GSE7390
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbi.cohort_pool.geo import (  # noqa: E402
    SERIES_BASE,
    _geo_dir,
    _split_stage,
    collapse_probes,
    download,
    parse_platform_annotation,
    platform_annotation_url,
)
from cbi.util import write_json  # noqa: E402

# Per-series mapping.  Values are lists of candidate characteristic keys in
# priority order; ``unit`` is the unit of the time field ("days"/"years").
SPECS: dict[str, dict[str, Any]] = {
    "GSE20685": {
        "event": ["event_death"],
        "time": ["follow_up_duration (years)"],
        "time_unit": "years",
        "age": ["age at diagnosis"],
        "sex": ["gender"],
        "stage": [],
        "tissue": ["tissue"],
        "neoadjuvant": ["neoadjuvant chemotherapy"],
        "neoadjuvant_default": False,
        "notes": "Asian (Taiwan) primary breast cancer cohort; event_death is "
                 "overall-survival death, follow_up_duration is years of follow-up.",
    },
    "GSE42568": {
        "event": ["overall survival event"],
        "time": ["overall survival time_days"],
        "time_unit": "days",
        "age": ["age"],
        "sex": [],
        "stage": [],
        "tissue": ["tissue"],
        "neoadjuvant": [],
        "neoadjuvant_default": False,
        "notes": "Irish primary breast cancer series; 17 normal breast samples in "
                 "the matrix are excluded, only tumour samples enter the cohort.",
    },
    "GSE7390": {
        "event": ["e.os"],
        "time": ["t.os"],
        "time_unit": "days",
        "age": ["age"],
        "sex": [],
        "stage": [],
        "tissue": [],
        "neoadjuvant": [],
        "neoadjuvant_default": False,
        "notes": "TRANSBIG node-negative untreated cohort; t.os/e.os are days and "
                 "death indicator.",
    },
    "GSE58812": {
        "event": ["death"],
        "time": ["os (days)"],
        "time_unit": "days",
        "age": ["age at diag"],
        "sex": [],
        "stage": [],
        "tissue": ["diagnosis"],
        "neoadjuvant": [],
        "neoadjuvant_default": False,
        "notes": "Triple-negative breast cancer cohort (all patients ER/PR/HER2 "
                 "negative); os (days) with death indicator.",
    },
}

PLATFORM_UNIT = "log2_intensity"


def series_matrix_url(gse: str) -> str:
    return f"{SERIES_BASE}/{_geo_dir(gse, 'GSE')}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def read_header(path: Path) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    """Return (sample_ids, per-sample key/value dicts, series info)."""
    sample_ids: list[str] = []
    titles: list[str] = []
    per_sample: list[dict[str, str]] = []
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
            elif key == "!Sample_title":
                titles = values
            elif key == "!Sample_characteristics_ch1":
                if not per_sample:
                    per_sample = [{} for _ in values]
                for i, cell in enumerate(values):
                    if i >= len(per_sample) or ":" not in cell:
                        continue
                    name, value = cell.split(":", 1)
                    name, value = name.strip().lower(), value.strip()
                    if name and value and value.upper() != "NA":
                        per_sample[i].setdefault(name, value)
            elif key == "!Series_platform_id" and platform is None:
                platform = values[0] if values and values[0] else None
            elif key == "!Series_title":
                title = values[0] if values else ""
    if not sample_ids:
        raise ValueError(f"{path}: no !Sample_geo_accession row")
    if platform is None:
        raise ValueError(f"{path}: no !Series_platform_id row")
    for i, title_i in enumerate(titles):
        if i < len(per_sample) and title_i:
            per_sample[i].setdefault("_title", title_i)
    return sample_ids, per_sample, {"platform": platform, "title": title}


def _pick(record: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        if key in record and record[key] not in {"", "NA", "na"}:
            return record[key]
    return None


def _num(text: str | None) -> float:
    if text is None:
        return np.nan
    try:
        return float(str(text).replace(",", "."))
    except ValueError:
        return np.nan


def _event(text: str | None) -> int | None:
    if text is None:
        return None
    low = str(text).strip().lower()
    if low in {"1", "1.0", "yes", "dead", "deceased", "event", "true", "y"}:
        return 1
    if low in {"0", "0.0", "no", "alive", "censored", "false", "n"}:
        return 0
    return None


def _sex(text: str | None) -> str:
    if text is None:
        return "female"  # breast-cancer series below are all-female cohorts
    low = str(text).strip().lower()
    if low.startswith("f") or low in {"2", "2.0"}:
        return "female"
    if low.startswith("m") or low in {"1", "1.0"}:
        return "male"
    return "female"


def _neoadjuvant(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return str(text).strip().lower() in {"yes", "y", "1", "true"}


def ingest(gse: str, out_root: str | Path, *, force: bool = False) -> dict[str, Any]:
    gse = gse.upper()
    spec = SPECS[gse]
    out = Path(out_root) / gse
    out.mkdir(parents=True, exist_ok=True)
    series = download(series_matrix_url(gse), out / f"{gse}_series_matrix.txt.gz", force=force)
    sample_ids, records, info = read_header(series)
    platform = str(info["platform"]).upper()

    rows = []
    dropped = {"no_time": 0, "no_event": 0, "normal_tissue": 0, "no_age": 0}
    for sample_id, rec in zip(sample_ids, records):
        tissue = (_pick(rec, spec["tissue"]) or "").lower()
        if tissue and any(word in tissue for word in ("normal", "adjacent")):
            dropped["normal_tissue"] += 1
            continue
        event = _event(_pick(rec, spec["event"]))
        time_raw = _num(_pick(rec, spec["time"]))
        days = time_raw * (365.25 if spec["time_unit"] == "years" else 1.0)
        if not np.isfinite(days) or days <= 0:
            dropped["no_time"] += 1
            continue
        if event is None:
            dropped["no_event"] += 1
            continue
        age = _num(_pick(rec, spec["age"]))
        if not np.isfinite(age):
            dropped["no_age"] += 1
            continue
        stage = _pick(rec, spec["stage"])
        rows.append({
            "sample_id": sample_id,
            "patient_id": sample_id,
            "os_time_days": float(days),
            "os_event": int(event),
            "stage_ajcc": _split_stage(stage) if stage else "NA",
            "age_years": float(age),
            "sex": _sex(_pick(rec, spec["sex"])),
            "neoadjuvant": _neoadjuvant(_pick(rec, spec["neoadjuvant"]), spec["neoadjuvant_default"]),
            "sample_type": "primary",
            "platform": platform,
            "expr_unit": PLATFORM_UNIT,
            "cohort_id": gse,
        })
    clinical = pd.DataFrame(rows)
    if len(clinical) < 80:
        raise ValueError(f"{gse}: only {len(clinical)} usable samples (dropped {dropped})")
    if clinical["os_event"].sum() < 15:
        raise ValueError(f"{gse}: only {int(clinical['os_event'].sum())} OS events")

    annot = parse_platform_annotation(
        download(platform_annotation_url(platform), out / f"{platform}.annot.gz", force=force)
    )
    table = pd.read_csv(series, sep="\t", comment="!", index_col=0)
    table.index = table.index.astype(str)
    table.columns = [str(c).strip().strip('"') for c in table.columns]
    table = table.apply(pd.to_numeric, errors="coerce")
    gene_expr = collapse_probes(table, annot)
    # Several series are already deposited on a log2 scale (MAS5/RMA); only
    # linear intensity matrices are log2-transformed, so the value scale is
    # log2 for every cohort and nothing is logged twice.
    peak = float(np.nanmax(gene_expr.to_numpy(dtype=float)))
    if peak > 50.0:
        gene_expr = np.log2(gene_expr.clip(lower=0) + 1.0)
        transform = "log2(x+1) from linear intensities"
    else:
        transform = "series matrix already log2-scaled; no further transform"
    gene_expr = gene_expr.reindex(clinical["sample_id"].tolist())

    clinical.to_parquet(out / "clinical.parquet", index=False)
    gene_expr.to_parquet(out / "expression.parquet")
    meta = {
        "cohort_id": gse,
        "cancer_type": "BRCA",
        "accession": gse,
        "platform": platform,
        "expr_unit": PLATFORM_UNIT,
        "source": f"GEO:{gse}",
        "license": "public GEO record; see series citation and GEO terms of use",
        "n": int(len(clinical)),
        "n_genes": int(gene_expr.shape[1]),
        "events_os": int(clinical["os_event"].sum()),
        "median_os_days": float(clinical["os_time_days"].median()),
        "n_deposited": int(len(sample_ids)),
        "sample_type": "primary tumour only",
        "sex_source": "characteristics field" if spec["sex"] else "cohort is an all-female breast-cancer series",
        "dropped": dropped,
        "notes": (
            f"{spec['notes']} Parsed by scripts/ingest_brca_geo.py: per-cell "
            "characteristics key/value mapping, probes collapsed to gene symbols "
            f"(max-mean probe), {transform}; no sample-level values were imputed."
        ),
    }
    write_json(out / "meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest-brca-geo")
    parser.add_argument("series", nargs="*", default=None)
    parser.add_argument("--out-root", default=str(ROOT / "data" / "raw" / "geo"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for gse in (args.series or sorted(SPECS)):
        meta = ingest(gse, args.out_root, force=args.force)
        print(f"{meta['cohort_id']}: n={meta['n']} events={meta['events_os']} "
              f"genes={meta['n_genes']} platform={meta['platform']} dropped={meta['dropped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
