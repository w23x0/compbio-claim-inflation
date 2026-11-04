"""Cohort cleaning: raw -> versioned processed/eval-restricted parquet."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbi.config import Config
from cbi.util import read_json, write_json

REQUIRED_CLINICAL = [
    "sample_id", "patient_id", "os_time_days", "os_event", "stage_ajcc",
    "age_years", "sex", "neoadjuvant", "sample_type", "platform", "expr_unit",
]
OPTIONAL_CLINICAL = [
    "dss_time_days", "dss_event", "pfi_time_days", "pfi_event",
    "dfs_time_days", "dfs_event", "grade", "cohort_id", "cancer_type",
]


def _coerce_clinical(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLINICAL if c not in df.columns]
    if missing:
        raise ValueError(f"clinical table missing required columns: {missing}")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    # Survival analyses assume one observation per patient.  Collapse replicate
    # samples to the first primary specimen; keep NA-only patient IDs as-is.
    if df["patient_id"].nunique() > 1:
        df = df.drop_duplicates(subset=["patient_id"], keep="first")
    for prefix in ["os", "dss", "pfi", "dfs"]:
        tcol, ecol = f"{prefix}_time_days", f"{prefix}_event"
        if tcol in df.columns:
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
            df.loc[df[tcol] <= 0, tcol] = np.nan
        if ecol in df.columns:
            df[ecol] = pd.to_numeric(df[ecol], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["stage_ajcc"] = df["stage_ajcc"].astype(str).str.replace("Stage ", "", regex=False).str.strip()
    df["stage_ajcc"] = df["stage_ajcc"].where(df["stage_ajcc"].isin(["I", "II", "III", "IV"]), "NA")
    df["neoadjuvant"] = df["neoadjuvant"].astype(bool)
    df["sex"] = df["sex"].astype(str).str.lower()
    df["sample_type"] = df["sample_type"].astype(str).str.lower()
    return df


def _coerce_expression(expr: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    expr = expr.copy()
    expr.index = expr.index.astype(str)
    expr.columns = [str(c) for c in expr.columns]
    for c in expr.columns:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.reindex(sample_ids)
    if expr.isna().any().any():
        # Drop genes with any missing values rather than imputing a survival
        # predictor silently; record is handled by sanity checks.
        keep = [c for c in expr.columns if expr[c].notna().all() and expr[c].std(ddof=0) > 1e-9]
        expr = expr[keep]
    return expr


def clean_cohort(cfg: Config, cohort_id: str, role: str) -> Path:
    raw_dir = cfg.paths.raw / "geo" / cohort_id
    if not raw_dir.is_dir():
        raw_dir = cfg.paths.restriction / "raw" / cohort_id
    if not raw_dir.is_dir():
        raw_dir = cfg.paths.raw / cohort_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw cohort not found under {cfg.paths.raw}")
    clinical = pd.read_parquet(raw_dir / "clinical.parquet")
    expr = pd.read_parquet(raw_dir / "expression.parquet")
    meta = read_json(raw_dir / "meta.json")
    clinical = _coerce_clinical(clinical)
    expr = _coerce_expression(expr, clinical["sample_id"].tolist())
    out_dir = cfg.paths.cohort_data_dir(cohort_id, role)
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical.to_parquet(out_dir / "clinical.parquet", index=False)
    expr.to_parquet(out_dir / "expression.parquet")
    meta = dict(meta)
    meta["role"] = role
    meta.setdefault("n_deposited", int(meta.get("n") or len(clinical)))
    meta["n"] = int(len(clinical))
    meta["n_clean"] = int(len(clinical))
    meta["events_os"] = int(pd.to_numeric(clinical["os_event"], errors="coerce").fillna(0).sum())
    meta["n_genes"] = int(expr.shape[1])
    write_json(out_dir / "meta.json", meta)
    if role != "dev":
        try:
            for path in [out_dir, out_dir.parent, out_dir.parent.parent]:
                path.chmod(0o700)
        except OSError:
            pass
    return out_dir


def clean_all(cfg: Config, *, force: bool = False) -> list[Path]:
    if not cfg.cohort_split.get("roles"):
        raise RuntimeError("cohort_split.yaml is missing; run P0 first")
    out: list[Path] = []
    for entry in cfg.cohort_pool.get("cohorts", []):
        cid = entry["cohort_id"]
        role = cfg.cohort_split["roles"][cid]
        out.append(clean_cohort(cfg, cid, role))
    return out
