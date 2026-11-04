"""Analysis specifications and per-cohort effect estimation (spec M2)."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from cbi.m2.survival import cox_ph, maxstat_cutpoint
from cbi.schemas import Claim


@dataclass(frozen=True)
class AnalysisSpec:
    spec_id: str
    grouping: str
    covariates: str = "none"
    sample_scope: str = "all_primary"
    followup_cap_years: float | None = None
    normalization: str = "platform_native"
    probe_mapping: str = "max_mean"
    in_reference_set: bool = True
    percentile: float | None = None
    subgroup: str | None = None

    @property
    def contrast_type(self) -> str:
        return "continuous" if self.grouping == "per_sd" else "dichotomous"

    def signature(self) -> str:
        return (
            f"g={self.grouping};p={self.percentile};c={self.covariates};"
            f"s={self.sample_scope};t={self.followup_cap_years};"
            f"n={self.normalization};pr={self.probe_mapping};sub={self.subgroup}"
        )


@dataclass(frozen=True)
class CohortEffect:
    cohort_id: str
    gene: str
    endpoint: str
    log_effect: float
    se: float
    p: float
    n: int
    events: int
    spec_id: str
    role: str = "dev"


def reference_specs() -> list[AnalysisSpec]:
    specs: list[AnalysisSpec] = []
    for grouping in ["median", "tertile", "per_sd"]:
        for cov in ["none", "age_stage"]:
            for scope in ["all_primary", "exclude_neoadjuvant"]:
                for cap in [None, 5.0]:
                    sid = f"{grouping}|{cov}|{scope}|{'5y' if cap else 'full'}"
                    specs.append(AnalysisSpec(
                        spec_id=sid,
                        grouping=grouping,
                        covariates=cov,
                        sample_scope=scope,
                        followup_cap_years=cap,
                        in_reference_set=True,
                    ))
    return specs


def default_spec(contrast_type: str) -> AnalysisSpec:
    grouping = "median" if contrast_type == "dichotomous" else "per_sd"
    return AnalysisSpec(
        spec_id=f"{grouping}|none|all_primary|full",
        grouping=grouping,
        covariates="none",
        sample_scope="all_primary",
        followup_cap_years=None,
        in_reference_set=True,
    )


def aligned_reference_specs(contrast_type: str) -> list[AnalysisSpec]:
    """Specs whose effect is on the same scale as the claim (R1)."""
    grouping = "median" if contrast_type == "dichotomous" else "per_sd"
    return [s for s in reference_specs() if s.grouping == grouping]


def _stage_ordinal(stage: pd.Series) -> pd.Series:
    return stage.map({"I": 1, "II": 2, "III": 3, "IV": 4}).astype(float)


def _subset_mask(clinical: pd.DataFrame, spec: AnalysisSpec) -> np.ndarray:
    mask = np.ones(len(clinical), dtype=bool)
    st = clinical["sample_type"].astype(str).str.lower()
    mask &= st.isin(["primary", "tumor", "tumour"]) | (st == "na")
    if spec.sample_scope == "exclude_neoadjuvant":
        mask &= ~clinical["neoadjuvant"].astype(bool).to_numpy()
    if spec.subgroup == "stage_I_II":
        mask &= clinical["stage_ajcc"].isin(["I", "II"]).to_numpy()
    elif spec.subgroup == "stage_III_IV":
        mask &= clinical["stage_ajcc"].isin(["III", "IV"]).to_numpy()
    return mask


def _design(clinical: pd.DataFrame, grouping: str, spec: AnalysisSpec,
            percentile: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (design matrix, row mask).  Row alignment is explicit because
    tertile contrasts drop the middle third and covariates may be missing."""
    expr = np.asarray(clinical["_x"], dtype=float)
    n = len(expr)
    finite = np.isfinite(expr)
    row_mask = finite.copy()
    if grouping in {"median", "tertile", "optimal", "percentile_custom"}:
        if grouping == "median":
            cut = float(np.median(expr[finite]))
            high = expr > cut
            low = ~high
        elif grouping == "tertile":
            low_cut = float(np.quantile(expr[finite], 1 / 3))
            high_cut = float(np.quantile(expr[finite], 2 / 3))
            high = expr > high_cut
            low = expr <= low_cut
            row_mask &= high | low
        elif grouping == "percentile_custom":
            cut = float(np.quantile(expr[finite], percentile if percentile is not None else 0.5))
            high = expr > cut
            low = ~high
        else:
            cut, _, _ = maxstat_cutpoint(expr, clinical["_time"].to_numpy(), clinical["_event"].to_numpy())
            high = expr > cut
            low = ~high
        x = high.astype(float)
    else:
        mu = float(np.nanmean(expr))
        sd = float(np.nanstd(expr, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return None
        x = (expr - mu) / sd
    cols = [x]
    names = ["expr"]
    if spec.covariates in {"age_stage", "age_stage_grade"}:
        age = pd.to_numeric(clinical["age_years"], errors="coerce").to_numpy(dtype=float)
        age_mu, age_sd = np.nanmean(age), np.nanstd(age, ddof=1)
        if np.isfinite(age_sd) and age_sd > 0:
            cols.append((age - age_mu) / age_sd)
            names.append("age_z")
            row_mask &= np.isfinite(age)
        else:
            row_mask &= False
        stage = _stage_ordinal(clinical["stage_ajcc"]).to_numpy(dtype=float)
        if np.isfinite(stage).sum() > 0:
            cols.append(stage - np.nanmean(stage))
            names.append("stage_ord")
            row_mask &= np.isfinite(stage)
        else:
            row_mask &= False
        if spec.covariates == "age_stage_grade" and "grade" in clinical.columns:
            grade = pd.to_numeric(clinical["grade"], errors="coerce").to_numpy(dtype=float)
            cols.append(grade - np.nanmean(grade))
            names.append("grade_ord")
            row_mask &= np.isfinite(grade)
    X = np.column_stack([c[row_mask] for c in cols])
    if X.shape[1] > 1:
        keep_cols = [0] + [j for j in range(1, X.shape[1]) if np.std(X[:, j]) > 1e-10]
        X = X[:, keep_cols]
    return X, row_mask


def estimate_effect(clinical: pd.DataFrame, expr: pd.Series, gene: str,
                    spec: AnalysisSpec, endpoint: str, *, min_samples: int = 80,
                    min_events: int = 15) -> CohortEffect | None:
    if endpoint not in {"OS", "DSS", "PFI", "DFS"}:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    tcol, ecol = f"{endpoint.lower()}_time_days", f"{endpoint.lower()}_event"
    if tcol not in clinical.columns or ecol not in clinical.columns:
        return None
    mask = _subset_mask(clinical, spec)
    df = clinical.loc[mask].copy()
    x = pd.to_numeric(expr.reindex(df["sample_id"]).to_numpy(), errors="coerce")
    df["_x"] = x
    df["_time"] = pd.to_numeric(df[tcol], errors="coerce")
    df["_event"] = pd.to_numeric(df[ecol], errors="coerce").fillna(0).astype(int)
    valid = np.isfinite(df["_x"].to_numpy()) & np.isfinite(df["_time"].to_numpy()) & (df["_time"] > 0).to_numpy()
    df = df.loc[valid].copy()
    if spec.followup_cap_years is not None:
        cap = float(spec.followup_cap_years) * 365.25
        beyond = df["_time"].to_numpy() > cap
        df.loc[beyond, "_event"] = 0
        df.loc[beyond, "_time"] = cap
    try:
        design = _design(df, spec.grouping, spec, spec.percentile)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if design is None:
        return None
    X, row_mask = design
    df = df.loc[row_mask].copy()
    if len(df) < min_samples or int(df["_event"].sum()) < min_events:
        return None
    try:
        fit = cox_ph(df["_time"].to_numpy(), df["_event"].to_numpy(), X)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(fit.beta[0]) or not np.isfinite(fit.se[0]) or fit.se[0] <= 0:
        return None
    cid = str(df["cohort_id"].iloc[0]) if "cohort_id" in df.columns else "unknown"
    return CohortEffect(
        cohort_id=cid,
        gene=gene,
        endpoint=endpoint,
        log_effect=float(fit.beta[0]),
        se=float(fit.se[0]),
        p=float(fit.p[0]),
        n=int(len(df)),
        events=int(df["_event"].sum()),
        spec_id=spec.spec_id,
    )


def claim_spec(claim: Claim) -> AnalysisSpec:
    ch = claim.reported_choices
    grouping = {
        "median": "median",
        "tertile": "tertile",
        "optimal": "optimal",
        "percentile_custom": "percentile_custom",
        "per_sd": "per_sd",
        "per_unit": "per_sd",
        "unspecified": "median" if ch.contrast_type == "dichotomous" else "per_sd",
        "other": "median" if ch.contrast_type == "dichotomous" else "per_sd",
    }[ch.cutpoint]
    percentile = None
    if grouping == "percentile_custom":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", (ch.selection_note or ""))
        percentile = float(m.group(1)) / 100.0 if m else 0.5
    subgroup = None
    if ch.subset:
        low = ch.subset.lower().replace(" ", "")
        if "stagei-ii" in low or "stagei–ii" in low or "i-ii" in low:
            subgroup = "stage_I_II"
        elif "stageiii-iv" in low or "stageiii–iv" in low or "iii-iv" in low:
            subgroup = "stage_III_IV"
    covariates = "none"
    if ch.covariates:
        joined = ",".join(ch.covariates).lower()
        if "grade" in joined:
            covariates = "age_stage_grade"
        elif "stage" in joined or "age" in joined:
            covariates = "age_stage"
    cap = ch.time_cap_years
    if cap is not None and cap >= 50:
        cap = None
    return AnalysisSpec(
        spec_id=f"claim|{ch.cutpoint}|{ch.covariates}|{ch.time_cap_years}|{ch.subset}",
        grouping=grouping,
        covariates=covariates,
        sample_scope="subgroup_stage" if subgroup else "all_primary",
        followup_cap_years=cap,
        normalization=ch.normalization,
        probe_mapping=ch.probe_selection or "max_mean",
        in_reference_set=(ch.cutpoint in {"median", "tertile", "per_sd", "unspecified"}),
        percentile=percentile,
        subgroup=subgroup,
    )


def registry_specs(specs: Iterable[AnalysisSpec]) -> list[AnalysisSpec]:
    return [replace(s, in_reference_set=False) for s in specs]
