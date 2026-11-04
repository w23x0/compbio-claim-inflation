"""Cohort sanity checks (spec P0).

The checks deliberately include a known-direction test: later AJCC stage must
carry worse overall survival.  A reversed event code or inconsistent time unit
fails that test.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cbi.m2.survival import cox_ph, km_median


def sanity_check_cohort(clinical: pd.DataFrame, expr: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {"cohort_id": meta.get("cohort_id"), "passed": True, "checks": {}}

    def add(name: str, ok: bool, detail: Any, *, critical: bool = True) -> None:
        checks["checks"][name] = {"ok": bool(ok), "detail": detail, "critical": critical}
        if critical and not ok:
            checks["passed"] = False

    t = pd.to_numeric(clinical["os_time_days"], errors="coerce")
    e = pd.to_numeric(clinical["os_event"], errors="coerce")
    add("time_positive", bool((t > 0).all()), {"min_days": float(t.min()), "max_days": float(t.max())})
    add("event_binary", bool(e.isin([0, 1]).all()), {"unique": sorted(e.dropna().unique().tolist())})
    add("event_rate_reasonable", bool(0.05 <= e.mean() <= 0.95), {"event_rate": float(e.mean())}, critical=False)
    med = km_median(t.to_numpy(), e.to_numpy())
    add("median_os_plausible", med is not None and 30 <= med <= 15 * 365.25,
        {"median_days": med}, critical=False)
    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    stages = clinical["stage_ajcc"].map(stage_map)
    if stages.notna().sum() >= 30 and e.sum() >= 10:
        mask = stages.notna() & t.notna() & e.notna()
        try:
            fit = cox_ph(t[mask].to_numpy(), e[mask].to_numpy(), stages[mask].to_numpy(dtype=float))
            add("stage_direction", bool(fit.beta[0] > 0 and fit.p[0] < 0.10),
                {"log_hr_per_stage": float(fit.beta[0]), "p": float(fit.p[0])})
        except ValueError as exc:
            add("stage_direction", False, f"fit failed: {exc}")
    else:
        add("stage_direction", False, "too few staged events", critical=False)
    add("sample_ids_unique", bool(not clinical["sample_id"].duplicated().any()),
        {"n": int(len(clinical))})
    add("expression_finite", bool(np.isfinite(expr.to_numpy(dtype=float)).all()),
        {"n_genes": int(expr.shape[1]), "n_samples": int(expr.shape[0])})
    add("expression_samples_align", bool(set(expr.index) == set(clinical["sample_id"])),
        {"n_expr": int(expr.shape[0]), "n_clinical": int(len(clinical))})
    add("deposited_ge_used", bool(meta.get("n_deposited", 0) >= len(clinical)),
        {"n_deposited": meta.get("n_deposited"), "n_used": int(len(clinical))}, critical=False)
    return checks
