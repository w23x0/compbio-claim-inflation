"""Random-effects meta-analysis.

Primary estimator: REML for tau^2 with known within-study variances (the usual
meta-analysis model).  k < 3 falls back to a fixed-effect estimate, as required
by spec R2, and is flagged by the caller as lower-confidence.

The implementation follows Viechtbauer (2005) and is cross-checked against
R metafor reference values in tests/test_meta.py and scripts/validate_meta.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class MetaAnalysis:
    method: str
    pooled: float
    se: float
    tau2: float
    pred_low: float
    pred_high: float
    k: int
    q: float
    i2: float
    p: float
    weights: np.ndarray


def _pooled(y: np.ndarray, v: np.ndarray, tau2: float) -> tuple[float, float, np.ndarray]:
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return mu, se, w


def _neg_reml(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    if tau2 < 0:
        return np.inf
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    # Constants independent of tau2 are omitted.
    ll = -0.5 * (np.sum(np.log(v + tau2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))
    return -float(ll)


def meta_random_effects(y: np.ndarray, se: np.ndarray, *, method: str = "REML") -> MetaAnalysis:
    y = np.asarray(y, dtype=float).reshape(-1)
    se = np.asarray(se, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[mask], se[mask]
    k = len(y)
    if k == 0:
        raise ValueError("meta-analysis needs at least one estimate")
    v = se ** 2
    w_fe = 1.0 / v
    mu_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    q = float(np.sum(w_fe * (y - mu_fe) ** 2))
    if k < 3:
        se_fe = float(np.sqrt(1.0 / np.sum(w_fe)))
        p = float(2 * stats.norm.sf(abs(mu_fe / se_fe)))
        # t with df=1 would be absurdly wide; use a normal interval for the
        # fixed-effect fallback and mark the method.
        return MetaAnalysis(
            method="FE (k<3)" if method == "REML" else method,
            pooled=mu_fe,
            se=se_fe,
            tau2=0.0,
            pred_low=float(mu_fe - 1.959963984540054 * se_fe),
            pred_high=float(mu_fe + 1.959963984540054 * se_fe),
            k=k,
            q=q,
            i2=float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
            p=p,
            weights=w_fe,
        )
    # DerSimonian-Laird start; REML refinement.
    c = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2_start = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    upper = max(10.0 * float(np.max(v)), 1.0, 10.0 * tau2_start)
    res = optimize.minimize_scalar(
        lambda t: _neg_reml(float(t), y, v),
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    tau2 = float(res.x) if res.success else float(tau2_start)
    if tau2 < 1e-10:
        tau2 = 0.0
    mu, se_mu, w = _pooled(y, v, tau2)
    p = float(2 * stats.norm.sf(abs(mu / se_mu)))
    # 95% prediction interval, normal quantile (matches metafor's predict()).
    zcrit = 1.959963984540054
    pred = zcrit * float(np.sqrt(tau2 + se_mu ** 2))
    i2 = float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0
    return MetaAnalysis(
        method="REML",
        pooled=mu,
        se=se_mu,
        tau2=tau2,
        pred_low=mu - pred,
        pred_high=mu + pred,
        k=k,
        q=q,
        i2=i2,
        p=p,
        weights=w,
    )


def meta_from_effects(log_effects: list[float], ses: list[float]) -> MetaAnalysis:
    return meta_random_effects(np.asarray(log_effects), np.asarray(ses))
