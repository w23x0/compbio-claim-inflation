"""Survival analysis primitives.

Cox proportional hazards is implemented directly (Breslow ties, Newton-Raphson)
so the ground-truth engine has no heavy runtime dependency and is deterministic
across snapshots.  The implementation is checked against known closed-form and
simulation cases in tests/test_survival.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CoxFit:
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    loglik: float
    converged: bool
    n: int
    events: int
    iterations: int

    def log_hr(self, index: int = -1) -> float:
        return float(self.beta[index])

    def hr(self, index: int = -1) -> float:
        return float(np.exp(self.beta[index]))

    def se_at(self, index: int = -1) -> float:
        return float(self.se[index])


def _prepare(time: np.ndarray, event: np.ndarray, X: np.ndarray):
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=int).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(time) & np.isfinite(event) & np.isfinite(X).all(axis=1) & (time > 0)
    time, event, X = time[mask], event[mask], X[mask]
    if len(time) == 0 or event.sum() < 1:
        raise ValueError("no analyzable observations/events")
    # Center columns for numerical stability; Cox partial likelihood is
    # invariant to column shifts because each risk set contains every subject.
    X = X - X.mean(axis=0, keepdims=True)
    return time, event, X


def _risk_groups(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (group_end_indices, events_per_group) for time sorted descending."""
    n = len(time)
    starts = np.flatnonzero(np.r_[True, time[1:] != time[:-1]])
    ends = np.r_[starts[1:] - 1, n - 1]
    return ends, starts


def cox_ph(time: np.ndarray, event: np.ndarray, X: np.ndarray, *, max_iter: int = 50,
           tol: float = 1e-9) -> CoxFit:
    time, event, X = _prepare(time, event, X)
    n, p = X.shape
    order = np.argsort(-time, kind="mergesort")
    time_s, event_s, X_s = time[order], event[order], X[order]
    ends, starts = _risk_groups(time_s)
    event_counts = np.array([event_s[s:e + 1].sum() for s, e in zip(starts, ends)], dtype=float)
    if event_counts.sum() < 1:
        raise ValueError("no events after grouping")
    # only compute for groups that contain at least one event
    active = event_counts > 0
    ends_a = ends[active]
    counts_a = event_counts[active]

    beta = np.zeros(p, dtype=float)
    loglik = -np.inf
    converged = False
    hess = np.eye(p) * 1e-8
    iterations = 0
    for iterations in range(1, max_iter + 1):
        eta = X_s @ beta
        eta = eta - eta.max()
        w = np.exp(eta)
        s0 = np.cumsum(w)
        s1 = np.cumsum(X_s * w[:, None], axis=0)
        s2 = np.cumsum(X_s[:, :, None] * X_s[:, None, :] * w[:, None, None], axis=0)
        s0_a = np.maximum(s0[ends_a], 1e-12)
        s1_a = s1[ends_a]
        s2_a = s2[ends_a]
        mean = s1_a / s0_a[:, None]
        var = s2_a / s0_a[:, None, None] - mean[:, :, None] * mean[:, None, :]
        # Aggregate by event-time group (Breslow ties).
        score = np.zeros(p)
        info = np.zeros((p, p))
        for gi, (end, cnt) in enumerate(zip(ends_a, counts_a)):
            ev_idx = np.flatnonzero(event_s[starts[active][gi]:end + 1]) + starts[active][gi]
            score += X_s[ev_idx].sum(axis=0) - cnt * mean[gi]
            info += cnt * var[gi]
        new_loglik = float(np.sum(eta[event_s.astype(bool)]) - np.sum(counts_a * np.log(s0_a)))
        try:
            step = np.linalg.solve(info + np.eye(p) * 1e-10, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ score
        # Step-halving keeps the partial likelihood monotone.
        improved = False
        alpha = 1.0
        for _ in range(20):
            cand = beta + alpha * step
            eta_c = X_s @ cand
            eta_c = eta_c - eta_c.max()
            wc = np.exp(eta_c)
            s0c = np.cumsum(wc)
            ll = float(np.sum(eta_c[event_s.astype(bool)]) - np.sum(counts_a * np.log(np.maximum(s0c[ends_a], 1e-12))))
            if np.isfinite(ll) and ll >= new_loglik - 1e-12:
                beta = cand
                loglik = ll
                improved = True
                break
            alpha *= 0.5
        if not improved:
            loglik = new_loglik
            break
        if np.max(np.abs(alpha * step)) < tol:
            converged = True
            break
    # Final information matrix at the estimate.
    eta = X_s @ beta
    eta = eta - eta.max()
    w = np.exp(eta)
    s0 = np.cumsum(w)
    s1 = np.cumsum(X_s * w[:, None], axis=0)
    s2 = np.cumsum(X_s[:, :, None] * X_s[:, None, :] * w[:, None, None], axis=0)
    s0_a = np.maximum(s0[ends_a], 1e-12)
    s1_a, s2_a = s1[ends_a], s2[ends_a]
    mean = s1_a / s0_a[:, None]
    var = s2_a / s0_a[:, None, None] - mean[:, :, None] * mean[:, None, :]
    info = np.zeros((p, p))
    for gi, cnt in enumerate(counts_a):
        info += cnt * var[gi]
    try:
        cov = np.linalg.pinv(info)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        pvals = 2 * stats.norm.sf(np.abs(beta / se))
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    return CoxFit(
        beta=beta,
        se=se,
        p=pvals,
        loglik=float(loglik),
        converged=bool(converged),
        n=int(n),
        events=int(event.sum()),
        iterations=iterations,
    )


def logrank_chi2(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    """Two-group log-rank statistic and two-sided p value."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=float)
    mask = np.isfinite(time) & np.isfinite(event) & np.isfinite(group) & (time > 0)
    time, event, group = time[mask], event[mask], group[mask]
    if event.sum() < 1 or len(np.unique(group)) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-time, kind="mergesort")
    t, e, g = time[order], event[order], group[order]
    ends, starts = _risk_groups(t)
    s0 = np.cumsum(np.ones(len(t)))
    s1 = np.cumsum(g)
    s2 = np.cumsum(g * g)
    score = 0.0
    info = 0.0
    for s, end in zip(starts, ends):
        cnt = e[s:end + 1].sum()
        if cnt == 0:
            continue
        n_risk = s0[end]
        sum_g = s1[end]
        sum_g2 = s2[end]
        mean = sum_g / n_risk
        var = max(sum_g2 / n_risk - mean * mean, 0.0)
        score += g[s:end + 1].sum() - cnt * mean
        if n_risk > 1:
            info += cnt * var * (n_risk - cnt) / (n_risk - 1)
    # score and info above are the standard log-rank numerator/variance
    if info <= 0:
        return float("nan"), float("nan")
    chi2 = score * score / info
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def maxstat_cutpoint(expr: np.ndarray, time: np.ndarray, event: np.ndarray, *,
                     low_q: float = 0.15, high_q: float = 0.85) -> tuple[float, float, float]:
    """Outcome-dependent cutpoint scan: returns (cutoff, chi2, p)."""
    expr = np.asarray(expr, dtype=float)
    valid = np.isfinite(expr)
    expr, time, event = expr[valid], np.asarray(time)[valid], np.asarray(event)[valid]
    if len(expr) < 20 or event.sum() < 5:
        raise ValueError("too few samples/events for cutpoint scan")
    qs = np.linspace(low_q, high_q, 71)
    candidates = np.unique(np.quantile(expr, qs))
    best = (float("nan"), -np.inf, float("nan"))
    for cut in candidates:
        group = (expr > cut).astype(float)
        if group.sum() < 5 or (1 - group).sum() < 5:
            continue
        chi2, p = logrank_chi2(time, event, group)
        if np.isfinite(chi2) and chi2 > best[1]:
            best = (float(cut), float(chi2), float(p))
    if not np.isfinite(best[1]):
        raise ValueError("cutpoint scan found no valid split")
    return best


def km_median(time: np.ndarray, event: np.ndarray) -> float | None:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    mask = np.isfinite(time) & (time > 0)
    time, event = time[mask], event[mask]
    if len(time) == 0 or event.sum() == 0:
        return None
    order = np.argsort(time)
    time, event = time[order], event[order]
    survival = 1.0
    n = len(time)
    i = 0
    while i < n:
        t = time[i]
        d = int(event[i:][time[i:] == t].sum())
        at_risk = int((time >= t).sum())
        if at_risk > 0 and d > 0:
            survival *= 1.0 - d / at_risk
            if survival <= 0.5:
                return float(t)
        i += int((time[i:] == t).sum())
    return None


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C (higher risk -> shorter survival)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    mask = np.isfinite(time) & np.isfinite(risk) & (time > 0)
    time, event, risk = time[mask], event[mask], risk[mask]
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")
