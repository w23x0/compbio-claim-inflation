from __future__ import annotations

import math

import numpy as np
import pytest

from cbi.m2.survival import (CoxFit, cox_ph, concordance_index, km_median, logrank_chi2,
                             maxstat_cutpoint)


def _simulate(n=2000, beta=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(1.0 / np.exp(beta * x))
    c = rng.exponential(3.0, size=n)
    return np.minimum(t, c), (t <= c).astype(int), x


def test_cox_recovers_known_effect():
    time, event, x = _simulate()
    fit = cox_ph(time, event, x)
    assert fit.converged
    assert abs(fit.beta[0] - 0.7) < 0.12
    assert 0 < fit.p[0] < 1e-6
    assert fit.events == int(event.sum())


def test_cox_matches_lifelines_if_present():
    pytest.importorskip("lifelines")
    from lifelines import CoxPHFitter
    import pandas as pd

    time, event, x = _simulate(n=600, beta=0.5, seed=4)
    ours = cox_ph(time, event, x)
    df = pd.DataFrame({"T": time, "E": event, "x": x})
    ll = CoxPHFitter().fit(df, "T", "E")
    assert abs(ours.beta[0] - ll.params_["x"]) < 1e-6
    assert abs(ours.se[0] - ll.standard_errors_["x"]) < 1e-6


def test_logrank_and_maxstat():
    rng = np.random.default_rng(8)
    x = np.r_[rng.normal(-1.0, 0.5, 200), rng.normal(1.0, 0.5, 200)]
    t = rng.exponential(1.0 / np.exp(0.9 * (x > 0)))
    c = rng.exponential(4.0, size=len(x))
    time, event = np.minimum(t, c), (t <= c).astype(int)
    chi2, p = logrank_chi2(time, event, (x > 0).astype(float))
    assert chi2 > 20 and p < 1e-5
    cut, chi2b, pb = maxstat_cutpoint(x, time, event)
    assert 0.0 < cut < 1.0
    assert chi2b >= chi2


def test_km_and_cindex():
    time = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    event = np.ones(8, dtype=int)
    assert km_median(time, event) == 4.0
    assert concordance_index(time, event, -time) == 1.0  # earlier time = higher risk
    assert concordance_index(time, event, time) == 0.0   # reversed risk
