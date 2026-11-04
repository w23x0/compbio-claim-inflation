from __future__ import annotations

import math

import numpy as np
import pytest

from cbi.m2.meta import meta_random_effects


def test_reml_matches_metafor_bcg_reference():
    # yi/vi from metafor::dat.bcg as printed in the metadat reference page.
    yi = np.array([-0.8893, -1.5854, -1.3481, -1.4416, -0.2175, -0.7861, -1.6209,
                   0.0120, -0.4694, -1.3713, -0.3394, 0.4459, -0.0173])
    vi = np.array([0.3256, 0.1946, 0.4154, 0.0200, 0.0512, 0.0069, 0.2230, 0.0040,
                   0.0564, 0.0730, 0.0124, 0.5325, 0.0714])
    m = meta_random_effects(yi, np.sqrt(vi))
    # metafor: estimate -0.7145, se 0.1798, tau^2 0.3132, PI exp 0.1546-1.5490.
    assert m.method == "REML"
    assert abs(m.pooled - (-0.7145)) < 5e-4
    assert abs(m.se - 0.1798) < 5e-4
    assert abs(m.tau2 - 0.3132) < 5e-4
    assert abs(math.exp(m.pred_low) - 0.1546) < 2e-3
    assert abs(math.exp(m.pred_high) - 1.5490) < 3e-3


def test_k_lt_three_falls_back_to_fixed_effect():
    m = meta_random_effects(np.array([0.2, 0.3]), np.array([0.1, 0.1]))
    assert m.method.startswith("FE")
    assert m.tau2 == 0.0
    assert abs(m.pooled - 0.25) < 1e-12


def test_zero_heterogeneity():
    y = np.array([0.2, 0.2, 0.2, 0.2])
    se = np.array([0.1, 0.1, 0.1, 0.1])
    m = meta_random_effects(y, se)
    assert m.tau2 == 0.0
    assert abs(m.pooled - 0.2) < 1e-12
    assert abs(m.se - 0.05) < 1e-12
