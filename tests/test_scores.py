from __future__ import annotations

import math

import pytest

from cbi.m2.analysis import aligned
from cbi.m2.specs import CohortEffect


def test_inflation_table():
    s = math.log(2.0)
    cases = [(1.15, 1.00, 0.20), (2.1, 1.00, 0.79), (3.5, 1.00, 0.95), (2.1, 1.9, 0.14)]
    for reported, pooled, expected in cases:
        got = math.tanh(max(0.0, math.log(reported) - math.log(pooled)) / s)
        assert abs(got - expected) < 0.011


def test_direction_alignment():
    e = CohortEffect("c", "G", "OS", 0.4, 0.1, 0.01, 100, 30, "s")
    assert aligned(e, "high_expression_worse").log_effect == pytest.approx(0.4)
    assert aligned(e, "high_expression_better").log_effect == pytest.approx(-0.4)
    assert aligned(e, "low_expression_worse").log_effect == pytest.approx(-0.4)
    assert aligned(e, "low_expression_better").log_effect == pytest.approx(0.4)


def test_robustness_bounds():
    from cbi.util import clip01
    assert clip01(-1) == 0.0
    assert clip01(2) == 1.0
