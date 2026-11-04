"""Unit tests for the P8 downstream prior aggregation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from cbi.m8.downstream import _risk, build_cancer_prior


def _claim(cid: str, gene: str, cancer: str, hr: float) -> SimpleNamespace:
    return SimpleNamespace(
        claim_id=cid,
        entities=SimpleNamespace(gene=gene, cancer_type=cancer),
        reported_effect=SimpleNamespace(value=hr),
    )


def test_prior_aggregates_signed_weights_and_directions():
    claims = [
        _claim("c1", "G1", "BRCA", 2.0),
        _claim("c2", "G1", "BRCA", 0.5),
        _claim("c3", "G2", "BRCA", 0.5),
        _claim("c4", "G3", "LUAD", 1.0),
    ]
    weights = {
        "c1": {"weight": 0.8},
        "c2": {"weight": 0.2},
        "c3": {"weight": 0.0},
        "c4": {"weight": 1.0},
    }
    prior = build_cancer_prior(claims, weights, "BRCA")
    assert set(prior) == {"G1", "G2"}
    # Conflicting claims partially cancel: (0.8 - 0.2) / 2.
    assert prior["G1"].signed_weight == pytest.approx(0.3)
    assert prior["G1"].direction == 1
    # A zero-weight claim keeps its reported direction for the uniform arm.
    assert prior["G2"].signed_weight == 0.0
    assert prior["G2"].direction == -1


def test_risk_uses_same_gene_set_and_flips_with_direction():
    expr = pd.DataFrame({
        "G1": [1.0, 2.0, 3.0, 4.0],
        "G2": [1.0, 2.0, 3.0, 4.0],
    })
    risk_both_worse = _risk(expr, ["G1", "G2"], np.asarray([1.0, 1.0]))
    risk_opposite = _risk(expr, ["G1", "G2"], np.asarray([1.0, -1.0]))
    assert np.all(np.diff(risk_both_worse) > 0)
    # Equal-magnitude opposite directions cancel exactly.
    assert np.allclose(risk_opposite, 0.0)
