"""Regression test: the combined detector arm must not reuse evidence scores."""
from __future__ import annotations

from datetime import date

import numpy as np

from cbi.m4.context_prior import Fusion
from cbi.m4.train_llm import combine_with_context
from cbi.schemas import ContextFeatures


class _Model:
    base_rate = 0.5

    def predict(self, vector: np.ndarray, coverage: float) -> float:
        return 0.9


def _features(cid: str) -> ContextFeatures:
    return ContextFeatures(
        claim_id=cid,
        as_of_date=date(2020, 1, 1),
        author_cluster_id="ac_0001",
        supporting_papers=0,
        effective_independent_support=None,
        citation_cascade_score=0.0,
        lab_historical_replication_rate=None,
        self_citation_ratio=0.0,
        review_time_days=None,
        missing_mask={
            "supporting_papers": False,
            "effective_independent_support": True,
            "citation_cascade_score": False,
            "lab_historical_replication_rate": True,
            "self_citation_ratio": False,
            "review_time_days": True,
        },
    )


def test_combined_uses_context_and_falls_back_without_it():
    fusion = Fusion(slope=1.0, max_weight=0.5)
    ctx, combined = combine_with_context(
        ["with_context", "without_context"], [0.2, 0.2],
        {"with_context": _features("with_context")}, _Model(), fusion,
    )
    assert ctx[0] == 0.9
    assert combined[0] == 0.55  # 0.2 + min(0.5, 0.571) * (0.9 - 0.2)
    assert ctx[1] == 0.5
    assert combined[1] == 0.2
    assert not np.allclose(combined, [0.2, 0.2])
