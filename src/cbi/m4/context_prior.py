"""Context prior and coverage-aware fusion (M4).

The context model is trained ONLY on natural labels (never red-team samples).
Fusion enforces combined == evidence when context_coverage == 0, so the main
red-team loop cannot be affected by absent/manipulated context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from cbi.schemas import ContextFeatures

CONTEXT_KEYS = [
    "supporting_papers",
    "effective_independent_support",
    "citation_cascade_score",
    "lab_historical_replication_rate",
    "self_citation_ratio",
    "review_time_days",
    "coi_count",
    "retraction",
    "correction",
    "pubpeer_threads",
]


def context_vector(features: ContextFeatures) -> tuple[np.ndarray, float]:
    values: list[float] = []
    for key in CONTEXT_KEYS:
        if key == "coi_count":
            v = float(len(features.coi_flags)) if features.coi_flags else 0.0
        elif key == "retraction":
            v = float(bool(features.post_pub_signals.get("retraction")))
        elif key == "correction":
            v = float(bool(features.post_pub_signals.get("correction")))
        elif key == "pubpeer_threads":
            v = float(features.post_pub_signals.get("pubpeer_threads", 0) or 0)
        else:
            raw = getattr(features, key, None)
            v = float(raw) if raw is not None else 0.0
        if key in {"supporting_papers", "review_time_days"}:
            v = float(np.log1p(max(v, 0.0)))
        values.append(v)
    missing = [
        float(features.missing_mask.get(k, getattr(features, k, None) is None))
        for k in ["supporting_papers", "effective_independent_support",
                  "citation_cascade_score", "lab_historical_replication_rate",
                  "self_citation_ratio", "review_time_days"]
    ]
    coverage = float(features.context_coverage)
    return np.asarray(values + missing + [coverage], dtype=float), coverage


@dataclass
class ContextPrior:
    model: Any
    base_rate: float
    feature_count: int
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None,
            seed: int = 0) -> "ContextPrior":
        base = float(np.average(y, weights=sample_weight)) if len(y) else 0.0
        if len(y) == 0 or len(np.unique(y)) < 2:
            return cls(model=None, base_rate=base, feature_count=X.shape[1] if X.ndim > 1 else 0)
        model = HistGradientBoostingClassifier(
            max_iter=120, max_depth=3, learning_rate=0.05, l2_regularization=1.0,
            min_samples_leaf=12, random_state=seed,
        )
        model.fit(X, y, sample_weight=sample_weight)
        return cls(model=model, base_rate=base, feature_count=X.shape[1])

    def predict(self, vector: np.ndarray, coverage: float) -> float:
        if coverage <= 0.05 or self.model is None:
            return float(self.base_rate)
        return float(self.model.predict_proba(vector.reshape(1, -1))[0, 1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "base_rate": self.base_rate,
                     "feature_count": self.feature_count, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ContextPrior":
        obj = joblib.load(path)
        return cls(model=obj.get("model"), base_rate=float(obj.get("base_rate", 0.0)),
                   feature_count=int(obj.get("feature_count", 0)),
                   metrics=dict(obj.get("metrics", {})))


@dataclass
class Fusion:
    slope: float = 1.0
    max_weight: float = 0.5
    evidence_threshold: float = 0.5
    evidence_threshold_1pct: float = 0.5
    combined_threshold: float = 0.5
    metrics: dict[str, Any] = field(default_factory=dict)

    def weight(self, coverage: float) -> float:
        if coverage <= 0.0:
            return 0.0
        return float(min(self.max_weight, self.slope * coverage))

    def combine(self, evidence: float, context: float | None, coverage: float) -> float:
        if context is None or coverage <= 0.0:
            return float(evidence)
        w = self.weight(coverage)
        return float(np.clip(evidence + w * (context - evidence), 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope": self.slope, "max_weight": self.max_weight,
            "evidence_threshold": self.evidence_threshold,
            "evidence_threshold_1pct": self.evidence_threshold_1pct,
            "combined_threshold": self.combined_threshold,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Fusion":
        return cls(
            slope=float(obj.get("slope", 1.0)),
            max_weight=float(obj.get("max_weight", 0.5)),
            evidence_threshold=float(obj.get("evidence_threshold", 0.5)),
            evidence_threshold_1pct=float(obj.get("evidence_threshold_1pct", 0.5)),
            combined_threshold=float(obj.get("combined_threshold", 0.5)),
            metrics=dict(obj.get("metrics", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Fusion":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
