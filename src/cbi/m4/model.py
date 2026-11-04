"""Evidence checklist model (M4).

The score is a transparent signed checklist, calibrated to natural-label
probabilities with a one-dimensional Platt fit.  This avoids unstable
coefficients when checklist items are correlated and keeps the sign of every
indicator interpretable: risk indicators can only raise the score, transparency
indicators can only lower it.  FPR thresholds are calibrated on natural
validation data.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from cbi.m4.evidence import FEATURE_LABELS

# Signed checklist weights.  Zero means the item is recorded but not scored.
FEATURE_WEIGHTS: dict[str, float] = {
    "optimal_cutpoint": 2.0,
    "dichotomous_unreported_cutpoint": 0.6,
    "subgroup_signal": 1.0,
    "subgroup_claim": 1.0,
    "no_validation": 0.2,
    "multiplicity_unreported": 1.5,
    "many_candidates": 0.8,
    "endpoint_switch": 0.8,
    "covariate_fishing": 0.8,
    "probe_selection": 0.4,
    "followup_custom": 0.6,
    "best_subset_language": 0.4,
    "sample_count_gap": 0.4,
    "n_reported_exceeds_deposited": 1.0,
    "small_n": 0.2,
    "large_effect": 0.4,
    "no_code": 0.4,
    "no_raw_data": 0.3,
    "ci_p_inconsistent": 1.0,
    "no_ci": 0.2,
    "transparent_cutpoint": -1.0,
    "validation_signal": -0.5,
    "prespecified": -0.8,
    "limitations": -0.2,
    "ci": -0.2,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        return 1.0
    allowed = int(math.floor(target * len(neg)))
    if allowed >= len(neg):
        return float(neg[0])
    if allowed == 0:
        return float(np.nextafter(neg[-1], np.inf))
    return float(np.nextafter(neg[-allowed - 1], np.inf))


def fpr_recall(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(scores) >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return (fp / neg if neg else 0.0, tp / pos if pos else 0.0)


@dataclass
class EvidenceModel:
    version: str
    feature_names: list[str]
    weights: dict[str, float]
    calibration_a: float
    calibration_b: float
    thresholds: dict[str, float]
    learned_patterns: dict[str, list[str]]
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None,
            version: str = "det-1.1.0", fpr_targets: list[float] | None = None,
            learned_patterns: dict[str, list[str]] | None = None) -> "EvidenceModel":
        weights = {name: float(FEATURE_WEIGHTS.get(name, 0.0)) for name in FEATURE_LABELS}
        raw = np.asarray(X, dtype=float) @ np.asarray([weights[n] for n in FEATURE_LABELS])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(raw.reshape(-1, 1), y, sample_weight=sample_weight)
        a = float(clf.coef_[0, 0])
        b = float(clf.intercept_[0])
        # A negative slope is a real (bad) fit on a small sample; record it
        # instead of silently replacing the model with an uncalibrated score.
        inverted = a < 0
        model = cls(
            version=version, feature_names=list(FEATURE_LABELS), weights=weights,
            calibration_a=a, calibration_b=b, thresholds={},
            learned_patterns=learned_patterns or {}, metrics={"calibration_inverted": inverted},
        )
        scores = np.asarray([model.predict(dict(zip(FEATURE_LABELS, row))) for row in X])
        for target in (fpr_targets or [0.05]):
            model.thresholds[f"fpr_{target:g}"] = threshold_at_fpr(y, scores, target)
        return model

    def raw_score(self, features: dict[str, float]) -> float:
        return float(sum(self.weights.get(n, 0.0) * float(features.get(n, 0.0)) for n in self.feature_names))

    def linear(self, features: dict[str, float]) -> float:
        return self.calibration_a * self.raw_score(features) + self.calibration_b

    def predict(self, features: dict[str, float]) -> float:
        return sigmoid(self.linear(features))

    def contributions(self, features: dict[str, float]) -> dict[str, float]:
        scale = max(self.calibration_a, 0.0)
        return {n: float(scale * self.weights.get(n, 0.0) * features.get(n, 0.0))
                for n in self.feature_names}

    def threshold(self, fpr_target: float = 0.05) -> float:
        return float(self.thresholds.get(f"fpr_{fpr_target:g}", 0.5))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "checklist-v2",
            "version": self.version,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "calibration": {"a": self.calibration_a, "b": self.calibration_b},
            "thresholds": self.thresholds,
            "learned_patterns": self.learned_patterns,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "EvidenceModel":
        if obj.get("format") == "checklist-v2":
            return cls(
                version=obj.get("version", "det-1.1.0"),
                feature_names=list(obj["feature_names"]),
                weights={k: float(v) for k, v in obj.get("weights", {}).items()},
                calibration_a=float(obj.get("calibration", {}).get("a", 1.0)),
                calibration_b=float(obj.get("calibration", {}).get("b", 0.0)),
                thresholds={k: float(v) for k, v in obj.get("thresholds", {}).items()},
                learned_patterns={k: list(v) for k, v in obj.get("learned_patterns", {}).items()},
                metrics=dict(obj.get("metrics", {})),
            )
        # Legacy linear model: preserve predictions, but new reports use v2.
        coefs = np.asarray(obj["coefficients"], dtype=float)
        names = list(obj["feature_names"])
        return cls(
            version=obj.get("version", "det-1.1.0"), feature_names=names,
            weights={n: float(c) for n, c in zip(names, coefs)},
            calibration_a=1.0, calibration_b=float(obj["intercept"]),
            thresholds={k: float(v) for k, v in obj.get("thresholds", {}).items()},
            learned_patterns={k: list(v) for k, v in obj.get("learned_patterns", {}).items()},
            metrics=dict(obj.get("metrics", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
