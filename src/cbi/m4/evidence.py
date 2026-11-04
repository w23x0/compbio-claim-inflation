"""Content-only evidence checklist features (M4, no M2 calls at inference)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from cbi.config import Config
from cbi.schemas import Claim

FEATURE_LABELS = {
    "optimal_cutpoint": "data-driven / optimal cutpoint language",
    "transparent_cutpoint": "pre-specified or conventional cutpoint reported",
    "dichotomous_unreported_cutpoint": "dichotomous contrast without a cutpoint method",
    "subgroup_signal": "subgroup / restricted-analysis language",
    "subgroup_claim": "claim restricts the analysis to a subset",
    "validation_signal": "independent validation / replication language",
    "no_validation": "no independent validation mentioned",
    "multiple_testing": "multiplicity correction mentioned",
    "many_candidates": "many candidate genes/markers tested",
    "multiplicity_unreported": "many comparisons without correction",
    "endpoint_switch": "multiple/favourable endpoint language",
    "covariate_fishing": "outcome-based covariate/model selection language",
    "probe_selection": "probe selection by significance",
    "followup_custom": "non-standard follow-up truncation",
    "significance_language": "significance-focused reporting",
    "prespecified": "prespecified primary analysis language",
    "novelty": "novelty / first-report language",
    "limitations": "limitations disclosed",
    "ci": "confidence interval reported",
    "no_ci": "no confidence interval",
    "pvalue": "p value reported",
    "hr": "hazard ratio reported",
    "best_subset_language": "best/strongest association language",
    "sample_count_gap": "reported sample count below deposited count",
    "n_reported_exceeds_deposited": "reported sample count exceeds deposited count",
    "small_n": "small analysed sample size",
    "large_effect": "implausibly large effect",
    "no_code": "analysis code unavailable",
    "no_raw_data": "raw data unavailable",
    "se_derived_from_p": "SE derived from p value",
    "ci_p_inconsistent": "CI and p value inconsistent",
    "percentile_language": "percentile-based threshold",
    "early_stage_language": "early-stage focus",
    "advanced_stage_language": "advanced-stage focus",
    "best_model_language": "best model/subset reported",
    "text_length": "manuscript length (scaled)",
}


@dataclass
class EvidenceExtractor:
    cfg: Config
    lexicon: dict[str, Any]
    learned_patterns: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._compiled: dict[str, list[re.Pattern]] = {}
        base = self.lexicon.get("patterns", {})
        for group, patterns in base.items():
            self._compiled[group] = [re.compile(p, re.I) for p in dict.fromkeys(patterns)]
        for group, patterns in self.learned_patterns.items():
            combined = list(dict.fromkeys(list(base.get(group, [])) + list(patterns)))
            self._compiled[group] = [re.compile(p, re.I) for p in combined]

    def _matches(self, group: str, text: str) -> int:
        return sum(1 for p in self._compiled.get(group, []) if p.search(text))

    def extract(self, claim: Claim, manuscript: str) -> dict[str, float]:
        text = manuscript
        f: dict[str, float] = {}
        for group in self._compiled:
            f[group] = 1.0 if self._matches(group, text) else 0.0
        f["subgroup_signal"] = f.get("subgroup", 0.0)
        f["validation_signal"] = f.get("validation", 0.0)
        f["percentile_language"] = f.get("percentiles", 0.0)
        # Text-derived derived features.
        f["dichotomous_unreported_cutpoint"] = float(
            claim.reported_choices.contrast_type == "dichotomous"
            and not f.get("transparent_cutpoint")
            and not f.get("optimal_cutpoint")
            and not f.get("percentile_language")
        )
        f["no_validation"] = 1.0 - float(f.get("validation_signal", 0.0) > 0)
        f["multiplicity_unreported"] = float(
            (claim.reported_choices.n_comparisons or 0) > 10
            and claim.reported_choices.multiple_testing_correction is not True
        )
        f["subgroup_claim"] = float(bool(claim.reported_choices.subset or claim.reported_cohort.subset_desc))
        f["no_ci"] = float(claim.reported_effect.ci_low is None or claim.reported_effect.ci_high is None)
        f["no_code"] = float(not claim.data_availability.code_available)
        f["no_raw_data"] = float(not claim.data_availability.raw_deposited)
        f["se_derived_from_p"] = float(claim.reported_effect.se_derived_from_p)
        n_rep, dep = claim.reported_cohort.n_reported, claim.data_availability.deposited_n
        if n_rep and dep:
            f["sample_count_gap"] = float(n_rep < 0.9 * dep)
            f["n_reported_exceeds_deposited"] = float(n_rep > dep)
        else:
            f["sample_count_gap"] = 0.0
            f["n_reported_exceeds_deposited"] = 0.0
        f["small_n"] = float((claim.reported_cohort.n_reported or 10**6) < 100)
        hr = claim.reported_effect.value
        f["large_effect"] = float(hr > 3.0 or hr < 1 / 3.0)
        # CI/p consistency using the reported p value (normal approximation).
        inconsistent = 0.0
        if claim.reported_effect.ci_low is not None and claim.reported_effect.ci_high is not None \
                and claim.reported_effect.p is not None:
            excludes_one = (claim.reported_effect.ci_low > 1.0) or (claim.reported_effect.ci_high < 1.0)
            if excludes_one and claim.reported_effect.p > 0.05:
                inconsistent = 1.0
            if not excludes_one and claim.reported_effect.p < 0.01:
                inconsistent = 1.0
        f["ci_p_inconsistent"] = inconsistent
        # Short aliases used by rationale/model.
        f["percentile_language"] = float(f.get("percentiles", 0.0) > 0)
        low = text.lower()
        f["early_stage_language"] = float("early-stage" in low or "early stage" in low or "stage i-ii" in low)
        f["advanced_stage_language"] = float("advanced" in low or "stage iii-iv" in low)
        f["best_model_language"] = float(f.get("best_subset_language", 0.0) > 0 or f.get("covariate_fishing", 0.0) > 0)
        f["text_length"] = float(min(len(text) / 6000.0, 2.0))
        # Keep the vocabulary stable across detector versions.
        return {name: float(f.get(name, 0.0)) for name in FEATURE_LABELS}

    def feature_vector(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in FEATURE_LABELS]

    def rationale(self, features: dict[str, float], contributions: dict[str, float],
                  top: int = 4) -> list[str]:
        items = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
        out = []
        for name, value in items[:top]:
            if abs(value) < 0.05:
                continue
            label = FEATURE_LABELS.get(name, name)
            direction = "increases" if value > 0 else "decreases"
            out.append(f"{label} ({direction} inflation risk; contribution {value:+.2f})")
        return out
