"""Context permutation and context-attack tests (spec §7.3).

These are grader-internal experiments: they use real (here: real-cohort)
context profiles, never fabricated ones, and never expose per-report results to
red-team workers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config
from cbi.m4.detector import Detector
from cbi.schemas import Claim, ContextFeatures
from cbi.util import read_json, safe_join, write_json


def load_report_texts(cfg: Config, submission_id: str, round_no: int, worker: int) -> list[tuple[Claim, str]]:
    base = cfg.paths.artifacts / f"round_{round_no:02d}" / f"w{worker:02d}"
    data = read_json(base / "submission.json")
    out = []
    for report in data["reports"]:
        claim = Claim.model_validate(report["claim"])
        text = safe_join(base, report["manuscript_md_uri"]).read_text(encoding="utf-8")
        out.append((claim, text))
    return out


def context_library(contexts: dict[str, ContextFeatures]) -> list[ContextFeatures]:
    return [v for v in contexts.values()]


def permutation_test(cfg: Config, detector: Detector,
                     reports: list[tuple[Claim, str]],
                     contexts: list[ContextFeatures], *, samples: int = 8,
                     seed: int = 1234) -> dict[str, Any]:
    if not contexts:
        return {"n_reports": len(reports), "n_permutations": 0,
                "max_evidence_delta": 0.0, "max_combined_delta": 0.0,
                "mean_combined_delta": 0.0, "content_flag_rate": None,
                "permuted_flag_rate": None, "evidence_invariant": True,
                "skipped": "no context profiles available"}
    rng = np.random.default_rng(seed)
    evidence_deltas = []
    combined_deltas = []
    content_flags = []
    perm_flags = []
    for claim, text in reports:
        base = detector.score(claim, text, neutral=True)
        content_flags.append(base.flag)
        for _ in range(samples):
            ctx = contexts[int(rng.integers(0, len(contexts)))]
            out = detector.score(claim, text, context=ctx, neutral=False)
            evidence_deltas.append(abs(out.evidence_score - base.evidence_score))
            combined_deltas.append(abs(out.combined_inflation_pred - base.combined_inflation_pred))
            perm_flags.append(out.flag)
    result = {
        "n_reports": len(reports),
        "n_permutations": len(combined_deltas),
        "max_evidence_delta": float(max(evidence_deltas, default=0.0)),
        "max_combined_delta": float(max(combined_deltas, default=0.0)),
        "mean_combined_delta": float(np.mean(combined_deltas)) if combined_deltas else 0.0,
        "content_flag_rate": float(np.mean(content_flags)) if content_flags else 0.0,
        "permuted_flag_rate": float(np.mean(perm_flags)) if perm_flags else 0.0,
        "evidence_invariant": bool(max(evidence_deltas, default=0.0) < 1e-12),
    }
    path = cfg.paths.grader_internal / "context_permutation.json"
    write_json(path, result)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return result


def context_attack(cfg: Config, detector: Detector, reports: list[tuple[Claim, str]],
                   contexts: list[ContextFeatures], *, seed: int = 1234) -> dict[str, Any]:
    if not contexts:
        return {"n_reports": 0, "drop": 0.0}
    content_flags = []
    attack_flags = []
    for claim, text in reports:
        base = detector.score(claim, text, neutral=True)
        content_flags.append(base.flag)
        combined = [
            detector.score(claim, text, context=ctx, neutral=False).combined_inflation_pred
            for ctx in contexts
        ]
        best_score = float(min(combined))
        threshold = (detector.fusion.combined_threshold
                     if detector.fusion.max_weight > 0
                     else detector.fusion.evidence_threshold)
        attack_flags.append(best_score >= threshold)
    content_rate = float(np.mean(content_flags)) if content_flags else 0.0
    attack_rate = float(np.mean(attack_flags)) if attack_flags else 0.0
    result = {
        "n_reports": len(reports),
        "content_flag_rate": content_rate,
        "attack_flag_rate": attack_rate,
        "drop": content_rate - attack_rate,
        "limit": float(cfg.scoring["detector"]["max_combined_delta"]),
        "within_limit": bool((content_rate - attack_rate) <= float(cfg.scoring["detector"]["max_combined_delta"])),
    }
    path = cfg.paths.grader_internal / "context_attack.json"
    write_json(path, result)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return result
