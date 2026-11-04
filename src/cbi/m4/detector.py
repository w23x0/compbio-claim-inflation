"""Detector façade: content evidence + optional context prior, with a hard
neutral-context path used by the M6 main loop (C7 / R4)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cbi.config import Config
from cbi.m4.context_prior import ContextPrior, Fusion, context_vector
from cbi.m4.evidence import EvidenceExtractor
from cbi.m4.llm_judge import llm_checklist_score, llm_checklist_score_many
from cbi.m4.model import EvidenceModel
from cbi.schemas import Claim, ContextFeatures, DetectorOutput
from cbi.util import read_json, safe_join, sha256_file, sha256_obj


class Detector:
    def __init__(self, cfg: Config, evidence: EvidenceModel | None, context: ContextPrior | None,
                 fusion: Fusion, *, version: str | None = None, backend: str = "checklist",
                 learned_examples: list[str] | None = None, evidence_variant: str = "rubric"):
        if evidence_variant not in {"rubric", "holistic"}:
            raise ValueError(f"unknown evidence variant: {evidence_variant}")
        self.cfg = cfg
        self.evidence_model = evidence
        self.context_model = context
        self.fusion = fusion
        self.backend = backend
        self.evidence_variant = evidence_variant
        self.learned_examples = list(learned_examples or [])
        self.version = version or (evidence.version if evidence is not None else "det-llm-1.0")
        self.extractor = EvidenceExtractor(cfg, cfg.lexicon, evidence.learned_patterns) if evidence is not None else None

    def _evidence(self, claim: Claim, manuscript: str) -> tuple[float, dict[str, float], list[str]]:
        if self.backend == "llm":
            judgement = llm_checklist_score(claim, manuscript, learned_examples=self.learned_examples)
            return self.judgement_score(judgement), {}, list(judgement["rationale"])
        assert self.evidence_model is not None and self.extractor is not None
        features = self.extractor.extract(claim, manuscript)
        score = self.evidence_model.predict(features)
        contributions = self.evidence_model.contributions(features)
        rationale = self.extractor.rationale(features, contributions)
        return score, features, rationale

    def score(self, claim: Claim, manuscript: str, context: ContextFeatures | None = None,
              *, neutral: bool = True) -> DetectorOutput:
        evidence, features, rationale = self._evidence(claim, manuscript)
        if neutral or context is None or self.context_model is None:
            coverage = 0.0
            context_pred = None
            combined = evidence
            threshold = self.fusion.evidence_threshold
            if neutral:
                rationale = rationale + ["neutral context: evidence_score only"]
        else:
            vec, coverage = context_vector(context)
            context_pred = self.context_model.predict(vec, coverage)
            combined = self.fusion.combine(evidence, context_pred, coverage)
            # If context cannot move the score (zero fusion weight), the fused
            # score is exactly evidence; use the evidence threshold.  The same
            # applies below minimal coverage.
            use_fused = coverage > 0.05 and self.fusion.max_weight > 0
            threshold = (self.fusion.combined_threshold if use_fused
                         else self.fusion.evidence_threshold)
        return DetectorOutput(
            claim_id=claim.claim_id,
            detector_version=self.version,
            evidence_score=float(evidence),
            context_prior=context_pred,
            context_coverage=float(coverage),
            combined_inflation_pred=float(combined),
            flag=bool(combined >= threshold),
            threshold_used=float(threshold),
            rationale=rationale,
            features=features,
        )

    def score_evidence_only(self, claim: Claim, manuscript: str) -> float:
        return self.score(claim, manuscript, neutral=True).evidence_score

    def judgement_score(self, judgement: dict) -> float:
        """Use the score variant selected with the detector's validation data."""
        key = "holistic" if self.evidence_variant == "holistic" else "score"
        return float(judgement[key])

    def flags_many(self, claims: list[Claim], manuscripts: list[str]) -> list[bool]:
        """Neutral-context flags for several reports, batched when possible."""
        if len(claims) != len(manuscripts):
            raise ValueError("claims and manuscripts must have the same length")
        if self.backend != "llm" or len(claims) < 2:
            return [self.score(c, m, neutral=True).flag for c, m in zip(claims, manuscripts)]
        judgements = llm_checklist_score_many(
            list(zip(claims, manuscripts)), learned_examples=self.learned_examples)
        out: list[bool] = []
        for claim, text, judgement in zip(claims, manuscripts, judgements):
            if judgement is None:
                out.append(self.score(claim, text, neutral=True).flag)
            else:
                out.append(bool(self.judgement_score(judgement) >= self.fusion.evidence_threshold))
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.evidence_model is not None:
            self.evidence_model.save(directory / "evidence.json")
        self.fusion.save(directory / "fusion.json")
        if self.context_model is not None:
            self.context_model.save(directory / "context.joblib")
        (directory / "meta.json").write_text(
            json.dumps({"version": self.version, "backend": self.backend,
                        "evidence_variant": self.evidence_variant,
                        "learned_examples": self.learned_examples}) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, cfg: Config, directory: str | Path) -> "Detector":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8")) if (directory / "meta.json").is_file() else {}
        backend = str(meta.get("backend", "checklist"))
        evidence = None if backend == "llm" else EvidenceModel.load(directory / "evidence.json")
        fusion = Fusion.load(directory / "fusion.json")
        context = None
        if (directory / "context.joblib").is_file():
            context = ContextPrior.load(directory / "context.joblib")
        return cls(cfg, evidence, context, fusion, version=meta.get("version"),
                   backend=backend, learned_examples=meta.get("learned_examples"),
                   evidence_variant=meta.get("evidence_variant", "rubric"))


def detector_fingerprint(directory: str | Path) -> str:
    directory = Path(directory)
    return sha256_obj({name: sha256_file(directory / name)
                       for name in ("meta.json", "fusion.json", "context.joblib", "evidence.json")
                       if (directory / name).is_file()})


def active_detector_dir(models_dir: str | Path) -> Path:
    """Return the newest trained detector directory.

    ``detector_llm_v2`` is the retrained content monitor (monitor-document
    input + red-flag rubric); ``detector_llm`` stays as the frozen v1 baseline.
    """
    models = Path(models_dir)
    for name in ("detector_llm_v2", "detector_llm"):
        candidate = models / name
        if (candidate / "meta.json").is_file() and (candidate / "fusion.json").is_file():
            pointer = models / "blue_llm" / "active.json"
            if name == "detector_llm_v2" and pointer.is_file():
                try:
                    active = read_json(pointer)
                    blue = safe_join(models, active["directory"])
                    if (blue.is_relative_to((models / "blue_llm").resolve())
                            and (blue / "meta.json").is_file() and (blue / "fusion.json").is_file()
                            and active["base_fingerprint"] == detector_fingerprint(candidate)
                            and active["detector_fingerprint"] == detector_fingerprint(blue)):
                        return blue
                except (ValueError, OSError, KeyError, TypeError):
                    pass
            return candidate
    raise RuntimeError("no trained detector found; run P4 first")
