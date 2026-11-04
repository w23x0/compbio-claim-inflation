"""Downstream claim/edge weighting API (M8)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config
from cbi.m4.detector import Detector
from cbi.m8 import export as weight_export
from cbi.schemas import Claim, ContextFeatures, GroundTruthResult
from cbi.util import read_json, write_json


@dataclass
class ClaimWeightStore:
    cfg: Config
    claims: dict[str, Claim]
    gt: dict[str, GroundTruthResult]
    contexts: dict[str, ContextFeatures]
    detector: Detector | None
    weights: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, cfg: Config, detector: Detector | None = None) -> "ClaimWeightStore":
        claims = {c.claim_id: c for c in
                  [Claim.model_validate(x) for x in read_json(cfg.paths.processed / "corpus" / "claims.json")]}
        gt = {cid: GroundTruthResult.model_validate(v) for cid, v in
              read_json(cfg.paths.processed / "ground_truth.json").items()}
        ctx_path = cfg.paths.processed / "context_features.json"
        contexts = {}
        if ctx_path.is_file():
            contexts = {cid: ContextFeatures.model_validate(v) for cid, v in read_json(ctx_path).items()}
        if detector is None:
            candidates = sorted(cfg.paths.models.glob("detector_v*"))
            if candidates:
                detector = Detector.load(cfg, candidates[-1])
        weights: dict[str, dict[str, Any]] = {}
        for cid, claim in claims.items():
            if cid in gt:
                r = gt[cid]
                weights[cid] = {
                    "claim_id": cid, "weight": float(r.robustness_score),
                    "source": "ground_truth", "inflation": float(r.inflation),
                    "gt_confidence": r.gt_confidence, "flags": r.reconciliation_flags,
                }
            elif detector is not None:
                text_path = cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md"
                text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
                out = detector.score(claim, text, context=contexts.get(cid), neutral=contexts.get(cid) is None)
                weights[cid] = {
                    "claim_id": cid, "weight": float(1.0 - out.combined_inflation_pred),
                    "source": "detector", "inflation": float(out.combined_inflation_pred),
                    "gt_confidence": None, "flags": [],
                }
            else:
                weights[cid] = {"claim_id": cid, "weight": 0.0, "source": "unscored",
                                "inflation": None, "gt_confidence": None, "flags": []}
        store = cls(cfg=cfg, claims=claims, gt=gt, contexts=contexts,
                    detector=detector, weights=weights)
        write_json(cfg.paths.processed / "claim_weights.json", weights)
        return store

    def get_claim_weight(self, claim_id: str) -> dict[str, Any]:
        if claim_id not in self.weights:
            raise KeyError(f"unknown claim: {claim_id}")
        return self.weights[claim_id]

    def get_edge_weight(self, gene: str, disease: str, relation: str = "prognostic_biomarker") -> dict[str, Any]:
        gene_u, disease_u = gene.upper(), disease.upper()
        matched = [c for c in self.claims.values()
                   if c.entities.gene.upper() == gene_u and c.entities.cancer_type.upper() == disease_u]
        if not matched:
            return {"gene": gene, "disease": disease, "relation": relation, "n_supporting": 0,
                    "effective_independent_support": 0.0, "total_weight": 0.0,
                    "edge_weight": 0.0, "claim_ids": []}
        total = sum(float(self.weights[c.claim_id]["weight"]) for c in matched)
        eis_values = [self.contexts[c.claim_id].effective_independent_support for c in matched
                      if c.claim_id in self.contexts and self.contexts[c.claim_id].effective_independent_support is not None]
        eis = float(max(1.0, np.mean(eis_values))) if eis_values else float(len(matched))
        return {
            "gene": gene, "disease": disease, "relation": relation,
            "n_supporting": len(matched),
            "effective_independent_support": eis,
            "total_weight": float(total),
            "edge_weight": float(total / eis),
            "claim_ids": [c.claim_id for c in matched],
        }

    def all_edges(self) -> list[dict[str, Any]]:
        keys = {(c.entities.gene, c.entities.cancer_type) for c in self.claims.values()}
        return [self.get_edge_weight(g, d) for g, d in sorted(keys)]

    def to_records(self) -> list[dict[str, Any]]:
        return weight_export.records(self.weights.values(), self.all_edges())

    def export(self, path: str | Path, fmt: str | None = None) -> Path:
        return weight_export.export(self.to_records(), path, fmt)
