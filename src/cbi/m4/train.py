"""Train and evaluate detector v0 (P4) on natural labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from cbi.config import Config, ensure_layout
from cbi.m4.context_prior import ContextPrior, Fusion, context_vector
from cbi.m4.detector import Detector
from cbi.m4.evidence import FEATURE_LABELS, EvidenceExtractor
from cbi.m4.model import EvidenceModel, fpr_recall, threshold_at_fpr
from cbi.schemas import Claim, ContextFeatures, GroundTruthResult
from cbi.util import read_json, write_json


def load_natural_corpus(cfg: Config) -> tuple[list[Claim], dict[str, str], dict[str, ContextFeatures]]:
    base = cfg.paths.processed / "corpus"
    claims = [Claim.model_validate(x) for x in read_json(base / "claims.json")]
    split = read_json(base / "natural_split.json")["by_claim"]
    context_path = cfg.paths.processed / "context_features.json"
    contexts: dict[str, ContextFeatures] = {}
    if context_path.is_file():
        for cid, obj in read_json(context_path).items():
            contexts[cid] = ContextFeatures.model_validate(obj)
    return claims, {cid: obj["split"] for cid, obj in split.items()}, contexts


def labels_and_weights(results: dict[str, GroundTruthResult], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    y, w = [], []
    for cid, r in results.items():
        pos = float(r.inflation_z >= cfg.scoring["detector"]["label_min_z"] and
                    r.inflation >= cfg.scoring["detector"]["label_min_inflation"])
        y.append(pos)
        w.append({"high": 1.0, "medium": 0.6, "low": 0.25}[r.gt_confidence])
    return np.asarray(y), np.asarray(w)


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(np.unique(y)) < 2:
        return out
    out["auc"] = float(roc_auc_score(y, scores))
    for target in [0.01, 0.05]:
        thr = threshold_at_fpr(y, scores, target)
        fpr, rec = fpr_recall(y, scores, thr)
        out[f"recall_at_fpr_{target:g}"] = rec
        out[f"threshold_fpr_{target:g}"] = thr
        out[f"empirical_fpr_{target:g}"] = fpr
    # Calibration bins.
    bins = []
    for lo in np.linspace(0, 0.9, 10):
        mask = (scores >= lo) & (scores < lo + 0.1)
        if mask.sum() >= 5:
            bins.append({"bin": [float(lo), float(lo + 0.1)], "n": int(mask.sum()),
                         "pred": float(scores[mask].mean()), "obs": float(y[mask].mean())})
    out["calibration"] = bins
    return out


def train_detector_v0(cfg: Config) -> tuple[Detector, dict[str, Any], dict[str, Any]]:
    ensure_layout(cfg)
    claims, split, contexts = load_natural_corpus(cfg)
    gt = {cid: GroundTruthResult.model_validate(x) for cid, x in read_json(cfg.paths.processed / "ground_truth.json").items()}
    # align by corpus order; only claims with independent evidence are labels
    eligible = {cid for cid, r in gt.items() if r.label_eligible and r.independent_cohorts}
    train_ids = [c.claim_id for c in claims if split[c.claim_id] == "train" and c.claim_id in eligible]
    val_ids = [c.claim_id for c in claims if split[c.claim_id] == "val" and c.claim_id in eligible]
    test_ids = [c.claim_id for c in claims if split[c.claim_id] == "test" and c.claim_id in eligible]
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError("natural split must contain train/val/test claims")
    extractor = EvidenceExtractor(cfg, cfg.lexicon)
    def matrix(ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
        X, y, w, feats = [], [], [], []
        for cid in ids:
            claim = next(c for c in claims if c.claim_id == cid)
            text = (cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md").read_text(encoding="utf-8")
            f = extractor.extract(claim, text)
            feats.append(f)
            X.append([f[n] for n in FEATURE_LABELS])
            yy, ww = labels_and_weights({cid: gt[cid]}, cfg)
            y.append(float(yy[0])); w.append(float(ww[0]))
        return np.asarray(X, dtype=float), np.asarray(y), np.asarray(w), feats
    Xtr, ytr, wtr, _ = matrix(train_ids)
    Xval, yval, wval, _ = matrix(val_ids)
    Xte, yte, wte, _ = matrix(test_ids)
    evidence = EvidenceModel.fit(Xtr, ytr, sample_weight=wtr, version="det-1.1.0")
    # Thresholds calibrated on validation at FPR 5% (primary) and 1%.
    val_scores = np.asarray([evidence.predict(dict(zip(FEATURE_LABELS, row))) for row in Xval])
    evidence.thresholds = {
        f"fpr_{t:g}": threshold_at_fpr(yval, val_scores, t) for t in [0.01, 0.05]
    }
    evidence.metrics = {"val": _metrics(yval, val_scores),
                        "test": _metrics(yte, np.asarray([evidence.predict(dict(zip(FEATURE_LABELS, row))) for row in Xte])),
                        "calibration_inverted": bool(evidence.calibration_a < 0)}
    # Context model: natural train only.  If no context snapshot is available,
    # keep the base rate so evidence-only training still works.
    if contexts:
        pairs = [(contexts[cid], yi, wi) for cid, yi, wi in zip(train_ids, ytr, wtr) if cid in contexts]
        if pairs:
            vec_train = np.asarray([context_vector(c)[0] for c, _, _ in pairs])
            ctx_y = np.asarray([yi for _, yi, _ in pairs])
            ctx_w = np.asarray([wi for _, _, wi in pairs])
            ctx_model = ContextPrior.fit(vec_train, ctx_y, sample_weight=ctx_w, seed=int(cfg.mvp["seed"]))
        else:
            ctx_model = ContextPrior(model=None, base_rate=float(np.average(ytr, weights=wtr)), feature_count=0)
    else:
        ctx_model = ContextPrior(model=None, base_rate=float(np.average(ytr, weights=wtr)), feature_count=0)
    # Fusion learned on validation only.
    fusion = Fusion()
    best = None
    val_ev = val_scores
    for slope in [0.5, 1.0, 2.0]:
        for wmax in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            comb = []
            covs = []
            for cid, ev in zip(val_ids, val_ev):
                if cid in contexts:
                    vec, cov = context_vector(contexts[cid])
                    cp = ctx_model.predict(vec, cov)
                    comb.append(min(wmax, slope * cov) * (cp - ev) + ev)
                    covs.append(cov)
                else:
                    comb.append(ev); covs.append(0.0)
            comb = np.asarray(comb)
            # Logistic loss as a proper scoring rule; tie-break to less context.
            eps = 1e-9
            ll = float(-np.mean(yval * np.log(comb + eps) + (1 - yval) * np.log(1 - comb + eps)))
            key = (ll, wmax, slope)
            if best is None or key < best[0]:
                best = (key, slope, wmax, comb)
    _, slope, wmax, comb_val = best
    fusion.slope = slope
    fusion.max_weight = wmax
    # Main-loop neutral screening uses the stricter 1% threshold; 5% is
    # reported for recall-at-fixed-FPR evaluation.
    fusion.evidence_threshold = float(evidence.threshold(0.01))
    fusion.combined_threshold = threshold_at_fpr(yval, comb_val, 0.05)
    # Test combined using val-learned fusion.
    test_scores = []
    for cid, row in zip(test_ids, Xte):
        ev = evidence.predict(dict(zip(FEATURE_LABELS, row)))
        if cid in contexts:
            vec, cov = context_vector(contexts[cid])
            cp = ctx_model.predict(vec, cov)
            test_scores.append(fusion.combine(ev, cp, cov))
        else:
            test_scores.append(ev)
    test_scores = np.asarray(test_scores)
    fusion.metrics = {"val": _metrics(yval, comb_val), "test": _metrics(yte, test_scores)}
    # Context-only ablation on the natural test split.
    ctx_test_scores = []
    for cid in test_ids:
        if cid in contexts:
            vec, cov = context_vector(contexts[cid])
            ctx_test_scores.append(ctx_model.predict(vec, cov))
        else:
            ctx_test_scores.append(ctx_model.base_rate)
    ctx_test_scores = np.asarray(ctx_test_scores)
    detector = Detector(cfg, evidence, ctx_model, fusion, version="det-1.1.0")
    det_dir = cfg.paths.models / "detector_v1"
    detector.save(det_dir)
    report = {
        "train_n": len(train_ids), "val_n": len(val_ids), "test_n": len(test_ids),
        "evidence_val": evidence.metrics.get("val"), "evidence_test": evidence.metrics.get("test"),
        "combined_val": fusion.metrics.get("val"), "combined_test": fusion.metrics.get("test"),
        "context_only_test": _metrics(yte, ctx_test_scores),
        "ablation": {"evidence_only": evidence.metrics.get("test"),
                     "context_only": _metrics(yte, ctx_test_scores),
                     "combined": fusion.metrics.get("test")},
        "fusion": fusion.to_dict(), "calibration_inverted": bool(evidence.calibration_a < 0),
        "positive_rate": {
            "train": float(ytr.mean()), "val": float(yval.mean()), "test": float(yte.mean())},
        "context_coverage": {
            "mean": float(np.mean([contexts[c].context_coverage for c in contexts])) if contexts else 0.0,
            "test_mean": float(np.mean([contexts[c].context_coverage for c in test_ids if c in contexts])) if contexts else 0.0,
        },
    }
    write_json(cfg.paths.reports / "P4_metrics.json", report)
    write_json(cfg.paths.models / "detector_v1" / "train_report.json", report)
    return detector, report, {"Xte": Xte, "yte": yte, "test_ids": test_ids, "contexts": contexts}
