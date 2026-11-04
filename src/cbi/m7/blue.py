"""Blue-team updates (M7): mine blind spots, extend the content checklist, and
re-run natural-label / holdout-family / context-permutation checks.

Only `evidence_score` is updated from red-team artifacts.  Context-prior and
fusion parameters remain those trained on natural labels.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from cbi.config import Config, ensure_layout
from cbi.m4.context_prior import context_vector
from cbi.m4.detector import Detector, active_detector_dir
from cbi.m4.evidence import FEATURE_LABELS, EvidenceExtractor
from cbi.m4.model import EvidenceModel, fpr_recall, threshold_at_fpr
from cbi.m4.train import _metrics, labels_and_weights, load_natural_corpus
from cbi.m5.orchestrator import run_round
from cbi.m6.context_sampler import context_attack, load_report_texts, permutation_test
from cbi.schemas import Claim, ContextFeatures, GroundTruthResult
from cbi.util import read_json, read_jsonl, safe_join, write_json

# Phrases are mined from blind-spot manuscripts; only phrases that are rare in
# the natural training corpus are added, so honest reporting does not get
# blanket-flagged.
CANDIDATE_PHRASES: dict[str, list[str]] = {
    "optimal_cutpoint": [
        "classified into high- and low-expression groups",
        "maximize separation", "maximise separation",
        "threshold was chosen", "cutoff was chosen", "threshold was selected",
        "after inspection of the survival curves",
        "data-driven change point", "data driven change point",
    ],
    "subgroup": [
        "focused on patients with early-stage", "focused on patients with locally advanced",
        "focused on early-stage", "focused on advanced-stage",
    ],
    "best_subset_language": [
        "most informative", "most promising", "strongest association", "best-performing",
    ],
    "many_candidates": [
        "most informative of the", "candidate markers is presented",
    ],
    "endpoint_switch": [
        "among the available endpoints", "most favourable endpoint", "most favorable endpoint",
    ],
    "covariate_fishing": [
        "routinely available in the cohort", "which were routinely available",
        "models included age, stage, grade",
    ],
    "followup_custom": [
        "five-year survival was analysed", "five-year survival was analyzed",
        "five-year overall survival was analysed",
    ],
}


def _artifact_dir(cfg: Config, entry: dict) -> Path | None:
    sid = str(entry.get("submission_id", ""))
    m = re.match(r"rt_r(\d+)_w(\d+)", sid)
    if not m:
        return None
    path = cfg.paths.artifacts / f"round_{int(m.group(1)):02d}" / f"w{int(m.group(2)):02d}"
    return path if (path / "submission.json").is_file() else None


def _submission_reports(cfg: Config, entry: dict) -> list[tuple[Claim, str]]:
    path = _artifact_dir(cfg, entry)
    if path is None:
        return []
    data = read_json(path / "submission.json")
    out = []
    for report in data.get("reports", []):
        claim = Claim.model_validate(report["claim"])
        text = safe_join(path, report["manuscript_md_uri"]).read_text(encoding="utf-8")
        out.append((claim, text))
    return out


def _natural_data(cfg: Config):
    claims, split, contexts = load_natural_corpus(cfg)
    gt = {cid: GroundTruthResult.model_validate(v) for cid, v in
          read_json(cfg.paths.processed / "ground_truth.json").items()}
    return claims, split, contexts, gt


def _matrix(cfg: Config, claims: list[Claim], ids: list[str], learned: dict[str, list[str]]):
    extractor = EvidenceExtractor(cfg, cfg.lexicon, learned)
    X, y, w = [], [], []
    for cid in ids:
        claim = next(c for c in claims if c.claim_id == cid)
        text = (cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md").read_text(encoding="utf-8")
        f = extractor.extract(claim, text)
        X.append([f[n] for n in FEATURE_LABELS])
        yy, ww = labels_and_weights({cid: _natural_data_gt(cfg)[cid]}, cfg)
        y.append(float(yy[0])); w.append(float(ww[0]))
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(w), extractor

_GT_CACHE: dict[str, GroundTruthResult] = {}
def _natural_data_gt(cfg: Config) -> dict[str, GroundTruthResult]:
    if not _GT_CACHE:
        _GT_CACHE.update({cid: GroundTruthResult.model_validate(v) for cid, v in
                          read_json(cfg.paths.processed / "ground_truth.json").items()})
    return _GT_CACHE


def _mine_patterns(cfg: Config, current: dict[str, list[str]], blind_texts: list[str],
                   natural_texts: list[str], *, max_add: int = 4) -> dict[str, list[str]]:
    learned = {k: list(v) for k, v in current.items()}
    existing = set()
    for pats in learned.values():
        existing.update(pats)
    for pats in cfg.lexicon.get("patterns", {}).values():
        existing.update(pats)
    added = 0
    for group, phrases in CANDIDATE_PHRASES.items():
        for phrase in phrases:
            if added >= max_add:
                return learned
            if phrase in existing or re.escape(phrase) in existing:
                continue
            if not any(phrase.lower() in t.lower() for t in blind_texts):
                continue
            natural_hits = sum(1 for t in natural_texts if phrase.lower() in t.lower())
            if natural_hits > max(2, int(0.02 * len(natural_texts))):
                continue
            escaped = re.escape(phrase)
            learned.setdefault(group, []).append(escaped)
            existing.add(escaped)
            added += 1
    return learned


def _train_variant(cfg: Config, learned: dict[str, list[str]], version: str, out_dir: Path,
                   prev: Detector) -> tuple[Detector, dict[str, Any]]:
    claims, split, contexts, gt = _natural_data(cfg)
    train_ids = [c.claim_id for c in claims if split[c.claim_id] == "train"]
    val_ids = [c.claim_id for c in claims if split[c.claim_id] == "val"]
    test_ids = [c.claim_id for c in claims if split[c.claim_id] == "test"]
    Xtr, ytr, wtr, _ = _matrix(cfg, claims, train_ids, learned)
    Xval, yval, wval, _ = _matrix(cfg, claims, val_ids, learned)
    Xte, yte, wte, _ = _matrix(cfg, claims, test_ids, learned)
    evidence = EvidenceModel.fit(Xtr, ytr, sample_weight=wtr, version=version,
                                 learned_patterns=learned)
    val_scores = np.asarray([evidence.predict(dict(zip(FEATURE_LABELS, row))) for row in Xval])
    test_scores = np.asarray([evidence.predict(dict(zip(FEATURE_LABELS, row))) for row in Xte])
    evidence.thresholds = {f"fpr_{t:g}": threshold_at_fpr(yval, val_scores, t) for t in [0.01, 0.05]}
    fusion = type(prev.fusion).from_dict(prev.fusion.to_dict())
    # Keep context/fusion natural-label parameters; recalibrate the combined
    # threshold on natural validation with the new evidence scores.
    comb_val = []
    for cid, ev in zip(val_ids, val_scores):
        if cid in contexts:
            vec, cov = context_vector(contexts[cid])
            cp = prev.context_model.predict(vec, cov) if prev.context_model is not None else ev
            comb_val.append(fusion.combine(float(ev), float(cp), float(cov)))
        else:
            comb_val.append(float(ev))
    comb_val = np.asarray(comb_val)
    fusion.evidence_threshold = float(evidence.threshold(0.01))
    fusion.combined_threshold = threshold_at_fpr(yval, comb_val, 0.05)
    detector = Detector(cfg, evidence, prev.context_model, fusion, version=version)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector.save(out_dir)
    metrics = {
        "version": version, "train_n": len(train_ids), "val_n": len(val_ids), "test_n": len(test_ids),
        "evidence_val": _metrics(yval, val_scores), "evidence_test": _metrics(yte, test_scores),
        "combined_val": _metrics(yval, comb_val),
        "positive_rate": {"train": float(ytr.mean()), "val": float(yval.mean()), "test": float(yte.mean())},
        "learned_patterns": learned,
    }
    write_json(out_dir / "train_report.json", metrics)
    return detector, metrics


def _holdout_recall(detectors: list[Detector], entries: list[dict], families: set[str],
                    cfg: Config) -> dict[str, Any]:
    out_rates = []
    n_reports = 0
    for det in detectors:
        flags = []
        for entry in entries:
            if entry.get("strategy_family") not in families:
                continue
            for claim, text in _submission_reports(cfg, entry):
                flags.append(det.score(claim, text, neutral=True).flag)
        n_reports = max(n_reports, len(flags))
        out_rates.append(float(np.mean(flags)) if flags else None)
    return {"families": sorted(families), "n_reports": n_reports, "rates": out_rates}


def _context_checks(cfg: Config, detector: Detector, contexts: dict[str, ContextFeatures],
                    entries: list[dict]) -> dict[str, Any]:
    reports: list[tuple[Claim, str]] = []
    for entry in entries:
        reports.extend(_submission_reports(cfg, entry))
        if len(reports) >= 20:
            break
    reports = reports[:20]
    if not reports:
        return {"permutation": None, "attack": None}
    lib = list(contexts.values())
    return {
        "permutation": permutation_test(cfg, detector, reports, lib, samples=5),
        "attack": context_attack(cfg, detector, reports, lib),
    }


def run_blue_cycles(cfg: Config, *, cycles: int | None = None,
                    rounds_after_update: int = 1, sandbox: bool | None = None) -> dict[str, Any]:
    ensure_layout(cfg)
    cycles = int(cfg.blue.get("cycles", 3) if cycles is None else cycles)
    if cycles < 1:
        raise ValueError("cycles must be positive")
    rng = np.random.default_rng(int(cfg.blue.get("seed", cfg.mvp["seed"])) + 515)
    claims, split, contexts, gt = _natural_data(cfg)
    natural_texts = [(cfg.paths.processed / "corpus" / "manuscripts" / f"{c.claim_id}.md").read_text(encoding="utf-8")
                     for c in claims]
    prev_dir = active_detector_dir(cfg.paths.models)
    prev = Detector.load(cfg, prev_dir)
    if getattr(prev, "backend", "checklist") == "llm":
        from cbi.m7.blue_llm import run_llm_cycle
        return run_llm_cycle(cfg, cycles=cycles,
                             workers=int(cfg.redteam.get("workers", 8)),
                             rounds_after_update=rounds_after_update,
                             sandbox=bool(cfg.redteam.get("sandbox", True)) if sandbox is None else sandbox)
    cycle_reports = []
    for cycle in range(1, cycles + 1):
        log = read_jsonl(cfg.paths.experiment_log)
        if not log:
            raise RuntimeError("P7 requires red-team submissions in the experiment log (run P6)")
        # `honest_reference` is a control, not an attack family; it must not be
        # used for blind-spot mining or held-out recall evaluation.
        families = sorted({e.get("strategy_family") for e in log
                           if e.get("strategy_family") and e.get("strategy_family") != "honest_reference"})
        if not families:
            families = sorted({e.get("strategy_family") for e in log if e.get("strategy_family")})
        n_hold = max(1, int(round(float(cfg.blue.get("family_holdout_fraction", 0.2)) * len(families))))
        hold = set(rng.choice(families, size=min(n_hold, len(families)), replace=False).tolist())
        blind_all = [e for e in log
                     if e.get("rejected_by") is None and float(e.get("score", 0.0)) > 0.05
                     and e.get("strategy_family") != "honest_reference"]
        # If every observed blind spot would be held out there is nothing to
        # learn from; swap one holdout family for a family with no blind spot.
        blind_families = {e.get("strategy_family") for e in blind_all}
        if blind_all and blind_families and blind_families <= hold:
            candidates = [f for f in families if f not in hold]
            if candidates:
                hold.discard(sorted(hold)[0])
                hold.add(candidates[0])
        blind_entries = [e for e in blind_all if e.get("strategy_family") not in hold]
        blind_texts: list[str] = []
        for e in blind_entries:
            blind_texts.extend(text for _, text in _submission_reports(cfg, e))
        learned = _mine_patterns(cfg, prev.evidence_model.learned_patterns, blind_texts, natural_texts)
        version = f"det-1.{cycle + 1}.0"
        new_dir = cfg.paths.models / f"detector_v{cycle + 1}"
        new_det, metrics = _train_variant(cfg, learned, version, new_dir, prev)
        # Natural test FPR must not rise more than tolerance.
        claims_gt = gt
        test_ids = [c.claim_id for c in claims if split[c.claim_id] == "test"]
        extractor = EvidenceExtractor(cfg, cfg.lexicon, learned)
        ytest, scores_old, scores_new = [], [], []
        for cid in test_ids:
            claim = next(c for c in claims if c.claim_id == cid)
            text = (cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md").read_text(encoding="utf-8")
            f = extractor.extract(claim, text)
            ytest.append(1.0 if (claims_gt[cid].inflation_z >= 1 and claims_gt[cid].inflation >= 0.2) else 0.0)
            scores_old.append(prev.score(claim, text, neutral=True).evidence_score)
            scores_new.append(new_det.score(claim, text, neutral=True).evidence_score)
        ytest = np.asarray(ytest)
        # Compare both detectors at the same operating point.
        operating_threshold = new_det.fusion.evidence_threshold
        old_flag = np.asarray(scores_old) >= operating_threshold
        new_flag = np.asarray(scores_new) >= operating_threshold
        fpr_old = float(old_flag[ytest == 0].mean()) if (ytest == 0).any() else 0.0
        fpr_new = float(new_flag[ytest == 0].mean()) if (ytest == 0).any() else 0.0
        recall_old = float(old_flag[ytest == 1].mean()) if (ytest == 1).any() else 0.0
        recall_new = float(new_flag[ytest == 1].mean()) if (ytest == 1).any() else 0.0
        tolerance = float(cfg.blue.get("fpr_tolerance", 0.01))
        adjusted = False
        if fpr_new > fpr_old + tolerance:
            # Conservative repair on natural validation negatives, not test.
            val_ids = [c.claim_id for c in claims if split[c.claim_id] == "val"]
            yval, val_scores = [], []
            for cid in val_ids:
                claim = next(c for c in claims if c.claim_id == cid)
                text = (cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md").read_text(encoding="utf-8")
                yval.append(1.0 if (claims_gt[cid].inflation_z >= 1 and claims_gt[cid].inflation >= 0.2) else 0.0)
                val_scores.append(new_det.score(claim, text, neutral=True).evidence_score)
            new_thr = threshold_at_fpr(np.asarray(yval), np.asarray(val_scores), 0.02)
            new_det.fusion.evidence_threshold = max(new_det.fusion.evidence_threshold, new_thr)
            new_det.evidence_model.thresholds["fpr_0.05"] = new_det.fusion.evidence_threshold
            new_det.save(new_dir)
            new_flag = np.asarray(scores_new) >= new_det.fusion.evidence_threshold
            fpr_new = float(new_flag[ytest == 0].mean()) if (ytest == 0).any() else 0.0
            recall_new = float(new_flag[ytest == 1].mean()) if (ytest == 1).any() else 0.0
            adjusted = bool(fpr_new > fpr_old + tolerance)
        hold_eval = _holdout_recall([prev, new_det], log, hold, cfg)
        ctx_eval = _context_checks(cfg, new_det, contexts, blind_entries or log)
        # Run the next red-team round against the updated detector.
        next_round = max(int(e.get("round", 0)) for e in log) + 1
        red = run_round(cfg, round_no=next_round, detector_dir=new_dir,
                        workers=int(cfg.redteam.get("workers", 8)),
                        reports_per_batch=int(cfg.redteam.get("reports_per_batch", 20)),
                        sandbox=(bool(cfg.redteam.get("sandbox", True)) if sandbox is None else sandbox),
                        log_entries=log)
        cycle_report = {
            "cycle": cycle, "version": version, "holdout_families": sorted(hold),
            "blind_spot_entries": len(blind_entries), "patterns_added": {
                k: v for k, v in learned.items() if v != prev.evidence_model.learned_patterns.get(k, [])
            },
            "natural_test": {"fpr_before": fpr_old, "fpr_after": fpr_new,
                             "recall_before": recall_old, "recall_after": recall_new,
                             "threshold_adjusted": adjusted},
            "holdout_recall": hold_eval,
            "context": ctx_eval,
            "red_round": {"round": next_round, "graded": len(red),
                          "best": max((r.score for r in red), default=0.0),
                          "mean": float(np.mean([r.score for r in red])) if red else 0.0},
            "metrics": {k: v for k, v in metrics.items() if k != "learned_patterns"},
        }
        cycle_reports.append(cycle_report)
        prev = new_det
        prev_dir = new_dir
    report = {"cycles": cycle_reports, "n_cycles": len(cycle_reports)}
    write_json(cfg.paths.reports / "P7_metrics.json", report)
    lines = ["# P7 — blue-team cycles", ""]
    for c in cycle_reports:
        lines += [
            f"## Cycle {c['cycle']} ({c['version']})",
            f"- Holdout families: {', '.join(c['holdout_families'])}",
            f"- Blind-spot batches: {c['blind_spot_entries']}",
            f"- Natural test FPR: {c['natural_test']['fpr_before']:.3f} -> {c['natural_test']['fpr_after']:.3f} "
            f"(tolerance 0.01); recall {c['natural_test']['recall_before']:.3f} -> {c['natural_test']['recall_after']:.3f}",
            f"- Holdout-family recall: {c['holdout_recall']['rates']}",
            f"- Context permutation: {None if c['context']['permutation'] is None else c['context']['permutation']}",
            f"- Context attack: {None if c['context']['attack'] is None else c['context']['attack']}",
            f"- Red-team next round: {c['red_round']}",
            "",
        ]
    (cfg.paths.reports / "P7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
