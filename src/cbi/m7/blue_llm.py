"""P7 blue-team cycle for the LLM content monitor.

The update is a checklist/few-shot update, not weight fitting.  Context prior
and fusion are carried over from natural labels.
"""
from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config
from cbi.m4.detector import Detector, active_detector_dir, detector_fingerprint
from cbi.m4.llm_judge import llm_checklist_score_many
from cbi.m4.model import fpr_recall, threshold_at_fpr
from cbi.m4.cache import prepare_documents
from cbi.m4.train import labels_and_weights
from cbi.m6.context_sampler import permutation_test
from cbi.m5.orchestrator import run_round
from cbi.schemas import GroundTruthResult
from cbi.util import new_id, read_json, read_jsonl, write_json


def _score_many(det: Detector, items: list[tuple[str, str, str]], workers: int):
    if det.backend != "llm" or len(items) < 2:
        def one(item):
            claim, text, _ = item
            return det.score(claim, text, neutral=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, items))

    batch = max(1, int(os.environ.get("CBI_EVIDENCE_BATCH_SIZE", "4")))
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]

    def score_chunk(chunk):
        try:
            judgements = llm_checklist_score_many(
                [(claim, text) for claim, text, _ in chunk],
                learned_examples=det.learned_examples)
        except Exception:
            judgements = [None] * len(chunk)
        out = []
        for (claim, text, _), judgement in zip(chunk, judgements):
            if judgement is None:
                out.append(det.score(claim, text, neutral=True))
                continue
            score = det.judgement_score(judgement)
            out.append(SimpleNamespace(
                evidence_score=score,
                context_prior=None,
                context_coverage=0.0,
                combined_inflation_pred=score,
                flag=bool(score >= det.fusion.evidence_threshold),
                threshold_used=float(det.fusion.evidence_threshold),
                rationale=list(judgement.get("rationale") or []),
                features={},
            ))
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as pool:
        nested = list(pool.map(score_chunk, chunks))
    return [row for group in nested for row in group]


def _rates(labels, outputs, threshold):
    scores = np.asarray([o.evidence_score for o in outputs])
    fpr, recall = fpr_recall(labels, scores, threshold)
    return {"fpr": fpr, "recall": recall, "threshold": float(threshold)}


def _write_progress(cfg, run_dir, report):
    write_json(run_dir / "metrics.json", report)
    write_json(cfg.paths.reports / "P7_metrics.json", report)
    lines = ["# P7 — LLM blue-team cycles", "", f"- Run: {report['run_id']}",
             f"- Completed/requested cycles: {report['n_cycles']}/{report['requested_cycles']}",
             f"- Active detector: {report['active_detector_directory']}",
             "- Family holdout is fixed for this run and excluded from all prompt updates.",
             "- Thresholds are fitted on validation only. The spec's test-FPR acceptance gate",
             "  is reused across cycles; these test results are adaptive diagnostics, not an untouched final test.", ""]
    for row in report["cycles"]:
        test = row["natural_test"]
        lines.extend([f"## Cycle {row['cycle']} ({row['version']})", "",
                      f"- Accepted: {row['accepted']}; patterns added: {len(row['patterns_added'])}",
                      f"- Test FPR: {test['fpr_before']:.3f} -> {test['fpr_after']:.3f}; "
                      f"recall: {test['recall_before']:.3f} -> {test['recall_after']:.3f}",
                      f"- Holdout recall: {row['holdout_recall']}",
                      f"- Red round: {row['red_round']}", ""])
    (cfg.paths.reports / "P7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_llm_cycle(cfg: Config, *, cycles: int = 3, workers: int = 8,
                  rounds_after_update: int = 1, sandbox: bool = True) -> dict[str, Any]:
    from cbi.m7.blue import CANDIDATE_PHRASES, _natural_data, _submission_reports

    if cycles < 1 or rounds_after_update < 1 or workers < 1:
        raise ValueError("cycles, rounds_after_update and workers must be positive")
    # The v1 monitor is an invalid baseline; never spend on blue updates to it.
    baseline = cfg.paths.models / "detector_llm_v2"
    if not (baseline / "meta.json").is_file():
        raise RuntimeError("P7 requires a completed P4 v2 detector and matching P6 submissions")
    prev_dir = active_detector_dir(cfg.paths.models)
    prev = Detector.load(cfg, prev_dir)
    log = read_jsonl(cfg.paths.experiment_log)
    current_log = [e for e in log if e.get("detector_version") == prev.version]
    families = sorted({e["strategy_family"] for e in current_log
                       if e.get("strategy_family") not in {None, "unknown", "honest_reference"}})
    if prev_dir != baseline:
        # A resumed lineage must keep the original untouched families; drawing
        # a fresh holdout could include families already learned in a prior run.
        lineage = read_json(cfg.paths.models / "blue_llm" / "active.json")
        hold = set(lineage["holdout_families"])
        hold_entries = lineage["holdout_entries"]
        if not set(families) - hold:
            raise RuntimeError("P7 needs matching P6 submissions from a non-held-out attack family")
    else:
        if len(families) < 2:
            raise RuntimeError("P7 requires matching P6 submissions from at least two attack families")
        rng = np.random.default_rng(int(cfg.blue.get("seed", cfg.mvp["seed"])))
        n_hold = min(len(families) - 1, max(1, round(float(cfg.blue.get("family_holdout_fraction", .2)) * len(families))))
        hold = set(rng.choice(families, size=n_hold, replace=False).tolist())
        hold_entries = [e for e in current_log if e.get("strategy_family") in hold
                        and e.get("rejected_by") is None]
    # Keep this holdout fixed across cycles; rotating it would leak previous
    # update examples into later 'held-out' evaluations.
    report_limit = int(cfg.blue.get("evaluation_report_limit", 10))
    hold_reports = []
    for entry in hold_entries:
        hold_reports.extend(_submission_reports(cfg, entry))
    hold_reports = hold_reports[:report_limit]
    claims, split, contexts, gt = _natural_data(cfg)
    eligible = {cid for cid, r in gt.items() if r.label_eligible and r.independent_cohorts}
    selected = [c for c in claims if c.claim_id in eligible and split[c.claim_id] in {"val", "test"}]
    texts, _, _ = prepare_documents(cfg, selected)
    items = {role: [(c, texts[c.claim_id], c.claim_id) for c in selected if split[c.claim_id] == role]
             for role in ("val", "test")}
    labels, _ = labels_and_weights(gt, cfg)
    label_map = dict(zip(gt, labels))
    ys = {role: np.asarray([label_map[cid] for _, _, cid in group]) for role, group in items.items()}
    if any(len(np.unique(y)) != 2 for y in ys.values()):
        raise RuntimeError("P7 requires eligible positive and negative labels in validation and test")
    score_workers = max(1, int(os.environ.get("CBI_P7_SCORE_WORKERS", str(workers))))
    run_id = new_id("p7")
    run_dir = cfg.paths.models / "blue_llm" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "requested_cycles": cycles, "n_cycles": 0, "cycles": [],
              "base_detector_directory": str(baseline), "base_fingerprint": detector_fingerprint(baseline),
              "holdout_families": sorted(hold), "active_detector_directory": str(prev_dir),
              "status": "running", "evaluation": "adaptive_test_fpr_gate_per_spec"}
    _write_progress(cfg, run_dir, report)
    try:
        for cycle in range(1, cycles + 1):
            log = read_jsonl(cfg.paths.experiment_log)
            blind = [e for e in log if e.get("detector_version") == prev.version
                     and e.get("rejected_by") is None and float(e.get("score", 0)) > .05
                     and e.get("strategy_family") not in hold | {None, "unknown", "honest_reference"}]
            training_reports = []
            for entry in blind[:report_limit]:
                training_reports.extend(_submission_reports(cfg, entry))
            found = []
            for group, phrases in CANDIDATE_PHRASES.items():
                for phrase in phrases:
                    example = f"{group}: {phrase}"
                    if len(found) < 10 and example not in prev.learned_examples and any(
                            phrase.lower() in text.lower() for _, text in training_reports):
                        found.append(example)
            version = f"det-llm-blue-{run_id}-{cycle}"
            new_dir = run_dir / f"cycle_{cycle:03d}"
            new_det = Detector(cfg, None, prev.context_model, copy.deepcopy(prev.fusion),
                               version=version, backend="llm", evidence_variant=prev.evidence_variant,
                               learned_examples=sorted(set(prev.learned_examples) | set(found)))
            val_out = _score_many(new_det, items["val"], score_workers)
            val_scores = np.asarray([o.evidence_score for o in val_out])
            primary = float(cfg.scoring["detector"].get("primary_fpr", .05))
            new_det.fusion.evidence_threshold = threshold_at_fpr(ys["val"], val_scores, primary)
            new_det.fusion.evidence_threshold_1pct = threshold_at_fpr(ys["val"], val_scores, .01)
            # Context weights/model remain fixed. Recalibrate the combined
            # threshold only on natural validation after the evidence update.
            from cbi.m4.train_llm import combine_with_context
            if new_det.context_model is not None:
                _, combined = combine_with_context([cid for _, _, cid in items["val"]],
                                                    val_scores.tolist(), contexts,
                                                    new_det.context_model, new_det.fusion)
            else:
                combined = val_scores
            new_det.fusion.combined_threshold = threshold_at_fpr(ys["val"], combined, primary)
            before = _rates(ys["test"], _score_many(prev, items["test"], score_workers), prev.fusion.evidence_threshold)
            after = _rates(ys["test"], _score_many(new_det, items["test"], score_workers), new_det.fusion.evidence_threshold)
            hold_items = [(c, text, c.claim_id) for c, text in hold_reports]
            hold_rates = []
            for detector in (prev, new_det):
                outputs = _score_many(detector, hold_items, score_workers) if hold_items else []
                hold_rates.append(float(np.mean([o.evidence_score >= detector.fusion.evidence_threshold
                                                 for o in outputs])) if outputs else None)
            context_eval = permutation_test(cfg, new_det, (hold_reports or training_reports)[:2],
                                            list(contexts.values()), samples=2,
                                            seed=int(cfg.blue.get("seed", 0)) + cycle)
            accepted = after["fpr"] <= before["fpr"] + float(cfg.blue.get("fpr_tolerance", .01))
            new_det.save(new_dir)  # Retain rejected candidates without promoting them.
            row = {"cycle": cycle, "version": version, "parent_version": prev.version,
                   "detector_directory": str(new_dir), "holdout_families": sorted(hold),
                   "training_families": sorted({e["strategy_family"] for e in blind}),
                   "blind_spot_entries": len(blind), "patterns_added": found, "accepted": accepted,
                   "natural_test": {"fpr_before": before["fpr"], "fpr_after": after["fpr"],
                                    "recall_before": before["recall"], "recall_after": after["recall"],
                                    "threshold_before": before["threshold"], "threshold_after": after["threshold"]},
                   "holdout_recall": {"n_reports": len(hold_reports), "rates": hold_rates,
                                      "gain": hold_rates[1] - hold_rates[0] if hold_reports else None},
                   "context": {"permutation": context_eval, "attack": None},
                   "red_round": {"status": "pending", "rounds": []}}
            report["cycles"].append(row)
            if accepted:
                prev, prev_dir = new_det, new_dir
                write_json(cfg.paths.models / "blue_llm" / "active.json", {
                    "directory": str(new_dir.relative_to(cfg.paths.models)),
                    "base_fingerprint": report["base_fingerprint"],
                    "holdout_families": sorted(hold), "holdout_entries": hold_entries,
                    "detector_fingerprint": detector_fingerprint(new_dir)})
            report["active_detector_directory"] = str(prev_dir)
            _write_progress(cfg, run_dir, report)
            # Rejections also alternate with red work against the retained model.
            for _ in range(rounds_after_update):
                log = read_jsonl(cfg.paths.experiment_log)
                next_round = max([int(e.get("round", 0)) for e in log]
                                 + [int(p.name[6:]) for p in cfg.paths.artifacts.glob("round_*")
                                    if p.name[6:].isdigit()] + [0]) + 1
                red = run_round(cfg, round_no=next_round, detector_dir=prev_dir, workers=workers,
                                reports_per_batch=int(cfg.blue.get("reports_per_batch", 1)),
                                sandbox=sandbox, log_entries=log)
                row["red_round"]["rounds"].append({"round": next_round, "graded": len(red),
                                                    "best": max((r.score for r in red), default=0.0)})
                _write_progress(cfg, run_dir, report)
                if not red:
                    raise RuntimeError(f"P7 red round {next_round} produced no graded submissions; cycle is incomplete")
            row["red_round"]["status"] = "complete"
            report["n_cycles"] += 1
            _write_progress(cfg, run_dir, report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = type(exc).__name__
        _write_progress(cfg, run_dir, report)
        raise
    _write_progress(cfg, run_dir, report)
    return report
