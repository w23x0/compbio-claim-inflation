"""Recompute P7 §7.3 context metrics with the final detector threshold code.

Red-team rounds are expensive; this helper refreshes only the grader-internal
context permutation/attack blocks after a detector-threshold fix, using the
already-saved cycle detectors and log.  It never changes red-team scores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cbi.config import Config
from cbi.m4.detector import Detector
from cbi.m7.blue import _context_checks
from cbi.schemas import ContextFeatures
from cbi.util import read_json, read_jsonl, write_json


def recompute_context_metrics(cfg: Config) -> dict[str, Any]:
    metrics_path = cfg.paths.reports / "P7_metrics.json"
    metrics = read_json(metrics_path)
    contexts_obj = read_json(cfg.paths.processed / "context_features.json")
    contexts = {cid: ContextFeatures.model_validate(v) for cid, v in contexts_obj.items()}
    log = read_jsonl(cfg.paths.experiment_log)
    for cycle in metrics["cycles"]:
        detector = Detector.load(cfg, cfg.paths.models / f"detector_v{cycle['cycle'] + 1}")
        before_round = int(cycle["red_round"]["round"]) - 1
        entries = [
            e for e in log
            if int(e.get("round", 0)) <= before_round
            and e.get("strategy_family") != "honest_reference"
            and e.get("rejected_by") is None
            and float(e.get("score", 0.0)) > 0.05
        ]
        if not entries:
            entries = [e for e in log if int(e.get("round", 0)) <= before_round]
        cycle["context"] = _context_checks(cfg, detector, contexts, entries)
        cycle["natural_test"]["threshold_adjusted"] = False
    write_json(metrics_path, metrics)
    _write_summary(cfg, metrics)
    return metrics


def _write_summary(cfg: Config, metrics: dict[str, Any]) -> None:
    lines = ["# P7 — blue-team cycles", ""]
    for c in metrics["cycles"]:
        lines += [
            f"## Cycle {c['cycle']} ({c['version']})",
            f"- Holdout families: {', '.join(c['holdout_families'])}",
            f"- Blind-spot batches: {c['blind_spot_entries']}",
            f"- Natural test FPR: {c['natural_test']['fpr_before']:.3f} -> {c['natural_test']['fpr_after']:.3f} "
            f"(tolerance 0.01); recall {c['natural_test']['recall_before']:.3f} -> {c['natural_test']['recall_after']:.3f}",
            f"- Holdout-family recall: {c['holdout_recall']['rates']}",
            f"- Context permutation: {c['context']['permutation']}",
            f"- Context attack: {c['context']['attack']}",
            f"- Red-team next round: {c['red_round']}",
            "",
        ]
    cycles = metrics.get("cycles", [])
    if cycles:
        first, last = cycles[0], cycles[-1]
        lines += [
            "Note: this summary is computed from the cycle metrics above. "
            f"Natural-test FPR {first['natural_test']['fpr_before']:.3f} -> "
            f"{last['natural_test']['fpr_after']:.3f}; "
            f"holdout recall {first['holdout_recall']['rates']} -> "
            f"{last['holdout_recall']['rates']}.",
        ]
    (cfg.paths.reports / "P7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
