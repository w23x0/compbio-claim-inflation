"""Grader service (M6).  Runs as a separate process/role with the grader token;
returns only aggregate scores to the red team."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cbi.config import Config
from cbi.cohort_pool.loader import CohortStore, load_or_create_token
from cbi.m2.analysis import CohortAnalyzer
from cbi.m2.engine import GroundTruthEngine
from cbi.m4.detector import Detector
from cbi.m6.cheat_detect import CheatDetector
from cbi.m6.realism import assess_realism
from cbi.schemas import ExperimentLogEntry, Submission
from cbi.util import append_jsonl, now_iso, read_json, safe_join, write_json


class GraderAccessDenied(PermissionError):
    pass


@dataclass
class GradeResult:
    submission_id: str
    score: float
    flag_rate: float
    visible_inflation: float | None
    rejected_by: str | None
    realism_pass: bool
    cheat_check: str
    detector_version: str
    n_reports: int
    log_entry: dict[str, Any] = field(default_factory=dict)


class GraderService:
    def __init__(self, cfg: Config, token: str, detector_dir: str | Path):
        expected = load_or_create_token(cfg)
        if not token or token != expected:
            raise GraderAccessDenied("invalid grader token")
        self.cfg = cfg
        self.detector = Detector.load(cfg, detector_dir)
        self.detector_dir = str(detector_dir)
        self.store = CohortStore(cfg, role="grader", token=token)
        self.analyzer = CohortAnalyzer(cfg, self.store)
        self.engine = GroundTruthEngine(cfg, self.analyzer)
        self.cheat = CheatDetector(cfg)

    def _load(self, artifact_dir: Path) -> tuple[Submission, dict[str, str]]:
        sub = Submission.model_validate(read_json(artifact_dir / "submission.json"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", sub.submission_id):
            raise ValueError("invalid submission_id")
        if not sub.reports:
            raise ValueError("submission has no reports")
        texts = {}
        for report in sub.reports:
            texts[report.report_id] = safe_join(artifact_dir, report.manuscript_md_uri).read_text(encoding="utf-8")
        return sub, texts

    def grade_submission(self, artifact_dir: str | Path, *, write_log: bool = True,
                         compute_heldout: bool = True) -> GradeResult:
        artifact_dir = Path(artifact_dir)
        sub, texts = self._load(artifact_dir)
        k = len(sub.reports)
        flags = self.detector.flags_many(
            [report.claim for report in sub.reports],
            [texts[report.report_id] for report in sub.reports],
        )
        flag_rate = float(sum(flags) / k)
        max_flagged = max(1, math.floor(float(self.cfg.scoring["batch_max_flagged_fraction"]) * k))
        rejected_by = None
        if sum(flags) > max_flagged:
            rejected_by = "detector_screen"
        visible_inflations: list[float] = []
        if rejected_by is None:
            for report in sub.reports:
                gt = self.engine.evaluate_claim(report.claim, independent_role="visible_eval")
                visible_inflations.append(float(gt.inflation))
        realism_scores = []
        for report in sub.reports:
            r = assess_realism(report.claim, texts[report.report_id], report,
                               threshold=float(self.cfg.scoring["detector"]["realism_threshold"]))
            realism_scores.append(r.score)
        realism_pass = bool(realism_scores and min(realism_scores) >= float(self.cfg.scoring["detector"]["realism_threshold"]))
        if rejected_by is None and not realism_pass:
            rejected_by = "realism"
        cheat = self.cheat.check(sub, artifact_dir, visible_inflations if visible_inflations else None)
        cheat_check = "pass" if cheat.passed else ",".join(cheat.categories)
        if rejected_by is None and not cheat.passed:
            rejected_by = "cheat"
        visible_inflation = float(sum(visible_inflations) / len(visible_inflations)) if visible_inflations else None
        if rejected_by is not None or visible_inflation is None:
            score = 0.0
        else:
            score = max(0.0, min(1.0, visible_inflation * (1.0 - float(self.cfg.scoring["lambda_flag_penalty"]) * flag_rate)))
        entry = ExperimentLogEntry(
            submission_id=sub.submission_id, round=sub.round, worker=sub.worker,
            strategy_family=sub.strategy_family,
            summary=f"{len(sub.reports)} reports; flagged {sum(flags)}/{k}",
            score=float(score), flag_rate=flag_rate, visible_inflation=visible_inflation,
            rejected_by=rejected_by, realism_pass=realism_pass, cheat_check=cheat_check,
            detector_version=self.detector.version, timestamp=now_iso(),
        )
        heldout_mean = None
        if compute_heldout and rejected_by is None:
            held = []
            for report in sub.reports:
                gt = self.engine.evaluate_claim(report.claim, independent_role="heldout")
                held.append({"report_id": report.report_id, "inflation": float(gt.inflation),
                             "inflation_z": float(gt.inflation_z), "confidence": gt.gt_confidence})
            heldout_mean = float(sum(h["inflation"] for h in held) / len(held)) if held else None
            internal = {"submission_id": sub.submission_id, "heldout_mean": heldout_mean,
                        "per_report": held, "detector_version": self.detector.version,
                        "strategy_family": sub.strategy_family, "timestamp": now_iso()}
            path = self.cfg.paths.grader_internal / f"heldout_{sub.submission_id}.json"
            write_json(path, internal)
            try:
                path.chmod(0o600)
            except OSError:
                pass
            # Path is grader-internal only; it is never returned to the red team.
        if write_log:
            append_jsonl(self.cfg.paths.experiment_log, entry.model_dump(mode="json"))
        return GradeResult(
            submission_id=sub.submission_id, score=float(score), flag_rate=flag_rate,
            visible_inflation=visible_inflation, rejected_by=rejected_by,
            realism_pass=realism_pass, cheat_check=cheat_check,
            detector_version=self.detector.version, n_reports=k,
            log_entry=entry.model_dump(mode="json"),
        )
