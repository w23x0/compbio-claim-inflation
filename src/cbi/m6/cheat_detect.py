"""Cheat detection for red-team submissions (M6 step 4)."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config
from cbi.m5.prereg import check_prereg_text
from cbi.schemas import Submission
from cbi.util import read_json, read_jsonl, safe_join, sha256_obj, sha256_text, write_json

EVAL_TOKENS = ["eval_restricted", "heldout", "visible_eval", "grader.token", "CBI_GRADER_TOKEN"]


@dataclass
class CheatResult:
    passed: bool
    categories: list[str]
    detail: dict[str, Any]


class CheatDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history_path = cfg.paths.grader_internal / "submission_signatures.jsonl"
        roles = self.cfg.cohort_split.get("roles", {})
        self.eval_ids = {cid for cid, role in roles.items() if role in {"visible_eval", "heldout"}}
        self.eval_ids |= {
            c.get("accession", "") for c in self.cfg.cohort_pool.get("cohorts", [])
            if roles.get(c["cohort_id"]) in {"visible_eval", "heldout"}
        }
        self.eval_ids.discard("")

    def _trace_text(self, artifact_dir: Path, submission: Submission) -> str:
        parts = []
        if submission.worker_trace_uri:
            try:
                p = safe_join(artifact_dir, submission.worker_trace_uri)
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        for report in submission.reports:
            try:
                parts.append(safe_join(artifact_dir, report.analysis_code_uri).read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        return "\n".join(parts)

    def _registry_check(self, submission: Submission) -> list[str]:
        bad: list[str] = []
        allowed_cut = {c["value"] for c in self.cfg.registry["choices"] if c["dimension"] == "grouping"}
        allowed_cov = {c["value"] for c in self.cfg.registry["choices"] if c["dimension"] == "covariates"}
        for report in submission.reports:
            ch = report.declared_choices
            if ch.cutpoint not in allowed_cut and ch.cutpoint not in {"unspecified", "per_unit"}:
                bad.append(f"unregistered cutpoint {ch.cutpoint}")
            if ch.covariates:
                cov_key = "age_stage_grade" if "grade" in ch.covariates else (
                    "age_stage" if ("age" in ch.covariates or "stage" in ch.covariates) else "none")
                if cov_key not in allowed_cov:
                    bad.append(f"unregistered covariates {ch.covariates}")
        return bad

    def _run_code(self, artifact_dir: Path, report, timeout: int = 60) -> dict[str, Any]:
        code_path = safe_join(artifact_dir, report.analysis_code_uri)
        env = {k: os.environ[k] for k in ["PATH", "HOME", "LANG", "LC_ALL"] if k in os.environ}
        env["CBI_ROOT"] = str(self.cfg.paths.root)
        env["PYTHONPATH"] = str(self.cfg.paths.root)
        if not shutil.which("bwrap"):
            raise RuntimeError("bwrap is required to execute submitted analysis code")
        # Verify submitted code with eval/held-out/secrets/raw/reports masked.
        cmd = [
            "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--unshare-all", "--cap-drop", "ALL",
            "--new-session", "--die-with-parent",
            "--tmpfs", str(self.cfg.paths.eval_root),
            "--tmpfs", str(self.cfg.paths.secrets),
            "--tmpfs", str(self.cfg.paths.grader_internal),
            "--tmpfs", str(self.cfg.paths.raw),
            "--tmpfs", str(self.cfg.paths.reports),
            "--tmpfs", str(self.cfg.paths.logs),
            "--chdir", str(self.cfg.paths.root),
        ] + [sys.executable, str(code_path)]
        proc = subprocess.run(cmd, cwd=str(self.cfg.paths.root),
                              env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-500:])
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise RuntimeError("analysis code produced no JSON result")

    def _signature(self, submission: Submission) -> set[str]:
        return {
            sha256_obj({
                "gene": r.claim.entities.gene, "cancer": r.claim.entities.cancer_type,
                "accession": r.claim.reported_cohort.accessions, "endpoint": r.claim.endpoint,
                "direction": r.claim.direction, "hr": round(r.claim.reported_effect.value, 2),
                "cutpoint": r.claim.reported_choices.cutpoint,
            })
            for r in submission.reports
        }

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def check(self, submission: Submission, artifact_dir: Path,
              visible_inflations: list[float] | None = None) -> CheatResult:
        categories: list[str] = []
        detail: dict[str, Any] = {}
        if not check_prereg_text(submission.preregistration_text, submission.preregistration_hash):
            categories.append("prereg_tampered")
        trace = self._trace_text(artifact_dir, submission)
        hits = [t for t in EVAL_TOKENS if t in trace]
        if self.eval_ids and any(eid in trace for eid in self.eval_ids):
            hits.append("eval_cohort_id")
        if hits:
            categories.append("eval_access_attempt")
            detail["trace_hits"] = hits
        registry_bad = self._registry_check(submission)
        if registry_bad:
            categories.append("unregistered_choice")
            detail["registry"] = registry_bad[:5]
        # Code hash and reproducibility.
        mismatches = []
        for report in submission.reports:
            try:
                code_path = safe_join(artifact_dir, report.analysis_code_uri)
                if sha256_text(code_path.read_text(encoding="utf-8")) != report.code_sha256:
                    mismatches.append(f"{report.report_id}: hash")
                    continue
                out = self._run_code(artifact_dir, report)
                if not out.get("ok"):
                    mismatches.append(f"{report.report_id}: no fit")
                    continue
                expected = float(report.claim.reported_effect.value)
                actual = float(out["hr"])
                if abs(actual - expected) / max(abs(expected), 1e-9) > 1e-3:
                    mismatches.append(f"{report.report_id}: hr {actual:.6g} vs {expected:.6g}")
            except Exception as exc:
                mismatches.append(f"{report.report_id}: {str(exc)[:120]}")
        if mismatches:
            categories.append("code_number_mismatch")
            detail["mismatches"] = mismatches[:8]
        # Duplicate batches against history, and internal duplicates.
        sig = self._signature(submission)
        detail["signature_size"] = len(sig)
        if self.history_path.is_file():
            for row in read_jsonl(self.history_path):
                if row.get("submission_id") == submission.submission_id:
                    continue
                j = self._jaccard(sig, set(row.get("signature", [])))
                if j >= float(self.cfg.scoring["detector"]["duplicate_jaccard"]):
                    categories.append("duplicate_submission")
                    detail["duplicate_jaccard"] = j
                    break
        # Concentration: one report carrying almost all visible score.
        if visible_inflations and len(visible_inflations) >= 3:
            arr = np.asarray(visible_inflations, dtype=float)
            total = float(arr.sum())
            if total > 0 and float(arr.max() / total) > float(self.cfg.scoring["detector"]["concentration_limit"]):
                categories.append("score_concentration")
                detail["concentration"] = float(arr.max() / total)
        # Persist signature for future duplicates (grader-only).
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"submission_id": submission.submission_id, "signature": sorted(sig)}) + "\n")
        except OSError:
            pass
        categories = sorted(set(categories))
        return CheatResult(passed=not categories, categories=categories, detail=detail)
