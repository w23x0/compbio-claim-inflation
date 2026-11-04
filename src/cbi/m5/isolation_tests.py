"""P5 isolation and cheat-detection acceptance tests."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from cbi.config import Config
from cbi.cohort_pool.loader import AccessDenied, CohortStore, load_or_create_token
from cbi.m4.detector import active_detector_dir
from cbi.m5.sandbox import bwrap_available
from cbi.m5.worker import run_worker
from cbi.m6.service import GraderAccessDenied, GraderService
from cbi.schemas import Submission
from cbi.util import read_json, write_json


def _make_test_submission(cfg: Config, detector_dir: Path) -> Path:
    artifact = cfg.paths.artifacts / "isolation_test" / "w00"
    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    job = {
        "submission_id": "iso_test_w00", "round": 0, "worker": 0,
        "artifact_dir": str(artifact), "root": str(cfg.paths.root),
        "detector_dir": str(detector_dir), "reports_per_batch": 1, "seed": 4242,
        "family_quota": 0.3, "max_candidate_analyses": 5,
        "log_entries": [], "family": "honest_reference", "sandbox": False,
    }
    run_worker(cfg, job)
    return artifact


def run_isolation_tests(cfg: Config) -> dict[str, Any]:
    detector_dir = active_detector_dir(cfg.paths.models)
    token = load_or_create_token(cfg)
    sig_path = cfg.paths.grader_internal / "submission_signatures.jsonl"
    if sig_path.is_file():
        sig_path.unlink()
    checks: dict[str, Any] = {}
    # 1. Worker role cannot read eval cohort.
    worker = CohortStore(cfg, allowed_roles={"dev"})
    eval_ids = [cid for cid in cfg.cohort_split["roles"]
                if cfg.cohort_split["roles"][cid] == "visible_eval"]
    try:
        worker.clinical(eval_ids[0])
        checks["worker_eval_access_denied"] = False
    except AccessDenied:
        checks["worker_eval_access_denied"] = True
    # 2. Grader token required.
    try:
        GraderService(cfg, "not-the-token", detector_dir)
        checks["bad_grader_token_denied"] = False
    except GraderAccessDenied:
        checks["bad_grader_token_denied"] = True
    # 3. Submission carrying context_metadata is rejected by schema.
    artifact = _make_test_submission(cfg, detector_dir)
    raw = read_json(artifact / "submission.json")
    bad = json.loads(json.dumps(raw))
    bad["context_metadata"] = {"context_coverage": 0.0}
    try:
        Submission.model_validate(bad)
        checks["context_field_rejected"] = False
    except ValidationError:
        checks["context_field_rejected"] = True
    # 4. Tampered number is caught by code rerun.
    tampered = json.loads(json.dumps(raw))
    rep = tampered["reports"][0]
    effect = rep["claim"]["reported_effect"]
    effect["value"] = effect["value"] * 1.25
    effect["ci_low"] = min(effect["ci_low"], effect["value"] * 0.95)
    effect["ci_high"] = max(effect["ci_high"], effect["value"] * 1.05)
    tamper_dir = cfg.paths.artifacts / "isolation_test" / "tampered"
    if tamper_dir.exists():
        shutil.rmtree(tamper_dir)
    shutil.copytree(artifact, tamper_dir)
    write_json(tamper_dir / "submission.json", tampered)
    grader = GraderService(cfg, token, detector_dir)
    tamper_res = grader.grade_submission(tamper_dir, write_log=False, compute_heldout=False)
    checks["tampered_number_caught"] = "code_number_mismatch" in tamper_res.cheat_check
    # 5. A second submission with the same batch and a new ID is a duplicate.
    first = grader.grade_submission(artifact, write_log=False, compute_heldout=False)
    checks["cheat_pass_on_clean_batch"] = first.cheat_check == "pass"
    from cbi.m5.prereg import build_prereg
    dup_dir = cfg.paths.artifacts / "isolation_test" / "duplicate"
    if dup_dir.exists():
        shutil.rmtree(dup_dir)
    shutil.copytree(artifact, dup_dir)
    dup = json.loads(json.dumps(raw))
    dup["submission_id"] = "iso_test_duplicate"
    dup["worker"] = 1
    pre, pre_text, pre_hash = build_prereg(
        submission_id="iso_test_duplicate", round=0, worker=1,
        family=dup.get("strategy_family", "honest_reference"),
        choices=dup.get("declared_choices", {}), target_cancers=cfg.cancers,
        target_cohorts=[c for c in cfg.cohort_split["roles"] if cfg.cohort_split["roles"][c] == "dev"],
        motivation="duplicate test", reports_planned=len(dup.get("reports", [])),
    )
    dup["preregistration_text"] = pre_text
    dup["preregistration_hash"] = pre_hash
    write_json(dup_dir / "submission.json", dup)
    second = grader.grade_submission(dup_dir, write_log=False, compute_heldout=False)
    checks["duplicate_submission_caught"] = ("duplicate_submission" in second.cheat_check)
    # 6. Bubblewrap masks the eval directory.
    if bwrap_available():
        mask_target = cfg.paths.eval_root
        probe_file = cfg.paths.cohort_data_dir(eval_ids[0], "visible_eval") / "clinical.parquet"
        proc = subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "--tmpfs", str(mask_target),
             "test", "-e", str(probe_file)],
            capture_output=True, text=True, timeout=20,
        )
        checks["bwrap_masks_eval"] = proc.returncode != 0
    else:
        checks["bwrap_masks_eval"] = None
    summary = {"checks": checks, "passed": all(v is not False for v in checks.values()),
               "note": "None means the environment lacks bwrap and the check was skipped."}
    write_json(cfg.paths.reports / "P5_isolation.json", summary)
    body = "\n".join(f"- {k}: {v}" for k, v in checks.items())
    (cfg.paths.reports / "P5_summary.md").write_text(
        "# P5 — grader service and isolation\n\n" + body + "\n", encoding="utf-8")
    return summary
