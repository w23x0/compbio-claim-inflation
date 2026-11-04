"""Bubblewrap isolation for red-team workers.

Masks eval_restricted, held-out data, grader secrets/internal logs, and all
other workers' artifacts.  Only dev cohort data, the code, the evidence model,
the safe config view, and the worker's own artifact directory are visible.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from cbi.config import Config, write_yaml
from cbi.util import write_json


def bwrap_available() -> bool:
    return shutil.which("bwrap") is not None


_VIEW_LOCK = threading.Lock()


def prepare_worker_view(cfg: Config) -> Path:
    view = cfg.paths.root / ".worker_view"
    with _VIEW_LOCK:
        return _prepare_worker_view(cfg, view)


def _prepare_worker_view(cfg: Config, view: Path) -> Path:
    view.mkdir(parents=True, exist_ok=True)
    roles = cfg.cohort_split.get("roles", {})
    dev_roles = {cid: role for cid, role in roles.items() if role == "dev"}
    write_yaml(view / "cohort_split.yaml", {
        "version": 1, "snapshot": cfg.snapshot, "frozen": True,
        "rule": "worker view: dev cohorts only", "roles": dev_roles,
        "split_hash": "worker-view",
    })
    dev_entries = [c for c in cfg.cohort_pool.get("cohorts", [])
                   if roles.get(c["cohort_id"]) == "dev"]
    write_yaml(view / "cohort_pool.yaml", {
        "version": 1, "snapshot": cfg.snapshot, "source": "worker_view",
        "cohorts": dev_entries, "manifest_hash": "worker-view",
    })
    return view


def build_bwrap_command(cfg: Config, artifact_dir: Path, job_path: Path) -> list[str]:
    root = cfg.paths.root
    view = prepare_worker_view(cfg)
    cmd = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--unshare-all",
        "--share-net",
        "--cap-drop", "ALL",
        "--new-session",
        "--die-with-parent",
        # Restricted data and secrets are invisible.
        "--tmpfs", str(cfg.paths.eval_root),
        "--tmpfs", str(cfg.paths.secrets),
        "--tmpfs", str(cfg.paths.grader_internal),
        # Raw cohorts, reports and logs are not worker-visible.  Only the
        # processed dev cohorts under data/processed are readable.
        "--tmpfs", str(cfg.paths.raw),
        "--tmpfs", str(cfg.paths.reports),
        "--tmpfs", str(cfg.paths.logs),
        # Other workers' artifacts are invisible; this worker's directory is writable.
        "--tmpfs", str(cfg.paths.artifacts),
        "--dir", str(artifact_dir),
        "--bind", str(artifact_dir), str(artifact_dir),
        # Safe config view replaces the global split and pool.
        "--ro-bind", str(view / "cohort_split.yaml"), str(cfg.paths.configs / "cohort_split.yaml"),
        "--ro-bind", str(view / "cohort_pool.yaml"), str(cfg.paths.configs / "cohort_pool.yaml"),
        "--ro-bind", str(job_path), str(job_path),
        "--chdir", str(root),
        str(sys.executable),
        "-m", "cbi.m5.worker",
    ]
    return cmd


def run_worker_sandboxed(cfg: Config, job: dict[str, Any], artifact_dir: Path, *,
                         timeout: int = 240) -> subprocess.CompletedProcess:
    if not bwrap_available():
        raise RuntimeError("bubblewrap (bwrap) is required for sandboxed red-team workers")
    view = prepare_worker_view(cfg)
    job_path = view / f"job_{job['submission_id']}.json"
    write_json(job_path, job)
    try:
        job_path.chmod(0o600)
    except OSError:
        pass
    cmd = build_bwrap_command(cfg, artifact_dir, job_path)
    env = {k: os.environ[k] for k in ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH"]
           if k in os.environ}
    env.update({
        "CBI_ROOT": str(cfg.paths.root),
        "CBI_WORKER_JOB": str(job_path),
        "PYTHONPATH": str(cfg.paths.root),
    })
    # The red team is allowed to query the evidence model; the LLM backend needs
    # its endpoint credential inside the worker.
    for key in ["CBI_LLM_ENDPOINT", "CBI_LLM_MODEL", "CBI_LLM_API_KEY", "CBI_LLM_SESSION",
                "CBI_EVIDENCE_ENDPOINT", "CBI_EVIDENCE_MODEL", "CBI_EVIDENCE_API_KEY"]:
        if key in os.environ:
            env[key] = os.environ[key]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=str(cfg.paths.root))


def run_worker_inprocess(cfg: Config, job: dict[str, Any], artifact_dir: Path,
                         timeout: int | None = None) -> Any:
    raise RuntimeError("in-process workers are disabled; bwrap is mandatory")
