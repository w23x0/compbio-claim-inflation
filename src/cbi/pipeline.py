"""Reproducible P0/P1/P2/P3/P4/P5/P6/P7/P8 entry points.

Each stage either runs or fails loudly; there is no synthetic fallback.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cbi.config import Config, find_root, load_config
from cbi.cohort_pool.cleaning import clean_all
from cbi.cohort_pool.loader import load_or_create_token
from cbi.cohort_pool.split import validate_split
from cbi.m4.detector import active_detector_dir
from cbi.m4.train_llm import train_llm_detector
from cbi.m5.isolation_tests import run_isolation_tests
from cbi.m5.orchestrator import run_campaign
from cbi.m7.blue import run_blue_cycles
from cbi.m8.api import ClaimWeightStore
from cbi.util import write_json


def _check_memory(workers: int = 1) -> None:
    try:
        info = Path("/proc/meminfo").read_text()
        available_kb = next(int(line.split()[1]) for line in info.splitlines()
                            if line.startswith("MemAvailable:"))
    except Exception:
        return
    available_gb = available_kb / (1024 * 1024)
    # Column-projected worker memory is small; budget ~1 GiB/worker plus 2 GiB.
    required_gb = 2.0 + 1.0 * max(1, workers)
    if available_gb < required_gb:
        raise RuntimeError(
            f"only {available_gb:.1f} GiB RAM available; need ~{required_gb:.0f} GiB for {workers} workers. "
            "Stop other processes or lower --workers."
        )


def _script(name: str, *args: str) -> None:
    proc = subprocess.run([sys.executable, name, *args], cwd=find_root(),
                          text=True, capture_output=True)
    if proc.returncode != 0:
        detail = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
        raise RuntimeError(f"{name} failed ({proc.returncode}):\n{detail[-2000:]}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def run_p0(cfg: Config) -> dict:
    _script("scripts/build_p0_config.py")
    cfg = load_config(cfg.paths.root)
    validate_split(cfg)
    clean_all(cfg)
    _script("scripts/validate_p0.py")
    return {"cohorts": len(cfg.cohort_pool.get("cohorts", []))}


def run_p1(cfg: Config, *, workers: int = 8, limit: int | None = None,
           batch_size: int | None = None) -> dict:
    args = ["--dir", "data/raw/pmc_real", "--workers", str(workers)]
    if limit is not None:
        args += ["--limit", str(limit)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    args += ["--cache-dir", str(cfg.paths.processed / "p1_cache")]
    _script("-m", "cbi.m1.public_audit", *args)
    _script("scripts/filter_real_claims.py")
    _script("scripts/build_real_corpus.py", "--endpoints", "OS")
    return {"claims": len(json.loads((cfg.paths.reports / "P1_public_pmc_claims_valid.json").read_text()))}


def run_p2(cfg: Config) -> dict:
    _script("scripts/run_p2_real.py", "--endpoints", "OS")
    _script("scripts/build_real_corpus.py", "--endpoints", "OS")
    return {"ground_truth": len(json.loads((cfg.paths.processed / "ground_truth.json").read_text()))}


def run_p3(cfg: Config) -> dict:
    _script("scripts/build_p3_context.py")
    _script("scripts/report_p3.py")
    path = cfg.paths.processed / "context_features.json"
    report = json.loads((cfg.paths.reports / "P3_context.json").read_text())
    return {"claims": len(json.loads(path.read_text())),
            "claim_coverage": report["claim_coverage"],
            "mean_context_coverage": report["variants"]["pub_plus_3y"]["mean_context_coverage"]}


def run_p4(cfg: Config, *, workers: int | None = None, prepare_only: bool = False,
           offline: bool = False) -> dict:
    workers = int(workers or os.environ.get("CBI_P4_WORKERS", "8"))
    _, report = train_llm_detector(cfg, workers=workers, prepare_only=prepare_only, offline=offline)
    return report


def run_p5(cfg: Config) -> dict:
    return run_isolation_tests(cfg)


def run_p6(cfg: Config, *, rounds: int | None = None, workers: int | None = None) -> dict:
    _check_memory(int(workers or cfg.redteam.get("workers", 8)))
    if not os.environ.get("CBI_GRADER_TOKEN"):
        os.environ["CBI_GRADER_TOKEN"] = load_or_create_token(cfg)
    detector_dir = active_detector_dir(cfg.paths.models)
    return run_campaign(cfg, rounds=int(rounds or cfg.redteam.get("rounds", 3)),
                        detector_dir=detector_dir,
                        workers=workers, sandbox=True)


def run_p7(cfg: Config, *, cycles: int | None = None) -> dict:
    _check_memory(int(cfg.redteam.get("workers", 8)))
    if not os.environ.get("CBI_GRADER_TOKEN"):
        os.environ["CBI_GRADER_TOKEN"] = load_or_create_token(cfg)
    return run_blue_cycles(cfg, cycles=cycles, sandbox=True)


def run_p8(cfg: Config) -> dict:
    store = ClaimWeightStore.load(cfg)
    path = store.export(cfg.paths.reports / "claim_weights.csv", "csv")
    from cbi.m8.downstream import run_downstream

    summary = run_downstream(cfg, store=store)
    return {"claims": len(store.weights), "export": str(path),
            "downstream": summary}
