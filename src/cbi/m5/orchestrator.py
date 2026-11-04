"""Red-team campaign orchestration and shared-log updates (M5/M6)."""
from __future__ import annotations

import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config
from cbi.m5.sandbox import run_worker_sandboxed
from cbi.m6.service import GradeResult, GraderService
from cbi.util import append_jsonl, percentile_rank, read_jsonl, write_json


def _entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = {x: items.count(x) for x in set(items)}
    n = len(items)
    return float(-sum((c / n) * math.log(c / n, 2) for c in counts.values()))


def run_round(cfg: Config, *, round_no: int, detector_dir: str | Path, workers: int,
              reports_per_batch: int, sandbox: bool = True,
              log_entries: list[dict] | None = None,
              family: str | None = None) -> list[GradeResult]:
    log_entries = log_entries if log_entries is not None else read_jsonl(cfg.paths.experiment_log)
    token = os.environ.get("CBI_GRADER_TOKEN")
    if not token:
        raise RuntimeError("CBI_GRADER_TOKEN must be supplied to the grader control plane")
    grader = GraderService(cfg, token, detector_dir)
    artifact_root = cfg.paths.artifacts / f"round_{round_no:02d}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for worker in range(workers):
        sid = f"rt_r{round_no:02d}_w{worker:02d}"
        artifact_dir = artifact_root / f"w{worker:02d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "submission_id": sid, "round": round_no, "worker": worker,
            "artifact_dir": str(artifact_dir), "root": str(cfg.paths.root),
            "detector_dir": str(detector_dir), "reports_per_batch": reports_per_batch,
            "seed": int(cfg.mvp["seed"]) + round_no * 1000 + worker,
            "family_quota": float(cfg.redteam.get("family_quota", 0.30)),
            "max_candidate_analyses": int(cfg.redteam.get("max_candidate_analyses", 120)),
            "log_entries": log_entries,
            "family": family,
            "sandbox": sandbox,
        }
        jobs.append((worker, sid, artifact_dir, job))
    results: list[GradeResult] = []

    def run_one(worker: int, sid: str, artifact_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
        # bwrap is mandatory; there is no in-process fallback.
        try:
            proc = run_worker_sandboxed(cfg, job, artifact_dir,
                                        timeout=int(cfg.redteam.get("worker_timeout_seconds", 900)))
        except Exception as exc:
            return {"ok": False, "worker": worker, "error": f"{type(exc).__name__}: {exc}"}
        if proc.returncode != 0:
            return {"ok": False, "worker": worker, "error": proc.stderr[-1000:]}
        return {"ok": True, "worker": worker, "submission_id": sid, "artifact_dir": str(artifact_dir)}

    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run_one, w, sid, ad, job) for w, sid, ad, job in jobs]
        for fut in as_completed(futures):
            outcomes.append(fut.result())
    outcomes.sort(key=lambda x: x["worker"])

    def grade_one(outcome: dict[str, Any]) -> GradeResult | None:
        if not outcome.get("ok"):
            append_jsonl(cfg.paths.experiment_log, {
                "submission_id": f"rt_r{round_no:02d}_w{outcome['worker']:02d}",
                "round": round_no, "worker": outcome["worker"], "strategy_family": family or "unknown",
                "summary": "worker error", "score": 0.0, "flag_rate": 0.0,
                "visible_inflation": None, "rejected_by": "worker_error", "realism_pass": False,
                "cheat_check": "worker_error", "detector_version": "unknown",
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": outcome.get("error", "")[:500],
            })
            return None
        artifact_dir = Path(outcome["artifact_dir"])
        try:
            # One grader instance per thread; the engine and cohort cache are
            # not shared across concurrent submissions.
            local = GraderService(cfg, token, detector_dir)
            return local.grade_submission(artifact_dir, write_log=True)
        except Exception as exc:
            append_jsonl(cfg.paths.experiment_log, {
                "submission_id": outcome["submission_id"], "round": round_no,
                "worker": outcome["worker"], "strategy_family": family or "unknown",
                "summary": f"grader error: {type(exc).__name__}", "score": 0.0,
                "flag_rate": 0.0, "visible_inflation": None, "rejected_by": "grader_error",
                "realism_pass": False, "cheat_check": "grader_error",
                "detector_version": "unknown", "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(exc)[:500],
            })
            return None

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(outcomes)))) as pool:
        graded = list(pool.map(grade_one, outcomes))
    results = [r for r in graded if r is not None]
    return results


def run_campaign(cfg: Config, *, rounds: int, detector_dir: str | Path, start_round: int = 1,
                 sandbox: bool | None = None, workers: int | None = None,
                 reports_per_batch: int | None = None, family: str | None = None) -> dict[str, Any]:
    workers = int(workers or cfg.redteam.get("workers", 8))
    k = int(reports_per_batch or cfg.redteam.get("reports_per_batch", 20))
    sandbox = bool(cfg.redteam.get("sandbox", True)) if sandbox is None else sandbox
    all_results: list[GradeResult] = []
    for r in range(start_round, start_round + rounds):
        log = read_jsonl(cfg.paths.experiment_log)
        results = run_round(cfg, round_no=r, detector_dir=detector_dir, workers=workers,
                            reports_per_batch=k, sandbox=sandbox, log_entries=log, family=family)
        all_results.extend(results)
        log = read_jsonl(cfg.paths.experiment_log)
        print(f"round {r}: {len(results)} graded; log entries {len(log)}")
    return summarize_log(cfg, all_results)


def summarize_log(cfg: Config, results: list[GradeResult]) -> dict[str, Any]:
    log = read_jsonl(cfg.paths.experiment_log)
    by_round: dict[str, list[dict]] = {}
    for entry in log:
        by_round.setdefault(str(entry["round"]), []).append(entry)
    curves = []
    for r in sorted(by_round, key=lambda x: int(x)):
        entries = by_round[r]
        scores = [float(e.get("score", 0.0)) for e in entries]
        curves.append({
            "round": int(r), "n": len(entries), "best": max(scores) if scores else 0.0,
            "mean": float(np.mean(scores)) if scores else 0.0,
            "passed_batches": int(sum(1 for e in entries if not e.get("rejected_by"))),
            "entropy": _entropy([e.get("strategy_family", "") for e in entries]),
        })
    summary = {
        "n_submissions": len(log),
        "curves": curves,
        "best_score": max([float(e.get("score", 0.0)) for e in log], default=0.0),
        "mean_flag_rate": float(np.mean([float(e.get("flag_rate", 0.0)) for e in log])) if log else 0.0,
        "family_counts": {f: sum(1 for e in log if e.get("strategy_family") == f)
                          for f in sorted({e.get("strategy_family", "") for e in log})},
        "results": [r.log_entry for r in results],
    }
    write_json(cfg.paths.reports / "P6_redteam.json", summary)
    from cbi.m5.plotting import write_curve_svg

    write_curve_svg(cfg.paths.reports / "P6_curve.svg", curves)
    lines = ["# P6 — red-team campaign", ""]
    for c in curves:
        lines.append(
            f"- Round {c['round']}: best **{c['best']:.3f}**, mean {c['mean']:.3f}, "
            f"passed {c['passed_batches']}/{c['n']}, family entropy {c['entropy']:.2f}"
        )
    lines += [
        "",
        f"Family counts: `{summary['family_counts']}`",
        "",
        "![score curve](P6_curve.svg)",
        "",
        "Flag rates use the neutral-context content screen. Held-out results are "
        "grader-internal and are not included in this report.",
    ]
    (cfg.paths.reports / "P6_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
