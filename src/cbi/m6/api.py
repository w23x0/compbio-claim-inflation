"""HTTP grader service (ported from comp01's FastAPI layout).

The CLI entry point (`cbi.m6.service_cli`) grades one artifact directory per
process.  This module adds a long-running HTTP wrapper for a grader-owned
service: the token is checked on every non-health route, per-report results are
never returned, and held-out numbers are written only to the grader-internal
directory.

Run:

    CBI_GRADER_TOKEN=... python -m cbi.m6.api --root . --detector models/detector_v1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from cbi.config import load_config
from cbi.m6.service import GraderService
from pydantic import BaseModel, Field

from cbi.schemas import Claim


class GradeRequest(BaseModel):
    artifact_dir: str
    compute_heldout: bool = True


class ScreenReport(BaseModel):
    claim: dict
    manuscript: str = ""


class ScreenRequest(BaseModel):
    reports: list[ScreenReport] = Field(default_factory=list)


class VisibleRequest(BaseModel):
    claim: dict


def _public(result: Any) -> dict[str, Any]:
    return {
        "submission_id": result.submission_id,
        "score": result.score,
        "flag_rate": result.flag_rate,
        "visible_inflation": result.visible_inflation,
        "rejected_by": result.rejected_by,
        "realism_pass": result.realism_pass,
        "cheat_check": result.cheat_check,
        "detector_version": result.detector_version,
        "n_reports": result.n_reports,
    }


def create_app(cfg, detector_dir: str | Path, token: str | None = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("install the 'api' extra: fastapi and uvicorn") from exc

    expected = token or os.environ.get("CBI_GRADER_TOKEN")
    if not expected:
        raise RuntimeError("CBI_GRADER_TOKEN is required")
    grader = GraderService(cfg, expected, detector_dir)

    def require_token(x_grader_token: str = Header(default="")) -> None:
        if x_grader_token != expected:
            raise HTTPException(status_code=401, detail="invalid grader token")

    app = FastAPI(title="CBI grader service", version="1.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "detector_version": grader.detector.version}

    @app.post("/grade", dependencies=[Depends(require_token)])
    def grade(req: GradeRequest) -> dict[str, Any]:
        path = Path(req.artifact_dir).resolve()
        root = cfg.paths.artifacts.resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="artifact_dir outside redteam_artifacts")
        result = grader.grade_submission(path, compute_heldout=req.compute_heldout, write_log=False)
        return _public(result)

    @app.post("/screen", dependencies=[Depends(require_token)])
    def screen(req: ScreenRequest) -> dict[str, Any]:
        rows = []
        for item in req.reports:
            claim = Claim.model_validate(item.claim)
            out = grader.detector.score(claim, item.manuscript, neutral=True)
            rows.append({
                "claim_id": out.claim_id,
                "evidence_score": out.evidence_score,
                "flag": out.flag,
                "threshold_used": out.threshold_used,
                "rationale": out.rationale,
            })
        return {"n": len(rows), "reports": rows,
                "flag_rate": sum(1 for r in rows if r["flag"]) / len(rows) if rows else 0.0}

    @app.post("/visible_inflation", dependencies=[Depends(require_token)])
    def visible_inflation(req: VisibleRequest) -> dict[str, Any]:
        claim = Claim.model_validate(req.claim)
        gt = grader.engine.evaluate_claim(claim, independent_role="visible_eval")
        return {
            "claim_id": gt.claim_id,
            "role": "visible_eval",
            "inflation": gt.inflation,
            "inflation_z": gt.inflation_z,
            "robustness_score": gt.robustness_score,
            "gt_confidence": gt.gt_confidence,
        }

    @app.get("/isolation/check", dependencies=[Depends(require_token)])
    def isolation_check() -> dict[str, Any]:
        return {
            "eval_root_exists": cfg.paths.eval_root.exists(),
            "grader_internal_exists": cfg.paths.grader_internal.exists(),
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-grader-http")
    parser.add_argument("--root", default=None)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn

    cfg = load_config(args.root)
    app = create_app(cfg, args.detector)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
