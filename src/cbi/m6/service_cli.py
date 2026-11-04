"""Standalone grader process entry point.

Run with the grader OS user and CBI_GRADER_TOKEN set; workers never receive the
token.  The shared log receives only the public aggregate entry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from cbi.config import load_config
from cbi.m6.service import GraderService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-grader")
    parser.add_argument("--root", default=None)
    parser.add_argument("--detector", required=True)
    parser.add_argument("submission_dir")
    parser.add_argument("--no-heldout", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.root)
    token = os.environ.get("CBI_GRADER_TOKEN")
    if not token:
        print("CBI_GRADER_TOKEN is required", file=sys.stderr)
        return 2
    grader = GraderService(cfg, token, args.detector)
    result = grader.grade_submission(args.submission_dir, compute_heldout=not args.no_heldout)
    public = {
        "submission_id": result.submission_id, "score": result.score,
        "flag_rate": result.flag_rate, "visible_inflation": result.visible_inflation,
        "rejected_by": result.rejected_by, "realism_pass": result.realism_pass,
        "cheat_check": result.cheat_check, "detector_version": result.detector_version,
    }
    print(json.dumps(public, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
