"""Command-line entry point (MD-aligned, real-data only).

P0-P8 require a populated real cohort pool; P1/P4/P6/P7 additionally require a
configured LLM.  The synthetic demo pipeline has been removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbi.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi", description="computational-biology inflation harness")
    parser.add_argument("--root", default=None, help="project root (default: nearest parent of cbi/)")
    parser.add_argument("stage", choices=["status", "m1-audit", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])
    parser.add_argument("--dir", default="data/raw/pmc_real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="articles per M1 request (default: CBI_M1_BATCH_SIZE or 4)")
    p4_mode = parser.add_mutually_exclusive_group()
    p4_mode.add_argument("--prepare-only", action="store_true",
                         help="P4: prepare documents and report cache readiness without API calls")
    p4_mode.add_argument("--offline", action="store_true",
                         help="P4: train from complete cached scores without API calls")
    args = parser.parse_args(argv)
    if (args.prepare_only or args.offline) and args.stage != "p4":
        parser.error("--prepare-only and --offline are only available for p4")
    cfg = load_config(args.root)

    if args.stage == "status":
        from cbi.m4.detector import active_detector_dir

        try:
            detector_dir = active_detector_dir(cfg.paths.models)
        except RuntimeError:
            detector_dir = None
        print(json.dumps({
            "root": str(cfg.paths.root),
            "snapshot": cfg.snapshot,
            "detector": detector_dir is not None,
            "detector_directory": str(detector_dir) if detector_dir else None,
            "p4_v2_trained": (cfg.paths.models / "detector_llm_v2" / "meta.json").is_file()
                             and (cfg.paths.models / "detector_llm_v2" / "fusion.json").is_file(),
            "ground_truth": (cfg.paths.processed / "ground_truth.json").is_file(),
            "pmc_cache": len(list((cfg.paths.raw / "pmc_real").glob("*.xml"))),
            "real_cohorts": sorted(p.name for p in (cfg.paths.raw / "geo").glob("*") if p.is_dir())
            if (cfg.paths.raw / "geo").is_dir() else [],
            "pool_cohorts": len(cfg.cohort_pool.get("cohorts", [])),
            "split_roles": len(cfg.cohort_split.get("roles", {})),
        }, indent=2))
        return 0

    if args.stage == "m1-audit":
        from cbi.m1.extractor import LLMRequiredError
        from cbi.m1.public_audit import main as audit_main

        argv = ["--root", str(cfg.paths.root), "--dir", args.dir]
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        if args.workers is not None:
            argv += ["--workers", str(args.workers)]
        if args.batch_size is not None:
            argv += ["--batch-size", str(args.batch_size)]
        try:
            return audit_main(argv)
        except LLMRequiredError as exc:
            print(f"BLOCKED: {exc}")
            return 2

    from cbi import pipeline

    try:
        _run_stage(args, cfg, pipeline)
    except Exception as exc:
        print(f"STAGE FAILED: {type(exc).__name__}: {exc}")
        return 2
    return 0


def _run_stage(args, cfg, pipeline) -> None:
    if args.stage == "p0":
        print(json.dumps(pipeline.run_p0(cfg), indent=2))
    elif args.stage == "p1":
        print(json.dumps(pipeline.run_p1(cfg, workers=max(1, args.workers or 8), limit=args.limit,
                                    batch_size=args.batch_size), indent=2))
    elif args.stage == "p2":
        print(json.dumps(pipeline.run_p2(cfg), indent=2))
    elif args.stage == "p3":
        print(json.dumps(pipeline.run_p3(cfg), indent=2))
    elif args.stage == "p4":
        print(json.dumps(pipeline.run_p4(cfg, workers=args.workers,
                                        prepare_only=args.prepare_only, offline=args.offline), indent=2))
    elif args.stage == "p5":
        print(json.dumps(pipeline.run_p5(cfg), indent=2))
    elif args.stage == "p6":
        print(json.dumps(pipeline.run_p6(cfg, rounds=args.rounds, workers=args.workers), indent=2))
    elif args.stage == "p7":
        print(json.dumps(pipeline.run_p7(cfg, cycles=args.rounds), indent=2))
    elif args.stage == "p8":
        print(json.dumps(pipeline.run_p8(cfg), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
