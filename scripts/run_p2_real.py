"""Run M2 on the real claim corpus + P0 cohort pool.

Usage: python3 scripts/run_p2_real.py
Reads reports/P1_public_pmc_claims.json and writes reports/P2_real.json/md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.cohort_pool.loader import CohortStore
from cbi.m2.analysis import CohortAnalyzer
from cbi.m2.engine import GroundTruthEngine
from cbi.schemas import Claim
from cbi.util import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", default="OS", help="comma-separated, default OS")
    args = parser.parse_args()
    endpoints = {e.strip().upper() for e in args.endpoints.split(",") if e.strip()}
    cfg = load_config(ROOT)
    claims_path = ROOT / "reports" / "P1_public_pmc_claims_valid.json"
    claims = [Claim.model_validate(x) for x in read_json(claims_path) if x.get("endpoint") in endpoints]
    store = CohortStore(cfg, allowed_roles={"dev"})
    analyzer = CohortAnalyzer(cfg, store)
    engine = GroundTruthEngine(cfg, analyzer)
    results = {}
    for claim in claims:
        try:
            result = engine.evaluate_claim(claim, independent_role="dev")
        except Exception as exc:
            results[claim.claim_id] = {"claim_id": claim.claim_id, "error": f"{type(exc).__name__}: {exc}"}
            continue
        results[claim.claim_id] = result.model_dump(mode="json")
    write_json(ROOT / "reports" / "P2_real.json", results)
    ok = [r for r in results.values() if "error" not in r]
    lines = ["# P2 — real ground-truth run", "",
             f"- Claims: {len(claims)}; evaluated: {len(ok)}; errors: {len(claims) - len(ok)}", "",
             "| claim | gene | cancer | endpoint | robustness | inflation | z | confidence |",
             "|---|---|---|---|---:|---:|---:|---|"]
    for claim in claims:
        r = results.get(claim.claim_id, {})
        if "error" in r:
            lines.append(f"| {claim.claim_id} | {claim.entities.gene} | {claim.entities.cancer_type} | {claim.endpoint} | — | — | — | {r['error']} |")
            continue
        lines.append(f"| {claim.claim_id} | {claim.entities.gene} | {claim.entities.cancer_type} | {claim.endpoint} | "
                     f"{r['robustness_score']:.3f} | {r['inflation']:.3f} | {r['inflation_z']:.2f} | {r['gt_confidence']} |")
    (ROOT / "reports" / "P2_real.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"claims": len(claims), "evaluated": len(ok), "errors": len(claims) - len(ok)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
