"""Build the M4 corpus from real PMC claims + P2 ground truth.

No synthetic claims.  Writes:
  data/processed/<snapshot>/corpus/claims.json
  data/processed/<snapshot>/corpus/claims.jsonl
  data/processed/<snapshot>/corpus/manuscripts/<claim_id>.md
  data/processed/<snapshot>/corpus/natural_split.json
  data/processed/<snapshot>/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m1.pmc import parse_pmc_xml
from cbi.util import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", default="OS", help="comma-separated, default OS")
    args = parser.parse_args()
    endpoints = {e.strip().upper() for e in args.endpoints.split(",") if e.strip()}
    cfg = load_config(ROOT)
    claims = read_json(ROOT / "reports" / "P1_public_pmc_claims_valid.json")
    claims = [c for c in claims if c.get("endpoint") in endpoints]
    base = cfg.paths.processed / "corpus"
    (base / "manuscripts").mkdir(parents=True, exist_ok=True)
    pmc_dir = cfg.paths.raw / "pmc_real"
    rows = []
    for claim in claims:
        pmcid = claim.get("source", {}).get("pmcid")
        if not pmcid:
            continue
        xml_path = pmc_dir / f"{pmcid}.xml"
        if not xml_path.is_file():
            continue
        article = parse_pmc_xml(xml_path)
        text = article.full_text
        (base / "manuscripts" / f"{claim['claim_id']}.md").write_text(text, encoding="utf-8")
        rows.append(claim)
    write_json(base / "claims.json", rows)
    (base / "claims.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    split = {}
    for claim in rows:
        year = int(claim["source"]["pub_date"][:4])
        bucket = "train" if year <= 2019 else ("val" if year <= 2021 else "test")
        split[claim["claim_id"]] = {"split": bucket, "year": year,
                                    "cancer": claim["entities"]["cancer_type"]}
    write_json(base / "natural_split.json", {"version": 1, "by_claim": split})
    gt_src = ROOT / "reports" / "P2_real.json"
    if gt_src.is_file():
        gt = read_json(gt_src)
        errors = {k: v for k, v in gt.items() if "error" in v}
        clean = {k: v for k, v in gt.items() if "error" not in v}
        write_json(cfg.paths.processed / "ground_truth.json", clean)
        write_json(cfg.paths.processed / "ground_truth_errors.json", errors)
        missing = sorted({r["claim_id"] for r in rows} - set(clean))
        if missing:
            print(f"warning: {len(missing)} claims have no ground-truth result: {missing[:5]}")
    counts = {b: sum(1 for v in split.values() if v["split"] == b) for b in ("train", "val", "test")}
    print(json.dumps({"claims": len(rows), "split": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
