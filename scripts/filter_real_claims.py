"""Apply M1 validity gates and attach accessions found in the PMC text.

Reads reports/P1_public_pmc_claims.json, parses the cached PMC XML for
GSE/TCGA/ArrayExpress accessions, writes reports/P1_public_pmc_claims_valid.json.
No model calls; deterministic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m1.audit import audit_claim
from cbi.m1.hgnc import HGNCResolver
from cbi.m1.pmc import parse_pmc_xml
from cbi.schemas import Claim
from cbi.util import read_json, write_json

ACC_RE = re.compile(r"\b(GSE\d{3,}|TCGA-[A-Z0-9]{2,8}|E-[A-Z]+-\d+)\b")


def main() -> int:
    cfg = load_config(ROOT)
    resolver = HGNCResolver(cfg.paths.raw / "hgnc" / "hgnc_complete.tsv")
    allowed = set(cfg.mvp.get("cancers", []))
    pool = {str(c.get("accession") or c.get("cohort_id")).upper() for c in cfg.cohort_pool.get("cohorts", [])}
    raw = read_json(ROOT / "reports" / "P1_public_pmc_claims.json")
    kept, rejected = [], []
    for obj in raw:
        claim = Claim.model_validate(obj)
        pmcid = claim.source.pmcid
        xml = ROOT / "data" / "raw" / "pmc_real" / f"{pmcid}.xml" if pmcid else None
        if xml is not None and xml.is_file():
            text = parse_pmc_xml(xml).full_text
            found = sorted({m.group(1).upper() for m in ACC_RE.finditer(text)})
            claim = claim.model_copy(update={"reported_cohort": claim.reported_cohort.model_copy(
                update={"accessions": sorted(set(claim.reported_cohort.accessions) | set(found))}
            )})
        if claim.reported_cohort.accessions:
            claim.reported_cohort.in_standard_pool = all(a.upper() in pool for a in claim.reported_cohort.accessions)
        issues = audit_claim(claim, allowed_cancers=allowed, resolver=resolver)
        if issues:
            rejected.append({"claim_id": claim.claim_id, "issues": issues})
        else:
            kept.append(claim.model_dump(mode="json"))
    write_json(ROOT / "reports" / "P1_public_pmc_claims_valid.json", kept)
    write_json(ROOT / "reports" / "P1_public_pmc_rejected.json", rejected)
    p1_path = ROOT / "reports" / "P1_public_pmc.json"
    if p1_path.is_file():
        p1 = read_json(p1_path)
        p1["validity_gate"] = {
            "raw_claims": len(raw), "valid_claims": len(kept),
            "rejected_claims": len(rejected),
            "valid_os_claims": sum(1 for c in kept if c.get("endpoint") == "OS"),
        }
        write_json(p1_path, p1)
    from collections import Counter
    counts = Counter(i.split(":", 1)[0] for r in rejected for i in r["issues"])
    print(json.dumps({"raw": len(raw), "valid": len(kept), "rejected": len(rejected),
                      "issues": dict(counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
