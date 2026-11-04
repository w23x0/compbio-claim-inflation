"""Reproduce one red-team report number on its declared data (M6 check 3).

Generated report code calls `reproduce` with the cohort, gene, endpoint and the
serialized AnalysisSpec.  Only dev cohorts are admissible here; the grader runs
the code in a separate process/role.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cbi.config import load_config
from cbi.cohort_pool.loader import AccessDenied, CohortStore
from cbi.m2.specs import AnalysisSpec, estimate_effect


def reproduce(cohort_id: str, gene: str, endpoint: str, spec_dict: dict[str, Any],
              *, root: str | None = None) -> dict[str, Any]:
    cfg = load_config(root or os.environ.get("CBI_ROOT") or ".")
    store = CohortStore(cfg, allowed_roles={"dev"})
    cid = cohort_id
    if cid not in {c["cohort_id"] for c in cfg.cohort_pool.get("cohorts", [])}:
        cid = next((c["cohort_id"] for c in cfg.cohort_pool.get("cohorts", [])
                    if c.get("accession") == cohort_id), cohort_id)
    clinical = store.clinical(cid)
    series = store.gene_expression(cid, gene)
    if series is None:
        return {"ok": False, "error": f"gene not in cohort: {gene}"}
    spec = AnalysisSpec(**spec_dict)
    effect = estimate_effect(clinical, series, gene, spec, endpoint,
                             min_samples=int(cfg.scoring["gt"]["min_samples_per_cohort"]),
                             min_events=int(cfg.scoring["gt"]["min_events_per_cohort"]))
    if effect is None:
        return {"ok": False, "error": "no analyzable fit"}
    return {
        "ok": True,
        "cohort_id": cid,
        "gene": gene,
        "endpoint": endpoint,
        "log_effect": effect.log_effect,
        "se": effect.se,
        "p": effect.p,
        "n": effect.n,
        "events": effect.events,
        "hr": float(__import__("math").exp(effect.log_effect)),
    }


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m cbi.m5.reproduce spec.json", file=sys.stderr)
        return 2
    spec_obj = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    out = reproduce(
        spec_obj["cohort_id"], spec_obj["gene"], spec_obj["endpoint"], spec_obj["spec"],
        root=spec_obj.get("root"),
    )
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
