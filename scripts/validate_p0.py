"""Validate the real P0 cohort pool and emit a manifest.

Scans data/raw/geo/*/ for the standard contract, checks sample/event counts and
runs a small real-cohort Cox fit per cohort.  No synthetic fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.cohort_pool.cleaning import REQUIRED_CLINICAL
from cbi.cohort_pool.sanity import sanity_check_cohort
from cbi.config import load_config
from cbi.m2.real_cohort import evaluate_genes
from cbi.util import read_json, write_json

GEO = ROOT / "data" / "raw" / "geo"
RESTRICTED = ROOT / "data" / "eval_restricted" / "2026-11-01" / "raw"
GENES = {
    "LIHC": ["TP53", "MKI67", "SOX9"],
    "BRCA": ["TP53", "MKI67", "ESR1"],
    "LUAD": ["TP53", "MKI67", "EGFR"],
    "KIRC": ["TP53", "MKI67", "VHL"],
}


def main() -> int:
    cfg = load_config(ROOT)
    roles = cfg.cohort_split.get("roles", {})
    rows = []
    for cohort_dir in sorted(p for root in (GEO, RESTRICTED) for p in root.glob("*") if p.is_dir()):
        clinical_path = cohort_dir / "clinical.parquet"
        expr_path = cohort_dir / "expression.parquet"
        meta_path = cohort_dir / "meta.json"
        if not (clinical_path.is_file() and expr_path.is_file()):
            rows.append({"cohort_id": cohort_dir.name, "status": "missing_files"})
            continue
        try:
            clinical = pd.read_parquet(clinical_path)
            expression = pd.read_parquet(expr_path)
            meta = read_json(meta_path) if meta_path.is_file() else {}
        except Exception as exc:
            rows.append({"cohort_id": cohort_dir.name, "status": f"read_error:{exc}"})
            continue
        missing = [c for c in REQUIRED_CLINICAL if c not in clinical.columns]
        n = len(clinical)
        events = int(pd.to_numeric(clinical.get("os_event", []), errors="coerce").fillna(0).sum()) if "os_event" in clinical else 0
        cancer = str(meta.get("cancer_type") or clinical.get("cancer_type", pd.Series(["NA"])).iloc[0])
        genes = GENES.get(cancer, ["TP53", "MKI67"])
        fit = evaluate_genes(cohort_dir, genes)
        fits = [r for r in fit if "error" not in r]
        sanity = sanity_check_cohort(clinical, expression, meta) if not missing else {"passed": False, "checks": {}}
        status = "ok" if not missing and n >= 80 and events >= 15 and fits and sanity.get("passed") else "fail"
        rows.append({
            "cohort_id": cohort_dir.name, "accession": meta.get("accession"),
            "role": roles.get(cohort_dir.name),
            "cancer_type": cancer, "n": n, "events": events,
            "genes": int(expression.shape[1]), "platform": meta.get("platform"),
            "missing_columns": missing, "fits": len(fits), "sanity": bool(sanity.get("passed")),
            "status": status,
        })
    summary = {
        "n_cohorts": len(rows),
        "by_cancer": {},
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "cohorts": rows,
    }
    for row in rows:
        summary["by_cancer"].setdefault(row.get("cancer_type", "NA"), []).append(row["cohort_id"])
    internal = cfg.paths.grader_internal
    internal.mkdir(parents=True, exist_ok=True)
    try:
        internal.chmod(0o700)
    except OSError:
        pass
    write_json(internal / "P0_real_cohorts.json", summary)
    # Public report: only dev cohort identities; held-out/visible IDs and roles
    # stay grader-internal.
    dev_rows = [r for r in rows if r.get("role") == "dev"]
    public = {"n_cohorts": len(rows), "ok": summary["ok"],
              "by_cancer": {k: len(v) for k, v in summary["by_cancer"].items()},
              "role_counts": {role: sum(1 for r in rows if r.get("role") == role)
                              for role in ("dev", "visible_eval", "heldout")},
              "dev_cohorts": dev_rows}
    write_json(ROOT / "reports" / "P0_real_cohorts.json", public)
    lines = ["# P0 — real cohort pool", "",
             f"- Cohorts: {len(rows)}; ok: {summary['ok']}",
             f"- Roles: dev {public['role_counts']['dev']}, visible_eval {public['role_counts']['visible_eval']}, heldout {public['role_counts']['heldout']}",
             f"- By cancer: {public['by_cancer']}", "",
             "Dev cohorts (worker-visible):", "",
             "| cohort | cancer | n | events | genes | platform | status |",
             "|---|---|---:|---:|---:|---|---|"]
    for row in dev_rows:
        lines.append(f"| {row['cohort_id']} | {row.get('cancer_type','')} | {row.get('n','')} | "
                     f"{row.get('events','')} | {row.get('genes','')} | {row.get('platform','')} | {row['status']} |")
    lines += ["", "Held-out/visible cohort identities and roles are in",
              "`logs/grader_internal/P0_real_cohorts.json` (grader-only)."]
    if public["by_cancer"].get("KIRC", 0) < 5:
        lines += ["", "KIRC has 4 valid cohorts; no fifth independent public ccRCC",
                  "cohort with >=80 samples and >=15 OS events exists. No duplicate",
                  "patients were used to pad the pool."]
    (ROOT / "reports" / "P0_real_cohorts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
