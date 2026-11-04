"""Build frozen P0 cohort_pool/split configs from data/raw/geo.

Deterministic: sort by OS events then sample count, take up to 9 per cancer,
assign dev/visible_eval/heldout (>=1 each).  No human confirmation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config, write_yaml
from cbi.cohort_pool.cleaning import REQUIRED_CLINICAL
from cbi.util import read_json, sha256_obj, write_json

GEO = ROOT / "data" / "raw" / "geo"
RESTRICTED = ROOT / "data" / "eval_restricted" / "2026-11-01" / "raw"
MAX_PER_CANCER = 9


def cohort_dir(cid: str) -> Path:
    for root in (GEO, RESTRICTED):
        path = root / cid
        if path.is_dir():
            return path
    raise FileNotFoundError(f"cohort not found: {cid}")


def main() -> int:
    cfg = load_config(ROOT)
    max_per_cancer = int(cfg.mvp.get("cohorts", {}).get("per_cancer", 9))
    entries = []
    for d in sorted(p for root in (GEO, RESTRICTED) for p in root.glob("*") if p.is_dir()):
        cpath, epath, mpath = d / "clinical.parquet", d / "expression.parquet", d / "meta.json"
        if not (cpath.is_file() and epath.is_file() and mpath.is_file()):
            continue
        clinical = pd.read_parquet(cpath)
        expression = pd.read_parquet(epath)
        meta = read_json(mpath)
        events = int(pd.to_numeric(clinical["os_event"], errors="coerce").fillna(0).sum())
        missing = [c for c in REQUIRED_CLINICAL if c not in clinical.columns]
        if missing or len(clinical) < 80 or events < 15 or expression.shape[1] < 100:
            continue
        entries.append({
            "cohort_id": d.name,
            "cancer_type": str(meta.get("cancer_type") or "NA"),
            "accession": meta.get("accession") or d.name,
            "n": int(len(clinical)),
            "events_os": events,
            "platform": meta.get("platform"),
        })
    # Patient-overlap screen: same-cancer cohorts that share >=80% of the
    # smaller cohort's exact clinical survival signature are treated as one
    # dataset; keep the larger cohort.
    dropped: list[dict] = []
    by_cancer: dict[str, list[dict]] = {}
    for e in entries:
        by_cancer.setdefault(e["cancer_type"], []).append(e)
    for cancer, rows in list(by_cancer.items()):
        keep: list[dict] = []
        rows.sort(key=lambda r: (r["n"], r["events_os"]), reverse=True)
        for e in rows:
            if any(e["cohort_id"] == d["drop"] for d in dropped):
                continue
            ed = pd.read_parquet(cohort_dir(e["cohort_id"]) / "clinical.parquet")
            esig = set(zip(ed["age_years"].astype(str), ed["sex"].astype(str),
                           ed["stage_ajcc"].astype(str), ed["os_event"].astype(str),
                           ed["os_time_days"].round(2).astype(str)))
            duplicate_of = None
            for k in keep:
                kd = pd.read_parquet(cohort_dir(k["cohort_id"]) / "clinical.parquet")
                ksig = set(zip(kd["age_years"].astype(str), kd["sex"].astype(str),
                               kd["stage_ajcc"].astype(str), kd["os_event"].astype(str),
                               kd["os_time_days"].round(2).astype(str)))
                overlap = len(esig & ksig)
                if overlap >= 0.8 * min(len(esig), len(ksig)):
                    duplicate_of = k["cohort_id"]
                    break
            if duplicate_of:
                dropped.append({"drop": e["cohort_id"], "keep": duplicate_of,
                                "n": e["n"], "events_os": e["events_os"]})
            else:
                keep.append(e)
        by_cancer[cancer] = keep
    if dropped:
        print("duplicate cohorts dropped:", dropped)
    chosen = []
    roles: dict[str, str] = {}
    for cancer, rows in sorted(by_cancer.items()):
        rows.sort(key=lambda r: (r["events_os"], r["n"]), reverse=True)
        rows = rows[:max_per_cancer]
        if len(rows) < 5:
            print(f"warning: {cancer} has only {len(rows)} valid cohorts")
        n = len(rows)
        n_dev = max(2, n - 4)
        n_vis = max(1, min(2, n - n_dev - 1))
        n_held = n - n_dev - n_vis
        for i, row in enumerate(rows):
            role = "dev" if i < n_dev else ("visible_eval" if i < n_dev + n_vis else "heldout")
            roles[row["cohort_id"]] = role
            chosen.append({k: row[k] for k in ("cohort_id", "cancer_type", "accession", "n", "events_os", "platform")})
    pool = {"version": 1, "snapshot": cfg.snapshot, "source": "public_geo", "cohorts": chosen}
    split = {"version": 1, "snapshot": cfg.snapshot, "frozen": True,
             "rule": "deterministic: events/n desc; dev then visible_eval then heldout",
             "roles": roles, "split_hash": sha256_obj(roles)}
    write_yaml(ROOT / "configs" / "cohort_pool.yaml", pool)
    write_yaml(ROOT / "configs" / "cohort_split.yaml", split)
    cfg.paths.grader_internal.mkdir(parents=True, exist_ok=True)
    try:
        cfg.paths.grader_internal.chmod(0o700)
    except OSError:
        pass
    write_json(cfg.paths.grader_internal / "P0_config.json", {"pool": pool, "split": split,
                                                    "duplicate_screen_dropped": dropped})
    print(json.dumps({"cohorts": len(chosen), "by_cancer": {k: len(v) for k, v in by_cancer.items()},
                      "roles": {r: list(roles.values()).count(r) for r in ("dev", "visible_eval", "heldout")}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
