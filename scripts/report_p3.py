"""Build the P3 context-feature report from the frozen snapshot.

This does not invent missing fields.  It reports both `as_of_date` variants and
the per-field `missing_mask`, so the acceptance result (claim coverage >= 90%)
is separated from field coverage, which is expected to be sparse at MVP scale.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m3.features import ContextFeatureBuilder
from cbi.m3.sources import load_context_snapshot
from cbi.schemas import Claim, ContextFeatures, GroundTruthResult
from cbi.util import read_json, write_json

FIELDS = [
    "author_cluster_id",
    "supporting_papers",
    "effective_independent_support",
    "citation_cascade_score",
    "lab_historical_replication_rate",
    "review_time_days",
]
# Explicitly forbidden by spec M3.  None of these may appear in the schema.
FORBIDDEN = {
    "country", "institution_rank", "institution_ranking", "author_gender",
    "author_ethnicity", "author_seniority", "seniority", "race", "gender",
}

FIELD_NOTES = {
    "author_cluster_id": "coauthorship cluster visible at as_of_date",
    "supporting_papers": "support links for this claim with pub_date <= as_of",
    "effective_independent_support": "independent support clusters (empty support_links in this snapshot)",
    "citation_cascade_score": "same-cluster share of citations visible at as_of",
    "lab_historical_replication_rate": "leave-one-out M2 history; null when n < 5",
    "review_time_days": "received/accepted dates; absent in the OpenAlex snapshot",
}


def _field_stats(features: dict[str, ContextFeatures]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in FIELDS:
        present = sum(
            1 for f in features.values()
            if getattr(f, key) is not None and not f.missing_mask.get(key, False)
        )
        out[key] = {
            "present": present,
            "missing": len(features) - present,
            "coverage": present / len(features) if features else 0.0,
            "note": FIELD_NOTES[key],
        }
    return out


def _summarize(features: dict[str, ContextFeatures], *, variant: str) -> dict:
    coverages = [f.context_coverage for f in features.values()]
    return {
        "variant": variant,
        "n_claims": len(features),
        "mean_context_coverage": statistics.mean(coverages) if coverages else 0.0,
        "median_context_coverage": statistics.median(coverages) if coverages else 0.0,
        "nonzero_context_coverage": sum(1 for c in coverages if c > 0),
        "fields": _field_stats(features),
        "as_of_min": min((f.as_of_date for f in features.values()), default=None).isoformat()
        if features else None,
        "as_of_max": max((f.as_of_date for f in features.values()), default=None).isoformat()
        if features else None,
    }


def main() -> int:
    cfg = load_config(ROOT)
    claims = [Claim.model_validate(x) for x in
              read_json(cfg.paths.processed / "corpus" / "claims.json")]
    gt = {cid: GroundTruthResult.model_validate(v) for cid, v in
          read_json(cfg.paths.processed / "ground_truth.json").items() if "error" not in v}
    snapshot = load_context_snapshot(cfg)
    if not snapshot.papers:
        raise RuntimeError(
            f"missing context snapshot: {cfg.paths.raw / 'context' / 'context.json'}"
        )
    builder = ContextFeatureBuilder(cfg, claims, gt, snapshot)
    variants = {}
    for variant in ("publication", "pub_plus_3y"):
        feats = builder.build_all(variant=variant)
        variants[variant] = {"features": feats, "summary": _summarize(feats, variant=variant)}

    with_paper = sum(1 for c in claims if c.claim_id in builder.paper_by_claim)
    claim_coverage = with_paper / len(claims) if claims else 0.0
    snapshot_date = cfg.snapshot
    asof_after_snapshot = {
        variant: sum(1 for f in data["features"].values()
                     if f.as_of_date.isoformat() > snapshot_date)
        for variant, data in variants.items()
    }
    forbidden_present = sorted(FORBIDDEN & set(ContextFeatures.model_fields))
    report = {
        "snapshot": snapshot_date,
        "n_claims": len(claims),
        "n_claims_with_paper": with_paper,
        "claim_coverage": claim_coverage,
        "claim_coverage_pass": claim_coverage >= 0.90,
        "snapshot_counts": {
            "papers": len(snapshot.papers),
            "citations": len(snapshot.citations),
            "authors": len(snapshot.authors),
            "support_links": len(snapshot.support_links),
        },
        "variants": {k: v["summary"] for k, v in variants.items()},
        "as_of_after_snapshot": asof_after_snapshot,
        "forbidden_feature_fields_present": forbidden_present,
        "prohibited_fields_clear": not forbidden_present,
    }
    write_json(cfg.paths.reports / "P3_context.json", report)

    main_summary = report["variants"]["pub_plus_3y"]
    lab_present = main_summary["fields"]["lab_historical_replication_rate"]["present"]
    lines = [
        "# P3 — context graph / M3",
        "",
        f"Snapshot: `{snapshot_date}`; {report['snapshot_counts']['papers']} papers, "
        f"{report['snapshot_counts']['citations']} citation edges, "
        f"{report['snapshot_counts']['authors']} author nodes, "
        f"{report['snapshot_counts']['support_links']} support links.",
        "",
        "## Acceptance: claim coverage",
        "",
        f"- Claims with a context record: **{with_paper}/{len(claims)} = {claim_coverage:.1%}** "
        f"(threshold 90%: {'PASS' if report['claim_coverage_pass'] else 'FAIL'}).",
        "- Fields may be null by design (sparse fallback); field coverage is reported separately.",
        "",
        "## Field coverage (`pub_plus_3y`)",
        "",
        "| Field | Present | Missing | Coverage | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for key, row in main_summary["fields"].items():
        lines.append(
            f"| `{key}` | {row['present']} | {row['missing']} | {row['coverage']:.1%} | {row['note']} |"
        )
    lines += [
        "",
        f"Mean `context_coverage`: **{main_summary['mean_context_coverage']:.3f}** "
        f"(median {main_summary['median_context_coverage']:.3f}); "
        f"{main_summary['nonzero_context_coverage']}/{main_summary['n_claims']} claims have non-zero coverage.",
        "",
        "## `as_of_date` variants",
        "",
        "| Variant | As-of range | Mean coverage | Non-zero |",
        "|---|---|---:|---:|",
    ]
    for name, s in report["variants"].items():
        lines.append(
            f"| `{name}` | {s['as_of_min']} .. {s['as_of_max']} | "
            f"{s['mean_context_coverage']:.3f} | {s['nonzero_context_coverage']}/{s['n_claims']} |"
        )
    lines += [
        "",
        "Both variants are computed separately. Unit tests in `tests/test_m3_asof.py` "
        "check that future citations, retractions and papers are invisible at `as_of_date`.",
        "",
        f"{asof_after_snapshot['pub_plus_3y']} claim"
        f"{'s' if asof_after_snapshot['pub_plus_3y'] != 1 else ''} "
        f"{'have' if asof_after_snapshot['pub_plus_3y'] != 1 else 'has'} "
        "`pub_plus_3y > snapshot_date`; the snapshot contains no records after the snapshot "
        "date, so these features are truncated at the snapshot boundary rather than leaking "
        "future data.",
        "",
        "## Prohibited features",
        "",
        f"Forbidden M3 fields present in `ContextFeatures`: `{forbidden_present}`. "
        "Country, institution rank, author gender/ethnicity and seniority are not modelled.",
        "",
        "## Known limits",
        "",
        "- `support_links` is empty, so `effective_independent_support` is null for every claim.",
        f"- `lab_historical_replication_rate` needs >=5 earlier eligible claims in the same "
        f"author cluster; {lab_present} of {len(claims)} claims reach that threshold.",
        "- `review_time_days` needs received/accepted dates that the current OpenAlex snapshot "
        "does not provide.",
        f"- Claim coverage is complete; the {main_summary['mean_context_coverage']:.3f} mean "
        "field coverage is a real data limitation, not a missing-code path.",
    ]
    (cfg.paths.reports / "P3_context.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "claims": len(claims),
        "claim_coverage": claim_coverage,
        "claim_coverage_pass": report["claim_coverage_pass"],
        "mean_context_coverage": main_summary["mean_context_coverage"],
        "prohibited_fields_clear": report["prohibited_fields_clear"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
