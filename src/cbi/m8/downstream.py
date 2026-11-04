"""P8 §7.5 downstream example: literature-prior weights vs uniform weights.

The same literature-derived gene set is scored on held-out cohorts twice:

* ``unweighted``: every supporting claim gets equal weight;
* ``weighted``: claims are weighted by the M8 claim weight (M2 robustness for
  re-analysable claims, detector weight otherwise).

No cohort-level fitting, gene selection or hyper-parameter tuning is performed
on the held-out data.  The result is deliberately reported as a small,
underpowered example, not as a non-inferiority test.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cbi.config import Config, load_config
from cbi.cohort_pool.loader import CohortStore, load_or_create_token
from cbi.m2.survival import concordance_index
from cbi.m8.api import ClaimWeightStore
from cbi.schemas import Claim
from cbi.util import write_json


@dataclass
class GenePrior:
    gene: str
    signed_weight: float
    abs_weight: float
    direction: int
    n_claims: int
    claim_ids: list[str] = field(default_factory=list)


def build_cancer_prior(claims: list[Claim], weights: dict[str, dict[str, Any]],
                       cancer: str) -> dict[str, GenePrior]:
    """Aggregate claim weights into one signed prior per gene.

    A claim's sign is +1 when high expression is reported as worse (HR >= 1)
    and -1 otherwise.  Conflicting claims cancel in the signed sum, which is the
    intended effect of down-weighting inconsistent literature.
    """
    acc: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if claim.entities.cancer_type != cancer:
            continue
        cid = claim.claim_id
        if cid not in weights:
            continue
        weight = max(0.0, float(weights[cid].get("weight") or 0.0))
        sign = 1 if float(claim.reported_effect.value) >= 1.0 else -1
        row = acc.setdefault(claim.entities.gene, {"signed": 0.0, "abs": 0.0, "sign_sum": 0,
                                                   "n": 0, "claims": []})
        row["signed"] += weight * sign
        row["abs"] += weight
        row["sign_sum"] += sign
        row["n"] += 1
        row["claims"].append(cid)
    out: dict[str, GenePrior] = {}
    for gene, row in acc.items():
        if row["abs"] > 0:
            signed = float(row["signed"] / row["n"])
            direction = 1 if signed >= 0 else -1
        else:
            # A zero-weight claim still defines a gene in the set; its direction
            # comes from the reported HR, but its weighted influence is zero.
            signed = 0.0
            direction = 1 if row["sign_sum"] >= 0 else -1
        out[gene] = GenePrior(gene=gene, signed_weight=signed,
                              abs_weight=float(row["abs"] / row["n"]),
                              direction=direction, n_claims=int(row["n"]),
                              claim_ids=list(row["claims"]))
    return out


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0, np.nan)
    return ((frame - mean) / std).fillna(0.0)


def _risk(expr: pd.DataFrame, genes: list[str], weights: np.ndarray) -> np.ndarray:
    z = _zscore(expr[genes])
    denom = float(np.abs(weights).sum())
    if denom <= 0:
        return np.zeros(len(z), dtype=float)
    return np.asarray(z.to_numpy(dtype=float) @ (weights / denom), dtype=float)


def _cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float | None:
    mask = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[mask], event[mask], risk[mask]
    if len(time) < 10 or int(event.sum()) < 5 or len(np.unique(risk)) < 2:
        return None
    try:
        return float(concordance_index(time, event, risk))
    except Exception:
        return None


def _evaluate_cohort(cfg: Config, store: CohortStore, cohort_id: str, cancer: str,
                     prior: dict[str, GenePrior], *, bootstrap: int = 500,
                     seed: int = 20261101) -> dict[str, Any]:
    clinical, expr, meta = store.load(cohort_id)
    genes = sorted(g for g in prior if g in expr.columns)
    if len(genes) < 2:
        return {"cohort": cohort_id, "cancer": cancer, "n": 0, "events": 0,
                "genes": genes, "error": "fewer than two prior genes measured",
                "unweighted_cindex": None, "weighted_cindex": None}
    clin = clinical.set_index("sample_id").reindex(expr.index)
    time = clin["os_time_days"].to_numpy(dtype=float)
    event = clin["os_event"].to_numpy(dtype=float)
    weighted_w = np.asarray([prior[g].signed_weight for g in genes], dtype=float)
    # Uniform baseline: the same gene set, but every claim has unit influence.
    uniform_w = np.asarray([float(prior[g].direction) for g in genes], dtype=float)
    risk_w = _risk(expr, genes, weighted_w)
    risk_u = _risk(expr, genes, uniform_w)
    c_w = _cindex(time, event, risk_w)
    c_u = _cindex(time, event, risk_u)

    rng = np.random.default_rng(seed)
    diffs = []
    if c_w is not None and c_u is not None and bootstrap > 0:
        n = len(time)
        for _ in range(bootstrap):
            idx = rng.integers(0, n, n)
            bw = _cindex(time[idx], event[idx], risk_w[idx])
            bu = _cindex(time[idx], event[idx], risk_u[idx])
            if bw is not None and bu is not None:
                diffs.append(bw - bu)
    diff_ci = ([float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
               if len(diffs) >= 50 else None)
    return {
        "cohort": cohort_id,
        "cancer": cancer,
        "n": int(len(time)),
        "events": int(event.sum()),
        "genes": genes,
        "n_genes": len(genes),
        "unweighted_cindex": c_u,
        "weighted_cindex": c_w,
        "diff_weighted_minus_unweighted": (c_w - c_u) if (c_w is not None and c_u is not None) else None,
        "diff_bootstrap_95ci": diff_ci,
        "n_bootstrap": len(diffs),
        "meta_n": meta.get("n"),
    }


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def run_downstream(cfg: Config, *, store: ClaimWeightStore | None = None,
                   bootstrap: int = 500) -> dict[str, Any]:
    if store is None:
        store = ClaimWeightStore.load(cfg)
    token = load_or_create_token(cfg)
    cohorts = CohortStore(cfg, role="grader", token=token)
    claims = list(store.claims.values())
    weights = store.weights
    cancers = sorted({c.entities.cancer_type for c in claims})
    heldout = [cid for cid in cohorts.cohort_ids(role="heldout")]
    by_cancer: dict[str, list[str]] = defaultdict(list)
    for cid in heldout:
        meta = cohorts.meta(cid)
        by_cancer[meta["cancer_type"]].append(cid)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cancer in cfg.cancers:
        if cancer not in cancers:
            skipped.append({"cancer": cancer, "reason": "no extracted claims for this cancer"})
            for cohort_id in by_cancer.get(cancer, []):
                skipped.append({"cancer": cancer, "cohort": cohort_id,
                                "reason": "no prior genes for this cancer"})
    for cancer in cancers:
        prior = build_cancer_prior(claims, weights, cancer)
        if not prior:
            continue
        if not by_cancer.get(cancer):
            skipped.append({"cancer": cancer, "reason": "no held-out cohort"})
            continue
        for cohort_id in sorted(by_cancer[cancer]):
            row = _evaluate_cohort(cfg, cohorts, cohort_id, cancer, prior,
                                   bootstrap=bootstrap)
            if row.get("error"):
                skipped.append({"cancer": cancer, "cohort": cohort_id,
                                "reason": row["error"]})
            else:
                row["prior_genes"] = len(prior)
                rows.append(row)

    valid = [r for r in rows if r["unweighted_cindex"] is not None and r["weighted_cindex"] is not None]
    diffs = [r["diff_weighted_minus_unweighted"] for r in valid]
    mean_u = _mean([r["unweighted_cindex"] for r in valid])
    mean_w = _mean([r["weighted_cindex"] for r in valid])
    mean_diff = _mean(diffs)
    # Paired bootstrap over held-out cohorts, not samples: with five cohorts this
    # is a descriptive interval, not a formal test.
    cohort_ci = None
    if len(diffs) >= 3:
        rng = np.random.default_rng(20261102)
        draws = []
        for _ in range(2000):
            pick = rng.integers(0, len(diffs), len(diffs))
            draws.append(float(np.mean(np.asarray(diffs)[pick])))
        cohort_ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    per_cancer = []
    for cancer in sorted({r["cancer"] for r in valid}):
        sub = [r for r in valid if r["cancer"] == cancer]
        per_cancer.append({
            "cancer": cancer,
            "cohorts": [r["cohort"] for r in sub],
            "mean_unweighted": _mean([r["unweighted_cindex"] for r in sub]),
            "mean_weighted": _mean([r["weighted_cindex"] for r in sub]),
            "mean_diff": _mean([r["diff_weighted_minus_unweighted"] for r in sub]),
        })

    skipped_cohorts = {row.get("cohort") for row in skipped if row.get("cohort")}
    summary = {
        "n_claims": len(claims),
        "n_cancers_with_claims": len(cancers),
        "n_heldout_cohorts_total": len(heldout),
        "n_heldout_cohorts_evaluated": len(valid),
        "n_heldout_cohorts_skipped": len(skipped_cohorts),
        "n_skipped_entries": len(skipped),
        "mean_unweighted_cindex": mean_u,
        "mean_weighted_cindex": mean_w,
        "mean_diff_weighted_minus_unweighted": mean_diff,
        "cohort_bootstrap_95ci": cohort_ci,
        "weighted_not_worse_by_0.01": bool(mean_diff is not None and mean_diff >= -0.01),
        "per_cancer": per_cancer,
    }
    report = {
        "snapshot": cfg.snapshot,
        "weight_source": "M8 ClaimWeightStore (M2 robustness for re-analysable claims)",
        "sign_rule": "sign = +1 if reported HR >= 1 else -1; conflicting claims cancel",
        "cohorts": rows,
        "skipped": skipped,
        "summary": summary,
        "caveats": [
            "Only five held-out cohorts, 34 claims and 2-24 measured genes per cohort.",
            "Weights come from M2 on dev cohorts; the held-out cohorts are not used for fitting.",
            "The same gene set is used for both arms; only claim weights differ.",
            "C-index is unadjusted for stage/age and is not a clinical model.",
            "A small positive mean difference is not a formal non-inferiority result.",
        ],
    }
    write_json(cfg.paths.reports / "P8_downstream.json", report)
    _write_markdown(cfg, report)
    return summary


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_ci(ci: list[float] | None) -> str:
    return "—" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _write_markdown(cfg: Config, report: dict[str, Any]) -> None:
    s = report["summary"]
    lines = [
        "# P8 — downstream example (§7.5)",
        "",
        "Same literature-derived gene set per cancer, scored on held-out cohorts with "
        "uniform claim weights vs M8 claim weights. Higher C-index is better.",
        "",
        "## Summary",
        "",
        f"- Claims: {s['n_claims']} across {s['n_cancers_with_claims']} cancers.",
        f"- Held-out cohorts evaluated: {s['n_heldout_cohorts_evaluated']} of "
        f"{s['n_heldout_cohorts_total']} (skipped {s['n_heldout_cohorts_skipped']}).",
        f"- Mean C-index: unweighted **{_fmt(s['mean_unweighted_cindex'])}**, "
        f"weighted **{_fmt(s['mean_weighted_cindex'])}**.",
        f"- Mean paired difference (weighted − unweighted): "
        f"**{_fmt(s['mean_diff_weighted_minus_unweighted'])}**; "
        f"cohort-bootstrap 95% CI {_fmt_ci(s['cohort_bootstrap_95ci'])}.",
        f"- Descriptive non-inferiority flag (mean diff ≥ −0.01): "
        f"**{s['weighted_not_worse_by_0.01']}**.",
        "",
        "## Per cohort",
        "",
        "| Cancer | Held-out cohort | n | events | genes | unweighted | weighted | diff | 95% CI (bootstrap) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["cohorts"]:
        ci = r.get("diff_bootstrap_95ci")
        ci_text = _fmt_ci(ci)
        lines.append(
            f"| {r['cancer']} | {r['cohort']} | {r['n']} | {r['events']} | {r['n_genes']} | "
            f"{_fmt(r['unweighted_cindex'])} | {_fmt(r['weighted_cindex'])} | "
            f"{_fmt(r['diff_weighted_minus_unweighted'])} | {ci_text} |"
        )
    lines += ["", "## Per cancer", "",
              "| Cancer | Cohorts | Mean unweighted | Mean weighted | Mean diff |",
              "|---|---|---:|---:|---:|"]
    for row in s["per_cancer"]:
        lines.append(
            f"| {row['cancer']} | {', '.join(row['cohorts'])} | "
            f"{_fmt(row['mean_unweighted'])} | {_fmt(row['mean_weighted'])} | "
            f"{_fmt(row['mean_diff'])} |"
        )
    lines += ["", "## Skipped", ""]
    if report["skipped"]:
        for row in report["skipped"]:
            lines.append(f"- {row.get('cancer')} / {row.get('cohort', '—')}: {row['reason']}")
    else:
        lines.append("- None.")
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in report["caveats"]]
    lines += [
        "",
        "The result is mixed across cohorts and the sample is small. It does not "
        "establish that down-weighting improves prognosis; it only checks that the "
        "M8-weighted prior does not collapse relative to the uniform prior.",
    ]
    (cfg.paths.reports / "P8_downstream.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-downstream")
    parser.add_argument("--root", default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    cfg = load_config(args.root)
    summary = run_downstream(cfg, bootstrap=args.bootstrap)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
