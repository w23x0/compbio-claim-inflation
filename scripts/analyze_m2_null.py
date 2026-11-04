"""Compare support-statistic alternatives offline; never change production labels."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbi.config import load_config
from cbi.cohort_pool.loader import CohortStore
from cbi.m2.analysis import CohortAnalyzer
from cbi.m2.meta import meta_random_effects
from cbi.m2.specs import default_spec
from cbi.util import read_json, sha256_file, write_json

VARIANTS = ("phi", "positive_centered", "excess_above_95pct")


def support_variants(z):
    p = float(stats.norm.cdf(z))
    return {"phi": p, "positive_centered": max(0.0, 2 * p - 1),
            "excess_above_95pct": max(0.0, (p - .95) / .05)}


def compare_row(*, z, robustness, weight, **metadata):
    support = support_variants(z)
    residual = robustness - weight * support["phi"]
    if residual < -1e-6 or residual > 1 - weight + 1e-6:
        raise ValueError("Reconstructed support is inconsistent with stored robustness; do not compare mismatched runs")
    return {**metadata, "z": float(z), "support": support, "unchanged_other_components": residual,
            "robustness": {name: float(np.clip(residual + weight * value, 0, 1)) for name, value in support.items()}}


def summarize(rows):
    return {name: {"n": len(rows),
                   "median": float(np.median([r["robustness"][name] for r in rows])) if rows else None,
                   "mean": float(np.mean([r["robustness"][name] for r in rows])) if rows else None,
                   "below_0_2": sum(r["robustness"][name] < .2 for r in rows),
                   "above_0_6": sum(r["robustness"][name] > .6 for r in rows)}
            for name in VARIANTS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    cfg = load_config(args.root)
    controls_path = cfg.paths.reports / "M2_controls.json"
    controls = read_json(controls_path)
    # Only the same dev cohorts and the same already-selected genes are used.
    # Refit the reference model to recover Phi(mu/se); reuse the stored remainder
    # of robustness instead of rerunning the full multiverse.
    store = CohortStore(cfg, allowed_roles={"dev"})
    analyzer = CohortAnalyzer(cfg, store)
    weight = float(cfg.weights["independent_support"])
    control_rows = {"positive": [], "negative": []}
    skipped = []
    for group in control_rows:
        for row in controls[group]["rows"]:
            if "robustness" not in row:
                skipped.append({"group": group, **row})
                continue
            cancer, gene = row["cancer"], row["gene"]
            ids = controls["cohorts"][cancer]
            if any(cfg.cohort_split["roles"].get(cid) != "dev" for cid in ids):
                raise ValueError("Control analysis may use only dev cohorts")
            effects = analyzer.effects_for_cohorts(ids, gene, default_spec("continuous"), "OS")
            if not effects:
                skipped.append({"group": group, "cancer": cancer, "gene": gene, "error": "no reference effects"})
                continue
            meta = meta_random_effects(np.asarray([e.log_effect for e in effects]), np.asarray([e.se for e in effects]))
            control_rows[group].append(compare_row(z=meta.pooled / meta.se, robustness=row["robustness"], weight=weight,
                                                   cancer=cancer, gene=gene, independent_k=len(effects)))
        print(f"{group}: recovered {len(control_rows[group])} control statistics", flush=True)
    gt_path = cfg.paths.processed / "ground_truth.json"
    corpus_rows = []
    for cid, result in read_json(gt_path).items():
        meta = result.get("meta")
        if not result.get("label_eligible") or not meta or meta["se"] <= 0:
            continue
        weights = result["weights_used"]
        r, m = result["reported_log_effect"], result["multiverse"]["median_log_effect"]
        other = weights["multiverse_support"] * result["multiverse"]["frac_same_direction_sig"]
        other += weights["shrinkage"] * (float(np.clip(m / r, 0, 1)) if m is not None and r > 0 else 0)
        z = meta["pooled_log_effect"] / meta["se"]
        expected = float(np.clip(other + weights["independent_support"] * stats.norm.cdf(z), 0, 1))
        if not math.isclose(expected, result["robustness_score"], abs_tol=1e-8):
            raise ValueError(f"Stored corpus robustness cannot be reconstructed: {cid}")
        corpus_rows.append(compare_row(z=z, robustness=expected, weight=weights["independent_support"],
                                       claim_id=cid, cancer=result["cancer_type"]))
    summaries = {key: summarize(rows) for key, rows in control_rows.items()}
    report = {"mode": "exploratory_offline_comparison", "snapshot": cfg.snapshot, "api_requests": 0,
              "production_formula_changed": False, "production_labels_changed": False,
              "formulae": {"phi": "Phi(z)", "positive_centered": "max(0, 2*Phi(z)-1)",
                           "excess_above_95pct": "max(0, (Phi(z)-0.95)/0.05)"},
              "ideal_null": {"assumption": "Z ~ N(0,1), so Phi(Z) ~ Uniform(0,1)",
                             "support_at_z0": support_variants(0),
                             "expected_support": {"phi": .5, "positive_centered": .25, "excess_above_95pct": .025},
                             "expected_weighted_support": {"phi": weight * .5, "positive_centered": weight * .25,
                                                           "excess_above_95pct": weight * .025}},
              "controls": summaries, "control_rows": control_rows, "skipped_controls": skipped,
              "control_component_recovery": "refitted continuous reference meta-analysis; other components retained from original control report",
              "by_cancer": {group: {cancer: summarize([r for r in rows if r["cancer"] == cancer])
                                     for cancer in sorted({r["cancer"] for r in rows})}
                            for group, rows in control_rows.items()},
              "corpus": summarize(corpus_rows), "corpus_rows": corpus_rows,
              "input_hashes": {str(p.relative_to(cfg.paths.root)): sha256_file(p)
                               for p in (controls_path, gt_path, cfg.paths.configs / "scoring.yaml")},
              "recommendation": "Retain the current formula pending preregistered, outcome-permuted null controls and independent validation; passing these exploratory controls is not acceptance."}
    write_json(cfg.paths.reports / "M2_null_analysis.json", report)
    lines = ["# M2 — null baseline and alternative support statistics", "",
             "This is an offline sensitivity analysis. Production scoring, labels, thresholds and the original control results are unchanged.", "",
             "## Correction to the earlier '0.25 floor' explanation", "",
             "At z=0, 0.5*Phi(z)=0.25. This is a baseline, not a pointwise lower bound: negative z gives a smaller value.",
             "Under an ideal calibrated null Z~N(0,1), Phi(Z) is uniform, so its expected weighted contribution is 0.25.",
             "A finite negative-control median can be below 0.2; the existing failed median does not prove that passing is mathematically impossible.",
             "Random expressed genes are not guaranteed null genes, and survival/meta-analysis statistics need not follow the ideal normal null.", "",
             "## Empirical comparison (same controls and fixed other components)", "",
             "| Support statistic | Positive median (>0.6) | Negative median (<0.2) | Ideal null expected weighted support |",
             "|---|---:|---:|---:|"]
    for name in VARIANTS:
        pos, neg = summaries["positive"][name]["median"], summaries["negative"][name]["median"]
        lines.append(f"| {report['formulae'][name]} | {pos:.4f} | {neg:.4f} | {report['ideal_null']['expected_weighted_support'][name]:.4f} |")
    lines += ["", "The centered alternative is zero at z=0, but clipping negative values creates a positive null expectation (0.125 after weighting), not zero.",
              "The 95% excess alternative is a stronger nonlinear filter; it is not a probability of biological truth and may suppress genuine weak effects.",
              f"Reference effects were refitted for {len(control_rows['positive'])} positive and {len(control_rows['negative'])} negative controls using dev cohorts only.",
              "Only the independent-support term is replaced. The multiverse and shrinkage remainder is retained from the stored controls and checked for feasible bounds.",
              f"For {len(corpus_rows)} eligible corpus results, the current robustness formula was reconstructed exactly from stored components before comparison.",
              "", "## Recommendation", "",
              "Keep the current production formula for now. Treat the centered statistic as a candidate requiring a preregistered comparison on outcome-permuted dev controls and independent cohorts.",
              "Do not choose a formula or relax the <0.2 threshold merely because it passes these already-seen controls. The alternatives are exploratory, not new acceptance results.",
              "Inflation and inflation_z do not use robustness, so the M4 binary labels remain unchanged under these support-only alternatives; M8 robustness weights would change. The current M6 reward uses inflation and detector flags, so this support-only substitution would not directly change its reward formula.",
              "Control uncertainty, between-cancer dependence, and one-snapshot stability remain unresolved. No claim of calibrated null FPR follows from these summaries.",
              "Full per-gene results and per-cancer summaries are in M2_null_analysis.json."]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.linspace(-3, 3, 301)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name in VARIANTS:
            axes[0].plot(z, [weight * support_variants(x)[name] for x in z], label=name)
        axes[0].set(xlabel="Reference effect z", ylabel="Weighted independent support", title="Support term (weight 0.5)")
        axes[0].legend(fontsize=8)
        labels = [f"{group}\n{name}" for group in ("positive", "negative") for name in VARIANTS]
        data = [[r["robustness"][name] for r in control_rows[group]] for group in ("positive", "negative") for name in VARIANTS]
        axes[1].boxplot(data, labels=labels)
        axes[1].tick_params(axis="x", labelsize=7, rotation=30)
        axes[1].axhline(.2, color="gray", linestyle=":")
        axes[1].axhline(.6, color="gray", linestyle=":")
        axes[1].set(ylabel="Robustness", title="Same real-data controls; exploratory")
        fig.tight_layout()
        fig.savefig(cfg.paths.reports / "M2_null_analysis.svg")
        plt.close(fig)
        lines += ["", "![Support statistic comparison](M2_null_analysis.svg)"]
    except ImportError:
        lines += ["", "Plot omitted: optional matplotlib dependency is unavailable."]
    (cfg.paths.reports / "M2_null_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({k: {n: s["median"] for n, s in v.items()} for k, v in summaries.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
