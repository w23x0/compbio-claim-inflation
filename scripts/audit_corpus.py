"""Offline audit of paper separation and the provenance of natural labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbi.config import load_config
from cbi.util import read_json, sha256_file, sha256_text, write_json


def paper_tokens(claim):
    source = claim.get("source", {})
    tokens = []
    for key in ("pmcid", "pmid", "doi"):
        value = str(source.get(key) or "").strip().lower()
        if key == "doi":
            value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
        if value:
            tokens.append(f"{key}:{value}")
    title = re.sub(r"\W+", " ", str(source.get("title") or "").lower()).strip()
    if title:
        tokens.append("title:" + title)
    return tokens


def audit_corpus(claims, split, gt, pool, thresholds, document_hashes=None):
    """Identifiers are unioned transitively so aliases cannot hide paper overlap."""
    parents = list(range(len(claims)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    token_owner, no_identity = {}, []
    for i, claim in enumerate(claims):
        tokens = paper_tokens(claim)
        if not tokens:
            no_identity.append(claim["claim_id"])
        for token in tokens:
            if token in token_owner:
                parents[find(i)] = find(token_owner[token])
            token_owner[token] = i
    groups = defaultdict(list)
    for i, claim in enumerate(claims):
        groups[find(i)].append(claim["claim_id"])
    def bucket(cid):
        value = split.get(cid, {})
        return value.get("split") if isinstance(value, dict) else value
    overlaps = [{"claim_ids": sorted(set(ids)), "splits": sorted({bucket(cid) for cid in ids if bucket(cid)})}
                for ids in groups.values() if len({bucket(cid) for cid in ids if bucket(cid)}) > 1]
    docs = defaultdict(list)
    for cid, digest in (document_hashes or {}).items():
        docs[digest].append(cid)
    duplicate_docs = [{"claim_ids": ids, "splits": sorted({bucket(cid) for cid in ids if bucket(cid)})}
                      for ids in docs.values() if len({bucket(cid) for cid in ids if bucket(cid)}) > 1]
    mapping = {}
    for cohort in pool.get("cohorts", []):
        for alias in (cohort.get("accession"), cohort.get("cohort_id")):
            if alias:
                mapping[str(alias).upper()] = cohort["cohort_id"]
    rows, exclusions, direct_overlap, metadata_errors = [], Counter(), [], []
    for claim in claims:
        cid = claim["claim_id"]
        result = gt.get(cid)
        if result is None:
            exclusions["missing_ground_truth"] += 1
            continue
        if not result.get("label_eligible", True) or not result.get("independent_cohorts"):
            exclusions["no_eligible_independent_evidence"] += 1
            continue
        role = bucket(cid)
        if role not in {"train", "val", "test"}:
            exclusions["missing_or_invalid_split"] += 1
            continue
        cancer = claim["entities"]["cancer_type"]
        status = result["original_cohort_repro"]["status"]
        # Use the actual engine reproduction outcome, not M1's preliminary flag.
        pool_group = "out_of_pool" if status == "out_of_pool" else "in_pool"
        positive = (result["inflation_z"] >= thresholds["label_min_z"]
                    and result["inflation"] >= thresholds["label_min_inflation"])
        source_cohorts = {mapping[a.upper()] for a in claim.get("reported_cohort", {}).get("accessions", [])
                          if a.upper() in mapping}
        independent = {e["cohort_id"] for e in result["independent_cohorts"]}
        shared = source_cohorts & independent
        if shared:
            direct_overlap.append({"claim_id": cid, "cohorts": sorted(shared)})
        for key, expected in (("gene", claim["entities"]["gene"]), ("cancer_type", cancer),
                              ("endpoint", claim["endpoint"])):
            if result.get(key) != expected:
                metadata_errors.append({"claim_id": cid, "field": key, "claim": expected, "ground_truth": result.get(key)})
        rows.append({"claim_id": cid, "split": role, "cancer": cancer,
                     "year": int(claim["source"]["pub_date"][:4]), "positive": int(positive),
                     "pool": pool_group, "confidence": result["gt_confidence"],
                     "independent_k": len(independent), "reproduction": status,
                     "extraction_pool_flag": claim.get("reported_cohort", {}).get("in_standard_pool"),
                     "gene": claim["entities"]["gene"], "inflation": result["inflation"],
                     "inflation_z": result["inflation_z"]})
    def summary(selected):
        n = len(selected)
        positive = sum(r["positive"] for r in selected)
        return {"n": n, "positive": positive, "negative": n - positive,
                "positive_rate": positive / n if n else None,
                "confidence": dict(Counter(r["confidence"] for r in selected)),
                "reproduction": dict(Counter(r["reproduction"] for r in selected)),
                "independent_k": dict(Counter(str(r["independent_k"]) for r in selected))}
    stratified = {key: {value: summary([r for r in rows if str(r[key]) == value])
                        for value in sorted({str(r[key]) for r in rows})}
                  for key in ("split", "cancer", "pool", "confidence", "year")}
    cancer_split = {cancer: {role: summary([r for r in rows if r["cancer"] == cancer and r["split"] == role])
                              for role in ("train", "val", "test")}
                    for cancer in sorted({r["cancer"] for r in rows})}
    counts = Counter(c["claim_id"] for c in claims)
    genes = {role: {(r["gene"], r["cancer"]) for r in rows if r["split"] == role}
             for role in ("train", "val", "test")}
    return {"corpus_claims": len(claims), "eligible": summary(rows), "excluded": dict(exclusions),
            "duplicate_claim_ids": [cid for cid, n in counts.items() if n > 1],
            "paper_groups": len(groups), "paper_overlap": overlaps,
            "missing_paper_identity": no_identity, "document_overlap": duplicate_docs,
            "original_independent_cohort_overlap": direct_overlap,
            "label_metadata_mismatches": metadata_errors,
            "missing_split_ids": sorted(c["claim_id"] for c in claims if bucket(c["claim_id"]) not in {"train", "val", "test"}),
            "extra_split_ids": sorted(set(split) - set(counts)),
            "stratified": stratified, "cancer_by_split": cancer_split,
            "gene_cancer_overlap": {f"{a}/{b}": len(genes[a] & genes[b])
                                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
            "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    cfg = load_config(args.root)
    base = cfg.paths.processed / "corpus"
    claims_path, split_path, gt_path = base / "claims.json", base / "natural_split.json", cfg.paths.processed / "ground_truth.json"
    claims = read_json(claims_path)
    hashes, missing_docs = {}, []
    for c in claims:
        path = base / "manuscripts" / f"{c['claim_id']}.md"
        if path.is_file() and (text := path.read_text(encoding="utf-8").strip()):
            hashes[c["claim_id"]] = sha256_text(" ".join(text.split()))
        else:
            missing_docs.append(c["claim_id"])
    report = audit_corpus(claims, read_json(split_path)["by_claim"], read_json(gt_path),
                          cfg.cohort_pool, cfg.scoring["detector"], hashes)
    report.update(snapshot=cfg.snapshot, api_requests=0, missing_manuscripts=missing_docs,
                  input_hashes={str(p.relative_to(cfg.paths.root)): sha256_file(p)
                                for p in (claims_path, split_path, gt_path, cfg.paths.configs / "cohort_pool.yaml")})
    write_json(cfg.paths.reports / "corpus_audit.json", report)
    lines = ["# Natural corpus and label audit", "", f"- Snapshot: {cfg.snapshot}; API requests: 0.",
             f"- Corpus: {len(claims)} claims; eligible: {report['eligible']['n']}; exclusions: {report['excluded']}.",
             f"- Cross-split paper groups: {len(report['paper_overlap'])}; identical manuscript groups: {len(report['document_overlap'])}.",
             f"- Claims without paper identity: {len(report['missing_paper_identity'])}; missing manuscripts: {len(missing_docs)}.",
             f"- Original/independent cohort intersections: {len(report['original_independent_cohort_overlap'])}.",
             f"- Label metadata mismatches: {len(report['label_metadata_mismatches'])}.", "",
             "## Eligible label provenance", "", "| Group | N | Positive | Positive rate | Confidence | Reproduction |",
             "|---|---:|---:|---:|---|---|"]
    for key in ("pool", "split"):
        for value, s in report["stratified"][key].items():
            lines.append(f"| {key}: {value} | {s['n']} | {s['positive']} | {s['positive_rate']:.1%} | {s['confidence']} | {s['reproduction']} |")
    lines += ["", "## Cancer coverage by split", "", "| Cancer | Train (positive/negative) | Validation | Test |", "|---|---|---|---|"]
    for cancer, splits in report["cancer_by_split"].items():
        values = [f"{splits[r]['positive']}/{splits[r]['negative']}" for r in ("train", "val", "test")]
        lines.append(f"| {cancer} | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation and limits", "",
              "- In-pool/out-of-pool refers to the ground-truth engine's original-cohort reproduction outcome. M1's preliminary pool flag is not used to classify labels.",
              "- An out-of-pool label can still have independent evidence, but does not establish original-cohort reproduction. Report detector performance separately for these groups.",
              "- The existing year-based split is audited, not rewritten. Sparse or absent cancer/class cells limit subgroup conclusions.",
              f"- Gene/cancer overlap counts: {report['gene_cancer_overlap']}. This is reuse of biological targets, not by itself paper leakage; the current split does not measure unseen-gene generalization.",
              "- Identifier/title and exact-text checks cannot rule out paraphrases, author/lab overlap, or shared patients across distinct cohort accessions.",
              "- Ground-truth labels share a fixed dev cohort pool. This audit does not turn it into an independent clinical validation sample.",
              "- Full counts, claim IDs and source hashes are in corpus_audit.json; existing labels and splits are unchanged."]
    (cfg.paths.reports / "corpus_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"eligible={report['eligible']['n']}; paper_overlap={len(report['paper_overlap'])}; cohort_overlap={len(report['original_independent_cohort_overlap'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
