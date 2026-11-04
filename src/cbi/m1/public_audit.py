"""Automated public-PMC extraction audit (M1 v1.2).

Reads a directory of PMC OA XML files (fetched by `cbi.m1.pmc_fetch`), extracts
claims, and reports both claim yield and consistency-audit pass rate.  No human
annotation is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cbi.config import load_config
from cbi.llm_usage import reset as reset_usage, snapshot as usage_snapshot
from cbi.m1.audit import audit_claim
from cbi.m1.extractor import ClaimExtractor, LLMRequiredError
from cbi.m1.pmc import parse_pmc_xml
from cbi.util import read_json, write_json


def _cacheable(row: dict[str, Any]) -> bool:
    """Only persist deterministic negatives; never cache a transient 5xx."""
    if not row.get("error"):
        return True
    error = str(row["error"])
    return ("rule validation" in error or "ValidationError" in error
            or error.startswith("parse:"))


def _row(path: Path, article, claim, error: Exception | None) -> dict[str, Any]:
    if error is not None:
        return {"file": path.name, "pmcid": getattr(article, "pmcid", None),
                "error": f"extract:{type(error).__name__}:{error}"}
    if claim is None:
        return {"file": path.name, "pmcid": getattr(article, "pmcid", None), "extracted": False}
    issues = audit_claim(claim)
    return {
        "file": path.name, "pmcid": article.pmcid, "extracted": True,
        "gene": claim.entities.gene, "cancer": claim.entities.cancer_type,
        "endpoint": claim.endpoint, "direction": claim.direction,
        "hr": claim.reported_effect.value, "accessions": claim.reported_cohort.accessions,
        "issues": issues, "passed": not issues,
        "claim": claim.model_dump(mode="json"),
    }


def run_public_audit(cfg, xml_dir: str | Path, *, limit: int | None = None,
                     workers: int = 1, batch_size: int | None = None,
                     cache_dir: str | Path | None = None) -> dict[str, Any]:
    xml_dir = Path(xml_dir)
    paths = sorted(xml_dir.glob("*.xml"))
    if limit is not None:
        paths = paths[:limit]
    extractor = ClaimExtractor(cfg)
    reset_usage()
    size = max(1, int(batch_size if batch_size is not None
                      else os.environ.get("CBI_M1_BATCH_SIZE", "4")))

    parsed: list[tuple[Path, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            parsed.append((path, parse_pmc_xml(path)))
        except Exception as exc:
            parse_rows.append({"file": path.name, "error": f"parse:{type(exc).__name__}:{exc}"})

    cache_root = Path(cache_dir) if cache_dir else None
    cached_rows: list[dict[str, Any]] = []
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        pending = []
        for path, article in parsed:
            key = (article.pmcid or path.stem).replace("/", "_")
            cpath = cache_root / f"{key}.json"
            if cpath.is_file():
                try:
                    obj = read_json(cpath)
                except Exception:
                    obj = None
                if isinstance(obj, dict) and _cacheable(obj):
                    cached_rows.append(obj)
                    continue
            pending.append((path, article))
        parsed = pending

    def cache_row(path: Path, article, row: dict[str, Any]) -> None:
        if cache_root is None or not _cacheable(row):
            return
        key = (article.pmcid or path.stem).replace("/", "_")
        write_json(cache_root / f"{key}.json", row)

    if size == 1:
        def process_one(item: tuple[Path, Any]) -> dict[str, Any]:
            path, article = item
            try:
                claim = extractor.extract_article(article)
            except LLMRequiredError:
                raise
            except Exception as exc:
                return _row(path, article, None, exc)
            row = _row(path, article, claim, None)
            cache_row(path, article, row)
            return row
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(process_one, parsed))
        else:
            rows = [process_one(item) for item in parsed]
    else:
        chunks = [parsed[i:i + size] for i in range(0, len(parsed), size)]

        def process_chunk(chunk: list[tuple[Path, Any]]) -> list[dict[str, Any]]:
            try:
                results = extractor.extract_articles([a for _, a in chunk], batch_size=size)
            except LLMRequiredError:
                raise
            except Exception as exc:
                return [_row(path, article, None, exc) for path, article in chunk]
            out = []
            for (path, article), (_, claim, error) in zip(chunk, results):
                row = _row(path, article, claim, error)
                cache_row(path, article, row)
                out.append(row)
            return out
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                nested = list(pool.map(process_chunk, chunks))
            rows = [row for group in nested for row in group]
        else:
            rows = [row for chunk in chunks for row in process_chunk(chunk)]
    rows = parse_rows + cached_rows + rows

    parse_errors = sum(1 for r in rows if str(r.get("error", "")).startswith("parse:"))
    extract_errors = sum(1 for r in rows if str(r.get("error", "")).startswith("extract:"))
    extracted = sum(1 for r in rows if r.get("extracted"))
    audited = [r for r in rows if r.get("extracted")]
    passed = sum(1 for r in audited if r.get("passed"))
    issue_counts: dict[str, int] = {}
    for row in audited:
        for issue in row["issues"]:
            key = issue.split(":", 1)[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
    result = {
        "n_files": len(paths),
        "n_parsed": len(paths) - parse_errors,
        "parse_errors": parse_errors,
        "n_claims": extracted,
        "extract_errors": extract_errors,
        "claim_yield": float(extracted / len(paths)) if paths else 0.0,
        "audit_passed": passed,
        "audit_pass_rate": float(passed / (extracted + extract_errors)) if (extracted + extract_errors) else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
        "batch_size": size,
        "workers": workers,
        "cached_rows": len(cached_rows),
        "token_usage": usage_snapshot(),
        "rows": rows,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbi-public-pmc-audit")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dir", default="data/raw/pmc_real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="optional per-article cache for resumable runs")
    args = parser.parse_args(argv)
    cfg = load_config(args.root)
    try:
        result = run_public_audit(cfg, args.dir, limit=args.limit, workers=args.workers,
                                 batch_size=args.batch_size, cache_dir=args.cache_dir)
    except LLMRequiredError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    write_json(cfg.paths.reports / "P1_public_pmc.json", result)
    claims = [r["claim"] for r in result["rows"] if r.get("claim")]
    write_json(cfg.paths.reports / "P1_public_pmc_claims.json", claims)
    lines = [
        "# P1 — public PMC OA extraction audit", "",
        f"- Files: {result['n_files']}",
        f"- Parsed: {result['n_parsed']} (parse errors {result['parse_errors']})",
        f"- Claims extracted: {result['n_claims']} (yield {result['claim_yield']:.1%})",
        f"- Extraction/validation errors: {result['extract_errors']}",
        f"- Batch size: {result.get('batch_size')} (workers {result.get('workers')})",
        f"- Token usage: `{result.get('token_usage')}`",
        f"- Cached rows: {result.get('cached_rows', 0)}",
        f"- Coverage flag: {'below 20%' if result['claim_yield'] < 0.2 else 'ok'}",
        f"- Consistency audit: {result['audit_passed']}/{result['n_claims'] + result['extract_errors']} "
        f"extraction attempts passed ({result['audit_pass_rate']:.1%})",
        f"- Issue counts: `{result['issue_counts']}`", "",
        "This audit is automatic: schema, HGNC resolution, cancer mapping, GEO format,",
        "HR/CI/p consistency, direction/effect agreement, and evidence-span validity.",
        "It does not replace a semantic precision benchmark; it catches unusable claims.",
        "",
        "| PMC | gene | cancer | HR | issues |",
        "|---|---|---|---:|---|",
    ]
    for row in result["rows"]:
        if not row.get("extracted"):
            continue
        issues = ", ".join(row["issues"]) or "—"
        lines.append(f"| {row['pmcid']} | {row['gene']} | {row['cancer']} | {row['hr']:.3g} | {issues} |")
    (cfg.paths.reports / "P1_public_pmc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
