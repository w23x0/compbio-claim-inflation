"""One red-team worker (M5).  LLM planner chooses the strategy family; the
per-report analysis executor is deterministic and every number is reproducible.

The worker never receives eval/held-out data.  When launched by the orchestrator
it runs under bubblewrap with those directories masked and only dev cohorts
mounted.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config, load_config
from cbi.cohort_pool.loader import CohortStore
from cbi.m5.claim_factory import make_claim
from cbi.m2.analysis import CohortAnalyzer, align_sign
from cbi.m2.meta import meta_random_effects
from cbi.m2.specs import AnalysisSpec, CohortEffect, default_spec
from cbi.m4.detector import Detector
from cbi.m4.llm_judge import llm_checklist_score_many
from cbi.m5.llm_agent import choose_family
from cbi.m5.prereg import build_prereg
from cbi.m5.strategies import FAMILIES, candidate_specs, endpoint_candidates, family
from cbi.schemas import Claim, Report, Submission
from cbi.textgen import locate_evidence_span, render_manuscript
from cbi.util import (append_jsonl, atomic_write_text, ensure_watermark, now_iso,
                      sha256_text, write_json)


@dataclass
class Candidate:
    claim: Claim
    manuscript: str
    spec: AnalysisSpec
    endpoint: str
    style: str
    evidence_score: float
    flagged: bool
    proxy_inflation: float
    utility: float


def _proxy_inflation(cfg: Config, analyzer: CohortAnalyzer, *, cancer: str, gene: str,
                     endpoint: str, direction: str, contrast: str, paper_cid: str) -> float:
    other = [c for c in analyzer.store.cohort_ids(cancer=cancer, role="dev") if c != paper_cid]
    if not other:
        return 0.0
    spec = default_spec(contrast)
    effects = analyzer.effects_for_cohorts(other, gene, spec, endpoint)
    if not effects:
        return 0.0
    try:
        m = meta_random_effects(
            np.asarray([align_sign(direction) * e.log_effect for e in effects]),
            np.asarray([e.se for e in effects]),
        )
        mu = m.pooled
    except ValueError:
        mu = float(np.median([align_sign(direction) * e.log_effect for e in effects]))
    # The report's r is available from the direction-aligned claim; caller
    # overwrites this placeholder after constructing the claim.
    return mu


def _report_id(submission_id: str, index: int) -> str:
    return f"{submission_id}_s{index:03d}"


def _code_for_claim(chunk: str, cohort_id: str, gene: str, endpoint: str, spec: AnalysisSpec,
                    watermark: str) -> str:
    spec_obj = asdict(spec)
    return ensure_watermark(
        f'''"""Auto-generated red-team analysis code (watermarked)."""
import json, os
from cbi.m5.reproduce import reproduce

SPEC = {spec_obj!r}

if __name__ == "__main__":
    result = reproduce({cohort_id!r}, {gene!r}, {endpoint!r}, SPEC, root=os.environ.get("CBI_ROOT"))
    print(json.dumps(result, sort_keys=True))
''',
        marker="#",
    )


def run_worker(cfg: Config, job: dict[str, Any]) -> Submission:
    rng = np.random.default_rng(int(job["seed"]))
    store = CohortStore(cfg, allowed_roles={"dev"})
    store.assert_dev_only()
    analyzer = CohortAnalyzer(cfg, store)
    detector = Detector.load(cfg, job["detector_dir"])
    artifact_dir = Path(job["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "worker_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    trace: list[dict[str, Any]] = []

    def log_event(action: str, **data: Any) -> None:
        event = {"action": action, "timestamp": now_iso(), **data}
        trace.append(event)
        append_jsonl(trace_path, event, lock=False)

    fam_name, agent_rationale = choose_family(
        job.get("log_entries", []), float(job.get("family_quota", 0.30)), job.get("family"))
    fam = family(fam_name)
    submission_id = job["submission_id"]
    round_no, worker_no = int(job["round"]), int(job["worker"])
    k_reports = int(job.get("reports_per_batch", 20))
    log_event("agent_decision", family=fam_name, rationale=agent_rationale,
              submission_id=submission_id)
    cancers = list(cfg.cancers)
    dev_ids = {c: store.cohort_ids(cancer=c, role="dev") for c in cancers}
    all_dev = [cid for ids in dev_ids.values() for cid in ids]
    prereg, prereg_text, prereg_hash = build_prereg(
        submission_id=submission_id, round=round_no, worker=worker_no, family=fam_name,
        choices={"registry_choices": fam.registry_choices, "styles": fam.styles,
                 "n_comparisons": fam.n_comparisons, "agent_rationale": agent_rationale},
        target_cancers=cancers, target_cohorts=all_dev,
        motivation=fam.motivation, reports_planned=k_reports,
    )
    atomic_write_text(artifact_dir / "prereg.txt", ensure_watermark(prereg_text, marker="#"))
    log_event("prereg_frozen", hash=prereg_hash)
    max_candidates = int(job.get("max_candidate_analyses", 120))
    min_samples = int(cfg.scoring["gt"]["min_samples_per_cohort"])
    min_events = int(cfg.scoring["gt"]["min_events_per_cohort"])
    candidates: dict[tuple[str, str], Candidate] = {}
    attempts = 0
    while attempts < max_candidates:
        attempts += 1
        cancer = cancers[int(rng.integers(0, len(cancers)))]
        paper_cid = dev_ids[cancer][int(rng.integers(0, len(dev_ids[cancer])))]
        meta = store.meta(paper_cid)
        genes = store.genes(paper_cid)
        gene = genes[int(rng.integers(0, len(genes)))]
        key = (cancer, gene)
        if key in candidates:
            continue
        specs = candidate_specs(fam_name, rng)
        endpoints = endpoint_candidates(fam_name, rng)
        best_for_gene: Candidate | None = None
        for spec in specs:
            for endpoint in endpoints:
                try:
                    effect = analyzer.effect(paper_cid, gene, spec, endpoint)
                except Exception:
                    effect = None
                if effect is None or abs(effect.log_effect) > 3.5:
                    continue
                if not math.isfinite(effect.se) or effect.se <= 0 or effect.se > 5:
                    continue
                raw = float(effect.log_effect)
                direction = "high_expression_worse" if raw >= 0 else "high_expression_better"
                year = 2018 + int(rng.integers(0, 6))
                pub = date(year, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
                style = fam.styles[int(rng.integers(0, len(fam.styles)))]
                claim = make_claim(
                    claim_id=_report_id(submission_id, 0), gene=gene, cancer=cancer,
                    paper_cid=paper_cid, spec=spec, endpoint=endpoint, effect=effect,
                    meta=meta, rng=rng, policy=fam_name, style=style, pub_date=pub,
                    journal="Journal of Synthetic Oncology", n_comparisons=fam.n_comparisons,
                    code_available=fam.code_available,
                )
                manuscript = render_manuscript(claim, style=style, watermark=True,
                                               dataset_n=int(meta.get("n_deposited") or meta.get("n") or effect.n))
                span = locate_evidence_span(claim, manuscript)
                if span is not None:
                    from cbi.schemas import EvidenceSpan
                    claim = claim.model_copy(update={"evidence_span": EvidenceSpan(
                        start=span[0], end=span[1], sentence_hash=sha256_text(manuscript[span[0]:span[1]]))})
                mu = _proxy_inflation(cfg, analyzer, cancer=cancer, gene=gene, endpoint=endpoint,
                                      direction=direction, contrast=claim.reported_choices.contrast_type,
                                      paper_cid=paper_cid)
                proxy = float(math.tanh(max(0.0, claim.reported_log_effect - mu) / float(cfg.scoring["inflation_scale"])))
                if getattr(detector, "backend", "checklist") == "llm":
                    # Defer expensive LLM scoring to a parallel pass after the
                    # candidate pool is built.
                    det_score, det_flag = 0.5, False
                else:
                    det = detector.score(claim, manuscript, neutral=True)
                    det_score, det_flag = float(det.evidence_score), bool(det.flag)
                utility = proxy - 0.35 * det_score - (2.0 if det_flag else 0.0)
                cand = Candidate(claim=claim, manuscript=manuscript, spec=spec, endpoint=endpoint,
                                 style=style, evidence_score=det_score,
                                 flagged=det_flag, proxy_inflation=proxy, utility=float(utility))
                log_event("candidate", cancer=cancer, cohort=paper_cid, gene=gene,
                          spec=spec.spec_id, endpoint=endpoint, style=style,
                          evidence_score=cand.evidence_score, flagged=cand.flagged,
                          reported_hr=claim.reported_effect.value, proxy=proxy)
                if best_for_gene is None or cand.utility > best_for_gene.utility:
                    best_for_gene = cand
        if best_for_gene is not None:
            candidates[key] = best_for_gene
    if getattr(detector, "backend", "checklist") == "llm" and candidates:
        score_budget = min(len(candidates), max(k_reports * 2, 12))
        to_score = sorted(candidates.values(), key=lambda c: c.proxy_inflation, reverse=True)[:score_budget]
        batch_size = max(1, int(os.environ.get("CBI_M5_SCORE_BATCH", "4")))
        chunks = [to_score[i:i + batch_size] for i in range(0, len(to_score), batch_size)]

        def _score_chunk(chunk: list[Candidate]) -> list[dict | None]:
            try:
                return llm_checklist_score_many(
                    [(cand.claim, cand.manuscript) for cand in chunk],
                    learned_examples=detector.learned_examples)
            except Exception:
                return [None] * len(chunk)

        with ThreadPoolExecutor(max_workers=int(os.environ.get("CBI_M5_SCORE_WORKERS", "2"))) as pool:
            nested = list(pool.map(_score_chunk, chunks))
        judgements = [row for group in nested for row in group]
        for cand, judgement in zip(to_score, judgements):
            if judgement is None:
                # Missing batch entry: retry singly, never score it as zero.
                det = detector.score(cand.claim, cand.manuscript, neutral=True)
                cand.evidence_score = float(det.evidence_score)
                cand.flagged = bool(det.flag)
            else:
                cand.evidence_score = detector.judgement_score(judgement)
                cand.flagged = bool(cand.evidence_score >= detector.fusion.evidence_threshold)
            cand.utility = float(cand.proxy_inflation - 0.35 * cand.evidence_score - (2.0 if cand.flagged else 0.0))
            log_event("llm_score", gene=cand.claim.entities.gene, cancer=cand.claim.entities.cancer_type,
                      evidence_score=cand.evidence_score, flagged=cand.flagged)
    ranked = sorted(candidates.values(), key=lambda c: (c.flagged, -c.utility, c.evidence_score))
    selected = ranked[:k_reports]
    if len(selected) < k_reports:
        log_event("short_batch", requested=k_reports, selected=len(selected))
    reports: list[Report] = []
    for i, cand in enumerate(selected):
        rid = _report_id(submission_id, i)
        claim = cand.claim.model_copy(update={"claim_id": rid})
        manuscript = cand.manuscript
        span = locate_evidence_span(claim, manuscript)
        if span is not None:
            from cbi.schemas import EvidenceSpan
            claim = claim.model_copy(update={"evidence_span": EvidenceSpan(
                start=span[0], end=span[1], sentence_hash=sha256_text(manuscript[span[0]:span[1]]))})
        md_path = artifact_dir / "manuscripts" / f"{rid}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(md_path, manuscript)
        code = _code_for_claim(rid, cand.claim.reported_cohort.accessions[0], claim.entities.gene,
                               cand.endpoint, cand.spec, "")
        code_path = artifact_dir / "code" / f"{rid}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(code_path, code)
        reports.append(Report(
            report_id=rid,
            claim=claim,
            manuscript_md_uri=f"manuscripts/{rid}.md",
            analysis_code_uri=f"code/{rid}.py",
            code_sha256=sha256_text(code),
            declared_data=list(claim.reported_cohort.accessions),
            declared_choices=claim.reported_choices,
            reported_numbers={
                "HR": float(claim.reported_effect.value),
                "ci_low": float(claim.reported_effect.ci_low or 0.0),
                "ci_high": float(claim.reported_effect.ci_high or 0.0),
                "p": float(claim.reported_effect.p or 0.0),
                "log_effect": float(claim.reported_log_effect),
            },
        ))
    parents = [e["submission_id"] for e in job.get("log_entries", [])
               if e.get("strategy_family") == fam_name]
    parent_ids = [max(parents, key=lambda sid: next(float(e.get("score", 0)) for e in job["log_entries"] if e["submission_id"] == sid))] if parents else []
    mean_score = float(np.mean([c.evidence_score for c in selected])) if selected else 0.0
    summary = f"{fam_name}: {len(reports)} reports; mean evidence {mean_score:.2f}"
    submission = Submission(
        submission_id=submission_id, round=round_no, worker=worker_no, strategy_family=fam_name,
        parent_ids=parent_ids, preregistration_text=prereg_text,
        preregistration_hash=prereg_hash, reports=reports,
        worker_trace_uri="worker_trace.jsonl", created_at=now_iso(),
        declared_choices={"family": fam_name, "registry_choices": fam.registry_choices,
                          "styles": fam.styles},
    )
    write_json(artifact_dir / "submission.json", submission.model_dump(mode="json"))
    write_json(artifact_dir / "summary.json", {
        "submission_id": submission_id, "family": fam_name, "n_reports": len(reports),
        "mean_evidence": mean_score, "n_flagged": sum(c.flagged for c in selected),
        "mean_proxy": float(np.mean([c.proxy_inflation for c in selected])) if selected else 0.0,
        "attempts": attempts, "summary": summary,
    })
    return submission


def _main() -> int:
    raw = os.environ["CBI_WORKER_JOB"]
    if os.path.isfile(raw):
        job = json.loads(open(raw, encoding="utf-8").read())
    else:
        job = json.loads(raw)
    cfg = load_config(job.get("root"))
    run_worker(cfg, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
