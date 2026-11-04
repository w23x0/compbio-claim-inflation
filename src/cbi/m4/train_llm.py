"""P4 with the MD content monitor: LLM evidence_score + small context prior."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from cbi.config import Config, ensure_layout
from cbi.m4.cache import input_fingerprints, prepare_documents
from cbi.m4.context_prior import ContextPrior, Fusion, context_vector
from cbi.m4.detector import Detector
from cbi.m4.llm_judge import (FLAG_KEYS, llm_checklist_score, llm_checklist_score_many,
                              llm_ping)
from cbi.m4.model import fpr_recall, threshold_at_fpr
from cbi.m4.train import _metrics, load_natural_corpus, labels_and_weights
from cbi.schemas import GroundTruthResult
from cbi.util import read_json, sha256_file, atomic_write_text, write_json


def _valid_cached_score(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    for key in ("score", "holistic"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not np.isfinite(value) or not 0 <= value <= 1:
            return False
    return (isinstance(row.get("flags"), dict)
            and isinstance(row.get("rationale"), list)
            and all(isinstance(x, str) for x in row["rationale"]))


def _operating_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Evaluate an already calibrated threshold without fitting on test labels."""
    fpr, recall = fpr_recall(y, scores, threshold)
    return {"threshold": float(threshold), "fpr": fpr, "recall": recall,
            "flagged": int(np.sum(scores >= threshold)), "n": len(y)}


def combine_with_context(ids: list[str], evidence: list[float], contexts: dict,
                         ctx_model, fusion) -> tuple[np.ndarray, np.ndarray]:
    """Return (context_only, combined) predictions aligned with ``ids``.

    A test-time combined score must use the validation-learned fusion weights.
    Keeping this in one function avoids re-using the raw evidence vector, which
    made the earlier ablation identical by construction.
    """
    context_scores: list[float] = []
    combined_scores: list[float] = []
    for cid, ev in zip(ids, evidence):
        if cid in contexts:
            vec, cov = context_vector(contexts[cid])
            cp = float(ctx_model.predict(vec, cov))
            context_scores.append(cp)
            combined_scores.append(float(fusion.combine(float(ev), cp, cov)))
        else:
            context_scores.append(float(ctx_model.base_rate))
            combined_scores.append(float(ev))
    return np.asarray(context_scores), np.asarray(combined_scores)


def _fill_cache(ids: list[str], cache: dict[str, dict], cache_path: Path,
                score_chunk, *, meta: dict[str, Any], workers: int = 1,
                batch_size: int = 1, waves: int = 4,
                deadline_seconds: float | None = None) -> None:
    """Score every missing id, persisting each finished request immediately.

    The LLM endpoint is only reachable inside short operator-opened windows, so
    a cancelled run must keep every score it already paid for.  Each completed
    request is merged into ``cache`` and flushed atomically immediately.
    ``deadline_seconds`` stops submitting new work; requests already in flight
    finish and are persisted before returning. An entirely failed worker group
    stops the run rather than retrying the rest of the corpus during an outage.
    """
    def flush() -> None:
        write_json(cache_path, {**meta, "claims": cache})

    started = time.monotonic()

    def out_of_time() -> bool:
        return deadline_seconds is not None and time.monotonic() - started >= deadline_seconds

    for _ in range(waves):
        missing = [cid for cid in ids if cid not in cache]
        if not missing or out_of_time():
            return
        chunks = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
        gained_total = 0
        while chunks and not out_of_time():
            gained_before = gained_total
            group = [chunks.pop(0) for _ in range(min(max(1, workers), len(chunks)))]
            if workers > 1 and len(group) > 1:
                with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                    futures = [pool.submit(score_chunk, chunk) for chunk in group]
                    try:
                        for future in as_completed(futures):
                            rows = future.result()
                            for cid, value in (rows.items() if isinstance(rows, dict) else rows):
                                cache[cid] = value
                            gained_total += len(rows)
                            if rows:
                                flush()
                    finally:
                        if gained_total:
                            flush()
            else:
                for chunk in group:
                    rows = score_chunk(chunk)
                    for cid, value in (rows.items() if isinstance(rows, dict) else rows):
                        cache[cid] = value
                    gained_total += len(rows)
                    if rows:
                        flush()
            if gained_total == gained_before:
                return
        if not gained_total:
            # A wave with no gain means the window is closed or every request
            # failed; retrying all 391 ids again only wastes wall time.
            return


def train_llm_detector(cfg: Config, *, workers: int = 8, prepare_only: bool = False,
                       offline: bool = False) -> tuple[Detector | None, dict[str, Any]]:
    ensure_layout(cfg)
    claims, split, contexts = load_natural_corpus(cfg)
    gt = {cid: GroundTruthResult.model_validate(x)
          for cid, x in read_json(cfg.paths.processed / "ground_truth.json").items()}
    eligible = {cid for cid, r in gt.items() if r.label_eligible and r.independent_cohorts}
    train_ids = [c.claim_id for c in claims if split[c.claim_id] == "train" and c.claim_id in eligible]
    val_ids = [c.claim_id for c in claims if split[c.claim_id] == "val" and c.claim_id in eligible]
    test_ids = [c.claim_id for c in claims if split[c.claim_id] == "test" and c.claim_id in eligible]
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError("LLM P4 requires non-empty eligible train/val/test splits")

    by_id = {c.claim_id: c for c in claims}
    ids = list(train_ids + val_ids + test_ids)
    selected_claims = [by_id[cid] for cid in ids]
    texts, sources, document_stats = prepare_documents(cfg, selected_claims)
    batch_size = max(1, int(os.environ.get("CBI_EVIDENCE_BATCH_SIZE", "1")))
    fingerprints = input_fingerprints(selected_claims, texts, sources, batch_size=batch_size)
    cache_path = cfg.paths.processed / "p4_scores_v2.json"
    cache_model = os.environ.get("CBI_EVIDENCE_MODEL") or os.environ.get("CBI_LLM_MODEL", "")
    cache_version = "monitor_docs_v2+flags_rubric_v1"
    cache: dict[str, dict] = {}
    cache_status = "absent"
    invalid_ids: list[str] = []
    stale_ids: list[str] = []
    if cache_path.is_file():
        try:
            cached = read_json(cache_path)
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Cannot read P4 cache {cache_path}; preserve and repair it before resuming") from exc
        if not isinstance(cached, dict) or not isinstance(cached.get("claims"), dict):
            raise RuntimeError(f"Invalid P4 cache structure: {cache_path}")
        # Offline work can identify the scoring model from the cache itself.
        if not cache_model and (prepare_only or offline):
            cache_model = cached.get("model", "")
        if cached.get("model") != cache_model or cached.get("version") != cache_version:
            cache_status = "incompatible"
        else:
            cache_status = "compatible"
            for cid in ids:
                if cid in cached["claims"]:
                    row = cached["claims"][cid]
                    if not _valid_cached_score(row):
                        invalid_ids.append(cid)
                    elif row.get("input_fingerprint") != fingerprints[cid]:
                        stale_ids.append(cid)
                    else:
                        cache[cid] = row

    missing = [cid for cid in ids if cid not in cache]
    if prepare_only:
        labels, _ = labels_and_weights(gt, cfg)
        label_map = dict(zip(gt, labels))
        report = {
            "mode": "offline_preparation", "api_requests": 0,
            "snapshot": cfg.snapshot, "eligible_claims": len(ids),
            "documents": len(texts), "document_provenance": document_stats,
            "mean_doc_chars": float(np.mean([len(text) for text in texts.values()])),
            "splits": {name: {"n": len(group), "positive": int(sum(label_map[c] for c in group)),
                               "negative": int(sum(1 - label_map[c] for c in group))}
                       for name, group in (("train", train_ids), ("val", val_ids), ("test", test_ids))},
            "context_records": sum(cid in contexts for cid in ids),
            "cache_path": str(cache_path), "cache_status": cache_status,
            "cache_model": cache_model or None, "cached_scores": len(cache),
            "invalid_cached_ids": invalid_ids, "stale_cached_ids": stale_ids,
            "missing_scores": len(missing),
            "missing_ids": missing, "ready_for_offline_training": not missing,
        }
        write_json(cfg.paths.reports / "P4_preflight.json", report)
        return None, report

    if cache_status == "incompatible":
        raise RuntimeError("P4 cache model/version mismatch; archive the existing cache before scoring with a different model/version")
    if missing and offline:
        raise RuntimeError(f"Offline P4 requires complete valid scores: {len(cache)}/{len(ids)} cached; no API requests were sent")
    if missing and not llm_ping():
        raise RuntimeError("LLM endpoint unavailable (API window closed); no P4 scoring requests were sent")

    # Preserve paid results before replacing invalid/unverifiable rows. A legacy
    # cache without input fingerprints is stale, never silently trusted.
    if missing and cache_path.is_file() and (stale_ids or invalid_ids):
        archive = cache_path.with_name(cache_path.stem + ".archive-" + sha256_file(cache_path)[7:19] + ".json")
        if not archive.exists():
            atomic_write_text(archive, cache_path.read_text(encoding="utf-8"))

    def score_chunk(chunk: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        try:
            judgements = llm_checklist_score_many(
                [(by_id[cid], texts[cid]) for cid in chunk])
        except Exception as exc:
            # A 5xx batch is a service outage, not four independent bad items.
            if "503" in str(exc) or "429" in str(exc):
                return {}
            judgements = [None] * len(chunk)
        for cid, judgement in zip(chunk, judgements):
            if judgement is None:
                try:
                    judgement = llm_checklist_score(by_id[cid], texts[cid])
                except Exception:
                    continue
            out[cid] = {"input_fingerprint": fingerprints[cid],
                        "score": float(judgement["score"]),
                        "holistic": float(judgement.get("holistic", judgement["score"])),
                        "flags": dict(judgement.get("flags") or {}),
                        "rationale": list(judgement["rationale"])}
        return out

    deadline = float(os.environ.get("CBI_P4_DEADLINE", "0") or 0) or None
    _fill_cache(ids, cache, cache_path, score_chunk,
                meta={"model": cache_model, "version": cache_version},
                workers=workers, batch_size=batch_size, deadline_seconds=deadline)
    missing = [cid for cid in ids if cid not in cache]
    if missing:
        raise RuntimeError(
            f"P4 scored {len(ids) - len(missing)}/{len(ids)} claims and stopped "
            f"(window deadline or endpoint failure); rerun to resume, "
            f"first missing: {missing[:5]}")
    scores = {cid: float(cache[cid]["score"]) for cid in ids}
    holistic_scores = {cid: float(cache[cid].get("holistic", cache[cid]["score"])) for cid in ids}
    rationales = {cid: list(cache[cid].get("rationale") or []) for cid in ids}
    y, w = labels_and_weights(gt, cfg)
    ymap = dict(zip(gt.keys(), y))
    wmap = dict(zip(gt.keys(), w))
    ytr = np.asarray([ymap[c] for c in train_ids])
    yval = np.asarray([ymap[c] for c in val_ids])
    yte = np.asarray([ymap[c] for c in test_ids])
    wtr = np.asarray([wmap[c] for c in train_ids])
    tr_scores = np.asarray([scores[c] for c in train_ids])
    val_scores = np.asarray([scores[c] for c in val_ids])
    te_scores = np.asarray([scores[c] for c in test_ids])
    te_rubric = te_scores.copy()
    val_holistic = np.asarray([holistic_scores[c] for c in val_ids])
    te_holistic = np.asarray([holistic_scores[c] for c in test_ids])
    # Pick the evidence score on validation only: the deterministic red-flag
    # rubric and the model's holistic score are both stored for every claim.
    val_auc_rubric = _metrics(yval, val_scores).get("auc")
    val_auc_holistic = _metrics(yval, val_holistic).get("auc")
    evidence_variant = "holistic" if (val_auc_holistic or 0.0) > (val_auc_rubric or 0.0) else "rubric"
    if evidence_variant == "holistic":
        tr_scores = np.asarray([holistic_scores[c] for c in train_ids])
        val_scores = val_holistic
        te_scores = te_holistic

    # Small model: context_prior only, trained on natural train labels.
    pairs = [(contexts[c], ymap[c], wmap[c]) for c in train_ids if c in contexts]
    if pairs:
        vec = np.asarray([context_vector(c)[0] for c, _, _ in pairs])
        cy = np.asarray([yi for _, yi, _ in pairs])
        cw = np.asarray([wi for _, _, wi in pairs])
        ctx_model = ContextPrior.fit(vec, cy, sample_weight=cw, seed=int(cfg.mvp["seed"]))
    else:
        ctx_model = ContextPrior(model=None, base_rate=float(np.average(ytr, weights=wtr)), feature_count=0)

    # Fusion weight learned on natural validation only.
    fusion = Fusion()
    best = None
    for slope in [0.5, 1.0, 2.0]:
        for wmax in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            comb = []
            for cid, ev in zip(val_ids, val_scores):
                if cid in contexts:
                    vec, cov = context_vector(contexts[cid])
                    cp = ctx_model.predict(vec, cov)
                    comb.append(min(wmax, slope * cov) * (cp - ev) + ev)
                else:
                    comb.append(ev)
            comb = np.asarray(comb)
            eps = 1e-9
            ll = float(-np.mean(yval * np.log(comb + eps) + (1 - yval) * np.log(1 - comb + eps)))
            key = (ll, wmax, slope)
            if best is None or key < best[0]:
                best = (key, slope, wmax, comb)
    _, slope, wmax, comb_val = best
    fusion.slope, fusion.max_weight = slope, wmax
    # Spec §M4: the evidence threshold is calibrated at FPR <= 5%.  The 1% point
    # is kept next to it so the stricter operating point stays reportable.
    fusion.evidence_threshold = float(threshold_at_fpr(yval, val_scores, 0.05))
    fusion.evidence_threshold_1pct = float(threshold_at_fpr(yval, val_scores, 0.01))
    fusion.combined_threshold = float(threshold_at_fpr(yval, comb_val, 0.05))
    # Test-time fusion must use the validation-learned weights, not the raw
    # evidence scores.  The previous version re-used ``te_scores`` for the
    # combined arm, which made the ablation identical by construction.
    ctx_test_arr, comb_test_arr = combine_with_context(
        test_ids, [float(x) for x in te_scores], contexts, ctx_model, fusion)
    fusion.metrics = {"val": _metrics(yval, comb_val),
                      "test": _metrics(yte, comb_test_arr)}
    detector = Detector(cfg, None, ctx_model, fusion, version="det-llm-2.0", backend="llm",
                        evidence_variant=evidence_variant)
    det_dir = cfg.paths.models / "detector_llm_v2"
    detector.save(det_dir)
    flag_rates = {
        key: float(np.mean([1.0 if cache[cid].get("flags", {}).get(key) is True else 0.0
                            for cid in ids]))
        for key in FLAG_KEYS
    }
    report = {
        "backend": "llm", "train_n": len(train_ids), "val_n": len(val_ids), "test_n": len(test_ids),
        "input": {"doc_version": cache_version,
                  "mean_doc_chars": float(np.mean([len(texts[cid]) for cid in ids]))},
        "positive_rate": {"train": float(ytr.mean()), "val": float(yval.mean()), "test": float(yte.mean())},
        "evidence_variant": evidence_variant,
        "evidence_val_auc_by_variant": {"rubric": val_auc_rubric, "holistic": val_auc_holistic},
        "evidence_val": _metrics(yval, val_scores), "evidence_test": _metrics(yte, te_scores),
        "evidence_rubric_test": _metrics(yte, te_rubric),
        "test_at_validation_threshold": {
            "evidence_5pct": _operating_metrics(yte, te_scores, fusion.evidence_threshold),
            "evidence_1pct": _operating_metrics(yte, te_scores, fusion.evidence_threshold_1pct),
            "combined_5pct": _operating_metrics(yte, comb_test_arr, fusion.combined_threshold),
        },
        "evidence_holistic_val": _metrics(yval, val_holistic),
        "evidence_holistic_test": _metrics(yte, te_holistic),
        "flag_rates": flag_rates,
        "combined_val": _metrics(yval, comb_val), "combined_test": _metrics(yte, comb_test_arr),
        "context_only_test": _metrics(yte, ctx_test_arr),
        "ablation": {
            "evidence_only_test": _metrics(yte, te_scores),
            "context_only_test": _metrics(yte, ctx_test_arr),
            "combined_test": _metrics(yte, comb_test_arr),
        },
        "test_predictions": {
            cid: {"label": int(y), "evidence": float(ev), "holistic": float(ho),
                  "context": float(cp), "combined": float(cb),
                  "flags": dict(cache[cid].get("flags") or {})}
            for cid, y, ev, ho, cp, cb in zip(test_ids, yte, te_scores, te_holistic,
                                               ctx_test_arr, comb_test_arr)
        },
        "fusion": fusion.to_dict(), "context_coverage": {
            "mean": float(np.mean([contexts[c].context_coverage for c in contexts])) if contexts else 0.0,
            "test_mean": float(np.mean([contexts[c].context_coverage for c in test_ids if c in contexts])) if contexts else 0.0,
        },
        "errors": 0, "rationale_examples": {cid: rationales[cid] for cid in test_ids[:3]},
    }
    write_json(cfg.paths.reports / "P4_metrics_v2.json", report)
    write_json(det_dir / "train_report.json", report)
    lines = ["# P4 — LLM content monitor, revision 2", "",
             f"- Train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}",
             f"- Input: section-aware monitor document ({report['input']['mean_doc_chars']:.0f} chars mean)",
             f"- Evidence variant (chosen on val): {evidence_variant}",
             f"- Red-flag rubric AUC: val {val_auc_rubric}, test {report['evidence_rubric_test'].get('auc')}",
             f"- Holistic-score AUC: val {report['evidence_holistic_val'].get('auc')}, "
             f"test {report['evidence_holistic_test'].get('auc')}",
             f"- Test recall at validation 5% FPR threshold: {report['test_at_validation_threshold']['evidence_5pct']['recall']}; "
             f"observed test FPR: {report['test_at_validation_threshold']['evidence_5pct']['fpr']}",
             "- The *_test recall_at_fpr metrics in JSON describe the test ROC curve; deployment uses validation thresholds above.",
             f"- Evidence threshold (5% FPR): {fusion.evidence_threshold:.4f}; "
             f"1% FPR: {fusion.evidence_threshold_1pct:.4f}",
             f"- Combined AUC: val {report['combined_val'].get('auc')}, "
             f"test {report['combined_test'].get('auc')}",
             f"- Context-only test AUC: {report['context_only_test'].get('auc')}",
             f"- Context coverage: {report['context_coverage']['mean']:.3f}",
             "- Red-flag rates: " + ", ".join(f"{k}={v:.2f}" for k, v in flag_rates.items()),
             "- Evidence backend: LLM; small model only context_prior.",
             "- Ablation uses the same test claims; missing context falls back to base rate."]
    (cfg.paths.reports / "P4_llm_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return detector, report
