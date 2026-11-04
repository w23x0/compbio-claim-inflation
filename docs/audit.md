# Audit — what went wrong before, with evidence

Written after reviewing the 2026-10-31 build session, the 2026-11-01 scale-up
and the resulting artifacts. `docs/archive/freeze-2026-11-01.*` describe the 2026-11-01 snapshot; this
file describes the defects found in it and what has been corrected since.

## 1. Previous-session process errors

1. **Deviated from the spec's primary path.** The content monitor was first
   built as a small checklist model instead of the LLM judge that §M4 makes
   primary. It was switched only after the operator objected.
2. **Fed the judge the wrong input.** The batched prompt truncated each
   manuscript to `CBI_EVIDENCE_BATCH_CHARS=6000`. For the sampled paper the
   Methods start at char 7,654, the HR results at 15,907 and the HR table at
   29,880 — the checklist questions were unanswerable from the supplied text.
3. **Continued after a negative result.** P5/P6 were run with the near-chance
   `det-llm-1.0`, and "best blind spot 0.448" was recorded as a deliverable.
4. **Documents contradicted each other.** `README.md:25` said P4 was not
   retrained (503); `docs/handoff.md` and `docs/status.md` said it was retrained;
   `docs/status.md` said it was not. `P3_context.md` still described a
   "34-claim corpus" and a "0.56 mean field coverage" next to the 0.540 figure.
5. **Misdiagnosed the next step** as "prompt/label alignment" instead of the
   input truncation.
6. **Spent $487.79 / 34.8M tokens** (`reports/session_usage.json`) while the
   central P4 artifact was invalid.

## 2. Acceptance defects that were marked "Done"

| Requirement | Actual |
|---|---|
| P1 synthetic-fixture regression, `reports/P1_metrics.json` | Did not exist; harness added in `scripts/p1_fixture_audit.py` (unrun) |
| P1 public audit pass rate ≥90% | 531/1020 = 52.1% (the report printed "531/531 (52.1%)") |
| P1 claim yield ≥20% | 13.2%, flagged "below 20%" but P1 marked done |
| M2 positive/negative controls, cross-snapshot stability | Never implemented; now `reports/M2_controls.md`: positive PASS 0.870, negative FAIL 0.278, sensitivity PASS 0.062 (p=1.8e-5), snapshots not evaluable |
| M2 in-pool reproduction ≥60% | 52/217 = 24% strict (146/217 = 67% counting partial); never reported |
| P0 5–10 cohorts per cancer | KIRC has 4 |
| P4 threshold at FPR ≤5% for evidence | v1 used 1% |
| Split by cancer and year | Year only; 58 of 391 eligible claims in train |
| P7 ≥3 blue cycles | `run_llm_cycle` runs one cycle |
| P8 current | 34-claim example from before the scale-up |
| Label quality | 230/391 labels out-of-pool (59%); confidence medium/low only, no `high` |

## 3. Errors made during this audit pass

1. Started the 391-claim rerun without an open window; the in-flight results
   were lost because the score cache was written only after a whole wave.
   Fixed: per-request persistence (`_fill_cache`) and a preflight probe
   (`llm_ping`) that aborts before sending any scoring request.
2. First version of the monitor-document builder could drop the data-availability
   block when Methods were long. Fixed by reserving that budget; covered by
   `tests/test_monitor_doc.py`.
3. A cache-merge bug (iterating a dict instead of `.items()`) wasted one full
   scoring pass. Fixed and covered by `tests/test_p4_cache.py`, whose fakes now
   return the real `dict[str, dict]` shape.
4. Treated the 40-claim pilot AUC as directional evidence. It is not a result:
   20 positives give roughly ±0.10.

## 4. What is still open

- P4 v2 scoring (needs the API window), then P5/P6/P8 reruns.
- P7 real multi-cycle acceptance (the loop is now implemented and tested offline).
- P0 KIRC cohort count.
- A decision on M2 support calibration. The earlier claim that "<0.2 cannot
  be met" was too strong: 0.5*Φ(0)=0.25 is a zero-effect baseline, not a hard
  floor. See `reports/M2_null_analysis.md`; retain the production formula until
  independently validated rather than tuning it to these observed controls.

## 5. Subsequent offline corrections (2026-11-02)

- **Training/inference mismatch:** P4 selected rubric versus holistic on
  validation, but `Detector` always used rubric at inference and did not save
  the choice. The choice now survives save/load and is used by single scoring,
  batch flags, M5 candidate scoring and M7 batch evaluation.
- **Metric presentation:** the rubric test AUC line displayed holistic AUC
  when holistic won. The report now retains both, and explicitly reports test
  recall/FPR at validation-derived thresholds instead of presenting test-ROC
  threshold selection as the deployed operating point.
- **Offline readiness:** preparation no longer requires an endpoint probe;
  `p4 --prepare-only` verified all 391 documents and context records, while
  recording that the formal v2 score cache is absent. `p4 --offline` refuses
  incomplete scores before network access and supports local calibration once
  the cache is complete.
- **Cache preservation:** model overrides and invalid rows are checked before
  reuse; damaged/incompatible caches are not silently overwritten. Parallel
  results are flushed after each completion and an entirely failed worker
  group stops the run. Regression suite: 56 passed, 1 skipped.

## 6. Four offline follow-ups completed (2026-11-02)

- P7 previously ignored `cycles`, shared mutable thresholds between candidate
  and baseline, wrote into P4's directory, compared boolean flags to a numeric
  threshold, and evaluated training examples as a family holdout. These are
  corrected. Fixed family exclusion persists across the accepted model lineage;
  validation calibrates thresholds, and the spec's natural-test FPR gate remains
  explicitly an adaptive diagnostic. Offline tests cover three cycles with a
  rejection, interrupted red work, separate directories and baseline integrity.
- Paid cache rows now carry content fingerprints. Source-backed documents are
  rebuilt on source/claim/builder changes; old rows without fingerprints are not
  silently accepted. Stale results are archived before successful rescoring can
  replace them. Full source provenance is available for all 391 current inputs.
- `reports/corpus_audit.md` records zero detected paper/text split overlaps and
  zero direct original/independent cohort overlaps. It also exposes 230/391
  out-of-pool labels, no high-confidence labels, and only three BRCA validation
  negatives. The 161 eligible in-pool count is a subset of the 217 previously
  evaluated in-pool claims, not a contradiction.
- `reports/M2_null_analysis.md` corrects the hard-floor explanation and compares
  current/centered/95%-excess support on the same controls. Centered support's
  positive/negative medians are 0.8696/0.0518, but its ideal weighted null mean
  remains 0.125 because of clipping. No production statistic, label or acceptance
  threshold was changed to make this comparison pass.
- Full suite: 72 passed, 1 skipped. No API calls or live P4/P7 results in this pass.
