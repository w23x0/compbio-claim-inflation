# Project status — after the 2026-11-02 audit

This file replaces the previous STATUS. `docs/archive/freeze-2026-11-01.json` / `.sha256` remain
the manifest of the **2026-11-01 snapshot**; they are historical and are not
re-frozen until P4 v2 has produced numbers.

## One paragraph

The 2026-11-01 P4 result (test AUC 0.453) is not evidence that the content
monitor cannot work: the batch judge read only the first 6,000 characters of
each manuscript, i.e. abstract + introduction, while the checklist asks about
methods, results and data availability. The input was rebuilt from titled JATS
sections, the checklist now returns eight fixed red flags with a deterministic
score, and the evidence threshold follows the spec's 5% FPR. A 40-claim pilot
of the new input scored AUC 0.547 (rubric) / 0.612 (holistic) — above the old
0.453/0.483, but with 20 positives the interval is ~±0.10, so it is not a
result. The full 391-claim rerun needs the provider API, which is only open
inside operator-opened windows; the run has not completed.

## What changed in this pass (all local, no API spend)

| Change | Evidence |
|---|---|
| Section-aware monitor input (`src/cbi/m4/monitor_doc.py`, `PMCArticle.body_sections`) | 391 docs, mean 12,273 chars; methods/results/data-availability present |
| Fixed 8-key red-flag rubric with offline recomputable score (`llm_judge.py`) | `tests/test_monitor_doc.py` |
| Evidence threshold calibrated at 5% FPR (was 1%) | `train_llm.py` |
| Evidence variant (rubric vs holistic) chosen on validation only | `train_llm.py` |
| Score cache persisted per finished request; preflight aborts on a closed window | `_fill_cache`, `llm_ping`; `tests/test_p4_cache.py`, `tests/test_llm_ping.py` |
| New detector written to `models/detector_llm_v2/`; consumers resolve the newest | `active_detector_dir()` |
| M2 acceptance controls implemented and run | `reports/M2_controls.md` |
| M1 synthetic-fixture harness implemented | `scripts/p1_fixture_audit.py` (unrun: needs LLM) |
| P1 audit denominator fixed; P3 report stale strings fixed | `src/cbi/m1/public_audit.py`, `scripts/report_p3.py` |

## Acceptance status (honest)

| Stage | State | Evidence / gap |
|---|---|---|
| P0 | Partial | KIRC has 4 cohorts; spec asks 5–10 |
| P1 | **Not accepted** | `P1_metrics.json` did not exist until now; fixture harness unrun; yield 13.2% (spec flags <20%); audit 531/1020 = 52.1% |
| P2 | **Not accepted** | positive control passes (0.870), negative control fails (0.278; the weighted Φ zero-effect baseline is 0.25, not a hard floor); exploratory alternatives are in `M2_null_analysis.md`; inflation sensitivity passes; cross-snapshot test not evaluable (one snapshot); in-pool reproduction 52/217 = 24% across all evaluated in-pool claims |
| P3 | Pass (formal) | 496/496 coverage, but `support_links=0` ⇒ `effective_independent_support` null for all claims |
| P4 | **Pending** | v1 AUC 0.453 is invalid as a ceiling; v2 not scored; 40-claim pilot 0.547/0.612 |
| P5 | Pass | 7/7 isolation checks (independent of detector quality) |
| P6 | **Invalid** | ran on the v1 near-chance detector; blind-spot claims are noise |
| P7 | Implemented; live acceptance pending | Three-cycle control flow, rejection, holdout separation and interruption tested offline; requires valid P4 v2, matching P6 logs and API for real results |
| P8 | Stale | 34-claim example; rerun after P4 v2 |

## Tests

`python3 -m pytest -q` → 72 passed, 1 skipped.

## Offline preparation and scoring fixes (2026-11-02)

`python3 -m cbi.cli p4 --prepare-only` now builds/reuses monitor documents and
reports split/label counts and cache readiness without contacting the API.
`reports/P4_preflight.json` confirms 391 documents/context records, 58/128/205
train/validation/test claims, and 0/391 cached v2 scores. This is preparation,
not P4 acceptance. `p4 --offline` requires a complete valid cache and never
contacts the endpoint; it currently fails with an explicit missing-score count.

The saved detector now retains the validation-selected holistic/rubric score
and uses it consistently in single/batch inference. P4 reports separate test
ROC summaries from FPR/recall at validation-calibrated thresholds and correct
the rubric AUC display when holistic wins. Cache reuse respects the effective
evidence model, rejects invalid scores, preserves corrupt/incompatible files,
persists each parallel completion, and stops after an entirely failed worker
group. `status` reports the active LLM detector directory and v2 availability.

## Completed follow-up work (2026-11-02, no API)

- **P7:** honors cycle count, separates P4/candidate output directories, copies
  fusion state, uses numeric scores at each detector's own threshold, calibrates
  on eligible natural validation labels, and preserves family holdout across a
  model lineage. Accepted candidates are promoted by a baseline-bound pointer;
  rejected models are retained only for inspection. Progress is persisted per
  cycle, and empty/failed red rounds are not reported as completed cycles.
- **Cache provenance:** fingerprints cover source files, claim, document,
  prompt/parser source, rubric and request settings. The 391 monitor documents
  were refreshed from original XML with provenance recorded. Legacy/stale
  scores require rescore; existing results are archived before replacement.
- **Corpus audit:** no paper/text cross-split overlap or original/independent
  cohort overlap found. Eligible pool composition is 161 in-pool / 230
  out-of-pool, with 208 medium / 183 low confidence labels and no high-confidence
  labels. BRCA validation has just 3 negatives. See `reports/corpus_audit.md`.
- **M2 analysis:** 20 positive and 40 negative controls were compared with two
  alternative support terms using dev cohorts. Centered support yields medians
  0.8696/0.0518 versus current 0.8700/0.2782. These are exploratory results,
  not changed acceptance rules; production labels and formulas are untouched.

## Next command (needs an open API window)

```bash
export CBI_P4_WORKERS=8
python3 -m cbi.cli p4
```

The run preflights the endpoint, scores 391 claims one per request with
incremental persistence, and resumes from `p4_scores_v2.json` if interrupted.
Expected: ~10–20 min, ~$15–20 at the observed token prices.
