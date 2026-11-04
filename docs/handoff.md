# Handoff — 2026-11-02

**Status: idle.** No job is running. The previous handoff (`2026-11-01` freeze)
is superseded where this file says so; `docs/archive/freeze-2026-11-01.*` remains the snapshot manifest
of that date.

## Start here

```bash
cd compbio-claim-inflation
python3 -m pytest -q                      # 72 passed, 1 skipped
python3 -m cbi.cli status
python3 -m cbi.cli p4 --prepare-only       # no API; reports/P4_preflight.json

# needs an operator-opened API window:
export CBI_P4_WORKERS=8
# optional: stop cleanly after N seconds of the open window
export CBI_P4_DEADLINE=600
python3 -m cbi.cli p4
```

`p4` probes the endpoint (`llm_ping`) only when scores are missing; the probe is
an API request, while `--prepare-only` and `--offline` send none. A failed probe
aborts before scoring. It writes each completed result atomically to
`data/processed/2026-11-01/p4_scores_v2.json`, so an interrupted run resumes
where it stopped. 391 claims, roughly 10–20 minutes and $15–20.

## Latest local optimization (no API calls)

- `p4 --prepare-only` verified 391 monitor documents and 391 context records;
  train/val/test = 58/128/205. The formal `p4_scores_v2.json` is absent (0/391
  cached). See `reports/P4_preflight.json`.
- `p4 --offline` can calibrate/save from a complete valid cache without endpoint
  configuration. In the current state it exits with 0/391 cached and sends no
  request. Corrupt/incompatible caches are never silently overwritten.
- Fixed the validation-selected holistic/rubric variant being lost on model
  save/load and ignored during inference, including M5/M7 batch consumers.
- P4 reports now distinguish test ROC summaries from test recall/FPR at a
  validation-calibrated threshold; rubric AUC stays rubric AUC when holistic wins.
- Cache writes happen after each completed request even with parallel workers;
  an entirely failed worker group stops further submission. Deadlines stop new
  work and allow in-flight requests to finish.
- Tests: 72 passed, 1 skipped. No new production detector or live P4/P7
  performance result was generated.
- P4 caches now require content fingerprints (claim, XML/manuscript, monitor
  text, judge source/rubric and request settings). All 391 source-backed monitor
  documents were rebuilt once to record provenance; no source-less fallback was
  needed. Stale paid scores are archived before online replacement.
- P7 runs the requested cycles, isolates candidate directories, copies threshold
  state, evaluates fixed untouched families, and retains the last accepted model
  on rejection. Accepted candidates use a baseline-bound active pointer; a P4
  retrain invalidates it. Failed red rounds retain progress but do not count as
  completed cycles. Live runs require P4 v2 and matching P6 logs.
- Corpus audit: 0 paper/text cross-split overlaps and 0 original/independent
  cohort intersections. Of 391 eligible labels, 161 are in-pool and 230 out-of-pool;
  confidence is medium/low only. See `reports/corpus_audit.md`.
- M2 exploratory comparison: current positive/negative medians 0.8700/0.2782;
  positive-centered support 0.8696/0.0518. Production formulas and thresholds
  are unchanged. See `reports/M2_null_analysis.md`.

## What is true now

- P4 v1 (test AUC 0.453) is **not** a valid ceiling: its judge only saw the
  first 6,000 characters of each manuscript (abstract + introduction). See
  `docs/audit.md` §1.
- P4 v2 feeds a section-aware document (claim, abstract, methods, data
  availability, claim-relevant results, gene rows), scores eight fixed red
  flags plus a holistic score, calibrates at 5% FPR, and selects the evidence
  variant on validation. 40-claim pilot: AUC 0.547 rubric / 0.612 holistic —
  not a result, 20 positives give ±0.10.
- The 391 monitor documents are already built and cached in
  `data/processed/2026-11-01/corpus/monitor_docs_v2/`.
- M2 acceptance controls now exist and were run: `reports/M2_controls.md`.
  Positive controls pass (0.870); negative controls fail (0.278). The specified
  Φ term has a 0.25 weighted zero-effect baseline, not a hard floor; inflation sensitivity passes
  (0.062, p=1.8e-5); cross-snapshot stability is not evaluable (one snapshot).
- P1 fixture harness exists (`scripts/p1_fixture_audit.py`) but has not run; it
  is the missing `reports/P1_metrics.json` acceptance item.
- P5 7/7 was measured against the v1 detector; the checks themselves are
  detector-independent, but they must be re-run after P4 v2.
- P6 results (`best blind spot 0.448`) came from the v1 near-chance detector
  and must not be cited.
- P7 multi-cycle code is implemented and verified offline; real three-cycle
  acceptance remains pending the API and valid P4 v2/P6 artifacts.
- P8 is still the 34-claim example.

## What to do after P4 v2 finishes

```bash
cat reports/P4_llm_v2.md
cat reports/P4_metrics_v2.json
```

- If test recall at 5% FPR is clearly above the v1 0.035, rerun P5, P6 and P8
  with `models/detector_llm_v2/` (the pipeline resolves it automatically).
- If it is not, stop calling P4 "done": restrict labels to in-pool claims or
  improve the label source before spending on more red-teaming.
- Evaluate the Φ-based support alternatives on preregistered, outcome-permuted
  null controls and independent cohorts; current comparisons are exploratory.

## Warnings

- Do not rerun P1 on the 4,025 files without `--cache-dir
  data/processed/2026-11-01/p1_cache`; it would re-spend the extraction budget.
- Keep concurrency at 10–20; 30 workers caused a 503 retry storm.
- `secrets/llm.env` and `secrets/ncbi.env` hold exposed keys; rotate them and
  never commit `secrets/`.
- The API is only reachable inside operator-opened windows. A closed window
  returns 503; every call retries 5× with backoff, which is why an unguarded
  run can idle for 40 minutes. The preflight prevents that now.
