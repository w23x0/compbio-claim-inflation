# P8 — downstream example (§7.5)

Same literature-derived gene set per cancer, scored on held-out cohorts with uniform claim weights vs M8 claim weights. Higher C-index is better.

## Summary

- Claims: 34 across 3 cancers.
- Held-out cohorts evaluated: 5 of 6 (skipped 1).
- Mean C-index: unweighted **0.589**, weighted **0.600**.
- Mean paired difference (weighted − unweighted): **0.011**; cohort-bootstrap 95% CI [-0.013, +0.041].
- Descriptive non-inferiority flag (mean diff ≥ −0.01): **True**.

## Per cohort

| Cancer | Held-out cohort | n | events | genes | unweighted | weighted | diff | 95% CI (bootstrap) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BRCA | GSE42568 | 104 | 35 | 24 | 0.603 | 0.662 | 0.060 | [+0.017, +0.106] |
| BRCA | GSE58812 | 107 | 29 | 24 | 0.584 | 0.555 | -0.029 | [-0.082, +0.012] |
| LIHC | GSE76427 | 115 | 23 | 3 | 0.481 | 0.508 | 0.028 | [-0.028, +0.071] |
| LUAD | GSE13213 | 117 | 49 | 2 | 0.568 | 0.569 | 0.000 | [-0.008, +0.010] |
| LUAD | GSE31210 | 204 | 30 | 2 | 0.709 | 0.705 | -0.004 | [-0.010, +0.002] |

## Per cancer

| Cancer | Cohorts | Mean unweighted | Mean weighted | Mean diff |
|---|---|---:|---:|---:|
| BRCA | GSE42568, GSE58812 | 0.594 | 0.609 | 0.015 |
| LIHC | GSE76427 | 0.481 | 0.508 | 0.028 |
| LUAD | GSE13213, GSE31210 | 0.639 | 0.637 | -0.002 |

## Skipped

- KIRC / —: no extracted claims for this cancer
- KIRC / E-MTAB-1980: no prior genes for this cancer

## Caveats

- Only five held-out cohorts, 34 claims and 2-24 measured genes per cohort.
- Weights come from M2 on dev cohorts; the held-out cohorts are not used for fitting.
- The same gene set is used for both arms; only claim weights differ.
- C-index is unadjusted for stage/age and is not a clinical model.
- A small positive mean difference is not a formal non-inferiority result.

The result is mixed across cohorts and the sample is small. It does not establish that down-weighting improves prognosis; it only checks that the M8-weighted prior does not collapse relative to the uniform prior.
