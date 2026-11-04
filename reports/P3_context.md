# P3 — context graph / M3

Snapshot: `2026-11-01`; 496 papers, 7719 citation edges, 2908 author nodes, 0 support links.

## Acceptance: claim coverage

- Claims with a context record: **496/496 = 100.0%** (threshold 90%: PASS).
- Fields may be null by design (sparse fallback); field coverage is reported separately.

## Field coverage (`pub_plus_3y`)

| Field | Present | Missing | Coverage | Note |
|---|---:|---:|---:|---|
| `author_cluster_id` | 489 | 7 | 98.6% | coauthorship cluster visible at as_of_date |
| `supporting_papers` | 496 | 0 | 100.0% | support links for this claim with pub_date <= as_of |
| `effective_independent_support` | 0 | 496 | 0.0% | independent support clusters (empty support_links in this snapshot) |
| `citation_cascade_score` | 428 | 68 | 86.3% | same-cluster share of citations visible at as_of |
| `lab_historical_replication_rate` | 35 | 461 | 7.1% | leave-one-out M2 history; null when n < 5 |
| `review_time_days` | 0 | 496 | 0.0% | received/accepted dates; absent in the OpenAlex snapshot |

Mean `context_coverage`: **0.540** (median 0.571); 496/496 claims have non-zero coverage.

## `as_of_date` variants

| Variant | As-of range | Mean coverage | Non-zero |
|---|---|---:|---:|
| `publication` | 2010-07-27 .. 2026-10-12 | 0.251 | 496/496 |
| `pub_plus_3y` | 2013-07-27 .. 2029-10-12 | 0.540 | 496/496 |

Both variants are computed separately. Unit tests in `tests/test_m3_asof.py` check that future citations, retractions and papers are invisible at `as_of_date`.

134 claims have `pub_plus_3y > snapshot_date`; the snapshot contains no records after the snapshot date, so these features are truncated at the snapshot boundary rather than leaking future data.

## Prohibited features

Forbidden M3 fields present in `ContextFeatures`: `[]`. Country, institution rank, author gender/ethnicity and seniority are not modelled.

## Known limits

- `support_links` is empty, so `effective_independent_support` is null for every claim.
- `lab_historical_replication_rate` needs >=5 earlier eligible claims in the same author cluster; the 34-claim corpus does not reach that threshold.
- `review_time_days` needs received/accepted dates that the current OpenAlex snapshot does not provide.
- Claim coverage is complete; the 0.56 mean field coverage is a real data limitation, not a missing-code path.
