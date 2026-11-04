# Natural corpus and label audit

- Snapshot: 2026-11-01; API requests: 0.
- Corpus: 496 claims; eligible: 391; exclusions: {'missing_ground_truth': 22, 'no_eligible_independent_evidence': 83}.
- Cross-split paper groups: 0; identical manuscript groups: 0.
- Claims without paper identity: 0; missing manuscripts: 0.
- Original/independent cohort intersections: 0.
- Label metadata mismatches: 0.

## Eligible label provenance

| Group | N | Positive | Positive rate | Confidence | Reproduction |
|---|---:|---:|---:|---|---|
| pool: in_pool | 161 | 78 | 48.4% | {'medium': 32, 'low': 129} | {'failed': 36, 'partial': 75, 'reproduced': 44, 'no_data': 6} |
| pool: out_of_pool | 230 | 144 | 62.6% | {'medium': 176, 'low': 54} | {'out_of_pool': 230} |
| split: test | 205 | 115 | 56.1% | {'medium': 116, 'low': 89} | {'failed': 17, 'out_of_pool': 125, 'partial': 42, 'reproduced': 18, 'no_data': 3} |
| split: train | 58 | 36 | 62.1% | {'medium': 34, 'low': 24} | {'out_of_pool': 33, 'partial': 11, 'reproduced': 10, 'failed': 4} |
| split: val | 128 | 71 | 55.5% | {'low': 70, 'medium': 58} | {'out_of_pool': 72, 'partial': 22, 'reproduced': 16, 'failed': 15, 'no_data': 3} |

## Cancer coverage by split

| Cancer | Train (positive/negative) | Validation | Test |
|---|---|---|---|
| BRCA | 18/7 | 24/3 | 34/14 |
| KIRC | 4/4 | 10/23 | 15/24 |
| LIHC | 6/9 | 16/9 | 20/20 |
| LUAD | 8/2 | 21/22 | 46/32 |

## Interpretation and limits

- In-pool/out-of-pool refers to the ground-truth engine's original-cohort reproduction outcome. M1's preliminary pool flag is not used to classify labels.
- An out-of-pool label can still have independent evidence, but does not establish original-cohort reproduction. Report detector performance separately for these groups.
- The existing year-based split is audited, not rewritten. Sparse or absent cancer/class cells limit subgroup conclusions.
- Gene/cancer overlap counts: {'train/val': 0, 'train/test': 1, 'val/test': 5}. This is reuse of biological targets, not by itself paper leakage; the current split does not measure unseen-gene generalization.
- Identifier/title and exact-text checks cannot rule out paraphrases, author/lab overlap, or shared patients across distinct cohort accessions.
- Ground-truth labels share a fixed dev cohort pool. This audit does not turn it into an independent clinical validation sample.
- Full counts, claim IDs and source hashes are in corpus_audit.json; existing labels and splits are unchanged.
