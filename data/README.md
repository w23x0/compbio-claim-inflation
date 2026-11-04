# Data directory (not committed)

Everything under `data/` except this file is ignored by Git. Source and derived
data are large and partially redistribution-restricted, so the repository ships
only the code, configs and scripts that rebuild them.

| Path | Produced by | Contents |
|---|---|---|
| `data/raw/geo/<GSE>/` | `scripts/ingest_*.py`, `cbi.cohort_pool.geo` | series matrix, platform annotation, clinical/expression parquet |
| `data/raw/pmc_real/` | `python3 -m cbi.m1.pmc_fetch` | cached PMC Open Access XML + manifest |
| `data/raw/hgnc/hgnc_complete.tsv` | manual download | HGNC complete set |
| `data/raw/context/context.json` | `scripts/fetch_openalex_context.py` | as-of context snapshot (P3) |
| `data/processed/<snapshot>/` | P0–P4 | cleaned cohorts, claims, ground truth, context features, caches |
| `data/eval_restricted/` | P0/P2 | evaluation and held-out material; never readable by red-team workers |

Real cohorts must follow the clinical contract in
`src/cbi/cohort_pool/cleaning.py` and the frozen split in
`configs/cohort_split.yaml`.
