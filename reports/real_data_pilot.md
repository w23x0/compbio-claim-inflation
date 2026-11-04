# Real-data pilot: GSE76427 (LIHC)

Automated ingestion only; no manual sample editing and no human audit step.

```bash
python3 -m cbi.cohort_pool.geo --gse GSE76427 --out data/raw/geo/GSE76427
python3 -m cbi.m2.real_cohort --cohort data/raw/geo/GSE76427 --genes TP53,MKI67,BIRC5,AFP,CTNNB1
```

- Source: GEO GSE76427 (public), platform GPL10558 (Illumina HumanHT-12 V4).
- Cohort: 115 primary HCC tumors; 23 OS events; 20,759 genes after probe collapse.
- Clinical fields parsed from `!Sample_characteristics_ch1`: `event_os`,
  `duryears_os`, age, sex, TNM stage, patient id, tissue.
- Expression: log2 intensity, max-mean probe per gene.

Example per-gene Cox fits with the reference dichotomous (median) spec
(`cbi.m2.specs.default_spec`):

| gene | HR | p | n | events |
|---|---:|---:|---:|---:|
| TP53 | 0.879 | 0.763 | 115 | 23 |
| MKI67 | 0.466 | 0.113 | 115 | 23 |
| BIRC5 | 0.978 | 0.960 | 115 | 23 |
| AFP | 1.220 | 0.642 | 115 | 23 |
| CTNNB1 | 1.184 | 0.688 | 115 | 23 |

Caveat: 23 OS events is a small event count for stable Cox estimates; this
cohort is a pipeline validation, not a replacement for the larger demo pool.
