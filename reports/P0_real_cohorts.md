# P0 — real cohort pool

- Cohorts: 22; ok: 22
- Roles: dev 9, visible_eval 7, heldout 6
- By cancer: {'KIRC': 4, 'LIHC': 5, 'LUAD': 6, 'BRCA': 7}

Dev cohorts (worker-visible):

| cohort | cancer | n | events | genes | platform | status |
|---|---|---:|---:|---:|---|---|
| GSE14520 | LIHC | 242 | 96 | 12304 | GPL3921+GPL571 | ok |
| GSE337928 | KIRC | 127 | 56 | 77307 | GPL24676 (Illumina NovaSeq 6000, RNA-seq) | ok |
| GSE37745 | LUAD | 106 | 77 | 21755 | GPL570 | ok |
| GSE68465 | LUAD | 442 | 236 | 13101 | GPL96 | ok |
| GSE96058 | BRCA | 3069 | 322 | 30865 | GPL11154 | ok |
| METABRIC | BRCA | 1979 | 1143 | 20373 | Illumina_HumanHT-12_v3 | ok |
| TCGA-BRCA | BRCA | 1080 | 151 | 20530 | IlluminaHiSeq_RNASeqV2 | ok |
| TCGA-KIRC | KIRC | 531 | 175 | 20530 | Illumina HiSeq 2000 RNA-seq (Xena HiSeqV2) | ok |
| TCGA-LIHC | LIHC | 365 | 130 | 20530 | Illumina HiSeq (RNA-seq) | ok |

Held-out/visible cohort identities and roles are in
`logs/grader_internal/P0_real_cohorts.json` (grader-only).

KIRC has 4 valid cohorts; no fifth independent public ccRCC
cohort with >=80 samples and >=15 OS events exists. No duplicate
patients were used to pad the pool.
