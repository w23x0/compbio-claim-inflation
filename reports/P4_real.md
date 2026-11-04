# P4 — legacy checklist ablation (not the MD detector)

- Train/val/test: 11/11/8
- Positive rate: {'train': 0.7272727272727273, 'val': 0.7272727272727273, 'test': 0.75}
- Evidence AUC: val 0.4166666666666667, test 0.5
- Calibration inverted: True
- Evidence threshold: 0.8628341536495535
- Negative counts: val/test = 3/ 2

Baseline only. The fitted calibration slope is negative on 11 training claims,
so the checklist direction is not supported by this sample. The 1%/5% FPR
thresholds are identical because there are too few negatives. Do not present
this as a working detector.
