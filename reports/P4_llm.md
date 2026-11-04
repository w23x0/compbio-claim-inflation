# P4 — LLM content monitor (MD)

## Run on the 391-label corpus

- Train/val/test: **58 / 128 / 205**
- Positive rate: 0.621 / 0.555 / 0.561
- Evidence AUC: val **0.513**, test **0.453**
- Recall at 5% FPR (test): **0.035**
- Combined AUC: val 0.519, test 0.465
- Context-only test AUC: 0.500
- Evidence threshold (1% FPR): 0.950; combined threshold (5% FPR): 0.7795
- Context coverage: 0.540

## Honest interpretation

The retrain **completed**, but the LLM content score has **no useful signal**
on this corpus (test AUC below 0.5; 5% FPR recall 3.5%). The detector is worse
than the previous 30-label baseline (test AUC 0.542). This is a real negative
result, not a failed run.

Likely causes: the LLM checklist score is not aligned with the M2
inflation label at this scale; prompts/truncation and label noise (391 labels,
many low-confidence, 257 out-of-pool) dominate. The context prior is also at
the base rate (AUC 0.5).

P5 isolation passed with this detector, and P6 ran three rounds; the P6 blind
spots should be read against this weak detector, not as a working monitor.
