# M2 — null baseline and alternative support statistics

This is an offline sensitivity analysis. Production scoring, labels, thresholds and the original control results are unchanged.

## Correction to the earlier '0.25 floor' explanation

At z=0, 0.5*Phi(z)=0.25. This is a baseline, not a pointwise lower bound: negative z gives a smaller value.
Under an ideal calibrated null Z~N(0,1), Phi(Z) is uniform, so its expected weighted contribution is 0.25.
A finite negative-control median can be below 0.2; the existing failed median does not prove that passing is mathematically impossible.
Random expressed genes are not guaranteed null genes, and survival/meta-analysis statistics need not follow the ideal normal null.

## Empirical comparison (same controls and fixed other components)

| Support statistic | Positive median (>0.6) | Negative median (<0.2) | Ideal null expected weighted support |
|---|---:|---:|---:|
| Phi(z) | 0.8700 | 0.2782 | 0.2500 |
| max(0, 2*Phi(z)-1) | 0.8696 | 0.0518 | 0.1250 |
| max(0, (Phi(z)-0.95)/0.05) | 0.8609 | 0.0014 | 0.0125 |

The centered alternative is zero at z=0, but clipping negative values creates a positive null expectation (0.125 after weighting), not zero.
The 95% excess alternative is a stronger nonlinear filter; it is not a probability of biological truth and may suppress genuine weak effects.
Reference effects were refitted for 20 positive and 40 negative controls using dev cohorts only.
Only the independent-support term is replaced. The multiverse and shrinkage remainder is retained from the stored controls and checked for feasible bounds.
For 391 eligible corpus results, the current robustness formula was reconstructed exactly from stored components before comparison.

## Recommendation

Keep the current production formula for now. Treat the centered statistic as a candidate requiring a preregistered comparison on outcome-permuted dev controls and independent cohorts.
Do not choose a formula or relax the <0.2 threshold merely because it passes these already-seen controls. The alternatives are exploratory, not new acceptance results.
Inflation and inflation_z do not use robustness, so the M4 binary labels remain unchanged under these support-only alternatives; M8 robustness weights would change. The current M6 reward uses inflation and detector flags, so this support-only substitution would not directly change its reward formula.
Control uncertainty, between-cancer dependence, and one-snapshot stability remain unresolved. No claim of calibrated null FPR follows from these summaries.
Full per-gene results and per-cancer summaries are in M2_null_analysis.json.

Plot omitted: optional matplotlib dependency is unavailable.
