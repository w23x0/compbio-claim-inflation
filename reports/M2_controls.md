# M2 — acceptance controls

- Positive controls (canonical markers): n=20, median robustness **0.8699682849196342** (threshold > 0.6: PASS)
- Negative controls (random genes): n=40, median robustness **0.2781885661478399** (threshold < 0.2: FAIL)
- At zero effect, 0.5*Phi(0)=0.25 is a baseline, not a hard lower bound. Negative z values can score below 0.2. See [M2_null_analysis.md](M2_null_analysis.md) for the ideal-null expectation and exploratory alternatives. The measured controls and original acceptance thresholds are unchanged.
- Inflation sensitivity: n=40, median optimal-cutpoint inflation **0.06191099383019492** vs reference 0.0 (threshold > 0.05: PASS)
- Cross-snapshot stability: not evaluable (one snapshot).

Positive basis: canonical proliferation/hypoxia markers with a published prognostic association in the cancer; list fixed before any dev-cohort effect was computed for them.
