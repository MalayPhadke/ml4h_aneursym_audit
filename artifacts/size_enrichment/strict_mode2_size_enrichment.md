# Strict Mode-2 small-aneurysm enrichment

**Result: the enrichment survives explicit-ID correction, with a revised effect size.**

| Scope | Strict small / n | Non-strict small / n | OR | 95% CI | Fisher p |
|---|---:|---:|---:|---:|---:|
| all | 204/232 (87.9%) | 738/969 (76.2%) | 2.28 | 1.50–3.48 | 5.71e-05 |
| train | 146/170 (85.9%) | 530/698 (75.9%) | 1.93 | 1.21–3.07 | 0.00522 |
| val | 25/28 (89.3%) | 105/137 (76.6%) | 2.54 | 0.72–8.96 | 0.204 |
| test | 33/34 (97.1%) | 103/134 (76.9%) | 9.93 | 1.30–75.59 | 0.00606 |

Across all 1,201 positives, strict patches were small in 87.9% of cases versus 76.2% for non-strict patches (OR=2.28, Fisher p=5.71e-05).
Within train only, the corresponding values were 85.9% versus 75.9% (OR=1.93, Fisher p=0.00522).

The old OR≈2.88 must not be reused. The corrected reportable all-positive estimate is OR=2.28 (Woolf 95% CI 1.50–3.48), with explicit patch-ID linkage. Report this as a patch-level association and retain the size-estimation and within-patient dependence caveats.
