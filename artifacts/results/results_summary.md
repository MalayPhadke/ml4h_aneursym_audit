# CDEL condition-grid results

Completed conditions: 32; missing initial conditions: 0.

Full baseline: FPR@90=0.0661, AUC=0.9671.

Best guided condition selected on validation only: `b1_neg_radial_high_10`.
Its fixed-test result is FPR@90=0.0816, AUC=0.9628.

Matched positive-removal control: strict Mode-2 pruning and seeded random pruning each removed 170/868 train positives and zero negatives.
Strict Mode-2 fixed-test result: FPR@90=0.0671, AUC=0.9589; matched random: FPR@90=0.1012, AUC=0.9508.
Targeted removal is less harmful than random removal at equal budget, but does not improve on the full n1 baseline.

CAS-dependent A4, repair, CAS-guided pruning, and CAS ablations remain unavailable because true paired recrops/source volumes are absent.

See `condition_results.csv` and `grid_summary.json` for all per-run and aggregate values.
