# Phase 0 gate report

Overall: **PASS**

- G1 complete per-patch coverage: **PASS** (7742/7742)
- IG faithfulness (100-step trapezoid; p95 gap <=5%): **PASS** (p95=0.0144)
- G2 exact-zero Mode 2 point mass reproduced: **PASS**
- G3 frozen split/hash/patient integrity: **PASS**

## Train-positive Mode 2 counts

- Total: 868
- Exact radial_slope == 0: 170 (19.59%)
- PAR NaN: 215 (24.77%)
- Strict radial_slope==0 plus IG exact-zero: 170 (19.59%)
- Exact-zero slope inputs that are all-zero after preprocessing: 238/238
- Exact-zero slope inputs whose raw NPZ resized array is all-zero: 238/238

The old 232/321 counts are historical only; the clean train-only checkpoint determines the new values. Exact-zero slope patches are raw all-zero NPZ inputs, so they must not be described as information-bearing tissue inputs with mysteriously absent gradients.
