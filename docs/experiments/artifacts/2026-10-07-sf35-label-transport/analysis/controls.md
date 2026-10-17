# SF-35 positive controls (A1-A4)

## A1: k = 1 (Y = 0 cells, eps = 0.5)

| N | status | max\|psi1 - x2\| | max\|psi2 - x3\| | A1 (<= 1e-13) |
|---|---|---|---|---|
| 64 | ok | 5.551e-17 | 5.551e-17 | pass |

## A2: k = exp(0.5 cos 2 pi x2) (Y_cell = cos(2 pi (j + 1/2) h), eps = 0.5)

| N | status | max\|dpsi1(j)-dpsi1(0)\| | max\|dpsi2(j)-dpsi2(0)\| | straight (<= 1e-10) | e_v | r_F |
|---|---|---|---|---|---|---|
| 16 | ok | 0.000e+00 | 0.000e+00 | pass | 5.663e-04 | 1.188e-03 |
| 32 | ok | 0.000e+00 | 0.000e+00 | pass | 3.741e-05 | 7.816e-05 |
| 64 | ok | 0.000e+00 | 0.000e+00 | pass | 2.371e-06 | 4.949e-06 |

| quantity | pair | observed order | >= 3.5 |
|---|---|---|---|
| e_v | 16->32 | 3.92 | pass |
| e_v | 32->64 | 3.98 | pass |
| r_F | 16->32 | 3.93 | pass |
| r_F | 32->64 | 3.98 | pass |

- A2 order verdict e_v: **pass**
- A2 order verdict r_F: **pass**
- A2 straightness verdict: **pass**

## A3: control2d (psi2 = x3), eps = 0.5

| N | status | max\|psi2 - x3\| | A3 (<= 1e-10) |
|---|---|---|---|
| 32 | ok | 5.529e-14 | pass |

## A4 and fast tests (ctest)

| test | result | time (s) | passed and <= 60 s |
|---|---|---|---|
| inlet_slab_trace16 | Passed | 10.54 | pass |
| inlet_slab_controls | Passed | 10.34 | pass |

## Summary

| criterion | verdict |
|---|---|
| A1 | pass |
| A2 | pass |
| A3 | pass |
| A4/ctest:inlet_slab_trace16 | pass |
| A4/ctest:inlet_slab_controls | pass |
