`B_CONFIG`: `tier=RR4 grid=1664x128x128 ell=0.25 lambda_over_h=32 eps=2 seed=6001 L1=13`.

#### Correctors (`B_SOLVE`) and metrics (`B_METRIC`; points = lattice 1e5 + cell centres of three x1 lines inside `[0.5, L1 - 0.5]`)

| corrector | converged | iterations | initial | final | rel |
|---|---|---|---|---|---|
| 1 | 1 | 30 | 1.194356e+06 | 8.365622e-07 | 7.004294e-13 |
| 2 | 1 | 30 | 1.075872e+06 | 2.996660e-07 | 2.785331e-13 |
| 3 | 1 | 30 | 1.191288e+06 | 2.564746e-07 | 2.152919e-13 |

G=7.187359468623e-01,2.459384430307e-02,2.171976215826e-02 mg_levels=6

| field | quantity | rel difference | gate | verdict | points |
|---|---|---|---|---|---|
| h_tilde | value | 9.936224e-16 | 1e-11 | PASS | 104608 |
| h_tilde | gradient | 2.112960e-14 | 1e-11 | PASS | 104608 |
| h_tilde | hessian | 9.723778e-13 | 1e-11 | PASS | 104608 |
| lnK | value | 7.595407e-16 | none | RECORDED | 104608 |
| lnK | gradient | 1.568518e-14 | none | RECORDED | 104608 |
| lnK | hessian | 9.189621e-13 | none | RECORDED | 104608 |

log verdict line: `B_VERDICT all_pass=1`.

#### Recorded: per-knot decay factor of the end effect (`B_DECAY`, fitted on knots above 1e3 times the interior floor)

| field | line (j,k) | face | quantity | factor | knots used | floor |
|---|---|---|---|---|---|---|
| h_tilde | 0,0 | west | gradient | 0.267955 | 9 | 3.404e-15 |
| h_tilde | 0,0 | east | gradient | 0.267947 | 7 | 3.404e-15 |
| h_tilde | 0,0 | west | hessian | 0.267958 | 9 | 2.477e-13 |
| h_tilde | 0,0 | east | hessian | 0.267949 | 8 | 2.477e-13 |
| h_tilde | 64,32 | west | gradient | 0.267953 | 11 | 2.975e-15 |
| h_tilde | 64,32 | east | gradient | 0.267955 | 11 | 2.975e-15 |
| h_tilde | 64,32 | west | hessian | 0.267943 | 11 | 2.040e-13 |
| h_tilde | 64,32 | east | hessian | 0.267951 | 11 | 2.040e-13 |
| h_tilde | 37,101 | west | gradient | 0.267951 | 11 | 3.349e-15 |
| h_tilde | 37,101 | east | gradient | 0.267952 | 11 | 3.349e-15 |
| h_tilde | 37,101 | west | hessian | 0.267950 | 11 | 2.250e-13 |
| h_tilde | 37,101 | east | hessian | 0.267952 | 11 | 2.250e-13 |
| lnK | 0,0 | west | gradient | 0.267943 | 13 | 2.718e-15 |
| lnK | 0,0 | east | gradient | 0.267944 | 13 | 2.718e-15 |
| lnK | 0,0 | west | hessian | 0.267949 | 12 | 1.884e-13 |
| lnK | 0,0 | east | hessian | 0.267938 | 13 | 1.884e-13 |
| lnK | 64,32 | west | gradient | 0.267944 | 13 | 2.304e-15 |
| lnK | 64,32 | east | gradient | 0.267949 | 13 | 2.304e-15 |
| lnK | 64,32 | west | hessian | 0.267956 | 13 | 1.499e-13 |
| lnK | 64,32 | east | hessian | 0.267939 | 13 | 1.499e-13 |
| lnK | 37,101 | west | gradient | 0.267947 | 13 | 2.389e-15 |
| lnK | 37,101 | east | gradient | 0.267955 | 13 | 2.389e-15 |
| lnK | 37,101 | west | hessian | 0.267952 | 12 | 1.737e-13 |
| lnK | 37,101 | east | hessian | 0.267944 | 12 | 1.737e-13 |

derived: h_tilde gradient factor over 6 fits: mean 0.267952, min 0.267947, max 0.267955.
derived: h_tilde hessian factor over 6 fits: mean 0.267950, min 0.267943, max 0.267958.
derived: lnK gradient factor over 6 fits: mean 0.267947, min 0.267943, max 0.267955.
derived: lnK hessian factor over 6 fits: mean 0.267946, min 0.267938, max 0.267956.

`2 - sqrt(3)` = 0.267949 (derived formula; SPEC "~0.268").

#### Recorded: relative difference profile `B_PROFILE` (9984 rows in the log; per knot distance from the face; first knot = nearest cell centre)

| field | line | face | knots per face | gradient @knot0 | hessian @knot0 | gradient @knot8 | hessian @knot8 | gradient median knots 64.. | hessian median knots 64.. |
|---|---|---|---|---|---|---|---|---|---|
| h_tilde | 0,0 | east | 832 | 2.598e-08 | 2.832e-06 | 6.890e-13 | 7.528e-11 | 3.822e-15 | 2.351e-13 |
| h_tilde | 0,0 | west | 832 | 2.747e-07 | 1.531e-05 | 7.301e-12 | 4.070e-10 | 3.098e-15 | 2.598e-13 |
| h_tilde | 37,101 | east | 832 | 5.277e-06 | 2.942e-04 | 1.402e-10 | 7.817e-09 | 3.414e-15 | 2.331e-13 |
| h_tilde | 37,101 | west | 832 | 3.769e-06 | 2.101e-04 | 1.002e-10 | 5.583e-09 | 3.253e-15 | 2.139e-13 |
| h_tilde | 64,32 | east | 832 | 3.396e-06 | 1.893e-04 | 9.024e-11 | 5.031e-09 | 2.968e-15 | 2.185e-13 |
| h_tilde | 64,32 | west | 832 | 5.155e-06 | 2.874e-04 | 1.370e-10 | 7.636e-09 | 2.976e-15 | 1.894e-13 |
| lnK | 0,0 | east | 832 | 3.075e-05 | 1.733e-03 | 8.170e-10 | 4.604e-08 | 2.631e-15 | 1.933e-13 |
| lnK | 0,0 | west | 832 | 2.394e-05 | 1.349e-03 | 6.363e-10 | 3.586e-08 | 2.787e-15 | 1.815e-13 |
| lnK | 37,101 | east | 832 | 1.906e-05 | 1.074e-03 | 5.064e-10 | 2.854e-08 | 2.428e-15 | 1.736e-13 |
| lnK | 37,101 | west | 832 | 1.750e-05 | 9.863e-04 | 4.650e-10 | 2.621e-08 | 2.344e-15 | 1.778e-13 |
| lnK | 64,32 | east | 832 | 2.877e-05 | 1.621e-03 | 7.645e-10 | 4.308e-08 | 2.244e-15 | 1.553e-13 |
| lnK | 64,32 | west | 832 | 2.396e-05 | 1.350e-03 | 6.367e-10 | 3.588e-08 | 2.352e-15 | 1.433e-13 |

`RECORDED B_PROFILE` line of `results.txt`: `B_PROFILE rows=9984 (raw in the log)`.
