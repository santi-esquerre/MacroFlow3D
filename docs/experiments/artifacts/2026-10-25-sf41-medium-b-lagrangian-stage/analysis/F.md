Rungs: `ny` = 32, 64, 128, 256 (`lambda/h` = 16, 32, 64, 128; `ell` 0.5), planes `x1` = 3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5 (`RECORDED F_PLANES`).

```
F_VERDICT eps=1 cells_met=42/42 runs_ok=12/12 MET
F_VERDICT eps=2 cells_met=42/42 runs_ok=12/12 MET
F_VERDICT ALL MET
```

#### Observed order of the RMS `e_v` ladder per amplitude (`F_CELL`; order of a pair = log2(e_N / e_2N); last pair = 128 -> 256; 42 cells = 3 seeds x 14 planes)

| eps | cells | strictly decreasing | MET | min order 32->64 | max order 32->64 | min order 64->128 | max order 64->128 | min last-pair order (cell) | max last-pair order (cell) | min e_v at ny=256 | max e_v at ny=256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 42 | 42 | 2.0001 | 2.0928 | 1.9945 | 2.0435 | 1.9920 (seed 6103 plane 1) | 2.0263 (seed 6101 plane 9) | 2.90e-05 | 1.15e-04 |
| 2 | 42 | 42 | 42 | 1.9762 | 2.0220 | 1.9956 | 2.0063 | 1.9959 (seed 6103 plane 7) | 2.0040 (seed 6101 plane 9) | 7.11e-05 | 4.76e-04 |

RMS `e_v` of the first and last plane per seed (`F_CELL`; four rungs ny = 32, 64, 128, 256):

| eps | seed | plane | x1 | e_v at ny=32 / 64 / 128 / 256 | orders |
|---|---|---|---|---|---|
| 1 | 6101 | 0 | 3 | 2.878377e-03 / 7.167827e-04 / 1.786441e-04 / 4.467375e-05 | 2.0056 / 2.0044 / 1.9996 |
| 1 | 6101 | 13 | 9.5 | 4.832941e-03 / 1.184314e-03 / 2.954404e-04 / 7.338998e-05 | 2.0288 / 2.0031 / 2.0092 |
| 1 | 6102 | 0 | 3 | 2.781594e-03 / 6.948803e-04 / 1.730313e-04 / 4.333496e-05 | 2.0011 / 2.0057 / 1.9974 |
| 1 | 6102 | 13 | 9.5 | 6.610420e-03 / 1.634113e-03 / 4.043280e-04 / 1.013403e-04 | 2.0162 / 2.0149 / 1.9963 |
| 1 | 6103 | 0 | 3 | 4.856242e-03 / 1.211155e-03 / 3.033636e-04 / 7.566049e-05 | 2.0035 / 1.9973 / 2.0034 |
| 1 | 6103 | 13 | 9.5 | 5.402744e-03 / 1.328329e-03 / 3.310511e-04 / 8.221937e-05 | 2.0241 / 2.0045 / 2.0095 |
| 2 | 6101 | 0 | 3 | 7.897289e-03 / 1.984870e-03 / 4.965732e-04 / 1.241984e-04 | 1.9923 / 1.9990 / 1.9994 |
| 2 | 6101 | 13 | 9.5 | 2.121238e-02 / 5.300103e-03 / 1.326211e-03 / 3.309887e-04 | 2.0008 / 1.9987 / 2.0025 |
| 2 | 6102 | 0 | 3 | 7.399537e-03 / 1.855418e-03 / 4.639829e-04 / 1.160390e-04 | 1.9957 / 1.9996 / 1.9995 |
| 2 | 6102 | 13 | 9.5 | 3.036325e-02 / 7.614222e-03 / 1.903636e-03 / 4.758619e-04 | 1.9956 / 1.9999 / 2.0001 |
| 2 | 6103 | 0 | 3 | 1.752398e-02 / 4.350705e-03 / 1.087634e-03 / 2.715606e-04 | 2.0100 / 2.0001 / 2.0018 |
| 2 | 6103 | 13 | 9.5 | 2.646440e-02 / 6.540733e-03 / 1.628007e-03 / 4.073098e-04 | 2.0165 / 2.0063 / 1.9989 |

#### The 24 runs (exit code, wall s from `walls.tsv`, `B_SOLVER`, `STATUS` lines)

| run | rc | wall s | PCG iterations | rel | balance | non_ok plane | non_ok time | exit_inlet plane | exit_outlet plane | exit_inlet time | exit_outlet time | seed_backflow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F_e1_s6101_g32 | 0 | 5 | 40 | 8.712231e-13 | 9.820473e-13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g64 | 0 | 6 | 40 | 1.928585e-12 | 8.023446e-13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g128 | 0 | 12 | 40 | 4.059601e-12 | 1.844127e-12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6101_g256 | 0 | 47 | 40 | 8.081399e-12 | 1.771732e-12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_e1_s6102_g32 | 0 | 5 | 30 | 6.497745e-11 | 1.146443e-10 | 0 | 0 | 0 | 0 | 0 | 279 | 0 |
| F_e1_s6102_g64 | 0 | 6 | 40 | 2.862392e-14 | 1.676735e-13 | 0 | 0 | 0 | 0 | 0 | 277 | 0 |
| F_e1_s6102_g128 | 0 | 12 | 40 | 7.647650e-14 | 5.847627e-13 | 0 | 0 | 0 | 0 | 0 | 276 | 0 |
| F_e1_s6102_g256 | 0 | 48 | 40 | 1.703020e-13 | 2.629606e-12 | 0 | 0 | 0 | 0 | 0 | 276 | 0 |
| F_e1_s6103_g32 | 0 | 5 | 40 | 6.518864e-14 | 6.043527e-14 | 0 | 0 | 0 | 0 | 0 | 117 | 0 |
| F_e1_s6103_g64 | 0 | 7 | 40 | 1.678768e-13 | 3.450564e-14 | 0 | 0 | 0 | 0 | 0 | 111 | 0 |
| F_e1_s6103_g128 | 0 | 11 | 40 | 3.615968e-13 | 9.712946e-14 | 0 | 0 | 0 | 0 | 0 | 111 | 0 |
| F_e1_s6103_g256 | 0 | 49 | 40 | 7.481393e-13 | 3.290102e-13 | 0 | 0 | 0 | 0 | 0 | 109 | 0 |
| F_e2_s6101_g32 | 0 | 6 | 50 | 1.329922e-11 | 9.832689e-12 | 0 | 0 | 0 | 0 | 0 | 2200 | 0 |
| F_e2_s6101_g64 | 0 | 7 | 50 | 5.069808e-11 | 7.550515e-12 | 0 | 0 | 0 | 0 | 0 | 2207 | 0 |
| F_e2_s6101_g128 | 0 | 13 | 60 | 2.280992e-12 | 6.473766e-13 | 0 | 0 | 0 | 0 | 0 | 2204 | 0 |
| F_e2_s6101_g256 | 0 | 55 | 60 | 4.795790e-12 | 7.380071e-13 | 0 | 0 | 0 | 0 | 0 | 2204 | 0 |
| F_e2_s6102_g32 | 0 | 5 | 40 | 3.213466e-11 | 1.051728e-10 | 0 | 0 | 0 | 0 | 0 | 540 | 0 |
| F_e2_s6102_g64 | 0 | 7 | 40 | 7.668435e-11 | 1.928310e-09 | 0 | 0 | 0 | 0 | 0 | 484 | 0 |
| F_e2_s6102_g128 | 0 | 12 | 50 | 4.870740e-13 | 3.696217e-12 | 0 | 0 | 0 | 0 | 0 | 482 | 0 |
| F_e2_s6102_g256 | 0 | 50 | 50 | 8.223086e-13 | 5.679360e-12 | 0 | 0 | 0 | 0 | 0 | 482 | 0 |
| F_e2_s6103_g32 | 0 | 6 | 50 | 8.374818e-12 | 1.489147e-11 | 0 | 0 | 0 | 0 | 0 | 2288 | 0 |
| F_e2_s6103_g64 | 0 | 7 | 50 | 3.662682e-11 | 5.130068e-11 | 0 | 0 | 0 | 0 | 0 | 2260 | 0 |
| F_e2_s6103_g128 | 0 | 13 | 50 | 8.776535e-11 | 6.402827e-11 | 0 | 0 | 0 | 0 | 0 | 2252 | 0 |
| F_e2_s6103_g256 | 0 | 56 | 60 | 3.683051e-11 | 1.226788e-11 | 0 | 0 | 0 | 0 | 0 | 2249 | 0 |

derived: wall of the 24 runs: min 5 s, max 56 s, sum 450 s.

#### Recorded (no threshold): `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` from the campaign `F_DP` lines of `results.txt` (computed on the V100 from the per-particle `.npy` arrays, absent from `raw/`; 84 lines)

| eps | cells | concordant (campaign) | discordant (campaign) | recomputed class agrees | median |D_P| ny=32/64/128/256 | max |D_P| ny=32/64/128/256 | consumer sigma_T^2 range at ny=256 |
|---|---|---|---|---|---|---|---|
| 1 | 42 | 21 | 21 | 42 / 42 | 5.07e-05 / 1.36e-05 / 1.86e-05 / 2.80e-06 | 3.31e-04 / 1.82e-04 / 1.33e-04 / 8.61e-06 | 2.9053e-02 - 7.5889e-02 |
| 2 | 42 | 28 | 14 | 42 / 42 | 2.69e-04 / 5.72e-05 / 3.73e-05 / 6.38e-06 | 1.06e-03 / 3.94e-04 / 2.73e-04 / 3.06e-05 | 1.5400e-02 - 9.8512e-02 |

V3b recomputation (derived, RR8: concordant iff |D_P| decreases on every refinement and |D_P(ny=256)| <= max(0.05 sigma_T^2(consumer, ny=256), 2 SE_D(ny=256))): agrees with the campaign label on 84 of 84 cells. Recorded only; no `alpha` content.

`F_V3B_SUMMARY` lines of the campaign: `F_V3B_SUMMARY eps=1 concordant=21 discordant=21 not_evaluable=0 (of 42 cells; non-blocking)`; `F_V3B_SUMMARY eps=2 concordant=28 discordant=14 not_evaluable=0 (of 42 cells; non-blocking)`.
derived: max `csv_max_rel` (recomputed var_x2/var_x3 against the programs' moments CSVs) over all F_DP lines: 2.410e-14.
Lines in the campaign `results.txt` that are not `D_P` rows: 0.
