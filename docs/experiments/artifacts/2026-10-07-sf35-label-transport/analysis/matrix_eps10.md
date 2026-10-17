# SF-35 matrix digest, eps = 1 (sigma_Y = 1)

```text
Verdict rules (pre-registered, SF-35 spec "Acceptance thresholds", implemented literally):
- B. Round trip <= 1e-8 on every plane at the primary (h/8, 1e-8) for every matrix point; non_ok = 0; at 32 and 64 the
  ladder gives max|dpsi| between the primary and (h/32, 1e-10) <= 1e-8.
- C. Convergence per amplitude (32/64/128/256). e_v, e_i1, e_i2, e_div, r_F each decrease on every refinement and show
  observed order >= 1.8 on 128 -> 256: `accept`; any of them changing < 10 % between 128 and 256 while above 1e-12:
  `reject` (floor); otherwise `unresolved`. Amplitude verdict: `accept` iff all five quantities are `accept`, else
  `reject` if any is `reject`, else `unresolved`. Blocking: eps = 0.5 and 1.0 must be `accept`; eps = 1.5 must be
  `accept` unless the stage reports `inlet_backflow` (then recorded; owner decision).
- Implementation reading: `decrease on every refinement` = q(2N) < q(N) for every consecutive pair of the expected grid
  list; the observed order on the last pair is log2(q(128)/q(256)); the relative change is |q(256) - q(128)| / |q(128)|;
  a non-finite value or a missing grid makes the quantity `not_evaluable` (never silently `unresolved`).
```

Grids present: [32, 64, 128, 256]; expected for the verdicts: [32, 64, 128, 256].

## Status per grid

| N | status | exit_code | stage |
|---|---|---|---|
| 32 | roundtrip_exceeded | 21 | traced |
| 64 | ok | 0 | traced |
| 128 | ok | 0 | traced |
| 256 | ok | 0 | traced |

## Quantities per grid

| quantity | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|
| e_v | 6.789e-02 | 1.501e-02 | 3.164e-03 | 4.845e-04 |
| e_i1 | 2.473e-02 | 3.763e-03 | 4.484e-04 | 4.835e-05 |
| e_i2 | 1.161e-02 | 4.189e-03 | 9.350e-04 | 1.139e-04 |
| e_div | 4.986e-01 | 2.339e-01 | 9.523e-02 | 1.874e-02 |
| r_F | 4.404e+00 | 1.905e+00 | 7.579e-01 | 1.506e-01 |
| r_out | 6.783e-02 | 2.289e-02 | 4.671e-03 | 5.460e-04 |
| min_c | 3.327e-02 | 4.166e-02 | 4.154e-02 | 4.152e-02 |
| p0.1 | 5.222e-02 | 5.176e-02 | 5.162e-02 | 5.165e-02 |
| p1 | 9.123e-02 | 9.277e-02 | 9.314e-02 | 9.337e-02 |
| p5 | 2.024e-01 | 2.037e-01 | 2.041e-01 | 2.043e-01 |
| p50 | 8.862e-01 | 8.866e-01 | 8.863e-01 | 8.864e-01 |
| max_roundtrip | 2.282e-08 | 2.402e-09 | 3.227e-10 | 4.337e-11 |
| non_ok | 0 | 0 | 0 | 0 |
| status | roundtrip_exceeded | ok | ok | ok |
| inlet_vmin | 4.910e-02 | 4.807e-02 | 4.789e-02 | 4.789e-02 |
| vD1_min | 4.098e-02 | 4.115e-02 | 4.115e-02 | 4.112e-02 |
| vD1_nonpos | 0 | 0 | 0 | 0 |
| seed_backflow | 0 | 0 | 0 | 0 |
| backflow_enc | 0 | 0 | 0 | 0 |
| landings_discarded | 0 | 0 | 0 | 0 |
| min_g1_hat | 6.609e-01 | 6.580e-01 | 6.573e-01 | 6.571e-01 |
| status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k) | 32768,0,0,0,0,0,0 | 262144,0,0,0,0,0,0 | 2097152,0,0,0,0,0,0 | 16777216,0,0,0,0,0,0 |
| e_v_N/e_v_1 | 9.789e+00 | 1.549e+01 | 1.438e+01 | 7.824e+00 |
| e_i_N/e_i_1 | 2.560e+01 | 1.234e+02 | 3.647e+02 | 6.239e+02 |
| wall (s, driver total) | 8.59 | 106.41 | 194.01 | 3122.79 |
| wall (s, /usr/bin/time) | 12.2 | 110.1 | 197.6 | 3126.5 |
| max RSS (kB) | 613648 | 667312 | 995624 | 3766680 |
| peak device (bytes) | 531628032 | 638582784 | 988807168 | 3647995904 |

## Observed orders log2(q_N / q_2N)

| quantity | 32->64 | 64->128 | 128->256 |
|---|---|---|---|
| e_v | 2.18 | 2.25 | 2.71 |
| e_i1 | 2.72 | 3.07 | 3.21 |
| e_i2 | 1.47 | 2.16 | 3.04 |
| e_div | 1.09 | 1.30 | 2.35 |
| r_F | 1.21 | 1.33 | 2.33 |

## Per-plane growth along x1 (E: recorded, no threshold)

`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: `docs/experiments/artifacts/2026-10-07-sf35-label-transport/raw/matrix_eps10/planes_eps1.csv`.

N = 32

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 7.864e-03 | 1.650e-03 | 0.000e+00 |
| 1 | 0.0312 | 9.011e-03 | 7.226e-04 | 2.183e-09 |
| 8 | 0.2500 | 1.763e-02 | 2.643e-03 | 9.482e-09 |
| 16 | 0.5000 | 4.964e-02 | 1.525e-02 | 1.199e-08 |
| 24 | 0.7500 | 1.082e-01 | 1.618e-02 | 1.112e-08 |
| 32 | 1.0000 | 8.822e-02 | 1.850e-02 | 2.282e-08 |

N = 64

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.654e-03 | 1.069e-04 | 0.000e+00 |
| 1 | 0.0156 | 1.898e-03 | 4.259e-05 | 1.510e-10 |
| 16 | 0.2500 | 4.180e-03 | 2.314e-04 | 7.752e-10 |
| 32 | 0.5000 | 8.321e-03 | 2.670e-03 | 9.356e-10 |
| 48 | 0.7500 | 1.874e-02 | 2.539e-03 | 9.691e-10 |
| 64 | 1.0000 | 2.939e-02 | 5.257e-03 | 2.402e-09 |

N = 128

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 4.057e-04 | 6.796e-06 | 0.000e+00 |
| 1 | 0.0078 | 4.376e-04 | 2.616e-06 | 9.818e-12 |
| 32 | 0.2500 | 1.038e-03 | 1.719e-05 | 1.175e-10 |
| 64 | 0.5000 | 1.273e-03 | 2.219e-04 | 7.782e-11 |
| 96 | 0.7500 | 2.600e-03 | 3.311e-04 | 1.521e-10 |
| 128 | 1.0000 | 6.293e-03 | 9.541e-04 | 2.626e-10 |

N = 256

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.012e-04 | 4.246e-07 | 0.000e+00 |
| 1 | 0.0039 | 1.051e-04 | 1.700e-07 | 6.138e-13 |
| 64 | 0.2500 | 2.596e-04 | 1.616e-06 | 1.377e-11 |
| 128 | 0.5000 | 2.851e-04 | 1.519e-05 | 1.066e-11 |
| 192 | 0.7500 | 3.978e-04 | 3.002e-05 | 1.420e-11 |
| 256 | 1.0000 | 8.226e-04 | 1.061e-04 | 3.053e-11 |

## Integration ladder (B)

| N | hmax | tol | max\|dpsi1\| vs primary | max\|dpsi2\| vs primary | max_roundtrip | status |
|---|---|---|---|---|---|---|
| 32 | h/16 | 1e-08 | 1.513e-08 | 1.012e-08 | 3.018e-09 | ok |
| 32 | h/16 | 1e-10 | 1.513e-08 | 1.012e-08 | 3.018e-09 | ok |
| 32 | h/32 | 1e-10 | 1.655e-08 | 1.018e-08 | 3.458e-10 | ok |
| 64 | h/16 | 1e-08 | 1.468e-09 | 1.093e-09 | 2.847e-10 | ok |
| 64 | h/16 | 1e-10 | 1.468e-09 | 1.093e-09 | 2.847e-10 | ok |
| 64 | h/32 | 1e-10 | 1.475e-09 | 1.129e-09 | 5.160e-11 | ok |

## Verdict B (integration certificate)

| N | status | max_roundtrip (every plane) | non_ok | ladder max\|dpsi\| (primary vs h/32,1e-10; N=32,64) | point |
|---|---|---|---|---|---|
| 32 | roundtrip_exceeded | 2.282e-08 | 0 | 1.655e-08 | fail |
| 64 | ok | 2.402e-09 | 0 | 1.475e-09 | pass |
| 128 | ok | 3.227e-10 | 0 | n/a | pass |
| 256 | ok | 4.337e-11 | 0 | n/a | pass |

**B verdict: fail**

## Verdict C (convergence per amplitude)

| quantity | q(128) | q(256) | order last pair | rel change last pair | decreases on every refinement | verdict | note |
|---|---|---|---|---|---|---|---|
| e_v | 3.164e-03 | 4.845e-04 | 2.71 | 0.847 | True | accept |  |
| e_i1 | 4.484e-04 | 4.835e-05 | 3.21 | 0.892 | True | accept |  |
| e_i2 | 9.350e-04 | 1.139e-04 | 3.04 | 0.878 | True | accept |  |
| e_div | 9.523e-02 | 1.874e-02 | 2.35 | 0.803 | True | accept |  |
| r_F | 7.579e-01 | 1.506e-01 | 2.33 | 0.801 | True | accept |  |

**Amplitude verdict (C): accept**
