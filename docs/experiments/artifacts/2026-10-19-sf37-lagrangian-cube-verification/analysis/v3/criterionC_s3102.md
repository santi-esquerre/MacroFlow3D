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

Grids present: [64, 128, 256]; expected for the verdicts: [64, 128, 256].

## Status per grid

| N | status | exit_code | stage |
|---|---|---|---|
| 64 | ok | 0 | traced |
| 128 | ok | 0 | traced |
| 256 | ok | 0 | traced |

## Quantities per grid

| quantity | N=64 | N=128 | N=256 |
|---|---|---|---|
| e_v | 3.868e-03 | 9.341e-04 | 2.306e-04 |
| e_i1 | 6.476e-04 | 8.507e-05 | 7.239e-06 |
| e_i2 | 3.745e-04 | 4.627e-05 | 3.829e-06 |
| e_div | 1.003e-02 | 1.562e-03 | 1.435e-04 |
| r_F | 8.845e-02 | 1.908e-02 | 4.129e-03 |
| r_out | 2.089e-03 | 4.825e-04 | 1.181e-04 |
| min_c | 9.071e-02 | 9.033e-02 | 9.013e-02 |
| p0.1 | 1.117e-01 | 1.113e-01 | 1.111e-01 |
| p1 | 1.793e-01 | 1.790e-01 | 1.787e-01 |
| p5 | 2.711e-01 | 2.704e-01 | 2.701e-01 |
| p50 | 7.670e-01 | 7.680e-01 | 7.687e-01 |
| max_roundtrip | 2.549e-09 | 3.641e-10 | 3.417e-11 |
| non_ok | 0 | 0 | 0 |
| status | ok | ok | ok |
| inlet_vmin | 2.675e-01 | 2.672e-01 | 2.671e-01 |
| vD1_min | 8.795e-02 | 8.794e-02 | 8.792e-02 |
| vD1_nonpos | 0 | 0 | 0 |
| seed_backflow | 0 | 0 | 0 |
| backflow_enc | 0 | 0 | 0 |
| landings_discarded | 0 | 0 | 0 |
| min_g1_hat | 7.787e-01 | 7.784e-01 | 7.783e-01 |
| status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k) | 262144,0,0,0,0,0,0 | 2097152,0,0,0,0,0,0 | 16777216,0,0,0,0,0,0 |
| e_v_N/e_v_1 | 1.720e+00 | 1.731e+00 | 1.749e+00 |
| e_i_N/e_i_1 | 1.472e+01 | 3.127e+01 | 4.935e+01 |
| wall (s, driver total) | 2.68 | 24.21 | 527.51 |
| wall (s, /usr/bin/time) | 6.1 | 27.7 | 531.3 |
| max RSS (kB) | 656428 | 1012792 | 3769372 |
| peak device (bytes) | 649068544 | 1005584384 | 3664773120 |

## Observed orders log2(q_N / q_2N)

| quantity | 64->128 | 128->256 |
|---|---|---|
| e_v | 2.05 | 2.02 |
| e_i1 | 2.93 | 3.55 |
| e_i2 | 3.02 | 3.59 |
| e_div | 2.68 | 3.44 |
| r_F | 2.21 | 2.21 |

## Per-plane growth along x1 (E: recorded, no threshold)

`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: `../sf37_campaign/analysis/v3/criterionC_s3102_planes.csv`.

N = 64

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 2.898e-03 | 1.330e-04 | 0.000e+00 |
| 1 | 0.0156 | 3.129e-03 | 5.449e-05 | 1.911e-10 |
| 16 | 0.2500 | 3.151e-03 | 5.934e-05 | 4.023e-10 |
| 32 | 0.5000 | 3.761e-03 | 1.817e-04 | 7.843e-10 |
| 48 | 0.7500 | 3.909e-03 | 8.088e-04 | 2.549e-09 |
| 64 | 1.0000 | 5.383e-03 | 8.021e-04 | 1.922e-09 |

N = 128

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 7.211e-04 | 7.563e-06 | 0.000e+00 |
| 1 | 0.0078 | 7.512e-04 | 3.407e-06 | 1.249e-11 |
| 32 | 0.2500 | 7.875e-04 | 5.266e-06 | 3.976e-11 |
| 64 | 0.5000 | 9.321e-04 | 1.412e-05 | 1.065e-10 |
| 96 | 0.7500 | 9.227e-04 | 1.023e-04 | 1.831e-10 |
| 128 | 1.0000 | 1.300e-03 | 1.065e-04 | 2.286e-10 |

N = 256

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.802e-04 | 4.827e-07 | 0.000e+00 |
| 1 | 0.0039 | 1.840e-04 | 2.200e-07 | 7.940e-13 |
| 64 | 0.2500 | 1.970e-04 | 6.986e-07 | 5.853e-12 |
| 128 | 0.5000 | 2.329e-04 | 1.224e-06 | 1.021e-11 |
| 192 | 0.7500 | 2.257e-04 | 8.607e-06 | 1.805e-11 |
| 256 | 1.0000 | 3.217e-04 | 1.086e-05 | 2.162e-11 |

## Integration ladder (B)

No ladder entries in the inputs (the campaign runs --ladder at N = 32 and 64).

## Verdict B (integration certificate)

| N | status | max_roundtrip (every plane) | non_ok | ladder max\|dpsi\| (primary vs h/32,1e-10; N=32,64) | point |
|---|---|---|---|---|---|
| 64 | ok | 2.549e-09 | 0 | ladder missing | not_evaluable |
| 128 | ok | 3.641e-10 | 0 | n/a | pass |
| 256 | ok | 3.417e-11 | 0 | n/a | pass |

**B verdict: not_evaluable**

## Verdict C (convergence per amplitude)

| quantity | q(128) | q(256) | order last pair | rel change last pair | decreases on every refinement | verdict | note |
|---|---|---|---|---|---|---|---|
| e_v | 9.341e-04 | 2.306e-04 | 2.02 | 0.753 | True | accept |  |
| e_i1 | 8.507e-05 | 7.239e-06 | 3.55 | 0.915 | True | accept |  |
| e_i2 | 4.627e-05 | 3.829e-06 | 3.59 | 0.917 | True | accept |  |
| e_div | 1.562e-03 | 1.435e-04 | 3.44 | 0.908 | True | accept |  |
| r_F | 1.908e-02 | 4.129e-03 | 2.21 | 0.784 | True | accept |  |

**Amplitude verdict (C): accept**
