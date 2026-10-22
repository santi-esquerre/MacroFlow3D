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
| e_v | 3.465e-03 | 7.956e-04 | 1.966e-04 |
| e_i1 | 3.241e-04 | 2.902e-05 | 2.252e-06 |
| e_i2 | 2.476e-04 | 2.193e-05 | 1.923e-06 |
| e_div | 1.521e-02 | 1.401e-03 | 1.001e-04 |
| r_F | 6.635e-02 | 1.165e-02 | 2.966e-03 |
| r_out | 1.054e-03 | 2.183e-04 | 5.366e-05 |
| min_c | 9.133e-02 | 9.054e-02 | 9.044e-02 |
| p0.1 | 1.069e-01 | 1.061e-01 | 1.060e-01 |
| p1 | 1.448e-01 | 1.443e-01 | 1.441e-01 |
| p5 | 2.187e-01 | 2.178e-01 | 2.174e-01 |
| p50 | 8.091e-01 | 8.090e-01 | 8.090e-01 |
| max_roundtrip | 1.370e-09 | 1.145e-10 | 1.274e-11 |
| non_ok | 0 | 0 | 0 |
| status | ok | ok | ok |
| inlet_vmin | 2.619e-01 | 2.624e-01 | 2.624e-01 |
| vD1_min | 8.673e-02 | 8.643e-02 | 8.638e-02 |
| vD1_nonpos | 0 | 0 | 0 |
| seed_backflow | 0 | 0 | 0 |
| backflow_enc | 0 | 0 | 0 |
| landings_discarded | 0 | 0 | 0 |
| min_g1_hat | 7.527e-01 | 7.515e-01 | 7.512e-01 |
| status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k) | 262144,0,0,0,0,0,0 | 2097152,0,0,0,0,0,0 | 16777216,0,0,0,0,0,0 |
| e_v_N/e_v_1 | 2.858e+00 | 2.862e+00 | 2.910e+00 |
| e_i_N/e_i_1 | 1.100e+01 | 1.736e+01 | 3.266e+01 |
| wall (s, driver total) | 3.35 | 24.40 | 519.68 |
| wall (s, /usr/bin/time) | 6.9 | 27.9 | 523.3 |
| max RSS (kB) | 656256 | 997776 | 3769376 |
| peak device (bytes) | 649068544 | 1005584384 | 3664773120 |

## Observed orders log2(q_N / q_2N)

| quantity | 64->128 | 128->256 |
|---|---|---|
| e_v | 2.12 | 2.02 |
| e_i1 | 3.48 | 3.69 |
| e_i2 | 3.50 | 3.51 |
| e_div | 3.44 | 3.81 |
| r_F | 2.51 | 1.97 |

## Per-plane growth along x1 (E: recorded, no threshold)

`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: `../sf37_campaign/analysis/v3/criterionC_s3101_planes.csv`.

N = 64

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.344e-03 | 5.990e-05 | 0.000e+00 |
| 1 | 0.0156 | 1.448e-03 | 2.040e-05 | 1.578e-10 |
| 16 | 0.2500 | 2.719e-03 | 6.016e-05 | 4.001e-10 |
| 32 | 0.5000 | 3.776e-03 | 3.519e-04 | 4.636e-10 |
| 48 | 0.7500 | 4.291e-03 | 4.390e-04 | 7.204e-10 |
| 64 | 1.0000 | 4.139e-03 | 2.244e-04 | 1.370e-09 |

N = 128

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 3.345e-04 | 3.287e-06 | 0.000e+00 |
| 1 | 0.0078 | 3.484e-04 | 1.281e-06 | 1.047e-11 |
| 32 | 0.2500 | 6.786e-04 | 5.778e-06 | 4.649e-11 |
| 64 | 0.5000 | 8.452e-04 | 2.889e-05 | 4.971e-11 |
| 96 | 0.7500 | 9.217e-04 | 4.031e-05 | 7.066e-11 |
| 128 | 1.0000 | 9.972e-04 | 2.224e-05 | 9.559e-11 |

N = 256

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 8.356e-05 | 1.976e-07 | 0.000e+00 |
| 1 | 0.0039 | 8.531e-05 | 9.480e-08 | 6.644e-13 |
| 64 | 0.2500 | 1.696e-04 | 8.051e-07 | 4.689e-12 |
| 128 | 0.5000 | 2.086e-04 | 2.296e-06 | 5.742e-12 |
| 192 | 0.7500 | 2.248e-04 | 3.103e-06 | 7.004e-12 |
| 256 | 1.0000 | 2.483e-04 | 3.097e-06 | 1.238e-11 |

## Integration ladder (B)

No ladder entries in the inputs (the campaign runs --ladder at N = 32 and 64).

## Verdict B (integration certificate)

| N | status | max_roundtrip (every plane) | non_ok | ladder max\|dpsi\| (primary vs h/32,1e-10; N=32,64) | point |
|---|---|---|---|---|---|
| 64 | ok | 1.370e-09 | 0 | ladder missing | not_evaluable |
| 128 | ok | 1.145e-10 | 0 | n/a | pass |
| 256 | ok | 1.274e-11 | 0 | n/a | pass |

**B verdict: not_evaluable**

## Verdict C (convergence per amplitude)

| quantity | q(128) | q(256) | order last pair | rel change last pair | decreases on every refinement | verdict | note |
|---|---|---|---|---|---|---|---|
| e_v | 7.956e-04 | 1.966e-04 | 2.02 | 0.753 | True | accept |  |
| e_i1 | 2.902e-05 | 2.252e-06 | 3.69 | 0.922 | True | accept |  |
| e_i2 | 2.193e-05 | 1.923e-06 | 3.51 | 0.912 | True | accept |  |
| e_div | 1.401e-03 | 1.001e-04 | 3.81 | 0.929 | True | accept |  |
| r_F | 1.165e-02 | 2.966e-03 | 1.97 | 0.745 | True | accept |  |

**Amplitude verdict (C): accept**
