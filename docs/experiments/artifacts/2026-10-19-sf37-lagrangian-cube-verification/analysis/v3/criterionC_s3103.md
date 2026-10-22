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
| e_v | 2.618e-03 | 6.429e-04 | 1.605e-04 |
| e_i1 | 1.469e-04 | 1.134e-05 | 9.780e-07 |
| e_i2 | 6.784e-05 | 6.574e-06 | 7.626e-07 |
| e_div | 6.177e-03 | 5.493e-04 | 3.823e-05 |
| r_F | 5.076e-02 | 1.228e-02 | 3.180e-03 |
| r_out | 8.893e-04 | 2.157e-04 | 5.355e-05 |
| min_c | 1.204e-01 | 1.196e-01 | 1.194e-01 |
| p0.1 | 1.398e-01 | 1.395e-01 | 1.394e-01 |
| p1 | 2.111e-01 | 2.116e-01 | 2.118e-01 |
| p5 | 3.028e-01 | 3.027e-01 | 3.026e-01 |
| p50 | 8.327e-01 | 8.326e-01 | 8.326e-01 |
| max_roundtrip | 1.017e-09 | 9.731e-11 | 1.122e-11 |
| non_ok | 0 | 0 | 0 |
| status | ok | ok | ok |
| inlet_vmin | 1.320e-01 | 1.314e-01 | 1.314e-01 |
| vD1_min | 1.177e-01 | 1.175e-01 | 1.174e-01 |
| vD1_nonpos | 0 | 0 | 0 |
| seed_backflow | 0 | 0 | 0 |
| backflow_enc | 0 | 0 | 0 |
| landings_discarded | 0 | 0 | 0 |
| min_g1_hat | 8.574e-01 | 8.571e-01 | 8.570e-01 |
| status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k) | 262144,0,0,0,0,0,0 | 2097152,0,0,0,0,0,0 | 16777216,0,0,0,0,0,0 |
| e_v_N/e_v_1 | 1.478e+00 | 1.506e+00 | 1.526e+00 |
| e_i_N/e_i_1 | 5.162e+00 | 1.026e+01 | 2.321e+01 |
| wall (s, driver total) | 2.67 | 24.01 | 520.88 |
| wall (s, /usr/bin/time) | 6.0 | 27.5 | 524.3 |
| max RSS (kB) | 654872 | 1014136 | 3769360 |
| peak device (bytes) | 649068544 | 1005584384 | 3664773120 |

## Observed orders log2(q_N / q_2N)

| quantity | 64->128 | 128->256 |
|---|---|---|
| e_v | 2.03 | 2.00 |
| e_i1 | 3.70 | 3.54 |
| e_i2 | 3.37 | 3.11 |
| e_div | 3.49 | 3.84 |
| r_F | 2.05 | 1.95 |

## Per-plane growth along x1 (E: recorded, no threshold)

`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: `../sf37_campaign/analysis/v3/criterionC_s3103_planes.csv`.

N = 64

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.884e-03 | 7.943e-05 | 0.000e+00 |
| 1 | 0.0156 | 1.957e-03 | 2.534e-05 | 7.561e-11 |
| 16 | 0.2500 | 1.821e-03 | 7.885e-05 | 3.248e-10 |
| 32 | 0.5000 | 3.737e-03 | 1.462e-04 | 3.450e-10 |
| 48 | 0.7500 | 2.545e-03 | 5.340e-05 | 6.773e-10 |
| 64 | 1.0000 | 2.892e-03 | 1.308e-04 | 5.925e-10 |

N = 128

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 4.669e-04 | 5.073e-06 | 0.000e+00 |
| 1 | 0.0078 | 4.789e-04 | 1.739e-06 | 4.780e-12 |
| 32 | 0.2500 | 4.445e-04 | 6.026e-06 | 3.363e-11 |
| 64 | 0.5000 | 9.251e-04 | 1.144e-05 | 4.532e-11 |
| 96 | 0.7500 | 6.359e-04 | 4.885e-06 | 9.382e-11 |
| 128 | 1.0000 | 7.215e-04 | 1.784e-05 | 7.118e-11 |

N = 256

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.165e-04 | 3.204e-07 | 0.000e+00 |
| 1 | 0.0039 | 1.182e-04 | 1.311e-07 | 2.992e-13 |
| 64 | 0.2500 | 1.109e-04 | 6.363e-07 | 3.489e-12 |
| 128 | 0.5000 | 2.312e-04 | 1.061e-06 | 4.229e-12 |
| 192 | 0.7500 | 1.590e-04 | 6.438e-07 | 9.557e-12 |
| 256 | 1.0000 | 1.803e-04 | 3.043e-06 | 9.331e-12 |

## Integration ladder (B)

No ladder entries in the inputs (the campaign runs --ladder at N = 32 and 64).

## Verdict B (integration certificate)

| N | status | max_roundtrip (every plane) | non_ok | ladder max\|dpsi\| (primary vs h/32,1e-10; N=32,64) | point |
|---|---|---|---|---|---|
| 64 | ok | 1.017e-09 | 0 | ladder missing | not_evaluable |
| 128 | ok | 9.731e-11 | 0 | n/a | pass |
| 256 | ok | 1.122e-11 | 0 | n/a | pass |

**B verdict: not_evaluable**

## Verdict C (convergence per amplitude)

| quantity | q(128) | q(256) | order last pair | rel change last pair | decreases on every refinement | verdict | note |
|---|---|---|---|---|---|---|---|
| e_v | 6.429e-04 | 1.605e-04 | 2.00 | 0.750 | True | accept |  |
| e_i1 | 1.134e-05 | 9.780e-07 | 3.54 | 0.914 | True | accept |  |
| e_i2 | 6.574e-06 | 7.626e-07 | 3.11 | 0.884 | True | accept |  |
| e_div | 5.493e-04 | 3.823e-05 | 3.84 | 0.930 | True | accept |  |
| r_F | 1.228e-02 | 3.180e-03 | 1.95 | 0.741 | True | accept |  |

**Amplitude verdict (C): accept**
