# SF-35 matrix digest, eps = 0.5 (sigma_Y = 0.5)

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
| 32 | ok | 0 | traced |
| 64 | ok | 0 | traced |
| 128 | ok | 0 | traced |
| 256 | ok | 0 | traced |

## Quantities per grid

| quantity | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|
| e_v | 5.984e-03 | 1.473e-03 | 3.682e-04 | 9.211e-05 |
| e_i1 | 7.798e-04 | 6.611e-05 | 5.491e-06 | 6.601e-07 |
| e_i2 | 2.840e-04 | 2.851e-05 | 3.434e-06 | 5.457e-07 |
| e_div | 8.714e-03 | 8.226e-04 | 5.992e-05 | 4.305e-06 |
| r_F | 9.028e-02 | 2.046e-02 | 5.330e-03 | 1.481e-03 |
| r_out | 1.050e-03 | 2.234e-04 | 5.405e-05 | 1.328e-05 |
| min_c | 2.294e-01 | 2.288e-01 | 2.286e-01 | 2.284e-01 |
| p0.1 | 2.539e-01 | 2.528e-01 | 2.527e-01 | 2.528e-01 |
| p1 | 3.278e-01 | 3.296e-01 | 3.303e-01 | 3.308e-01 |
| p5 | 4.802e-01 | 4.798e-01 | 4.807e-01 | 4.810e-01 |
| p50 | 9.699e-01 | 9.698e-01 | 9.698e-01 | 9.698e-01 |
| max_roundtrip | 2.740e-09 | 3.403e-10 | 3.123e-11 | 3.770e-12 |
| non_ok | 0 | 0 | 0 | 0 |
| status | ok | ok | ok | ok |
| inlet_vmin | 2.445e-01 | 2.427e-01 | 2.421e-01 | 2.420e-01 |
| vD1_min | 2.271e-01 | 2.278e-01 | 2.280e-01 | 2.279e-01 |
| vD1_nonpos | 0 | 0 | 0 | 0 |
| seed_backflow | 0 | 0 | 0 | 0 |
| backflow_enc | 0 | 0 | 0 | 0 |
| landings_discarded | 0 | 0 | 0 | 0 |
| min_g1_hat | 9.248e-01 | 9.238e-01 | 9.235e-01 | 9.235e-01 |
| status_hist (ok,seed_bf,cap,stag,uflow,landfail,invalid_k) | 32768,0,0,0,0,0,0 | 262144,0,0,0,0,0,0 | 2097152,0,0,0,0,0,0 | 16777216,0,0,0,0,0,0 |
| e_v_N/e_v_1 | 1.879e+00 | 2.049e+00 | 2.148e+00 | 2.203e+00 |
| e_i_N/e_i_1 | 3.528e+00 | 5.571e+00 | 1.148e+01 | 2.617e+01 |
| wall (s, driver total) | 7.40 | 99.43 | 183.68 | 3002.15 |
| wall (s, /usr/bin/time) | 10.9 | 103.0 | 187.3 | 3005.9 |
| max RSS (kB) | 613636 | 665620 | 995688 | 3766692 |
| peak device (bytes) | 531628032 | 638582784 | 988807168 | 3647995904 |

## Observed orders log2(q_N / q_2N)

| quantity | 32->64 | 64->128 | 128->256 |
|---|---|---|---|
| e_v | 2.02 | 2.00 | 2.00 |
| e_i1 | 3.56 | 3.59 | 3.06 |
| e_i2 | 3.32 | 3.05 | 2.65 |
| e_div | 3.40 | 3.78 | 3.80 |
| r_F | 2.14 | 1.94 | 1.85 |

## Per-plane growth along x1 (E: recorded, no threshold)

`e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)` (as the GROWTH line). Full lists: `docs/experiments/artifacts/2026-10-07-sf35-label-transport/raw/matrix_eps05/planes_eps0.5.csv`.

N = 32

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 3.042e-03 | 3.806e-04 | 0.000e+00 |
| 1 | 0.0312 | 3.534e-03 | 1.476e-04 | 3.789e-10 |
| 8 | 0.2500 | 5.553e-03 | 2.618e-04 | 1.463e-09 |
| 16 | 0.5000 | 5.930e-03 | 6.050e-04 | 2.054e-09 |
| 24 | 0.7500 | 7.187e-03 | 6.351e-04 | 1.593e-09 |
| 32 | 1.0000 | 6.641e-03 | 5.208e-04 | 2.149e-09 |

N = 64

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 7.317e-04 | 2.817e-05 | 0.000e+00 |
| 1 | 0.0156 | 8.042e-04 | 9.348e-06 | 2.358e-11 |
| 16 | 0.2500 | 1.399e-03 | 2.398e-05 | 1.435e-10 |
| 32 | 0.5000 | 1.472e-03 | 4.924e-05 | 2.109e-10 |
| 48 | 0.7500 | 1.734e-03 | 5.204e-05 | 1.896e-10 |
| 64 | 1.0000 | 1.648e-03 | 5.208e-05 | 3.403e-10 |

N = 128

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 1.818e-04 | 1.912e-06 | 0.000e+00 |
| 1 | 0.0078 | 1.912e-04 | 6.221e-07 | 1.838e-12 |
| 32 | 0.2500 | 3.510e-04 | 3.098e-06 | 1.120e-11 |
| 64 | 0.5000 | 3.691e-04 | 4.024e-06 | 1.453e-11 |
| 96 | 0.7500 | 4.313e-04 | 5.121e-06 | 2.227e-11 |
| 128 | 1.0000 | 4.107e-04 | 7.145e-06 | 2.804e-11 |

N = 256

| j | x1 | e_v_j | e_i_j | roundtrip_j |
|---|---|---|---|---|
| 0 | 0.0000 | 4.539e-05 | 1.226e-07 | 0.000e+00 |
| 1 | 0.0039 | 4.657e-05 | 4.555e-08 | 1.275e-13 |
| 64 | 0.2500 | 8.784e-05 | 5.121e-07 | 1.313e-12 |
| 128 | 0.5000 | 9.241e-05 | 4.954e-07 | 1.814e-12 |
| 192 | 0.7500 | 1.077e-04 | 7.480e-07 | 2.628e-12 |
| 256 | 1.0000 | 1.026e-04 | 1.192e-06 | 2.783e-12 |

## Integration ladder (B)

| N | hmax | tol | max\|dpsi1\| vs primary | max\|dpsi2\| vs primary | max_roundtrip | status |
|---|---|---|---|---|---|---|
| 32 | h/16 | 1e-08 | 1.802e-09 | 1.413e-09 | 4.168e-10 | ok |
| 32 | h/16 | 1e-10 | 1.802e-09 | 1.413e-09 | 4.168e-10 | ok |
| 32 | h/32 | 1e-10 | 1.824e-09 | 1.453e-09 | 4.656e-11 | ok |
| 64 | h/16 | 1e-08 | 1.551e-10 | 1.459e-10 | 4.017e-11 | ok |
| 64 | h/16 | 1e-10 | 1.551e-10 | 1.459e-10 | 4.017e-11 | ok |
| 64 | h/32 | 1e-10 | 1.579e-10 | 1.536e-10 | 5.013e-12 | ok |

## Verdict B (integration certificate)

| N | status | max_roundtrip (every plane) | non_ok | ladder max\|dpsi\| (primary vs h/32,1e-10; N=32,64) | point |
|---|---|---|---|---|---|
| 32 | ok | 2.740e-09 | 0 | 1.824e-09 | pass |
| 64 | ok | 3.403e-10 | 0 | 1.579e-10 | pass |
| 128 | ok | 3.123e-11 | 0 | n/a | pass |
| 256 | ok | 3.770e-12 | 0 | n/a | pass |

**B verdict: pass**

## Verdict C (convergence per amplitude)

| quantity | q(128) | q(256) | order last pair | rel change last pair | decreases on every refinement | verdict | note |
|---|---|---|---|---|---|---|---|
| e_v | 3.682e-04 | 9.211e-05 | 2.00 | 0.750 | True | accept |  |
| e_i1 | 5.491e-06 | 6.601e-07 | 3.06 | 0.880 | True | accept |  |
| e_i2 | 3.434e-06 | 5.457e-07 | 2.65 | 0.841 | True | accept |  |
| e_div | 5.992e-05 | 4.305e-06 | 3.80 | 0.928 | True | accept |  |
| r_F | 5.330e-03 | 1.481e-03 | 1.85 | 0.722 | True | accept |  |

**Amplitude verdict (C): accept**
