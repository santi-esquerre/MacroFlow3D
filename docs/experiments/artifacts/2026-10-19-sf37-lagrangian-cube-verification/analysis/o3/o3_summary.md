# SF-37 O3 (blind): robust self-convergence metric of the gridded labels

- python 3.11.7, numpy 1.26.4, scipy 1.11.4
- bootstrap matrix: `analysis/bootstrap_blocks.npy`, sha256 `16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39` (verified against the pre-registered value)
- command: `o3.py /home/sesquerre/sf37_campaign /home/sesquerre/sf37_campaign/analysis/o3 --design full`
- design: `full` (eps codes [15, 20], seeds [3101, 3102, 3103], chain of fine grids [128, 256, 512]; pairs (N/2, N))
- q = 1e-3 (exact), d > 0.1 threshold, O3a order >= 1.8, reject if relative change < 0.10 and e > 1e-12
- peak resident set size of this run (ru_maxrss): 4.92 GB (estimate in the module docstring: ~1.5 GB at the 512 pair)

## Verdicts (O3a, O3b) per amplitude and seed

| sigma2 | seed | O3a | order last pair | rel change last pair | e_v^(q) N=128 | e_v^(q) N=256 | e_v^(q) N=512 | O3b | f_half N=128 | f_half N=256 | f_half N=512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.25 | 3101 | accept | 1.98629 | 0.747613 | 0.00923625 | 0.00237677 | 0.000599867 | shrinking | 0.00457279 | 0.00375006 | 0.00336794 |
| 2.25 | 3102 | accept | 2.12746 | 0.77114 | 0.00933969 | 0.00234119 | 0.000535804 | shrinking | 0.0020035 | 0.000141294 | 2.68674e-05 |
| 2.25 | 3103 | accept | 1.99345 | 0.748863 | 0.00867114 | 0.0020787 | 0.000522038 | shrinking | 0.00179036 | 0.000970795 | 0.00076276 |
| 4 | 3101 | accept | 2.20155 | 0.782596 | 0.0486779 | 0.0171976 | 0.00373881 | shrinking | 0.00178649 | 0.00164507 | 0.000276812 |
| 4 | 3102 | unresolved | 1.65159 | 0.68171 | 0.122639 | 0.0737958 | 0.0234884 | shrinking | 0.000193762 | 2.54714e-05 | 8.2577e-06 |
| 4 | 3103 | accept | 2.59793 | 0.834825 | 0.23737 | 0.150569 | 0.0248703 | shrinking | 0.00023639 | 0.000102847 | 4.45198e-05 |

Reasons:

- sigma2 2.25 seed 3101: O3a accept (decreases on every refinement, order 1.986 >= 1.8 on 256 -> 512); O3b shrinking (f_half = 0.00457279, 0.00375006, 0.00336794)
- sigma2 2.25 seed 3102: O3a accept (decreases on every refinement, order 2.127 >= 1.8 on 256 -> 512); O3b shrinking (f_half = 0.0020035, 0.000141294, 2.68674e-05)
- sigma2 2.25 seed 3103: O3a accept (decreases on every refinement, order 1.993 >= 1.8 on 256 -> 512); O3b shrinking (f_half = 0.00179036, 0.000970795, 0.00076276)
- sigma2 4 seed 3101: O3a accept (decreases on every refinement, order 2.202 >= 1.8 on 256 -> 512); O3b shrinking (f_half = 0.00178649, 0.00164507, 0.000276812)
- sigma2 4 seed 3102: O3a unresolved (decreases=True, order 1.652, relative change 0.6817); O3b shrinking (f_half = 0.000193762, 2.54714e-05, 8.2577e-06)
- sigma2 4 seed 3103: O3a accept (decreases on every refinement, order 2.598 >= 1.8 on 256 -> 512); O3b shrinking (f_half = 0.00023639, 0.000102847, 4.45198e-05)

## O3c - fraction of the O6 weight landing within one coarse cell of a marked vertex

| eps_code | seed | N | x1 | status | weighted fraction | contributors | near marked | note |
|---|---|---|---|---|---|---|---|---|
| 15 | 3101 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3101 | 128 | 0.5 | ok | 0 | 65536 | 0 |  |
| 15 | 3101 | 128 | pooled | ok | 0 | 65536 | 0 |  |
| 15 | 3101 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3101 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3101 | 256 | 0.5 | ok | 0.000599844 | 65536 | 82 |  |
| 15 | 3101 | 256 | pooled | ok | 0.000599844 | 65536 | 82 |  |
| 15 | 3101 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3101 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3101 | 512 | 0.5 | ok | 0.000446769 | 65536 | 62 |  |
| 15 | 3101 | 512 | pooled | ok | 0.000446769 | 65536 | 62 |  |
| 15 | 3101 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3102 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 128 | 0.5 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 128 | pooled | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3102 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 256 | 0.5 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 256 | pooled | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3102 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 512 | 0.5 | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 512 | pooled | ok | 0 | 65536 | 0 |  |
| 15 | 3102 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3103 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3103 | 128 | 0.5 | ok | 0.0033348 | 65536 | 2297 |  |
| 15 | 3103 | 128 | pooled | ok | 0.0033348 | 65536 | 2297 |  |
| 15 | 3103 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3103 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3103 | 256 | 0.5 | ok | 0.00203459 | 65536 | 1614 |  |
| 15 | 3103 | 256 | pooled | ok | 0.00203459 | 65536 | 1614 |  |
| 15 | 3103 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 15 | 3103 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 15 | 3103 | 512 | 0.5 | ok | 0.00140361 | 65536 | 1209 |  |
| 15 | 3103 | 512 | pooled | ok | 0.00140361 | 65536 | 1209 |  |
| 15 | 3103 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3101 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3101 | 128 | 0.5 | ok | 0.00213087 | 65536 | 429 |  |
| 20 | 3101 | 128 | pooled | ok | 0.00213087 | 65536 | 429 |  |
| 20 | 3101 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3101 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3101 | 256 | 0.5 | ok | 0.00192127 | 65536 | 419 |  |
| 20 | 3101 | 256 | pooled | ok | 0.00192127 | 65536 | 419 |  |
| 20 | 3101 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3101 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3101 | 512 | 0.5 | ok | 0.00149105 | 65536 | 335 |  |
| 20 | 3101 | 512 | pooled | ok | 0.00149105 | 65536 | 335 |  |
| 20 | 3101 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3102 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 128 | 0.5 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 128 | pooled | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3102 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 256 | 0.5 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 256 | pooled | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3102 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 512 | 0.5 | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 512 | pooled | ok | 0 | 65536 | 0 |  |
| 20 | 3102 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3103 | 128 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3103 | 128 | 0.5 | ok | 0.00250004 | 65536 | 2456 |  |
| 20 | 3103 | 128 | pooled | ok | 0.00250004 | 65536 | 2456 |  |
| 20 | 3103 | 128 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3103 | 256 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3103 | 256 | 0.5 | ok | 0.00166711 | 65536 | 2233 |  |
| 20 | 3103 | 256 | pooled | ok | 0.00166711 | 65536 | 2233 |  |
| 20 | 3103 | 256 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |
| 20 | 3103 | 512 | 0.25 | ok | 0 | 65536 | 0 |  |
| 20 | 3103 | 512 | 0.5 | ok | 0.0011395 | 65536 | 1989 |  |
| 20 | 3103 | 512 | pooled | ok | 0.0011395 | 65536 | 1989 |  |
| 20 | 3103 | 512 | 1 | structurally_empty | nan | 0 | 0 | x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only) |

x1 = 1 is the outlet coarse plane: structurally empty (no marks by construction).

## D7 - concentration of the defect (recorded, no thresholds)

Reading note: f_half shrinking means concentration of the defect, not convergence (N_half growth ~8 uniform, ~4 sheet, ~2 curve per doubling).

| eps_code | seed | pair N | pair 2N | N_half | N_half(2N) | growth | RMS d unmarked | RMS d unmarked (2N) | pair order |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 3101 | 128 | 256 | 1180 | 7803 | 6.61271 | 0.0221174 | 0.00665037 | 1.73368 |
| 15 | 3101 | 256 | 512 | 7803 | 56284 | 7.21312 | 0.00665037 | 0.00176216 | 1.91609 |
| 15 | 3102 | 128 | 256 | 517 | 294 | 0.568665 | 0.0154792 | 0.00525383 | 1.55889 |
| 15 | 3102 | 256 | 512 | 294 | 449 | 1.52721 | 0.00525383 | 0.00149141 | 1.8167 |
| 15 | 3103 | 128 | 256 | 462 | 2020 | 4.37229 | 0.0172225 | 0.00564243 | 1.60991 |
| 15 | 3103 | 256 | 512 | 2020 | 12747 | 6.3104 | 0.00564243 | 0.00152094 | 1.89135 |
| 20 | 3101 | 128 | 256 | 461 | 3423 | 7.42516 | 0.0693531 | 0.028639 | 1.27598 |
| 20 | 3101 | 256 | 512 | 3423 | 4626 | 1.35145 | 0.028639 | 0.00897063 | 1.6747 |
| 20 | 3102 | 128 | 256 | 50 | 53 | 1.06 | 0.0974509 | 0.0511989 | 0.928561 |
| 20 | 3102 | 256 | 512 | 53 | 138 | 2.60377 | 0.0511989 | 0.0183319 | 1.48176 |
| 20 | 3103 | 128 | 256 | 61 | 214 | 3.5082 | 0.141714 | 0.0833591 | 0.765569 |
| 20 | 3103 | 256 | 512 | 214 | 744 | 3.47664 | 0.0833591 | 0.0278276 | 1.58282 |

Per-cell values (n_common, n_marked, RMS|c_N|, v_rms of vD over the common vertices and vD.json, e_v^(q) and the unmasked e_v, f_half, N_half, fraction d > 0.1, d_max and its location in cube units, RMS d): `o3_cells.csv`.

## Input manifest

176 input files (path, sha256, bytes in `input_manifest_o3.csv`); sha256 of the manifest listing: `36f1831a46fa7fb5e1060d884f2a81fe3bd038a2ba837c2bd151f06eb233dbb2`.
