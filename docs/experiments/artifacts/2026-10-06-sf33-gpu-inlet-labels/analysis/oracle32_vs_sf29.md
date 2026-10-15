# SF-33 N8' — production oracle (GPU) vs the SF-29 spectral oracle on the step-8 `gauss` field (acceptance (d), second half)

Question: do the labels of the production oracle (SF-19 affine-periodic Darcy on the SF-28 spline of the potential,
SF-30 integrator, D-1 inlet labels from the SF-19 inlet-face flux) agree with the SF-29 spectral oracle (DOP853 on
the spectral reference Darcy flow, inlet labels from the spectral face velocity) on the same field, to the order of
the SF-19 / spline discretization? Recorded; no threshold (spec acceptance (d): "agree ... to the order of the
SF-19/spline discretization (recorded)").

## Inputs and commands

- Field: the closure-probe `gauss` field of SF-29 (`2026-10-02-closure-probes/scripts/closure_probe.py`: modes
  `|m_d| <= 3`, Gaussian covariance `ell = 1/4`, unit variance, seed 7) at amplitude 0.25, i.e. `ln k = 0.25 f`.
  The N4 exports `exports/crosscheck_gauss_0.25_N/Y_cells.npy` hold `0.25 f` at the cell centres (x1 fastest). The
  field is band-limited (`|m| <= 3 < N/2`), so the driver's spectral vertex evaluation of the cell samples is exact
  to roundoff at N = 16, 24, 32.
- GPU (job `sf33c-oracle32-vs-sf29`, V100, build `build/v100-release` of commit `94c5d53`):
  `inlet_slab --production --n N --eps 1 --cells exports/crosscheck_gauss_0.25_N/Y_cells.npy --oracle-ladder
  --threads 32 --save-oracle exports/oracle_gpu/N<N> --summary raw/oracle32/N<N>.json` for N = 16, 24, 32 (driver
  options `--cells` / `--save-oracle` added by the N8' commit `94c5d53`, additive; `--eps 1` so that the target stage
  field is `k = exp(Y_cells)` exactly; the continuation path `0.25 -> 0.5 -> 1` of the multiplier is irrelevant to
  the oracle, which runs on the target stage only). Logs `logs/oracle32/N<N>.log`, `/usr/bin/time -v` in
  `logs/oracle32/N<N>.time`.
- SF-29 reference: `exports/gauss_0.25_N/psi_or_{1,2}.npy` (N4 export of `cases.load_case("gauss", 0.25, N)`,
  full labels, same slab layout `[j, m2, m3]`).
- Comparison: `scripts/compare_oracle_sf29.py --case N exports/oracle_gpu/N<N> exports/gauss_0.25_N ...` (numpy)
  -> `raw/oracle32/oracle_vs_sf29.md` (GPU primary run, `h_max = h/8`, tol 1e-8), `..._h16_tol1e-08.md`,
  `..._h16_tol1e-10.md` (GPU ladder runs). Per label: `d = psi_or^GPU - psi_or^SF29` on planes 0..N, normalized by
  `RMS(psi_or^SF29 - affine)` (affine = x2 for psi1, x3 for psi2); plane 0 (inlet labels only) and planes 1..N
  separately; mean and de-meaned RMS.

## GPU solver / oracle facts (target stage)

| N | STATUS | PATH | r_F | oracle round trip max (h/8, 1e-8) | (h/16, 1e-8) | (h/16, 1e-10) | ORACLE_LABELDIFF max (h/16) vs primary | inlet vmin | SF-19 v1 vs spline rms_rel |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 0.25->0.5->1 | 3.762e-15 | 9.698e-09 | 8.218e-10 | 8.218e-10 | 3.3e-09 / 5.4e-09 | 0.6501 | 6.13e-03 |
| 24 | converged | 0.25->0.5->1 | 6.310e-15 | 2.576e-09 | 2.023e-10 | 2.023e-10 | 9.7e-10 / 1.3e-09 | 0.6390 | 2.73e-03 |
| 32 | converged | 0.25->0.5->1 | 9.745e-15 | 9.557e-10 | 7.723e-11 | 7.723e-11 | 4.2e-10 / 6.0e-10 | 0.6348 | 1.54e-03 |

All round trips <= 1e-8 at h/8 (16^3: 9.7e-9, the closest to the gate). The ladder runs change the labels by <= 5.4e-9,
i.e. the comparison below is insensitive to the oracle's integration tolerance (the three tables are identical to the
printed digits).

## Result (GPU primary oracle vs SF-29 spectral oracle; `raw/oracle32/oracle_vs_sf29.md`, verbatim)

| N | label | scale RMS(psi_or^SF29 - affine) | rms | max | mean | rms_demeaned | rms_inlet | max_inlet | rms_interior | max_interior | plane of max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | psi1 | 3.135897e-02 | 2.326e-02 | 7.358e-02 | -1.099e-02 | 2.050e-02 | 2.143e-02 | 6.582e-02 | 2.337e-02 | 7.358e-02 | 16 |
| 16 | psi2 | 2.226783e-02 | 2.533e-02 | 6.438e-02 | 2.057e-02 | 1.478e-02 | 1.912e-02 | 2.747e-02 | 2.566e-02 | 6.438e-02 | 10 |
| 24 | psi1 | 3.132811e-02 | 1.035e-02 | 3.363e-02 | -4.870e-03 | 9.138e-03 | 9.543e-03 | 2.965e-02 | 1.039e-02 | 3.363e-02 | 24 |
| 24 | psi2 | 2.229936e-02 | 1.132e-02 | 3.060e-02 | 9.185e-03 | 6.623e-03 | 8.481e-03 | 1.220e-02 | 1.143e-02 | 3.060e-02 | 23 |
| 32 | psi1 | 3.131215e-02 | 5.829e-03 | 1.900e-02 | -2.735e-03 | 5.148e-03 | 5.373e-03 | 1.690e-02 | 5.843e-03 | 1.900e-02 | 32 |
| 32 | psi2 | 2.231560e-02 | 6.380e-03 | 1.702e-02 | 5.185e-03 | 3.717e-03 | 4.767e-03 | 6.857e-03 | 6.423e-03 | 1.702e-02 | 31 |

Observed orders `log(e_a/e_b)/log(N_b/N_a)`:

| pair | label | rms | max | mean | rms_demeaned | rms_inlet | max_inlet | rms_interior | max_interior |
|---|---|---|---|---|---|---|---|---|---|
| 16->24 | psi1 | 2.00 | 1.93 | 2.01 | 1.99 | 1.99 | 1.97 | 2.00 | 1.93 |
| 16->24 | psi2 | 1.99 | 1.83 | 1.99 | 1.98 | 2.00 | 2.00 | 2.00 | 1.83 |
| 24->32 | psi1 | 2.00 | 1.98 | 2.01 | 1.99 | 2.00 | 1.95 | 2.00 | 1.98 |
| 24->32 | psi2 | 1.99 | 2.04 | 1.99 | 2.01 | 2.00 | 2.00 | 2.00 | 2.04 |

## Reading (recorded, no threshold)

- At 32^3 the GPU production oracle differs from the SF-29 spectral oracle by 0.58 % / 0.64 % (RMS, psi1 / psi2,
  relative to the fluctuating part of the labels; max 1.9 % / 1.7 %). The difference converges at observed order
  2.00 (RMS) on 16->24 and 24->32 for both labels, the order of the SF-19 2nd-order Darcy discretization (the step-8
  cross-check measured 1.92-2.04 for the SF-19 inlet `v1` / spline `v_perp`, `analysis/sf19_crosscheck.md`).
- The inlet plane alone (plane 0: D-1 labels from the SF-19 face flux vs the spectral face velocity) carries a
  difference of the same size and order (rms_inlet 5.4e-3 / 4.8e-3 at 32^3, order 2.00) as the interior planes
  (5.8e-3 / 6.4e-3): the oracle difference is dominated by the inlet labels' SF-19 input plus the same-order
  difference of the traced flow; the integration itself contributes < 1e-8 (round trips, ladder differences).
- A non-zero mean difference (label gauge of the D-1 construction applied to two discretely different inlet fluxes)
  is part of the difference and converges at the same order (2.0); it is not removed in the headline numbers.
- On the 16^3 case the local RTX 3050 run (debug build, development check of the `--cells` option) gave the same
  table entries to the printed digits.

Caveat: one field, one amplitude (0.25), three grids; the SF-29 oracle is itself a numerical object (DOP853 on a
spectral Darcy solve with its own tolerance, round trips recorded in its `case.json`), assumed exact at the level of
these differences.
