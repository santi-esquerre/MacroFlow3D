# SF-41 pre-activation review: thresholds A and B (design emulation, numpy; NOT SF-41 evidence)

Everything below is **design emulation (numpy), not SF-41 evidence**. SF-41 is not activated; nothing here is an SF-41 result, and no
SF-41 threshold is tuned or re-registered by this note. Emulated spline: uniform cubic B-spline on the SF-28 cell-centred lattice;
x1 closure = not-a-knot on the raw samples (no ghost samples), unknowns `c_{-1}..c_n`, plus continuation coefficients `c_{-2}`,
`c_{n+1}` (4th difference = 0 at `x_0`, `x_{n-1}`), evaluation domain `[-h/2, L1 + h/2)`; periodic in x2/x3 as SF-28.
x line solve variants: `reduced` (not-a-knot rows eliminated analytically: `c_1 = (8 f_1 - f_0 - f_2)/6`, then a plain `[1 4 1]`
tridiagonal solve), `dense` LU, `sparse` LU. y/z prefilter variants: `fft` (GPU-like) and `cyc` (cyclic tridiagonal + Sherman-Morrison,
CPU-mirror-like). Environment: python3 3.x, numpy 2.5.0, scipy 1.18.0.

Files (all under `docs/experiments/artifacts/2026-10-24-sf41-pre-activation-review/`): `scripts/a_spline.py` (shared emulation),
`scripts/a_ladder.py`, `scripts/b_roundoff.py`, `analysis/a_ladder.txt`, `analysis/b_roundoff.txt`.

Commands (from the artifact directory; each re-run reproduces the analysis files byte for byte, checked with `cmp` on a second run):

```
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I scripts/a_ladder.py  > analysis/a_ladder.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I scripts/b_roundoff.py > analysis/b_roundoff.txt
```

## 0. Validation of the emulation

| check (a_ladder.txt section 0) | result |
|---|---|
| 1-D vs `scipy.interpolate.CubicSpline(bc_type='not-a-knot')`, natural field and `exp(x) sin(3x)`, n = 32/64/128, 5000 points incl. both margins, relative to max of the same derivative order | sparse/dense: value <= 1.1e-15, d1 <= 1.7e-14, d2 <= 3.7e-13; reduced: value <= 4.1e-15, d1 <= 6.7e-14, d2 <= 1.2e-12 (round-off growth ~ 1/h^k; the `reduced` elimination is 3-5x noisier than LU) |
| 3-D with y/z-constant data vs 1-D | value 1.8e-15, d/dx 4.3e-14, d2/dx2 6.9e-13 (fft: 1.1e-15 / 3.8e-15 / 7.2e-14) |
| node interpolation, random 32x16x16 samples, all 6 prefilter variants | <= 1.9e-15 (A gate 1e-12) |

## B. Cross-representation round-off and end effect (`b_roundoff.py`, `analysis/b_roundoff.txt`)

Reference P = periodic SF-28 spline; candidate NP = not-a-knot spline; same samples; 1e5 random points >= 2 lambda from both x faces plus the
cell centres of 3 x1 lines (4608 points); seed 6001; norms: value max/max, gradient Euclidean per point, Hessian component-wise max / max|H|.
Pairs: P1 = FFT periodic vs (y/z FFT + sparse-LU x) [the original probe]; P2 = FFT periodic vs (y/z cyclic tridiagonal + reduced x);
P3 = cyclic-tridiagonal periodic vs P2's NP. Drivers: head1 (linearised head fluctuation, probe driver), head2 (`Z = Y + (Y^2-1)/2`),
Klike (`exp(2Y) - mean`).

1664x128x128, lambda/h = 32 (B gate 1e-11):

| driver | pair | value | gradient | Hessian | fitted per-knot factor (west / east, knots > 1e3 x floor) |
|---|---|---|---|---|---|
| head1 | P1 | 7.6e-16 | 4.9e-14 | 2.2e-12 | 0.2680 / 0.2679 (10 / 10 knots) |
| head1 | P2 | 9.1e-16 | 6.0e-14 | 3.4e-12 | 0.2679 / 0.2679 |
| head1 | P3 | 7.6e-16 | 4.1e-14 | 1.9e-12 | 0.2679 / 0.2680 |
| head2 | P1 / P2 / P3 | 6.8e-16 / 8.5e-16 / 6.8e-16 | 2.9e-14 / 3.3e-14 / 2.1e-14 | 1.0e-12 / 1.2e-12 / 7.9e-13 | 0.2679 |
| Klike | P1 / P2 / P3 | 4.4e-16 / 7.6e-16 / 2.2e-16 | 3.2e-15 / 7.2e-15 / 4.2e-15 | 2.7e-14 / 4.6e-14 / 3.8e-14 | 0.2679 |

All within 1e-11 with a margin of >= 2.9x (worst: Hessian 3.4e-12 for head1/P2). The worst quantity is the Hessian (floor ~ eps (lambda/h)^2 order).
Coefficient end-effect profile (head1/P1, west): 4.9e-7, 1.3e-7, 3.5e-8, ... decays by `2 - sqrt(3) = 0.2679` per knot and reaches the
interior coefficient floor (1.5e-15) at knot ~15; the floor of the P2/P3 variants (cyclic y/z) is 2-3x higher (2.5e-15..4.7e-15; interior coefficient floor).

416x32x32 (L1 = 13), seeds 6001 and 6002: at lambda/h = 32 (here `ell = 1` = the box length: few Fourier modes, a degenerate case) all 18 rows
are <= 2.4e-12 in the Hessian (value <= 1.1e-15, gradient <= 5.4e-14). **At lambda/h = 8** (same physical `lambda = 1/4` on 8x coarser cells; "2 lambda" = 16 knots)
the gate is **exceeded**: head1/head2 value 9.6e-14..3.5e-13, gradient 2.9e-12..8.8e-12, Hessian 3.0e-11..7.8e-11 (Klike 3.4e-12..9.2e-12 passes). Mechanism:
`0.2679^(2 lambda/h)` is 7e-10 at 16 knots but 2e-37 at 64 knots, so the B distance rule "2 lambda" is only safe for lambda/h >~ 16-32; the
fitted factor is unchanged (0.2679). The B box (lambda/h = 32) is not affected.

## A. Spline contracts (`a_ladder.py`, `analysis/a_ladder.txt`)

### (a) Reproduction of the prior review numbers

1-D natural field `sin(2 pi 1.1 x + 0.3) + 0.5 cos(2 pi 0.6 x)`, 1e4 random + 2000 points within 1/16 of each face, max norms, min orders over 16->32 and 32->64:

| L1 | value | grad | Hessian | gates 3.8 / 2.8 / 1.8 | prior in-memory |
|---|---|---|---|---|---|
| 1 | 3.8612 | 2.8312 | 1.7697 | PASS / PASS / FAIL | 3.88 / 2.84 / 1.77 |
| 2 | 3.7879 | 2.8013 | 1.8308 | FAIL / PASS / PASS | 3.79 / 2.80 / 1.83 |

300 random 3-mode fields (`kappa/2pi ~ U(0.2,0.6)`, random phases; amplitude law not recorded by the prior review, four choices run; seed 20261024), fields failing
the gates (value/grad/Hessian) as `min over 16->32,32->64 | same with gate+0.05 | 64->128 alone`:

| amplitude | L1 | min over both pairs | gate+0.05 | 64->128 alone |
|---|---|---|---|---|
| U(0.5,1) | 1 | 29/31/30 | 38/38/38 | 8/8/6 |
| U(0.5,1) | 2 | 48/48/42 | 61/62/59 | 4/4/2 |
| U(0,1) | 2 | 45/43/43 | 62/62/55 | 8/8/6 |
| 1 | 2 | 57/58/49 | 73/72/68 | 8/7/5 |
| N(0,1) | 2 | 46/45/40 | 62/58/52 | 2/2/1 |
| (prior) | ? | 49/48/49 | - | 6/6/5 |

(U(0,1), 1, N(0,1) at L1 = 1 and the smallest 128-cell errors, all >= 4e-10 / 2e-7 / 5e-5, are in the file; no round-off contamination.)

### (b) Test-field selection for threshold A (criterion fixed in `a_ladder.py` before any order was computed; all six candidates reported)

Boxes (32,16,16), (64,32,32), (128,64,64), L1 = 2; 14000 fixed points (all inside `[0,L1]`); `S = sin(2 pi x2) cos(2 pi x3)`. CRIT-1: finest-level argmax of
value and Hessian error within 1/16 of an x face. CRIT-2: min orders (16->32, 32->64) >= gate + 0.05 for value max/RMS, gradient max/RMS, Hessian max.

| candidate | min orders val max / val RMS / grad max / grad RMS / Hess max | min margin over gate+0.05 | CRIT-1 | CRIT-2 | (post hoc) x-closure share of finest val / Hess error |
|---|---|---|---|---|---|
| C1 `exp(0.7 x1)(1+0.3 S)` | 4.0062 / 4.0148 / 3.0065 / 3.0169 / 1.9845 | +0.1345 | PASS | PASS | 0.022 / 0.010 |
| C2 `exp(x1) sin(3 x1)(1+0.3 S)` | 3.9109 / 4.3575 / 2.9021 / 3.3816 / 1.9152 | +0.0521 | PASS | PASS | 0.766 / 0.776 |
| C3 `exp(x1) sin(3 x1)(1+0.05 S)` | 3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154 | +0.0524 | PASS | PASS | 0.952 / 0.954 |
| C4 `exp(x1) sin(3 x1)(1+0.01 S)` | 3.9006 / 4.3668 / 2.9046 / 3.3863 / 1.9155 | +0.0506 | PASS | PASS | 0.991 / 0.991 |
| C5 `exp(0.7 x1)(1+0.01 S)` | 3.9995 / 4.0548 / 3.0252 / 3.0977 / 1.9845 | +0.1345 | PASS | PASS | 0.455 / 0.294 |
| C6 `natural1D(x1)(1+0.01 S)` | 3.7884 / 4.5074 / 2.8015 / 3.5226 / 1.8309 | -0.0616 | PASS | FAIL | 0.991 / 0.991 |

**Recommended by the stated rule (largest minimum margin among passing candidates): C1** (C5 ties at +0.1345). Predicted orders (4 decimals):
value max 4.0062 / 4.0615 (16->32 / 32->64), value RMS 4.0594 / 4.0148, gradient max 3.0065 / 3.0512, gradient RMS 3.0612 / 3.0169, Hessian max 1.9845 / 2.0341.
Errors at h = 1/16, 1/32, 1/64: value max 1.457354e-04, 9.069405e-06, 5.431978e-07; value RMS 2.890388e-05, 1.733620e-06, 1.072462e-07; gradient max 5.242634e-03,
6.523718e-04, 7.870193e-05; gradient RMS 1.245374e-03, 1.492027e-04, 1.843283e-05; Hessian max 5.711422e-01, 1.443310e-01, 3.524033e-02.

**Caveat found after the orders were seen (post-hoc diagnostic "(b2)", does not change the rule's pick):** C1 passes CRIT-1 only because the y/z
factor (`0.3 S`) is multiplied by `exp(0.7 x1)`, which grows toward the east face; the 3-D error of C1 is 98% (value) / 99% (Hessian) the periodic y/z spline error, so
its 3-D ladder says almost nothing about the new x closure. The x closure alone (same `g(x1)`, `a = 0`) has min orders 3.9567 / 4.4305 / 2.9583 / 3.4477 / 1.9623 (passes
gate+0.05). If the owner wants the 3-D ladder itself to exercise the closure, C3 is the candidate that does so and still passes (x share 0.95, margin +0.0524; value max
3.9024 against the raw gate 3.8, i.e. +0.10); otherwise C1 should be paired with (c). Candidates C2-C4 share the same x-only orders (3.8998 / 4.3670 / 2.9045 / 3.3864 / 1.9153),
so their margin is the x closure's own and is small (+0.05 over gate+0.05) by construction of that profile.

### (c) x-only near-face sub-ladder for C1's `g(x1) = exp(0.7 x1)` (1-D, L1 = 2, 4000 points within 1/16 of the faces; recorded quantity, no gate)

| pair | value max | value RMS | grad max | grad RMS | Hess max |
|---|---|---|---|---|---|
| 16->32 | 3.9569 | 4.4460 | 2.9584 | 3.4881 | 1.9624 |
| 32->64 | 3.9789 | 4.4740 | 2.9795 | 3.4842 | 1.9813 |
| 64->128 | 3.9905 | 4.4371 | 2.9904 | 3.4687 | 1.9910 |

### (d) Polynomial reproduction through the prefilter (boxes 64x16x16 and 128x16x16, h = 1/16; 10000 random + 2 x 2000 near-face + 256 margin points)

`p = (x1 - L1/2)^2` and the cubic `1 + 0.5 t - 0.7 t^2 + 0.3 t^3`, `t = x1 - L1/2`; normwise relative errors (gate 1e-12, norm unspecified in the spec):

| quantity | nx = 64 (range over 8 rows: p/cubic x yz fft/cyc x reduced/dense) | nx = 128 |
|---|---|---|
| value | 1.3e-15 .. 3.7e-15 | 1.1e-15 .. 4.4e-15 |
| gradient | 2.3e-14 .. 1.2e-13 | 3.6e-14 .. 2.7e-13 |
| Hessian (component max) | 7.5e-13 .. 7.7e-12 | 1.3e-12 .. 2.9e-11 |

Hessian > 1e-12 in 13 of 16 rows (all 8 at nx = 128). Best case (yz fft, x dense): 8.3e-13 / 7.5e-13 at nx = 64 and 3.7e-12 / 1.3e-12 at nx = 128. The error is round-off amplified by `1/h^2`
(coefficient noise gain 33 at `c_{-2}`, see (f)) and, for `yz=cyc`, mostly in the spurious y/z/cross Hessian components (up to 2.9e-11).

### (e) Partition of unity (constant 3.0; boxes of the ladder; gate 1e-14)

Value max abs error: dense LU 4.4e-15..5.3e-15 on `[0,L1]` (14000 points) and 7.1e-15..7.6e-15 in the margins; reduced 2.7e-15..3.6e-15 and 2.2e-15..3.1e-15; sparse 3.1e-15..3.6e-15 and 3.6e-15..4.4e-15
(y/z cyclic: 2.7e-15..7.1e-15 over all rows). Gradient max abs (dense, yz fft; h = 1/16, 1/32, 1/64): 1.07e-13, 2.02e-13, 3.86e-13 on `[0,L1]`; 1.50e-13, 2.86e-13, 6.09e-13 in the margins
(reduced: 2.9e-14, 6.8e-14, 1.2e-13 and 2.5e-14, 4.4e-14, 8.9e-14). The gradient errors scale as `1/h` and are 2.5-61x above 1e-14 if the gate applies to the gradient; the value gate has a margin of 1.3x
(dense) to 2.8x (reduced) at the worst point. Margin points fixed in physical space: `x_m = -1/128` is at `x/h = -0.125, -0.25, -0.5` for h = 1/16, 1/32, 1/64 (inside; exactly at the limit `-h/2` for 1/64) and outside the domain for h <= 1/128.

### (f) CPU-vs-GPU surrogate (y/z FFT vs cyclic tridiagonal; same reduced x solve; relative to the max of the reference)

Recommended field C1, full pipeline (value / gradient / Hessian):

| h | interior (0.5 <= x <= L1 - 0.5) | within 1/16 of a face | margins `[-h/2,0)`, `[L1, L1+h/2)` (= ALL) | periodic SF-28 pair, interior |
|---|---|---|---|---|
| 1/16 | 5.0e-16 / 8.2e-15 / 5.9e-14 | 2.7e-15 / 7.3e-14 / 2.9e-13 | 3.7e-15 / 8.6e-14 / 5.3e-13 | 7.2e-16 / 1.1e-14 / 9.6e-14 |
| 1/32 | 4.2e-16 / 1.8e-14 / 2.4e-13 | 1.5e-15 / 9.8e-14 / 8.4e-13 | 4.7e-15 / 1.9e-13 / 1.6e-12 | 8.4e-16 / 2.3e-14 / 3.2e-13 |
| 1/64 | 5.9e-16 / 3.6e-14 / 1.1e-12 | 1.7e-15 / 1.7e-13 / 3.3e-12 | 5.6e-15 / 4.6e-13 / 8.2e-12 | 8.4e-16 / 5.1e-14 / 1.4e-12 |

Stress field (Gaussian, lambda/h = 8, seed 6001): ALL = 3.1e-15 / 3.1e-14 / 2.0e-13 (h = 1/16), 1.8e-15 / 2.1e-14 / 1.7e-13 (1/32), 4.1e-15 / 3.5e-14 / 2.1e-13 (1/64); interior Hessian 2.5e-14..2.9e-14.
Noise gain of the x line solve for white noise on the samples (rms): `c_{-2}` 33, `c_{-1}` 8, `c_0..c_3` 1.3-1.9, middle 1.85 (same at both faces, n = 32, 128, 1664): the near-face and margin points inherit
the y/z prefilter difference amplified up to 33x, then multiplied by `1/h` and `1/h^2`. The interior difference is already as large as that of the existing periodic pair. At the coefficient level the NP difference is 4.6e-14 / 5.0e-14 / 6.5e-14 (h = 1/16, 1/32, 1/64; SF-28 states its CPU/GPU contract on coefficients at 1e-13): the 1e-13 gate is met on coefficients and by the value, but not by the evaluated gradient (> 1e-13 from h = 1/32 near faces) or Hessian; the spec line does not say which quantity the gate bounds.

## Confirmed vs not reproduced (prior in-memory numbers)

| item | status |
|---|---|
| B probe 1664x128x128, lambda/h 32: 6.1e-16 / 4.7e-14 / 2.4e-12 | **confirmed** (head1/P1: 7.6e-16 / 4.9e-14 / 2.2e-12; differences are the field normalisation/seed stream) |
| B per-knot factor 0.268 over knots 0..12, floor ~14 knots | **confirmed** (0.2679-0.2680 over 9-11 knots > 1e3 x floor; floor reached at knot ~15) |
| (1) natural 1-D orders L1 = 1, 2 | **confirmed** to <= 0.02 (point set differs) |
| (1) 300 random 3-mode fields: 49/48/49 fail, 6/6/5 on 64->128 | **order of magnitude confirmed, counts not reproduced** (42-58 of 300; 2-8 on 64->128; the amplitude law is not recorded) |
| (1) `exp(0.7x)`, `exp(x) sin(3x)` pass by 0.10-0.16 | **confirmed in 1-D x-only** (margin over the raw gates 0.10-0.16: C1 x-only 3.9567/2.9583/1.9623; C2 x-only 3.8998/2.9045/1.9153); 3-D with y/z structure also passes (C1: 4.0062 / 4.0148 / 3.0065 / 3.0169 / 1.9845 vs the prior's >= 4.11 / 4.09 / 3.01 / 3.10 / 1.98: same structure, value orders slightly lower) |
| (2) cubic reproduction, Hessian 1.5-5.3e-13 (nx 64), 1.2-1.4e-12 (nx 128), gradient <= 3.4e-14 | **not reproduced**: same mechanism, 2-5x larger (8e-13 / 7.5e-13 at nx 64; 3.7e-12 / 1.3e-12 at nx 128 for yz fft, x dense; up to 2.9e-11 with cyclic y/z); gradient 2.3e-14..2.7e-13. Polynomial scaling and norm of the prior are not recorded |
| (3) partition of unity: value <= 3.1e-15 on `[0,L1]`, 7.5e-15 in margins (dense LU); gradient 1.1e-13 / 2.2e-13 / 4.1e-13 | **margin value and gradient confirmed** (7.6e-15; 1.07e-13 / 2.02e-13 / 3.86e-13); value on `[0,L1]` 4.4e-15..5.3e-15 for dense (3.1e-15 matches the reduced solve) |
| (4) CPU-vs-GPU: value <= 8.3e-16, gradient <= 5.2e-15, Hessian 5.8e-15 / 1.7e-14 / 6.7e-14 | **not reproduced for C1**: interior value 6e-16 matches, but gradient 8e-15..4e-14 and Hessian 6e-14..1.1e-12 (interior), up to 4.6e-13 / 8.2e-12 near faces and in the margins; the stress Gaussian field (interior Hessian 2.5e-14..2.9e-14) is the same order as the prior. The prior field and point set are not recorded |
| (5) margin points fixed in physical space leave the domain on finer grids | **confirmed** (`-1/128` is `-h/2` at h = 1/64, outside for h <= 1/128) |
