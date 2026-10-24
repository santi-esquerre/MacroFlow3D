# SF-41 pre-activation C2/C3/C4 design emulations (numpy) — design emulation, not SF-41 evidence

Base: master `a3d98ae`. SF-41 is not activated; nothing here is a pre-registered threshold or a measured SF-41 result.
Every script was run as
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I <script> > analysis/<name>.txt`
(numpy 2.5.0, scipy 1.18.0; deterministic: two runs of `c2_tau.py` and `c3_shift.py` gave identical md5; no timestamps).
Outputs: `c2_tau.txt` (`scripts/c2_tau.py`), `c3_shift.txt` (`scripts/c3_shift.py`), `c4_lobe.txt` (`scripts/c4_lobe.py`);
shared code `scripts/c_common.py` (splines, DP5(4), exact discrete heads, plane pass with Henon landing, quadrature).
Not modelled: FMA contraction, device libm, the solver (heads are written exactly), the 3-D transverse components
(justified below), the real injection code.

## 1. What reproduced the prior in-memory numbers

| item | prior | this emulation | |
|---|---|---|---|
| SF-38 F2, h = 1/32, 64x32 and 128x32 (periodic splines) | 4.771e-10 (measured V100) | 4.771e-10, 4.771e-10 | reproduced |
| SF-36 A2 / F2 at h = 1/16 | 4.6e-9 (measured) | 4.620e-09 | reproduced |
| SF-36 A2 at h = 1/32 | 4.77e-10 (measured) | 4.771e-10 | reproduced |
| C2 candidate, nak f + nak ln k, 64x32 / 128x32 | 5.0e-11 | 5.023e-11 / 5.023e-11 | reproduced |
| candidate at h = 1/16 | 2.0e-10 | 1.981e-10 (L1 = 2 and 4) | reproduced |
| candidate with periodic ln k spline | 0.93e-9 / 1.35e-9 | 9.323e-10 / 1.347e-09 | reproduced (fails the gate at L1 = 4) |
| profile 0.5 cos(2 pi x1 + 0.3) | 2.44e-9 | 2.442e-09 | reproduced |

Nothing failed to reproduce. The predictions below therefore come from a code path that matches three measured
values to all printed digits (the F2 reproduction uses the exact discrete periodic head; the measured runs used
solved heads, so the tau-vs-quadrature number is insensitive to the head's solver error at this level).

## 2. C2 (tau against an independent quadrature of 1/v1 on the same spline)

Why 1-D suffices: samples depend on x1 only, so both spline transverse gradients vanish to roundoff, `|g| = |g1|`,
`dx1/ds = g1/|g| = 1`, and the positions-only error estimate (the DP5 `e_i` sum to zero) is rounding-level, far below
tol = 1e-8. Every step is accepted with factor 5: steps are 0.1, 0.5, 1, 1, ... x h_max (h_max = h/8), no adaptivity;
tau rides passively, so the tau error is the DP5 quadrature error at fixed step plus the Henon-landing steps.
The error is a sum of knot-straddling errors: the grid is offset 0.6 h_max from the spline knots (multiples of
h/2 = 4 h_max) until the first landing, which re-aligns it (`c2_tau.txt` section vi: single plane 0.5 gives 4.1e-10,
planes {0.25, 0.5} give 1.3e-11 at both). Max tau relative error over the planes (`c2_tau.txt`):

| case (medium (b), exact discrete Dirichlet head, X0 = 0, tol 1e-8, h_max = h/8) | planes | max tau rel | gate 1e-9 |
|---|---|---|---|
| **C2 candidate** ln k = 0.5 cos(2 pi 0.6 x1 + 0.3), nak f, nak ln k, 64 cells (L1 = 2) | .25 .5 1 1.5 1.75 | **5.023e-11** | pass |
| same, 128 cells (L1 = 4) | .25 .5 1 2 3 3.75 | **5.023e-11** | pass |
| periodic ln k spline (nak f), L1 = 2 / L1 = 4 | as above | 9.323e-10 / 1.347e-09 | pass (marginal) / FAIL |
| profile 0.5 cos(2 pi x1 + 0.3), nak/nak, L1 = 2 / 4 | as above | 2.442e-09 / 2.442e-09 | FAIL |
| candidate at h = 1/16, L1 = 2 / 4 (32 / 64 cells) | as above | 1.981e-10 / 1.981e-10 | pass |
| F2 profile 0.5 cos(2 pi x1), medium (b), nak/nak, L1 = 2 / 4 | as above | 6.334e-10 / 6.334e-10 | pass |

Sensitivity of the candidate (h = 1/32, L1 = 2, nak/nak): phase scan nu = 0.6, phi = 0.1..0.5: 1.7e-10, 1.2e-10,
5.0e-11, 5.0e-11, 1.7e-10 (stable); but nu = 0.6, phi = 1.0: 1.05e-9 (fails); nu = 0.9: 1.2e-9 (phi 0.3), 5.0e-9
(phi 1.0); nu = 1.5: 1.8e-8; amplitude 0.25 instead of 0.5 is not monotone (2.7e-10 at nu 0.6 phi 0.3). The margin of the
candidate (20x) is real for this exact profile, plane list and phase, but the gate is not robust to the profile choice:
fix the profile (nu = 0.6, phi = 0.3, amplitude 0.5) in the activation row rather than describing it loosely.

## 3. C4 (inlet exits on a harmonic k = 1 field with a backflow lobe)

Field: G1 = 1, f = a cos(2 pi x2) exp(-2 pi x1), psi = G1 x2 - a exp(-2 pi x1) sin(2 pi x2); x1s = ln(2 pi a)/(2 pi)
= 0.18219 (a = 0.5), 0.14668 (a = 0.4). X0 = 0.1, h = 1/32, grids 64x32x32 (L1 = 2) and 128x32x32 (L1 = 4).
Methods: A = analytic field + DOP853 (rtol 1e-12; max |psi_end - psi0| 8.4e-14); B = DP5(4) lock-step on the 2-D
not-a-knot x periodic spline of f at cell centres (tol 1e-8, h_max = h/8; the Henon landing is not emulated: it does not
change the status); C = Pollock RT0 on analytically integrated face fluxes (max |div_cell| = 1.4e-14).
Findings (`c4_lobe.txt`):

* Exact outcomes at the seed: for a = 0.5 the v1 > 0 seeds with |x2| in (0.1482, 0.2655) exit through the inlet, those
  beyond reach the outlet; the inlet-exit seeds are bounded by the v1 = 0 curve (psi0 = -/+ 0.0658) and by the separatrix
  branch (psi0 = 0 at |x2| = 0.2655). a = 0.4 has a thinner lobe (psi0 in +/-0.026).
* Scan lattice Ms = 512 (360 seeds with v1 > 0 for a = 0.5, 394 for a = 0.4): 0 disagreements of B_nak or C with A in all four
  (a, grid) cases.
* Dense scans (401 seeds, spacing 5e-5 in x2) through the separatrix branch: B_nak disagrees only at the seed
  placed exactly on the separatrix (|psi0 - psi_sep| = 0 to 3e-16, a coin flip); C disagrees for margins up to
  **4.64e-4** (a = 0.5) / 3.74e-4 (a = 0.4) (RT0 shifts the separatrix by O(4e-4)). Dense scan through the v1 = 0 curve at
  the seed (analytic v1 down to 6.3e-4 for a = 0.5, 4.2e-4 for a = 0.4): 0 disagreements. So a margin delta >= 1e-3 is enough for these representations; delta = 0.02
  leaves a factor >40.
* Periodic (SF-28) closure in x1 for the same samples (diagnostic): 36 of 40 included seeds are not traceable
  (step collapse near the x1 ends, cap_exceeded), 4 exit_outlet — unusable for medium (b), as expected.

Recommended parameters (a = 0.5, G1 = 1, X0 = 0.1, M = 64 lattice x2 = (m + 1/2)/64 - 1/2, any x3; include a seed iff
analytic v1(seed) >= 0.05 and |psi0 - round(psi0)| >= 0.02; plane x1 = L1/2; T = 3.0 for L1 = 2, 5.5 for L1 = 4):

| a | grid | x2 values included / 64 | 3-D seeds (x M) | exit_inlet | outlet | min margin | min v1 | disagreements B_nak / C | max exit tau A (B, C) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 64x32x32 | 40 | 2560 | 768 (12 x 64) | 1792 (28 x 64) | 2.4235e-2 | 0.1384 | 0 / 0 | 2.1600 (2.1629, 2.1586) |
| 0.5 | 128x32x32 | 40 | 2560 | 768 | 1792 | 2.4235e-2 | 0.1384 | 0 / 0 | 4.1601 (4.1629, 4.1586) |
| 0.4 | 64x32x32 | 38 | 2432 | 256 (4 x 64) | 2176 | 2.2963e-2 | 0.0996 | 0 / 0 | 2.1714 (2.1720, 2.1700) |
| 0.4 | 128x32x32 | 38 | 2432 | 256 | 2176 | 2.2963e-2 | 0.0996 | 0 / 0 | 4.1714 (4.1720, 4.1700) |

24 (a = 0.5) / 26 (a = 0.4) of the 64 x2 values are excluded (v1 < 0.05 near the lobe, or within delta of the separatrix).
Plane pass (inlet exits + reached plane) and time pass (inlet exits + outlet exits at the recommended T) give identical
counts for A, B_nak and C. Larger M (128) and other
(delta, v1min) pairs also agree (`c4_lobe.txt` section 2); a = 0.4 with delta = 0.03, v1min = 0.10 loses all inlet exits
(do not use). The test needs an explicit seed list (or a filter of the M x M lattice by the analytic psi0 and v1).

## 4. C3 (shift by +1 in x2 / x3)

Point set of SF-38 F3 (4096 points, 1/4 pushed to the edge list), numpy PCG64 seed 20261022 (not the F3 mt19937 stream),
not-a-knot x1, periodic x2, x3, 3-D prefilter by dense per-axis solves. numpy has no FMA and sums in a different
order: **these are magnitude estimates of the roundoff, not bit-level predictions**. Max over 4096 points
(`c3_shift.txt`; metric: per component abs <= 1e-14 and <= 1e-14 |v|):

| field | grid | max abs (v1, v2, v3), shift x2 | max abs, shift x3 | max rel (any) | violations |
|---|---|---|---|---|---|
| V0 as specified (a = 0.5, ln k = 0.3 sin cos) | 64x16x16 | 3.1e-15, 3.2e-15, 9.4e-16 | 4.7e-15, 1.9e-15, 6.7e-16 | 1.10e-14 (v2, at |v| = 0.05) | 1 |
| V0 | 128x16x16 | 2.6e-15, 3.2e-15, 9.4e-16 | 3.6e-15, 2.0e-15, 6.7e-16 | 7.0e-15 | 0 |
| V1 x3-dependent harmonic f + x1 profile in ln k (a = 0.5) | 64x16x16 | 8.0e-15, 5.0e-15, 2.8e-15 | 7.1e-15, 2.7e-15, 4.6e-15 | 5.0e-15 | 0 |
| V1 | 128x16x16 | 8.0e-15, 5.0e-15, 1.8e-15 | 7.1e-15, 2.7e-15, 4.6e-15 | 2.3e-15 | 0 |
| V2 = V0 with a = 0.1, ln k amp 0.1 | 64x16x16 | 7.8e-16, 7.1e-16, 1.7e-16 | 1.0e-15, 2.2e-16, 1.9e-16 | 2.5e-15 | 0 |
| V2 | 128x16x16 | 7.8e-16, 7.1e-16, 1.7e-16 | 1.0e-15, 2.4e-16, 1.5e-16 | 2.5e-15 | 0 |
| V3 = V1 with f amp 0.1, ln k amps 0.1 / 0.2 | 64x16x16 | 1.1e-15, 6.4e-16, 3.9e-16 | 1.1e-15, 3.6e-16, 6.6e-16 | 3.7e-15 | 0 |
| V3 | 128x16x16 | 1.0e-15, 6.3e-16, 2.5e-16 | 1.1e-15, 3.3e-16, 5.5e-16 | 3.7e-15 | 0 |

Dyadic points k/2^16 (4096 per field, shift exact in floating point): v(x + e) == v(x) bitwise for all four fields,
both shifts, both grids (0 mismatches).
Reading: the error is `|d v/d x_perp| x 1.1e-16` (rounding of x + 1); with the specified amplitudes (|v| up to 4.7 and
`|v| ~ 0.05` near the inlet lobe) the 1e-14 absolute/relative gate is only 1-2x away (V0 64x16x16 already violates it
once in the relative metric, V1 reaches 8e-15 absolute). Recommended: V3 or V2 (amplitudes <= 0.1, `|v| >= 0.2`):
predicted <= 1.1e-15 absolute and <= 3.7e-15 relative, margin >= 9x absolute / 2.7x relative. V3 also exercises
the x3 wrap of the potential spline and the not-a-knot x1 closure of ln k.

## 5. Residual risks of these emulations

* The C2/C4 emulations use the exact discrete heads and written samples; the real stage's tracker also integrates
  covectors and I, which ride passively and do not change positions/tau (SF-36 design), and uses FMA-free device
  arithmetic (`--fmad=false`), so tau agreement to ~1e-16 relative of these numbers is expected, not bitwise.
* The tracker's exit semantics for x1 < 0 / x1 >= L1 are not yet implemented; B uses "an accepted step ends outside";
  the real code may resolve the exit point differently (status counts are unaffected by that choice for seeds with
  margin >= delta, tau_exit differs by O(h_max)).
* The Pollock non-periodic mode is emulated from the header's contract (cell stepping, log1p/expm1 exits), not from the
  future implementation.
