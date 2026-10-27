# SF-39 pre-activation scientific review of the macrodispersion study (literature, code reading, adversarial reviews, own calculations)

- Date: 2026-10-27
- Status: `complete — design review before activation; NOT SF-39 evidence; no GPU run, no build, no SF-39 data; the calculations are numpy/scipy on CPU`
- Increment: [`SF-39-macrodispersion-study.md`](../plans/active/lester-eq14/increments/SF-39-macrodispersion-study.md) (State `pending` at the time of this
  review; base `master 782c9d9a6c377b4def376aad4eee51ee6224a6c2`)
- Decision fed: [`2026-10-27-sf39-pre-activation-decisions.md`](../decisions/2026-10-27-sf39-pre-activation-decisions.md) (F1-F14)
- Precedents: [`2026-10-24-sf41-pre-activation-review.md`](2026-10-24-sf41-pre-activation-review.md); decision 2026-10-23 D21 (SF-38 threshold A
  amended before activation for an established cause)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) (Rule 1: no `alpha_L` / `alpha_T` statement is made here),
  [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md); post-hoc context
  [`2026-10-23-resolution-vs-variance.md`](2026-10-23-resolution-vs-variance.md)
- Artifact: [`artifacts/2026-10-27-sf39-pre-activation-review/scripts/`](artifacts/2026-10-27-sf39-pre-activation-review/scripts/README.md)

## 1. Question and scope

Before the activation row pre-registers SF-39's thresholds verbatim, (a) is the methodology (estimators, decision rules, ladders, domains)
executable and free of rules that mislabel by construction, and (b) is it adequate to the physical question the Goal asks — alpha_L and
alpha_T at `sigma^2` in {1, 2.25, 4} with the Lagrangian tracker, with convergence in `lambda/h`, tolerance and realization count, `alpha_L`
against Par2, and each transverse growth classified physical, numerical or unresolved — given the state of the art?

Hypothesis tested: none about `alpha`. The review hypothesizes only that some specification items fail for derivable reasons (undefined
quantities, a rule whose pass probability is computable without data), which is checked by reading, derivation and simulation.

Scope: **no SF-39 data exist** (seeds 5001+ are unused; nothing was run for this review on the V100). No `alpha` claim is made or implied.
Every figure below is a design calculation, a literature value, or a post-hoc reading of earlier increments' data.

**Date of knowledge (integrity statement, same substance as decision 2026-10-27).** The SF-39 thresholds D15 were fixed before any
`sigma^2 = 4` plume statistic existed. The reviewer saw, before writing this note, (i) SF-38's A/B post-hoc plume statistics
(Pollock plume, seeds 4001-4010: last-octave plane slopes, `alpha_T` of about 0.065-0.070 `lambda` at 8 `lambda` width, `sigma^2 = 4`, and the
per-realization tables of its `C_D.md`), and (ii) SF-41's F-ladder plane variances (3 seeds, 2 `lambda` section, <= 24 `lambda`). E4 and E5
(2026-10-24) were already fixed after (i). Every amendment motivated by this note is therefore data-informed in that sense; each is
classified in the decision as derivable without data or prompted by a post-hoc reading. No SF-39 realization had been generated.

## 2. Current paradigm

Locators are as read; sources marked (abstract) were not read in full text (section 8).

**Consensus: first order gives `alpha_T -> 0`.** Linear theory, 3D isotropic, purely advective: Gelhar & Axness 1983; Dagan 1984/1988;
summarized by Neuman & Zhang 1990. In 2D the zero is exact also non-linearly (Attinger, Dentz & Kinzelbach 2004 §6; de Dreuzy et al. 2007;
Beaudoin et al. 2010 ¶19). The first-order `alpha_T(tau)` for the Gaussian covariance is derived in section 5 (own calculation, independently
confirmed in closed form by a reviewer; the printed formula of Dagan 1988 was not checked).

**3D non-linear: disputed.**
- For positive asymptotic `alpha_T`: Neuman & Zhang 1990 (quasi-linear, Corrsin-type, exponential); Attinger et al. 2004 §7 (on
  *linearized* Darcy fields, not of the form `k grad phi`, not helicity-free); Jankovic, Fiori & Dagan 2003/2009 (inclusions, non-smooth K);
  **Beaudoin & de Dreuzy 2013** (Gaussian covariance eq. 1, `lambda/Delta = 10`, `L_T = 25.6 lambda`, finite-volume Darcy plus **explicit Euler
  with linear interpolation**, about 10 steps per cell, `N_p = 1e4`, `N_R = 500`): `alpha_TAM` = 0.01 / 0.044 / about 0.11 at `sigma^2` =
  1 / 2.25 / 4 (Fig. 4b), proportional to `sigma^4` (eq. 7, roughly: 4.4 and 11 against the exact-`sigma^4` 5.1 and 16 times the `sigma^2 = 1`
  value); `alpha_T(t)` non-monotone (peak `t` of about 1-3, plateau `t` of about 10-30); the only convergence check is at `sigma^2 = 6.25`
  (+-4 % in `alpha_T`); numerical artifacts are never considered; Dartois et al. 2018 (same code).
- For `alpha_T = 0` with smooth, locally isotropic, stationary K: Sposito 1994 (zero helicity implies no chaos); Lester et al. 2021 (JFM 918),
  2022 (JFM 945: bounded transverse deformation), 2023 (WRR 59), 2025 (JFM 1018: `lambda_inf = 0` in the isotropic case).
- What the Lester argument assumes: bounded fluctuations of `psi~` as an **explicit assumption** (2023 §4, eq. 27, p. 8); stationarity and
  ergodicity (§3); `Pe -> infinity` (eq. 7); no stagnation (eq. 11). What they never measured: the pseudo-symplectic tracker was not run in
  pure advection (called "trivial", p. 15); the zero of the periodic surrogate holds by construction (pp. 10, 13); no random Darcy flow
  was integrated to test the bounded-`psi~` assumption.
- Lester's positive results use non-smooth K, linearized/anisotropic flows or schemes that break the kinematic constraint: (A) reconstruction
  that breaks zero helicity (Pollock/RT0 or trilinear: `sigma^2` proportional to `Delta^2`, Fig. 3c); (B) integration that drifts off the
  streamsurfaces (RK: `sigma^2` proportional to `sqrt(tol)`, Fig. 4b). Lester 2023 p. 11 recommends interpolating `phi` and `k` separately and
  forming `v = -k grad phi` (helicity-free although not solenoidal): **this is the project's tracker construction**. Lester 2025 App. D, for
  `delta = 0`, nevertheless avoided RK5 on spline-`k grad phi` and used `psi_1`, `psi_2`, citing spurious drift: the group treats this
  construction as insufficient without drift evidence.
- Project's own finding: in the periodic cell the bounded-`psi~` assumption **fails** (closure oracle R5; SF-30 R2: Darcy streamlines do
  not close on the torus). In a random, non-periodic-in-`x1` medium it is **not established**.

**No published 3D `alpha_T` convergence study** (grid or tracking scheme) was found; Beaudoin 2010 ¶20 is the closest, in 2D. This is the gap
the project fills.

## 3. Starting point

Verified (from the closed increments' own notes; locators there):

- SF-36: tracker core correct (exact controls, GPU equals host bit for bit, identity `a x b = v_s e^{-I}` to 5e-11 on a C-infinity field).
- SF-37 (periodic cube 4 `lambda`, 3 seeds, 64 `lambda`): convergence in `h` of order about 2 (V2); integration drift `h/8` vs `h/32`,
  `MS(dx_T)/sigma_T^2 <= 1.27e-13` at `sigma^2 = 4` (RR15, the only integration evidence); `e_v` of order 2 (V4: 45/45 cells strictly
  decreasing); the tolerance ladder is **empty** (the `h/8` cap always governs); production `lambda/h` 8/16/32 (`sigma^2 = 1` raised to 10 by D1).
- SF-38: medium (b) selected by the pre-registered rule (5 equivalent, 0 non-equivalent, 5 inconclusive: not a demonstrated difference);
  per-realization tilt rms `|theta|` = 0.0227 at `sigma^2 = 4` (8 `lambda`); boundary layer about 4 `lambda`; Par2 pilot.
- SF-40/SF-41: lean stage (b) fits 30000 MiB at 3328x384^2 and 6656x256^2 (no R3 reduction); non-periodic-in-`x1` spline; explicit `x` exits;
  33.7 / 110 / 129 / 144 s per realization (E1-E4).

Not established: plume-statistic convergence in medium (b) at production length; integration drift beyond 64 `lambda`; the effect of the
transverse width; any value of `alpha`; the stretching law (V5 has no named fit; SF-30 §9 and V5 show exponential separation in the cube's
spline field — continuum property or discretization: open).

Post-hoc and noisy (not pre-registered): Pollock plume plane slopes of the SF-38 A/B (last octave 32-64 `lambda`, `N_R = 10`):
`sigma^2 = 1`: `alpha_T` of about 4e-4 `lambda` with SE of about 2.9e-3 `lambda` (not informative); `sigma^2 = 4`: about 0.065-0.070
`lambda` (width **8 `lambda`**, `lambda/h` 32).

## 4. Nature of the Lagrangian tracker (what follows from the construction)

- Velocity `v_s = k_s grad Phi_s`, `k_s = exp(C^2 spline of ln k)`, `Phi_s = G.x + C^2 spline of the finite-volume head fluctuation` (medium (b):
  not-a-knot in `x1`). **Pointwise zero helicity, exactly**: `curl(k grad Phi) = grad k x grad Phi`, orthogonal to `v`.
- Trajectories are field lines of `grad Phi_s`: `Phi_s` increases monotonically along a path, so there are no closed orbits and no
  recirculation. **The geometry of a path does not depend on `k_s`**: `k_s` fixes only the time parametrization and the flux weights. Plane
  statistics of the transverse position are therefore purely geometric (this underlies the plane estimator of section 5/6, M1).
- What does **not** follow: absence of exponential sensitivity (`Phi_s` is a Lyapunov function along the flow, not an invariant of the
  transverse map; "no chaos" is not implied by a monotone potential); absence of spurious extrema of `Phi_s` (a spline obeys no maximum
  principle, so trapping near genuine stagnation points is possible; report non-`ok` statuses by type).
- Not solenoidal: `div v_s = O(h^2)` (measured, order about 2; `e_v` 0.2-3.5 % by `sigma^2`, `h` and distance). Source (A) of Lester 2023 is
  reduced to a divergence: the helicity-breaking part is absent by construction; the divergence is not.
- Integration (B): DP5(4) in arclength, absolute position tolerance, step capped at `h_max = h/8` (the cap governs, the tolerance does not);
  no structural guarantee; with a nearly constant step the error is a smooth deterministic modification of the field, not of the form
  `k grad Phi` (weak helicity `O(h_max^p)`): measured, not guaranteed.
- Labels: the seed is the label (D-1) by definition; in medium (b) the gridded labels are NaN and the covectors are switched off in the runner.
- Contrast: Pollock/RT0 (exactly solenoidal, C0, helicity in the face laminae: violates (A)); Par2 and B&dD (explicit Euler plus face-linear
  interpolation: (A) and (B)); Par2 is essentially the B&dD scheme.
- Consequence: any transverse growth of the tracker is (i) a property of the represented flow (which converges to Darcy with `h`), (ii) a
  discretization effect (the `lambda/h` ladder) or (iii) integration drift (the `h_max` ladder). Refuting Lester 2025 App. D needs **pathwise
  drift measured at production length**, not only equivalence of `alpha`.

## 5. Own calculations

Scripts: [`artifacts/2026-10-27-sf39-pre-activation-review/scripts/`](artifacts/2026-10-27-sf39-pre-activation-review/scripts/README.md). Local CPU,
seconds each. Environment: Python 3.13.13, numpy 2.5.0, scipy 1.18.0.

### 5.1 First-order purely advective macrodispersivities (`firstorder2.py`)

3D isotropic, `C_Y(r) = sigma^2 exp(-(r/lambda)^2)`, `tau = U t / lambda`, wave-number quadrature of the linearized transverse and longitudinal
velocity covariances:

```
alpha_T(tau)/(sigma^2 lambda) = (tau/2) int_0^1 mu^2 (1 - mu^2) exp(-tau^2 mu^2) dmu
                              = sqrt(pi)/(8 tau^2) - 3 sqrt(pi)/(16 tau^4) + O(exp(-tau^2))
alpha_L(inf) = (sqrt(pi)/2) sigma^2 lambda      (= B&dD eq. 2)
X_22(inf)    = sigma^2 lambda^2 / 3
```

`alpha_T(tau)` is positive at every finite `tau`, peaks at `4.46e-2 sigma^2` near `tau = 1` and decays as `tau^-2`. (The closed form was
confirmed independently by a reviewer; the script integrates in wave number, so the table below is an independent numerical check of the
asymptote `alpha_T tau^2 -> sqrt(pi)/8 = 0.2216`.) Mean over `[0.75 tau_b, tau_b]`: `0.2953 sigma^2 / tau_b^2`.

| `tau` | 1 | 10 | 20 | 30 | 100 |
|---|---|---|---|---|---|
| `alpha_T/(sigma^2 lambda)` | 4.46e-2 | 2.18e-3 | 5.52e-4 | 2.46e-4 | 2.21e-5 |

### 5.2 Window and plane estimator biases at the measured breakthrough

- `tau_b` from Par2's measured centroid velocity (`U/J` = 1.193 / 1.470 / 1.977 at `sigma^2` = 1 / 2.25 / 4): `tau_b` = 122.6-139.4 /
  90.8-118.7 / 29.0-37.9 (upper bounds at snapshot resolution; the orchestrator's analysis also quotes about 26.7-37.9 at `sigma^2 = 4`: see
  section 8).
- First-order window bias `0.2953 sigma^2 lambda / tau_b^2`: 1.5-2.0e-5 / 4.7-8.1e-5 / 0.82-1.65e-3 `lambda` at `sigma^2` 1 / 2.25 / 4
  (checked by arithmetic: the `sigma^2 = 4` upper end 1.65e-3 needs `tau_b = 26.7`; with `tau_b` 29.0-37.9 the range is 0.82-1.40e-3).
- Plane estimator: mean of `sqrt(pi)/(8 tau^2) sigma^2` over the last half, `[47, 94] lambda`, at `sigma^2 = 4`: about `5.0e-5 sigma^2` =
  `2.0e-4 lambda`. The window bias exceeds it by 4-8 times at `sigma^2 = 4`; the plane value was computed for `sigma^2 = 4` only, so the
  advantage is not claimed for `sigma^2` 1 and 2.25.

### 5.3 Transfer-rule pass rates (`transfer.py`)

The 2026-10-23 D15 rule (narrow-ladder and production-twin relative 1/2 -> 1 differences have the same sign and a ratio in [1/2, 2]),
simulated as two independent normal estimates (4e5 draws, seed 1).

### 5.4 Medium (b) sign check on the SF-41 F ladder (`sign_planes.py`)

Per (eps, seed): plane `sigma_T^2 = (var x2 + var x3)/2` (the script sums the two variances; the sign is unaffected) at `lambda/h` 16 / 32 / 64
against the `lambda/h` 128 reference, 14 planes, eps 1 and 2, seeds 6101-6103. Planes where the coarse grid is **below** the reference:

| eps | seed 6101 | seed 6102 | seed 6103 |
|---|---|---|---|
| 1 | 1/14 | 1/14 | 14/14 |
| 2 | 3/14 | 0/14 | 5/14 |

The count is identical for the three coarse `lambda/h` of a seed and the relative differences shrink by about 4 per doubling (error order about
2). By majority of planes the coarse grid is **above** the reference in 5 of 6 (eps, seed) pairs (the exception is eps 1, seed 6103).

### Observed outputs

`python3 -I .../scripts/firstorder2.py` (exit 0):

```
first-order alpha_L(inf)/(s2 lam) = sqrt(pi)/2 = 0.8862269254527579
tau=   0.5 aL/s2=0.25753 aT/s2=+3.000e-02 aT*tau^2=+0.0075
tau=   1.0 aL/s2=0.46815 aT/s2=+4.460e-02 aT*tau^2=+0.0446
tau=   2.0 aL/s2=0.70577 aT/s2=+3.531e-02 aT*tau^2=+0.1413
tau=   3.0 aL/s2=0.79596 aT/s2=+2.052e-02 aT*tau^2=+0.1846
tau=   5.0 aL/s2=0.85184 aT/s2=+8.331e-03 aT*tau^2=+0.2083
tau=  10.0 aL/s2=0.87743 aT/s2=+2.182e-03 aT*tau^2=+0.2182
tau=  15.0 aL/s2=0.88230 aT/s2=+9.781e-04 aT*tau^2=+0.2201
tau=  20.0 aL/s2=0.88402 aT/s2=+5.518e-04 aT*tau^2=+0.2207
tau=  30.0 aL/s2=0.88524 aT/s2=+2.458e-04 aT*tau^2=+0.2212
tau=  40.0 aL/s2=0.88567 aT/s2=+1.383e-04 aT*tau^2=+0.2213
tau=  50.0 aL/s2=0.88587 aT/s2=+8.857e-05 aT*tau^2=+0.2214
tau=  75.0 aL/s2=0.88607 aT/s2=+3.938e-05 aT*tau^2=+0.2215
tau= 100.0 aL/s2=0.88614 aT/s2=+2.215e-05 aT*tau^2=+0.2215
tau= 150.0 aL/s2=0.88619 aT/s2=+9.846e-06 aT*tau^2=+0.2215
tau= 200.0 aL/s2=0.88620 aT/s2=+5.539e-06 aT*tau^2=+0.2215
t_b=10: windowed aT/s2 = +2.893e-03
t_b=15: windowed aT/s2 = +1.301e-03
t_b=20: windowed aT/s2 = +7.347e-04
t_b=30: windowed aT/s2 = +3.275e-04
t_b=50: windowed aT/s2 = +1.181e-04
t_b=80: windowed aT/s2 = +4.614e-05
t_b=100: windowed aT/s2 = +2.953e-05
t_b=190: windowed aT/s2 = +8.183e-06
```

`python3 -I .../scripts/transfer.py` (exit 0):

```
no effect, equal SE          pass=0.203
no effect, SE .056/.037      pass=0.192
true 5% both                 pass=0.381
true 5%, SE .02              pass=0.737
true 5%, SE .01              pass=0.974
narrow 10%, twin 0           pass=0.110
true 1%, SE 1%               pass=0.362
```

`python3 -I .../scripts/sign_planes.py docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/raw/sf41_campaign/evladder`
(exit 0):

```
eps=1 seed=6101 lh= 16 sigmaT2<ref  1/14 rel first/last +5.73e-04/+1.55e-03
eps=1 seed=6101 lh= 32 sigmaT2<ref  1/14 rel first/last +1.36e-04/+3.71e-04
eps=1 seed=6101 lh= 64 sigmaT2<ref  1/14 rel first/last +2.73e-05/+7.43e-05
eps=1 seed=6102 lh= 16 sigmaT2<ref  1/14 rel first/last -1.32e-04/+2.71e-03
eps=1 seed=6102 lh= 32 sigmaT2<ref  1/14 rel first/last -3.11e-05/+6.43e-04
eps=1 seed=6102 lh= 64 sigmaT2<ref  1/14 rel first/last -6.22e-06/+1.29e-04
eps=1 seed=6103 lh= 16 sigmaT2<ref 14/14 rel first/last -4.51e-04/-2.47e-04
eps=1 seed=6103 lh= 32 sigmaT2<ref 14/14 rel first/last -1.08e-04/-5.91e-05
eps=1 seed=6103 lh= 64 sigmaT2<ref 14/14 rel first/last -2.15e-05/-1.18e-05
eps=2 seed=6101 lh= 16 sigmaT2<ref  3/14 rel first/last +1.76e-03/+1.77e-03
eps=2 seed=6101 lh= 32 sigmaT2<ref  3/14 rel first/last +4.18e-04/+4.27e-04
eps=2 seed=6101 lh= 64 sigmaT2<ref  3/14 rel first/last +8.36e-05/+8.57e-05
eps=2 seed=6102 lh= 16 sigmaT2<ref  0/14 rel first/last +4.81e-04/+7.75e-03
eps=2 seed=6102 lh= 32 sigmaT2<ref  0/14 rel first/last +1.16e-04/+1.84e-03
eps=2 seed=6102 lh= 64 sigmaT2<ref  0/14 rel first/last +2.32e-05/+3.67e-04
eps=2 seed=6103 lh= 16 sigmaT2<ref  5/14 rel first/last -9.19e-04/+3.40e-03
eps=2 seed=6103 lh= 32 sigmaT2<ref  5/14 rel first/last -2.18e-04/+8.13e-04
eps=2 seed=6103 lh= 64 sigmaT2<ref  5/14 rel first/last -4.35e-05/+1.63e-04
```

Results read from these outputs: the table of 5.1 (rows `tau` 1, 10, 20, 30, 100 are 4.460e-2, 2.182e-3, 5.518e-4, 2.458e-4, 2.215e-5);
the window mean `0.2953/t_b^2` (`t_b = 100`: 2.953e-5); transfer-rule pass probabilities 0.203 / 0.192 with no effect (equal SE; SE
0.056/0.037), 0.381 / 0.737 / 0.974 with a true 5 % effect (SE 0.056-0.037 / 0.02 / 0.01), 0.110 (narrow 10 %, twin 0), 0.362 (true
1 %, SE 1 %); the sign table of 5.4. Quantities of 5.2 beyond the window mean (the `tau_b` values, `U/J`, the bias range, the `[47, 94] lambda`
plane mean) come from the orchestrator's earlier reduction of Par2's measured centroid velocity (not in the artifact) and the first-order law; the plane mean was
re-derived by hand here (`0.2215 (1/47 - 1/94)/47 = 5.0e-5`) and the window biases by arithmetic from the table; the Par2 reduction
itself is not in the artifact (section 8).

## 6. Findings

Classes: **B** blocking (must close before activation), **M** major (affects the scientific interpretation), **m** minor. Disposition
names the decision item (`docs/decisions/2026-10-27-sf39-pre-activation-decisions.md`).

### Blocking

| id | finding | evidence | disposition |
|---|---|---|---|
| B1 | The specification is not executable as written: (i) `alpha_AM` and `t_b` per realization undefined (A, E5 and C need them; which `t_b` for a pair of rungs); (ii) no snapshot-time rule (automatic `lambda/2` gives few samples in the window at `sigma^2 = 4`); (iii) "measured mean velocity" undefined: Par2 normalizes by nominal `J` and **overstates alpha by 1.19 / 1.47 / 1.98** (measured Par2 centroid velocity); (iv) the common window of Gate 5 is deferred without a rule; (v) covector statistics are not producible (the runner has no option; contradicts "scripts and configs only"); (vi) aligned moments (E4) need a new script over per-particle data (`times_x.npy`, about 80-110 MB per realization and pass); (vii) `nr_rule.py`, `run_sf39.sh` and the whole analysis layer (`t_b`, `alpha_AM`, bootstrap, TOST, transfer, classifier D) do not exist; (viii) Par2 configs fix `NR: 10`, `seed: 7001`; (ix) `sigma^2 = 6.25` has no geometry, `lambda/h` or `N_R`; (x) validation commands incomplete (no sync/build, no check of narrow sizes, no owner checkpoint; `REMOTE_GPU_WAIT=0` while SF-34 is also READY); (xi) contradictions: failure policy says "SF-19 rtol 1e-10" vs In scope (medium (b) `solve_head` with its 1e-8 fallback), two `N_R` caps, and `docs/runbooks/remote-v100.md:76-80` requiring serialization of remote nodes vs E6 (two concurrent jobs) | specification and runner read at `782c9d9a`; runbook lines cited; Par2 normalization from the measured centroid velocity | -> decision 2026-10-27 F2 (i-iv, M8), F12 (v-xi), F13 (x-xi, E6 vs runbook), F9 (two caps) |
| B2 | The transfer rule ("same sign and ratio in [1/2, 2]" of two noisy relative differences) passes about **20 %** of the time with no grid effect, 0.38 with a true 5 % effect at `N = 20`-like SE, 0.74 at SE 0.02, 0.97 at SE 0.01: a converged study comes out `inconclusive` at production about 80 % of the time, so Gate 4 is `unresolved` by design | section 5.3 (`transfer.py`); derivable without data | -> F4 (replace by the narrow-ladder verdict plus a TOST on the paired twin difference with a margin `2^1.5 m`) |
| B3 | Rule D branch 1's premise ("coarse grids under-disperse `sigma_T^2`", resolution-vs-variance §3.7) is post-hoc, from the cube, `n_eff` about 3; **on medium (b) the existing data invert it**: SF-41 F ladder (`lambda/h` 16/32/64 vs 128) has the coarse grid **above** the reference in 5 of 6 (eps, seed) pairs, error order about 2 (verified by re-running the script). Under the null, an `inconclusive` with an increasing point estimate fires the branch about 50 % of the time and labels "grid-limited lower bound" | section 5.4 (`sign_planes.py`); post-hoc, short narrow ladder | -> F5 (labels require a 90 % CI excluding 0 for increase/decrease; branch not applied to the `h_max` ladder; sign test pre-registered on SF-39's own ladder; citation removed) |
| B4 | "Zero within resolution" (95 % CI contains 0 and half-width <= 2.5e-3) (a) mixes 95 % with the 90 % of the rest, (b) turns "not rejected" into `physical`, (c) implies a 97.5 % bound of 5e-3 (half of B&dD's `sigma^2 = 1` value) | arithmetic on the rule; derivable | -> F6 (`physical (bounded)`: one-sided 95 % upper limit `U <= 2.5e-3 lambda`, reported `alpha_T <= U`; `physical (positive)`; else `unresolved (imprecise)`) |

### Major

| id | finding | evidence | disposition |
|---|---|---|---|
| M1 | Estimator versus the physical question. The temporal `alpha_TAM` at `sigma^2 = 4` is truncated at `tau_b` of about 27-38 (Par2 centroid velocity; SF-39 probably earlier with flux-weighted injection in the window): first-order positive bias 0.8-1.7e-3 `lambda` (`sigma^2 = 1`: about 2e-5; 2.25: 5-8e-5), about half the floor, plus non-linear amplification (the mean is at about 29 `lambda` when the first particle reaches 94 `lambda`), residual flow meander (SF-38 P4: quarter-length means of `<v2>` fluctuate 7-16 times their global mean) and residual `q_r` / drift misalignment (`theta_res^2 alpha_L` about 1e-4 to 5e-4 `lambda`). The plane variance `S_T` uses the full length, is geometric (independent of `k_s`), insensitive to tilt and meander, with a first-order pre-asymptotic bias 4-8 times smaller at `sigma^2 = 4` (2.0e-4 `lambda` over `[47, 94] lambda`). Caveat: A is unaffected (a common bias cancels in paired differences); the risk is D3, "CI excludes 0" | section 5.2; SF-38 P4; own estimates | -> F3 (conjunctive co-primary `alpha_TP`; `physical` only if both estimators say so; discordance is `unresolved (estimator-discordant)`) |
| M2 | "Asymptotic" is never proven: no finite domain separates a slow decay from a small plateau (even the octave ratio is degenerate between `a + c x^-2` and `x^-1`) | first-order law (5.1) | -> F7 (values "at the measured distance"; `Delta_half` and a monotone-decay upper bound used only to downgrade wording; full `sigma_T^2(x1)` and `sigma_T^2(t)` series) |
| M3 | Integration drift and production length. Evidence stops at 64 `lambda` in the cube. The `h_max` ladder probably reads `converged` trivially (error falls 2^4-2^5 per rung) unless trajectories decorrelate. Cheap response: per-realization `RMS(dx_T)/sigma_T` against `x1` between the `h/8` and `h/16` passes of the same stage and seeds, to full length (shadow per trajectory as in SF-37). It answers Lester 2025 App. D directly and measures perturbation growth (finite-time Lyapunov); Lester predicts bounded transverse deformation, the project measured exponential separation in the cube; the test is kinematic and independent of `alpha_T`. Needs the per-pass positions stored (decision F8: the per-pass `planes_x23.npy` arrays already exist in the `EnsembleRunner.cuh` multi-pass layout; not re-verified in this note) | SF-37 RR15, V5; SF-30 §9 | -> F8 (recorded non-gating readings `pathwise resolved` / `decorrelated`; fits recorded) |
| M4 | Power at `sigma^2 = 4`. With the A/B pairing as a proxy: `N` about 221 for `alpha_T` equivalence (`delta = 0`, power 0.8), about 957 for `alpha_L` at +-2.5 %; at `sigma^2 = 1` `alpha_T` `N` is 5-11 if the pairing holds, `alpha_L` about 25, objective B about 53. With about 19-91 GPU-h, `sigma^2 = 4` will probably end `unresolved`; it must be pre-registered as an anticipated outcome. E5 must size `N` from the transverse estimators only (else `alpha_L` fixes the cap); the cap formula pre-specified; the pilot checkpoint reports only SDs, `N` and GPU-h (the margins depend on the finer-rung mean: partial unblinding otherwise) | A/B pairing proxy (section 8) | -> F9 |
| M5 | Transverse width / periodicity. `sigma^2 = 4`: production `L_T = 12 lambda`, narrow ladder **4 `lambda`** (the cube's period, whose transverse growth is largely a periodic-medium artifact), B&dD 25.6 `lambda`, Pollock post-hoc about 0.07 `lambda` at 8 `lambda`. Meander: the section mean of `<v2>/q` does not vanish in a finite periodic section (point sd 0.132 at `sigma^2 = 4`, 8 `lambda`; 0.013 at `sigma^2 = 1`, 25.6 `lambda`; correlation about 1 `lambda`); it belongs to the field (present in (a) and in B&dD's setup); the tilt is its zero mode (predicted `sqrt(S0/L)` = 0.021 vs measured 0.016); it vanishes at first order, so its variance scales as `sigma^4 (lambda/L_T)^2`. Contribution to temporal `alpha_T` (not to plane estimators): 8 `lambda`: (1-2.5)e-3 `lambda`; 12 `lambda`: (0.45-1.1)e-3 `lambda` (about the floor of the margin); 4 `lambda`: (4-10)e-3 `lambda` (10-15 %); 25.6 `lambda`, `sigma^2 = 1`: about 1e-5. Factor about 2 uncertainty. No project evidence quantifies the width effect on plume statistics at `sigma^2 = 4`; the specification contains an implicit 2x2 (width 4/12 `lambda` x `lambda/h` 16/32) without a decision rule and with a source size that changes with the width (window `0.8 L_T`: 3.2 vs 9.6 `lambda`) | SF-38 `C_D.md` (per realization, Pollock, planes 32-64 `lambda`); meander estimate (section 8) | -> F10 (exploratory, non-gating width check with a fixed 9.6 `lambda` source; 24 vs 12 and 12 vs 4 `lambda`) |
| M6 | Method contrast (the project thesis: transport free of spurious macrodispersion) requires on the same fields (1) the tracker converges in `h` and `h_max`, (2) classical schemes carry an excess that shrinks with refinement. Gate 4 already demands the Pollock comparison (`docs/validation/acceptance-gates.md:176`); SF-39 only "records" it. The discriminating pattern is a difference-in-differences (Pollock approaches the consumer under refinement **while** the consumer has converged): low power, reported honestly. Uniform (twin) weights, since RT0 and the spline have different weights; Pollock on the narrow rungs too. Par2 (about the B&dD scheme) descriptive with the corrected normalization; one-sided comparisons with B&dD (0.01, 0.044) are hypothesis-generating (motivated after SF-38), at fixed axes as B&dD | Gate 4 text; SF-37 RR7 | -> F11 (pre-registered estimation, no verdict labels) |
| M7 | Pre-registration integrity: "frozen before SF-38 produced any `sigma^2 = 4` plume statistic" is only partly true: E4 and E5 (now inside A and D) were fixed after SF-38's A/B (disclosed in the bitácora, not in the threshold text); every amendment needs a date-of-knowledge statement | decision history 2026-10-24 | -> F1 |
| M8 | `t_b` is an extreme-value statistic: first arrival falls with `N_R` and differs between rungs (the fast tails are the most grid-sensitive); a paired difference would mix grid effect with window shift. Freeze one `t_b` per amplitude from the pilot for all rungs and arms | design argument | -> F2 |

### Minor

| id | finding | disposition |
|---|---|---|
| m1 | Percentile bootstrap covers 0.87-0.92 at `N = 20` (simulated by a reviewer): use BCa or `t`, the same method in A, B, D | -> F6 |
| m2 | Analysis hierarchy (primary / key secondary / estimation / exploratory) and one table with the count of decision points (about 21 frozen plus new) | -> F12 |
| m3 | Effective `N_p` with weights about `N_p/(1 + V1)` of about 2400 at `sigma^2 = 4` (rule for "`N_p` limits the CI") | -> F12 (split-half particle component, 25 % rule) |
| m4 | Gate 5 sensitivity by ANCOVA on Par2's realized `sigma^2`; Par2 seam (non-periodic K with periodic BC), closed outlet, `cg_rtol 1e-6`, `t_b` resolved to 2-3.5 `tau`, wrapped snapshots (not for transverse moments) | -> F12 |
| m5 | Report non-`ok` by type (possible spurious extrema of `Phi_s`); sensitivity with weights `w exp(-I)`; assert `active` constant over the window; report `alpha_T` in the plume-drift frame beside `q_r`; say "A's verdicts for `alpha_TAM`" in rule D; the frame of `alpha_LAM` in A (aligned or fixed) unstated | -> F2, F12 |
| m6 | Theory-note corrections (assumptions of Lester 2023 §4; p. 11; Lester 2025 App. D; scalings with locators; §9 is project inference; B&dD Euler stepping, `sigma^4`, non-monotone `alpha_T(t)`, Tables 1-2 numbers; MECOM slides 2/33 state `alpha_T` proportional to `exp(sigma^2)`, contradicting B&dD (`sigma^4`; the exponential is for `alpha_L`, eq. 9) and linear theory (0)) | -> F14 |

## 7. Adversarial verification summary

Four independent reviewers (theory/estimator, numerics, domain/periodicity, statistics/pre-registration) tried to refute each draft finding.

Confirmed: the closed-form first-order law and constants (closed form by a reviewer; this note's quadrature agrees, section 5.1); the pass
probabilities of the transfer rule (simulation, 5.3; derivable); the inverted sign on medium (b) (script re-run, 5.4). The remaining findings
are design arguments or post-hoc readings classified in the decision's table, not independently reproduced here.

Corrected during the review:
- The plane estimator's pre-asymptotic bias advantage is **4-8 times, not 20 times**, and holds at `sigma^2 = 4` only.
- The octave ratio is a **weak** decisive test of "asymptotic" (degenerate between decay laws): M2 is a wording rule, not a test.
- "No chaos" does **not** follow from a monotone `Phi_s`: it is a Lyapunov function, not a transverse-map invariant (section 4).
- Post-hoc `sigma^2 = 1` Pollock `alpha_T` of 4.1e-4 +- 2.9e-3 `lambda` (SE; 4 of 10 realizations negative) is **noise**, not a small positive
  value; `sigma^2 = 4`: 0.0696 +- 0.0059 `lambda` (medium (b)) and 0.0652 +- 0.0071 `lambda` (medium (a)) at 8 `lambda` width, per-realization CV 0.27-0.34
  (about 350 times the first-order tail; about 0.6 times the B&dD value at 25.6 `lambda`): a large and precise transverse signal whose nature
  (physical, Pollock's source (A), width effect) is exactly what SF-39 must discriminate; this note does not attribute it.

## 8. Caveats

- Power numbers (M4) use the SF-38 A/B pairing (a change of medium) as a proxy for the pairing of a grid-ladder rung; only the SF-39 pilot resolves it.
- `tau_b` comes from Par2 (another generator, uniform injection), not from the Lagrangian runs; SF-39 with flux-weighted injection in a window
  breaks through earlier. The Par2 reduction that produced `U/J` and `tau_b` and the reviewers' simulations behind the bootstrap coverage, the
  `N` values and the meander magnitudes are not part of this artifact's scripts (only `firstorder2.py`, `transfer.py`, `sign_planes.py` are).
- The `sigma^2 = 4` lower bound of `tau_b` is quoted as 29.0 in one reduction and 26.7 in the bias range (section 5.2); the bias range
  0.82-1.65e-3 `lambda` is the wider reading.
- The printed formula of Dagan 1988 was not checked (full text unavailable); the `tau^-2` law is an own derivation confirmed independently by a
  reviewer and numerically here, not compared with the printed formula.
- Meander contributions are estimates within about a factor 2 (10 realizations, one width, `alpha_L/sigma_L` assumed).
- The medium (b) sign inversion comes from a short, narrow ladder (2 `lambda` section, <= 24 `lambda`, 3 seeds): it shows the premise does not
  transfer; it does not establish the sign in the production geometry.
- Several literature sources were read at abstract level only; slide-only content of the MECOM presentation was not inspected; the claims
  attributed to them are limited to what the abstracts state.
- The calculations are first order in `sigma^2`; at `sigma^2 = 4` they are a design reference for the pre-asymptotic bias, not a prediction of the
  non-linear value.
- Nothing here is acceptance evidence of SF-39; every threshold is measured by SF-39's own pre-registered campaign, and no `alpha` claim is made.

## Outcome

Owner delegation of 2026-10-27: the orchestrator decided each finding; the decisions are F1-F14 of
[`2026-10-27-sf39-pre-activation-decisions.md`](../decisions/2026-10-27-sf39-pre-activation-decisions.md). The Goal is unchanged; no margin or
CI target frozen by D15 was changed.
